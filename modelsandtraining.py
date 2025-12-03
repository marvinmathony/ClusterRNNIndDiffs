import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd
import math
import numpy as np
import wandb

class Decoder(nn.Module):
    def __init__(self, in_dim, z_dim, hid, A=2):
        super().__init__()
        self.rnn   = nn.GRU(in_dim+z_dim, hid, batch_first=True)
        #self.rnn   = nn.GRU(in_dim, hid, batch_first=True)
        self.lin   = nn.Linear(hid, A)
        self.z2h0  = nn.Linear(z_dim, hid)
        self.z2h0_mlp = nn.Sequential(nn.Linear(z_dim,hid), nn.ReLU(),
                                 nn.Linear(hid,hid),    nn.ReLU(),
                                 nn.Linear(hid,hid),    nn.ReLU())
        self.z_lin = nn.Linear(hid,hid)

    def forward(self, seq, z, hidden=None, ID_test=False):
        """
        seq: (B, T, in_dim)
        z:   (B, z_dim)
        hidden: optional previous hidden state (1, B, hid)
        """

        if ID_test:
            if z.dim() > 2:
                z = z.view(z.size(0), -1)  # flatten (B, 1, 1, zdim) -> (B, zdim)
            if z.dim() == 1:
                z = z.unsqueeze(0)        # (zdim,) -> (1, zdim)

        if hidden is None:
            hidden = self.z2h0(z).unsqueeze(0).contiguous().to(seq.device)  # initialize hidden state from z


        # Repeat z across time steps
        zexp = z.unsqueeze(1).expand(-1, seq.size(1), -1)
        rnn_input = torch.cat([seq, zexp], -1)
        #rnn_input = seq

        if torch.isnan(rnn_input).any() or torch.isinf(rnn_input).any():
            print("NaNs or Infs in rnn_input!")
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print("NaNs or Infs in hidden state!")
        out, hidden = self.rnn(rnn_input, hidden)
        logits = self.lin(out)

        return logits, hidden
    
class LatentRNN_secondstep(nn.Module):
    def __init__(self, encoder, hid, z_dim=3, in_dim=2, A=2, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else Decoder(in_dim, z_dim, hid, A)
        self.return_per_timestep = False

    def forward(self, xenc, blocks, sample_z=False):
        """
        xenc: inputs to encoder (e.g., (B, B_blk, T, in_dim))
        blocks: input sequences to policy decoder
        sample_z: whether to sample from q(z|x)
        """
        mu, lv = self.encoder(xenc, return_per_timestep=self.return_per_timestep)
        if sample_z:
            std = torch.exp(0.5 * lv)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # optional h0 export
        h0_ = self.decoder.z2h0(z)  # (B,H)
        logits, hidden = self.decoder(blocks.squeeze(1), z)
        return logits.unsqueeze(1), mu, lv, z, h0_

class LookupEncoderZ(nn.Module):
    """
    Learnable per-participant z using nn.Embedding.
    """
    def __init__(self, n_participants: int, z_dim: int):
        super().__init__()
        self.embed = nn.Embedding(n_participants, z_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long().view(-1)                 # (B,)
        return self.embed(idx)
    
class LatentRNNz(nn.Module):
    def __init__(self, encoder, decoder, hid, z_dim=3, in_dim=2, A=2, block_structure=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.block_structure = block_structure
    def forward(self, xenc, blocks):            # blocks: (B,Bk,T,2)
        
        z  = self.encoder(xenc)
        #print(f"[{self.name}] mu shape: {mu.shape}, lv shape: {lv.shape}, z shape: {z.shape}")
        h0_ = self.decoder.z2h0(z)          # (B,H) - just for usage outside of RNN
        if self.block_structure:    
            outs = [ self.decoder(blocks[:,b], z) for b in range(blocks.size(1)) ]
            return torch.stack(outs,1), z, h0_
        else:
            logits, hidden = self.decoder(blocks, z) #not sure about hidden
            return logits, hidden, h0_, z
        


class IDRNN(nn.Module):
    def __init__(self, in_dim, z_dim, hid=128):
        super().__init__()
        self.block_encoder = nn.GRU(in_dim, hid, batch_first=True)
        self.mu_head       = nn.Linear(hid, z_dim)
        self.logvar_head   = nn.Linear(hid, z_dim)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # small variance
        self.count_training = 0
        self.count_testing = 0

    def forward(self, blocks_tensor, return_per_timestep=True):
        """
        blocks_tensor: (B, B_blk, T, in_dim)
          - B      = number of participants in batch
          - B_blk  = blocks per participant
          - T      = trials per block (padded if needed)
        """
        B, B_blk, T, in_dim = blocks_tensor.shape
        x = blocks_tensor.view(B * B_blk, T, in_dim)  # flatten blocks

        out, _ = self.block_encoder(x)  # (B*B_blk, T, hid) --> am I not putting all participants together here?
        
        if return_per_timestep:
            h = out.view(B, B_blk, T, -1)         # (B, B_blk, T, hid)
            mu     = self.mu_head(h)              # (B, B_blk, T, z_dim)
            logvar = self.logvar_head(h).clamp(-10, 5)
            if self.count_training < 1:
                print(f"[IDRNN] return_per_timestep is True")
                #print(f"[IDRNN] mu shape: {mu.shape}, logvar shape: {logvar.shape}")
                self.count_training += 1
            return mu, logvar                     # full sequence
        else:
            h_final = out[:, -1]                  # (B*B_blk, hid) — last step
            h_block = h_final.view(B, B_blk, -1)  # (B, B_blk, hid)
            h_participant = h_block.mean(dim=1)   # (B, hid)
            mu     = self.mu_head(h_participant)  # (B, z_dim)
            logvar = self.logvar_head(h_participant).clamp(-10, 5)
            if self.count_testing < 1:
                print(f"[IDRNN] return_per_timestep is False")
                #print(f"[IDRNN] mu shape: {mu.shape}, logvar shape: {logvar.shape}")
                self.count_testing += 1
            return mu, logvar
        
class AblatedDecoder(nn.Module):
    def __init__(self, in_dim, hid, A=2):
        super().__init__()
        #self.h0 = nn.Parameter(torch.zeros(1,1,hid))
        self.register_buffer('h0', torch.zeros(1, 1, hid))
        self.rnn= nn.GRU(in_dim, hid, batch_first=True)
        self.lin= nn.Linear(hid,A)

    """def forward(self, seq, hidden=None):
        if hidden is None:
            hidden = self.h0.expand(1, seq.size(0), -1).contiguous()
        out, hidden_out = self.rnn(seq, hidden)
        logits=self.lin(out)
        return logits, hidden_out, out"""

    def forward(self, seq, h0=None):
        if h0 is None:
            h0 = self.h0.expand(1, seq.size(0), -1).contiguous()
            h0 = h0.to(seq.device)
        #h0 = self.h0            
        out, hid = self.rnn(seq, h0) #might have to expand this as well
        return self.lin(out), hid, out

class AblatedRNN(nn.Module):
    def __init__(self, hid, in_dim=2, A=2, block_structure = True):
        super().__init__()
        self.in_dim = in_dim
        self.dec = AblatedDecoder(in_dim,hid,A)
        self.block_structure = block_structure
    def forward(self, blocks, h0=None):
        if self.block_structure:
            out_list = [ self.dec(blocks[:,b]) for b in range(blocks.size(1)) ]
            logits, final_hid, hidden_tr = zip(*out_list)
            return torch.stack(logits,1), torch.stack(final_hid, 1).squeeze(0), torch.stack(hidden_tr, 1)
        else:
            logits, final_hid, hidden_tr = self.dec(blocks, h0=h0)
            return logits, final_hid, hidden_tr
    
### TRAINING SCRIPTS ###
def train_latentrnn_noblocks_palminteri(
    model: nn.Module,
    ids_train: torch.Tensor,           # (B,)
    X_train: torch.Tensor,             # (B, T, in_dim)
    y_onehot: torch.Tensor,            # (B, T, A) one-hot
    ids_val: torch.Tensor, 
    X_val: torch.Tensor,
    y_val_onehot: torch.Tensor,
    epochs: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu")
):
    model = model.to(device)
    ids_train = ids_train.to(device)
    X_train = X_train.to(device)
    y_onehot = y_onehot.to(device)
    #y_train = torch.argmax(y_onehot, dim=-1).long()          # (B, T)
    y_train = y_onehot #just in sloutsky case

    ids_val = ids_val.to(device)
    X_val = X_val.to(device)
    y_val_onehot = y_val_onehot.to(device)
    #y_val = torch.argmax(y_val_onehot, dim=-1).long()          # (B, T)
    y_val = y_val_onehot # just in sloutsky case

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}
    accuracy_dict = {}
    val_accuracy_dict = {}
    best_val_acc = 0

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits, hidden, h0, z = model(ids_train, X_train)            # logits: (B,T,A)
        if ep == 1:
            print(f"shape logits: {logits.shape}")
            print(f"shape logits after reshape: {logits.reshape(-1, logits.size(-1)).shape}")
            print(f"shape y_train: {y_train.shape}")
            print(f"shape y_train after reshape: {y_train.reshape(-1).shape}")
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_train.reshape(-1).long(),
            reduction='mean'
        )
        nll.backward()
        opt.step()
        #probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        #flatten
        predictions_flat = predictions.reshape(-1)
        y_flat = y_train.reshape(-1)
        total = y_flat.numel()
        #acc
        train_acc = torch.sum(predictions_flat == y_flat)
        final_train_acc = (train_acc/total).item()
        train_losses.append(nll.item())
        accuracy_dict[ep] = final_train_acc

    # accuracy monitoring 
        model.eval()
        with torch.no_grad():
            logits_val, hidden_val, h0_val, z_val = model(ids_val, X_val)
            nll_val = F.cross_entropy(
            logits_val.reshape(-1, logits_val.size(-1)),
            y_val.reshape(-1).long(),
            reduction='mean'
            )
            val_losses.append(nll_val.item())

            #probs = F.softmax(logits, dim=-1)
            predictions_val = torch.argmax(logits_val, dim=-1)
            #flatten
            predictions_flat_val = predictions_val.reshape(-1)
            y_val_flat = y_val.reshape(-1)
            total_val = y_val_flat.numel()
            #acc
            val_acc = torch.sum(predictions_flat_val == y_val_flat)
            final_val_acc = (val_acc/total_val).item()
            val_accuracy_dict[ep] = final_val_acc

            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                z_val_best = z_val
                best_epoch = ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                

        if ep % 100 == 0:
            print(f"[LatentRNNz] epoch {ep:4d}  loss {nll.item():.4f}, train_acc: {final_train_acc}, val_acc: {final_val_acc}, val_loss {nll_val.item():.4f}")

    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_best, hidden_best, h0_best, z_best = model(ids_train, X_train)
        pA_best = F.softmax(logits_best, dim=-1)              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch, #ep #best_epoch
        "best_val_loss": best_val_loss,
        "z": z_best,
        "z_val": z_val_best,
        "h0": h0_best,
        "best_val_acc": best_val_acc,
    }
    print(f"best validation accuracy: {best_val_acc} in epoch: {best_epoch}")

    return model, train_losses, val_losses, pA_per_epoch, training_dict

def train_latentrnn_noblocks(
    model: nn.Module,
    ids_train: torch.Tensor,           # (B,)
    ids_test: torch.Tensor,
    X_train: torch.Tensor,             # (B, T, in_dim)
    y_onehot: torch.Tensor,            # (B, T, A) one-hot
    X_test: torch.Tensor,
    y_test_onehot: torch.Tensor,
    p_target: torch.Tensor = None,     # (B, T) prob(arm0), optional
    p_test_target: torch.Tensor = None,
    epochs: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu")
):
    model = model.to(device)
    ids_train = ids_train.to(device)
    ids_test = ids_test.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_onehot = y_onehot.to(device)
    y_train = torch.argmax(y_onehot, dim=-1).long()          # (B, T)
    y_test_onehot = y_test_onehot.to(device)
    y_test = torch.argmax(y_test_onehot, dim=-1).long()

    train_losses = []
    test_losses = []
    kl_vals = []
    kl_test_vals = []
    accuracy = []
    test_accuracy = []
    best_kl = float('inf')
    best_test_kl = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits, hidden, h0, z = model(ids_train, X_train)            # logits: (B,T,A)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_train.reshape(-1),
            reduction='mean'
        )
        nll.backward()
        opt.step()
        train_losses.append(nll.item())
        preds = logits.reshape(-1, logits.size(-1)).argmax(dim=-1)

        # Ground truth labels
        targets = y_train.reshape(-1).long()

        # Accuracy: compare and compute mean
        acc = (preds == targets).float().mean().item()
        accuracy.append(acc)

        # KL monitoring (optional)
        if p_target is not None:
            model.eval()
            with torch.no_grad():
                p_model = F.softmax(logits, dim=-1)[:, :, 0]   # prob(arm 0)
                kl = compute_kl_divergence_bernoulli(p_target.to(device), p_model)
                kl_vals.append(kl)

                if kl < best_kl:
                    best_kl = kl
                    best_epoch = ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        
        # test 
        if p_test_target is not None:
            model.eval()
            logits_test, hidden_test, h0_test, z_test = model(ids_test, X_test)
            nll_test = F.cross_entropy(logits_test.reshape(-1, logits_test.size(-1)), y_test.reshape(-1),reduction='mean')
            test_losses.append(nll_test.item())
            preds_test = logits_test.reshape(-1, logits_test.size(-1)).argmax(dim=-1)

            # Ground truth labels
            targets_test = y_test.reshape(-1).long()

            # Accuracy: compare and compute mean
            acc_test = (preds_test == targets_test).float().mean().item()
            test_accuracy.append(acc_test)
            
            with torch.no_grad():
                p_model_test = F.softmax(logits_test, dim=-1)[:, :, 0]   # prob(arm 0)
                kl_test = compute_kl_divergence_bernoulli(p_test_target.to(device), p_model_test)
                kl_test_vals.append(kl_test)

                if kl_test < best_test_kl:
                    best_test_kl = kl_test
                    best_test_epoch_kl = ep
                    best_test_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 100 == 0:
            if p_target is not None:
                print(f"[LatentRNNz] epoch {ep:4d}  loss {nll.item():.4f}  KL {kl:.4f}  accuracy {acc}  test loss {nll_test}  test KL {kl_test}  test accuracy {acc_test}")
            else:
                print(f"[LatentRNNz] epoch {ep:4d}  loss {nll.item():.4f}")

    best_epoch_test_acc = test_accuracy.index(max(test_accuracy))+1
    best_epoch_test_loss = test_losses.index(min(test_losses))+1
    best_epoch_train_loss = train_losses.index(min(train_losses))+1
    best_epoch_acc = accuracy.index(max(accuracy))+1

    print(f"Best epoch kl: {best_epoch}")
    print(f"Best test kl epoch: {best_test_epoch_kl}")
    print(f"Best loss epoch: {best_epoch_train_loss}")
    print(f"Best test loss epoch: {best_epoch_test_loss}")
    print(f"Best accuracy loss epoch: {best_epoch_acc}")
    print(f"Best test accuracy loss epoch: {best_epoch_test_acc}")
    # fallback: if no KL tracked, keep last state
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs
        best_kl = float('nan')

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_best, hidden_best, h0_best, z_best = model(ids_train, X_train)
        pA_best = F.softmax(logits_best, dim=-1)              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch,
        "best_kl": best_kl,
        "z": z_best,
        "h0": h0_best,
    }

    return model, train_losses, kl_vals, pA_per_epoch, training_dict


# Causal Importance-Weighted Test Function (per-timestep z)
@torch.no_grad()
def test_latentrnn_secondstep_causal_posterior_weighting(model, blocks, y, N=100, num_actions=2, first_n=None, id_case = True):
    """
    Args:
        model: LatentRNN_secondstep instance with encoder + frozen decoder
        blocks: (B, Bk, T, in_dim) input histories
        y:      (B, Bk, T) action labels
        N:      number of importance samples
        num_actions: number of actions (e.g., 2)
        first_n: if set, truncate to first_n time steps
    Returns:
        IWLL scalar (mean across participants), IWLL per participant (B,)
    """
    model.eval()
    y = y.squeeze(-1)  # ensure shape (B, Bk, T)

    if first_n is not None:
        blocks = blocks[:, :, :first_n]
        y = y[:, :, :first_n]

    B, Bk, T, A = blocks.shape[0], blocks.shape[1], blocks.shape[2], num_actions
    z_dim = model.encoder.mu_head.out_features
    device = blocks.device

    log_likelihoods_per_participant = torch.zeros(B, device=device)
    #print(f"z_samples shape: {z_samples.shape}")
    #print(f"z_samples: {z_samples}")
    #z_samples = z_samples.unsqueeze(2).unsqueeze(3)  # (N, B, 1, 1, z_dim) for broadcasting
    #print(f"z_samples expanded shape: {z_samples.shape}")
    #print(f"z_samples after formatting: {z_samples}")
    mu_tensor = torch.ones(B, T, z_dim)
    
    for participant_idx in range(B):
        participant_log_likelihood = 0.0
        
        for block_idx in range(Bk):
            block_data = blocks[participant_idx, block_idx]  # (T, in_dim)
            block_actions = y[participant_idx, block_idx]  # (T,)
            
            block_log_likelihood = 0.0
            for t in range(1, T + 1):
                # Only use data up to time t (causality)
                data_up_to_t = block_data[:t]  # (t, in_dim)
                data_batched = data_up_to_t.unsqueeze(0).unsqueeze(0)  # (1, 1, t, in_dim)
                
                if id_case:
                    # Get posterior parameters at time t
                    mu_t, logvar_t = model.encoder(data_batched, return_per_timestep=True)
                    mu_t = mu_t[0, 0, -1]  # (z_dim,)
                    mu_tensor[participant_idx, t-1] = mu_t # append latents to tensor
                    logvar_t = logvar_t[0, 0, -1]  # (z_dim,)
                    
                    # Sample z from the posterior q(z|x_{1:t})
                    std_t = torch.exp(0.5 * logvar_t)
                    z_samples = mu_t + std_t * torch.randn((N, z_dim), device=device)
                else:
                    z_samples = torch.zeros((N, z_dim), device=device) #torch.ones((N, z_dim), device=device) 
                
                # Vectorized decoder - only uses data up to t
                data_batch = data_up_to_t.unsqueeze(0).expand(N, -1, -1)  # (N, t, in_dim)
                logits,_ = model.decoder(data_batch, z_samples, ID_test=True)  # (N, t, A)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)  # (N, A)
                
                action_t = block_actions[t-1].long()
                log_probs_t = log_probs[:, action_t]  # (N,)
                
                # Marginalize over z samples (Monte Carlo integration)
                log_p_at = torch.logsumexp(log_probs_t, dim=0) - math.log(N)
                block_log_likelihood += log_p_at.item()
            
            participant_log_likelihood += block_log_likelihood 
        
        log_likelihoods_per_participant[participant_idx] = participant_log_likelihood
    total_log_likelihood = log_likelihoods_per_participant.sum()
    total_trials = B * Bk * T
    geometric_mean_per_trial = torch.exp(total_log_likelihood / total_trials)
    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant, mu_tensor, geometric_mean_per_trial


# --- Compute RNN likelihoods ---
def compute_rnn_likelihoods_torch(test_function, model, test_xin, choice_test, train_xin, latent, choice_train=None, N=500, id=True):
    rnn_results = []
    B, T = choice_test.shape
    with torch.no_grad():
        if latent:
            if test_xin.ndim == 3:
                test_xin = test_xin.unsqueeze(1)
                choice_test = choice_test.unsqueeze(1)
                train_xin = train_xin.unsqueeze(1)
                if choice_train is not None:
                    choice_train = choice_train.unsqueeze(1)
                    mean_ll_train, ll_per_participant_train, latent_tensor_train, geometric_mean_per_trial_train = test_function(
                    model=model, blocks=train_xin, y=choice_train, N=N, id_case=True
                    )
                else:
                    latent_tensor_train = None
            mean_ll, ll_per_participant, latent_tensor, geometric_mean_per_trial_test = test_function(
                model=model, blocks=test_xin, y=choice_test, N=N, id_case=id
            )
            #test_latentrnn_secondstep_causal_posterior_weighting

            #probably do linear probe here - or in the test function
            norm_ll = (ll_per_participant / T).exp()  # Normalize and convert to probabilities
            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": norm_ll.cpu().numpy(),
                "model": "IDRNN" if id else "common_process_RNN",#"LatentRNN_causalIW" - change code to implement other names as well
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train, geometric_mean_per_trial_test
        else:
            logits_test, final_hidden, latent_tensor = model(test_xin)
            #also access train tensor
            _, _, latent_tensor_train = model(train_xin)

            probs_test = F.softmax(logits_test, dim=-1)

            choices = choice_test.reshape(-1)
            probs_flat = probs_test.reshape(-1, probs_test.size(-1))
            chosen_probs = probs_flat[torch.arange(B*T), choices.long()].view(B, T)

            log_likelihoods = chosen_probs.clamp(min=1e-8).log()
            log_ll_per_session = log_likelihoods.sum(dim=1)
            norm_ll = (log_ll_per_session / T).exp()
            total_trials = B*T
            geometric_mean_per_trial = torch.exp(log_ll_per_session.sum()/total_trials).item()
            #print(f"normalized_ll_per_trial: {normalized_ll_per_trial}")
            #print(f"norm_ll mean: {norm_ll.sum()/B}")
            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": norm_ll.cpu().numpy(),
                #"normalized_likelihood_per_trial": normalized_ll_per_trial,
                "model": "vanillaRNN"
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train, geometric_mean_per_trial
        
def gaussian_nll_point(mu, logvar, z_T):

    var = logvar.exp()
    return 0.5 * (((z_T - mu)**2) / (var + 1e-8) + logvar).sum(-1).mean()
        
def elbo_lossZ(logits, targets):
    B,Bk,T,A = logits.shape
    nll = F.cross_entropy(logits.reshape(-1,A), targets.long().reshape(-1), reduction='mean')
    return nll, nll.detach()

def step_two_loss(lmbd, mu, logvar, z_T, L_pol):
    kl_loss = gaussian_nll_point(mu, logvar, z_T)
    loss = lmbd * kl_loss + (1-lmbd) * L_pol
    #print(f"kl loss: {kl_loss}, policy loss: {L_pol}")
    return loss, kl_loss

def compute_kl_divergence_bernoulli(p_true, p_pred):
    epsilon = 1e-10  # To avoid division by zero and log(0) issues
    kl_div = p_true * ((p_true + epsilon) / (p_pred + epsilon)).log() + \
             (1 - p_true) * ((1 - p_true + epsilon) / (1 - p_pred + epsilon)).log()
    return kl_div.mean().item()

def train_latentrnn_IDRNN_palminteri(model, xenc, blocks, y, lookup_z, xenc_val, y_val, z_val_lookup, epochs=60, patience = 300, lr=1e-3, window_size=20):
    #print(f"x_enc shape: {xenc.shape}, blocks shape: {blocks.shape}, y shape: {y.shape}")
    #xenc input must have shape (B, B_blk, T, in_dim)
    train_elbos = []
    val_elbos = []
    opt = torch.optim.Adam(model.parameters(), lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    kl_vals = []
    best_kl = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}
    accuracy_dict = {}
    val_accuracy_dict = {}
    val_acc_history = []
    best_val_acc = 0
    model_state_dict = {}
    # freeze decoder
    for p in model.decoder.parameters():
        assert not p.requires_grad, "Decoder parameters should be frozen before training"

    for ep in range(1,epochs+1):
        """model.train()
        logits, mu, lv, z, h0_ = model(xenc, blocks, sample_z=False) # two times x data for modularity 
        #(other architectures need two arguments here, one for position encoding, one for actual input. Here we just give input twice)
        policy_loss, nll = elbo_lossZ(logits, y)
        #policy_loss.backward(); opt.step()
        
        lambda_post, lambda_pol = 1.0, 1.0 #try setting lambda pol to 0 in order to only weigh z in training
        loss = step_two_loss(lambda_post, lambda_pol, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()
        opt.zero_grad()"""
        # implementing random prefixes to prepare model for causal structure at test time
        model.train()
        opt.zero_grad()

        # randomly pick a prefix length to simulate causal training
        max_len = blocks.size(2)
        prefix_len = np.random.randint(5, max_len+1)  # random length
        blocks_prefix = blocks[:, :, :prefix_len, :]
        xenc_prefix = xenc[:, :, :prefix_len, :]
        y_prefix = y[:, :, :prefix_len]
        #p_target_prefix = p_target[:,:prefix_len]

        logits, mu, lv, z, h0_ = model(xenc_prefix, blocks_prefix, sample_z=False)
        policy_loss, nll = elbo_lossZ(logits, y_prefix)

        lmbd = 0.5 #try setting lambda pol to 0 in order to only weigh z in training
        loss, _ = step_two_loss(lmbd, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()
        train_elbos.append(loss.item())

        xenc_val_prefix = xenc_val[:, :, :prefix_len, :]
        yval_prefix = y_val[:, :, :prefix_len]

        model.eval()
        val_logits, mu_val, lv_val, _, _ = model(xenc_val_prefix, xenc_val_prefix, sample_z=False)
        val_policy_loss, val_nll = elbo_lossZ(val_logits, yval_prefix)
        val_loss, _ = step_two_loss(lmbd, mu_val, lv_val, z_val_lookup, val_policy_loss)
        val_elbos.append(val_loss.item())
            
        #accuracy tracking
        #TRAINING
        predictions = torch.argmax(logits, dim=-1)
        #flatten
        predictions_flat = predictions.reshape(-1)
        y_flat = y_prefix.reshape(-1)
        total = y_flat.numel()
        #acc
        acc = torch.sum(predictions_flat == y_flat)
        final_acc = (acc/total).item()
        accuracy_dict[ep] = final_acc


        #VALIDATION
        predictions_val = torch.argmax(val_logits, dim=-1)
        #flatten
        predictions_flat_val = predictions_val.reshape(-1)
        y_val_flat = yval_prefix.reshape(-1)
        total_val = y_val_flat.numel()
        #acc
        val_acc = torch.sum(predictions_flat_val == y_val_flat)
        final_val_acc = (val_acc/total_val).item()
        val_accuracy_dict[ep] = final_val_acc
        val_acc_history.append(final_val_acc)
        model_state_dict[ep] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            
            epochs_no_improve = 0
            
        else:
            epochs_no_improve += 1

        if ep % 50 == 0:
            print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  val loss {val_loss.item():.3f},  train_acciracy: {final_acc},  val_accuracy: {final_val_acc}")
            

        #if epochs_no_improve >= patience:
            #print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            #break

    #best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    window = window_size
    smoothed_val_acc = np.convolve(val_acc_history, np.ones(window)/window, mode='valid')
    best_epoch = np.argmax(smoothed_val_acc) + window // 2
    best_model_state = model_state_dict[best_epoch]
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        logits_best, mu_best, lv_best, z_best, h0_best = model(xenc, blocks, sample_z=False)
        pA_best = F.softmax(logits_best, dim=-1).squeeze(1)[:, :, 0]              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch, #best_epoch, #just last epoch for now
        "best_kl": best_kl,
        "z": mu_best,
        "h0": h0_best,
        "val_acc": final_val_acc
    }
    print(f"best validation accuracy: {val_accuracy_dict[best_epoch]} at epoch {best_epoch}")

        #val_elbos.append(val_loss)
    return model, mu_best, lv_best, train_elbos, val_elbos, training_dict, pA_per_epoch

def train_latentrnn_IDRNN(model, xenc, blocks, y, lookup_z, xenc_val, y_val, p_target, device, epochs=60, patience = 300, lr=1e-3, window_size=40, lmbd = 0.5):
    #print(f"x_enc shape: {xenc.shape}, blocks shape: {blocks.shape}, y shape: {y.shape}")
    #xenc input must have shape (B, B_blk, T, in_dim)
    train_elbos = []
    val_elbos = []
    opt = torch.optim.Adam(model.parameters(), lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    kl_vals = []
    best_kl = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}
    accuracy_dict = {}
    val_accuracy_dict = {}
    val_acc_history = []
    model_state_dict = {}
    # freeze decoder
    for p in model.decoder.parameters():
        assert not p.requires_grad, "Decoder parameters should be frozen before training"

    kl_loss_list = []
    CE_loss_list = []
    for ep in range(1,epochs+1):
        """model.train()
        logits, mu, lv, z, h0_ = model(xenc, blocks, sample_z=False) # two times x data for modularity 
        #(other architectures need two arguments here, one for position encoding, one for actual input. Here we just give input twice)
        policy_loss, nll = elbo_lossZ(logits, y)
        #policy_loss.backward(); opt.step()
        
        lambda_post, lambda_pol = 1.0, 1.0 #try setting lambda pol to 0 in order to only weigh z in training
        loss = step_two_loss(lambda_post, lambda_pol, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()
        opt.zero_grad()"""
        # implementing random prefixes to prepare model for causal structure at test time
        model.train()
        opt.zero_grad()

        # randomly pick a prefix length to simulate causal training
        max_len = blocks.size(2)
        prefix_len = np.random.randint(5, max_len+1)  # random length
        blocks_prefix = blocks[:, :, :prefix_len, :]
        xenc_prefix = xenc[:, :, :prefix_len, :]
        y_prefix = y[:, :, :prefix_len]
        p_target_prefix = p_target[:,:prefix_len]

        xenc_val_prefix = xenc_val[:, :, :prefix_len, :]
        yval_prefix = y_val[:, :, :prefix_len]

        logits, mu, lv, z, h0_ = model(xenc_prefix, blocks_prefix, sample_z=False)
        policy_loss, nll = elbo_lossZ(logits, y_prefix)

        #think about z and action loss and implementing it step wise
        loss, kl_loss_train = step_two_loss(lmbd, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()

        train_elbos.append(loss.item())

        model.eval()
        val_logits, _, _, _, _ = model(xenc_val_prefix, xenc_val_prefix, sample_z=False)
        val_policy_loss, val_nll = elbo_lossZ(val_logits, yval_prefix)
        val_loss, kl_loss = step_two_loss(lmbd, mu, lv, lookup_z, val_policy_loss)
        val_elbos.append(val_loss.item())

  


        #accuracy tracking
        #TRAINING
        predictions = torch.argmax(logits, dim=-1)
        #flatten
        predictions_flat = predictions.reshape(-1)
        y_flat = y_prefix.reshape(-1)
        total = y_flat.numel()
        #acc
        acc = torch.sum(predictions_flat == y_flat)
        final_acc = (acc/total).item()
        accuracy_dict[ep] = final_acc

        


        # KL monitoring (optional)
        if p_target is not None:
            model.eval()
            with torch.no_grad():
                p_model = F.softmax(logits, dim=-1).squeeze(1)[:, :, 0]   # prob(arm 0)
                #print(f"shape pA_model: {p_model.shape}")
                #print(f"shape pA_target: {p_target_prefix.shape}")
                kl = compute_kl_divergence_bernoulli(p_target_prefix.to(device), p_model)
                kl_vals.append(kl)

                if kl < best_kl:
                    best_kl = kl
                    best_epoch = ep
                    #best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_mu = mu
                    best_lv = lv


        #VALIDATION
        predictions_val = torch.argmax(val_logits, dim=-1)
        #flatten
        predictions_flat_val = predictions_val.reshape(-1)
        y_val_flat = yval_prefix.reshape(-1)
        total_val = y_val_flat.numel()
        #acc
        val_acc = torch.sum(predictions_flat_val == y_val_flat)
        final_val_acc = (val_acc/total_val).item()
        val_accuracy_dict[ep] = final_val_acc
        val_acc_history.append(final_val_acc)
        model_state_dict[ep] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        wandb.log({"loss_test": val_policy_loss, "kl_loss": kl_loss, "accuracy_test": final_val_acc}, step=ep)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_loss_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            
        else:
            epochs_no_improve += 1

        if ep % 100 == 0:
            ckpt_path = f"checkpoints/IDRNN_epoch_{ep:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  val loss {val_loss.item():.3f},  train_acciracy: {final_acc},  val_accuracy: {final_val_acc}")
            

        #if epochs_no_improve >= patience:
            #print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            #break

    #best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    window = window_size
    smoothed_val_acc = np.convolve(val_acc_history, np.ones(window)/window, mode='valid')
    best_epoch = np.argmax(smoothed_val_acc) + window // 2
    best_model_state = model_state_dict[best_epoch]
    model.load_state_dict(best_model_state)


    print(f"best model state at epoch {best_epoch} with acc: {val_accuracy_dict[best_epoch]}")
    #model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_best, mu_best, lv_best, z_best, h0_best = model(xenc, blocks, sample_z=False)
        pA_best = F.softmax(logits_best, dim=-1).squeeze(1)[:, :, 0]              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch,
        "best_kl": best_kl,
        "z": mu_best,
        "h0": h0_best
    }

        #val_elbos.append(val_loss)
    return model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_per_epoch

def train_ablated_noblocks_palminteri(model, X_train, y_train, X_val, y_val, epochs=5000, lr=0.001):
    train_elbos = []
    val_elbos = []
    kl_vals = []
    best_val_loss = float('inf')
    best_val_acc = 0
    #y_train = torch.argmax(y_train, dim=-1)
    pA_per_epoch = {} # dictionary to store pA predictions for later comparison
    train_accuracy_dict = {}
    val_accuracy_dict = {}

    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits, _, _ = model(X_train)  # (B, T, A)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y_train.view(-1).long())
        nll.backward()
        opt.step()
        train_elbos.append(nll)
        #print(model.dec.h0.grad)

        ### accuracy ###
        # Predictions: take argmax over logits
        preds = logits.view(-1, logits.size(-1)).argmax(dim=-1)

        # Ground truth labels
        targets = y_train.view(-1).long()

        # Accuracy: compare and compute mean
        acc = (preds == targets).float().mean().item()
        train_accuracy_dict[ep] = acc

        """val_loss = val_elbo_ablated(model, X_val, y_val)
        train_elbos.append(nll.item())
        val_elbos.append(val_loss)"""

        """if ep % 50 == 0:
            print(f"Epoch {ep:3d}  Train loss {nll.item():.3f}")"""

        # === Validation loss ===
        model.eval()
        with torch.no_grad():
            logits_val, _, _ = model(X_val)  # (B, T, A)
            nll_val = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1).long())
            val_elbos.append(nll_val.item())

            ### accuracy ###
            # Predictions: take argmax over logits
            preds_val = logits_val.view(-1, logits_val.size(-1)).argmax(dim=-1)

            # Ground truth labels
            targets_val = y_val.view(-1).long()

            # Accuracy: compare and compute mean
            acc_val = (preds_val == targets_val).float().mean().item()
            val_accuracy_dict[ep] = acc_val

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep

            """if nll_val.item() > best_val_loss:
                best_val_loss = nll_val.item()
                #best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                #best_epoch = ep"""
            
            if ep % 100 == 0:
                print(f"Epoch {ep:3d},  Train loss: {nll.item():.3f},  Train accuracy: {acc}, Validation accuracy: {acc_val}")

    print(f"best validation accuracy: {best_val_acc} at epoch: {best_epoch}")
    model.load_state_dict(best_model_state)
    logits_best_model, _, _ = model(X_train)
    nll_best = F.cross_entropy(logits_best_model.view(-1, logits_best_model.size(-1)), y_val.view(-1).long())
    pA_best_model = F.softmax(logits_best_model, dim=-1)
    pA_per_epoch[str(best_epoch)] = pA_best_model
    training_dict = {"predictions": pA_best_model,
            "weights": best_model_state,
            "best_model": model,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc}
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict

def train_ablated_noblocks(model, X_train, y_train, X_test, y_test, device, p_target=None, p_test = None, epochs=5000, lr=0.001):
    train_elbos = []
    val_elbos = []
    kl_vals = []
    kl_test_vals = []
    accuracy = []
    test_accuracy = []
    test_loss = []
    best_kl = float('inf')
    best_test_kl = float('inf')
    y_train = torch.argmax(y_train, dim=-1)
    y_test = torch.argmax(y_test, dim=-1)
    pA_per_epoch = {} # dictionary to store pA predictions for later comparison

    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits, _, _ = model(X_train)  # (B, T, A)
        
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y_train.view(-1).long())
        nll.backward()
        opt.step()
        train_elbos.append(nll)
        #print(model.dec.h0.grad)

        ### accuracy ###
        # Predictions: take argmax over logits
        preds = logits.view(-1, logits.size(-1)).argmax(dim=-1)

        # Ground truth labels
        targets = y_train.view(-1).long()

        # Accuracy: compare and compute mean
        acc = (preds == targets).float().mean().item()
        accuracy.append(acc)

        """val_loss = val_elbo_ablated(model, X_val, y_val)
        train_elbos.append(nll.item())
        val_elbos.append(val_loss)"""

        """if ep % 50 == 0:
            print(f"Epoch {ep:3d}  Train loss {nll.item():.3f}")"""

        # === KL Divergence Monitoring ===
        if p_target is not None:
            model.eval()
            with torch.no_grad():
                p_target = p_target.to(device)
                p_model = F.softmax(logits, dim=-1)  # (B, T, A)
                p_model = p_model[:,:,0]
                #print("p_target shape:", p_target.shape)
                #print("p_model  shape:", p_model.shape)
                kl = compute_kl_divergence_bernoulli(p_target, p_model)
                kl_vals.append(kl)

                logits_test, _, _ = model(X_test)  # (B, T, A)
        
                nll_test = F.cross_entropy(logits_test.view(-1, logits_test.size(-1)), y_test.view(-1).long())
                preds_test = logits_test.view(-1, logits_test.size(-1)).argmax(dim=-1)

                # Ground truth labels
                targets_test = y_test.view(-1).long()

                # Accuracy: compare and compute mean
                acc_test = (preds_test == targets_test).float().mean().item()
                test_accuracy.append(acc_test)
                test_loss.append(nll_test)

                #test kl divergence
                p_test = p_test.to(device)
                p_model_test = F.softmax(logits_test, dim=-1)  # (B, T, A)
                p_model_test = p_model_test[:,:,0]
                #print("p_target shape:", p_target.shape)
                #print("p_model  shape:", p_model.shape)
                kl_test = compute_kl_divergence_bernoulli(p_test, p_model_test)
                kl_test_vals.append(kl_test)

                if kl < best_kl:
                    best_kl = kl
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                if kl_test < best_test_kl:
                    best_test_kl = kl_test
                    best_model_state_test = {k: v.clone() for k, v in model.state_dict().items()}
                    best_test_epoch_kl = ep

                if ep % 100 == 0:
                    print(f"Epoch {ep:3d}  Train loss {nll.item():.3f}  KL {kl:.3f}  accuracy {acc}  test loss {nll_test}  test accuracy {acc_test}  test kl {kl_test}")

    
    best_epoch_test_acc = test_accuracy.index(max(test_accuracy))+1
    best_epoch_test_loss = test_loss.index(min(test_loss))+1
    best_epoch_train_loss = train_elbos.index(min(train_elbos))+1
    best_epoch_acc = accuracy.index(max(accuracy))+1
    #normalized_ll = torch.exp(nll_test / targets_test.numel()).cpu().item()

    print(f"Best epoch kl: {best_epoch}")
    print(f"Best test kl epoch: {best_test_epoch_kl}")
    print(f"Best loss epoch: {best_epoch_train_loss}")
    print(f"Best test loss epoch: {best_epoch_test_loss}")
    print(f"Best accuracy loss epoch: {best_epoch_acc}")
    print(f"Best test accuracy loss epoch: {best_epoch_test_acc}")
    #print(f"normalzed_ll: {normalized_ll}")

    model.load_state_dict(best_model_state)
    logits_best_model, _, _ = model(X_train)
    pA_best_model = F.softmax(logits_best_model, dim=-1)
    pA_per_epoch[str(best_epoch)] = pA_best_model
    training_dict = {"predictions": pA_best_model,
            "weights": best_model_state,
            "best_model": model,
            "best_epoch": best_epoch,
            "best_kl": best_kl}
            #"normalized_ll": normalized_ll}
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict