#!/usr/bin/env python3
# compare_models.py
"""
Train three variants of the latent-RNN bandit model and plot their test log-likelihoods.

 * Model-psy  : MLP encoder with 4 psychometric factors  → z
 * Model-emb  : lookup-table (learnable μ_i, logσ²_i)    → z
 * Model-abl  : no latent state, shared initial GRU h₀
"""
import os, random, math, json, itertools, warnings, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from data_utils import load_preprocess_data_VAE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# reproducibility ----------------------------------------------------------------
torch.manual_seed(0);  np.random.seed(0);  random.seed(0)
torch.set_printoptions(linewidth=200)
# --------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# 1.  model definitions ----------------------------------------------------------
# --------------------------------------------------------------------------------
hidden_dim = 100
class PsychometricEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim=hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim,hidden_dim),    nn.ReLU(),
                                 nn.Linear(hidden_dim,hidden_dim),    nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden_dim,z_dim), nn.Linear(hidden_dim,z_dim)
    def forward(self,x):
        #print(f"[{self.__class__.__name__}] psychometrics shape: {x.shape}")
        
        x = x.float()
        h = self.net(x)
        #print(f"[{self.__class__.__name__}] hidden shape: {h}")
        return self.mu(h), self.logvar(h)
    

class LookupEncoder(nn.Module):
    def __init__(self, n_participants, z_dim):
        super().__init__()
        #print(f"number of participants: {n_participants}, z_dim: {z_dim}")
        self.mu     = nn.Embedding(n_participants, z_dim)
        self.logvar = nn.Embedding(n_participants, z_dim)
        nn.init.zeros_(self.mu.weight);  nn.init.zeros_(self.logvar.weight) #initialize to 0
    def forward(self, idx):
        return self.mu(idx), self.logvar(idx)

"""class LookupEncoderZ(nn.Module):
    def __init__(self, n_participants, z_dim):
        super().__init__()
        #print(f"number of participants: {n_participants}, z_dim: {z_dim}")
        self.z = torch.randn(n_participants, z_dim, dtype=torch.float32)
        # store as a buffer so it's not trainable but will move with the module and be checkpointed
        self.register_buffer("z_table", self.z, persistent=True)

    @torch.no_grad()
    def resample_(self, seed: int = None):
        Optionally resample all z's in-place (e.g., for ablations).
        self.z_table.copy_(torch.randn_like(self.z_table))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        
        idx: LongTensor of participant ids with arbitrary shape, e.g. (B,) or (B,T).
        Returns z with the same leading shape + (z_dim,).
        
        # ensure long dtype for indexing
        idx = idx.long()
        z = self.z_table.index_select(0, idx.reshape(-1))
        return z.view(*idx.shape, self.z_table.size(1))"""

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

class Decoder(nn.Module):
    def __init__(self, in_dim, z_dim, hid=hidden_dim, A=2):
        super().__init__()
        #self.rnn   = nn.GRU(in_dim+z_dim, hid, batch_first=True)
        self.rnn   = nn.GRU(in_dim, hid, batch_first=True)
        self.lin   = nn.Linear(hid, A)
        self.z2h0  = nn.Linear(z_dim, hid)
        self.z20_mlp = nn.Sequential(nn.Linear(z_dim,hid), nn.ReLU(),
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
        #rnn_input = torch.cat([seq, zexp], -1)
        rnn_input = seq

        if torch.isnan(rnn_input).any() or torch.isinf(rnn_input).any():
            print("⚠️ NaNs or Infs in rnn_input!")
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print("⚠️ NaNs or Infs in hidden state!")
        out, hidden = self.rnn(rnn_input, hidden)
        logits = self.lin(out)

        return logits, hidden
        
    """def forward(self, seq, z, ID_test=False):
        if ID_test:
            if z.dim() > 2:
                # Flatten extra dimensions: e.g., (B, 1, 1, z_dim) -> (B, z_dim)
                z = z.view(z.size(0), -1)  # Keep batch dimension, flatten the rest
                # Or more specifically: z = z.squeeze() to remove singleton dimensions
            
            # Ensure z has the right shape
            if z.dim() == 1:
                z = z.unsqueeze(0)  # (z_dim) -> (1, z_dim)

        #z shape: [B, z_dim]
        h0 = self.z2h0(z).unsqueeze(0)          # (1,B,H)
        zexp = z.unsqueeze(1).expand(-1, seq.size(1), -1)
        out,_ = self.rnn(torch.cat([seq,zexp], -1), h0)
        return self.lin(out) """                  # (B,T,A)
    

class Decoder_nocat(nn.Module):
    def __init__(self, in_dim, z_dim, hid=hidden_dim, A=2):
        super().__init__()
        self.rnn   = nn.GRU(in_dim, hid, batch_first=True)
        self.lin   = nn.Linear(hid, A)
        self.z2h0  = nn.Linear(z_dim, hid)
    def forward(self, seq, z):
        # no zexp concatenation
        h0 = self.z2h0(z).unsqueeze(0)          # (1,B,H)
        out,_ = self.rnn(seq, h0)
        return self.lin(out)

class LatentRNN(nn.Module):
    def __init__(self, encoder, z_dim=3, in_dim=2, hid=hidden_dim, A=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder(in_dim, z_dim, hid, A)
    def forward(self, xenc, blocks):            # blocks: (B,Bk,T,2)
        mu, lv  = self.encoder(xenc)
        std     = torch.exp(0.5*lv)
        z       = mu + torch.randn_like(std)*std
        #print(f"[{self.name}] mu shape: {mu.shape}, lv shape: {lv.shape}, z shape: {z.shape}")
        h0_ = self.decoder.z2h0(z)          # (B,H) - just for usage outside of RNN
        outs = [ self.decoder(blocks[:,b], z) for b in range(blocks.size(1)) ]
        return torch.stack(outs,1), mu, lv, z, h0_

class LatentRNN_secondstep(nn.Module):
    def __init__(self, encoder, z_dim=3, in_dim=2, hid=hidden_dim, A=2, decoder=None):
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

        #logits_list = []
        #hidden_list = []
        #for b in range(blocks.size(1)):
        #    logits_b, hidden_b = self.decoder(blocks[:, b], z)
        #    logits_list.append(logits_b)
        #    hidden_list.append(hidden_b)
        #hidden = torch.stack(hidden_list, dim=1) # in case I want access to this
        #return torch.stack(logits_list, dim=1), mu, lv, z, h0_
    
class LatentRNNz(nn.Module):
    def __init__(self, encoder, z_dim=3, in_dim=2, hid=hidden_dim, A=2, block_structure=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder(in_dim, z_dim, hid, A)
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


    
class LatentRNN_nocat(nn.Module):
    def __init__(self, encoder, z_dim=3, in_dim=2, hid=hidden_dim, A=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder_nocat(in_dim, z_dim, hid, A)
    def forward(self, xenc, blocks):            # blocks: (B,Bk,T,2)
        mu, lv  = self.encoder(xenc)
        std     = torch.exp(0.5*lv)
        z       = mu + torch.randn_like(std)*std
        #print(f"[{self.name}] mu shape: {mu.shape}, lv shape: {lv.shape}, z shape: {z.shape}")
        outs = [ self.decoder(blocks[:,b], z) for b in range(blocks.size(1)) ]
        return torch.stack(outs,1), mu, lv, z

class AblatedDecoder(nn.Module):
    def __init__(self, in_dim, hid=hidden_dim, A=2):
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
    def __init__(self, in_dim=2, hid=hidden_dim, A=2, block_structure = True):
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
         
        
    
    
class GlobalLatentRNN(nn.Module):
    def __init__(self, z_dim=3, in_dim=2, hid=hidden_dim, A=2):
        super().__init__()
        self.decoder = Decoder(in_dim, z_dim, hid, A)
        self.z = nn.Parameter(torch.randn(1, z_dim))  # Learnable global z
        
    def forward(self, blocks):  # blocks: (B, Bk, T, 2)
        z_broadcasted = self.z.expand(blocks.size(0), -1)  # (B, z_dim)
        outs = [self.decoder(blocks[:, b], z_broadcasted) for b in range(blocks.size(1))]
        return torch.stack(outs, 1)

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




# --------------------------------------------------------------------------------
# 2.  utilities  -----------------------------------------------------------------
# --------------------------------------------------------------------------------
nr_epochs = 60

def gaussian_nll_point(mu, logvar, z_T):

    var = logvar.exp()
    return 0.5 * (((z_T - mu)**2) / (var + 1e-8) + logvar).sum(-1).mean()

def step_two_loss(lambda_post, lambda_pol, mu, logvar, z_T, L_pol):
    print(f"z shape: {mu.shape}")
    loss = lambda_post * gaussian_nll_point(mu, logvar, z_T) + lambda_pol * L_pol
    return loss 

def elbo_loss(logits, targets, mu, lv, beta=0.5):
    B,Bk,T,A = logits.shape
    nll = F.cross_entropy(logits.view(-1,A), targets.long().view(-1), reduction='mean')
    kl  = -.5*torch.mean(torch.sum(1+lv-mu.pow(2)-lv.exp(),1))
    return nll + beta*kl, nll.detach(), kl.detach()


def elbo_lossZ(logits, targets):
    B,Bk,T,A = logits.shape
    nll = F.cross_entropy(logits.reshape(-1,A), targets.long().reshape(-1), reduction='mean')
    return nll, nll.detach()

def train_latentrnn(model, xenc, blocks, y, epochs=nr_epochs, lr=1e-3):
    #print(f"x_enc shape: {xenc.shape}, blocks shape: {blocks.shape}, y shape: {y.shape}")
    train_elbos = []
    val_elbos = []
    opt = torch.optim.Adam(model.parameters(), lr)
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        logits, mu,lv = model(xenc, blocks)
        #print(f"[{model.name}] logits shape: {logits.shape}, mu shape: {mu.shape}, lv shape: {lv.shape}")
        loss, nll, kl = elbo_loss(logits,y,mu,lv)
        loss.backward(); opt.step()
        #val_loss = val_elbo(model, xenc_val, Xs_val, Ys_val, mu, lv)
        if ep%50==0: print(f"[{model.name}] epoch {ep:3d}  loss {loss.item():.3f}"\
                           f" → val epoch {ep}: ELBO=")
        train_elbos.append(loss)
        #val_elbos.append(val_loss)
    return model, mu, lv, train_elbos, val_elbos


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
        "h0": h0_best
    }

    return model, train_losses, kl_vals, pA_per_epoch, training_dict

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
        "best_val_acc": best_val_acc
    }
    print(f"best validation accuracy: {best_val_acc} in epoch: {best_epoch}")

    return model, train_losses, val_losses, pA_per_epoch, training_dict

@torch.no_grad()
def extract_latents(model, xenc):
    """
    xenc: Tensor of shape (B, Bk, T, in_dim)
    Returns:
        mu: (B, z_dim) participant-specific latent embeddings
    """
    model.eval()
    mu, logvar = model.encoder(xenc, return_per_timestep=False)
    return mu.cpu().numpy()

def train_latentrnn_IDRNN(model, xenc, blocks, y, lookup_z, xenc_val, y_val, p_target, epochs=nr_epochs, patience = 300, lr=1e-3):
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
        lambda_post, lambda_pol = 1.0, 1.0 #try setting lambda pol to 0 in order to only weigh z in training
        loss = step_two_loss(lambda_post, lambda_pol, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()

        train_elbos.append(loss.item())

        model.eval()
        val_logits, _, _, _, _ = model(xenc_val_prefix, xenc_val_prefix, sample_z=False)
        val_policy_loss, val_nll = elbo_lossZ(val_logits, yval_prefix)
        val_loss = step_two_loss(lambda_post, lambda_pol, mu, lv, lookup_z, val_policy_loss)
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

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_loss_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            
        else:
            epochs_no_improve += 1

        if ep % 100 == 0:
            print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  val loss {val_loss.item():.3f},  train_acciracy: {final_acc},  val_accuracy: {final_val_acc}")
            

        #if epochs_no_improve >= patience:
            #print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            #break

    #best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    window = 40
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

def train_latentrnn_IDRNN_palminteri(model, xenc, blocks, y, lookup_z, xenc_val, y_val, z_val_lookup, epochs=nr_epochs, patience = 300, lr=1e-3):
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

        lambda_post, lambda_pol = 1.0, 1.0 #try setting lambda pol to 0 in order to only weigh z in training
        loss = step_two_loss(lambda_post, lambda_pol, mu, lv, lookup_z, policy_loss)
        loss.backward()
        opt.step()
        train_elbos.append(loss.item())

        xenc_val_prefix = xenc_val[:, :, :prefix_len, :]
        yval_prefix = y_val[:, :, :prefix_len]

        model.eval()
        val_logits, mu_val, lv_val, _, _ = model(xenc_val_prefix, xenc_val_prefix, sample_z=False)
        val_policy_loss, val_nll = elbo_lossZ(val_logits, yval_prefix)
        val_loss = step_two_loss(lambda_post, lambda_pol, mu_val, lv_val, z_val_lookup, val_policy_loss)
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
    window = 20
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


def train_latentrnn_early_stopping(model, xenc, blocks, y, blocks_val, y_val, 
                                   epochs=nr_epochs, lr=1e-3, patience=300):
    opt = torch.optim.Adam(model.parameters(), lr)
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_elbos = []
    val_elbos = []

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits, mu, lv, z, h0 = model(xenc, blocks)
        loss, nll, kl = elbo_loss(logits, y, mu, lv)
        loss.backward(); opt.step()

        train_elbos.append(loss.item())
        #print(f"xenc_val shape: {xenc_val.shape}, blocks_val shape: {blocks_val.shape}, y_val shape: {y_val.shape}")
        #logits_val, mu_val, lv_val = model(xenc_val, blocks_val)
        #val_loss, _ = val_elbo(model, blocks_val, y_val, mu, lv)
        model.eval()
        logits_val, _, _, _,_ = model(xenc, blocks_val)
        val_loss, _, _ = elbo_loss(logits_val, y_val, mu, lv) # using the same mu, lv from training
        #val_loss = -val_loss  # Convert to positive loss for early stopping

        val_elbos.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_mu = mu
            best_lv = lv
        else:
            epochs_no_improve += 1

        if ep % 50 == 0:
            print(f"[{model.name}] epoch {ep:3d}  train loss {loss.item():.3f}  val loss {val_loss:.3f}")

        if epochs_no_improve >= patience:
            print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
    return model, best_mu, best_lv, train_elbos, val_elbos, h0


def train_latentrnn_early_stoppingZ(model, xenc, blocks, y, blocks_val, y_val, 
                                   epochs=nr_epochs, lr=1e-3, patience=300):
    opt = torch.optim.Adam(model.parameters(), lr)
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_elbos = []
    val_elbos = []

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits, z, h0 = model(xenc, blocks)
        loss, nll = elbo_lossZ(logits, y)
        loss.backward(); opt.step()

        train_elbos.append(loss.item())
        #print(f"xenc_val shape: {xenc_val.shape}, blocks_val shape: {blocks_val.shape}, y_val shape: {y_val.shape}")
        #logits_val, mu_val, lv_val = model(xenc_val, blocks_val)
        #val_loss, _ = val_elbo(model, blocks_val, y_val, mu, lv)
        model.eval()
        logits_val, _, _,  = model(xenc, blocks_val)
        val_loss, _= elbo_lossZ(logits_val, y_val) # using the same mu, lv from training
        #val_loss = -val_loss  # Convert to positive loss for early stopping

        val_elbos.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_z = z
        else:
            epochs_no_improve += 1

        if ep % 50 == 0:
            print(f"[{model.name}] epoch {ep:3d}  train loss {loss.item():.3f}  val loss {val_loss:.3f}")

        if epochs_no_improve >= patience:
            print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
    return model, best_z, train_elbos, val_elbos, h0

@torch.no_grad()
def val_elbo(model, blocks, y, mu, lv, N=30, beta=0.1):
    model.eval()
    B, Bk, T, _ = y.shape
    mu_emp, lv_emp = sample_participant_moments(mu, lv, B)
    std = torch.exp(0.5 * lv_emp)
    LL = []
    for _ in range(N):
        z = mu_emp + torch.randn_like(std) * std
        lp = []
        for b in range(Bk):
            logits = model.decoder(blocks[:, b], z)  # (B, T, A)
            log_probs = F.log_softmax(logits, -1)    # (B, T, A)
            #print(f"log_probs shape: {log_probs.shape}, y[:, b].shape: {y[:, b].shape}")
            lp.append(log_probs.gather(-1, y[:, b].long()).squeeze(-1))  # (B, T)
        LL.append(torch.stack(lp, 1))  # [N, B, Bk, T]
    ll = torch.logsumexp(torch.stack(LL, 0), 0) - math.log(N)  # (B, Bk, T) - averaged over N samples
    per_part = ll.sum((1, 2))  # (B,) - adding all likelihoods per participant
    return per_part.mean().item(), per_part

def compute_kl_divergence(p_target, p_model, eps=1e-8):
    """
    Compute KL(p_target || p_model) for each trial and average.
    Both p_target and p_model should have shape [B, T, A]
    """
    p_target = p_target.clamp(min=eps)
    p_model = p_model.clamp(min=eps)

    kl = (p_target * (p_target.log() - p_model.log())).sum(-1)  # shape [B, T]
    return kl.mean().item()  # average over batch and time

def compute_kl_divergence_bernoulli(p_true, p_pred):
    epsilon = 1e-10  # To avoid division by zero and log(0) issues
    kl_div = p_true * ((p_true + epsilon) / (p_pred + epsilon)).log() + \
             (1 - p_true) * ((1 - p_true + epsilon) / (1 - p_pred + epsilon)).log()
    return kl_div.mean().item()

def train_ablated(model, blocks, y, Xs_val, Ys_val, p_target, epochs=nr_epochs, lr=1e-3):
    train_elbos = []
    val_elbos = []
    kl_vals = []
    best_kl = float('inf')
    opt = torch.optim.Adam(model.parameters(), lr)
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        logits, _, _ = model(blocks)
        nll = F.cross_entropy(logits.view(-1,2), y.long().view(-1))
        nll.backward(); opt.step()
        val_loss = val_elbo_ablated(model, Xs_val, Ys_val)
        if ep%50==0: print(f"[{model.name}] epoch {ep:3d}  loss {nll.item():.3f}")
        train_elbos.append(nll)
        val_elbos.append(val_loss)

        # === KL Divergence Monitoring ===
        if p_target is not None:
            model.eval()
            with torch.no_grad():
                #logits_val, _, _ = model(Xs_val)
                p_target = p_target.to(device)
                p_model_tr = F.softmax(logits, dim=-1)  # (B, Bk, T, A)
                #print(f"p_target after transform: {p_target.shape}")
                B, Bk, T, A = p_model_tr.shape
                #print(f"p_model_tr shape: {p_model_tr.shape}") # has shape of val data
                p_model_tr = p_model_tr.view(B, Bk*T, A).to(device)
                #print(f"p_model_tr after transform: {p_model_tr.shape}")
                
                
                kl = compute_kl_divergence_bernoulli(p_target, p_model_tr)
                kl_vals.append(kl)

                if kl < best_kl:
                    best_kl = kl
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                print(f"[{model.name}] Epoch {ep:3d}  train loss {nll.item():.3f}  val loss {val_loss:.3f}  val KL {kl:.3f}")
    print(f"best epoch: {ep}")
    model.load_state_dict(best_model_state)
    return model, train_elbos, val_elbos, kl_vals

def train_ablated_noblocks(model, X_train, y_train, X_test, y_test, p_target=None, p_test = None, epochs=5000, lr=0.001):
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

    print(f"Best epoch kl: {best_epoch}")
    print(f"Best test kl epoch: {best_test_epoch_kl}")
    print(f"Best loss epoch: {best_epoch_train_loss}")
    print(f"Best test loss epoch: {best_epoch_test_loss}")
    print(f"Best accuracy loss epoch: {best_epoch_acc}")
    print(f"Best test accuracy loss epoch: {best_epoch_test_acc}")

    model.load_state_dict(best_model_state)
    logits_best_model, _, _ = model(X_train)
    pA_best_model = F.softmax(logits_best_model, dim=-1)
    pA_per_epoch[str(best_epoch)] = pA_best_model
    training_dict = {"predictions": pA_best_model,
            "weights": best_model_state,
            "best_model": model,
            "best_epoch": best_epoch,
            "best_kl": best_kl}
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict

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

            """if nll_val.item() < best_val_loss:
                best_val_loss = nll_val.item()
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep"""
            
            if ep % 100 == 0:
                print(f"Epoch {ep:3d},  Train loss: {nll.item():.3f},  Train accuracy: {acc}, Validation accuracy: {acc_val}")

    print(f"best validation accuracy: {best_val_acc} at epoch: {best_epoch}")
    model.load_state_dict(best_model_state)
    logits_best_model, _, _ = model(X_train)
    pA_best_model = F.softmax(logits_best_model, dim=-1)
    pA_per_epoch[str(best_epoch)] = pA_best_model
    training_dict = {"predictions": pA_best_model,
            "weights": best_model_state,
            "best_model": model,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc}
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict

def train_ablated_early_stopping(model, blocks, y, Xs_val, Ys_val, epochs=nr_epochs, lr=1e-3, patience=300):
    train_elbos = []
    val_elbos = []
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        logits, _, _ = model(blocks)
        loss = F.cross_entropy(logits.view(-1,2), y.long().view(-1))
        loss.backward(); opt.step()
        val_loss = val_elbo_ablated(model, Xs_val, Ys_val)
        train_elbos.append(loss)
        val_elbos.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ep % 50 == 0:
            print(f"[{model.name}] epoch {ep:3d}  train loss {loss.item():.3f}  val loss {val_loss:.3f}")

        if epochs_no_improve >= patience:
            print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
        
    return model, train_elbos, val_elbos

def train_global_latent(model, blocks, y, Xs_val, Ys_val, epochs=100, lr=1e-3, patience = 300):
    train_elbos = []
    val_elbos = []
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    opt = torch.optim.Adam(model.parameters(), lr)

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits = model(blocks)  # logits: (B, Bk, T, A)
        loss = F.cross_entropy(logits.view(-1, 2), y.long().view(-1))
        loss.backward(); opt.step()
        train_elbos.append(loss.item())
        val_loss = val_elbo_global(model, Xs_val, Ys_val)
        val_elbos.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ep % 50 == 0:
            print(f"[{model.name}] epoch {ep:3d}  train loss {loss.item():.3f}  val loss {val_loss:.3f}")

        if epochs_no_improve >= patience:
            print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
    return model, train_elbos

def _safe_gather(logits, y_block):
    """
    logits : (B, T, A)
    y_block: (B, T)       ← **no extra dims yet**
    """
    idx = y_block.unsqueeze(-1).long()         # (B, T, 1)

    # DEBUG — remove once script runs
    assert logits.dim() == idx.dim() == 3, \
        f"gather shape mismatch: logits {logits.shape}, idx {idx.shape}"
    # -------------------------------------
    return (F.log_softmax(logits, -1).gather(-1, idx).squeeze(-1))

def mixture_moments(mu, lv, participant_dim):
        # 1) recover σ_i^2
    sigma2 = torch.exp(lv)                # (M, V)

    # 2) mixture mean
    mu_mix    = mu.mean(dim=0)            # (   V)

    # 3) mixture variance
    var_mix   = sigma2.mean(dim=0) \
                + (mu**2).mean(dim=0) \
                - mu_mix**2             # (   V)

    # 4) mixture log-variance
    lv_mix    = torch.log(var_mix)        # (   V)

    # 5) expand to test-participants
    mu_test   = mu_mix.unsqueeze(0).expand(participant_dim, -1)   # (B, V)
    lv_test   = lv_mix.unsqueeze(0).expand(participant_dim, -1)   # (B, V)
    return mu_test, lv_test

def sample_participant_moments(mu, lv, participant_dim):
    """
    For each test participant, randomly sample a (mu, lv) pair 
    from a participant in the training set.
    
    Args:
        mu (Tensor): Means from training participants, shape (M, V)
        lv (Tensor): Log-variances from training participants, shape (M, V)
        participant_dim (int): Number of test participants (B)

    Returns:
        mu_test (Tensor): Sampled means for test participants, shape (B, V)
        lv_test (Tensor): Sampled log-variances for test participants, shape (B, V)
    """
    M, V = mu.shape

    # Sample indices with replacement from training participants
    sampled_indices = torch.randint(0, M, (participant_dim,))

    # Gather the corresponding mu and lv
    mu_test = mu[sampled_indices]         # (B, V)
    lv_test = lv[sampled_indices]         # (B, V)

    return mu_test, lv_test


@torch.no_grad()
def test_latentrnn(model, mu, lv, blocks, y, N=100, first_n=None):
    model.eval()
    B, Bk, T, _ = y.shape  # Now accounts for [B, Bk, T, 1]
    y = y.squeeze(-1)
    if first_n is not None:
        blocks = blocks[:, :, :first_n]        # (B, Bk, first_n, 2)
        y      = y[:, :, :first_n]
    mu_empirical, lv_empirical = sample_participant_moments(mu, lv, B)
    #print(f"[{model.name}] mu_empirical shape: {mu_empirical.shape}, lv_empirical shape: {lv_empirical.shape}")
    #print(f"lv empirical: {lv_empirical}")
    #print(f"mu empirical: {mu_empirical}")
    std = torch.exp(0.5 * lv_empirical)
    LL = []
    for _ in range(N):
        z = mu_empirical + torch.randn_like(std) * std
        lp = []
        for b in range(Bk):
            logits = model.decoder(blocks[:, b], z)  # (B, T, A)
            log_probs = F.log_softmax(logits, -1)    # (B, T, A)
            lp.append(log_probs.gather(-1, y[:, b].unsqueeze(-1).long()).squeeze(-1))  # (B, T)
        LL.append(torch.stack(lp, 1))  # [N, B, Bk, T]
    ll = torch.logsumexp(torch.stack(LL, 0), 0) - math.log(N)  # (B, Bk, T) - averaged over N samples
    per_part = ll.sum((1, 2))  # (B,) - adding all likelihoods per participant
    return per_part.mean().item(), per_part

@torch.no_grad()
def test_latentrnnZ(model, z, blocks, y, N=100, first_n=None):
    model.eval()
    B, Bk, T, _ = y.shape  # Now accounts for [B, Bk, T, 1]
    y = y.squeeze(-1)
    if first_n is not None:
        blocks = blocks[:, :, :first_n]        # (B, Bk, first_n, 2)
        y      = y[:, :, :first_n]
    # sample one z vector from the learned z's for each test participant
    
             # (B, V)
    #print(f"[{model.name}] mu_empirical shape: {mu_empirical.shape}, lv_empirical shape: {lv_empirical.shape}")
    #print(f"lv empirical: {lv_empirical}")
    #print(f"mu empirical: {mu_empirical}")
    LL = []
    for _ in range(N):
        # sample participant index with replacement so that each test participant gets a z from a random training participant
        M = z.size(0)
        sampled_indices = torch.randint(0, M, (B,))
        z_sampled = z[sampled_indices]
        lp = []
        for b in range(Bk):
            logits = model.decoder(blocks[:, b], z_sampled)  # (B, T, A)
            log_probs = F.log_softmax(logits, -1)    # (B, T, A)
            lp.append(log_probs.gather(-1, y[:, b].unsqueeze(-1).long()).squeeze(-1))  # (B, T)
        LL.append(torch.stack(lp, 1))  # [N, B, Bk, T]
    ll = torch.logsumexp(torch.stack(LL, 0), 0) - math.log(N)  # (B, Bk, T) - averaged over N samples
    per_part = ll.sum((1, 2))  # (B,) - adding all likelihoods per participant
    return per_part.mean().item(), per_part

def log_prob_diag_gaussian(z, mu, logvar):
    """
    Simplified version for single elements
    z: (N, z_dim)
    mu: (N, z_dim) or (z_dim,)  
    logvar: (N, z_dim) or (z_dim,)
    returns: (N,) - log probabilities per sample
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0).expand(z.shape[0], -1)
    if logvar.dim() == 1:
        logvar = logvar.unsqueeze(0).expand(z.shape[0], -1)
    
    var = torch.exp(logvar)
    log_density = -0.5 * (((z - mu)**2) / var + logvar + math.log(2 * math.pi))
    return log_density.sum(-1)  # Sum over z_dim, not batch

# Causal Importance-Weighted Test Function (per-timestep z)
@torch.no_grad()
def test_latentrnn_secondstep_causal(model, blocks, y, N=500, num_actions=2, first_n=None):
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

    log_likelihoods = torch.zeros(B, Bk, device=device)
    z_samples = torch.randn((N, B, z_dim), device=device)  # (N, B, z_dim)
    #print(f"z_samples shape: {z_samples.shape}")
    #print(f"z_samples: {z_samples}")
    z_samples = z_samples.unsqueeze(2).unsqueeze(3)  # (N, B, 1, 1, z_dim) for broadcasting
    z_samples = z_samples.expand(-1, -1, Bk, -1, -1)
    #print(f"z_samples expanded shape: {z_samples.shape}")
    #print(f"z_samples after formatting: {z_samples}")
    



    for t in range(1, T + 1):
        # Slice input up to time t
        blocks_t = blocks[:, :, :t]  # (B, Bk, t, in_dim)

        # Posterior q(z_t | d_{1:t})
        mu_t, logvar_t = model.encoder(blocks_t, return_per_timestep=True)
        #print(f"mu_t shape: {mu_t.shape}, logvar_t shape: {logvar_t.shape}")
        #print(f"mu_t: {mu_t}, logvar_t: {logvar_t}")
        mu_t = mu_t.unsqueeze(0).expand(N, -1, -1, -1, -1)         # (N, B, Bk, t, z_dim)
        logvar_t = logvar_t.unsqueeze(0).expand(N, -1, -1, -1, -1)   # (N, B, Bk, t, z_dim)
        #print(f"mu_t expanded shape: {mu_t.shape}, logvar_t expanded shape: {logvar_t.shape}")
        #print(f"mu_t expanded: {mu_t}, logvar_t expanded: {logvar_t}")
        #now mu_t, logvar_t and z_samples all have shape (N, B, Bk, t, z_dim)

        # Importance samples from prior p(z)
        
        
        # Compute log q(z_t | d_{1:t}) and log p(z_t)
        log_qzx = log_prob_diag_gaussian(z_samples, mu_t, logvar_t)  # (N, B, Bk, t)
        zeros_mu = torch.zeros_like(mu_t)
        zeros_lv = torch.zeros_like(logvar_t)
        log_pz = log_prob_diag_gaussian(z_samples, zeros_mu, zeros_lv)  # (N, B, Bk, t)
        log_w = log_qzx - log_pz  # (N, B, Bk, t)

        # Evaluate decoder P(a_t | d_{1:t}, z_t)
        LL_t = torch.zeros(N, B, Bk, device=device)
        for i in range(N):
            z_i = z_samples[i]  # (B, z_dim)
            #print(f"z_i shape before squeeze: {z_i.shape}")
            z_i = z_i[:, 0, 0, :]

            #print(f"z_i shape after squeeze: {z_i.shape}")
            for b in range(Bk):
                logits = model.decoder(blocks[:, b, :t], z_i)  # (B, t, A)
                #print(f"logits shape: {logits.shape}")
                log_probs = F.log_softmax(logits, dim=-1)  # (B, A)
                #print(f"log_probs shape: {log_probs.shape}")
                action_t = y[:, b, t - 1]  # (B,)
                #print(f"action_t shape: {action_t.shape}")
                logp_t = log_probs[:, -1, :].gather(-1, action_t.unsqueeze(-1).long()).squeeze(-1)  # (B,)
                #print(f"logp_t shape: {logp_t.shape}")
                #print(f"logp_t: {logp_t}")
                LL_t[i, :, b] = logp_t  # (N, B, Bk) - likelihood for action t at each (sample, participant, block)
        
        # Step 5: Combine with log_w (shape: N, B, Bk, t)
        log_w_t = log_w[:, :, :, t - 1]  # (N, B, Bk)
        # Combine log p(a_t | d_{<t}, z) and importance weights
        log_weighted_t = LL_t + log_w_t  # (N, B, Bk)

        # Average across blocks
        #log_weighted_t = log_weighted_t.mean(dim=2)  # (N, B)
        # Marginalize over z-samples using logsumexp
        log_px_t = torch.logsumexp(log_weighted_t, dim=0) - math.log(N)  # (B,)
        # Accumulate per-timestep log-likelihood
        log_likelihoods += log_px_t  # (B,)
    log_likelihoods_per_participant = log_likelihoods.sum(dim=1)  # (B,)
    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant  # scalar, (B,)


# Causal Importance-Weighted Test Function (per-timestep z)
@torch.no_grad()
def test_latentrnn_secondstep_causal_2(model, blocks, y, N=100, num_actions=2, first_n=None):
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
    z_samples = torch.randn((N, B, z_dim), device=device)  # (N, B, z_dim)
    #print(f"z_samples shape: {z_samples.shape}")
    #print(f"z_samples: {z_samples}")
    z_samples = z_samples.unsqueeze(2).unsqueeze(3)  # (N, B, 1, 1, z_dim) for broadcasting
    #print(f"z_samples expanded shape: {z_samples.shape}")
    #print(f"z_samples after formatting: {z_samples}")
    
    for participant_idx in range(B):
        print(f"Processing participant {participant_idx+1}/{B}")
        participant_log_likelihood = 0.0

        for block_idx in range(Bk):
            block_log_likelihood = 0.0

            for t in range(1,T+1):
                data_up_to_t = blocks[participant_idx:participant_idx+1, block_idx:block_idx+1, :t]  # (1, 1, t, in_dim)

                #get posterior parameters for time t
                mu_t, logvar_t = model.encoder(data_up_to_t, return_per_timestep=True)

                # Extract parameters for the last timestep
                mu_t = mu_t[0, 0, -1]  # (z_dim,) - last timestep of this block
                logvar_t = logvar_t[0, 0, -1]  # (z_dim,)
                # Get samples for this participant
                z_participant = z_samples[:, participant_idx]  # (N, 1,1,z_dim)
                #print(f"z_participant shape before squeeze: {z_participant.shape}")
                z_participant = z_participant.squeeze(2)  # (N, 1, z_dim)
                z_participant = z_participant.squeeze(1)

                # Expand posterior parameters for all samples
                mu_t_expanded = mu_t.unsqueeze(0).expand(N, -1)  # (N, z_dim)
                logvar_t_expanded = logvar_t.unsqueeze(0).expand(N, -1)  # (N, z_dim)

                # Compute log q(z|d_{1:t}) and log p(z)
                log_qzx = log_prob_diag_gaussian(z_participant, mu_t_expanded, logvar_t_expanded)  # (N,)
                log_pz = log_prob_diag_gaussian(z_participant, 
                                              torch.zeros_like(mu_t_expanded),
                                              torch.zeros_like(logvar_t_expanded))  # (N,)
                # Importance weight
                log_w = log_qzx - log_pz  # (N,)

                # Evaluate decoder p(a_t | d_{1:t}, z)
                log_probs_t = torch.zeros(N, device=device)

                for sample_idx in range(N):
                    z_sample = z_participant[sample_idx]  # (z_dim,)
                    
                    # Get the actual data for this block (without batch dimensions)
                    block_data = blocks[participant_idx, block_idx, :t]  # (t, in_dim)
                    
                    # Get decoder output
                    logits = model.decoder(block_data.unsqueeze(0), z_sample.unsqueeze(0), ID_test = True) # (1, t, A)
                    #print(f"logits shape: {logits.shape}")

                    # Get probability for action at time t
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)  # (A,)
                    #print(f"log_probs shape: {log_probs.shape}, log_probs: {log_probs}")
                    action_t = y[participant_idx, block_idx, t-1]
                    #print(f"action_t before long: {action_t}, type: {type(action_t)}")
                    action_t = action_t.long()
                    log_probs_t[sample_idx] = log_probs[action_t]

                # Combine log p(a_t | d_{<t}, z) and importance weights
                log_weighted = log_probs_t + log_w  # (N,)
                #print(f"shape of log_weighted: {log_weighted.shape}")
                # Marginalize over samples
                log_p_at = torch.logsumexp(log_weighted, dim=0) - math.log(N)
                #print(f"shape of log_p_at: {log_p_at.shape}")
                print(f"type of log_p_at: {type(log_p_at)}")
                block_log_likelihood += log_p_at

            # Accumulate block likelihood (average later)
            participant_log_likelihood += block_log_likelihood
        
        #print(f"z_participant shape: {z_participant.shape}")  # Should be [100, z_dim]
        #print(f"mu_t_expanded shape: {mu_t_expanded.shape}")
        #print(f"logvar_t_expanded shape: {logvar_t_expanded.shape}")
        #print(f"log_qzx shape: {log_qzx.shape}")
        #print(f"log_pz shape: {log_pz.shape}")
        #print(f"log_w shape: {log_w.shape}")  # This might already have wrong dimensions

        # Average over blocks for this participant
        log_likelihoods_per_participant[participant_idx] = participant_log_likelihood 
        #print(f"participant_log_likelihood type: {type(participant_log_likelihood)}")
        #print(f"participant_log_likelihood shape: {participant_log_likelihood.shape if hasattr(participant_log_likelihood, 'shape') else 'No shape'}")

    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant


# Causal Importance-Weighted Test Function (per-timestep z)
@torch.no_grad()
def test_latentrnn_secondstep_causal_2_vectorized(model, blocks, y, N=100, num_actions=2, first_n=None):
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
    z_samples = torch.randn((N, B, z_dim), device=device)  # (N, B, z_dim)
    #print(f"z_samples shape: {z_samples.shape}")
    #print(f"z_samples: {z_samples}")
    #z_samples = z_samples.unsqueeze(2).unsqueeze(3)  # (N, B, 1, 1, z_dim) for broadcasting
    #print(f"z_samples expanded shape: {z_samples.shape}")
    #print(f"z_samples after formatting: {z_samples}")
    
    for participant_idx in range(B):
        z_participant = z_samples[:, participant_idx]  # (N, z_dim)
        print(f"Processing participant {participant_idx+1}/{B}")
        print(f"z_participant shape: {z_participant.shape}")  # Should be [100, z_dim]
        participant_log_likelihood = 0.0
        
        for block_idx in range(Bk):
            block_data = blocks[participant_idx, block_idx]  # (T, in_dim)
            block_actions = y[participant_idx, block_idx]  # (T,)
            
            block_log_likelihood = 0.0
            for t in range(1, T + 1):
                # Only use data up to time t (causality)
                data_up_to_t = block_data[:t]  # (t, in_dim)
                data_batched = data_up_to_t.unsqueeze(0).unsqueeze(0)  # (1, 1, t, in_dim)
                
                mu_t, logvar_t = model.encoder(data_batched, return_per_timestep=True)
                mu_t = mu_t[0, 0, -1]  # (z_dim,)
                logvar_t = logvar_t[0, 0, -1]  # (z_dim,)
                print(f"mu_t shape: {mu_t.shape}, logvar_t shape: {logvar_t.shape}")
                
                # Expand parameters
                mu_t_expanded = mu_t.unsqueeze(0).expand(N, -1)
                logvar_t_expanded = logvar_t.unsqueeze(0).expand(N, -1)
                print(f"mu_t_expanded shape: {mu_t_expanded.shape}, logvar_t_expanded shape: {logvar_t_expanded.shape}")
                
                # Compute weights
                log_qzx = log_prob_diag_gaussian(z_participant, mu_t_expanded, logvar_t_expanded)
                log_pz = log_prob_diag_gaussian(z_participant, 
                                              torch.zeros_like(mu_t_expanded),
                                              torch.zeros_like(logvar_t_expanded))
                print(f"log_qzx shape: {log_qzx.shape}, log_pz shape: {log_pz.shape}")
                log_w = log_qzx - log_pz
                print(f"log_w shape: {log_w.shape}")
                
                # Vectorized decoder (still causal - only uses data up to t)
                data_batch = data_up_to_t.unsqueeze(0).expand(N, -1, -1)  # (N, t, in_dim)
                logits = model.decoder(data_batch, z_participant, ID_test=True)  # (N, t, A)
                print(f"logits shape: {logits.shape}")
                log_probs = F.log_softmax(logits[:, -1], dim=-1)  # (N, A)
                print(f"log_probs shape: {log_probs.shape}")
                
                action_t = block_actions[t-1].long()
                print(f"action_t: {action_t}, type: {type(action_t)}")
                log_probs_t = log_probs[:, action_t]
                print(f"log_probs_t shape: {log_probs_t.shape}")
                
                # Combine and marginalize
                log_weighted = log_probs_t + log_w
                print(f"log_weighted shape: {log_weighted.shape}")
                log_p_at = torch.logsumexp(log_weighted, dim=0) - math.log(N)
                block_log_likelihood += log_p_at.item()
            
            participant_log_likelihood += block_log_likelihood 
        
        log_likelihoods_per_participant[participant_idx] = participant_log_likelihood
    
    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant


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
    
    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant, mu_tensor

@torch.no_grad()
def test_latentrnnZones(model, z, blocks, y, N=100, first_n=None):
    model.eval()
    B, Bk, T, _ = y.shape  # Now accounts for [B, Bk, T, 1]
    y = y.squeeze(-1)
    if first_n is not None:
        blocks = blocks[:, :, :first_n]        # (B, Bk, first_n, 2)
        y      = y[:, :, :first_n]
    # sample one z vector from the learned z's for each test participant
    
             # (B, V)
    #print(f"[{model.name}] mu_empirical shape: {mu_empirical.shape}, lv_empirical shape: {lv_empirical.shape}")
    #print(f"lv empirical: {lv_empirical}")
    #print(f"mu empirical: {mu_empirical}")

    LL = []
    for _ in range(N):
        # sample participant index with replacement so that each test participant gets a z from a random training participant
        M = z.size(0)
        #sampled_indices = torch.randint(0, M, (B,))
        # sample ones such that sampled tensor has same shape as z
        z_sampled = torch.ones((B,), dtype=torch.long)
        #z_sampled = z[sampled_indices]
        lp = []
        for b in range(Bk):
            logits = model.decoder(blocks[:, b], z_sampled)  # (B, T, A)
            log_probs = F.log_softmax(logits, -1)    # (B, T, A)
            lp.append(log_probs.gather(-1, y[:, b].unsqueeze(-1).long()).squeeze(-1))  # (B, T)
        LL.append(torch.stack(lp, 1))  # [N, B, Bk, T]
    ll = torch.logsumexp(torch.stack(LL, 0), 0) - math.log(N)  # (B, Bk, T) - averaged over N samples
    per_part = ll.sum((1, 2))  # (B,) - adding all likelihoods per participant
    return per_part.mean().item(), per_part


@torch.no_grad()
def test_latentrnn_ones(model, xenc, blocks, y, N=100, first_n=None):
    model.eval()
    B, Bk, T, _ = y.shape  # Now accounts for [B, Bk, T, 1]
    y = y.squeeze(-1)
    if first_n is not None:
        blocks = blocks[:, :, :first_n]        # (B, Bk, first_n, 2)
        y      = y[:, :, :first_n]
    mu, lv = model.encoder(xenc)
    std = torch.exp(0.5 * lv)
    LL = []
    for _ in range(N):
        z = mu + torch.randn_like(std) * std
        lp = []
        for b in range(Bk):
            logits = model.decoder(blocks[:, b], z)  # (B, T, A)
            # Corrected: gather log-probs of true actions y[:,b]
            log_probs = F.log_softmax(logits, -1)    # (B, T, A)
            lp.append(log_probs.gather(-1, y[:, b].unsqueeze(-1).long()).squeeze(-1))  # (B, T)
        LL.append(torch.stack(lp, 1))  # (B, Bk, T)
    ll = torch.logsumexp(torch.stack(LL, 0), 0) - math.log(N)  # (B, Bk, T)
    per_part = ll.sum((1, 2))  # (B,)
    return per_part.mean().item(), per_part

@torch.no_grad()
def test_ablated(model, blocks, y, first_n=None):
    model.eval()
    ll = []
    y = y.squeeze(-1)
    hidden_trajectories = []
    if first_n is not None:
        blocks = blocks[:, :, :first_n]        # (B, Bk, first_n, 2)
        y      = y[:, :, :first_n]
    for b in range(blocks.size(1)):
        logits, hid, hid_tr = model.dec(blocks[:, b])                # (B, T, A)
        ll.append(_safe_gather(logits, y[:, b]))
        hidden_trajectories.append(hid_tr)        
    ll = torch.stack(ll, 1)                             # (B, Bk, T)
    per_part = ll.sum((1, 2))
    traj = torch.stack(hidden_trajectories, dim=1)
      # Collect hidden states for each block
    return per_part.mean().item(), per_part, traj

@torch.no_grad()
def test_global_latent(model, blocks, y, first_n=None):
    model.eval()
    y = y.squeeze(-1)
    if first_n is not None:
        blocks = blocks[:, :, :first_n]
        y      = y[:, :, :first_n]

    ll = []
    for b in range(blocks.size(1)):
        logits = model.decoder(blocks[:, b], model.z.expand(blocks.size(0), -1))  # (B, T, A)
        ll.append(_safe_gather(logits, y[:, b]))

    ll = torch.stack(ll, 1)         # (B, Bk, T)
    per_part = ll.sum((1, 2))       # (B,)
    return per_part.mean().item(), per_part



@torch.no_grad()
def val_elbo_ablated(model, blocks, y):
    model.eval()
    logits, _, _ = model(blocks)
    nll = F.cross_entropy(logits.view(-1,2), y.long().view(-1), reduction='mean')
    return nll.item()

@torch.no_grad()
def val_elbo_global(model, blocks, y):
    model.eval()
    logits = model(blocks)
    nll = F.cross_entropy(logits.view(-1,2), y.long().view(-1), reduction='mean')
    return nll.item()

def posterior_predictive_check(model, x, blocks, N=100):
    model.eval()
    B, Bk, T, input_dim = blocks.shape

    with torch.no_grad():
        mu, logvar = model.encoder(x)
        std = torch.exp(0.5 * logvar)

        predictions = []

        for _ in range(N):
            z = mu + torch.randn_like(std) * std  # sample z outside
            block_outputs = []
            for b in range(Bk):
                seq_b = blocks[:, b]  # (B, T, input_dim)
                logits = model.decoder(seq_b, z) 
                probs = F.softmax(logits, dim=-1)
                block_outputs.append(probs)
            all_blocks = torch.stack(block_outputs, dim=1)  # (B, Bk, T, A)
            predictions.append(all_blocks)

        return torch.stack(predictions, dim=0).mean(dim=0)  # (B, Bk, T, A)
    
def ablated_posterior_predictive_check(model, blocks):
    model.eval()
    B, Bk, T, input_dim = blocks.shape
    with torch.no_grad():
        block_outputs = []
        for b in range(Bk):
            seq_b = blocks[:, b]
            logits,_,_ = model.dec(seq_b)
            probs = F.softmax(logits, dim=-1)
            block_outputs.append(probs)
        return torch.stack(block_outputs, dim=1)  # (B, Bk, T, A)
    
def behaviour_stats(action_tensor):
    """
    Parameters
    ----------
    action_tensor : LongTensor (B, Bk, T)  with entries 0 or 1
    Returns
    -------
    left  : (B,)  proportion of 0's
    right : (B,)  proportion of 1's
    switch: (B,)  prop. of trials where a_t != a_{t-1}  (per block, then mean)
    """
    
    B, Bk, T = action_tensor.shape
    left  = (action_tensor == 0).float().mean((1, 2))          # over blocks+time
    right = 1.0 - left

    # difference along time; first trial has no previous, so ignore it
    switches = (action_tensor[:, :, 1:] != action_tensor[:, :, :-1]).float()
    switch_prop = switches.mean((1, 2))                        # mean over (Bk, T-1)
    return left, right, switch_prop

def hidden_trajectory(traj, n_components=2, perplexity=100, random_state=42):
    """
    Plot the hidden state trajectories using PCA, t-SNE, or UMAP.
    
    Parameters
    ----------
    traj : torch.Tensor (B, Bk, T, H)  where H is the hidden state dimension
    n_components : int  number of dimensions for the output embedding
    perplexity : int  for t-SNE
    random_state : int  for reproducibility
    """
    nr_considered_blocks = 0
    B, Bk, T, H = traj.shape
    traj_subset = traj[:, nr_considered_blocks]
    traj_flat = traj_subset.reshape(-1, H).cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(traj_flat)
    traj_2d = tsne_result.reshape(B, 1, T, 2)
    return traj_2d


def plot_hidden_trajectory(traj_2d, participant_ids):
    plt.figure(figsize=(10, 8))
    
    # Create a distinct color for each participant
    colors = cm.get_cmap('tab20', len(participant_ids))

    for i, pid in enumerate(participant_ids):
        color = colors(i)
        for b in range(traj_2d.shape[1]):  # Loop over blocks
            x = traj_2d[pid, b, :, 0]
            y = traj_2d[pid, b, :, 1]
            plt.plot(x, y, marker='o', alpha=0.6, color=color)
        
        # Optionally add label only once per participant (legend might get crowded with 101)
        plt.plot([], [], color=color, label=f'P{pid}')

    plt.title('Hidden State Trajectories (Colored by Participant)')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True)

    # Show legend only if number of participants is manageable
    if len(participant_ids) <= 20:
        plt.legend()
    
    plt.show()

def plot_participant_mean_trajectories(traj):
    traj_avg = traj.mean(axis=1)  # shape: (101, T=10, 2)

    plt.figure(figsize=(10, 8))
    for pid in range(traj_avg.shape[0]):
        x = traj_avg[pid, :, 0]
        y = traj_avg[pid, :, 1]
        plt.plot(x, y, marker='o', alpha=0.4, label=f'P{pid}')
    plt.title("Participant-wise Mean Hidden State Trajectories")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()

def plot_avg_trajectories_colored_by_time(traj_2d):
    traj_avg = traj_2d.mean(axis=1)  # shape: (101, T=10, 2)
    B, T, _ = traj_avg.shape
    plt.figure(figsize=(10, 8))

    for pid in range(B):
        x = traj_avg[pid, :, 0]
        y = traj_avg[pid, :, 1]
        timesteps = np.arange(T)
        plt.plot(x, y, alpha=0.2, color='gray')
        sc = plt.scatter(x, y, c=timesteps, cmap='viridis', marker='o', alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label("Time Step")
    plt.title("Average Participant Trajectories — Colored by Time Step")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()

def plot_tsne_with_participant_trajectories(traj_2d, values, value_name="Factor1_Somatic_Anxiety"):
    """
    Plots t-SNE trajectories with gray lines per participant and color-coded points.

    Parameters
    ----------
    traj_2d : np.ndarray of shape (B, 1, T, 2)
        t-SNE embeddings per participant
    values : np.ndarray of shape (B*T,)
        Scalar value to color by (e.g. RU, timestep, etc.)
    value_name : str
        Label for the colorbar
    """
    B, _, T, _ = traj_2d.shape
    traj_block = traj_2d[:, 0]  # shape: (B, T, 2)
    traj_flat = traj_block.reshape(-1, 2)  # shape: (B*T, 2)

    plt.figure(figsize=(10, 8))

    # 1. Plot gray trajectory lines for each participant
    for pid in range(B):
        traj = traj_block[pid]  # shape: (T, 2)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.2, color='gray')

    # 2. Overlay color-coded scatter plot
    sc = plt.scatter(traj_flat[:, 0], traj_flat[:, 1], c=values, cmap='viridis', alpha=0.8)

    # 3. Colorbar and labels
    cbar = plt.colorbar(sc)
    cbar.set_label(value_name)

    plt.title(f"t-SNE Hidden States (Block 0) Colored by {value_name}")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --------------------------------------------------------------------------------
# 3.  train the three models  ----------------------------------------------------
# --------------------------------------------------------------------------------


"""model_psy = LatentRNN(PsychometricEncoder(4,zdim), zdim).to(device)
model_psy.name="psy"
model_emb = LatentRNN(LookupEncoder(len(participantID),zdim), zdim).to(device)
model_emb.name="emb"
model_abl = AblatedRNN().to(device)

print("\n⇢ Training psychometric-VAE …")
model_psy = train_latentrnn(model_psy, psy_train, Xs, Ys)
print("\n⇢ Training lookup-embedding …")
model_emb = train_latentrnn(model_emb, train_idx, Xs, Ys)
print("\n⇢ Training ablated RNN …")
model_abl = train_ablated(model_abl, Xs, Ys)"""

"""# --------------------------------------------------------------------------------
# 4.  evaluate on *held-out* participants  ---------------------------------------
# --------------------------------------------------------------------------------
print("\n⇢ Evaluation …")
mnames, pop_ll, part_ll = [], [], []
print(f"ys_test shape: {Ys_test.shape}, Xs_test shape: {Xs_test.shape}")

μ_psy,  ll_psy  = test_latentrnn(model_psy, psy_test,  Xs_test, Ys_test, first_n=1)
μ_emb,  ll_emb  = test_latentrnn(model_emb, test_idx,  Xs_test, Ys_test, first_n=1)
μ_abl,  ll_abl  = test_ablated  (model_abl,           Xs_test, Ys_test, first_n=1)

mnames.extend(["psy"]*len(ll_psy) + ["emb"]*len(ll_emb) + ["abl"]*len(ll_abl))
pop_ll  = {"psy":μ_psy, "emb":μ_emb, "abl":μ_abl}

df_ll = pd.DataFrame({
    "participant": np.tile(test_id,3),
    "model"      : mnames,
    "loglik"     : torch.cat([ll_psy, ll_emb, ll_abl]).cpu().numpy()
})

# --------------------------------------------------------------------------------
# 5.  plots  ---------------------------------------------------------------------
# --------------------------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(11,4))

# (a) overall
ax[0].bar(pop_ll.keys(), pop_ll.values(), color=["tab:orange","tab:blue","tab:green"])
ax[0].set_ylabel("mean test log-likelihood"); ax[0].set_title("Overall")

# (b) per participant scatter
for i,(lab,col) in enumerate(zip(["psy","emb","abl"],["tab:orange","tab:blue","tab:green"])):
    jitter = (i-1)*0.08
    sub = df_ll[df_ll.model==lab]
    ax[1].scatter(np.arange(len(sub))+jitter, sub["loglik"], label=lab, s=16, alpha=.8, color=col)
ax[1].set_xticks([]); ax[1].set_xlabel("held-out participants")
ax[1].set_ylabel("test log-likelihood")
ax[1].set_title("Per-participant"); ax[1].legend()
plt.tight_layout();  plt.show()"""
if __name__ == "__main__":
    device = "cpu"
    A=2; zdim=3; hid=100
    # 0. data  -----------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    FILE = "data/exp1_bandit_task_scale.csv"
    X,  Y,  Xtest,  Ytest, df = load_preprocess_data_VAE(FILE, n_blocks_pp=30)
    xs_overall = torch.cat((X, Xtest), dim=0)
    ys_overall = torch.cat((Y, Ytest), dim=0)

    # extract participant IDs --------------------------------------------------------
    factor_cols = ["sub",
                "Factor1_Somatic_Anxiety",
                "Factor2_Cognitive_Anxiety",
                "Factor3_Negative_Affect",
                "Factor4_Low_Self_esteem"]

    psy_df        = df[factor_cols].drop_duplicates("sub").sort_values("sub").reset_index(drop=True)
    participantID = psy_df["sub"].to_numpy()               # length B
    id2idx        = {pid:i for i,pid in enumerate(participantID)}

    # train / test split -------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    train_id, test_id = train_test_split(participantID, test_size=.20, random_state=42, shuffle=False)
    print(f"Train participants: {len(train_id)}, Test participants: {len(test_id)}")

    train_idx = torch.tensor([id2idx[i] for i in train_id], dtype=torch.long)
    test_idx  = torch.tensor([id2idx[i] for i in test_id],  dtype=torch.long)
    print(f"train_idx shape: {train_idx.shape}, train idx tensor: {train_idx}")

    psy_tensor = torch.tensor(psy_df.drop(columns="sub").to_numpy(),
                            dtype=torch.float32)
    psy_train  = psy_tensor[train_idx]
    psy_test   = psy_tensor[test_idx]
    print(f"Psychometric train shape: {psy_train.shape}")  # Expected: (num_train, 4)
    print(f"Psychometric test shape: {psy_test.shape}")    # Expected: (num_test, 4)

    # handy views of the behavioural tensors ----------------------------------------
    Xs, Ys     = xs_overall[train_idx],  ys_overall[train_idx]       # (B_train, 30,10,2)  (B_train,30,10)
    Xs_test, Ys_test = xs_overall[test_idx], ys_overall[test_idx]

    #epoch_grid = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300]
    #epoch_grid = [1,2,3,4,5,6,7,8,9,10]#[750, 800]#[350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    epoch_grid = [100]
    rows = []  
    rows_mean_ll = []                         # collect results here → DataFrame
    for E in epoch_grid:
        print(f"\n=== training for {E} epochs ===")

        # ---- re-create fresh model objects each run ---------------------
        m_psy = LatentRNN(PsychometricEncoder(4, zdim), zdim).to(device); m_psy.name="psy"
        m_emb = LatentRNN(LookupEncoder(len(participantID), zdim), zdim).to(device); m_emb.name="emb"
        m_abl = AblatedRNN().to(device); m_abl.name="abl"
        m_psy_ones = LatentRNN(PsychometricEncoder(4, zdim), zdim).to(device); m_psy_ones.name="psy_ones"

        # ---- train ------------------------------------------------------
        m_psy,mu_mpsy, lv_mpsy, train_loss_mpsy, val_loss_mpsy = train_latentrnn(m_psy, psy_train, Xs, Ys, epochs=E)
        train_loss_mpsy = [t.item() for t in train_loss_mpsy]
        # include m_psy model that takes in vector of ones of shape (B) to model just one global latent
        # print(f"len psy_train: {len(psy_train)}, len psy_test: {len(psy_test)}")
        m_psy_ones,_,_, train_loss_mpsy_ones, val_loss_mpsy_ones = train_latentrnn(m_psy_ones, torch.ones_like((psy_train), device=device), Xs, Ys, epochs=E)
        train_loss_mpsy_ones = [t.item() for t in train_loss_mpsy_ones]
        m_emb,mu_emb, lv_emb, train_loss_emb, val_loss_emb = train_latentrnn(m_emb, train_idx, Xs, Ys, epochs=E)
        train_loss_emb = [t.item() for t in train_loss_emb]
        m_abl,train_loss_abl, val_loss_abl = train_ablated(m_abl, Xs, Ys, Xs_test, Ys_test, epochs=E)
        train_loss_abl = [t.item() for t in train_loss_abl]

        # ---- evaluate ---------------------------------------------------
        mean_ll_psy, ll_psy = test_latentrnn(m_psy, mu_mpsy, lv_mpsy,  Xs_test, Ys_test)#, first_n = 2)
        # include m_psy model that takes in vector of ones of shape (B) to model just one global latent
        mean_ll_psy_ones, ll_psy_ones = test_latentrnn_ones(m_psy_ones, torch.ones_like((psy_test), device=device), Xs_test, Ys_test) #first_n = 2)
        mean_ll_emb, ll_emb = test_latentrnn(m_emb, mu_emb, lv_emb,  Xs_test, Ys_test) #, first_n = 2)
        mean_ll_abl, ll_abl, hidden_states = test_ablated  (m_abl,           Xs_test, Ys_test)
        """print(f"hidden states shape: {len(hidden_states)}, {hidden_states[0].shape}")  # (B, T, A)
        print(f"hidden states: {hidden_states[0][0,0,:]}")  # print first hidden state of first block
        traj = hidden_trajectory(hidden_states, n_components=2, perplexity=30, random_state=42)
        initial_hidden_states = traj[:, :, 0]  # shape: (B, Bk, H)
        initial_avg = initial_hidden_states.mean(axis=1)  # shape: (B, H)
        diffs = np.linalg.norm(initial_avg - initial_avg[0], axis=1)
        print(f"diffs: {diffs}")
        participant_ids = list(range(101)) #replace with len(test_id)
        
        df = pd.read_csv("data/exp1_bandit_task_scale.csv")
        block1_df = df[df["block"] == 1]
        block1_df = block1_df[block1_df["sub"].isin(test_id)]
        colors = block1_df["Factor2_Cognitive_Anxiety"].values
        plot_tsne_with_participant_trajectories(traj, colors)"""

        
        
        

        # ---- stack per participant data into rows -------------------------------------------
        for model_name, vec in zip(["psy","emb","abl", "psy_ones"], [ll_psy, ll_emb, ll_abl, ll_psy_ones]):
            for pid, ll in zip(test_id, vec.cpu().numpy()):
                rows.append(dict(epoch=E, model=model_name,
                                participant=pid, loglik=ll))
                
        # ---- create DataFrame from rows --------------------------------# ---------------------------------------------------------------------
        # tidy df →  DataFrame
        # ---------------------------------------------------------------------
        #print(df_runs.head())

        # ---- stack mean ll data into rows -------------------------------------------
        
        for model_name, ll in zip(["psy", "emb", "abl", "psy_ones"], 
                                [mean_ll_psy, mean_ll_emb, mean_ll_abl, mean_ll_psy_ones]):
            rows_mean_ll.append({"epoch": E, "model": model_name, "loglik": ll})

    """# plot train and test losses - always plots the last trained model
    losses = {
        "psy":      (train_loss_mpsy,      val_loss_mpsy),
        "psy_ones": (train_loss_mpsy_ones, val_loss_mpsy_ones),
        "emb":      (train_loss_emb,       val_loss_emb),
        "abl":      (train_loss_abl,       val_loss_abl),
    }
    # assign a color to each model
    colors = {
        "psy":      "tab:orange",
        "psy_ones": "tab:purple",
        "emb":      "tab:blue",
        "abl":      "tab:green",
    }"""





    # Convert to wide format (one row per epoch, columns=models)
    df_mean_ll = pd.DataFrame(rows_mean_ll).pivot(
        index="epoch",      # Rows are epochs
        columns="model",    # Columns are models
        values="loglik"     # Values are mean log-likelihoods
    ).reset_index()
    df_runs = pd.DataFrame(rows)      # columns: epoch, model, participant, loglik

                
    #z_dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # z_dim grid
    z_dims = [3]
    ll_mean = []  # collect mean log-likelihoods for each z_dim
    latent_ll = []
    for z_dim in z_dims:
        print(f"\n=== training latent-RNN with z_dim={z_dim} ===")
        m_emb_2 = LatentRNN(LookupEncoder(len(participantID), z_dim), z_dim).to(device); m_emb_2.name="emb_2"
        _, mu, lv, _, _ = train_latentrnn(m_emb_2, train_idx, Xs, Ys, epochs=100)
        ll_mean, _ = test_latentrnn(m_emb_2, mu, lv, Xs_test, Ys_test)
        latent_ll.append(dict(z_dim=z_dim, ll=ll_mean))

    # convert to DataFrame
    df_latent_ll = pd.DataFrame(latent_ll)
    #UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(mu.detach().cpu().numpy())  # (B, z_dim) → (B, 2)
    print(f"UMAP embedding shape: {embedding.shape}")
    plt.scatter(embedding[:,0], embedding[:,1])
    plt.title("UMAP projection of participant-level latents")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()

    #PCA
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(mu.detach().cpu().numpy())  # (B, z_dim) → (B, 2)
    print(f"PCA embedding shape: {embedding.shape}")
    plt.scatter(embedding[:,0], embedding[:,1])
    plt.title("PCA projection of participant-level latents")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    #t-SNE
    reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    embedding = reducer.fit_transform(mu.detach().cpu().numpy())  # (B, z_dim) → (B, 2)
    print(f"t-SNE embedding shape: {embedding.shape}")
    plt.scatter(embedding[:,0], embedding[:,1])
    plt.title("t-SNE projection of participant-level latents")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()






    # ---------------------------------------------------------------------
    # plotting: one scatter panel per epoch (10 panels → 2×5 grid)
    # ---------------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
    axes = axes.ravel()

    palette = dict(psy="tab:orange", emb="tab:blue", abl="tab:green", psy_ones="tab:purple")
    for ax, E in zip(axes, epoch_grid):
        sub = df_runs[df_runs["epoch"] == E]
        for i, (m, col) in enumerate(palette.items()):
            pts = sub[sub["model"] == m]["loglik"].values
            ax.scatter(np.arange(len(pts)) + (i-1)*0.10,   # small horizontal jitter
                    pts, s=15, alpha=0.8, color=col, label=m if E==epoch_grid[0] else None)
        ax.set_title(f"{E} epochs");  ax.set_xticks([])


    handles, labels = axes[0].get_legend_handles_labels()
    fig.text(0, 0.5, "test log-likelihood", va="center", rotation="vertical")
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Per-participant test log-likelihoods vs. training length", y=0.97, fontsize=16)
    plt.tight_layout(rect=[0,0.05,1,0.95])
    plt.savefig("per_participant_likelihoods_over_epochs.png", dpi=150)
    plt.show()



    with torch.no_grad():


        # psychometric VAE
        pred_psy = posterior_predictive_check(m_psy,
                                            psy_test, Xs_test, N=100).argmax(-1)  # (B,Bk,T)
        print(f"pred_psy shape: {pred_psy.shape}")

        # psychometric VAE
        pred_psy_ones = posterior_predictive_check(m_psy,
                                            torch.ones_like(psy_test), Xs_test, N=100).argmax(-1)  # (B,Bk,T)
        print(f"pred_psy shape: {pred_psy_ones.shape}")

        # lookup embedding
        pred_emb = posterior_predictive_check(m_emb,
                                            test_idx,  Xs_test, N=100).argmax(-1)
        print(f"pred_emb shape: {pred_emb.shape}")

        # ablated
        pred_abl = ablated_posterior_predictive_check(m_abl,
                                                    Xs_test).argmax(-1)
        print(f"pred_abl shape: {pred_abl.shape}")

        
        true_actions = Ys_test.squeeze(-1)  # (B, Bk, T)  → no extra dim
        print(f"y shape: {true_actions.shape}")


        rows = []
        for label, tens in [("data", true_actions),
                        ("psy",  pred_psy),
                        ("emb",  pred_emb),
                        ("abl",  pred_abl),
                        ("psy_ones", pred_psy_ones)]:
            l, r, s = behaviour_stats(tens)
            for pid, pl, pr, psw in zip(test_id, l, r, s):
                rows.append(dict(source=label,
                                participant=pid,
                                prop_left = pl.item(),
                                prop_right= pr.item(),
                                prop_switch= psw.item()))

        df_beh = pd.DataFrame(rows)
        print(df_beh.head(200))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

        for ax, metric, title in zip(axes,
                                    ["prop_left", "prop_right", "prop_switch"],
                                    ["Left choices", "Right choices", "Switches"]):
            sns.barplot(ax=ax, data=df_beh, x="source", y=metric,
                        ci=95, capsize=.15, palette=["grey","tab:orange","tab:blue","tab:green", "tab:purple"])
            ax.set_xlabel(""); ax.set_title(title)
            ax.set_ylim(0, 1)
        axes[1].set_ylabel("proportion")
        fig.suptitle("Held-out participants · behavioural proportions", y=1)
        plt.tight_layout()
        plt.show()

    sns.lineplot(data=df_latent_ll, x="z_dim", y="ll", marker='o')
    plt.gca().invert_yaxis()  # Flip the y-axis so 0 is at the bottom
    #plt.ylim(top=df_latent_ll["ll"].min(), bottom=0)  # Set 0 as the lower bound
    plt.title("Mean test log-likelihood vs. latent dimension")
    plt.xlabel("Latent dimension (z_dim)")
    plt.ylabel("Mean test log-likelihood")
    plt.show()



    """# How many epochs?
    epochs = df_mean_ll["epoch"].unique()
    n_epochs = len(epochs)

    # Set grid size based on number of epochs
    ncols = 5
    nrows = (n_epochs + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), sharey=True)
    axes = axes.ravel()

    palette = {
        "psy": "tab:orange",
        "emb": "tab:blue",
        "abl": "tab:green",
        "psy_ones": "tab:purple"
    }

    for i, E in enumerate(epochs):
        ax = axes[i]
        epoch_data = df_mean_ll[df_mean_ll['epoch'] == E]

        for j, model in enumerate(palette.keys()):
            val = epoch_data[model].values[0]
            ax.bar(j, val, color=palette[model], width=0.6,
                label=model if i == 0 else None)
            offset = 5
            ax.text(j, val-offset, f"{val:.2f}", ha='center', va='bottom' if val < 0 else 'top', fontsize=9)

        ax.set_title(f"{E} epochs", pad=10)
        ax.set_xticks([])
        #ax.invert_yaxis()

    # Hide unused axes
    for j in range(n_epochs, len(axes)):
        axes[j].set_visible(False)

    # Legend & labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), frameon=True)
    fig.text(0, 0.5, 'Negative Log-Likelihood (Lower = Better)', va='center', rotation='vertical', fontsize=12)
    fig.suptitle('Model Comparison Across Training Epochs', y=0.97, fontsize=16)

    fig.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0.1)
    plt.gca().invert_yaxis()  # Flip the y-axis so 0 is at the bottom
    plt.savefig('model_performance_by_epoch.png', dpi=150, bbox_inches='tight')
    plt.show()"""

    # Set up plot grid and style
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
    axes = axes.ravel()

    # Custom color palette (matches your previous scheme)
    palette = {
        "psy": "tab:orange",
        "emb": "tab:blue",
        "abl": "tab:green", 
        "psy_ones": "tab:purple"
    }

    # Plot each epoch in separate panel
    for ax, E in zip(axes, df_mean_ll['epoch'].unique()):
        epoch_data = df_mean_ll[df_mean_ll['epoch'] == E]
        
        # Plot bars for each model
        for i, model in enumerate(palette.keys()):
            ax.bar(i, epoch_data[model].values[0], 
                color=palette[model],
                width=0.6,
                label=model if E == df_mean_ll['epoch'].min() else None)
        
        # Panel formatting
        ax.set_title(f"{E} epochs", pad=10)
        ax.set_xticks([])
        #ax.invert_yaxis()  # Better performance at bottom
        
        # Add value labels
        offset = 10  # Offset for text above bars
        for i, model in enumerate(palette.keys()):
            val = epoch_data[model].values[0]
            ax.text(i, val-offset, f"{val:.2f}", 
                    ha='center', va= 'top',#'bottom' if val < 0 else 'top',
                    fontsize=9)

    # Add shared legend and labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, 
            bbox_to_anchor=(0.5, 0), frameon=True)
    fig.text(0, 0.5, 'Negative Log-Likelihood (Lower = Better)', 
            va='center', rotation='vertical', fontsize=12)
    fig.suptitle('Model Comparison Across Training Epochs', y=0.97, fontsize=16)

    plt.gca().invert_yaxis()  # Flip the y-axis so 0 is at the bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('model_performance_by_epoch.png', dpi=150, bbox_inches='tight')
    plt.show()

    """fig, ax = plt.subplots(figsize=(12, 5))

    for name, (train, val) in losses.items():
        c = colors[name]
        ax.plot(train, color=c, linestyle="-",  label=f"train {name}")
        ax.plot(val,   color=c, linestyle="--", label=f"val   {name}")

    ax.set_title(f"Training vs. validation losses after {E} epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.show()"""



