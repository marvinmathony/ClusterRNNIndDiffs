import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd
import math
import numpy as np
import wandb
import plot_functions as plf
import os
from sklearn.model_selection import KFold


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
        z:   (B, z_dim) for static z (broadcast across trials), or
             (B, T, z_dim) for per-timestep z (one z per trial).
        hidden: optional previous hidden state (1, B, hid)
        """

        if ID_test:
            if z.dim() > 2 and z.size(1) == 1:
                z = z.view(z.size(0), -1)
            if z.dim() == 1:
                z = z.unsqueeze(0)

        if z.dim() == 3:
            # per-timestep z: assume (B, T, z_dim) aligned with seq
            assert z.size(1) == seq.size(1), (
                f"per-timestep z time dim {z.size(1)} != seq time dim {seq.size(1)}")
            zexp     = z
            z_for_h0 = z[:, 0]            # causal h0 init from first-trial μ
        else:
            zexp     = z.unsqueeze(1).expand(-1, seq.size(1), -1)
            z_for_h0 = z

        if hidden is None:
            hidden = self.z2h0(z_for_h0).unsqueeze(0).contiguous().to(seq.device)

        rnn_input = torch.cat([seq, zexp], -1)

        if torch.isnan(rnn_input).any() or torch.isinf(rnn_input).any():
            print("NaNs or Infs in rnn_input!")
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print("NaNs or Infs in hidden state!")
        out, hidden = self.rnn(rnn_input, hidden)
        logits = self.lin(out)

        return logits, hidden
    
class LatentRNN_secondstep(nn.Module):
    def __init__(self, encoder, hid, z_dim=3, in_dim=2, A=2, decoder=None,
                 n_tasks=None, task_emb_dim=0, reinit_decoder_per_block=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else Decoder(in_dim, z_dim, hid, A)
        self.return_per_timestep = False
        self.reinit_decoder_per_block = reinit_decoder_per_block
        # Learned per-task context vector appended to decoder input at each timestep.
        # Analogous to the per-participant z embedding; requires the decoder to be
        # built with in_dim = base_in_dim + task_emb_dim.
        # task_ids (set via set_task_ids) maps each block index to its task (0/1/...).
        if n_tasks is not None and task_emb_dim > 0:
            self.task_embedding = nn.Embedding(n_tasks, task_emb_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        else:
            self.task_embedding = None
        self._task_ids = None  # LongTensor (n_blocks,); set via set_task_ids()

    def set_task_ids(self, task_ids):
        """Register per-block task IDs (LongTensor) for task-embedding lookup.
        Also propagates to the encoder if it supports task conditioning."""
        self._task_ids = task_ids
        if hasattr(self.encoder, 'set_task_ids'):
            self.encoder.set_task_ids(task_ids)

    def _append_task_emb(self, block_input, b):
        """Concatenate the task embedding for block b to block_input (B, T, d)."""
        if self.task_embedding is None or self._task_ids is None:
            return block_input
        tid  = self._task_ids[b].long()
        temb = self.task_embedding(tid)                                    # (task_emb_dim,)
        temb = temb.unsqueeze(0).unsqueeze(0).expand(                      # (B, T, task_emb_dim)
            block_input.size(0), block_input.size(1), -1)
        return torch.cat([block_input, temb], dim=-1)

    def forward(self, xenc, blocks, sample_z=False):
        """
        xenc: inputs to encoder (e.g., (B, B_blk, T, enc_in_dim))
        blocks: input sequences to policy decoder (B, n_blocks, T, base_in_dim)
        sample_z: whether to sample from q(z|x)
        """
        mu, lv = self.encoder(xenc, return_per_timestep=self.return_per_timestep)
        if sample_z:
            std = torch.exp(0.5 * lv)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # optional h0 export.  If z is per-timestep (4D), use first-trial μ
        # of the first block as the canonical "subject prior" for reporting.
        h0_ = self.decoder.z2h0(z[:, 0, 0] if z.dim() == 4 else z)
        if blocks.dim() == 4 and blocks.size(1) > 1:
            # Multiple blocks: optionally reinitialise decoder from z at each block
            # boundary (reinit_decoder_per_block=True) to force z to carry individual
            # identity rather than letting the GRU hidden state do it implicitly.
            h = None
            block_logits = []
            for b in range(blocks.size(1)):
                if self.reinit_decoder_per_block and b > 0:
                    h = None   # reinitialise from z2h0(z) at each block start
                inp_b = self._append_task_emb(blocks[:, b], b)
                # Slice z per block when z is per-timestep (4D), else broadcast.
                z_b = z[:, b] if z.dim() == 4 else z
                logits_b, h = self.decoder(inp_b, z_b, hidden=h)
                block_logits.append(logits_b)
            logits = torch.stack(block_logits, dim=1)  # (B, n_blocks, T, A)
        else:
            inp = self._append_task_emb(blocks.squeeze(1), 0)
            z_for_dec = z[:, 0] if z.dim() == 4 else z
            logits, hidden = self.decoder(inp, z_for_dec)
            logits = logits.unsqueeze(1)               # (B, 1, T, A)
        return logits, mu, lv, z, h0_

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
    def __init__(self, encoder, decoder, hid, z_dim=3, in_dim=2, A=2, block_structure=True,
                 n_tasks=None, task_emb_dim=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.block_structure = block_structure
        if n_tasks is not None and task_emb_dim > 0:
            self.task_embedding = nn.Embedding(n_tasks, task_emb_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        else:
            self.task_embedding = None
        self._task_ids = None

    def set_task_ids(self, task_ids):
        self._task_ids = task_ids

    def _append_task_emb(self, block_input, b):
        if self.task_embedding is None or self._task_ids is None:
            return block_input
        tid  = self._task_ids[b].long()
        temb = self.task_embedding(tid)
        temb = temb.unsqueeze(0).unsqueeze(0).expand(block_input.size(0), block_input.size(1), -1)
        return torch.cat([block_input, temb], dim=-1)

    def forward(self, xenc, blocks):            # blocks: (B,Bk,T,base_in_dim)
        z  = self.encoder(xenc)
        h0_ = self.decoder.z2h0(z)          # (B,H) - just for usage outside of RNN
        if self.block_structure:
            # Process each block independently; decoder hidden state resets to h0(z) each time
            logits_list = [self.decoder(self._append_task_emb(blocks[:, b], b), z)[0]
                           for b in range(blocks.size(1))]
            return torch.stack(logits_list, dim=1), z, h0_
        else:
            logits, hidden = self.decoder(self._append_task_emb(blocks, 0), z)
            return logits, hidden, h0_, z
        


class IDRNN(nn.Module):
    def __init__(self, in_dim, z_dim, hid=128, n_tasks=None, task_emb_dim=0,
                 continuous_encoder=False):
        super().__init__()
        self.task_emb_dim = task_emb_dim
        # continuous_encoder=True: GRU carries hidden state across ALL block
        # boundaries (correct for multi-task datasets like Thalmann).
        # continuous_encoder=False (default): each block processed independently,
        # preserving the old behaviour for single-task datasets.
        self.continuous_encoder = continuous_encoder
        if n_tasks is not None and task_emb_dim > 0:
            self.task_embedding = nn.Embedding(n_tasks, task_emb_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        else:
            self.task_embedding = None
        self._task_ids = None   # (B_blk,) LongTensor set via set_task_ids()
        self.block_encoder = nn.GRU(in_dim + task_emb_dim, hid, batch_first=True)
        self.mu_head       = nn.Linear(hid, z_dim)
        self.logvar_head   = nn.Linear(hid, z_dim)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # small variance
        self.count_training = 0
        self.count_testing = 0

    def set_task_ids(self, task_ids):
        """Register per-block task IDs (LongTensor of length B_blk)."""
        self._task_ids = task_ids

    def forward(self, blocks_tensor, return_per_timestep=True):
        """
        blocks_tensor: (B, B_blk, T, in_dim)
          - B      = number of participants in batch
          - B_blk  = blocks per participant
          - T      = trials per block (padded if needed)

        The GRU runs as ONE continuous sequence of length B_blk*T per
        participant, carrying hidden state across ALL block and task
        boundaries.  Padding positions (-100) are zeroed out before the GRU
        so they do not corrupt the running hidden state; the GRU simply
        coasts through them with near-stable h.

        return_per_timestep=False  →  (B, z_dim): μ at the last valid trial
                                      of the entire sequence (= fully informed
                                      individual posterior).
        return_per_timestep=True   →  (B, B_blk, T, z_dim): per-position μ,
                                      each informed by all preceding trials
                                      across blocks and tasks.
        """
        B, B_blk, T, in_dim_raw = blocks_tensor.shape
        x_raw = blocks_tensor.view(B * B_blk, T, in_dim_raw)

        # Valid mask computed on raw input (before task embedding)
        valid_mask_flat = (x_raw != -100.0).any(dim=-1)  # (B*B_blk, T)

        # ── task embeddings (shared by both paths) ─────────────────────────────
        if self.task_embedding is not None and self._task_ids is not None:
            task_ids = self._task_ids.to(x_raw.device)[:B_blk]
            task_ids_flat = task_ids.unsqueeze(0).expand(B, -1).reshape(B * B_blk)
            temb = self.task_embedding(task_ids_flat)          # (B*B_blk, task_emb_dim)
            temb = temb.unsqueeze(1).expand(-1, T, -1)        # (B*B_blk, T, task_emb_dim)
            x = torch.cat([x_raw, temb], dim=-1)              # (B*B_blk, T, d+task_emb)
        else:
            x = x_raw

        if self.continuous_encoder:
            # ── CONTINUOUS path: GRU carries state across ALL block boundaries ─
            # Reshape to (B, B_blk*T, d+task_emb) and zero out padding so it
            # doesn't corrupt the running hidden state.
            valid_flat = valid_mask_flat.reshape(B, B_blk * T)  # (B, B_blk*T)
            x_cont = x.reshape(B, B_blk * T, -1)
            x_cont = x_cont * valid_flat.unsqueeze(-1).float()  # zero padding
            out, _ = self.block_encoder(x_cont)                 # (B, B_blk*T, hid)

            if return_per_timestep:
                h = out.view(B, B_blk, T, -1)
                mu     = self.mu_head(h)
                logvar = self.logvar_head(h).clamp(-10, 5)
                if self.count_training < 1:
                    print("[IDRNN] continuous encoder, return_per_timestep=True")
                    self.count_training += 1
                return mu, logvar
            else:
                # Last valid trial across the full sequence — fully informed posterior
                has_valid  = valid_flat.any(dim=-1)
                last_valid = (B_blk * T - 1
                              - valid_flat.flip(dims=[1]).long().argmax(dim=1))
                last_valid = torch.where(has_valid, last_valid,
                                         torch.zeros_like(last_valid))
                h_final = out[torch.arange(B, device=out.device), last_valid]
                mu     = self.mu_head(h_final)
                logvar = self.logvar_head(h_final).clamp(-10, 5)
                if self.count_testing < 1:
                    print("[IDRNN] continuous encoder, return_per_timestep=False")
                    self.count_testing += 1
                return mu, logvar

        else:
            # ── PER-BLOCK path (default): each block processed independently ───
            out, _ = self.block_encoder(x)  # (B*B_blk, T, hid)

            if return_per_timestep:
                h = out.view(B, B_blk, T, -1)
                mu     = self.mu_head(h)
                logvar = self.logvar_head(h).clamp(-10, 5)
                if self.count_training < 1:
                    print("[IDRNN] per-block encoder, return_per_timestep=True")
                    self.count_training += 1
                return mu, logvar
            else:
                has_valid  = valid_mask_flat.any(dim=-1)
                last_valid = T - 1 - valid_mask_flat.flip(dims=[1]).long().argmax(dim=1)
                last_valid = torch.where(has_valid, last_valid,
                                         torch.zeros_like(last_valid))
                h_final = out[torch.arange(B * B_blk, device=x.device), last_valid]
                h_block = h_final.view(B, B_blk, -1)
                h_participant = h_block[:, -1, :]          # last block only
                mu     = self.mu_head(h_participant)
                logvar = self.logvar_head(h_participant).clamp(-10, 5)
                if self.count_testing < 1:
                    print("[IDRNN] per-block encoder, return_per_timestep=False")
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
    def __init__(self, hid, in_dim=2, A=2, block_structure=True,
                 n_tasks=None, task_emb_dim=0):
        super().__init__()
        self.in_dim = in_dim
        self.dec = AblatedDecoder(in_dim, hid, A)
        self.block_structure = block_structure
        if n_tasks is not None and task_emb_dim > 0:
            self.task_embedding = nn.Embedding(n_tasks, task_emb_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        else:
            self.task_embedding = None
        self._task_ids = None

    def set_task_ids(self, task_ids):
        self._task_ids = task_ids

    def _append_task_emb(self, block_input, b):
        if self.task_embedding is None or self._task_ids is None:
            return block_input
        tid  = self._task_ids[b].long()
        temb = self.task_embedding(tid)
        temb = temb.unsqueeze(0).unsqueeze(0).expand(block_input.size(0), block_input.size(1), -1)
        return torch.cat([block_input, temb], dim=-1)

    def forward(self, blocks, h0=None):
        if self.block_structure:
            out_list = [self.dec(self._append_task_emb(blocks[:, b], b))
                        for b in range(blocks.size(1))]
            logits, final_hid, hidden_tr = zip(*out_list)
            return torch.stack(logits, 1), torch.stack(final_hid, 1).squeeze(0), torch.stack(hidden_tr, 1)
        else:
            inp = self._append_task_emb(blocks, 0)
            logits, final_hid, hidden_tr = self.dec(inp, h0=h0)
            return logits, final_hid, hidden_tr
        
# small helper
def vectorize_rsa(mat):
    # Take only the upper triangle, excluding diagonal
        return mat[np.triu_indices_from(mat, k=1)]
    
### TRAINING SCRIPTS ###
def train_latentrnn_noblocks_palminteri(
    model: nn.Module,
    ids_train: torch.Tensor,           # (B,)
    X_train: torch.Tensor,             # (B, T, in_dim)
    y_onehot: torch.Tensor,            # (B, T, A) one-hot
    ids_val: torch.Tensor = None,
    X_val: torch.Tensor = None,
    y_val_onehot: torch.Tensor = None,
    epoch_nr: int = 3000,
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

    has_val = ids_val is not None and X_val is not None and y_val_onehot is not None
    if has_val:
        ids_val = ids_val.to(device)
        X_val = X_val.to(device)
        y_val_onehot = y_val_onehot.to(device)
        y_val = y_val_onehot

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}
    accuracy_dict = {}
    val_accuracy_dict = {}

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epoch_nr + 1):
        model.train()
        opt.zero_grad()

        logits, z, h0 = model(ids_train, X_train)            # logits: (B,T,A)
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

        # model selection
        if has_val:
            model.eval()
            with torch.no_grad():
                logits_val, z_val, _ = model(ids_val, X_val)
                nll_val = F.cross_entropy(
                logits_val.reshape(-1, logits_val.size(-1)),
                y_val.reshape(-1).long(),
                reduction='mean'
                )
                val_losses.append(nll_val.item())
                predictions_val = torch.argmax(logits_val, dim=-1)
                predictions_flat_val = predictions_val.reshape(-1)
                y_val_flat = y_val.reshape(-1)
                total_val = y_val_flat.numel()
                val_acc = torch.sum(predictions_flat_val == y_val_flat)
                final_val_acc = (val_acc/total_val).item()
                val_accuracy_dict[ep] = final_val_acc
                if nll_val.item() < best_val_loss:
                    best_val_loss = nll_val.item()
                    z_val_best = z_val
                    best_epoch = ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ep % 100 == 0:
                print(f"[LatentRNNz] epoch {ep:4d}  loss {nll.item():.4f}, train_acc: {final_train_acc}, val_acc: {final_val_acc}, val_loss {nll_val.item():.4f}")
        else:
            if nll.item() < best_train_loss:
                best_train_loss = nll.item()
                best_epoch = ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ep % 100 == 0:
                print(f"[LatentRNNz] epoch {ep:4d}  loss {nll.item():.4f}, train_acc: {final_train_acc}")

    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_best, z_best, h0_best = model(ids_train, X_train)
        pA_best = F.softmax(logits_best, dim=-1)              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "z": z_best,
        "h0": h0_best,
    }
    if has_val:
        training_dict["z_val"] = z_val_best
    print(f"best Step1 loss: {best_train_loss if not has_val else best_val_loss:.4f} in epoch: {best_epoch}")

    return model, train_losses, val_losses, pA_per_epoch, training_dict

def train_latentrnn_noblocks(
    model: nn.Module,
    ids_train: torch.Tensor,           # (B,)
    ids_test: torch.Tensor,
    X_train: torch.Tensor,             # (B, T, in_dim)
    y_onehot: torch.Tensor,            # (B, T, A) one-hot
    X_test: torch.Tensor,
    y_test_onehot: torch.Tensor,
    train_alpha_values,
    p_target: torch.Tensor = None,     # (B, T) prob(arm0), optional
    p_test_target: torch.Tensor = None,
    epoch_nr: int = 3000,
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
    best_acc = 0
    best_kl = float('inf')
    best_test_kl = float('inf')
    best_epoch = None
    best_state = None
    pA_per_epoch = {}

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epoch_nr + 1):
        model.train()
        opt.zero_grad()

        logits, z, _ = model(ids_train, X_train)            # logits: (B,T,A)
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

        model.eval()
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
        with torch.no_grad():

            z_expanded = z.unsqueeze(1).expand(-1, X_train.size(1), -1)
            distance_matrix_latents, latents_order = plf.rsa_latents(latents = z_expanded, metric="euclidean",title="training_run_latentmodel", reduction="last", plot=False, original_data=False)
            distance_matrix_params, params_order = plf.rsa_latents(latents = train_alpha_values, metric="euclidean",title="test_params", reduction="entire", plot=False, original_data=True)
            vec_params  = vectorize_rsa(distance_matrix_params)
            vec_latents = vectorize_rsa(distance_matrix_latents)
            corr_matrix = np.corrcoef(vec_params,vec_latents)[0,1]
            wandb.log({"training accuracy CP": acc, "Correlation original params & latents CP": corr_matrix}, step=ep)
        
        # test 
        if p_test_target is not None:
            model.eval()
            logits_test, z_test, _ = model(ids_test, X_test)
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
        
        #best state computed based on test accuracy
        if acc_test > best_acc:
            best_acc = acc_test
            best_acc_test_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
        best_epoch = epoch_nr
        best_kl = float('nan')

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_best, z_best, h0_best = model(ids_train, X_train)
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
def test_latentrnn_secondstep_causal_posterior_weighting(model, blocks, y, N=200, num_actions=2, first_n=None, id_case = True, enc_blocks=None):
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
    # mu_tensor stores the causal posterior mean for every (block, trial) across all blocks
    # Shape (B, Bk*T, z_dim) so analyze_outer_cv.py can average over axis-1 to get (B, z_dim)
    mu_tensor = torch.zeros(B, Bk * T, z_dim)

    # Separate encoder input (participant-specific dims) from decoder input (full dims)
    enc_blocks_resolved = enc_blocks if enc_blocks is not None else blocks

    total_valid_trials = 0  # count non-padding trials for geometric mean

    # Task-embedding state: True when the model has a task embedding and stored task IDs.
    has_task_emb = (getattr(model, 'task_embedding', None) is not None and
                    getattr(model, '_task_ids', None) is not None)

    def _add_task_emb_to_seq(seq, block_idx):
        """Append task embedding for block_idx to seq (L, in_dim) → (L, in_dim+task_emb_dim)."""
        if not has_task_emb:
            return seq
        tid  = model._task_ids[block_idx].long()
        temb = model.task_embedding(tid)              # (task_emb_dim,)
        temb = temb.unsqueeze(0).expand(seq.size(0), -1)  # (L, task_emb_dim)
        return torch.cat([seq, temb], dim=-1)

    for participant_idx in range(B):
        participant_log_likelihood = 0.0
        # h_block_start: hidden state at the START of each block, carried across
        # blocks for IDRNN.  Initialised to None so the decoder uses z2h0(z) on
        # the first block; updated with a deterministic (mu) forward pass after
        # each block.  Vanilla (id_case=False) always resets to None (z=0 → h=0).
        h_block_start = None
        _reinit_per_block = id_case and getattr(model, 'reinit_decoder_per_block', False)

        for block_idx in range(Bk):
            if _reinit_per_block and block_idx > 0:
                h_block_start = None   # reinitialise decoder from z2h0(z) each block
            block_data     = blocks[participant_idx, block_idx]  # (T, in_dim)
            enc_block_data = enc_blocks_resolved[participant_idx, block_idx]  # (T, enc_in_dim)
            block_actions  = y[participant_idx, block_idx]  # (T,)

            valid_len = int((block_actions >= 0).float().sum().item())
            mu_final  = None   # last-trial posterior mean; used for state update

            # Pre-slice completed blocks for this participant (fixed for all t in this block).
            # Used to build the cross-block encoder input at each trial.
            enc_in_dim_enc = enc_blocks_resolved.shape[-1]
            if id_case and block_idx > 0:
                prev_enc_blocks = enc_blocks_resolved[participant_idx, :block_idx]  # (block_idx, T, enc_in_dim)
            else:
                prev_enc_blocks = None

            block_log_likelihood = 0.0
            for t in range(1, T + 1):
                action_t = block_actions[t - 1].long()
                if action_t < 0:   # -100 padding → skip this trial entirely
                    continue
                total_valid_trials += 1

                # Only use data up to time t (causality)
                data_up_to_t = block_data[:t]            # (t, in_dim)

                if id_case:
                    # Build cross-block encoder input: all completed blocks (0..block_idx-1) in
                    # full, plus the current block up to trial t (-100 for unobserved future trials).
                    # This gives z an accumulated view of all participant behaviour seen so far,
                    # matching the training setup where all blocks are passed simultaneously.
                    curr_enc_padded = torch.full(
                        (T, enc_in_dim_enc), -100.0, device=device
                    )
                    curr_enc_padded[:t] = enc_block_data[:t]

                    if prev_enc_blocks is not None:
                        enc_so_far = torch.cat(
                            [prev_enc_blocks, curr_enc_padded.unsqueeze(0)], dim=0
                        )  # (block_idx+1, T, enc_in_dim)
                    else:
                        enc_so_far = curr_enc_padded.unsqueeze(0)  # (1, T, enc_in_dim)

                    data_batched = enc_so_far.unsqueeze(0)  # (1, block_idx+1, T, enc_in_dim)
                    mu_t, logvar_t = model.encoder(data_batched, return_per_timestep=False)
                    mu_t     = mu_t[0]      # (z_dim,)
                    logvar_t = logvar_t[0]  # (z_dim,)
                    # Store at the flattened block-trial position
                    mu_tensor[participant_idx, block_idx * T + (t - 1)] = mu_t
                    std_t    = torch.exp(0.5 * logvar_t)
                    z_samples = mu_t + std_t * torch.randn((N, z_dim), device=device)
                    mu_final  = mu_t
                else:
                    z_samples = torch.zeros((N, z_dim), device=device)

                # Append task embedding to decoder input (if multi-task model)
                dec_seq = _add_task_emb_to_seq(data_up_to_t, block_idx)
                data_batch = dec_seq.unsqueeze(0).expand(N, -1, -1)  # (N, t, dec_in_dim)

                # IDRNN: warm-start decoder from previous blocks' carried state.
                # For block 0 (h_block_start is None) the decoder initialises from
                # z2h0(z_samples), giving each sample its own starting state — correct.
                # For blocks 1+ we expand the single deterministic state to N copies.
                if id_case and h_block_start is not None:
                    h_init = h_block_start.expand(1, N, -1).contiguous()
                else:
                    h_init = None   # decoder init from z (vanilla always, IDRNN block-0)

                logits, _  = model.decoder(data_batch, z_samples, ID_test=True, hidden=h_init)
                log_probs  = F.log_softmax(logits[:, -1], dim=-1)  # (N, A)

                log_probs_t = log_probs[:, action_t]  # (N,)
                log_p_at    = torch.logsumexp(log_probs_t, dim=0) - math.log(N)
                block_log_likelihood += log_p_at.item()

            participant_log_likelihood += block_log_likelihood

            # Update h_block_start for IDRNN: deterministic forward pass through
            # the valid trials of this block using the posterior mean as z.
            if id_case and valid_len > 0 and mu_final is not None:
                z_det            = mu_final.unsqueeze(0)                    # (1, z_dim)
                # Apply task embedding to valid block data before passing to decoder
                block_valid_raw  = block_data[:valid_len]                   # (valid_len, in_dim)
                block_valid_dec  = _add_task_emb_to_seq(block_valid_raw, block_idx).unsqueeze(0)
                _, h_block_start = model.decoder(
                    block_valid_dec, z_det, ID_test=True, hidden=h_block_start
                )
                # h_block_start: (1, 1, H) — carried to the next block

        log_likelihoods_per_participant[participant_idx] = participant_log_likelihood

    total_log_likelihood = log_likelihoods_per_participant.sum()
    denom = total_valid_trials if total_valid_trials > 0 else (B * Bk * T)
    geometric_mean_per_trial = torch.exp(total_log_likelihood / denom)
    print("test function being used")
    return log_likelihoods_per_participant.mean().item(), log_likelihoods_per_participant, mu_tensor, geometric_mean_per_trial


# --- Compute RNN likelihoods ---
def compute_rnn_likelihoods_torch(test_function, model, test_xin, choice_test, train_xin, latent, choice_train=None, N=200, id=True, test_xin_enc=None, train_xin_enc=None):
    rnn_results = []
    B = choice_test.shape[0]
    with torch.no_grad():
        if latent:
            # Ensure 4-D block format (B, Bk, T, in_dim); sloutsky arrives as 3-D
            if test_xin.ndim == 3:
                test_xin  = test_xin.unsqueeze(1)
                choice_test = choice_test.unsqueeze(1)
                train_xin = train_xin.unsqueeze(1)
                if test_xin_enc  is not None: test_xin_enc  = test_xin_enc.unsqueeze(1)
                if train_xin_enc is not None: train_xin_enc = train_xin_enc.unsqueeze(1)
                if choice_train  is not None: choice_train  = choice_train.unsqueeze(1)

            # Compute train-data latents (always, regardless of original dimensionality)
            if choice_train is not None:
                _, _, latent_tensor_train, _ = test_function(
                    model=model, blocks=train_xin, y=choice_train, N=N, id_case=True,
                    enc_blocks=train_xin_enc
                )
            else:
                latent_tensor_train = None

            mean_ll, ll_per_participant, latent_tensor, geometric_mean_per_trial_test = test_function(
                model=model, blocks=test_xin, y=choice_test, N=N, id_case=id,
                enc_blocks=test_xin_enc
            )
            valid_per_participant = (choice_test >= 0).reshape(B, -1).sum(dim=1).float().cpu()
            valid_per_participant = valid_per_participant.clamp(min=1)
            nll_per_trial = -ll_per_participant.cpu() / valid_per_participant
            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": nll_per_trial.numpy(),
                "model": "IDRNN" if id else "common_process_RNN",
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train, geometric_mean_per_trial_test
        else:
            logits_test,  final_hidden, latent_tensor       = model(test_xin)
            _,            _,            latent_tensor_train = model(train_xin)

            # Flatten latent tensors to (B, all_timesteps, hid) for analyze_outer_cv.py
            if latent_tensor.dim() == 4:       # (B, Bk, T, hid) — block-structured
                B_sz, Bk, T_blk, hid = latent_tensor.shape
                latent_tensor       = latent_tensor.view(B_sz, Bk * T_blk, hid)
            if latent_tensor_train.dim() == 4:
                B_tr, Bk_tr, T_tr, hid_tr = latent_tensor_train.shape
                latent_tensor_train = latent_tensor_train.view(B_tr, Bk_tr * T_tr, hid_tr)

            probs_test = F.softmax(logits_test, dim=-1)

            # Mask out padding positions (-100) when computing per-participant NLL
            valid_mask   = choice_test >= 0                              # (...) bool
            choices_safe = choice_test.clone()
            choices_safe[~valid_mask] = 0                                # avoid bad indices
            chosen_probs = probs_test.gather(-1, choices_safe.long().unsqueeze(-1)).squeeze(-1)
            chosen_probs[~valid_mask] = 1.0                              # log(1) = 0; no contribution
            log_likelihoods   = chosen_probs.clamp(min=1e-8).log()
            # Sum over all non-batch dims (blocks + trials)
            log_ll_per_session = log_likelihoods.reshape(B, -1).sum(dim=1)  # (B,)
            valid_per_participant = valid_mask.reshape(B, -1).sum(dim=1).float().clamp(min=1)
            total_valid_trials  = valid_mask.sum().item()
            geometric_mean_per_trial = torch.exp(log_ll_per_session.sum() / max(total_valid_trials, 1)).item()

            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": (-log_ll_per_session / valid_per_participant).cpu().numpy(),
                "model": "vanillaRNN"
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train, geometric_mean_per_trial
        
def gaussian_nll_point(mu, logvar, z_T):

    var = logvar.exp()
    return 0.5 * (((z_T - mu)**2) / (var + 1e-8) + logvar).sum(-1).mean()
        
def elbo_lossZ(logits, targets, block_weights=None):
    """
    block_weights: optional 1-D tensor of length Bk.
      If None: flat mean over all non-padding tokens (original behaviour).
      If given: per-block NLL averaged within each block, then weighted mean
        across blocks.  Blocks with weight 0 are excluded entirely.
        Use e.g. task_ids.float() to focus loss on the informative task only.
    """
    B, Bk, T, A = logits.shape
    if block_weights is None:
        nll = F.cross_entropy(logits.reshape(-1, A), targets.long().reshape(-1), reduction='mean')
        return nll, nll.detach()
    # Compute per-block NLL (mean over non-padding tokens within each block)
    nll_per_block = []
    for b in range(Bk):
        logits_b  = logits[:, b].reshape(-1, A)
        targets_b = targets[:, b].reshape(-1).long()
        nll_b = F.cross_entropy(logits_b, targets_b, ignore_index=-100, reduction='mean')
        nll_per_block.append(nll_b)
    nll_blocks = torch.stack(nll_per_block)             # (Bk,)
    w = block_weights.to(nll_blocks.device).float()
    w_sum = w.sum()
    nll = (nll_blocks * w).sum() / w_sum if w_sum > 0 else nll_blocks.mean()
    return nll, nll.detach()

def uniformity_loss(z, t=2.0):
    """
    Wang & Isola (2020) uniformity loss.
    Encourages z to be spread uniformly across the hypersphere.
    Minimising this loss pushes all pairwise distances to be large.

    z: (B, z_dim) — encoder means across participants in the batch.
    t: Gaussian kernel bandwidth (default 2.0 from the paper).
    Returns a scalar; add mu_unif * uniformity_loss(z) to the step-2 loss.
    """
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    z_norm = F.normalize(z, dim=1)              # project onto unit hypersphere
    sq_dists = torch.pdist(z_norm).pow(2)       # all pairwise squared distances
    return sq_dists.mul(-t).exp().mean().log()  # log-mean of Gaussian kernel


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

def train_latentrnn_IDRNN_palminteri(model, xenc, blocks, y, lookup_z, xenc_val=None, blocks_val=None, y_val=None, z_val_lookup=None, epoch_nr=3000, patience = 300, lr=1e-3, window_size=20, checkpoint_dir=None, loss_dir=None, lmbd=0.5, block_weights=None, unif_weight=0.0):
    #print(f"x_enc shape: {xenc.shape}, blocks shape: {blocks.shape}, y shape: {y.shape}")
    #xenc input must have shape (B, B_blk, T, in_dim)
    train_elbos = []
    val_nll_flats = []   # pure flat cross-entropy on val (comparable to vanilla cv_val_loss)
    val_z_std = []       # std of mu across participants on val set (collapse diagnostic)
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

    has_val = xenc_val is not None and blocks_val is not None and y_val is not None and z_val_lookup is not None

    for ep in range(1,epoch_nr+1):
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
        policy_loss, nll = elbo_lossZ(logits, y_prefix, block_weights=block_weights)

        loss, _ = step_two_loss(lmbd, mu, lv, lookup_z, policy_loss)
        if unif_weight > 0.0:
            loss = loss + unif_weight * uniformity_loss(mu)
        loss.backward()
        opt.step()
        train_elbos.append(loss.item())

        # Validation tracking (when val data available)
        if has_val:
            xenc_val_prefix = xenc_val[:, :, :prefix_len, :]
            blocks_val_prefix = blocks_val[:, :, :prefix_len, :]
            yval_prefix = y_val[:, :, :prefix_len]
            model.eval()
            val_logits, mu_val, lv_val, _, _ = model(xenc_val_prefix, blocks_val_prefix, sample_z=False)
            val_policy_loss, val_nll = elbo_lossZ(val_logits, yval_prefix, block_weights=block_weights)
            val_loss, _ = step_two_loss(lmbd, mu_val, lv_val, z_val_lookup, val_policy_loss)
            val_elbos.append(val_loss.item())
            # Flat NLL (unweighted cross-entropy, comparable to vanilla cv_val_loss)
            B_, Bk_, T_, A_ = val_logits.shape
            flat_nll = F.cross_entropy(
                val_logits.reshape(-1, A_), yval_prefix.reshape(-1).long(), reduction='mean'
            )
            val_nll_flats.append(flat_nll.item())
            # z spread diagnostic: std of mu across participants, averaged over z_dim
            val_z_std.append(mu_val.detach().std(dim=0).mean().item())

        #accuracy tracking
        predictions = torch.argmax(logits, dim=-1)
        predictions_flat = predictions.reshape(-1)
        y_flat = y_prefix.reshape(-1)
        total = y_flat.numel()
        acc = torch.sum(predictions_flat == y_flat)
        final_acc = (acc/total).item()
        accuracy_dict[ep] = final_acc

        if has_val:
            predictions_val = torch.argmax(val_logits, dim=-1)
            predictions_flat_val = predictions_val.reshape(-1)
            y_val_flat = yval_prefix.reshape(-1)
            total_val = y_val_flat.numel()
            val_acc = torch.sum(predictions_flat_val == y_val_flat)
            final_val_acc = (val_acc/total_val).item()
            val_accuracy_dict[ep] = final_val_acc
            val_acc_history.append(final_val_acc)
            model_state_dict[ep] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0:
            if has_val:
                print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  val loss {val_loss.item():.3f},  train_accuracy: {final_acc},  val_accuracy: {final_val_acc}")
            else:
                print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  train_accuracy: {final_acc}")

        # Save checkpoint and loss every 100 epochs
        if ep % 100 == 0:
            if checkpoint_dir is not None:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch{ep:04d}.pt")
                torch.save(model.state_dict(), ckpt_path)
            if loss_dir is not None:
                loss_path = os.path.join(loss_dir, f"epoch_{ep:04d}.npy")
                np.save(loss_path, np.array(loss.item()))

    # Post-hoc epoch selection: specificity criterion handles this externally.
    # Use smoothed val accuracy if available, otherwise use final model state.
    if has_val:
        best_epoch = int(np.argmin(val_elbos) + 1)  # +1 because epochs start at 1
        best_model_state = model_state_dict[best_epoch]
        model.load_state_dict(best_model_state)
        print(f"best validation loss: {val_elbos[best_epoch - 1]:.6f} at epoch {best_epoch}")
    else:
        best_epoch = epoch_nr+1
        print(f"No val data; using final epoch {best_epoch} (specificity selects best checkpoint post-hoc)")

    model.eval()
    with torch.no_grad():
        logits_best, mu_best, lv_best, z_best, h0_best = model(xenc, blocks, sample_z=False)
        pA_best = F.softmax(logits_best, dim=-1).squeeze(1)[:, :, 0]              # (B,T,A)
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "best_model": model,
        "best_epoch": best_epoch,
        "best_kl": best_kl,
        "z": mu_best,
        "h0": h0_best,
    }

    return model, mu_best, lv_best, train_elbos, val_elbos, val_nll_flats, val_z_std, training_dict, pA_per_epoch

def make_model(wrapper_model, encoder, hidden, z_dim, in_dim, A, decoder, enc_in_dim, enc_hidden, device,
               task_embedding=None, task_ids=None, n_tasks=None, task_emb_dim=0):
                encoder_model = encoder(in_dim=enc_in_dim, z_dim=z_dim, hid=enc_hidden,
                                        n_tasks=n_tasks, task_emb_dim=task_emb_dim)
                model = wrapper_model(encoder_model, hidden, z_dim, in_dim, A, decoder)
                model.name = "GRU"
                if task_embedding is not None:
                    model.task_embedding = task_embedding
                if task_ids is not None:
                    model.set_task_ids(task_ids)   # propagates to encoder via set_task_ids
                model.to(device)
                for p in model.decoder.parameters():
                    p.requires_grad = False
                return model

def train_latentrnn_IDRNN_palminteri_CV(wrapper_model, encoder, hidden, z_dim, in_dim, A, decoder, enc_in_dim, enc_hidden, make_model_fun, xenc, blocks, y, lookup_z, device, xenc_val=None, blocks_val=None, y_val=None, z_val_lookup=None, epoch_nr=3000, patience = 300, lr=1e-3, window_size=20, checkpoint_dir=None, loss_dir=None, lmbd=0.5, nr_splits=3, block_weights=None, unif_weight=0.0):
    #xenc input must have shape (B, B_blk, T, in_dim)
    has_val = xenc_val is not None and blocks_val is not None and y_val is not None and z_val_lookup is not None

    n_subjects = xenc.shape[0]
    indices = np.arange(n_subjects)
    kf = KFold(n_splits=nr_splits, shuffle=True)
    best_epochs = []
    fold_results = []
    all_val_elbos = []
    all_val_nll_flats = []
    all_z_std_per_epoch = []

    for fold, (train_idx,test_idx) in enumerate(kf.split(indices), start=1):
        print(f"----training fold {fold}/{nr_splits}----")
        xenc_tr = xenc[train_idx]
        xenc_te = xenc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        blocks_tr = blocks[train_idx]
        blocks_te = blocks[test_idx]
        z_tr = lookup_z[train_idx]
        z_te = lookup_z[test_idx]

        model = make_model_fun(wrapper_model, encoder, hidden, z_dim, in_dim, A, decoder, enc_in_dim, enc_hidden, device)

        model, mu_best, lv_best, train_elbos, val_elbos, val_nll_flats, val_z_std, training_dict, pA_per_epoch = train_latentrnn_IDRNN_palminteri(
            model=model,
            xenc=xenc_tr,
            blocks=blocks_tr,
            y=y_tr,
            lookup_z=z_tr,
            xenc_val=xenc_te,
            blocks_val=blocks_te,
            y_val=y_te,
            z_val_lookup=z_te,
            epoch_nr=epoch_nr,
            lr=lr,
            window_size=window_size,
            checkpoint_dir=checkpoint_dir,
            loss_dir=loss_dir,
            lmbd=lmbd,
            block_weights=block_weights,
            unif_weight=unif_weight,
        )

        best_epoch = training_dict["best_epoch"]
        best_epochs.append(best_epoch)
        all_val_elbos.append(np.array(val_elbos))
        all_val_nll_flats.append(np.array(val_nll_flats))
        all_z_std_per_epoch.append(np.array(val_z_std))

        print(f"Fold {fold}: best epoch = {best_epoch}")

    all_val_elbos     = np.stack(all_val_elbos, axis=0)      # (fold, epoch)
    all_val_nll_flats = np.stack(all_val_nll_flats, axis=0)  # (fold, epoch)
    mean_val_elbos     = all_val_elbos.mean(axis=0)
    mean_val_nll_flats = all_val_nll_flats.mean(axis=0)
    selected_epoch = int(np.argmin(mean_val_elbos) + 1)

    # z spread curve (collapse diagnostic)
    min_len = min(len(a) for a in all_z_std_per_epoch)
    mean_z_std_per_epoch = np.stack([a[:min_len] for a in all_z_std_per_epoch], axis=0).mean(axis=0)

    print("\n===== CV summary =====")
    print(f"Best epochs per fold: {best_epochs}")
    print(f"selected_epoch: {selected_epoch}")
    print(f"mean_val_nll (flat, at selected epoch): {mean_val_nll_flats[selected_epoch - 1]:.4f}")
    print(f"z_std at selected epoch: {mean_z_std_per_epoch[selected_epoch - 1]:.4f}")
    return {
        "best_epochs_per_fold": best_epochs,
        "mean_val_elbos": mean_val_elbos,
        "mean_val_nll_flats": mean_val_nll_flats,
        "mean_z_std_per_epoch": mean_z_std_per_epoch,
        "selected_epoch": selected_epoch}

def train_final_model_after_cv(
    wrapper_model,
    encoder,
    decoder,
    make_model_fun,
    hidden,
    z_dim,
    in_dim,
    A,
    enc_in_dim,
    enc_hidden,
    xenc,
    blocks,
    y,
    lookup_z,
    n_epochs,
    device,
    lr=1e-3,
    checkpoint_dir=None,
    loss_dir=None,
    lmbd=0.5,
    block_weights=None,
    unif_weight=0.0):
    model = make_model_fun(wrapper_model, encoder, hidden, z_dim, in_dim, A, decoder, enc_in_dim, enc_hidden, device)

    model, mu_best, lv_best, train_elbos, val_elbos, val_nll_flats, val_z_std, training_dict, pA_per_epoch = train_latentrnn_IDRNN_palminteri(
        model=model,
        xenc=xenc,
        blocks=blocks,
        y=y,
        lookup_z=lookup_z,
        xenc_val=None,
        blocks_val=None,
        y_val=None,
        z_val_lookup=None,
        epoch_nr=n_epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        loss_dir=loss_dir,
        lmbd=lmbd,
        block_weights=block_weights,
        unif_weight=unif_weight,
    )

    return model, mu_best, lv_best, train_elbos, val_elbos, training_dict, pA_per_epoch


def train_idrnn_step2_5(model, xenc, blocks, y, n_epochs, lr=5e-4,
                         weight_decay=1e-4, block_weights=None, log_every=50):
    """
    Step 2.5 — brief joint encoder + decoder fine-tune with **per-timestep z**.

    Motivation: the standard step-2 trainer keeps the decoder frozen and feeds
    it the encoder's full-session μ broadcast across trials.  At test time
    the causal-posterior evaluator instead presents the decoder with a μ that
    varies trial-by-trial (the encoder is run on accumulating prefixes).
    The trained decoder weights have therefore never seen "less-informed"
    early-trial μ during training — a train/test mismatch that limits how
    much identity information the decoder can extract.

    Step 2.5 closes the gap by:
      - setting `model.return_per_timestep = True` so the encoder's per-trial
        μ (computed causally by the unidirectional GRU) is exposed,
      - unfreezing all parameters,
      - training for `n_epochs` epochs using policy NLL with per-timestep z,
      - resetting `return_per_timestep = False` afterwards so downstream
        causal-posterior testing works unchanged.

    Empirically (dezfouli, fold 0, 3 seeds) this closes ~17 % of the
    IDRNN-vs-vanilla NLL gap; an identical control with static z + unfrozen
    decoder produced no improvement (p=0.85), attributing the effect to the
    per-timestep conditioning rather than the extra optimisation steps.

    Args
    ----
    model : LatentRNN_secondstep (already step-2 trained)
    xenc, blocks : (B, n_blocks, T, in_dim) inputs to encoder / decoder
    y     : (B, n_blocks, T) action targets, padding -100
    n_epochs : K, e.g. 200–500
    lr       : ~5e-4 works well; lower if you see seed-400-style regression.

    Returns
    -------
    (model, train_losses)
    """
    model.return_per_timestep = True
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _, _, _, _ = model(xenc, blocks, sample_z=False)
        nll, _ = elbo_lossZ(logits, y, block_weights=block_weights)
        nll.backward()
        opt.step()
        train_losses.append(float(nll))
        if ep == 1 or ep % log_every == 0 or ep == n_epochs:
            print(f"  [step 2.5] ep {ep:4d}/{n_epochs}  "
                  f"train_nll={nll.item():.4f}")

    # Reset for downstream test_function compatibility (causal eval expects
    # the encoder to default to last-trial μ on each prefix call).
    model.return_per_timestep = False
    return model, train_losses


def train_ablated_noblocks_palminteri_CV(hidden, in_dim, A, X_train, y_train, device,
                                         epoch_nr=3000, lr=0.001, nr_splits=3,
                                         dec_in_dim=None, n_tasks=None, task_emb_dim=0,
                                         task_ids=None):
    """
    KFold CV over subjects to select the optimal number of training epochs
    for AblatedRNN on human data.

    Returns a summary dict with 'selected_epoch' (1-indexed int).
    """
    _dec_in_dim = dec_in_dim if dec_in_dim is not None else in_dim
    n_subjects = X_train.shape[0]
    indices = np.arange(n_subjects)
    kf = KFold(n_splits=nr_splits, shuffle=True)
    all_val_elbos = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        print(f"----vanilla CV fold {fold}/{nr_splits}----")
        X_tr = X_train[train_idx]
        X_te = X_train[test_idx]
        y_tr = y_train[train_idx]
        y_te = y_train[test_idx]

        _block_structure = (X_train.dim() == 4)
        model_fold = AblatedRNN(hid=hidden, in_dim=_dec_in_dim, A=A,
                                block_structure=_block_structure,
                                n_tasks=n_tasks, task_emb_dim=task_emb_dim).to(device)
        if task_ids is not None:
            model_fold.set_task_ids(task_ids)
        _, _, val_elbos, _, _, training_dict = train_ablated_noblocks_palminteri(
            model_fold, X_tr, y_tr, X_val=X_te, y_val=y_te,
            epoch_nr=epoch_nr, lr=lr,
            checkpoint_dir=None, loss_dir=None,  # don't save fold checkpoints
        )
        all_val_elbos.append(np.array(val_elbos))
        print(f"  Fold {fold}: best epoch = {training_dict['best_epoch']}")

    all_val_elbos = np.stack(all_val_elbos, axis=0)   # (folds, epochs)
    mean_val_elbos = all_val_elbos.mean(axis=0)
    selected_epoch = int(np.argmin(mean_val_elbos) + 1)

    print("\n===== Vanilla CV summary =====")
    print(f"Best epochs per fold: {[int(np.argmin(all_val_elbos[i]) + 1) for i in range(nr_splits)]}")
    print(f"selected_epoch: {selected_epoch}")
    return {"mean_val_elbos": mean_val_elbos, "selected_epoch": selected_epoch}


def train_final_ablated_model_after_cv(hidden, in_dim, A, X_train, y_train, device,
                                        n_epochs, lr=0.001, checkpoint_dir=None, loss_dir=None,
                                        dec_in_dim=None, n_tasks=None, task_emb_dim=0,
                                        task_ids=None):
    """
    Train AblatedRNN on all data for exactly n_epochs (the CV-selected count).
    The final epoch's state is used as the checkpoint (force_last_epoch=True).
    """
    _dec_in_dim = dec_in_dim if dec_in_dim is not None else in_dim
    _block_structure = (X_train.dim() == 4)
    model = AblatedRNN(hid=hidden, in_dim=_dec_in_dim, A=A, block_structure=_block_structure,
                       n_tasks=n_tasks, task_emb_dim=task_emb_dim).to(device)
    if task_ids is not None:
        model.set_task_ids(task_ids)
    model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict = \
        train_ablated_noblocks_palminteri(
            model, X_train, y_train,
            X_val=None, y_val=None,
            epoch_nr=n_epochs, lr=lr,
            checkpoint_dir=checkpoint_dir, loss_dir=loss_dir,
            force_last_epoch=True,
        )
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict


def train_latentrnn_IDRNN(model, xenc, blocks, y, lookup_z, xenc_val, y_val, p_target, device, ctest, ctrain,alpha_values_test, checkpoint_dir, rsa_dir, loss_dir, epochs=60, patience = 300, lr=1e-3, window_size=40, lmbd = 0.5):
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
    best_val_acc = 0
    pA_per_epoch = {}
    accuracy_dict = {}
    val_accuracy_dict = {}
    val_acc_history = []
    model_state_dict = {}
    best_RSA_corr = 0
    
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
                    #best_epoch = ep
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
        #model_state_dict[ep] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 100 == 0 or ep == 1:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch{ep:04d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            # save this externally in folder for run 1-5 (structured after seeds)
            print(f"[{model.name}], epoch {ep:3d},  train loss {loss.item():.3f},  val loss {val_loss.item():.3f},  train_acciracy: {final_acc},  val_accuracy: {final_val_acc}")
            # roll out the model for testing here (just use current model state and call the posterior sampling function)
            # collect latents (from the model and also the original parameters - the latter will have to be passed to the training function as an argument)
            # run RSA on both and compute correlation. Log the correlation 
            _, latent_tensor, _, _ = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xenc_val.squeeze(1), ctest, xenc.squeeze(1), latent=True, choice_train=None, id=True)
            distance_matrix_latents, latent_order = plf.rsa_latents(latents = latent_tensor, metric="euclidean",title="training_run_latentmodel", reduction="last", plot=False, original_data=False,cluster_order=False)
            distance_matrix_params, params_order = plf.rsa_latents(latents = alpha_values_test, metric="euclidean",title="test_params", reduction="entire", plot=False, original_data=True, cluster_order=False)
            vec_params  = vectorize_rsa(distance_matrix_params)
            vec_latents = vectorize_rsa(distance_matrix_latents)
            #save latent RSA at each checkpoint in the same folder logic as checkpoints (folder 1-5, structured after seeds)
            rsa_path = os.path.join(rsa_dir, f"epoch_{ep:04d}.npy")
            np.save(rsa_path, vec_latents)
            corr_matrix = np.corrcoef(vec_params,vec_latents)[0,1]
            # log the training loss as well
            loss_path = os.path.join(loss_dir, f"epoch_{ep:04d}.npy")
            np.save(loss_path, loss.cpu().item())
            

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_val_acc_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        
        wandb.log({"Cross Entropy Loss": val_policy_loss, "kl_loss": kl_loss, "accuracy_test": final_val_acc, "Correlation original params & latents": corr_matrix}, step=ep)

        if corr_matrix > best_RSA_corr:
            best_RSA_corr = corr_matrix
            best_RSA_corr_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            

        
            

        #if epochs_no_improve >= patience:
            #print(f"[{model.name}] Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            #break

    #best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    """window = window_size
    smoothed_val_acc = np.convolve(val_acc_history, np.ones(window)/window, mode='valid')
    best_epoch = np.argmax(smoothed_val_acc) + window // 2
    best_model_state = model_state_dict[best_epoch]"""
    model.load_state_dict(best_RSA_corr_state)


    print(f"best model state at epoch {best_epoch} with acc: {best_val_acc}")
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

def train_ablated_noblocks_palminteri(model, X_train, y_train, X_val=None, y_val=None, epoch_nr=3000, lr=0.001, checkpoint_dir=None, loss_dir=None, force_last_epoch=False):
    train_elbos = []
    val_elbos = []
    kl_vals = []
    best_val_loss = float('inf')
    best_acc = 0
    has_val = X_val is not None and y_val is not None
    pA_per_epoch = {}
    train_accuracy_dict = {}
    val_accuracy_dict = {}

    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    for ep in range(1, epoch_nr + 1):
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

        # === Validation or train-based model selection ===
        if has_val:
            model.eval()
            with torch.no_grad():
                logits_val, _, _ = model(X_val)
                nll_val = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1).long())
                val_elbos.append(nll_val.item())
                preds_val = logits_val.view(-1, logits_val.size(-1)).argmax(dim=-1)
                targets_val = y_val.view(-1).long()
                acc_val = (preds_val == targets_val).float().mean().item()
                val_accuracy_dict[ep] = acc_val
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                if ep % 100 == 0:
                    print(f"Epoch {ep:3d},  Train loss: {nll.item():.3f},  Train accuracy: {acc}, Validation accuracy: {acc_val}")
        else:
            if acc > best_acc:
                best_acc = acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep
            if ep % 100 == 0:
                print(f"Epoch {ep:3d},  Train loss: {nll.item():.3f},  Train accuracy: {acc}")

        # Save checkpoint and loss every 100 epochs
        if ep % 100 == 0:
            if checkpoint_dir is not None:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch{ep:04d}.pt")
                torch.save(model.state_dict(), ckpt_path)
            if loss_dir is not None:
                loss_path = os.path.join(loss_dir, f"epoch_{ep:04d}.npy")
                np.save(loss_path, np.array(nll.item()))

    # When training the final model after CV (no val data), keep the last epoch's
    # state instead of the best-train-accuracy epoch.
    if not has_val and force_last_epoch:
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch_nr

    print(f"best accuracy: {best_acc} at epoch: {best_epoch}")
    model.load_state_dict(best_model_state)
    logits_best_model, _, _ = model(X_train)
    pA_best_model = F.softmax(logits_best_model, dim=-1)
    pA_per_epoch[str(best_epoch)] = pA_best_model
    training_dict = {"predictions": pA_best_model,
            "weights": best_model_state,
            "best_model": model,
            "best_epoch": best_epoch,
            "best_acc": best_acc}
    return model, train_elbos, val_elbos, kl_vals, pA_per_epoch, training_dict

def train_ablated_noblocks(model, X_train, y_train, X_test, y_test, device, ctest, ctrain, alpha_values_test, checkpoint_dir, rsa_dir, loss_dir, p_target=None, p_test = None, epoch_nr=3000, lr=0.001):
    train_elbos = []
    val_elbos = []
    kl_vals = []
    kl_test_vals = []
    accuracy = []
    test_accuracy = []
    test_loss = []
    best_kl = float('inf')
    best_test_kl = float('inf')
    best_acc = 0
    y_train = torch.argmax(y_train, dim=-1)
    y_test = torch.argmax(y_test, dim=-1)
    pA_per_epoch = {} # dictionary to store pA predictions for later comparison

    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    for ep in range(1, epoch_nr + 1):
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
        model.eval()
        with torch.no_grad():
            if p_target is not None:
                
                
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

            if acc_test > best_acc:
                best_acc = acc_test
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep

            """if kl < best_kl:
                best_kl = kl
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep
            if kl_test < best_test_kl:
                best_test_kl = kl_test
                best_model_state_test = {k: v.clone() for k, v in model.state_dict().items()}
                best_test_epoch_kl = ep"""

            if ep % 100 == 0 or ep == 1:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch{ep:04d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Epoch {ep:3d}  Train loss {nll.item():.3f}  KL {kl:.3f}  accuracy {acc}  test loss {nll_test}  test accuracy {acc_test}  test kl {kl_test}")
                # roll out the model for testing here (just use current model state and call the posterior sampling function)
                # collect latents (from the model and also the original parameters - the latter will have to be passed to the training function as an argument)
                # run RSA on both and compute correlation. Log the correlation 
                _, latent_tensor, _, _ = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, X_test, ctest, X_train, latent=False, choice_train=ctrain, id=False)
                distance_matrix_latents, latent_order = plf.rsa_latents(latents = latent_tensor, metric="euclidean",title="training_run_vanilla", reduction="avg", plot=False, original_data=False, cluster_order= False)
                distance_matrix_params, params_order = plf.rsa_latents(latents = alpha_values_test, metric="euclidean",title="test_params", reduction="entire", plot=False, original_data=True, cluster_order=False)
                vec_params  = vectorize_rsa(distance_matrix_params)
                vec_latents = vectorize_rsa(distance_matrix_latents)
                corr_matrix = np.corrcoef(vec_params,vec_latents)[0,1]
                rsa_path = os.path.join(rsa_dir, f"epoch_{ep:04d}.npy")
                np.save(rsa_path, vec_latents)
                # log the training loss as well
                loss_path = os.path.join(loss_dir, f"epoch_{ep:04d}.npy")
                np.save(loss_path, nll.cpu().item())
        
            wandb.log({"Cross Entropy Loss": nll, "Cross Entropy Test Loss": nll_test, "accuracy_test": acc_test, "Correlation original params & latents": corr_matrix}, step=ep)       

    
    best_epoch_test_acc = test_accuracy.index(max(test_accuracy))+1
    best_epoch_test_loss = test_loss.index(min(test_loss))+1
    best_epoch_train_loss = train_elbos.index(min(train_elbos))+1
    best_epoch_acc = accuracy.index(max(accuracy))+1
    #normalized_ll = torch.exp(nll_test / targets_test.numel()).cpu().item()

    print(f"Best epoch test accuracy: {best_epoch}")
    print(f"Best test accuracy: {best_acc}")
    #print(f"Best test kl epoch: {best_test_epoch_kl}")
    #print(f"Best loss epoch: {best_epoch_train_loss}")
    #print(f"Best test loss epoch: {best_epoch_test_loss}")
    #print(f"Best accuracy loss epoch: {best_epoch_acc}")
    #print(f"Best test accuracy loss epoch: {best_epoch_test_acc}")
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


def train_IDRNN_joint(model, xenc, blocks, y, xenc_val, y_val, p_target, device, ctest, ctrain,
                      alpha_values_test, checkpoint_dir, rsa_dir, loss_dir,
                      epochs=10000, lr=1e-3, beta=0.1):
    """
    Joint end-to-end training of IDRNN encoder + decoder.

    Key difference from two-step training:
    - Encoder and decoder are trained together from scratch
    - KL regularizer toward standard normal prior (not lookup embeddings)
    - beta controls KL weight (like beta-VAE)

    Args:
        model: LatentRNN_secondstep with trainable encoder AND decoder
        xenc: encoder input (B, 1, T, in_dim)
        blocks: decoder input (B, 1, T, in_dim) - same as xenc here
        y: action labels (B, 1, T)
        xenc_val: validation encoder input
        y_val: validation labels
        p_target: target action probabilities for KL monitoring
        beta: weight on KL divergence to prior (start small, e.g., 0.01-0.1)
    """
    train_losses = []
    val_losses = []
    kl_to_prior_vals = []
    best_val_acc = 0
    best_RSA_corr = 0
    best_epoch = 0
    best_state = None
    pA_per_epoch = {}

    # ALL parameters are trainable (encoder + decoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # Forward pass - encoder produces mu, logvar; decoder predicts actions
        logits, mu, lv, z, h0_ = model(xenc, blocks, sample_z=True)  # sample during training

        # Policy loss (cross-entropy on action prediction)
        B, Bk, T, A = logits.shape
        policy_loss = F.cross_entropy(logits.reshape(-1, A), y.reshape(-1).long(), reduction='mean')

        # KL divergence to standard normal prior: KL(q(z|x) || N(0,1))
        # For per-timestep z: mu, lv have shape (B, 1, T, z_dim)
        # KL = 0.5 * sum(mu^2 + var - 1 - log(var))
        kl_to_prior = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).mean()

        # Total loss
        loss = policy_loss + beta * kl_to_prior
        loss.backward()
        opt.step()

        train_losses.append(loss.item())
        kl_to_prior_vals.append(kl_to_prior.item())

        # Training accuracy
        preds = logits.reshape(-1, A).argmax(dim=-1)
        targets = y.reshape(-1).long()
        train_acc = (preds == targets).float().mean().item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, mu_val, lv_val, _, _ = model(xenc_val, xenc_val, sample_z=False)
            val_policy_loss = F.cross_entropy(val_logits.reshape(-1, A), y_val.reshape(-1).long(), reduction='mean')
            val_kl = 0.5 * (mu_val.pow(2) + lv_val.exp() - 1 - lv_val).mean()
            val_loss = val_policy_loss + beta * val_kl
            val_losses.append(val_loss.item())

            # Validation accuracy
            val_preds = val_logits.reshape(-1, A).argmax(dim=-1)
            val_targets = y_val.reshape(-1).long()
            val_acc = (val_preds == val_targets).float().mean().item()

        # Checkpointing and RSA logging (same as your existing code)
        if ep % 100 == 0 or ep == 1:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch{ep:04d}.pt")
            torch.save(model.state_dict(), ckpt_path)

            print(f"[IDRNN_joint] ep {ep:4d}  loss {loss.item():.3f}  "
                  f"policy {policy_loss.item():.3f}  kl_prior {kl_to_prior.item():.3f}  "
                  f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

            # RSA evaluation
            _, latent_tensor, _, _ = compute_rnn_likelihoods_torch(
                test_latentrnn_secondstep_causal_posterior_weighting,
                model, xenc_val.squeeze(1), ctest, xenc.squeeze(1),
                latent=True, choice_train=None, id=True
            )
            distance_matrix_latents, _ = plf.rsa_latents(
                latents=latent_tensor, metric="euclidean",
                title="joint_training", reduction="last",
                plot=False, original_data=False, cluster_order=False
            )
            distance_matrix_params, _ = plf.rsa_latents(
                latents=alpha_values_test, metric="euclidean",
                title="test_params", reduction="entire",
                plot=False, original_data=True, cluster_order=False
            )
            vec_params = vectorize_rsa(distance_matrix_params)
            vec_latents = vectorize_rsa(distance_matrix_latents)
            corr_matrix = np.corrcoef(vec_params, vec_latents)[0, 1]

            # Save RSA and loss
            np.save(os.path.join(rsa_dir, f"epoch_{ep:04d}.npy"), vec_latents)
            np.save(os.path.join(loss_dir, f"epoch_{ep:04d}.npy"), loss.cpu().item())

            wandb.log({
                "CE_loss": policy_loss.item(),
                "KL_to_prior": kl_to_prior.item(),
                "total_loss": loss.item(),
                "accuracy_test": val_acc,
                "RSA_correlation": corr_matrix
            }, step=ep)

        # Track best by RSA correlation (or change to val_acc if preferred)
        if ep % 100 == 0:
            if corr_matrix > best_RSA_corr:
                best_RSA_corr = corr_matrix
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = ep

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_best, mu_best, lv_best, z_best, h0_best = model(xenc, blocks, sample_z=False)
        pA_best = F.softmax(logits_best, dim=-1).squeeze(1)[:, :, 0]
        pA_per_epoch[str(best_epoch)] = pA_best

    training_dict = {
        "predictions": pA_best,
        "weights": best_state,
        "best_model": model,
        "best_epoch": best_epoch,
        "best_RSA_corr": best_RSA_corr,
        "z": mu_best,
        "h0": h0_best
    }

    print(f"Best RSA correlation: {best_RSA_corr:.3f} at epoch {best_epoch}")

    return model, mu_best, lv_best, train_losses, val_losses, training_dict, pA_per_epoch
