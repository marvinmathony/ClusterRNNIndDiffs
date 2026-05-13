#!/usr/bin/env python3
"""
Diagnose overtraining on Sloutsky data by sweeping checkpoints.

For each saved checkpoint epoch, computes:
  - Test NLL (cross-entropy on held-out test set, per trial)
  - Train NLL (from saved loss files, per trial)
  - Latent RSA: Spearman r of IDRNN latent RDM vs. age / cog-model param RDMs
  - Hidden-state RSA: same for Vanilla model (time-averaged GRU hidden states)

Uses hyperparameters from run_sloutsky.sbatch:
  lmbd=0.005, z=5, seeds=[12, 50, 76, 100, 142]

Run from the project root with the RNNproject conda env active:
    python diagnose_overtraining_sloutsky.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config, load_model_checkpoint
)
from sloutsky_cog_model import unpack_params

# ── Config (mirrors run_sloutsky.sbatch) ──────────────────────────────────────
SEEDS     = [12, 50, 76, 100, 142]
DATA_DIR  = "data_sloutsky"
IDRNN_DIR = "runs_sloutsky"
VAN_DIR   = "runs_vanilla_sloutsky"
DEVICE    = torch.device("cpu")   # CPU is fine for eval

# ── Load test data ─────────────────────────────────────────────────────────────
xin_test         = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float()
choice_one_hot   = torch.from_numpy(np.load(f"{DATA_DIR}/choice_one_hot_test.npy")).float()
c_test           = torch.from_numpy(np.load(f"{DATA_DIR}/c_test.npy")).float()
B, T, in_dim     = xin_test.shape

# Integer labels for cross-entropy: use c_test if 2-D, else argmax of one-hot
if c_test.ndim == 2 and c_test.shape == (B, T):
    y_test = c_test.long()                          # already integer labels
else:
    y_test = torch.argmax(choice_one_hot, dim=-1)   # (B, T)

# Encoder-specific input: default to split file; fall back to full input if absent.
# If the saved model config shows enc_in_dim == in_dim (trained with --same_enc_dec),
# override to use xin_test so dimensions match the encoder.
_enc_path    = f"{DATA_DIR}/xin_enc_test.npy"
xin_enc_test = (torch.from_numpy(np.load(_enc_path)).float()
                if os.path.exists(_enc_path) else xin_test)

_sample_cfg = load_model_config(os.path.join(IDRNN_DIR, f"seed_{SEEDS[0]}"))
if _sample_cfg and _sample_cfg.get("enc_in_dim", _sample_cfg["in_dim"]) == _sample_cfg["in_dim"]:
    print("Detected same_enc_dec mode: using full xin_test for IDRNN encoder.")
    xin_enc_test = xin_test

# ── RSA targets ────────────────────────────────────────────────────────────────
df_test     = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_all      = pd.read_csv(f"{DATA_DIR}/exp2_train_all_participants.csv")
test_subids = sorted(df_test["subid"].unique())
assert len(test_subids) == B, f"Expected {B} participants, got {len(test_subids)}"

age_map = df_all.drop_duplicates("subid").set_index("subid")["age"].to_dict()
age_ord = {"young_child": 0, "old_child": 1, "adult": 2}
y_age   = np.array([age_ord[age_map[s]] for s in test_subids])
age_rdm = pdist(y_age.reshape(-1, 1))

em           = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all_test   = em["h_all_test"]
cog_params   = [unpack_params(h_all_test[i]) for i in range(B)]
theta_rdm    = pdist(np.array([p["theta"]              for p in cog_params]).reshape(-1, 1))
w_value_rdm  = pdist(np.array([p["b_value_train"]      for p in cog_params]).reshape(-1, 1))
w_uncert_rdm = pdist(np.array([p["b_uncertain_train"]  for p in cog_params]).reshape(-1, 1))
w_novel_rdm  = pdist(np.array([p["b_novelty_train"]    for p in cog_params]).reshape(-1, 1))

# ── Helpers ───────────────────────────────────────────────────────────────────
def nll_vanilla(model, x, y_labels):
    """Mean NLL per trial (cross-entropy) for AblatedRNN.
    Also returns time-averaged hidden states (B, hid) for RSA."""
    model.eval()
    with torch.no_grad():
        logits, _, hidden_tr = model(x)  # hidden_tr: (B, T, hid)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_labels.reshape(-1))
        hidden_avg = hidden_tr.contiguous().mean(dim=1).cpu().numpy()  # (B, hid)
    return nll.item(), hidden_avg


def nll_idrnn(model, x, x_enc, y_labels):
    """Mean NLL per trial for IDRNN using full-sequence (non-causal) posterior."""
    model.eval()
    with torch.no_grad():
        xenc   = x_enc.unsqueeze(1)       # (B, 1, T, enc_in_dim)
        blocks = x.unsqueeze(1)           # (B, 1, T, in_dim)
        logits, mu, _, _, _ = model(xenc, blocks, sample_z=False)
        logits = logits.squeeze(1).contiguous()   # (B, T, A)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_labels.reshape(-1))
    return nll.item(), mu.cpu().numpy()   # also return latents (B, z_dim)


def _rsa_from_repr(repr_matrix):
    """Spearman r between pairwise-distance RDM of repr_matrix and each behavioral RDM."""
    rdm = pdist(repr_matrix)
    r_age,    _ = spearmanr(rdm, age_rdm)
    r_theta,  _ = spearmanr(rdm, theta_rdm)
    r_value,  _ = spearmanr(rdm, w_value_rdm)
    r_uncert, _ = spearmanr(rdm, w_uncert_rdm)
    r_novel,  _ = spearmanr(rdm, w_novel_rdm)
    return r_age, r_theta, r_value, r_uncert, r_novel


def rsa_idrnn(mu):
    """Spearman r using IDRNN mean latents (B, z_dim)."""
    return _rsa_from_repr(mu)


def rsa_vanilla(hidden_avg):
    """Spearman r using time-averaged vanilla GRU hidden states (B, hid)."""
    return _rsa_from_repr(hidden_avg)


def available_epochs(base_dir, seed):
    ckpt_dir = os.path.join(base_dir, f"seed_{seed}", "checkpoints")
    if not os.path.exists(ckpt_dir):
        return set()
    return {
        int(f.replace("epoch", "").replace(".pt", ""))
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch") and f.endswith(".pt")
    }


def load_train_nll(base_dir, seed, epoch):
    """Return saved training NLL for this epoch (scalar). None if missing."""
    p = os.path.join(base_dir, f"seed_{seed}", "loss", f"epoch_{epoch:04d}.npy")
    if os.path.exists(p):
        return float(np.load(p))
    return None


# ── Epochs to evaluate ────────────────────────────────────────────────────────
idrnn_epochs = set.intersection(*[available_epochs(IDRNN_DIR, s) for s in SEEDS])
van_epochs   = set.intersection(*[available_epochs(VAN_DIR,   s) for s in SEEDS])
all_epochs   = sorted(idrnn_epochs & van_epochs)

print(f"Found {len(all_epochs)} epochs common to both models across all seeds.")
print(f"Epochs: {all_epochs[:5]} ... {all_epochs[-3:]}")
print(f"B (test participants) = {B},  T (trials) = {T}\n")

# ── Main sweep ────────────────────────────────────────────────────────────────
rows = []

for epoch in all_epochs:
    van_test_nlls,   van_train_nlls   = [], []
    idrnn_test_nlls, idrnn_train_nlls = [], []
    van_rsa_vals  = []
    idrnn_rsa_vals = []

    for seed in SEEDS:
        # ---- Vanilla ----
        run_dir_v = os.path.join(VAN_DIR, f"seed_{seed}")
        ckpt_v    = os.path.join(run_dir_v, "checkpoints", f"epoch{epoch:04d}.pt")
        try:
            cfg_v   = load_model_config(run_dir_v)
            model_v = create_model_from_config(cfg_v, n_participants=B, device=DEVICE)
            model_v = load_model_checkpoint(ckpt_v, model_v, DEVICE)
            nll_v, hidden_avg = nll_vanilla(model_v, xin_test, y_test)
            van_test_nlls.append(nll_v)
            van_rsa_vals.append(rsa_vanilla(hidden_avg))
            t_nll = load_train_nll(VAN_DIR, seed, epoch)
            if t_nll is not None:
                van_train_nlls.append(t_nll)
        except Exception as e:
            print(f"  [vanilla seed={seed} ep={epoch}] {e}")

        # ---- IDRNN ----
        run_dir_i = os.path.join(IDRNN_DIR, f"seed_{seed}")
        ckpt_i    = os.path.join(run_dir_i, "checkpoints", f"epoch{epoch:04d}.pt")
        dec_path  = os.path.join(run_dir_i, "frozen_decoder", "policy_model.pt")
        try:
            cfg_i    = load_model_config(run_dir_i)
            model_i  = create_model_from_config(
                cfg_i, n_participants=B, device=DEVICE, frozen_decoder_path=dec_path
            )
            model_i  = load_model_checkpoint(ckpt_i, model_i, DEVICE)
            nll_i, mu = nll_idrnn(model_i, xin_test, xin_enc_test, y_test)
            idrnn_test_nlls.append(nll_i)
            idrnn_rsa_vals.append(rsa_idrnn(mu))
            t_nll = load_train_nll(IDRNN_DIR, seed, epoch)
            if t_nll is not None:
                idrnn_train_nlls.append(t_nll)
        except Exception as e:
            print(f"  [IDRNN  seed={seed} ep={epoch}] {e}")

    RSA_NAMES = ["r_age", "r_theta", "r_value", "r_uncert", "r_novel"]
    row = {
        "epoch":           epoch,
        "van_test_nll":    np.mean(van_test_nlls)    if van_test_nlls    else float("nan"),
        "van_test_std":    np.std(van_test_nlls)     if van_test_nlls    else float("nan"),
        "van_train_nll":   np.mean(van_train_nlls)   if van_train_nlls   else float("nan"),
        "idrnn_test_nll":  np.mean(idrnn_test_nlls)  if idrnn_test_nlls  else float("nan"),
        "idrnn_test_std":  np.std(idrnn_test_nlls)   if idrnn_test_nlls  else float("nan"),
        "idrnn_train_nll": np.mean(idrnn_train_nlls) if idrnn_train_nlls else float("nan"),
    }
    if van_rsa_vals:
        rsas = np.array(van_rsa_vals)
        for i, name in enumerate(RSA_NAMES):
            row[f"v_{name}"] = float(np.mean(rsas[:, i]))
    if idrnn_rsa_vals:
        rsas = np.array(idrnn_rsa_vals)
        for i, name in enumerate(RSA_NAMES):
            row[f"i_{name}"] = float(np.mean(rsas[:, i]))
    rows.append(row)

# ── Print summary ──────────────────────────────────────────────────────────────
RSA_NAMES = ["r_age", "r_theta", "r_value", "r_uncert", "r_novel"]
nan = float("nan")

W_NLL = 80
print("\n" + "=" * W_NLL)
print(f"{'Overtraining Diagnosis — Sloutsky  (↓ NLL is better)':^{W_NLL}}")
print(f"{'Test NLL = cross-entropy on held-out set  |  Train NLL = saved training loss':^{W_NLL}}")
print("=" * W_NLL)
print(f"{'Epoch':>6} | {'Van testNLL':>11} {'±':>1} {'std':>6} | {'Van trainNLL':>12} "
      f"| {'IDRNN testNLL':>13} {'±':>1} {'std':>6} | {'IDRNN trainNLL':>14}")
print("-" * W_NLL)
for r in rows:
    print(f"{r['epoch']:>6} | "
          f"{r['van_test_nll']:>11.4f} ± {r['van_test_std']:>6.4f} | "
          f"{r['van_train_nll']:>12.4f} | "
          f"{r['idrnn_test_nll']:>13.4f} ± {r['idrnn_test_std']:>6.4f} | "
          f"{r['idrnn_train_nll']:>14.4f}")
print("=" * W_NLL)

W_RSA = 110
print(f"\n{'RSA — Spearman r of representation RDM vs. behavioral RDMs (mean across seeds)':^{W_RSA}}")
print(f"{'IDRNN: mean latent z  |  Vanilla: time-averaged GRU hidden states':^{W_RSA}}")
print("=" * W_RSA)
col_hdr = (f"{'Epoch':>6} | "
           f"{'I_age':>6} {'I_theta':>7} {'I_value':>7} {'I_uncert':>8} {'I_novel':>7} | "
           f"{'V_age':>6} {'V_theta':>7} {'V_value':>7} {'V_uncert':>8} {'V_novel':>7}")
print(col_hdr)
print("-" * W_RSA)
for r in rows:
    i_vals = " ".join(f"{r.get(f'i_{n}', nan):>7.3f}" for n in RSA_NAMES)
    v_vals = " ".join(f"{r.get(f'v_{n}', nan):>7.3f}" for n in RSA_NAMES)
    print(f"{r['epoch']:>6} | {i_vals} | {v_vals}")
print("=" * W_RSA)

print("\nNotes:")
print("  - Test NLL: non-causal full-sequence posterior for IDRNN (consistent across epochs)")
print("  - Train NLL: NLL saved during training (training set only, different scale for IDRNN)")
print("  - RSA: Spearman r; I_ = IDRNN latents, V_ = Vanilla time-averaged hidden states")
print("  - Overtraining in NLL: test NLL increases after some epoch while train NLL keeps falling")
print("  - Overtraining in RSA: RSA peaks then degrades as representations become idiosyncratic")
