#!/usr/bin/env python3
"""
Compare IDRNN (with uniformity loss, seed_12) vs Vanilla on Thalmann data.

Metrics:
  1. Pure NLL on held-out test set
  2. LOO-CV ridge regression decoding of questionnaire scores from latents
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, LatentRNNz, LatentRNN_secondstep, AblatedRNN

# ── Config ───────────────────────────────────────────────────────────────────
IDRNN_RUN   = "runs_thalmann/seed_12"
VANILLA_DIR = "runs_vanilla_thalmann/fold0"  # pick one seed
VANILLA_SEED = "seed_200"
QUEST_PATH  = "data/finalQuestionnaireDataSession1.csv"
TASK_IDS_PATH = "data_thalmann/task_ids_per_block.npy"

# Questionnaire composite scales
PANAS_PA_ITEMS  = [0, 2, 4, 6, 8, 14, 16, 18]   # positive affect
PANAS_NA_ITEMS  = [1, 3, 5, 7, 9, 11, 13, 15]   # negative affect
STICSA_ITEMS    = list(range(22))
PHQ_ITEMS       = list(range(10))
CEI_ITEMS       = list(range(4))
BIG5_ITEMS      = list(range(6))

# ── Load questionnaire data ───────────────────────────────────────────────────
quest = pd.read_csv(QUEST_PATH)
quest = quest.set_index("ID")
quest["PANAS_PA"]  = quest[[f"PANAS_{i}" for i in PANAS_PA_ITEMS]].mean(axis=1)
quest["PANAS_NA"]  = quest[[f"PANAS_{i}" for i in PANAS_NA_ITEMS]].mean(axis=1)
quest["STICSA"]    = quest[[f"STICSA_{i}" for i in STICSA_ITEMS]].mean(axis=1)
quest["PHQ"]       = quest[[f"PHQ_9_{i}" for i in PHQ_ITEMS]].mean(axis=1)
quest["CEI"]       = quest[[f"CEI_{i}" for i in CEI_ITEMS]].mean(axis=1)
quest["BIG5_open"] = quest[[f"BIG_5_{i}" for i in BIG5_ITEMS]].mean(axis=1)
SCALES = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open"]

# ── Helper: LOO-CV ridge regression (within-fold z-score to avoid confound) ──
def loo_ridge_decode(Z, y):
    """
    Z: (N, d) latent matrix, y: (N,) target
    Returns mean Pearson r across LOO folds.
    """
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(Z):
        Z_tr, Z_te = Z[train_idx], Z[test_idx]
        y_tr = y[train_idx]
        # z-score within fold
        mu_z, sd_z = Z_tr.mean(0), Z_tr.std(0) + 1e-8
        mu_y, sd_y = y_tr.mean(), y_tr.std() + 1e-8
        Z_tr_s = (Z_tr - mu_z) / sd_z
        Z_te_s = (Z_te - mu_z) / sd_z
        y_tr_s = (y_tr - mu_y) / sd_y
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit(Z_tr_s, y_tr_s)
        preds[test_idx] = clf.predict(Z_te_s) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return r, p

# ── 1. Load IDRNN model and get encoder z ────────────────────────────────────
print("=" * 60)
print("Loading IDRNN (uniformity loss) model...")

with open(os.path.join(IDRNN_RUN, "config.json")) as f:
    idrnn_cfg = json.load(f)
mc = idrnn_cfg["model_config"]

state = torch.load(
    os.path.join(IDRNN_RUN, "checkpoints",
                 f"epoch{idrnn_cfg['cv_selected_epoch']:04d}.pt"),
    map_location="cpu"
)

enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
            n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
enc.load_state_dict(enc_state)
enc.eval()

task_ids = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)
enc.set_task_ids(task_ids)

xenc_train = torch.tensor(np.load("data_thalmann/xin_train.npy"), dtype=torch.float32)
xenc_test  = torch.tensor(np.load("data_thalmann/xin_test.npy"),  dtype=torch.float32)

with torch.no_grad():
    mu_train_full, _ = enc(xenc_train)   # (B, Bk, T, z_dim)
    mu_test_full,  _ = enc(xenc_test)

# Final z: last block (restless), last timestep
Z_idrnn_train = mu_train_full[:, -1, -1, :].numpy()   # (165, z_dim)
Z_idrnn_test  = mu_test_full[:,  -1, -1, :].numpy()   # (71, z_dim)

train_ids = pd.read_csv("data_thalmann/df_train.csv")["subid"].values
test_ids  = pd.read_csv("data_thalmann/df_test.csv")["subid"].values

print(f"IDRNN z train: {Z_idrnn_train.shape}, std mean: {Z_idrnn_train.std(0).mean():.4f}")
print(f"IDRNN z test:  {Z_idrnn_test.shape},  std mean: {Z_idrnn_test.std(0).mean():.4f}")

# ── 2. Compute IDRNN pure NLL on test set ────────────────────────────────────
print("\nComputing IDRNN pure NLL on test set...")

# Rebuild full model with frozen decoder for forward pass
# The frozen_decoder checkpoint is the full step-1 LatentRNNz model.
# Load it, extract its decoder submodule, then build LatentRNN_secondstep.
step1_state = torch.load(os.path.join(IDRNN_RUN, "frozen_decoder", "policy_model.pt"),
                         map_location="cpu")
dec = LatentRNNz(mc["hidden"], mc["z_dim"], mc["in_dim"], mc["dec_in_dim"], mc["A"],
                 n_tasks=mc.get("n_tasks", 2), task_emb_dim=mc.get("task_emb_dim", 0))
dec_only_state = {k[len("decoder."):]: v for k, v in step1_state.items() if k.startswith("decoder.")}
try:
    dec.decoder.load_state_dict(dec_only_state)
    dec.eval()
    model = LatentRNN_secondstep(enc, mc["hidden"], mc["z_dim"], mc["in_dim"], mc["A"], dec)
    # load encoder weights from step-2 checkpoint (already loaded enc above)
    model.eval()
    model.set_task_ids(task_ids)

    y_test = torch.tensor(
        np.argmax(np.load("data_thalmann/choice_one_hot_test.npy"), axis=-1),
        dtype=torch.long
    )  # (B, Bk, T)

    with torch.no_grad():
        logits_test, _, _, _, _ = model(xenc_test, xenc_test, sample_z=False)
    # Pure NLL (cross-entropy) — all blocks
    B, Bk, T, A = logits_test.shape
    idrnn_nll_all = F.cross_entropy(
        logits_test.reshape(-1, A), y_test.reshape(-1).long(), reduction="mean"
    ).item()
    # NLL on restless block only (block index 30)
    idrnn_nll_restless = F.cross_entropy(
        logits_test[:, -1].reshape(-1, A),
        y_test[:, -1].reshape(-1).long(), reduction="mean"
    ).item()
    print(f"IDRNN NLL (all blocks):     {idrnn_nll_all:.4f}")
    print(f"IDRNN NLL (restless only):  {idrnn_nll_restless:.4f}")
    got_idrnn_nll = True
except Exception as e:
    print(f"Could not compute IDRNN NLL: {e}")
    got_idrnn_nll = False

# ── 3. Load Vanilla model and get hidden states ───────────────────────────────
print("\n" + "=" * 60)
print("Loading Vanilla model...")

van_seed_dir = os.path.join(VANILLA_DIR, VANILLA_SEED)
with open(os.path.join(van_seed_dir, "config.json")) as f:
    van_cfg = json.load(f)
van_mc = van_cfg["model_config"]

van_model = AblatedRNN(
    hid=van_mc["hidden"], in_dim=van_mc.get("dec_in_dim", van_mc["in_dim"]),
    A=van_mc["A"], block_structure=True,
    n_tasks=van_mc.get("n_tasks", 2), task_emb_dim=van_mc.get("task_emb_dim", 0)
)
van_epoch = van_cfg["cv_selected_epoch"]
van_ckpt  = os.path.join(van_seed_dir, "checkpoints", f"epoch{van_epoch:04d}.pt")
if not os.path.exists(van_ckpt):
    ckpts = sorted(os.listdir(os.path.join(van_seed_dir, "checkpoints")))
    van_ckpt = os.path.join(van_seed_dir, "checkpoints", ckpts[-1])
    print(f"  Using checkpoint: {os.path.basename(van_ckpt)}")
van_state = torch.load(van_ckpt, map_location="cpu")
van_model.load_state_dict(van_state, strict=False)
van_model.eval()
if hasattr(van_model, "set_task_ids"):
    van_model.set_task_ids(task_ids)

# Get vanilla hidden states (final timestep, last block = restless)
# AblatedRNN returns (logits, hidden) or similar — check forward output
with torch.no_grad():
    van_logits_train, van_final_hid_train, _ = van_model(xenc_train)
    van_logits_test,  van_final_hid_test,  _ = van_model(xenc_test)
# van_final_hid shape: (B, Bk, hidden) — final GRU hidden state per block

# Vanilla NLL on test set
y_test_all = torch.tensor(
    np.argmax(np.load("data_thalmann/choice_one_hot_test.npy"), axis=-1),
    dtype=torch.long
)
B2, Bk2, T2, A2 = van_logits_test.shape
van_nll_all = F.cross_entropy(
    van_logits_test.reshape(-1, A2), y_test_all.reshape(-1).long(), reduction="mean"
).item()
van_nll_restless = F.cross_entropy(
    van_logits_test[:, -1].reshape(-1, A2),
    y_test_all[:, -1].reshape(-1).long(), reduction="mean"
).item()
print(f"Vanilla NLL (all blocks):    {van_nll_all:.4f}")
print(f"Vanilla NLL (restless only): {van_nll_restless:.4f}")
print(f"  (CV val loss from config:  {van_cfg['cv_val_loss']:.4f})")

# van_final_hid shape from AblatedRNN: (Bk, B, hidden) due to stack+squeeze
# Permute to (B, Bk, hidden), then average over blocks → (B, hidden)
Z_vanilla_train = van_final_hid_train.permute(1, 0, 2).mean(dim=1).numpy()
Z_vanilla_test  = van_final_hid_test.permute(1, 0, 2).mean(dim=1).numpy()
print("  Using mean final GRU hidden state across blocks as vanilla representation")

print(f"Vanilla Z shape: {Z_vanilla_train.shape}")

# ── 4. Decode questionnaire scores ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Decoding questionnaire scores (LOO-CV ridge regression)...")

# Match subjects to questionnaire scores
quest_train = quest.loc[quest.index.isin(train_ids)].reindex(train_ids)
quest_test  = quest.loc[quest.index.isin(test_ids)].reindex(test_ids)

results = []
for scale in SCALES:
    y_tr = quest_train[scale].values.astype(float)
    y_te = quest_test[scale].values.astype(float)

    # IDRNN decoding (train subjects LOO)
    mask_idrnn = ~np.isnan(y_tr)
    if mask_idrnn.sum() > 10:
        r_idrnn, p_idrnn = loo_ridge_decode(Z_idrnn_train[mask_idrnn], y_tr[mask_idrnn])
    else:
        r_idrnn, p_idrnn = np.nan, np.nan

    # Vanilla decoding (train subjects LOO)
    mask_van = ~np.isnan(y_tr)
    if mask_van.sum() > 10:
        r_van, p_van = loo_ridge_decode(Z_vanilla_train[mask_van], y_tr[mask_van])
    else:
        r_van, p_van = np.nan, np.nan

    results.append({
        "scale":   scale,
        "r_idrnn": r_idrnn, "p_idrnn": p_idrnn,
        "r_van":   r_van,   "p_van":   p_van,
    })
    print(f"  {scale:12s}  IDRNN r={r_idrnn:+.3f} (p={p_idrnn:.3f})   "
          f"Vanilla r={r_van:+.3f} (p={p_van:.3f})")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nNLL comparison (test set):")
if got_idrnn_nll:
    print(f"  IDRNN all blocks:     {idrnn_nll_all:.4f}")
    print(f"  IDRNN restless only:  {idrnn_nll_restless:.4f}")
print(f"  Vanilla all blocks:   {van_nll_all:.4f}")
print(f"  Vanilla restless:     {van_nll_restless:.4f}")

print("\nDecoding r (LOO-CV, train subjects):")
df_res = pd.DataFrame(results).set_index("scale")
print(df_res[["r_idrnn", "p_idrnn", "r_van", "p_van"]].round(3).to_string())

sig_idrnn = (df_res["p_idrnn"] < 0.05).sum()
sig_van   = (df_res["p_van"]   < 0.05).sum()
print(f"\nSignificant decoding (p<0.05): IDRNN={sig_idrnn}/{len(SCALES)}, Vanilla={sig_van}/{len(SCALES)}")
