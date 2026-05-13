#!/usr/bin/env python3
"""
RSA analysis for Thalmann outer-CV unif_weight sweep.

For each unif_weight in {0.0, 0.1, 0.5, 1.0}:
  - Loads IDRNN encoder z from test-fold participants (outer-CV, same as unif_sweep_decoding)
  - Computes latent RDM (pairwise Euclidean distances)
  - Correlates (Spearman) with psychometric score RDMs:
      - Full 6-scale composite (standardised Euclidean)
      - Per-scale (each questionnaire individually)
  - Permutation test for significance (row/col shuffle of latent RDM)
  - Also computes vanilla (mean GRU hidden state) RSA as baseline

Outputs:
  plots_thalmann/rsa_unif_sweep_per_scale.png   — RSA r per scale × unif_weight
  plots_thalmann/rsa_unif_sweep_composite.png   — composite RSA vs vanilla
  plots_thalmann/rsa_unif_sweep_summary.png     — summary bar + significance
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN

# ── Config ────────────────────────────────────────────────────────────────────
DGP      = "thalmann"
N_FOLDS  = 3
UW_LIST  = [0.0, 0.1, 0.5, 1.0]
PLOT_DIR = f"plots_{DGP}"
PERM_N   = 1000
os.makedirs(PLOT_DIR, exist_ok=True)

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

# ── Questionnaire scales ───────────────────────────────────────────────────────
QUEST_PATH     = "data/finalQuestionnaireDataSession1.csv"
PANAS_PA_ITEMS = [0, 2, 4, 6, 8, 14, 16, 18]
PANAS_NA_ITEMS = [1, 3, 5, 7, 9, 11, 13, 15]
SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in PANAS_PA_ITEMS], "PANAS Pos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in PANAS_NA_ITEMS], "PANAS Neg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],       "STICSA Anxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],       "PHQ-9 Depression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],        "CEI Curiosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],        "BIG5 Openness"),
}
SCALE_KEYS   = list(SCALES.keys())
SCALE_LABELS = [SCALES[k][1] for k in SCALE_KEYS]

quest = pd.read_csv(QUEST_PATH).set_index("ID")
for key, (items, _) in SCALES.items():
    quest[key] = quest[items].mean(axis=1)


# ── RSA helpers ───────────────────────────────────────────────────────────────
def rdm_vec(X):
    """Upper-triangle of Euclidean RDM."""
    return pdist(X, metric="euclidean")


def rsa_spearman(lat_vec, tgt_vec):
    """Spearman r between two RDM vectors (NaN-safe)."""
    mask = ~(np.isnan(lat_vec) | np.isnan(tgt_vec))
    if mask.sum() < 10:
        return np.nan, 1.0
    r, p = spearmanr(lat_vec[mask], tgt_vec[mask])
    return float(r), float(p)


def permutation_rsa(lat, tgt_vec, n_perm=PERM_N, seed=0):
    """
    Row/col permutation of the latent RDM.
    Returns (observed_r, p_perm, null_distribution).
    """
    rng = np.random.default_rng(seed)
    n   = lat.shape[0]
    mat = squareform(rdm_vec(lat))
    triu = np.triu_indices(n, k=1)

    obs_r, _ = spearmanr(mat[triu], tgt_vec)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n)
        mat_perm = mat[np.ix_(idx, idx)]
        null[i], _ = spearmanr(mat_perm[triu], tgt_vec)

    p_perm = float((null >= obs_r).mean())
    return float(obs_r), p_perm, null


# ── Data loading ───────────────────────────────────────────────────────────────
def load_idrnn_z_test(fold, suffix):
    """Load encoder z for test subjects of one outer fold."""
    base     = f"runs_{DGP}_{suffix}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"
    if not os.path.isdir(base):
        return None, None

    seed_dirs = sorted([d for d in os.listdir(base)
                        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))])
    if not seed_dirs:
        return None, None

    xin_test = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    df_test  = pd.read_csv(f"{data_dir}/df_test.csv")
    subids   = df_test["subid"].values if "subid" in df_test.columns else df_test["session"].values

    z_seeds = []
    for sd in seed_dirs:
        run_dir  = os.path.join(base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue
        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]
        try:
            state     = torch.load(ckpt_path, map_location="cpu")
            enc       = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                              hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                              task_emb_dim=mc["task_emb_dim"])
            enc_state = {k[len("encoder."):]: v for k, v in state.items()
                         if k.startswith("encoder.")}
            enc.load_state_dict(enc_state)
            enc.eval()
            enc.set_task_ids(task_ids_global)
            with torch.no_grad():
                mu, _ = enc(xin_test)
            if mu.dim() == 4:
                mu = mu[:, -1, -1, :]
            z_seeds.append(mu.numpy())
        except Exception as e:
            print(f"    Warning ({sd}): {e}")

    if not z_seeds:
        return None, None
    return np.stack(z_seeds).mean(0), subids


def load_vanilla_z_test(fold):
    """Load vanilla mean-GRU-hidden for test subjects of one outer fold."""
    base     = f"runs_vanilla_{DGP}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"
    if not os.path.isdir(base):
        return None, None

    seed_dirs = sorted([d for d in os.listdir(base)
                        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))])
    if not seed_dirs:
        return None, None

    xin_test = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    df_test  = pd.read_csv(f"{data_dir}/df_test.csv")
    subids   = df_test["subid"].values if "subid" in df_test.columns else df_test["session"].values

    z_seeds = []
    for sd in seed_dirs:
        run_dir  = os.path.join(base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue
        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            dec_in = mc.get("dec_in_dim", mc["in_dim"])
            van = AblatedRNN(hid=mc["hidden"], in_dim=dec_in, A=mc["A"],
                             block_structure=True,
                             n_tasks=mc.get("n_tasks", 2),
                             task_emb_dim=mc.get("task_emb_dim", 0))
            van.load_state_dict(state)
            van.eval()
            van.set_task_ids(task_ids_global)
            with torch.no_grad():
                _, hid, _ = van(xin_test)   # hid: (Bk, B, hidden)
            z = hid.permute(1, 0, 2).mean(1).numpy()   # (B, hidden)
            z_seeds.append(z)
        except Exception as e:
            print(f"    Vanilla warning ({sd}): {e}")

    if not z_seeds:
        return None, None
    return np.stack(z_seeds).mean(0), subids


def pool_across_folds(fold_data):
    """[(z, subids), ...] → (z_all, subids_all)"""
    zs, ids = zip(*[(z, s) for z, s in fold_data if z is not None])
    return np.concatenate(zs), np.concatenate(ids)


def get_quest_scores(subids):
    """Return (scores_dict, mask_dict) aligned to subids."""
    scores, masks = {}, {}
    for key in SCALE_KEYS:
        y = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                      for sid in subids])
        scores[key] = y
        masks[key]  = ~np.isnan(y)
    return scores, masks


# ── Main loop ─────────────────────────────────────────────────────────────────
print("Computing RSA for Thalmann unif_weight sweep (outer-CV test subjects)\n")

# Storage
rsa_idrnn    = {uw: {} for uw in UW_LIST}   # uw → scale → (r, p_perm)
rsa_composite= {}                             # uw → (r, p_perm)

# Vanilla
print("Loading vanilla...")
van_folds = [load_vanilla_z_test(f) for f in range(N_FOLDS)]
van_folds_ok = [(z, s) for z, s in van_folds if z is not None]
if van_folds_ok:
    z_van, ids_van = pool_across_folds(van_folds_ok)
    scores_van, masks_van = get_quest_scores(ids_van)
    rsa_vanilla_per_scale = {}
    for key in SCALE_KEYS:
        mask = masks_van[key]
        if mask.sum() < 10:
            rsa_vanilla_per_scale[key] = (np.nan, 1.0)
            continue
        tgt = rdm_vec(scores_van[key][mask].reshape(-1, 1))
        r, p, _ = permutation_rsa(z_van[mask], tgt, n_perm=PERM_N)
        rsa_vanilla_per_scale[key] = (r, p)
        print(f"  Vanilla {key:12s}: r={r:+.3f}  p_perm={p:.3f}")

    # Composite (standardised all scales together)
    full_mask = np.stack([masks_van[k] for k in SCALE_KEYS], axis=1).all(axis=1)
    if full_mask.sum() > 10:
        Y = np.column_stack([(scores_van[k][full_mask] - scores_van[k][full_mask].mean())
                              / (scores_van[k][full_mask].std() + 1e-8)
                              for k in SCALE_KEYS])
        tgt_composite_van = rdm_vec(Y)
        r_van_c, p_van_c, _ = permutation_rsa(z_van[full_mask], tgt_composite_van, n_perm=PERM_N)
        print(f"  Vanilla COMPOSITE:   r={r_van_c:+.3f}  p_perm={p_van_c:.3f}")
    else:
        r_van_c, p_van_c = np.nan, 1.0

print()

for uw in UW_LIST:
    suffix = f"unif{str(uw).replace('.', 'p')}"
    print(f"unif_weight={uw}  (suffix={suffix})")

    folds = [load_idrnn_z_test(f, suffix) for f in range(N_FOLDS)]
    folds_ok = [(z, s) for z, s in folds if z is not None]
    if not folds_ok:
        print(f"  No data — skipping.")
        continue

    z_all, ids_all = pool_across_folds(folds_ok)
    scores_all, masks_all = get_quest_scores(ids_all)
    print(f"  n_test = {len(ids_all)}")

    # Per-scale RSA
    for key in SCALE_KEYS:
        mask = masks_all[key]
        if mask.sum() < 10:
            rsa_idrnn[uw][key] = (np.nan, 1.0)
            continue
        tgt = rdm_vec(scores_all[key][mask].reshape(-1, 1))
        r, p, _ = permutation_rsa(z_all[mask], tgt, n_perm=PERM_N)
        rsa_idrnn[uw][key] = (r, p)
        print(f"  {key:12s}: r={r:+.3f}  p_perm={p:.3f}")

    # Composite RSA
    full_mask = np.stack([masks_all[k] for k in SCALE_KEYS], axis=1).all(axis=1)
    if full_mask.sum() > 10:
        Y = np.column_stack([(scores_all[k][full_mask] - scores_all[k][full_mask].mean())
                              / (scores_all[k][full_mask].std() + 1e-8)
                              for k in SCALE_KEYS])
        tgt_composite = rdm_vec(Y)
        r_c, p_c, _ = permutation_rsa(z_all[full_mask], tgt_composite, n_perm=PERM_N)
        rsa_composite[uw] = (r_c, p_c)
        print(f"  COMPOSITE:   r={r_c:+.3f}  p_perm={p_c:.3f}")
    else:
        rsa_composite[uw] = (np.nan, 1.0)

    print()


# ── Plot 1: Per-scale RSA across unif_weight values ───────────────────────────
uw_labels = [str(uw) for uw in UW_LIST]
n_scales  = len(SCALE_KEYS)
x = np.arange(len(UW_LIST))
width = 0.12
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(11, 5))
for si, (key, label) in enumerate(zip(SCALE_KEYS, SCALE_LABELS)):
    rs   = [rsa_idrnn[uw].get(key, (np.nan, 1.0))[0] for uw in UW_LIST]
    ps   = [rsa_idrnn[uw].get(key, (np.nan, 1.0))[1] for uw in UW_LIST]
    bars = ax.bar(x + si * width - (n_scales - 1) * width / 2,
                  rs, width, label=label, color=cmap(si), alpha=0.85)
    for xi, (r, p) in enumerate(zip(rs, ps)):
        if not np.isnan(r) and p < 0.05:
            ax.text(x[xi] + si * width - (n_scales - 1) * width / 2,
                    r + (0.003 if r >= 0 else -0.012),
                    "*", ha="center", va="bottom" if r >= 0 else "top", fontsize=10)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"uw={v}" for v in UW_LIST])
ax.set_ylabel("Spearman r (latent RDM vs. score RDM)")
ax.set_title("RSA: IDRNN latent geometry vs. psychometric scores\n(outer-CV test participants; *=p_perm<0.05)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "rsa_unif_sweep_per_scale.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ── Plot 2: Composite RSA — IDRNN vs vanilla ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

rs_idrnn  = [rsa_composite.get(uw, (np.nan, 1.0))[0] for uw in UW_LIST]
ps_idrnn  = [rsa_composite.get(uw, (np.nan, 1.0))[1] for uw in UW_LIST]

bars = ax.bar(x, rs_idrnn, 0.5, color="steelblue", alpha=0.85, label="IDRNN")
for xi, (r, p) in enumerate(zip(rs_idrnn, ps_idrnn)):
    if not np.isnan(r) and p < 0.05:
        ax.text(xi, r + 0.003, "*", ha="center", va="bottom", fontsize=12)

if van_folds_ok:
    ax.axhline(r_van_c, color="gray", ls="--", lw=1.5, label=f"Vanilla (r={r_van_c:+.3f})")
    if p_van_c < 0.05:
        ax.text(len(UW_LIST) - 0.5, r_van_c + 0.003, "*", color="gray",
                ha="center", va="bottom", fontsize=10)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"uw={v}" for v in UW_LIST])
ax.set_ylabel("Spearman r (composite RDM)")
ax.set_title("RSA: composite psychometric geometry\n(6 scales standardised; *=p_perm<0.05)")
ax.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "rsa_unif_sweep_composite.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ── Plot 3: Summary heatmap (scale × unif_weight) ─────────────────────────────
r_matrix = np.array([[rsa_idrnn[uw].get(k, (np.nan, 1.0))[0] for uw in UW_LIST]
                      for k in SCALE_KEYS])  # (n_scales, n_uw)
p_matrix = np.array([[rsa_idrnn[uw].get(k, (np.nan, 1.0))[1] for uw in UW_LIST]
                      for k in SCALE_KEYS])

fig, ax = plt.subplots(figsize=(7, 5))
vmax = max(0.15, np.nanmax(np.abs(r_matrix)))
im = ax.imshow(r_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label="Spearman r")
ax.set_xticks(range(len(UW_LIST)));  ax.set_xticklabels([f"uw={v}" for v in UW_LIST])
ax.set_yticks(range(n_scales));      ax.set_yticklabels(SCALE_LABELS, fontsize=9)
ax.set_title("RSA Spearman r: latent vs. psychometric scale RDM\n(*=p_perm<0.05)")
for i in range(n_scales):
    for j in range(len(UW_LIST)):
        r = r_matrix[i, j]
        p = p_matrix[i, j]
        txt = f"{r:.2f}" + ("*" if p < 0.05 else "")
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color="white" if abs(r) > vmax * 0.6 else "black")
plt.tight_layout()
out = os.path.join(PLOT_DIR, "rsa_unif_sweep_heatmap.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

print("\nDone.")
