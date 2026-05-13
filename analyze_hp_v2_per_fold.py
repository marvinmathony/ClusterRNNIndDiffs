#!/usr/bin/env python3
"""
Per-fold decoding analysis for HP search v2.

Fixes two bugs in analyze_hp_v2_outer_cv.py:
  1. Regression now runs within each fold (not on pooled z across folds),
     eliminating the between-fold mean-offset artifact.
  2. Correct encoder timestep: block 0, trial 9 (last real trial of task-0),
     not mu[:, -1, -1, :] which lands in padding.

For each HP combo:
  - Loads IDRNN encoder from runs_thalmann_hp_v2_{combo}/fold{k}/seed_*/
  - Extracts z at (blk=0, t=9), averaged over seeds
  - Runs LOO-CV ridge within each fold (z-scored inside the LOO loop)
  - Aggregates r across folds: mean ± std

NLL: per-seed cv_val_nll from outer-CV config.json, averaged over seeds × folds.

Outputs: plots_thalmann/hp_v2/
  summary_table_per_fold.csv
  nll_vs_decoding_scatter.png
  top_combos_bar.png
  z_collapse_grid.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from joblib import Parallel, delayed

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN

# ── Config ─────────────────────────────────────────────────────────────────────
RUNS_BASE = "runs_thalmann"
PLOT_DIR  = "plots_thalmann/hp_v2"
DGP       = "thalmann"
N_FOLDS   = 3
BLK_USE   = 0
T_USE     = 9    # last real trial of task-0 blocks

os.makedirs(PLOT_DIR, exist_ok=True)

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS Pos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS Neg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA Anxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9 Depression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI Curiosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5 Openness"),
}
SCALE_KEYS   = list(SCALES.keys())
SCALE_LABELS = [SCALES[k][1] for k in SCALE_KEYS]

quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
for k, (items, _) in SCALES.items():
    quest[k] = quest[items].mean(axis=1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def loo_ridge_within_fold(Z, y):
    """LOO-CV ridge, z-scored strictly within train split. Returns (r, p)."""
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z = Z[tr].mean(0); sd_z = Z[tr].std(0) + 1e-8
        mu_y = y[tr].mean();  sd_y = y[tr].std() + 1e-8
        clf  = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    return pearsonr(y, preds)


def load_z_for_fold(combo, fold):
    """
    Load seed-averaged encoder z at (blk=0, t=9) for test subjects of fold.
    Returns (z_test, subids) or (None, None).
    """
    run_base = f"{RUNS_BASE}_hp_v2_{combo}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"
    if not os.path.isdir(run_base):
        return None, None

    xin_test = torch.tensor(
        np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32
    )
    df_test  = pd.read_csv(f"{data_dir}/df_test.csv")
    subids   = df_test["subid"].values if "subid" in df_test.columns \
               else df_test["session"].values

    z_seeds, nlls = [], []
    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue
        if "cv_val_nll" in cfg:
            nlls.append(cfg["cv_val_nll"])

        ckpt = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt):
            continue
        try:
            state = torch.load(ckpt, map_location="cpu")
            enc   = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                          hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                          task_emb_dim=mc["task_emb_dim"])
            enc.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
            enc.eval(); enc.set_task_ids(task_ids_global)
            with torch.no_grad():
                mu, _ = enc(xin_test)          # (B, 31, 200, z_dim)
            z_seeds.append(mu[:, BLK_USE, T_USE, :].numpy())
        except Exception as e:
            print(f"  Warning ({combo} fold{fold} {sd}): {e}")

    if not z_seeds:
        return None, None

    return np.stack(z_seeds).mean(0), subids   # (B_test, z_dim)


def parse_combo(combo):
    parts = combo.split("_")
    try:
        uw   = {"00": 0.0, "01": 0.1, "05": 0.5}.get(parts[0][2:], float("nan"))
        lmbd = {"005": 0.05, "01": 0.1, "02": 0.2}.get(parts[1][4:], float("nan"))
        eh   = int(parts[2][2:]); h = int(parts[3][1:]); z = int(parts[4][1:])
        return uw, lmbd, eh, h, z
    except Exception:
        return None, None, None, None, None


def process_combo(combo):
    uw, lmbd, eh, h, z_dim = parse_combo(combo)
    if uw is None:
        return None

    fold_rs   = {k: [] for k in SCALE_KEYS}
    fold_nlls = []
    fold_zstd = []

    for fold in range(N_FOLDS):
        z_test, subids = load_z_for_fold(combo, fold)
        if z_test is None:
            continue

        # NLL from configs
        run_base = f"{RUNS_BASE}_hp_v2_{combo}/fold{fold}"
        for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
            cfg_p = os.path.join(run_base, sd, "config.json")
            if os.path.exists(cfg_p):
                with open(cfg_p) as f: cfg = json.load(f)
                if "cv_val_nll" in cfg:
                    fold_nlls.append(cfg["cv_val_nll"])

        fold_zstd.append(float(z_test.std(0).mean()))

        for scale in SCALE_KEYS:
            y = quest.reindex(subids)[scale].values.astype(float)
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue
            r, _ = loo_ridge_within_fold(z_test[mask], y[mask])
            fold_rs[scale].append(r)

    if not fold_nlls:
        return None

    row = {
        "combo": combo, "uw": uw, "lmbd": lmbd,
        "enc_hidden": eh, "hidden": h, "z_dim": z_dim,
        "cv_val_nll": np.mean(fold_nlls),
        "z_std_mean": np.mean(fold_zstd),
        "n_folds_complete": len(fold_zstd),
    }
    for scale in SCALE_KEYS:
        rs = fold_rs[scale]
        row[f"r_{scale}"]      = np.mean(rs) if rs else np.nan
        row[f"r_std_{scale}"]  = np.std(rs)  if rs else np.nan
    row["mean_abs_r"] = np.nanmean([abs(row[f"r_{k}"]) for k in SCALE_KEYS])
    return row


# ── Discover combos ────────────────────────────────────────────────────────────
all_combo_dirs = sorted(
    d for d in os.listdir(".")
    if d.startswith(f"{RUNS_BASE}_hp_v2_uw") and os.path.isdir(d)
)
combos = [d[len(f"{RUNS_BASE}_hp_v2_"):] for d in all_combo_dirs
          if not d.startswith(f"{RUNS_BASE}_hp_v2_uw05_lmbd0")]

# Include all uw/lmbd combos
combos = [d[len(f"{RUNS_BASE}_hp_v2_"):] for d in all_combo_dirs]
combos = sorted(set(combos))
print(f"Found {len(combos)} combos")

# ── Process in parallel ────────────────────────────────────────────────────────
results = Parallel(n_jobs=8, verbose=5)(
    delayed(process_combo)(c) for c in combos
)
results = [r for r in results if r is not None]
df = pd.DataFrame(results).sort_values("cv_val_nll")
df.to_csv(os.path.join(PLOT_DIR, "summary_table_per_fold.csv"), index=False)
print(f"\nProcessed {len(df)} combos. Saved summary_table_per_fold.csv")
print(df[["combo","cv_val_nll","z_std_mean",
          "r_PHQ","r_std_PHQ","mean_abs_r","n_folds_complete"]].head(20).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: NLL vs PHQ decoding r
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(df["cv_val_nll"], df["r_PHQ"].abs(),
                c=df["z_dim"], cmap="viridis", s=50, alpha=0.75,
                edgecolors="grey", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="z_dim")
# Annotate top combos
top = df.nlargest(5, "mean_abs_r")
for _, row in top.iterrows():
    ax.annotate(row["combo"].replace("_", "\n"), (row["cv_val_nll"], abs(row["r_PHQ"])),
                fontsize=6, alpha=0.8)
ax.set_xlabel("Mean cv_val_nll (outer-CV, lower = better)")
ax.set_ylabel("|r| PHQ decoding (per-fold LOO-CV)")
ax.set_title("NLL vs PHQ decoding — per-fold regression\n(bubble = z_dim)",
             fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "nll_vs_decoding_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Top 15 combos — bar chart of mean |r| across scales
# ══════════════════════════════════════════════════════════════════════════════
top15 = df.nlargest(15, "mean_abs_r").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top15))
colors = plt.cm.tab10(np.linspace(0, 1, len(SCALE_KEYS)))
w = 0.8 / len(SCALE_KEYS)
for j, (sk, col) in enumerate(zip(SCALE_KEYS, colors)):
    ax.bar(x + j*w - 0.4 + w/2,
           top15[f"r_{sk}"].abs(),
           w, color=col, alpha=0.8, label=SCALES[sk][1])
ax.set_xticks(x)
ax.set_xticklabels([r["combo"].replace("_", "\n") for _, r in top15.iterrows()],
                   fontsize=7, rotation=0)
ax.set_ylabel("|r| per-fold LOO-CV")
ax.set_ylim(0, 1)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_title("Top 15 combos by mean |r| — per-fold regression", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "top_combos_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: z_std vs NLL grid (collapse diagnostic)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
for zdim, grp in df.groupby("z_dim"):
    ax.scatter(grp["cv_val_nll"], grp["z_std_mean"],
               label=f"z={zdim}", s=40, alpha=0.7)
ax.set_xlabel("cv_val_nll"); ax.set_ylabel("Mean z_std (collapse diagnostic)")
ax.set_title("z_std vs NLL — all combos\n(low z_std = posterior collapse)", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "z_collapse_grid.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Full heatmap — all scales × top 20 combos
# ══════════════════════════════════════════════════════════════════════════════
top20 = df.nlargest(20, "mean_abs_r").reset_index(drop=True)
r_mat = top20[[f"r_{k}" for k in SCALE_KEYS]].values   # (20, 6)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(r_mat, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
plt.colorbar(im, ax=ax, label="Pearson r (mean over folds)")
ax.set_xticks(range(len(SCALE_KEYS))); ax.set_xticklabels(SCALE_LABELS, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(top20))); ax.set_yticklabels(
    [f"{r['combo']}  NLL={r['cv_val_nll']:.3f}" for _, r in top20.iterrows()], fontsize=7)
for i in range(len(top20)):
    for j, sk in enumerate(SCALE_KEYS):
        r_val = r_mat[i, j]
        std   = top20.iloc[i][f"r_std_{sk}"]
        ax.text(j, i, f"{r_val:.2f}\n±{std:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(r_val) > 0.4 else "black")
ax.set_title("Per-fold decoding r — top 20 combos by mean |r|\n(r ± std across 3 folds)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "top20_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nAll plots saved to", PLOT_DIR)
