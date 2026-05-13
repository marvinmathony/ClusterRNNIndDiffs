#!/usr/bin/env python3
"""
Analyze the uniformity-loss sweep for Thalmann outer-CV pipeline.

For each unif_weight value:
  - Loads IDRNN encoder z across all seeds × folds
  - Decodes questionnaire scores via LOO-CV ridge regression
  - Compares NLL to vanilla

Produces:
  plots_thalmann/unif_sweep_decoding_{uw}.png   — one per unif_weight
  plots_thalmann/unif_sweep_nll.png             — NLL comparison across uw values
  plots_thalmann/unif_sweep_summary.png         — decoding r vs unif_weight per scale
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--unif_weights", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0])
parser.add_argument("--dgp", type=str, default="thalmann")
parser.add_argument("--n_folds", type=int, default=3)
args = parser.parse_args()

DGP      = args.dgp
N_FOLDS  = args.n_folds
UW_LIST  = args.unif_weights
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Questionnaire scales ───────────────────────────────────────────────────────
QUEST_PATH     = "data/finalQuestionnaireDataSession1.csv"
PANAS_PA_ITEMS = [0, 2, 4, 6, 8, 14, 16, 18]
PANAS_NA_ITEMS = [1, 3, 5, 7, 9, 11, 13, 15]
SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in PANAS_PA_ITEMS], "PANAS Positive Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in PANAS_NA_ITEMS], "PANAS Negative Affect"),
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

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

# ── Helpers ───────────────────────────────────────────────────────────────────
def loo_ridge(Z, y):
    """LOO-CV ridge regression; returns (r, p). Within-fold z-score."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        Z_tr, Z_te, y_tr = Z[tr], Z[te], y[tr]
        mu_z, sd_z = Z_tr.mean(0), Z_tr.std(0) + 1e-8
        mu_y, sd_y = y_tr.mean(), y_tr.std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z_tr - mu_z) / sd_z, (y_tr - mu_y) / sd_y)
        preds[te] = clf.predict((Z_te - mu_z) / sd_z) * sd_y + mu_y
    return pearsonr(y, preds)


def load_idrnn_z_for_fold(fold, run_suffix):
    """Return (z_train, z_test, subids_train, subids_test) for given fold + suffix."""
    base = f"runs_{DGP}_{run_suffix}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    seed_dirs = sorted([
        d for d in os.listdir(base)
        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))
    ])
    if not seed_dirs:
        return None

    subids_train = pd.read_csv(f"{data_dir}/df_train.csv")["subid"].values
    subids_test  = pd.read_csv(f"{data_dir}/df_test.csv")["subid"].values
    xin_train = torch.tensor(np.load(f"{data_dir}/xin_train.npy"), dtype=torch.float32)
    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)

    z_train_seeds, z_test_seeds = [], []
    for sd in seed_dirs:
        run_dir = os.path.join(base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc = cfg["model_config"]

        # Load encoder weights from step-2 checkpoint
        ckpt_epoch = cfg["cv_selected_epoch"]
        ckpt_path  = os.path.join(run_dir, "checkpoints", f"epoch{ckpt_epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]

        state = torch.load(ckpt_path, map_location="cpu")
        enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                    n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        enc.load_state_dict(enc_state)
        enc.eval()
        enc.set_task_ids(task_ids_global)

        with torch.no_grad():
            mu_tr, _ = enc(xin_train)   # (B, Bk, T, z) if per_timestep
            mu_te, _ = enc(xin_test)

        # Last block (restless), last timestep
        if mu_tr.dim() == 4:
            mu_tr = mu_tr[:, -1, -1, :]
            mu_te = mu_te[:, -1, -1, :]

        z_train_seeds.append(mu_tr.numpy())
        z_test_seeds.append(mu_te.numpy())

    if not z_train_seeds:
        return None

    # Average z across seeds
    z_train = np.stack(z_train_seeds, axis=0).mean(axis=0)  # (B_train, z_dim)
    z_test  = np.stack(z_test_seeds,  axis=0).mean(axis=0)  # (B_test, z_dim)
    return z_train, z_test, subids_train, subids_test


def load_vanilla_z_for_fold(fold):
    """Average GRU hidden state across blocks for vanilla model."""
    base     = f"runs_vanilla_{DGP}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    seed_dirs = sorted([
        d for d in os.listdir(base)
        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))
    ])
    if not seed_dirs:
        return None

    subids_train = pd.read_csv(f"{data_dir}/df_train.csv")["subid"].values
    subids_test  = pd.read_csv(f"{data_dir}/df_test.csv")["subid"].values
    xin_train = torch.tensor(np.load(f"{data_dir}/xin_train.npy"), dtype=torch.float32)
    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)

    z_train_seeds, z_test_seeds = [], []
    for sd in seed_dirs:
        run_dir = os.path.join(base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc = cfg["model_config"]

        ckpt_epoch = cfg["cv_selected_epoch"]
        ckpt_path  = os.path.join(run_dir, "checkpoints", f"epoch{ckpt_epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts: continue
            ckpt_path = ckpts[-1]

        van = AblatedRNN(hid=mc["hidden"],
                         in_dim=mc.get("dec_in_dim", mc["in_dim"]),
                         A=mc["A"], block_structure=True,
                         n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0))
        van.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        van.eval()
        if hasattr(van, "set_task_ids"):
            van.set_task_ids(task_ids_global)

        with torch.no_grad():
            _, hid_tr, _ = van(xin_train)   # hid: (Bk, B, hidden)
            _, hid_te, _ = van(xin_test)

        # Mean across blocks → (B, hidden)
        z_train_seeds.append(hid_tr.permute(1, 0, 2).mean(1).numpy())
        z_test_seeds.append(hid_te.permute(1, 0, 2).mean(1).numpy())

    if not z_train_seeds:
        return None
    return (np.stack(z_train_seeds).mean(0), np.stack(z_test_seeds).mean(0),
            subids_train, subids_test)


def pool_z_across_folds(fold_results):
    """Concatenate test-set z and subids across folds."""
    z_all, y_all = [], {k: [] for k in SCALE_KEYS}
    for z_tr, z_te, ids_tr, ids_te in fold_results:
        z_all.append(z_te)
        for key in SCALE_KEYS:
            vals = quest.reindex(ids_te)[key].values.astype(float)
            y_all[key].append(vals)
    return np.concatenate(z_all), {k: np.concatenate(v) for k, v in y_all.items()}


# ── Collect results ────────────────────────────────────────────────────────────
print("Loading results...")

results = {}   # uw → {scale → (r, p)}
nll_dict = {}  # uw → mean cv_val_nll across seeds×folds

for uw in UW_LIST:
    suffix = f"unif{str(uw).replace('.', 'p')}"
    print(f"\nunif_weight={uw}  (suffix={suffix})")
    fold_results = []
    cv_nlls = []

    for fold in range(N_FOLDS):
        base = f"runs_{DGP}_{suffix}/fold{fold}"
        if not os.path.isdir(base):
            print(f"  Missing: {base}")
            continue

        res = load_idrnn_z_for_fold(fold, suffix)
        if res is None:
            print(f"  No valid runs in fold {fold}")
            continue
        fold_results.append(res)

        # Collect NLL values from config.json of all seeds
        for sd in os.listdir(base):
            cfg_p = os.path.join(base, sd, "config.json")
            if os.path.exists(cfg_p):
                with open(cfg_p) as f:
                    cfg = json.load(f)
                nll = cfg.get("cv_val_nll", cfg.get("cv_val_loss"))
                if nll is not None:
                    cv_nlls.append(nll)

    if not fold_results:
        print(f"  No data for uw={uw}, skipping.")
        continue

    z_pool, y_pool = pool_z_across_folds(fold_results)
    scale_results = {}
    for key in SCALE_KEYS:
        y = y_pool[key]
        mask = ~np.isnan(y)
        if mask.sum() > 20:
            r, p = loo_ridge(z_pool[mask], y[mask])
            scale_results[key] = (r, p)
            print(f"  {key:12s}  r={r:+.3f}  p={p:.3f}")
        else:
            scale_results[key] = (np.nan, np.nan)

    results[uw] = scale_results
    nll_dict[uw] = np.mean(cv_nlls) if cv_nlls else np.nan

# ── Vanilla baseline ───────────────────────────────────────────────────────────
print("\nLoading vanilla...")
van_fold_results = []
van_nlls = []

for fold in range(N_FOLDS):
    base = f"runs_vanilla_{DGP}/fold{fold}"
    if not os.path.isdir(base):
        continue
    res = load_vanilla_z_for_fold(fold)
    if res:
        van_fold_results.append(res)
    for sd in os.listdir(base):
        cfg_p = os.path.join(base, sd, "config.json")
        if os.path.exists(cfg_p):
            with open(cfg_p) as f:
                cfg = json.load(f)
            nll = cfg.get("cv_val_loss")
            if nll is not None:
                van_nlls.append(nll)

van_results = {}
if van_fold_results:
    z_van, y_van = pool_z_across_folds(van_fold_results)
    for key in SCALE_KEYS:
        y = y_van[key]
        mask = ~np.isnan(y)
        if mask.sum() > 20:
            r, p = loo_ridge(z_van[mask], y[mask])
            van_results[key] = (r, p)
            print(f"  Vanilla {key:12s}  r={r:+.3f}  p={p:.3f}")
van_nll = np.mean(van_nlls) if van_nlls else np.nan

# ── Plot 1: Per-unif_weight decoding bar charts ─────────────────────────────
palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
uw_colors = {uw: palette[i % len(palette)] for i, uw in enumerate(UW_LIST)}

for uw in UW_LIST:
    if uw not in results:
        continue
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SCALE_KEYS))
    width = 0.35

    r_idrnn = [abs(results[uw][k][0]) if not np.isnan(results[uw][k][0]) else 0 for k in SCALE_KEYS]
    p_idrnn = [results[uw][k][1] for k in SCALE_KEYS]
    r_van   = [abs(van_results.get(k, (np.nan, np.nan))[0]) for k in SCALE_KEYS]
    p_van   = [van_results.get(k, (np.nan, np.nan))[1] for k in SCALE_KEYS]

    bars_i = ax.bar(x - width/2, r_idrnn, width, label=f"IDRNN (uw={uw})",
                    color=uw_colors[uw], alpha=0.85, edgecolor="white")
    bars_v = ax.bar(x + width/2, r_van, width, label="Vanilla",
                    color="#BBBBBB", alpha=0.85, edgecolor="white")

    # Significance markers
    for i, (r, p) in enumerate(zip(r_idrnn, p_idrnn)):
        if not np.isnan(p) and p < 0.05:
            ax.text(i - width/2, r + 0.01, "*" if p < 0.05 else "", ha="center", va="bottom", fontsize=12)
    for i, (r, p) in enumerate(zip(r_van, p_van)):
        if not np.isnan(p) and p < 0.05:
            ax.text(i + width/2, r + 0.01, "*", ha="center", va="bottom", fontsize=12, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|Pearson r|  (LOO-CV ridge)  ↑ better")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(f"Thalmann: Decoding questionnaire scores\nIDRNN (unif_weight={uw}) vs Vanilla",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"unif_sweep_decoding_uw{str(uw).replace('.','p')}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

# ── Plot 2: Summary — |r| vs unif_weight per scale ──────────────────────────
uw_vals = sorted([uw for uw in UW_LIST if uw in results])
fig, ax = plt.subplots(figsize=(9, 5))
for i, key in enumerate(SCALE_KEYS):
    r_per_uw = [abs(results[uw][key][0]) if uw in results else np.nan for uw in uw_vals]
    ax.plot(uw_vals, r_per_uw, marker="o", label=SCALES[key][1],
            color=palette[i % len(palette)], linewidth=1.8, markersize=7)
    # Vanilla as dashed horizontal
    r_v = abs(van_results.get(key, (np.nan,))[0])
    if not np.isnan(r_v):
        ax.axhline(r_v, color=palette[i % len(palette)], linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xlabel("Uniformity loss weight")
ax.set_ylabel("|Pearson r|  (LOO-CV ridge)  ↑ better")
ax.set_title("Decoding |r| vs uniformity weight\n(dashed = vanilla baseline)", fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "unif_sweep_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 3: NLL comparison ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
nll_vals = [nll_dict.get(uw, np.nan) for uw in uw_vals]
ax.plot(uw_vals, nll_vals, marker="o", color="#4477AA",
        linewidth=2, markersize=8, label="IDRNN cv_val_nll")
ax.axhline(van_nll, color="#EE6677", linestyle="--", linewidth=2, label=f"Vanilla NLL ({van_nll:.3f})")
ax.set_xlabel("Uniformity loss weight")
ax.set_ylabel("Mean CV val NLL (cross-entropy)  ↓ better")
ax.set_title("NLL fit vs uniformity weight", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "unif_sweep_nll.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Text summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'':15s}" + "".join(f"{k:>12s}" for k in SCALE_KEYS))
for uw in uw_vals:
    row = f"uw={uw:<10}"
    for k in SCALE_KEYS:
        r, p = results[uw].get(k, (np.nan, np.nan))
        sig = "*" if not np.isnan(p) and p < 0.05 else " "
        row += f"  {r:+.3f}{sig}  "
    print(row)
row = f"{'Vanilla':<15}"
for k in SCALE_KEYS:
    r, p = van_results.get(k, (np.nan, np.nan))
    sig = "*" if not np.isnan(p) and p < 0.05 else " "
    row += f"  {r:+.3f}{sig}  "
print(row)
print("\nNLL (cv_val_nll):")
for uw in uw_vals:
    print(f"  uw={uw}: {nll_dict.get(uw, np.nan):.4f}")
print(f"  Vanilla: {van_nll:.4f}")
