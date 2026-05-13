#!/usr/bin/env python3
"""
Combined analysis script for CV-trained IDRNN and Vanilla models (Sloutsky data).

Reads pre-saved latents from  data_sloutsky/seed_{seed}/latents_tensor{nametag}.pt
(written by testing_script.py) and computes per seed:

  1. RSA       : Spearman r between latent RDM and behavioural RDMs
                 (age, w_novelty, w_value, w_uncert, theta, cog_nll)
  2. Cog-weight: Pearson r between PC1 of latents and each EM-fitted parameter
  3. Decoding  : LOOCV Ridge regression of each cog param from full latent

Results are averaged across seeds, plotted with mean ± SEM + seed scatter,
and saved to plots_sloutsky/.

Usage:
    python analyze_cv_seeds.py
    python analyze_cv_seeds.py --seeds 12,50,76
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, ttest_rel, sem as scipy_sem, linregress
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sloutsky_cog_model import unpack_params

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds to restrict analysis, e.g. '12,50,76'")
args = parser.parse_args()
FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)

# ── Configuration ─────────────────────────────────────────────────────────────
DGP       = "sloutsky"
DATA_DIR  = f"data_{DGP}"
IDRNN_DIR = f"runs_{DGP}"
VAN_DIR   = f"runs_vanilla_{DGP}"
PLOT_DIR  = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

_seed_tag = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
             if FILTER_SEEDS else "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Participant metadata ───────────────────────────────────────────────────────
df_test   = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique = df_test.drop_duplicates("subid").sort_values("subid").reset_index(drop=True)
test_subids = sorted(df_test["subid"].unique())
B_test = len(test_subids)

# Binary age label
y_age = np.where(df_unique["age"] == "young_child", 0, 1)

# Age group info for scatter plots
GROUP_ORDER  = ["young_child", "old_child", "adult"]
GROUP_LABELS = ["Young children", "Older children", "Adults"]
GROUP_COLORS = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
age_labels   = np.array([df_unique.loc[df_unique["subid"] == s, "age"].values[0]
                          for s in test_subids])

# ── Cognitive model parameters ────────────────────────────────────────────────
em = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all_test        = em["h_all_test"]
participants_test = em["participants_test"]
assert list(participants_test) == test_subids, \
    "EM participants_test order doesn't match test_subids"

_cog          = [unpack_params(h_all_test[i]) for i in range(B_test)]
theta_vals    = np.array([p["theta"]             for p in _cog])
w_value_vals  = np.array([p["b_value_train"]     for p in _cog])
w_uncert_vals = np.array([p["b_uncertain_train"] for p in _cog])
w_lag_vals    = np.array([p["b_lag_train"]       for p in _cog])
w_novel_vals  = np.array([p["b_novelty_train"]   for p in _cog])

df_cog   = pd.read_csv(f"{DATA_DIR}/cog_model_results.csv")
nll_vals = df_cog.sort_values("session")["normalized_likelihood"].values
assert len(nll_vals) == B_test

# ── Target RDMs for RSA ───────────────────────────────────────────────────────
def make_rdm_vec(values):
    return pdist(values.reshape(-1, 1), metric="euclidean")

targets = {
    "age":      make_rdm_vec(y_age),
    "w_novelty":make_rdm_vec(w_novel_vals),
    "w_value":  make_rdm_vec(w_value_vals),
    "w_uncert": make_rdm_vec(w_uncert_vals),
    "theta":    make_rdm_vec(theta_vals),
    "cog_nll":  make_rdm_vec(nll_vals),
}

cog_vars = {
    "theta":    theta_vals,
    "w_value":  w_value_vals,
    "w_uncert": w_uncert_vals,
    "w_lag":    w_lag_vals,
    "w_novelty":w_novel_vals,
}

COG_LABELS = {
    "theta":    "θ",
    "w_value":  "w value",
    "w_uncert": "w uncert",
    "w_lag":    "w lag",
    "w_novelty":"w novelty",
}

# ── LOOCV Ridge regression ────────────────────────────────────────────────────
def ridge_loocv(X, y_raw):
    """Returns (r, y_pred_full) where y_pred_full has nan for excluded rows."""
    X     = np.asarray(X, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    finite = np.isfinite(y_raw) & np.all(np.isfinite(X), axis=1)
    if finite.sum() < 4:
        return float("nan"), np.full(len(y_raw), np.nan)

    X_f, y_f = X[finite], y_raw[finite]
    alphas = np.logspace(-3, 3, 20)
    loo    = LeaveOneOut()
    y_pred = np.empty_like(y_f)

    for tr, te in loo.split(X_f):
        pipe = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, cv=min(5, len(tr)))
        )
        pipe.fit(X_f[tr], y_f[tr])
        y_pred[te] = pipe.predict(X_f[te])

    r, _ = pearsonr(y_f, y_pred)
    y_pred_full = np.full(len(y_raw), np.nan)
    y_pred_full[finite] = y_pred
    return float(r), y_pred_full


# ── Per-model analysis loop ───────────────────────────────────────────────────
def run_analysis(base_dir, nametag, label):
    seeds = sorted([
        int(d.split("_")[1])
        for d in os.listdir(base_dir)
        if d.startswith("seed_")
        and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
    ])
    print(f"\n{'='*70}")
    print(f"{label}  |  seeds: {seeds}")
    print(f"{'='*70}")

    rsa_per_seed      = {t: [] for t in targets}
    cogcorr_per_seed  = {v: [] for v in cog_vars}
    decode_per_seed   = {v: [] for v in cog_vars}
    novelty_pred      = {}
    lat_per_seed      = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        lat_path = os.path.join(DATA_DIR, f"seed_{seed}", f"latents_tensor{nametag}.pt")
        if not os.path.exists(lat_path):
            print(f"  No latent file found at {lat_path}, skipping.")
            continue

        lat = torch.load(lat_path, map_location="cpu").numpy()
        if lat.ndim > 2:
            # Flatten extra dims (B, 1, z_dim) → (B, z_dim)
            lat = lat.reshape(lat.shape[0], -1)
        lat_per_seed[seed] = lat

        # 1. RSA
        rdm = pdist(lat, metric="euclidean")
        for tname, trdm in targets.items():
            r, _ = spearmanr(rdm, trdm)
            rsa_per_seed[tname].append(r)
        print("  RSA: " + "  ".join(f"{t}={rsa_per_seed[t][-1]:+.3f}" for t in targets))

        # 2. Cog-weight Pearson r with PC1
        n_comp = min(2, lat.shape[1])
        pc1    = PCA(n_components=n_comp).fit_transform(lat)[:, 0]
        for vname, vals in cog_vars.items():
            finite = np.isfinite(pc1) & np.isfinite(vals)
            r = float(pearsonr(pc1[finite], vals[finite])[0]) if finite.sum() > 2 else float("nan")
            cogcorr_per_seed[vname].append(r)
        print("  Cog r(PC1): " + "  ".join(
            f"{v}={cogcorr_per_seed[v][-1]:+.3f}" for v in cog_vars))

        # 3. Decoding (LOOCV Ridge)
        for vname, vals in cog_vars.items():
            r, y_pred = ridge_loocv(lat, vals)
            decode_per_seed[vname].append(r)
            if vname == "w_novelty":
                novelty_pred[seed] = y_pred
        print("  Decoding r: " + "  ".join(
            f"{v}={decode_per_seed[v][-1]:+.3f}" for v in cog_vars))

    return (
        {t: np.array(v) for t, v in rsa_per_seed.items()},
        {v: np.array(r) for v, r in cogcorr_per_seed.items()},
        {v: np.array(r) for v, r in decode_per_seed.items()},
        novelty_pred,
        seeds,
        lat_per_seed,
    )


rsa_idrnn, cogcorr_idrnn, decode_idrnn, nov_pred_idrnn, seeds_idrnn, lats_idrnn = \
    run_analysis(IDRNN_DIR, "latentmodel", "IDRNN")
rsa_van,   cogcorr_van,   decode_van,   nov_pred_van,   seeds_van,   lats_van   = \
    run_analysis(VAN_DIR,   "vanilla",    "Vanilla")


# ── Shared plot helpers ───────────────────────────────────────────────────────
def scatter_seeds(ax, xi, vals, rng, jitter=0.05):
    jit = rng.uniform(-jitter, jitter, size=len(vals))
    ax.scatter(xi + jit, vals, color="k", s=18, zorder=5, alpha=0.7)


def grouped_bar_plot(ax, keys, idrnn_dict, van_dict, key_labels,
                     ylabel, title, primary_keys=()):
    x   = np.arange(len(keys))
    w   = 0.35
    rng = np.random.default_rng(0)

    for i, k in enumerate(keys):
        iv = idrnn_dict[k]
        vv = van_dict[k]
        alpha = 0.95 if k in primary_keys else 0.5

        mi  = float(np.nanmean(iv)) if len(iv) else 0
        sei = float(np.nanstd(iv) / np.sqrt(np.sum(~np.isnan(iv)))) if len(iv) > 1 else 0
        mv  = float(np.nanmean(vv)) if len(vv) else 0
        sev = float(np.nanstd(vv) / np.sqrt(np.sum(~np.isnan(vv)))) if len(vv) > 1 else 0

        ax.bar(i - w/2, mi, w, color="#4C72B0", alpha=alpha,
               label="IDRNN"   if i == 0 else "_")
        ax.bar(i + w/2, mv, w, color="#DD8452", alpha=alpha,
               label="Vanilla" if i == 0 else "_")
        ax.errorbar([i - w/2, i + w/2], [mi, mv], yerr=[sei, sev],
                    fmt="none", color="k", capsize=4, lw=1.2)
        scatter_seeds(ax, i - w/2, iv, rng)
        scatter_seeds(ax, i + w/2, vv, rng)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([key_labels.get(k, k) for k in keys])
    for lbl, k in zip(ax.get_xticklabels(), keys):
        if k in primary_keys:
            lbl.set_fontweight("bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


target_labels = {
    "age":       "Age",
    "w_novelty": "w novelty",
    "w_value":   "w value",
    "w_uncert":  "w uncert",
    "theta":     "θ",
    "cog_nll":   "Cog NLL",
}

# ── Plot 1: RSA ───────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 4))
grouped_bar_plot(
    ax1, list(targets.keys()), rsa_idrnn, rsa_van, target_labels,
    ylabel="Spearman r (RSA)",
    title="RSA: latent RDM vs behavioural RDMs\n(CV epoch, mean ± SEM)",
    primary_keys={"age", "w_novelty"},
)
fig1.tight_layout()
rsa_path = os.path.join(PLOT_DIR, f"rsa_cv_seeds{_seed_tag}.png")
fig1.savefig(rsa_path, dpi=150)
print(f"\nRSA plot saved → {rsa_path}")
plt.close(fig1)

# ── Plot 2: Cog-weight Pearson r with PC1 ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
grouped_bar_plot(
    ax2, list(cog_vars.keys()), cogcorr_idrnn, cogcorr_van, COG_LABELS,
    ylabel="Pearson r  (latent PC1 vs cog param)",
    title="Cog weights vs latent PC1\n(CV epoch, mean ± SEM)",
    primary_keys={"w_novelty"},
)
fig2.tight_layout()
cog_path = os.path.join(PLOT_DIR, f"cogcorr_cv_seeds{_seed_tag}.png")
fig2.savefig(cog_path, dpi=150)
print(f"Cog-weight correlation plot saved → {cog_path}")
plt.close(fig2)

# ── Plot 3: Decoding (LOOCV Ridge) ────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 4))
grouped_bar_plot(
    ax3, list(cog_vars.keys()), decode_idrnn, decode_van, COG_LABELS,
    ylabel="Pearson r  (LOOCV Ridge, predicted vs true)",
    title="Decoding cognitive parameters from latents\n(CV epoch, mean ± SEM)",
    primary_keys={"w_novelty"},
)
fig3.tight_layout()
dec_path = os.path.join(PLOT_DIR, f"decode_cv_seeds{_seed_tag}.png")
fig3.savefig(dec_path, dpi=150)
print(f"Decoding plot saved → {dec_path}")
plt.close(fig3)

# ── Plot 4: w_novelty scatter per seed (predicted vs true) ───────────────────
true_nov = cog_vars["w_novelty"]
n_idrnn  = len([s for s in seeds_idrnn if s in nov_pred_idrnn])
n_van    = len([s for s in seeds_van   if s in nov_pred_van])
n_cols   = max(n_idrnn, n_van, 1)
n_rows   = 1 if (n_idrnn == 0 or n_van == 0) else 2

fig4, axes4 = plt.subplots(n_rows, n_cols,
                            figsize=(3.5 * n_cols, 3.5 * n_rows),
                            squeeze=False)
fig4.suptitle("w_novelty: LOOCV predicted vs true (per seed)", fontsize=11)


def scatter_novelty(ax, seed, label, pred, color):
    finite = np.isfinite(true_nov) & np.isfinite(pred)
    if finite.sum() < 3:
        ax.set_visible(False)
        return
    for grp, gcol, glbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
        mask = (age_labels == grp) & finite
        ax.scatter(true_nov[mask], pred[mask],
                   color=gcol, s=35, alpha=0.8, edgecolors="none", label=glbl)
    coef = np.polyfit(true_nov[finite], pred[finite], 1)
    xr   = np.linspace(true_nov[finite].min(), true_nov[finite].max(), 100)
    ax.plot(xr, np.poly1d(coef)(xr), "--", color=color, lw=1.5)
    r, _ = pearsonr(true_nov[finite], pred[finite])
    ax.set_title(f"{label} seed {seed}\nr = {r:.3f}", fontsize=9)
    ax.set_xlabel("true w_novelty", fontsize=8)
    ax.set_ylabel("predicted", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


row_idrnn = 0
row_van   = 1 if n_rows == 2 else 0
for col_i, seed in enumerate(s for s in seeds_idrnn if s in nov_pred_idrnn):
    scatter_novelty(axes4[row_idrnn, col_i], seed, "IDRNN", nov_pred_idrnn[seed], "#4C72B0")
    if col_i == 0:
        axes4[row_idrnn, col_i].legend(fontsize=7, frameon=False)
for col_i, seed in enumerate(s for s in seeds_van if s in nov_pred_van):
    scatter_novelty(axes4[row_van, col_i], seed, "Vanilla", nov_pred_van[seed], "#DD8452")
for row in range(n_rows):
    for col in range(n_cols):
        if not axes4[row, col].collections and not axes4[row, col].lines:
            axes4[row, col].set_visible(False)

fig4.tight_layout()
scat_path = os.path.join(PLOT_DIR, f"novelty_scatter_cv_seeds{_seed_tag}.png")
fig4.savefig(scat_path, dpi=150)
print(f"Novelty scatter saved → {scat_path}")
plt.close(fig4)

# ── Plot 5: Likelihood comparison (IDRNN vs Vanilla vs Cog Model) ─────────────
def load_nll_across_seeds(base_dir, nametag):
    """
    Load per-participant NLL CSVs from all seed directories and return
    the mean NLL per participant (averaged across seeds).

    Returns (mean_nll_per_participant, n_seeds_used) or (None, 0) if no data.
    """
    seed_dirs = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith("seed_") and os.path.isdir(os.path.join(base_dir, d))
        and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
    ])
    all_nlls = []
    for seed_dir_name in seed_dirs:
        seed = int(seed_dir_name.split("_")[1])
        csv_path = os.path.join(DATA_DIR, f"seed_{seed}", f"rnn_results{nametag}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "normalized_likelihood" not in df.columns:
            continue
        all_nlls.append(df["normalized_likelihood"].values)

    if not all_nlls:
        return None, 0
    # Stack (n_seeds, n_participants) and average across seeds
    stacked = np.stack(all_nlls, axis=0)
    return stacked.mean(axis=0), len(all_nlls)


def _add_sig_bar(ax, x1, x2, y, h, p_val):
    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x1 + x2) / 2, y + h, sig_str, ha="center", va="bottom", fontsize=10)


def _load_csv_nll(path, sort_col="session"):
    """Load normalized_likelihood from a CSV; return array or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "normalized_likelihood" not in df.columns:
        return None
    if sort_col in df.columns:
        df = df.sort_values(sort_col)
    return df["normalized_likelihood"].values


def plot_likelihood_comparison():
    # ── Load RNN models (seed-averaged) ──────────────────────────────────────
    idrnn_ll,  n_idrnn = load_nll_across_seeds(IDRNN_DIR, "latentmodel")
    van_ll,    n_van   = load_nll_across_seeds(VAN_DIR,   "vanilla")
    rnn_cp_ll, n_cp    = load_nll_across_seeds(IDRNN_DIR, "_common_process")

    # ── Load cognitive models ─────────────────────────────────────────────────
    cog_em_ll = _load_csv_nll(os.path.join(DATA_DIR, "cog_model_results.csv"))
    cog_cp_ll = _load_csv_nll(os.path.join(DATA_DIR, "cog_model_cp_results.csv"))

    # ── Load ill-specified models ─────────────────────────────────────────────
    ill_em_ll = _load_csv_nll(os.path.join(DATA_DIR, "ill_specified_map_results.csv"))
    ill_cp_ll = _load_csv_nll(os.path.join(DATA_DIR, "ill_specified_cp_results.csv"))

    if idrnn_ll is None or van_ll is None:
        print("  Likelihood comparison: missing core RNN NLL data, skipping plot.")
        return

    N = len(idrnn_ll)
    print(f"\n── Likelihood comparison (N={N} test participants) ──")

    # ── Build ordered model list in three groups ──────────────────────────────
    # Group 1: Ill-specified cognitive model (CP then EM)
    # Group 2: Full cognitive model (CP then EM)
    # Group 3: RNN models (CP / IDRNN / Vanilla)
    # Groups are separated by a small x-gap of 0.5

    GROUP_GAP = 0.6
    BAR_W     = 0.55

    groups = [
        # (label, data_array, color, group_index)
        ("Ill-spec.\nCP",  ill_cp_ll,  "#9C6FA4"),
        ("Ill-spec.\nEM",  ill_em_ll,  "#C9A8D0"),
        (None, None, None),                          # gap sentinel
        ("Cog. model\nCP", cog_cp_ll,  "#3B8A3B"),
        ("Cog. model\nEM", cog_em_ll,  "#7DC87D"),
        (None, None, None),                          # gap sentinel
        ("RNN\n(no ID)",   rnn_cp_ll,  "#B07030"),
        ("IDRNN",          idrnn_ll,   "#2a82c2"),
        ("Vanilla\nRNN",   van_ll,     "#e1861f"),
    ]

    # Build x positions, skipping gaps
    x_pos, x_labels, x_data, x_colors = [], [], [], []
    cur_x = 0.0
    for entry in groups:
        if entry[0] is None:
            cur_x += GROUP_GAP
            continue
        label, data, color = entry
        if data is None or len(data) != N:
            if data is not None:
                print(f"  Skipping '{label}': length {len(data)} ≠ {N}")
            else:
                print(f"  Skipping '{label}': data not found")
            cur_x += 1.0
            continue
        x_pos.append(cur_x)
        x_labels.append(label)
        x_data.append(data)
        x_colors.append(color)
        cur_x += 1.0

    if not x_pos:
        print("  No models with valid data, skipping plot.")
        return

    means = [np.mean(d) for d in x_data]
    sems  = [float(scipy_sem(d)) for d in x_data]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(x_pos)), 6))

    ax.bar(x_pos, means, width=BAR_W, color=x_colors, alpha=0.85, zorder=2)
    ax.errorbar(x_pos, means, yerr=sems, fmt="none",
                capsize=5, capthick=1.2, ecolor="black", elinewidth=1.2, zorder=5)

    rng = np.random.default_rng(42)
    for xi, data in zip(x_pos, x_data):
        jit = rng.uniform(-BAR_W * 0.25, BAR_W * 0.25, size=len(data))
        ax.scatter(xi + jit, data, alpha=0.45, c="black", s=18, zorder=10, edgecolors="none")

    # ── Significance bars ─────────────────────────────────────────────────────
    ymax = max(m + s for m, s in zip(means, sems))
    h    = (max(means) - min(means)) * 0.025

    def idx_of(label_substr):
        for k, lbl in enumerate(x_labels):
            if label_substr in lbl.replace("\n", " "):
                return k
        return None

    def sig_bar(label_a, label_b, level=0):
        ia, ib = idx_of(label_a), idx_of(label_b)
        if ia is None or ib is None:
            return
        y = ymax + h * (2 + 4 * level)
        _, p = ttest_rel(x_data[ia], x_data[ib])
        _add_sig_bar(ax, x_pos[ia], x_pos[ib], y, h, p)

    # Within-group CP vs EM comparisons
    sig_bar("Ill-spec. CP", "Ill-spec. EM", level=0)
    sig_bar("Cog. model CP", "Cog. model EM", level=0)
    # Key RNN comparisons
    sig_bar("IDRNN", "Vanilla RNN", level=0)
    sig_bar("IDRNN", "RNN (no ID)",  level=1)
    # Best cog model vs IDRNN
    sig_bar("Cog. model EM", "IDRNN", level=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Mean Negative Log-Likelihood per Participant", fontsize=11)
    ax.set_title(
        "Model Likelihood Comparison — Sloutsky Data\n"
        "Lower is Better  ·  RNN NLL averaged over seeds  ·  CV epoch",
        fontsize=11, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group labels underneath
    def _group_label(ax, xi_list, text):
        if not xi_list:
            return
        mid = (xi_list[0] + xi_list[-1]) / 2
        ax.annotate(text, xy=(mid, ax.get_ylim()[0]),
                    xytext=(mid, ax.get_ylim()[0] - (ymax - ax.get_ylim()[0]) * 0.10),
                    ha="center", fontsize=8, color="dimgray",
                    annotation_clip=False)

    fig.tight_layout()
    ll_path = os.path.join(PLOT_DIR, f"likelihood_comparison_cv_seeds{_seed_tag}.png")
    fig.savefig(ll_path, dpi=150, bbox_inches="tight")
    print(f"Likelihood comparison plot saved → {ll_path}")
    for label, m, s in zip(x_labels, means, sems):
        print(f"  {label.replace(chr(10), ' '):25s}: {m:.3f} ± {s:.3f}")
    plt.close(fig)


plot_likelihood_comparison()

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {
    "idrnn": {
        "seeds":   seeds_idrnn,
        "rsa":     {t: rsa_idrnn[t].tolist()    for t in targets},
        "cogcorr": {v: cogcorr_idrnn[v].tolist() for v in cog_vars},
        "decoding":{v: decode_idrnn[v].tolist()  for v in cog_vars},
        "mean_rsa":     {t: float(np.nanmean(rsa_idrnn[t]))    for t in targets},
        "mean_decoding":{v: float(np.nanmean(decode_idrnn[v])) for v in cog_vars},
    },
    "vanilla": {
        "seeds":   seeds_van,
        "rsa":     {t: rsa_van[t].tolist()    for t in targets},
        "cogcorr": {v: cogcorr_van[v].tolist() for v in cog_vars},
        "decoding":{v: decode_van[v].tolist()  for v in cog_vars},
        "mean_rsa":     {t: float(np.nanmean(rsa_van[t]))    for t in targets},
        "mean_decoding":{v: float(np.nanmean(decode_van[v])) for v in cog_vars},
    },
}
json_path = os.path.join(PLOT_DIR, f"analyze_cv_seeds_summary{_seed_tag}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {json_path}")

# ── Best-seed novelty scatter ─────────────────────────────────────────────────
nov_r_per_seed = cogcorr_idrnn["w_novelty"]
best_idx  = int(np.argmax(np.abs(nov_r_per_seed)))
print(f"best novelty corr index: {best_idx}")
best_seed = seeds_idrnn[best_idx]
print(f"best novelty corr seed: {best_seed}")
best_r    = nov_r_per_seed[best_idx]
IDRNN_TAG = "latentmodel"
lat_path  = os.path.join(DATA_DIR, f"seed_{best_seed}", f"latents_tensor{IDRNN_TAG}.pt")
lat       = torch.load(lat_path, map_location="cpu").numpy().mean(axis=1)  # (N, features)

pca       = PCA(n_components=1)
pc1       = pca.fit_transform(lat).ravel()
var_exp   = pca.explained_variance_ratio_[0] * 100

# linear regression line
slope, intercept, *_ = linregress(w_novel_vals, pc1)
x_line = np.linspace(w_novel_vals.min(), w_novel_vals.max(), 100)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(5, 4.5))
sc = ax.scatter(w_novel_vals, pc1,
                c=w_novel_vals, cmap="viridis",
                s=50, edgecolors="k", linewidths=0.5, zorder=3)
plt.colorbar(sc, ax=ax, label="w novelty (EM)")

ax.plot(x_line, y_line, color="k", lw=1.5, ls="--", zorder=2)
ax.set_xlabel("w novelty (EM)", fontsize=11)
ax.set_ylabel(f"PC1 ({var_exp:.1f}% var. explained)", fontsize=11)
ax.set_title(f"Best seed (seed {best_seed})\nPearson r = {best_r:.3f}", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, f"best_novelty_pc1_scatter{_seed_tag}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved best-novelty scatter → {out}")
plt.close(fig)
