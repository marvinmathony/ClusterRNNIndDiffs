#!/usr/bin/env python3
"""
K-means group decoding: Do IDRNN latents separate children vs adults
better than Vanilla RNN hidden states?

Uses k-means (k=2) as an unsupervised classifier. Cluster labels are
aligned to ground truth via the Hungarian algorithm (best permutation).
Each model is tested against chance via a permutation test, and models
are compared via McNemar's test and a paired permutation test on
balanced accuracy.

Produces:
  plots_sloutsky/group_kmeans_clusters.png  – 2D scatter of clusters
  plots_sloutsky/group_kmeans_summary.png   – accuracy bars + null dists + model comparison
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data_sloutsky"
PLOT_DIR = "plots_sloutsky"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── load group labels ────────────────────────────────────────────────────────
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique = df_test.drop_duplicates(subset="subid").sort_values("subid").reset_index(drop=True)
group_map = {"Coin CollectorV6": 0, "Coin CollectorV5": 0, "Coin Collector": 1}
labels = np.array([group_map[g] for g in df_unique["game"].values])
n_children, n_adults = (labels == 0).sum(), (labels == 1).sum()
print(f"Test participants: {len(labels)}  (children={n_children}, adults={n_adults})")

# ── load latent tensors ──────────────────────────────────────────────────────
lat_idrnn_raw = torch.load(f"{DATA_DIR}/latents_tensorlatentmodel.pt", map_location="cpu").numpy()
lat_vanilla_raw = torch.load(f"{DATA_DIR}/latents_tensorvanilla.pt", map_location="cpu").numpy()

# IDRNN: last time step (best posterior estimate)  →  (B, D)
# Vanilla: time-average  →  (B, D)
lat_idrnn = lat_idrnn_raw[:, -1, :]
lat_vanilla = lat_vanilla_raw.mean(axis=1)


# ── helpers ──────────────────────────────────────────────────────────────────
def kmeans_classify(X, y, n_init=50, seed=42):
    """Run k-means (k=2), align cluster labels to ground truth via Hungarian method.
    Returns aligned predictions and balanced accuracy."""
    km = KMeans(n_clusters=2, n_init=n_init, random_state=seed)
    raw_labels = km.fit_predict(X)

    # Build cost matrix for Hungarian alignment
    # cost[i, j] = number of mismatches if cluster i is mapped to true label j
    cost = np.zeros((2, 2))
    for ci in range(2):
        for tj in range(2):
            cost[ci, tj] = np.sum((raw_labels == ci) & (y != tj))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = dict(zip(row_ind, col_ind))

    preds = np.array([mapping[c] for c in raw_labels])
    ba = balanced_accuracy_score(y, preds)
    return preds, ba, km


def permutation_null_kmeans(X, y, n_perm=2000, n_init=50, seed=42):
    """Permutation distribution of balanced accuracy under the null."""
    rng = np.random.RandomState(seed)
    null_accs = np.zeros(n_perm)
    for i in trange(n_perm, desc="Permutation test"):
        y_shuf = rng.permutation(y)
        _, ba, _ = kmeans_classify(X, y_shuf, n_init=n_init, seed=seed + i)
        null_accs[i] = ba
    return null_accs


def mcnemar_test(preds_a, preds_b, y):
    """McNemar's test comparing two classifiers on the same subjects."""
    correct_a = (preds_a == y).astype(int)
    correct_b = (preds_b == y).astype(int)
    b = ((correct_a == 1) & (correct_b == 0)).sum()
    c = ((correct_a == 0) & (correct_b == 1)).sum()
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = chi2.sf(chi2_stat, df=1)
    return chi2_stat, p_val, b, c


def paired_permutation_test(pred_a, pred_b, y, n_perm=2000, seed=42):
    """Permutation test on the difference in balanced accuracy between two models.
    Randomly swaps per-subject predictions between models."""
    rng = np.random.RandomState(seed)
    ba_a = balanced_accuracy_score(y, pred_a)
    ba_b = balanced_accuracy_score(y, pred_b)
    obs_delta = ba_a - ba_b

    null_deltas = np.zeros(n_perm)
    for i in range(n_perm):
        swap = rng.randint(0, 2, size=len(y)).astype(bool)
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        null_deltas[i] = balanced_accuracy_score(y, perm_a) - balanced_accuracy_score(y, perm_b)

    p_val = (np.abs(null_deltas) >= np.abs(obs_delta)).mean()
    return obs_delta, null_deltas, p_val


# ── run k-means classification ───────────────────────────────────────────────
print("Running k-means (k=2) classification ...")
pred_i, ba_i, km_i = kmeans_classify(lat_idrnn, labels)
pred_v, ba_v, km_v = kmeans_classify(lat_vanilla, labels)

print(f"  IDRNN   – bal-acc = {ba_i:.3f}")
print(f"  Vanilla – bal-acc = {ba_v:.3f}")

# ── permutation tests ────────────────────────────────────────────────────────
print("Running permutation tests (2000 permutations each) ...")
null_i = permutation_null_kmeans(lat_idrnn, labels)
null_v = permutation_null_kmeans(lat_vanilla, labels)
p_i = (null_i >= ba_i).mean()
p_v = (null_v >= ba_v).mean()
print(f"  IDRNN   – permutation p = {p_i:.4f}")
print(f"  Vanilla – permutation p = {p_v:.4f}")

# ── model comparison ─────────────────────────────────────────────────────────
print("Comparing models (McNemar + paired permutation) ...")
mcn_chi2, mcn_p, mcn_b, mcn_c = mcnemar_test(pred_i, pred_v, labels)
print(f"  McNemar's test: chi2={mcn_chi2:.3f}, p={mcn_p:.4f}  "
      f"(IDRNN-only correct={mcn_b}, Vanilla-only correct={mcn_c})")

delta_ba, null_deltas, p_delta = paired_permutation_test(pred_i, pred_v, labels)
print(f"  Paired permutation test: delta bal-acc = {delta_ba:+.3f}, p = {p_delta:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – 2D cluster visualisation (PCA for high-dim Vanilla)
# ═══════════════════════════════════════════════════════════════════════════════
palette = ["#4C72B0", "#DD8452"]  # blue=children, orange=adults
marker_correct = "o"
marker_wrong = "X"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, X, preds, km, y, model_name, ba in [
    (axes[0], lat_idrnn, pred_i, km_i, labels, "IDRNN", ba_i),
    (axes[1], lat_vanilla, pred_v, km_v, labels, "Vanilla RNN", ba_v),
]:
    # Reduce to 2D for plotting
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2).fit_transform(X)
        ax_label = "PC"
    elif X.shape[1] == 2:
        X_2d = X
        ax_label = "dim"
    else:
        # 1D: pad with zeros
        X_2d = np.column_stack([X, np.zeros(len(X))])
        ax_label = "dim"

    correct = (preds == y)

    for grp, grp_name, color in [(0, "Children", palette[0]), (1, "Adults", palette[1])]:
        mask_correct = (y == grp) & correct
        mask_wrong = (y == grp) & ~correct
        ax.scatter(X_2d[mask_correct, 0], X_2d[mask_correct, 1],
                   c=color, marker=marker_correct, s=50, edgecolors="k",
                   linewidths=0.5, label=f"{grp_name} (correct)", zorder=3)
        if mask_wrong.any():
            ax.scatter(X_2d[mask_wrong, 0], X_2d[mask_wrong, 1],
                       c=color, marker=marker_wrong, s=70, edgecolors="k",
                       linewidths=0.5, label=f"{grp_name} (wrong)", zorder=4)

    ax.set_xlabel(f"{ax_label} 1", fontsize=11)
    ax.set_ylabel(f"{ax_label} 2", fontsize=11)
    ax.set_title(f"{model_name}  (bal-acc = {ba:.3f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")

fig.suptitle("K-means (k=2) Group Classification", fontsize=14, fontweight="bold", y=1.02)
fig.savefig(f"{PLOT_DIR}/group_kmeans_clusters.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR}/group_kmeans_clusters.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Summary: accuracy bars + null distributions + model comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"wspace": 0.35})

# ── left panel: balanced accuracy ──
ax = axes[0]
ax.bar(["IDRNN", "Vanilla"], [ba_i, ba_v],
       color=["#4C72B0", "#DD8452"], edgecolor="k", width=0.5)
perm_95_i = np.percentile(null_i, 95)
perm_95_v = np.percentile(null_v, 95)
ax.axhline(np.mean([perm_95_i, perm_95_v]), color="grey", ls="--", lw=1,
           label=f"95th %ile null ({np.mean([perm_95_i, perm_95_v]):.2f})")
for x_pos, ba, p_val in [(0, ba_i, p_i), (1, ba_v, p_v)]:
    star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax.text(x_pos, ba + 0.02, f"p={p_val:.3f} {star}", ha="center", fontsize=9)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_title("K-means Classification", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.5)

# ── middle panel: permutation null distributions ──
ax = axes[1]
bins = np.linspace(0.2, 1.0, 30)
ax.hist(null_i, bins=bins, alpha=0.5, color="#4C72B0", label="IDRNN null", density=True)
ax.hist(null_v, bins=bins, alpha=0.5, color="#DD8452", label="Vanilla null", density=True)
ax.axvline(ba_i, color="#4C72B0", lw=2, ls="-", label=f"IDRNN obs ({ba_i:.2f})")
ax.axvline(ba_v, color="#DD8452", lw=2, ls="-", label=f"Vanilla obs ({ba_v:.2f})")
ax.set_xlabel("Balanced Accuracy", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Permutation Null Distributions", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")

# ── right panel: paired model comparison ──
ax = axes[2]
bins_delta = np.linspace(-0.5, 0.5, 40)
ax.hist(null_deltas, bins=bins_delta, alpha=0.6, color="#8C8C8C",
        density=True, label="Null (shuffled)")
ax.axvline(delta_ba, color="#C44E52", lw=2.5, ls="-",
           label=f"Observed delta = {delta_ba:+.3f}")
ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.5)
delta_star = "***" if p_delta < 0.001 else "**" if p_delta < 0.01 else "*" if p_delta < 0.05 else "n.s."
ax.set_xlabel("delta Balanced Accuracy  (IDRNN - Vanilla)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Model Comparison  (p={p_delta:.3f} {delta_star})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")

fig.savefig(f"{PLOT_DIR}/group_kmeans_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR}/group_kmeans_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("K-MEANS GROUP CLASSIFICATION SUMMARY  (children vs adults, k=2)")
print("=" * 65)
print(f"  {'Metric':<35s} {'IDRNN':>12s} {'Vanilla':>12s}")
print(f"  {'-'*35} {'-'*12} {'-'*12}")
print(f"  {'Latent dimensionality':<35s} {lat_idrnn.shape[1]:>12d} {lat_vanilla.shape[1]:>12d}")
print(f"  {'Balanced accuracy':<35s} {ba_i:>12.3f} {ba_v:>12.3f}")
print(f"  {'Permutation p-value':<35s} {p_i:>12.4f} {p_v:>12.4f}")
print("-" * 65)
print(f"  MODEL COMPARISON (IDRNN vs Vanilla)")
print(f"  {'delta bal-acc (IDRNN - Vanilla)':<35s} {delta_ba:>+12.3f}")
print(f"  {'Paired permutation p-value':<35s} {p_delta:>12.4f}")
print(f"  {'McNemar chi2':<35s} {mcn_chi2:>12.3f}")
print(f"  {'McNemar p-value':<35s} {mcn_p:>12.4f}")
print(f"  {'IDRNN-only correct (b)':<35s} {mcn_b:>12d}")
print(f"  {'Vanilla-only correct (c)':<35s} {mcn_c:>12d}")
print("=" * 65)
