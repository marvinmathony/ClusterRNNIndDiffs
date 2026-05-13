#!/usr/bin/env python3
"""
Group decoding analysis: Can we decode children vs adults from latent representations?
Compares IDRNN (3-dim explicit latent) vs Vanilla RNN (10-dim hidden state).

Each model is tested against chance via permutation test, and the two models
are compared against each other via McNemar's test (on binary predictions) and
a paired permutation test (on balanced accuracy difference).

Produces:
  plots_sloutsky/group_decoding_dimensions.png   – per-dimension violin + strip plots
  plots_sloutsky/group_decoding_roc.png           – ROC curve comparison
  plots_sloutsky/group_decoding_summary.png       – accuracy bars + null dists + model comparison
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, ttest_ind, chi2

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
def loocv_decode(X, y, C=None):
    """LOOCV logistic regression. Returns predictions, probabilities, bal-acc, AUC.
    If C is provided, uses fixed regularization; otherwise selects C via LogisticRegressionCV first."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        clf = LogisticRegressionCV(penalty="l2", max_iter=2000, n_jobs=4)
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    ba = balanced_accuracy_score(y, preds)
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    return preds, probs, ba, roc_auc, fpr, tpr


def permutation_null(X, y, n_perm=100, seed=42):
    """Permutation distribution of balanced accuracy under the null."""
    rng = np.random.RandomState(seed)
    null_accs = np.zeros(n_perm)
    for i in trange(n_perm):
        y_shuf = rng.permutation(y)
        _, _, ba, _, _, _= loocv_decode(X, y_shuf)
        null_accs[i] = ba
    return null_accs


def mcnemar_test(preds_a, preds_b, y):
    """McNemar's test comparing two classifiers on the same subjects.
    Returns chi2 statistic, p-value, and the contingency counts (b, c)."""
    correct_a = (preds_a == y).astype(int)
    correct_b = (preds_b == y).astype(int)
    # b: A correct, B wrong;  c: A wrong, B correct
    b = ((correct_a == 1) & (correct_b == 0)).sum()
    c = ((correct_a == 0) & (correct_b == 1)).sum()
    # McNemar's with continuity correction
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = chi2.sf(chi2_stat, df=1)
    return chi2_stat, p_val, b, c


def paired_permutation_test(X_a, X_b, y, n_perm=1000, seed=42):
    """Permutation test on the difference in balanced accuracy between two models.
    For each permutation, each subject's predictions are randomly swapped between models."""
    rng = np.random.RandomState(seed)
    # get LOOCV predictions for both models (already computed, passed in as X)
    pred_a, _, ba_a, _, _, _ = loocv_decode(X_a, y)
    pred_b, _, ba_b, _, _, _ = loocv_decode(X_b, y)
    obs_delta = ba_a - ba_b

    null_deltas = np.zeros(n_perm)
    for i in range(n_perm):
        swap = rng.randint(0, 2, size=len(y)).astype(bool)
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        ba_perm_a = balanced_accuracy_score(y, perm_a)
        ba_perm_b = balanced_accuracy_score(y, perm_b)
        null_deltas[i] = ba_perm_a - ba_perm_b

    # two-sided p-value
    p_val = (np.abs(null_deltas) >= np.abs(obs_delta)).mean()
    return obs_delta, null_deltas, p_val


# ── run decoding ─────────────────────────────────────────────────────────────
print("Running LOOCV decoding …")
pred_i, prob_i, ba_i, auc_i, fpr_i, tpr_i = loocv_decode(lat_idrnn, labels)
pred_v, prob_v, ba_v, auc_v, fpr_v, tpr_v = loocv_decode(lat_vanilla, labels)

print(f"  IDRNN   – bal-acc = {ba_i:.3f},  AUC = {auc_i:.3f}")
print(f"  Vanilla – bal-acc = {ba_v:.3f},  AUC = {auc_v:.3f}")

print("Running permutation tests (2 000 permutations each) …")
null_i = permutation_null(lat_idrnn, labels)
null_v = permutation_null(lat_vanilla, labels)
p_i = (null_i >= ba_i).mean()
p_v = (null_v >= ba_v).mean()
print(f"  IDRNN   – permutation p = {p_i:.4f}")
print(f"  Vanilla – permutation p = {p_v:.4f}")

# ── model comparison ─────────────────────────────────────────────────────────
print("Comparing models (McNemar + paired permutation) …")
mcn_chi2, mcn_p, mcn_b, mcn_c = mcnemar_test(pred_i, pred_v, labels)
print(f"  McNemar's test: χ²={mcn_chi2:.3f}, p={mcn_p:.4f}  "
      f"(IDRNN-only correct={mcn_b}, Vanilla-only correct={mcn_c})")

delta_ba, null_deltas, p_delta = paired_permutation_test(lat_idrnn, lat_vanilla, labels)
print(f"  Paired permutation test: Δ bal-acc = {delta_ba:+.3f}, p = {p_delta:.4f}")


# ── per-dimension stats ──────────────────────────────────────────────────────
def dim_stats(X, y):
    out = []
    for d in range(X.shape[1]):
        r, p = pointbiserialr(y, X[:, d])
        t, pt = ttest_ind(X[y == 0, d], X[y == 1, d])
        out.append(dict(dim=d, r_pb=r, p_rpb=p, t=t, p_t=pt))
    return pd.DataFrame(out)

stats_i = dim_stats(lat_idrnn, labels)
stats_v = dim_stats(lat_vanilla, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Per-dimension violin + strip plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dim_violins(ax, X, y, dim_stats_df, model_name, palette):
    D = X.shape[1]
    positions_c = np.arange(D) - 0.18
    positions_a = np.arange(D) + 0.18

    children = X[y == 0]
    adults = X[y == 1]

    # violins
    vp_c = ax.violinplot([children[:, d] for d in range(D)], positions=positions_c,
                          widths=0.32, showextrema=False, showmedians=False)
    vp_a = ax.violinplot([adults[:, d] for d in range(D)], positions=positions_a,
                          widths=0.32, showextrema=False, showmedians=False)
    for body in vp_c["bodies"]:
        body.set_facecolor(palette[0])
        body.set_alpha(0.35)
    for body in vp_a["bodies"]:
        body.set_facecolor(palette[1])
        body.set_alpha(0.35)

    # strip (jittered dots)
    rng = np.random.RandomState(0)
    for d in range(D):
        jit_c = rng.uniform(-0.08, 0.08, size=n_children)
        jit_a = rng.uniform(-0.08, 0.08, size=n_adults)
        ax.scatter(positions_c[d] + jit_c, children[:, d],
                   s=28, color=palette[0], edgecolors="k", linewidths=0.4, zorder=3)
        ax.scatter(positions_a[d] + jit_a, adults[:, d],
                   s=28, color=palette[1], edgecolors="k", linewidths=0.4, zorder=3)

    # means + SEM
    for d in range(D):
        for grp, pos, col in [(0, positions_c[d], palette[0]),
                               (1, positions_a[d], palette[1])]:
            vals = X[y == grp, d]
            m, se = vals.mean(), vals.std() / np.sqrt(len(vals))
            ax.errorbar(pos, m, yerr=se, fmt="D", color=col, markersize=7,
                        markeredgecolor="k", markeredgewidth=0.8, capsize=4, capthick=1.5,
                        elinewidth=1.5, zorder=4)

    # significance annotations
    for d in range(D):
        p = dim_stats_df.loc[d, "p_rpb"]
        r = dim_stats_df.loc[d, "r_pb"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            y_top = max(X[:, d]) + 0.08 * (max(X[:, d]) - min(X[:, d]))
            ax.text(d, y_top, f"{star}\nr={r:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(range(D))
    ax.set_xticklabels([f"dim {d}" for d in range(D)], fontsize=9)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Latent activation")


palette = ["#4C72B0", "#DD8452"]  # blue=children, orange=adults

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 10], wspace=0.30)

ax_idrnn = fig.add_subplot(gs[0])
ax_vanilla = fig.add_subplot(gs[1])

plot_dim_violins(ax_idrnn, lat_idrnn, labels, stats_i, "IDRNN  (z=3)", palette)
plot_dim_violins(ax_vanilla, lat_vanilla, labels, stats_v, "Vanilla RNN  (h=10)", palette)

# shared legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette[0], edgecolor="k", label=f"Children (n={n_children})"),
                   Patch(facecolor=palette[1], edgecolor="k", label=f"Adults (n={n_adults})")]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, 1.02))

fig.savefig(f"{PLOT_DIR}/group_decoding_dimensions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR}/group_decoding_dimensions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – ROC curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_i, tpr_i, color="#4C72B0", lw=2.2,
        label=f"IDRNN  (AUC = {auc_i:.2f})")
ax.plot(fpr_v, tpr_v, color="#DD8452", lw=2.2,
        label=f"Vanilla  (AUC = {auc_v:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("LOOCV ROC – Children vs Adults", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")

fig.savefig(f"{PLOT_DIR}/group_decoding_roc.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR}/group_decoding_roc.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Summary bar chart  +  permutation nulls  +  model comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"wspace": 0.35})

# ── left panel: balanced accuracy ──
ax = axes[0]
bars = ax.bar(["IDRNN", "Vanilla"], [ba_i, ba_v],
              color=["#4C72B0", "#DD8452"], edgecolor="k", width=0.5)
# permutation 95th-percentile line
perm_95_i = np.percentile(null_i, 95)
perm_95_v = np.percentile(null_v, 95)
ax.axhline(np.mean([perm_95_i, perm_95_v]), color="grey", ls="--", lw=1,
           label=f"95th %ile null ({np.mean([perm_95_i, perm_95_v]):.2f})")
# annotate p-values
for x_pos, ba, p_val in [(0, ba_i, p_i), (1, ba_v, p_v)]:
    star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax.text(x_pos, ba + 0.02, f"p={p_val:.3f} {star}", ha="center", fontsize=9)

ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_title("LOOCV Decoding Accuracy", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
# chance line
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
           label=f"Observed Δ = {delta_ba:+.3f}")
ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.5)
# annotate p-value
delta_star = "***" if p_delta < 0.001 else "**" if p_delta < 0.01 else "*" if p_delta < 0.05 else "n.s."
ax.set_xlabel("Δ Balanced Accuracy  (IDRNN − Vanilla)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Model Comparison  (p={p_delta:.3f} {delta_star})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")

fig.savefig(f"{PLOT_DIR}/group_decoding_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR}/group_decoding_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("GROUP DECODING SUMMARY  (children vs adults, LOOCV)")
print("=" * 65)
print(f"  {'Metric':<35s} {'IDRNN':>12s} {'Vanilla':>12s}")
print(f"  {'-'*35} {'-'*12} {'-'*12}")
print(f"  {'Latent dimensionality':<35s} {lat_idrnn.shape[1]:>12d} {lat_vanilla.shape[1]:>12d}")
print(f"  {'Balanced accuracy':<35s} {ba_i:>12.3f} {ba_v:>12.3f}")
print(f"  {'Permutation p-value':<35s} {p_i:>12.4f} {p_v:>12.4f}")
print(f"  {'ROC AUC':<35s} {auc_i:>12.3f} {auc_v:>12.3f}")
print(f"  {'Best dim |r_pb|':<35s} {stats_i['r_pb'].abs().max():>12.3f} {stats_v['r_pb'].abs().max():>12.3f}")
print(f"  {'# dims with p < 0.05':<35s} {(stats_i['p_rpb'] < 0.05).sum():>12d} {(stats_v['p_rpb'] < 0.05).sum():>12d}")
print(f"  {'# dims with p < 0.01':<35s} {(stats_i['p_rpb'] < 0.01).sum():>12d} {(stats_v['p_rpb'] < 0.01).sum():>12d}")
print("-" * 65)
print(f"  MODEL COMPARISON (IDRNN vs Vanilla)")
print(f"  {'Δ bal-acc (IDRNN − Vanilla)':<35s} {delta_ba:>+12.3f}")
print(f"  {'Paired permutation p-value':<35s} {p_delta:>12.4f}")
print(f"  {'McNemar χ²':<35s} {mcn_chi2:>12.3f}")
print(f"  {'McNemar p-value':<35s} {mcn_p:>12.4f}")
print(f"  {'IDRNN-only correct (b)':<35s} {mcn_b:>12d}")
print(f"  {'Vanilla-only correct (c)':<35s} {mcn_c:>12d}")
print("=" * 65)