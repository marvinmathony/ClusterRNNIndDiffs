#!/usr/bin/env python3
"""
Plot boxplots of the EM-fitted cognitive model weights (value, uncertainty,
lag, novelty) and inverse temperature (theta) for each age group
(young_child, old_child, adult).

Also plots novelty weight vs RNN latents (analogous to plot_alpha_vs_z in
analyze_synthetic_multi_dataset.py).

Requires:  data_sloutsky/em_results.npz  (produced by sloutsky_cog_model.py)
           data_sloutsky/latents_tensorlatentmodel.pt
           data_sloutsky/latents_tensorvanilla.pt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from scipy.stats import kruskal, pearsonr

PLOT_DIR = "plots_sloutsky"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load EM results
# ------------------------------------------------------------------
em = np.load("data_sloutsky/em_results.npz", allow_pickle=True)
h_all      = em["h_all"]        # (P, 5) unconstrained MAP estimates
participants = em["participants"]  # (P,) subids

# ------------------------------------------------------------------
# 2. Convert unconstrained params → interpretable quantities
# ------------------------------------------------------------------
def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

P = h_all.shape[0]
theta_all = np.exp(h_all[:, 0])                                   # (P,)
weights_all = np.array([softmax(h_all[p, 1:5]) for p in range(P)])  # (P, 4)

# ------------------------------------------------------------------
# 3. Build participant → age-group lookup from the full dataset CSV
# ------------------------------------------------------------------
df_full = pd.read_csv("data_sloutsky/exp2_train_all_participants.csv")
age_map = (df_full.groupby("subid")["age"]
                  .first()
                  .to_dict())

# test-split participants (sorted by subid, matching latent tensor ordering)
df_test = pd.read_csv("data_sloutsky/df_test.csv")
test_participants_sorted = np.array(sorted(df_test["subid"].unique()))

age_groups = np.array([age_map[s] for s in participants])

GROUP_ORDER  = ["young_child", "old_child", "adult"]
GROUP_LABELS = ["Young\nchildren", "Older\nchildren", "Adults"]
GROUP_COLORS = ["#5B9BD5", "#ED7D31", "#A5A5A5"]

# ------------------------------------------------------------------
# 4. Plotting helper
# ------------------------------------------------------------------
def boxplot_by_group(ax, values, age_groups, title, ylabel):
    """Draw boxplots + individual data points for one parameter."""
    data_by_group = [values[age_groups == g] for g in GROUP_ORDER]

    bp = ax.boxplot(
        data_by_group,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], GROUP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Jittered individual points
    for i, (data, color) in enumerate(zip(data_by_group, GROUP_COLORS), start=1):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
        ax.scatter(i + jitter, data, color=color, s=20, alpha=0.7,
                   zorder=3, edgecolors="none")

    # Kruskal-Wallis test
    if all(len(d) > 0 for d in data_by_group):
        stat, pval = kruskal(*data_by_group)
        stars = ("***" if pval < 0.001 else
                 "**"  if pval < 0.01  else
                 "*"   if pval < 0.05  else "n.s.")
        ax.set_title(f"{title}  (KW p={pval:.3f} {stars})", fontsize=10)
    else:
        ax.set_title(title, fontsize=10)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(GROUP_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ------------------------------------------------------------------
# 5. Figure: 5 panels (theta + 4 weights)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
fig.suptitle("Cognitive model parameters by age group (EM fit)", fontsize=13)

param_specs = [
    (theta_all,          "Inverse temperature (θ)", "θ"),
    (weights_all[:, 0],  "Value weight",            "w_value"),
    (weights_all[:, 1],  "Uncertainty weight",       "w_uncertainty"),
    (weights_all[:, 2],  "Lag weight",               "w_lag"),
    (weights_all[:, 3],  "Novelty weight",           "w_novelty"),
]

for ax, (values, title, ylabel) in zip(axes, param_specs):
    boxplot_by_group(ax, values, age_groups, title, ylabel)

plt.tight_layout()
outpath = os.path.join(PLOT_DIR, "cog_model_weights_by_age.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()


# ------------------------------------------------------------------
# 6. Novelty weight vs RNN latents  (analogous to plot_alpha_vs_z)
# ------------------------------------------------------------------
def plot_novelty_weight_vs_latents(
    em_participants,        # (P_em,) all participants from EM
    h_all,                  # (P_em, 5) unconstrained MAP estimates from EM
    test_participants_sorted,  # (P_test,) participants in latent tensor order
    age_map,                # dict subid → age group
    latent_path,            # path to .pt latent tensor  (P_test, T, z_dim)
    model_label,            # string for plot titles / filenames
    plot_dir,
):
    """
    Scatter plot of w_novelty (from EM) against the first PCA component of
    the RNN latent representations, coloured by age group.
    One figure with three panels: PCA scatter, w_novelty vs PC1, w_novelty vs PC2
    (mirrors the multidimensional branch of plot_alpha_vs_z).
    """
    # ---- load latent tensor ------------------------------------------------
    latent_tensor = torch.load(latent_path, map_location="cpu")
    if hasattr(latent_tensor, "detach"):
        latent_np = latent_tensor.detach().numpy()
    else:
        latent_np = np.array(latent_tensor)

    # shape: (P_test, T, z_dim)  → take last timestep → (P_test, z_dim)
    if latent_np.ndim == 3:
        z_values = latent_np[:, -1, :]
        z_dim = latent_np.shape[2]
    elif latent_np.ndim == 2:
        z_values = latent_np
        z_dim = latent_np.shape[1]
    else:
        z_values = latent_np.reshape(-1, 1)
        z_dim = 1

    # ---- align EM estimates to test-split participant order ----------------
    em_id_to_idx = {sid: i for i, sid in enumerate(em_participants)}

    w_novelty = []
    age_labels = []
    valid_mask = []

    for sid in test_participants_sorted:
        if sid in em_id_to_idx:
            idx = em_id_to_idx[sid]
            w = softmax(h_all[idx, 1:5])[3]   # w_novelty
            w_novelty.append(w)
            age_labels.append(age_map.get(sid, "unknown"))
            valid_mask.append(True)
        else:
            # participant not fitted by EM (shouldn't happen but be safe)
            w_novelty.append(np.nan)
            age_labels.append("unknown")
            valid_mask.append(False)

    w_novelty  = np.array(w_novelty)
    age_labels = np.array(age_labels)
    valid      = np.array(valid_mask)

    # filter out any missing
    z_values   = z_values[valid]
    w_novelty  = w_novelty[valid]
    age_labels = age_labels[valid]

    # ---- PCA reduction -----------------------------------------------------
    n_components = min(2, z_dim)
    pca = PCA(n_components=n_components)
    z_reduced = pca.fit_transform(z_values)
    explained_var = pca.explained_variance_ratio_

    # ---- map age groups to colours -----------------------------------------
    color_map = {
        "young_child": GROUP_COLORS[0],
        "old_child":   GROUP_COLORS[1],
        "adult":       GROUP_COLORS[2],
    }
    point_colors = np.array([color_map.get(a, "#888888") for a in age_labels])

    # ---- figure ------------------------------------------------------------
    if n_components >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{model_label}: Novelty weight vs latent space (last timestep)",
                     fontsize=13)

        # Panel 0: PCA scatter coloured by w_novelty
        sc = axes[0].scatter(z_reduced[:, 0], z_reduced[:, 1],
                             c=w_novelty, cmap="viridis",
                             alpha=0.8, edgecolors="k", linewidths=0.4, s=50)
        axes[0].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)", fontsize=11)
        axes[0].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)", fontsize=11)
        axes[0].set_title("Latent PCA\n(colour = w_novelty)", fontsize=11)
        plt.colorbar(sc, ax=axes[0], label="w_novelty")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Panels 1 & 2: w_novelty vs PC1 / PC2, coloured by age group
        for panel_idx, pc_idx in enumerate([0, 1], start=1):
            ax = axes[panel_idx]
            pc_label = f"PC{pc_idx+1} ({explained_var[pc_idx]*100:.1f}% var)"

            for grp, col, lbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
                mask = age_labels == grp
                ax.scatter(w_novelty[mask], z_reduced[mask, pc_idx],
                           color=col, alpha=0.75, s=50,
                           edgecolors="none", label=lbl.replace("\n", " "))

            # regression line + correlation over all participants
            finite = np.isfinite(w_novelty) & np.isfinite(z_reduced[:, pc_idx])
            if finite.sum() > 2:
                r, pval = pearsonr(w_novelty[finite], z_reduced[finite, pc_idx])
                coef = np.polyfit(w_novelty[finite], z_reduced[finite, pc_idx], 1)
                x_line = np.linspace(w_novelty[finite].min(),
                                     w_novelty[finite].max(), 100)
                ax.plot(x_line, np.poly1d(coef)(x_line), "r--",
                        linewidth=2, label=f"r = {r:.3f}, p = {pval:.3f}")

            ax.set_xlabel("w_novelty (EM fit)", fontsize=11)
            ax.set_ylabel(pc_label, fontsize=11)
            ax.set_title(f"w_novelty vs {pc_label.split(' ')[0]}", fontsize=11)
            ax.legend(fontsize=8, frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    else:
        # 1D latent space — single scatter
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"{model_label}: Novelty weight vs latent (last timestep)",
                     fontsize=13)

        for grp, col, lbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
            mask = age_labels == grp
            ax.scatter(w_novelty[mask], z_reduced[mask, 0],
                       color=col, alpha=0.75, s=50,
                       edgecolors="none", label=lbl.replace("\n", " "))

        finite = np.isfinite(w_novelty) & np.isfinite(z_reduced[:, 0])
        if finite.sum() > 2:
            r, pval = pearsonr(w_novelty[finite], z_reduced[finite, 0])
            coef = np.polyfit(w_novelty[finite], z_reduced[finite, 0], 1)
            x_line = np.linspace(w_novelty[finite].min(),
                                 w_novelty[finite].max(), 100)
            ax.plot(x_line, np.poly1d(coef)(x_line), "r--",
                    linewidth=2, label=f"r = {r:.3f}, p = {pval:.3f}")

        ax.set_xlabel("w_novelty (EM fit)", fontsize=11)
        ax.set_ylabel("Latent z", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"novelty_weight_vs_latents_{model_label.lower().replace(' ', '_')}.png"
    outpath = os.path.join(plot_dir, fname)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


# Run for both IDRNN and vanilla
for model_label, latent_file in [
    ("IDRNN",       "data_sloutsky/latents_tensorlatentmodel.pt"),
    ("Vanilla RNN", "data_sloutsky/latents_tensorvanilla.pt"),
]:
    plot_novelty_weight_vs_latents(
        em_participants=participants,
        h_all=h_all,
        test_participants_sorted=test_participants_sorted,
        age_map=age_map,
        latent_path=latent_file,
        model_label=model_label,
        plot_dir=PLOT_DIR,
    )
