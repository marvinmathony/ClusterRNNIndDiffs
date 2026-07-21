#!/usr/bin/env python3
"""
Standalone replot of the individual-fit and z-space learning-curve analyses,
loading curves from plots_thalmann/step1_learning_curves.npz so we don't have
to rerun the rollouts. Produces the same outputs as the corresponding section
of analyze_thalmann_logit_updates.py:

  step1_individual_fit_quality.png
  step1_individual_learning_curves.png            (advantage-stratified)
  step1_individual_learning_curves_zspace.png     (PC1-stratified)
  step1_zspace_advantage_map.png

Note: also reads the IDRNN latents (saved["z"], saved["subids"]) to build the
PCA of z-space.
"""
import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

PLOT_DIR = "plots_thalmann"
NPZ_PATH = os.path.join(PLOT_DIR, "step1_learning_curves.npz")
SMOOTH_W = 20
N_SAMPLE = 12

# ── Load saved curves ─────────────────────────────────────────────────────────
print(f"Loading curves: {NPZ_PATH}")
d = np.load(NPZ_PATH)
subids_v       = d["subids"]
human_curves   = d["human"]
idrnn_curves   = d["idrnn"]
vanilla_h_curves = d["vanilla_h"]
vanilla_curves = d["vanilla"]
N_TRIALS       = human_curves.shape[1]
trials         = np.arange(1, N_TRIALS + 1)
print(f"  N participants: {len(subids_v)}, T={N_TRIALS}")

# ── Load IDRNN latents for z-space PCA ────────────────────────────────────────
_latent_glob = glob.glob(
    "plots_thalmann/step1_vs_vanilla/latents_idrnn_step1_bestseed*.pt"
)
ALL_SUBJ_LATENTS = max(_latent_glob, key=os.path.getmtime)
print(f"Loading latents: {ALL_SUBJ_LATENTS}")
saved = torch.load(ALL_SUBJ_LATENTS, map_location="cpu", weights_only=False)
emb_all   = np.asarray(saved["z"])
subids_all = np.asarray(saved["subids"])

# Align emb to subids_v ordering (the npz)
sid_to_idx = {int(s): i for i, s in enumerate(subids_all)}
emb_v = np.stack([emb_all[sid_to_idx[int(s)]] for s in subids_v])
BEST_SEED = saved["seed"]
print(f"  emb_v shape: {emb_v.shape}, best seed: {BEST_SEED}")

N_RNG_SEEDS = 100   # for plot annotations only — not used in computation

# ── Truncating-window smoothing ───────────────────────────────────────────────
def _smooth(curves, w):
    half = w // 2
    T = curves.shape[1]
    c = np.concatenate([np.zeros((curves.shape[0], 1)), np.cumsum(curves, axis=1)], axis=1)
    out = np.empty_like(curves)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out[:, t] = (c[:, hi] - c[:, lo]) / (hi - lo)
    return out

human_sm    = _smooth(human_curves,    SMOOTH_W)
idrnn_sm    = _smooth(idrnn_curves,    SMOOTH_W)
vanillaH_sm = _smooth(vanilla_h_curves,SMOOTH_W)
vanilla_sm  = _smooth(vanilla_curves,  SMOOTH_W)

# ── Per-participant correlations ──────────────────────────────────────────────
def _per_subj_corr(A_, B_):
    out = np.full(A_.shape[0], np.nan)
    for i in range(A_.shape[0]):
        a, b = A_[i], B_[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 5 and a[m].std() > 0 and b[m].std() > 0:
            out[i] = np.corrcoef(a[m], b[m])[0, 1]
    return out

r_idrnn   = _per_subj_corr(human_sm, idrnn_sm)
r_vanH    = _per_subj_corr(human_sm, vanillaH_sm)
r_vanilla = _per_subj_corr(human_sm, vanilla_sm)

mask_id = np.isfinite(r_idrnn) & np.isfinite(r_vanilla)
mask_vh = np.isfinite(r_idrnn) & np.isfinite(r_vanH)

W_iv, p_iv = stats.wilcoxon(r_idrnn[mask_id], r_vanilla[mask_id], alternative="greater")
W_ih, p_ih = stats.wilcoxon(r_idrnn[mask_vh], r_vanH[mask_vh],    alternative="greater")
d_iv = ((r_idrnn[mask_id] - r_vanilla[mask_id]).mean() /
        (r_idrnn[mask_id] - r_vanilla[mask_id]).std(ddof=1))
d_ih = ((r_idrnn[mask_vh] - r_vanH[mask_vh]).mean() /
        (r_idrnn[mask_vh] - r_vanH[mask_vh]).std(ddof=1))

print(f"\nPer-subject Pearson r:")
print(f"  IDRNN     : mean={np.nanmean(r_idrnn):+.3f} (n={mask_id.sum()})")
print(f"  Vanilla+h : mean={np.nanmean(r_vanH):+.3f}")
print(f"  Vanilla   : mean={np.nanmean(r_vanilla):+.3f}")
print(f"IDRNN > Vanilla   : Wilcoxon W={W_iv:.0f}, p={p_iv:.2e}, d={d_iv:+.3f}")
print(f"IDRNN > Vanilla+h : Wilcoxon W={W_ih:.0f}, p={p_ih:.2e}, d={d_ih:+.3f}")

# ── Plot 1: fit-quality scatter + boxplot ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
ax.scatter(r_vanilla[mask_id], r_idrnn[mask_id], color="#4C72B0", s=30, alpha=0.7,
           edgecolors="black", linewidths=0.3, zorder=3)
lo = min(np.nanmin(r_vanilla), np.nanmin(r_idrnn)) - 0.05
hi = max(np.nanmax(r_vanilla), np.nanmax(r_idrnn)) + 0.05
ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1, alpha=0.6, label="y = x")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Pearson r (Vanilla vs human)", fontsize=11)
ax.set_ylabel("Pearson r (IDRNN vs human)", fontsize=11)
ax.set_title(f"Per-participant fit quality\n"
             f"Wilcoxon p={p_iv:.2e}, d={d_iv:+.3f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
data = [r_idrnn[mask_id], r_vanH[mask_vh], r_vanilla[mask_id]]
labels = [f"IDRNN\n(n={mask_id.sum()})",
          f"Vanilla+h\n(n={mask_vh.sum()})",
          f"Vanilla\n(n={mask_id.sum()})"]
colors_ = ["#4C72B0", "#8C564B", "#DD8452"]
bp = ax.boxplot(data, positions=range(3), widths=0.5,
                patch_artist=True, showfliers=False, zorder=2)
for patch, c in zip(bp["boxes"], colors_):
    patch.set_facecolor(c); patch.set_alpha(0.4)
for elem in ["whiskers", "caps", "medians"]:
    for line in bp[elem]:
        line.set_color("black")
for i, (vals, c) in enumerate(zip(data, colors_)):
    jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
    ax.scatter(i + jitter, vals, color=c, s=20, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)
ax.axhline(0, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Pearson r (model vs human, per subject)", fontsize=11)
ax.set_title("Distribution of per-subject fit quality",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_fit_quality.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 1b: per-subject mean reward — human vs each model, across windows ──
# Isolates *level matching* from *temporal-shape matching*. We compute the
# per-subject mean over multiple trial windows (early / full / late) so we
# can see whether IDRNN's per-subject mean tracks humans' uniformly across
# the task, only at asymptote, or only early on.
def _window_mean(curves, lo, hi):
    return np.nanmean(curves[:, lo:hi], axis=1)

windows = [
    ("First 50 (early)", 0, 50),
    ("Full 200",         0, N_TRIALS),
    ("Last 50 (late)",   N_TRIALS - 50, N_TRIALS),
]
models = [
    ("IDRNN",     idrnn_curves,     "#4C72B0"),
    ("Vanilla+h", vanilla_h_curves, "#8C564B"),
    ("Vanilla",   vanilla_curves,   "#DD8452"),
]

fig, axes = plt.subplots(len(windows), len(models),
                          figsize=(5 * len(models), 5 * len(windows)))
print("\nPer-subject mean reward correlations:")
for ri, (wlabel, lo, hi) in enumerate(windows):
    m_human = _window_mean(human_curves, lo, hi)
    print(f"  {wlabel:<22s} human mean={np.nanmean(m_human):.2f}, "
          f"std={np.nanstd(m_human):.2f}")
    # Symmetric axis range across models for this row
    all_in_row = [m_human]
    for _, curves, _ in models:
        all_in_row.append(_window_mean(curves, lo, hi))
    all_concat = np.concatenate(all_in_row)
    axis_lo = np.nanmin(all_concat) - 1
    axis_hi = np.nanmax(all_concat) + 1

    for ci, (mlabel, curves, color) in enumerate(models):
        m_model = _window_mean(curves, lo, hi)
        m = np.isfinite(m_human) & np.isfinite(m_model)
        r_pe, p_pe = stats.pearsonr(m_human[m], m_model[m])
        r_sp, p_sp = stats.spearmanr(m_human[m], m_model[m])
        print(f"    [{wlabel}] {mlabel:<10s}: "
              f"Pearson r={r_pe:+.3f} p={p_pe:.2e}  |  "
              f"Spearman r={r_sp:+.3f} p={p_sp:.2e}  (n={m.sum()})")

        ax = axes[ri, ci]
        ax.scatter(m_human[m], m_model[m], color=color, s=25, alpha=0.7,
                   edgecolors="black", linewidths=0.3, zorder=3)
        ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], color="grey",
                ls=":", lw=1, alpha=0.6, label="y = x")
        slope, intercept, *_ = stats.linregress(m_human[m], m_model[m])
        xf = np.linspace(np.nanmin(m_human[m]), np.nanmax(m_human[m]), 100)
        ax.plot(xf, slope * xf + intercept, color="black", lw=1.6, ls="--",
                label="OLS fit")
        ax.set_xlim(axis_lo, axis_hi); ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")
        if ri == len(windows) - 1:
            ax.set_xlabel(f"Human mean reward", fontsize=10)
        if ci == 0:
            ax.set_ylabel(f"{wlabel}\nModel mean reward", fontsize=10)
        ax.set_title(f"{mlabel}  (n={m.sum()})\n"
                     f"Pearson r={r_pe:+.3f}, p={p_pe:.2e}",
                     fontsize=9, fontweight="bold")
        if ri == 0 and ci == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

fig.suptitle("Per-subject mean reward: human vs model — across trial windows\n"
             "(rows = window; cols = model; level matching only, no temporal shape)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_per_subject_mean_reward.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 2: advantage-stratified individuals ─────────────────────────────────
adv = r_idrnn - r_vanilla
ok  = np.where(np.isfinite(adv))[0]
ok_sorted = ok[np.argsort(adv[ok])]
sample_idx = ok_sorted[np.linspace(0, len(ok_sorted) - 1, N_SAMPLE, dtype=int)]

cols = 4; rows = (N_SAMPLE + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                          sharex=True, sharey=True)
axes = axes.ravel()
for ax_i, pi in enumerate(sample_idx):
    ax = axes[ax_i]
    ax.plot(trials, human_sm[pi],    color="#666666", lw=1.5, label="Human")
    ax.plot(trials, idrnn_sm[pi],    color="#4C72B0", lw=1.5, label="IDRNN")
    ax.plot(trials, vanillaH_sm[pi], color="#8C564B", lw=1.5, label="Vanilla+h")
    ax.plot(trials, vanilla_sm[pi],  color="#DD8452", lw=1.5, label="Vanilla")
    sid = int(subids_v[pi])
    ax.set_title(f"sub {sid}  |  r_id={r_idrnn[pi]:+.2f}, "
                 f"r_v={r_vanilla[pi]:+.2f}", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for j in range(len(sample_idx), len(axes)):
    axes[j].set_visible(False)
axes[0].legend(fontsize=8, loc="lower right")
fig.supxlabel("Trial", fontsize=11)
fig.supylabel(f"Mean reward (rolling window={SMOOTH_W})", fontsize=11)
fig.suptitle(
    f"Sampled individual learning curves "
    f"(stratified by IDRNN−Vanilla r advantage)\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_RNG_SEEDS} rollouts/subject",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_learning_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 3: z-PC1 stratified individuals ─────────────────────────────────────
pca = PCA(n_components=min(3, emb_v.shape[1])).fit(emb_v)
z_scores = pca.transform(emb_v)
ev = pca.explained_variance_ratio_
print(f"\nz-space PCA: explained variance ratio = "
      f"{', '.join(f'{e:.2%}' for e in ev)}")

ok_z   = np.where(np.isfinite(human_curves[:, 0]))[0]
order  = ok_z[np.argsort(z_scores[ok_z, 0])]
sample_idx_z = order[np.linspace(0, len(order) - 1, N_SAMPLE, dtype=int)]

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                          sharex=True, sharey=True)
axes = axes.ravel()
for ax_i, pi in enumerate(sample_idx_z):
    ax = axes[ax_i]
    ax.plot(trials, human_sm[pi],    color="#666666", lw=1.5, label="Human")
    ax.plot(trials, idrnn_sm[pi],    color="#4C72B0", lw=1.5, label="IDRNN")
    ax.plot(trials, vanillaH_sm[pi], color="#8C564B", lw=1.5, label="Vanilla+h")
    ax.plot(trials, vanilla_sm[pi],  color="#DD8452", lw=1.5, label="Vanilla")
    sid = int(subids_v[pi])
    pc1 = z_scores[pi, 0]
    pc2 = z_scores[pi, 1] if z_scores.shape[1] > 1 else np.nan
    ax.set_title(f"sub {sid}  |  PC1={pc1:+.2f}, PC2={pc2:+.2f}\n"
                 f"r_id={r_idrnn[pi]:+.2f}, r_v={r_vanilla[pi]:+.2f}",
                 fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for j in range(len(sample_idx_z), len(axes)):
    axes[j].set_visible(False)
axes[0].legend(fontsize=8, loc="lower right")
fig.supxlabel("Trial", fontsize=11)
fig.supylabel(f"Mean reward (rolling window={SMOOTH_W})", fontsize=11)
fig.suptitle(
    f"Sampled individual learning curves (stratified along z-space PC1)\n"
    f"PC1 explained variance = {ev[0]:.2%}, "
    f"all-subjects step-1 seed {BEST_SEED}, {N_RNG_SEEDS} rollouts/subject",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_learning_curves_zspace.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 4: z-space scatter colored by IDRNN advantage ───────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))
adv_for_plot = np.where(np.isfinite(adv), adv, np.nan)
y_axis = z_scores[:, 1] if z_scores.shape[1] > 1 else np.zeros(len(z_scores))
sc = ax.scatter(z_scores[:, 0], y_axis,
                c=adv_for_plot, cmap="coolwarm", s=35, alpha=0.7,
                edgecolors="black", linewidths=0.3, zorder=3)
ax.scatter(z_scores[sample_idx_z, 0],
           y_axis[sample_idx_z],
           facecolors="none", edgecolors="black", s=160, lw=1.5, zorder=4,
           label="sampled (z-PC1 strata)")
ax.scatter(z_scores[sample_idx, 0],
           y_axis[sample_idx],
           facecolors="none", edgecolors="goldenrod", s=220, lw=1.5, zorder=5,
           label="sampled (advantage strata)")
plt.colorbar(sc, ax=ax, label="IDRNN advantage (r_idrnn − r_vanilla)")
ax.set_xlabel(f"z-PC1 ({ev[0]:.1%})", fontsize=11)
ax.set_ylabel(f"z-PC2 ({ev[1]:.1%})" if z_scores.shape[1] > 1 else "(constant)",
              fontsize=11)
ax.set_title("z-space PCA, colored by IDRNN advantage",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_zspace_advantage_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Across-participant variance trajectory: σ(t) = std across subjects at each
# trial. Asks: do the model groups REPRODUCE the temporal pattern of human
# heterogeneity (e.g. divergence into high/low performers as trials accumulate)?
# Computed on the *smoothed* curves so all four groups are on equal noise
# footing (single-realization human curve smoothed the same way as the
# already-averaged model curves).
# ══════════════════════════════════════════════════════════════════════════════
print("\nAcross-participant variance trajectory analysis")

def _std_across_subjects(curves):
    """Per-trial std across participants (ignores NaN rows)."""
    return np.nanstd(curves, axis=0, ddof=1)

def _bootstrap_std_ci(curves, n_boot=500, seed=0):
    """Bootstrap participants to get (low, high) per-trial std bands."""
    valid = curves[np.isfinite(curves[:, 0])]
    n, T = valid.shape
    rng_b = np.random.default_rng(seed)
    boots = np.zeros((n_boot, T))
    for b in range(n_boot):
        idx = rng_b.integers(0, n, size=n)
        boots[b] = valid[idx].std(axis=0, ddof=1)
    lo = np.percentile(boots, 2.5,  axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return lo, hi

# Use smoothed curves to put humans and models on equal noise footing
human_std    = _std_across_subjects(human_sm)
idrnn_std    = _std_across_subjects(idrnn_sm)
vanillaH_std = _std_across_subjects(vanillaH_sm)
vanilla_std  = _std_across_subjects(vanilla_sm)

human_ci    = _bootstrap_std_ci(human_sm,    seed=1)
idrnn_ci    = _bootstrap_std_ci(idrnn_sm,    seed=2)
vanillaH_ci = _bootstrap_std_ci(vanillaH_sm, seed=3)
vanilla_ci  = _bootstrap_std_ci(vanilla_sm,  seed=4)

# Temporal-shape correlations: does model variance trajectory track humans?
def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan
    return stats.pearsonr(a[m], b[m])

r_id_var, p_id_var = _corr(human_std, idrnn_std)
r_vh_var, p_vh_var = _corr(human_std, vanillaH_std)
r_v_var,  p_v_var  = _corr(human_std, vanilla_std)

print(f"  Pearson r between human σ(t) and model σ(t):")
print(f"    IDRNN     : r={r_id_var:+.3f}, p={p_id_var:.2e}")
print(f"    Vanilla+h : r={r_vh_var:+.3f}, p={p_vh_var:.2e}")
print(f"    Vanilla   : r={r_v_var:+.3f}, p={p_v_var:.2e}")

# Bootstrap test on the *between-participant* std within each window.
# This matches what the plot shows (per-trial std across subjects, averaged
# over the window), unlike Levene's which would conflate within- and
# between-subject variability.
WINDOWS_VAR = [
    ("Early (1–50)",    0, 50),
    ("Mid (75–125)",   75, 125),
    ("Late (151–200)",151, 200),
]
N_BOOT = 1000

def _window_between_subj_std(curves, lo, hi, idx=None):
    """Mean over the window of per-trial across-subject std (after smoothing).
    idx: optional bootstrap subject-index array (same set per call to keep
    bootstrap subjects matched across all four groups)."""
    valid_mask = np.isfinite(curves[:, 0])
    valid = curves[valid_mask]
    if idx is None:
        sub = valid
    else:
        sub = valid[idx]
    sd_per_trial = np.std(sub[:, lo:hi], axis=0, ddof=1)
    return float(np.nanmean(sd_per_trial))

def _bootstrap_window_std(group_curves, lo, hi, n_boot=N_BOOT, seed=0):
    """Bootstrap subjects, return n_boot estimates of the window-mean
    across-subject std for each group, with subjects matched across groups."""
    # Use the intersection of valid subjects across all groups so bootstrap
    # samples the same set of subjects for each group.
    valid = np.ones(group_curves[0].shape[0], dtype=bool)
    for c in group_curves:
        valid &= np.isfinite(c[:, 0])
    n = int(valid.sum())
    rng = np.random.default_rng(seed)
    boots = np.zeros((len(group_curves), n_boot))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for gi, c in enumerate(group_curves):
            sub = c[valid][idx]
            sd_per_trial = np.std(sub[:, lo:hi], axis=0, ddof=1)
            boots[gi, b] = float(np.mean(sd_per_trial))
    return boots  # (n_groups, n_boot)

print(f"\n  Bootstrap test on between-participant σ within window")
print(f"  ({N_BOOT} resamples, subjects matched across groups)")
group_curves = [human_sm, idrnn_sm, vanillaH_sm, vanilla_sm]
group_names  = ["Human", "IDRNN", "Vanilla+h", "Vanilla"]
boot_records = []
for wlabel, lo_t, hi_t in WINDOWS_VAR:
    point = [_window_between_subj_std(c, lo_t, hi_t) for c in group_curves]
    boots = _bootstrap_window_std(group_curves, lo_t, hi_t, seed=hash(wlabel) & 0xFFFF)
    cis = np.percentile(boots, [2.5, 97.5], axis=1)

    print(f"\n  {wlabel}:")
    for gi, gname in enumerate(group_names):
        print(f"    σ̄[{gname:<10s}] = {point[gi]:5.2f}  "
              f"95% CI [{cis[0, gi]:.2f}, {cis[1, gi]:.2f}]")

    # Bootstrap one-sided p-values: P(σ_model < σ_human)
    # — does the model UNDER-estimate human variance?
    sd_h = boots[0]
    for gi in range(1, 4):
        sd_m = boots[gi]
        p_under = float((sd_m < sd_h).mean())
        diff = sd_m - sd_h
        p_eq  = 2 * min((diff < 0).mean(), (diff > 0).mean())  # two-sided
        boot_records.append({
            "window": wlabel, "model": group_names[gi],
            "sd_human": point[0], "sd_model": point[gi],
            "ratio_to_human": point[gi] / point[0] if point[0] > 0 else np.nan,
            "p_two_sided": p_eq,
            "p_model<human": p_under,
        })
        print(f"      {group_names[gi]:<10s} vs Human: "
              f"ratio σ_m/σ_h = {point[gi] / point[0]:.2f}  |  "
              f"P(σ_m < σ_h) = {p_under:.3f}   "
              f"two-sided p = {p_eq:.3f}")

    # IDRNN vs Vanilla: does IDRNN better recover human variance?
    # Use |gap_idrnn| < |gap_vanilla| as the "better recovery" event
    sd_i, sd_v = boots[1], boots[3]
    gap_i = np.abs(sd_i - sd_h)
    gap_v = np.abs(sd_v - sd_h)
    p_better = float((gap_i < gap_v).mean())
    print(f"      IDRNN closer to human than Vanilla : "
          f"P(|σ_IDRNN−σ_H| < |σ_Vanilla−σ_H|) = {p_better:.3f}")

# ── Plot: variance trajectory ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
panels = [
    ("Humans",    human_std,    human_ci,    "#666666", r_id_var),  # r unused here
    ("IDRNN",     idrnn_std,    idrnn_ci,    "#4C72B0", r_id_var),
    ("Vanilla+h", vanillaH_std, vanillaH_ci, "#8C564B", r_vh_var),
    ("Vanilla",   vanilla_std,  vanilla_ci,  "#DD8452", r_v_var),
]
for label, sd, ci, color, _ in panels:
    rsuffix = ""
    if label != "Humans":
        rv, pv = _corr(human_std, sd)
        rsuffix = f"  (r vs human σ = {rv:+.3f})"
    ax.plot(trials, sd, color=color, lw=2, label=f"{label}{rsuffix}")
    ax.fill_between(trials, ci[0], ci[1], color=color, alpha=0.15)

# Shade analysis windows
for wlabel, lo_t, hi_t in WINDOWS_VAR:
    ax.axvspan(lo_t, hi_t, color="grey", alpha=0.05, zorder=0)

ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("σ across participants (smoothed)", fontsize=11)
ax.set_title("Per-trial across-participant std\n"
             "shaded bands: 95% bootstrap CI",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right panel: σ ratio (model / human) over time, makes scale differences
# easier to see
ax = axes[1]
def _safe_ratio(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    ok = (den > 0)
    out[ok] = num[ok] / den[ok]
    return out

ax.plot(trials, _safe_ratio(idrnn_std,    human_std), color="#4C72B0", lw=2,
        label="IDRNN / Human")
ax.plot(trials, _safe_ratio(vanillaH_std, human_std), color="#8C564B", lw=2,
        label="Vanilla+h / Human")
ax.plot(trials, _safe_ratio(vanilla_std,  human_std), color="#DD8452", lw=2,
        label="Vanilla / Human")
ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6, label="= Human (no gap)")
for wlabel, lo_t, hi_t in WINDOWS_VAR:
    ax.axvspan(lo_t, hi_t, color="grey", alpha=0.05, zorder=0)
ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("σ(model) / σ(human)", fontsize=11)
ax.set_title("Variance ratio over time\n"
             "1.0 = same heterogeneity as humans",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="best")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    "Across-participant variance trajectory: humans vs IDRNN vs Vanilla(+h)",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_variance_trajectory.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Per-subject regression: m_human(s) ~ m_vanilla(s) + m_IDRNN(s)
#   - Each subject contributes ONE observation per window (window-mean reward)
#   - Vanilla's per-subject mean is driven only by env easiness (no z) → "env"
#   - IDRNN's per-subject mean adds z-driven policy structure
#   - If β_IDRNN > 0 after controlling for vanilla, IDRNN encodes policy
#     signal beyond what env easiness alone explains.
#
# 236 independent observations per fit (no time-series autocorrelation), so
# OLS standard errors are valid. We additionally subject-bootstrap to
# corroborate.
# ══════════════════════════════════════════════════════════════════════════════
print("\nPer-subject regression: m_human(s) ~ m_vanilla (env) + m_IDRNN (env + policy)")

try:
    import statsmodels.api as sm
    _have_sm = True
except ImportError:
    _have_sm = False
    print("  (statsmodels not available — using numpy + bootstrap only)")

def _r2_lr(X, y_):
    """R² from numpy least squares (intercept already in X)."""
    coef, *_ = np.linalg.lstsq(X, y_, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y_ - pred) ** 2))
    ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
    return coef, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

WINDOWS_REG = [
    ("Early (1–50)",     0, 50),
    ("Mid (75–125)",    75, 125),
    ("Late (151–200)", 151, 200),
    ("Full (1–200)",     0, N_TRIALS),
]
N_BOOT_REG = 1000
reg_records = []

for wlabel, lo_t, hi_t in WINDOWS_REG:
    m_h = np.nanmean(human_curves[:,    lo_t:hi_t], axis=1)
    m_i = np.nanmean(idrnn_curves[:,    lo_t:hi_t], axis=1)
    m_v = np.nanmean(vanilla_curves[:,  lo_t:hi_t], axis=1)
    m_vh = np.nanmean(vanilla_h_curves[:,lo_t:hi_t], axis=1)
    valid = (np.isfinite(m_h) & np.isfinite(m_i) &
             np.isfinite(m_v) & np.isfinite(m_vh))
    h_w, i_w, v_w, vh_w = m_h[valid], m_i[valid], m_v[valid], m_vh[valid]
    n = len(h_w)

    # Diagnostic: across-subject std for each group, plus the analytic ceiling
    # for R²(m_h ~ m_v) when m_v is a pure env-only signal independent of skill
    sd_h = float(np.std(h_w, ddof=1))
    sd_i = float(np.std(i_w, ddof=1))
    sd_v = float(np.std(v_w, ddof=1))
    var_ratio_v_to_h = (sd_v / sd_h) ** 2 if sd_h > 0 else np.nan
    var_ratio_i_to_h = (sd_i / sd_h) ** 2 if sd_h > 0 else np.nan
    print(f"\n  {wlabel}   diagnostic: across-subject σ:  "
          f"σ_h={sd_h:5.2f}  σ_i={sd_i:5.2f}  σ_v={sd_v:5.2f}")
    print(f"    Ceiling for R²(env only)  = Var(m_v)/Var(m_h) = "
          f"{var_ratio_v_to_h:.4f}")
    print(f"    Ceiling for R²(IDRNN only)= Var(m_i)/Var(m_h) = "
          f"{var_ratio_i_to_h:.4f}")

    X_env  = np.column_stack([np.ones(n), v_w])
    X_id   = np.column_stack([np.ones(n), i_w])
    X_full = np.column_stack([np.ones(n), v_w, i_w])
    c_env,  r2_env  = _r2_lr(X_env,  h_w)
    c_id,   r2_id   = _r2_lr(X_id,   h_w)
    c_full, r2_full = _r2_lr(X_full, h_w)
    inc = r2_full - r2_env

    # Analytic p-values (OLS, classical)
    if _have_sm:
        fit_env  = sm.OLS(h_w, X_env ).fit()
        fit_id   = sm.OLS(h_w, X_id  ).fit()
        fit_full = sm.OLS(h_w, X_full).fit()
        p_b_env_in_env  = float(fit_env.pvalues[1])
        p_b_env_in_full = float(fit_full.pvalues[1])
        p_b_id_in_full  = float(fit_full.pvalues[2])
        # Nested F-test: does adding m_i to env-only model help?
        F, p_F = float("nan"), float("nan")
        try:
            from scipy.stats import f as f_dist_
            dfn = 1
            dfd = n - 3
            rss_red  = float(np.sum(fit_env.resid ** 2))
            rss_full = float(np.sum(fit_full.resid ** 2))
            F = ((rss_red - rss_full) / dfn) / (rss_full / dfd)
            p_F = float(f_dist_.sf(F, dfn, dfd))
        except Exception:
            pass
    else:
        p_b_env_in_env  = p_b_env_in_full = p_b_id_in_full = np.nan
        F = p_F = np.nan

    # Bootstrap subjects (with replacement)
    rng_b = np.random.default_rng(hash(wlabel) & 0xFFFF)
    r2_env_b   = np.zeros(N_BOOT_REG)
    r2_id_b    = np.zeros(N_BOOT_REG)
    r2_full_b  = np.zeros(N_BOOT_REG)
    beta_v_b   = np.zeros(N_BOOT_REG)
    beta_id_b  = np.zeros(N_BOOT_REG)
    for b in range(N_BOOT_REG):
        idx = rng_b.integers(0, n, size=n)
        hh, ii, vv = h_w[idx], i_w[idx], v_w[idx]
        Xe = np.column_stack([np.ones(n), vv])
        Xi = np.column_stack([np.ones(n), ii])
        Xf = np.column_stack([np.ones(n), vv, ii])
        _,   r2_env_b[b]  = _r2_lr(Xe, hh)
        _,   r2_id_b[b]   = _r2_lr(Xi, hh)
        cb, r2_full_b[b]  = _r2_lr(Xf, hh)
        beta_v_b[b]  = cb[1]
        beta_id_b[b] = cb[2]
    inc_b = r2_full_b - r2_env_b
    ci_env  = np.percentile(r2_env_b,  [2.5, 97.5])
    ci_id   = np.percentile(r2_id_b,   [2.5, 97.5])
    ci_full = np.percentile(r2_full_b, [2.5, 97.5])
    ci_inc  = np.percentile(inc_b,     [2.5, 97.5])
    ci_b_v  = np.percentile(beta_v_b,  [2.5, 97.5])
    ci_b_id = np.percentile(beta_id_b, [2.5, 97.5])
    p_inc_le0 = float((inc_b <= 0).mean())
    p_b_id_le0 = float((beta_id_b <= 0).mean())

    print(f"\n  {wlabel}   (n={n} subjects)")
    print(f"    R²(env only)  = {r2_env:.4f}  bootstrap 95% CI [{ci_env[0]:.4f}, {ci_env[1]:.4f}]"
          f"  analytic p(β_env)={p_b_env_in_env:.2e}")
    print(f"    R²(IDRNN only)= {r2_id:.4f}  bootstrap 95% CI [{ci_id[0]:.4f}, {ci_id[1]:.4f}]")
    print(f"    R²(full)      = {r2_full:.4f}  bootstrap 95% CI [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    print(f"    Δ R² (full−env)= {inc:.4f}  bootstrap 95% CI [{ci_inc[0]:.4f}, {ci_inc[1]:.4f}]"
          f"  P(ΔR²≤0)={p_inc_le0:.3f}")
    print(f"    β_env  (full): {c_full[1]:+.3f}  95% CI [{ci_b_v[0]:+.3f}, {ci_b_v[1]:+.3f}]"
          f"  analytic p={p_b_env_in_full:.2e}")
    print(f"    β_id   (full): {c_full[2]:+.3f}  95% CI [{ci_b_id[0]:+.3f}, {ci_b_id[1]:+.3f}]"
          f"  analytic p={p_b_id_in_full:.2e}  P(β_id≤0)={p_b_id_le0:.3f}")
    print(f"    Nested F-test (env vs full):  F={F:.3f}, p={p_F:.2e}")

    reg_records.append(dict(
        window=wlabel, lo=lo_t, hi=hi_t, n=n,
        m_h=h_w, m_i=i_w, m_v=v_w,
        r2_env=r2_env, r2_id=r2_id, r2_full=r2_full, incremental=inc,
        ci_env=ci_env, ci_id=ci_id, ci_full=ci_full, ci_inc=ci_inc,
        beta_env=c_full[1], beta_id=c_full[2],
        ci_b_v=ci_b_v, ci_b_id=ci_b_id,
        p_b_env=p_b_env_in_full, p_b_id=p_b_id_in_full,
        F=F, p_F=p_F, p_inc_le0=p_inc_le0, p_b_id_le0=p_b_id_le0,
        coef_env_only=c_env, coef_full=c_full,
    ))

# ── Plot: added-variable plot (full task) + bar chart summary ────────────────
# Restricted to the Full window for the manuscript figure.
rec_full = next(r for r in reg_records if r["window"].startswith("Full"))
fig, axes = plt.subplots(2, 1, figsize=(6, 9))

h_w, i_w, v_w = rec_full["m_h"], rec_full["m_i"], rec_full["m_v"]
n = rec_full["n"]

# Top: added-variable plot — residualize against vanilla
ax = axes[0]
c_env_only = rec_full["coef_env_only"]
res_h = h_w - (c_env_only[0] + c_env_only[1] * v_w)
Xe = np.column_stack([np.ones(n), v_w])
coef_iv, _ = _r2_lr(Xe, i_w)
res_i = i_w - (coef_iv[0] + coef_iv[1] * v_w)

ax.scatter(res_i, res_h, color="#4C72B0", s=22, alpha=0.65,
           edgecolors="black", linewidths=0.3, zorder=3)
slope = rec_full["coef_full"][2]
intercept = res_h.mean() - slope * res_i.mean()
xx = np.linspace(res_i.min(), res_i.max(), 50)
ax.plot(xx, intercept + slope * xx, color="black", lw=2, ls="--",
        label=f"slope = β_id = {slope:+.3f}")
ax.set_xlabel(r"m_IDRNN residual $\perp$ m_Vanilla", fontsize=10)
ax.set_ylabel(r"m_human residual $\perp$ m_Vanilla", fontsize=10)
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Bottom: R² bar chart with bootstrap CI
ax = axes[1]
labels = ["env\nonly", "IDRNN\nonly", "full"]
vals   = [rec_full["r2_env"], rec_full["r2_id"], rec_full["r2_full"]]
cis    = np.array([rec_full["ci_env"], rec_full["ci_id"], rec_full["ci_full"]])
err    = np.array([vals - cis[:, 0], cis[:, 1] - vals])
colors_ = ["#DD8452", "#4C72B0", "#55A868"]
ax.bar(range(3), vals, yerr=err, color=colors_, capsize=5,
       edgecolor="black", linewidth=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("R²", fontsize=10)
ax.set_ylim(0, max(float(cis[:, 1].max()), 0.05) + 0.05)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_variance_regression.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Camera-ready combined figure: nll_comparison_pooled.png (left) +
# variance regression Full panels (right top/mid) + BIG5 regression with
# Bayes factors (right bottom).
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

NLL_PNG = "plots_thalmann/comparison/nll_comparison_pooled.png"
if os.path.exists(NLL_PNG):
    nll_img = mpimg.imread(NLL_PNG)

    # ── Compute BIG5_open regression with Bayes factors ──────────────────────
    # IDRNN: regression of BIG5 on raw z (emb_v).
    # Vanilla: regression of BIG5 on vanilla h (loaded fresh from the saved
    # latents file so feature dimensionality is correct).
    import pandas as _pd
    _q = _pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    _q["BIG5_open"] = _q[[f"BIG_5_{i}" for i in range(6)]].mean(axis=1)
    y_big5 = _q.reindex(subids_v)["BIG5_open"].values.astype(float)

    # Vanilla latents
    import glob as _glob
    _van_glob = _glob.glob(
        "plots_thalmann/step1_vs_vanilla/latents_vanilla_bestseed*.pt")
    v_path = max(_van_glob, key=os.path.getmtime)
    _v_saved = torch.load(v_path, map_location="cpu", weights_only=False)
    v_h_all = np.asarray(_v_saved["h"])
    v_sids  = np.asarray(_v_saved["subids"])
    _v_idx  = {int(s): i for i, s in enumerate(v_sids)}
    v_h_for_big5 = np.full((len(subids_v), v_h_all.shape[1]),
                           np.nan, dtype=np.float32)
    for i, s in enumerate(subids_v):
        if int(s) in _v_idx:
            v_h_for_big5[i] = v_h_all[_v_idx[int(s)]]

    def _insample_r2(X, y):
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        Xv, yv = X[valid], y[valid]
        n_v = len(yv)
        Xa = np.column_stack([np.ones(n_v), Xv])
        coef, *_ = np.linalg.lstsq(Xa, yv, rcond=None)
        pred = Xa @ coef
        ss_res = float(np.sum((yv - pred) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return r2, n_v, Xv.shape[1]

    def _bic_bf10(r2, n_, k):
        """BIC-approximated BF10 for an OLS regression with k predictors:
        compares the full model against the intercept-only null.
        BF10 > 1 ⇒ evidence for alt; BF10 < 1 ⇒ evidence for null."""
        delta_BIC = n_ * np.log(max(1 - r2, 1e-300)) + k * np.log(n_)
        return float(np.exp(-delta_BIC / 2))

    r2_id_big5, n_id, k_id = _insample_r2(emb_v, y_big5)
    r2_v_big5,  n_v_, k_v  = _insample_r2(v_h_for_big5, y_big5)
    bf_id = _bic_bf10(r2_id_big5, n_id, k_id)
    bf_v  = _bic_bf10(r2_v_big5,  n_v_, k_v)
    r_id_big5 = float(np.sqrt(max(r2_id_big5, 0)))
    r_v_big5  = float(np.sqrt(max(r2_v_big5,  0)))

    print(f"\nBIG5_open decoding (BIC Bayes factors):")
    print(f"  IDRNN (k={k_id}): r=√R²={r_id_big5:.3f}, R²={r2_id_big5:.3f}, "
          f"BF10={bf_id:.2e}")
    print(f"  Vanilla (k={k_v}): r=√R²={r_v_big5:.3f}, R²={r2_v_big5:.3f}, "
          f"BF10={bf_v:.2e}")

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.0, 1.0],
                  hspace=0.40, wspace=0.20)

    # Left: NLL spanning all rows
    ax_nll = fig.add_subplot(gs[:, 0])
    ax_nll.imshow(nll_img)
    ax_nll.axis("off")

    # Right top: added-variable plot for the Full window
    rec = next(r for r in reg_records if r["window"].startswith("Full"))
    h_w, i_w, v_w = rec["m_h"], rec["m_i"], rec["m_v"]
    n = rec["n"]
    c_env_only = rec["coef_env_only"]
    res_h = h_w - (c_env_only[0] + c_env_only[1] * v_w)
    Xe = np.column_stack([np.ones(n), v_w])
    coef_iv, _ = _r2_lr(Xe, i_w)
    res_i = i_w - (coef_iv[0] + coef_iv[1] * v_w)
    slope = rec["coef_full"][2]
    intercept = res_h.mean() - slope * res_i.mean()
    xx = np.linspace(res_i.min(), res_i.max(), 50)

    ax_av = fig.add_subplot(gs[0, 1])
    ax_av.scatter(res_i, res_h, color="#4C72B0", s=18, alpha=0.65,
                  edgecolors="black", linewidths=0.3, zorder=3)
    ax_av.plot(xx, intercept + slope * xx, color="black", lw=2, ls="--",
               label=f"slope = β_id = {slope:+.3f}")
    ax_av.set_xlabel(r"m_IDRNN residual $\perp$ m_Vanilla", fontsize=10)
    ax_av.set_ylabel(r"m_human residual $\perp$ m_Vanilla", fontsize=10)
    ax_av.legend(fontsize=8)
    ax_av.spines["top"].set_visible(False)
    ax_av.spines["right"].set_visible(False)

    # Right middle: R² bar chart from variance regression
    ax_bar = fig.add_subplot(gs[1, 1])
    labels = ["env\nonly", "IDRNN\nonly", "full"]
    vals = [rec["r2_env"], rec["r2_id"], rec["r2_full"]]
    cis = np.array([rec["ci_env"], rec["ci_id"], rec["ci_full"]])
    err = np.array([vals - cis[:, 0], cis[:, 1] - vals])
    colors_ = ["#DD8452", "#4C72B0", "#55A868"]
    ax_bar.bar(range(3), vals, yerr=err, color=colors_, capsize=5,
               edgecolor="black", linewidth=0.6)
    ax_bar.set_xticks(range(3))
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_ylabel("R²", fontsize=10)
    ax_bar.set_ylim(0, max(float(cis[:, 1].max()), 0.05) + 0.05)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.3, zorder=0)

    # Right bottom: BIG5_open r for IDRNN vs Vanilla, BF10 annotated
    ax_big5 = fig.add_subplot(gs[2, 1])
    big5_vals   = [r_id_big5, r_v_big5]
    big5_colors = ["#4C72B0", "#DD8452"]
    big5_labels = ["IDRNN", "Vanilla"]
    ax_big5.bar(range(2), big5_vals, color=big5_colors,
                edgecolor="black", linewidth=0.6)
    ax_big5.set_xticks(range(2))
    ax_big5.set_xticklabels(big5_labels, fontsize=10)
    ax_big5.set_ylabel(r"BIG5-openness $r$ (= $\sqrt{R^2}$)", fontsize=10)
    # Annotations: BF10 above each bar
    for i, (v, bf) in enumerate(zip(big5_vals, [bf_id, bf_v])):
        if bf >= 100 or bf < 0.01:
            bf_str = f"BF$_{{10}}$ = {bf:.1e}"
        else:
            bf_str = f"BF$_{{10}}$ = {bf:.2f}"
        ax_big5.text(i, v + 0.01, bf_str, ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    ax_big5.set_ylim(0, max(big5_vals) * 1.35 + 0.02)
    ax_big5.spines["top"].set_visible(False)
    ax_big5.spines["right"].set_visible(False)
    ax_big5.grid(axis="y", alpha=0.3, zorder=0)

    out = os.path.join(PLOT_DIR, "comparison",
                       "nll_and_variance_regression.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
else:
    print(f"  (skipped combined figure — {NLL_PNG} not found)")

# ══════════════════════════════════════════════════════════════════════════════
# Decode questionnaire scales from on-policy rollout means.
# Rationale: z encodes structure that only becomes behavioural after passing
# through the trained decoder. So the rollout-mean reward is the "filtered"
# expression of z that captures only the part of z the decoder actually uses.
# Decode each scale from:
#   m_IDRNN(full)           — single feature, full-task mean reward
#   m_IDRNN(early/mid/late) — three-feature window summary
#   m_Vanilla(full)         — negative-control (env-only signal, no z)
#   raw z (LOO ridge)       — comparison with the latent-based decoding
# ══════════════════════════════════════════════════════════════════════════════
print("\nQuestionnaire decoding from rollout-mean reward")

import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression as _LR
from sklearn.model_selection import LeaveOneOut
from scipy.stats import f as _f_dist

quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
SCALES = {
    "PANAS_PA":   [f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]],
    "PANAS_NA":   [f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]],
    "STICSA":     [f"STICSA_{i}" for i in range(22)],
    "PHQ":        [f"PHQ_9_{i}"  for i in range(10)],
    "CEI":        [f"CEI_{i}"    for i in range(4)],
    "BIG5_open":  [f"BIG_5_{i}"  for i in range(6)],
    "Motiv_slot": ["motiv_slot_0"],
    "Motiv_mem":  ["motiv_mem_0"],
}
for name, items in SCALES.items():
    quest[name] = quest[items].mean(axis=1)
scale_names = list(SCALES.keys())

def _wmean(curves, lo, hi):
    return np.nanmean(curves[:, lo:hi], axis=1)

# Per-subject features (aligned to subids_v order)
m_i_full  = _wmean(idrnn_curves,    0, N_TRIALS)
m_i_early = _wmean(idrnn_curves,    0, 50)
m_i_mid   = _wmean(idrnn_curves,   75, 125)
m_i_late  = _wmean(idrnn_curves,  151, N_TRIALS)
m_v_full  = _wmean(vanilla_curves,  0, N_TRIALS)

def _insample_ols(X_no_intercept, y):
    """Returns (multiple r = sqrt(R²), F-test p-value, R²) where X has no
    intercept column (we add one)."""
    n = len(y)
    p = X_no_intercept.shape[1]
    X = np.column_stack([np.ones(n), X_no_intercept])
    clf = _LR(fit_intercept=False).fit(X, y)
    pred = clf.predict(X)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dfn, dfd = p, n - p - 1
    if r2 < 1 and dfd > 0:
        F = (r2 / dfn) / ((1 - r2) / dfd)
        p_val = float(_f_dist.sf(F, dfn, dfd))
    else:
        p_val = np.nan
    return float(np.sqrt(max(r2, 0))), p_val, float(r2)

def _loo_ridge(X_no_intercept, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    """LOO RidgeCV with per-fold standardization. Returns (Pearson r, p)."""
    loo = LeaveOneOut()
    if X_no_intercept.ndim == 1:
        X_no_intercept = X_no_intercept[:, None]
    n = len(y)
    preds = np.zeros(n)
    for tr, te in loo.split(X_no_intercept):
        Xtr = X_no_intercept[tr]; ytr = y[tr]
        mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-8
        mu_y, sd_y = ytr.mean(),  ytr.std() + 1e-8
        clf = RidgeCV(alphas=list(alphas))
        clf.fit((Xtr - mu_x) / sd_x, (ytr - mu_y) / sd_y)
        preds[te] = clf.predict(
            (X_no_intercept[te] - mu_x) / sd_x) * sd_y + mu_y
    r, p = stats.pearsonr(preds, y)
    return float(r), float(p)

# Build the feature sets for each subject (one column per feature)
feature_sets = [
    ("mIDRNN(full)",        np.column_stack([m_i_full])),
    ("mIDRNN(early|mid|late)", np.column_stack([m_i_early, m_i_mid, m_i_late])),
    ("mVanilla(full)",      np.column_stack([m_v_full])),
    ("mV + mI(full)",       np.column_stack([m_v_full, m_i_full])),
    ("raw z (latents)",     emb_v),
]

# Run decoding per scale × feature set
print(f"{'Scale':<11}{'Features':<24}{'n':>4}  "
      f"{'r_in':>7}  {'p_in':>9}  {'R²_in':>7}  "
      f"{'r_LOO':>7}  {'p_LOO':>9}")
print("-" * 86)
qdecode_records = []
for scale in scale_names:
    y_all = quest.reindex(subids_v)[scale].values.astype(float)
    for fname, X in feature_sets:
        valid = (np.isfinite(y_all) &
                 np.all(np.isfinite(X), axis=1) if X.ndim > 1
                 else np.isfinite(y_all) & np.isfinite(X.ravel()))
        if valid.sum() < 20:
            continue
        X_v = X[valid]; y_v = y_all[valid]
        r_in, p_in, r2_in = _insample_ols(X_v, y_v)
        r_lo, p_lo        = _loo_ridge   (X_v, y_v)
        qdecode_records.append(dict(
            scale=scale, features=fname, n=int(valid.sum()),
            r_in=r_in, p_in=p_in, r2_in=r2_in,
            r_loo=r_lo, p_loo=p_lo,
        ))
        print(f"{scale:<11}{fname:<24}{int(valid.sum()):>4}  "
              f"{r_in:>+7.3f}  {p_in:>9.2e}  {r2_in:>+7.3f}  "
              f"{r_lo:>+7.3f}  {p_lo:>9.2e}")

qdf = pd.DataFrame(qdecode_records)
csv_out = os.path.join(PLOT_DIR, "step1_rollout_questionnaire_decoding.csv")
qdf.to_csv(csv_out, index=False)
print(f"Saved → {csv_out}")

# ── Plot: bar chart comparing decoders per scale ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
feat_order = ["mIDRNN(full)", "mIDRNN(early|mid|late)",
              "mVanilla(full)", "raw z (latents)"]
feat_colors = {
    "mIDRNN(full)":            "#4C72B0",
    "mIDRNN(early|mid|late)":  "#3D5A80",
    "mVanilla(full)":          "#DD8452",
    "mV + mI(full)":           "#55A868",
    "raw z (latents)":         "#8C564B",
}

for ax, metric_col, ptitle in [
    (axes[0], "r_in",  "In-sample r (= √R²)"),
    (axes[1], "r_loo", "LOO RidgeCV r"),
]:
    x = np.arange(len(scale_names))
    w = 0.18
    for ki, feat in enumerate(feat_order):
        vals = []
        for sc in scale_names:
            row = qdf[(qdf["scale"] == sc) & (qdf["features"] == feat)]
            vals.append(float(row[metric_col].iloc[0]) if len(row) else np.nan)
        vals = np.array(vals)
        offset = (ki - 1.5) * w
        bars = ax.bar(x + offset, vals, w, color=feat_colors[feat],
                      edgecolor="black", linewidth=0.4, label=feat)
        # significance stars from analytic p-values
        for i, sc in enumerate(scale_names):
            row = qdf[(qdf["scale"] == sc) & (qdf["features"] == feat)]
            if not len(row): continue
            p_col = "p_in" if metric_col == "r_in" else "p_loo"
            p = float(row[p_col].iloc[0])
            s = ("***" if p < 0.001 else "**" if p < 0.01
                 else "*" if p < 0.05 else "")
            if s:
                ax.text(x[i] + offset, vals[i] + 0.005, s,
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(scale_names, fontsize=9, rotation=20)
    ax.set_ylabel("r", fontsize=11)
    ax.set_title(ptitle, fontsize=11, fontweight="bold")
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    "Questionnaire decoding from rollout-mean reward vs raw latents\n"
    "Same model — different readouts of z's behavioural content",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_rollout_questionnaire_decoding.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

print("\nDone.")
