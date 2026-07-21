"""
plot_publication_figure.py — Camera-ready 2x2 composite figure
(Nature Human Behaviour style) summarising the synthetic-data study.

Panel a — Aggregated model NLL across 20 datasets, BFs annotated.
Panel b — Aggregated RSA correlation (IDRNN vs Vanilla), BF annotated.
Panel c — Alpha vs PC1 of step-2 latents for dataset 0.
Panel d — Environment-decoding accuracy (train); IDRNN carries the least env info.
Panel e — Three-regression bars for dataset 0 (R1/R2 → true α).

Saves:
  plots/publication_figure.png
  plots/publication_figure.pdf
"""
import os, json, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from sklearn.decomposition import PCA
import plot_functions as plf
from pingouin import bayesfactor_ttest, bayesfactor_pearson
from modelsandtraining import vectorize_rsa


# ── Config ─────────────────────────────────────────────────────────────────
N_DATASETS = 20
SEEDS      = [12, 50, 76, 100, 142]
MIN_EPOCH, MAX_EPOCH = 1000, 3000
OUT_DIR    = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


# Style: Nature HB-ish
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    9.5,
    "axes.titlesize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def fmt_bf(bf):
    """Format BF_10 (number only — 'BF₁₀=' label belongs in the figure caption)."""
    if not np.isfinite(bf):
        return "—"
    if bf >= 1000:
        return f"{bf:.1e}"
    elif bf >= 10:
        return f"{bf:.1f}"
    else:
        return f"{bf:.2f}"


# ── Load per-dataset NLL + RSA arrays ──────────────────────────────────────
def load_nll(dataset_id):
    data_dir = f"data_dataset{dataset_id}"
    try:
        df_eval = pd.read_csv(f"{data_dir}/model_eval_dfvanilla.csv")
        rnn_id  = pd.read_csv(f"{data_dir}/rnn_resultslatentmodel.csv")
        rnn_cp  = pd.read_csv(f"{data_dir}/rnn_results_common_process.csv")
        rnn_van = pd.read_csv(f"{data_dir}/rnn_resultsvanilla.csv")
    except FileNotFoundError as e:
        print(f"  Skipping ds {dataset_id}: {e}")
        return None

    def m(df, label, col="model"):
        v = df[df[col] == label]["normalized_likelihood"].values
        return float(np.mean(v)) if len(v) else np.nan

    return {
        "Q_CP":     m(df_eval, "Q (common fit)"),
        "Q_EM":     m(df_eval, "Q (MAP)"),
        "FQ_CP":    m(df_eval, "FQ (common fit)"),
        "FQ_EM":    m(df_eval, "FQ (MAP)"),
        "True":     m(df_eval, "True model"),
        "RNN_CP":   m(rnn_cp,  "common_process_RNN"),
        "IDRNN":    m(rnn_id,  "IDRNN"),
        "Vanilla":  m(rnn_van, "vanillaRNN"),
    }


def load_best_seed_epoch(json_path):
    if not os.path.exists(json_path):
        return None, None
    with open(json_path) as f:
        d = json.load(f)
    return d.get("best_seed"), d.get("best_epoch")


def find_vanilla_best(runs_vanilla_dir):
    """Lowest loss within [MIN_EPOCH, MAX_EPOCH] across SEEDS."""
    best = (None, None, float("inf"))
    for seed in SEEDS:
        loss_dir = os.path.join(runs_vanilla_dir, f"seed_{seed}", "loss")
        if not os.path.exists(loss_dir): continue
        for f in glob.glob(os.path.join(loss_dir, "epoch_*.npy")):
            ep = int(os.path.basename(f).replace("epoch_", "").replace(".npy", ""))
            if ep < MIN_EPOCH or ep > MAX_EPOCH: continue
            try:
                loss = float(np.load(f))
                if loss < best[2]:
                    best = (seed, ep, loss)
            except Exception:
                continue
    return best[0], best[1]


def load_rsa_corr(dataset_id):
    """Pearson r between RSA(latents) and RSA(alpha) for the best-epoch model."""
    data_dir = f"data_dataset{dataset_id}"
    runs_dir = f"runs_dataset{dataset_id}"
    runs_v   = f"runs_vanilla_dataset{dataset_id}"

    # vec_params from true test alpha
    try:
        params = pd.read_csv(f"{data_dir}/true_test_parameter_values.csv")["alphaP_list"].values
    except FileNotFoundError:
        return None, None
    dist_p, _ = plf.rsa_latents(latents=params, metric="euclidean", title="p",
                                 reduction="entire", plot=False, original_data=True,
                                 cluster_order=False)
    vec_p = vectorize_rsa(dist_p)

    # Best epoch/seed for latent (from JSON) and vanilla (via lowest loss)
    lat_seed, lat_ep = load_best_seed_epoch(
        os.path.join(runs_dir, "best_epoch_by_specificity.json"))
    van_seed, van_ep = find_vanilla_best(runs_v)

    def corr(base, seed, ep):
        if seed is None: return np.nan
        rsa_file = os.path.join(base, f"seed_{seed}", "rsa", f"epoch_{ep}.npy")
        try:
            vec = np.load(rsa_file)
            return float(np.corrcoef(vec_p, vec)[0, 1])
        except FileNotFoundError:
            return np.nan

    return corr(runs_dir, lat_seed, lat_ep), corr(runs_v, van_seed, van_ep)


print("Loading per-dataset NLLs ...")
nll_records = []
for d in range(N_DATASETS):
    rec = load_nll(d)
    if rec is not None:
        nll_records.append(rec)
nll_df = pd.DataFrame(nll_records)
print(f"  loaded {len(nll_df)} datasets; NLL means:")
print(nll_df.mean().to_string())

print("\nLoading per-dataset RSA correlations ...")
rsa_records = []
for d in range(N_DATASETS):
    lat_r, van_r = load_rsa_corr(d)
    if np.isfinite(lat_r) and np.isfinite(van_r):
        rsa_records.append({"IDRNN": lat_r, "Vanilla": van_r})
rsa_df = pd.DataFrame(rsa_records)
print(f"  loaded {len(rsa_df)} datasets; mean IDRNN={rsa_df.IDRNN.mean():.3f}, "
      f"mean Vanilla={rsa_df.Vanilla.mean():.3f}")


# ── Compute BFs for the paired contrasts ───────────────────────────────────
def paired_bf(a, b):
    """Paired-samples BF_10 (Bayesian alternative to t-test)."""
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    t, _ = stats.ttest_rel(a[mask], b[mask])
    return float(bayesfactor_ttest(t=abs(t), nx=int(mask.sum()), paired=True))


bf_Q_pair       = paired_bf(nll_df["Q_CP"],   nll_df["Q_EM"])
bf_FQ_pair      = paired_bf(nll_df["FQ_CP"],  nll_df["FQ_EM"])
bf_RNNCP_IDRNN  = paired_bf(nll_df["RNN_CP"], nll_df["IDRNN"])
bf_Van_IDRNN    = paired_bf(nll_df["Vanilla"], nll_df["IDRNN"])
bf_RSA          = paired_bf(rsa_df["IDRNN"], rsa_df["Vanilla"])
print(f"\nPaired BFs:\n"
      f"  Ill-spec. CP vs EM     : {bf_Q_pair:.2g}\n"
      f"  Cog model CP vs EM     : {bf_FQ_pair:.2g}\n"
      f"  CP RNN vs IDRNN        : {bf_RNNCP_IDRNN:.2g}\n"
      f"  Vanilla RNN vs IDRNN   : {bf_Van_IDRNN:.2g}\n"
      f"  RSA IDRNN vs Vanilla   : {bf_RSA:.2g}")


# ── Panel C: load alpha + step-2 latents for dataset 0, PCA → PC1 ──────────
ds0_alpha = pd.read_csv("data_dataset0/true_test_parameter_values.csv")["alphaP_list"].values
ds0_lat   = torch.load("data_dataset0/latents_tensorlatentmodel.pt",
                       map_location="cpu", weights_only=False)
ds0_lat   = ds0_lat.detach().cpu().numpy() if hasattr(ds0_lat, "detach") else np.asarray(ds0_lat)
z_last    = ds0_lat[:, -1, :]                       # (200, 3)
pca       = PCA(n_components=min(2, z_last.shape[1])).fit(z_last)
z_pc1     = pca.transform(z_last)[:, 0]             # (200,)
ev1       = pca.explained_variance_ratio_[0]
r_panelC  = float(np.corrcoef(ds0_alpha, z_pc1)[0, 1])
bf_panelC = float(bayesfactor_pearson(r_panelC, len(ds0_alpha)))


# ── Panel D: three-regression bars data (training-time alphas) ─────────────
panelD_npz = np.load("plots_dataset0/step1_three_regressions.npz")
panelD_y   = pd.read_csv("data_dataset0/true_parameter_values.csv")["alphaP_list"].values
PANEL_D_BARS = [
    # (display model name, regression label, predictor key)
    ("ground truth", "GT", "R1_raw_mean_reward"),  # behavior of the data-generating Q-model
    ("IDRNN",        "R1", "R2_idrnn"),             # exact-env simulation (was R2)
    ("IDRNN",        "R2", "R3_idrnn"),             # marg-env simulation  (was R3)
    ("Vanilla",      "R1", "R2_van0"),
    ("Vanilla",      "R2", "R3_van0"),
    ("Vanilla+h",    "R1", "R2_vanH"),              # vanilla with subject's h-state init
    ("Vanilla+h",    "R2", "R3_vanH"),
]


def r_with_ci_and_boots(x, y, n_boot=2000, seed=0):
    """Pearson r + 95% bootstrap CI + n + the full bootstrap distribution."""
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]
    r0 = float(stats.pearsonr(xv, yv)[0])
    rng = np.random.default_rng(seed)
    n = len(xv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = xv[idx], yv[idx]
        boots[i] = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r0, (float(lo), float(hi)), int(m.sum()), boots


panelD_stats = []
for model, reg, pred_key in PANEL_D_BARS:
    x = np.asarray(panelD_npz[pred_key])
    r, ci, n, boots = r_with_ci_and_boots(x, panelD_y, seed=1)
    bf = float(bayesfactor_pearson(r, n))
    panelD_stats.append(dict(model=model, reg=reg, r=r, ci=ci, n=n, bf=bf,
                             boots=boots))


# ── Colours ────────────────────────────────────────────────────────────────
COL_Q_CP    = "#a19f9f"
COL_Q_EM    = "#545454"
COL_FQ_CP   = "#a4d3a2"
COL_FQ_EM   = "#3ba83b"
COL_RNN_CP  = "#a0c4e8"
COL_IDRNN   = "#2a82c2"
COL_VANILLA = "#e1861f"
COL_VAN_H   = "#8C564B"    # Vanilla+h — matches thalmann_s2_h10z3 plot
COL_TRUE    = "#222222"
COL_HUMAN   = "#666666"

MODEL_COLORS_D = {
    "ground truth": COL_FQ_EM,   # match Cog model EM in panel a
    "IDRNN":        COL_IDRNN,
    "Vanilla":      COL_VANILLA,
    "Vanilla+h":    COL_VAN_H,
}
# Sharper R1↔R2 contrast (R1 lighter, R2 full) so the two are visually distinct.
# Ground-truth bar uses full saturation; the green colour itself distinguishes it.
REG_ALPHA_D = {"GT": 1.00, "R1": 0.45, "R2": 1.00}


# ── Build the 2×2 figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(13.0, 8.8))
gs_outer = GridSpec(2, 1, figure=fig, hspace=0.25,
                    left=0.07, right=0.985, top=0.92, bottom=0.06)
gs_top = GridSpecFromSubplotSpec(1, 2, gs_outer[0],
                                  width_ratios=[1.45, 1.00], wspace=0.28)
# Right column of the top row is split into b (RSA) and d (env decoding) stacked
gs_top_right = GridSpecFromSubplotSpec(2, 1, gs_top[0, 1],
                                        height_ratios=[0.75, 1.00], hspace=0.45)
gs_bot = GridSpecFromSubplotSpec(1, 2, gs_outer[1],
                                  width_ratios=[1.00, 1.15], wspace=0.22)

# ── Panel A: NLL bars ───────────────────────────────────────────────────────
axA = fig.add_subplot(gs_top[0, 0])
labels_A = ["Ill-spec. CP", "Ill-spec. EM",
            "Cog model CP", "Cog model EM",
            "CP RNN",       "IDRNN",
            "Vanilla RNN"]
keys_A   = ["Q_CP", "Q_EM", "FQ_CP", "FQ_EM", "RNN_CP", "IDRNN", "Vanilla"]
cols_A   = [COL_Q_CP, COL_Q_EM, COL_FQ_CP, COL_FQ_EM,
            COL_RNN_CP, COL_IDRNN, COL_VANILLA]
means_A  = [nll_df[k].mean() for k in keys_A]
sems_A   = [stats.sem(nll_df[k].dropna()) for k in keys_A]

x_A = np.arange(len(labels_A))
bars_A = axA.bar(x_A, means_A, color=cols_A, alpha=0.92,
                 edgecolor="black", linewidth=0.5, zorder=2)
axA.errorbar(x_A, means_A, yerr=sems_A, fmt="none",
             ecolor="#444444", elinewidth=0.7, capsize=0, zorder=3)

# Individual dataset jitter dots — shared aesthetics across panels a, b, e
DOT_KW       = dict(c="black", alpha=0.40, zorder=4, edgecolors="none")
DOT_SIZE_AB  = 12   # panels a and b: one dot per dataset
DOT_SIZE_E   = 5    # panel e: bootstrap cloud, denser → smaller dots

rng = np.random.default_rng(0)
for i, k in enumerate(keys_A):
    v = nll_df[k].dropna().values
    if len(v) == 0: continue
    j = rng.normal(0, 0.05, size=len(v))
    axA.scatter(x_A[i] + j, v, s=DOT_SIZE_AB, **DOT_KW)

# True-model dotted line
true_mean = nll_df["True"].mean()
axA.axhline(true_mean, ls=":", color=COL_TRUE, lw=1.2, zorder=1)

# BF brackets between paired comparisons
def bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, c="black")
    ax.text((x1+x2)/2, y + h*1.1, txt, ha="center", va="bottom", fontsize=7.5)

ymax = max(means_A) + max(sems_A)
h_step = 0.5
yb = ymax + h_step
bracket(axA, 0, 1, yb,           h_step*0.3, fmt_bf(bf_Q_pair))
bracket(axA, 2, 3, yb,           h_step*0.3, fmt_bf(bf_FQ_pair))
bracket(axA, 4, 5, yb,           h_step*0.3, fmt_bf(bf_RNNCP_IDRNN))
bracket(axA, 5, 6, yb + h_step*1.3, h_step*0.3, fmt_bf(bf_Van_IDRNN))

axA.set_xticks([])
axA.set_ylabel("neg. log-likelihood per participant")
axA.set_ylim(bottom=true_mean - 2)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# Two panel-a-specific legends, stacked on the LEFT above panel a:
#   - Fit type   (CP vs EM)      — applies only to the NLL setup in panel a
#   - True model (dotted line)   — applies only to panel a (reference line)
fit_handles = [
    Patch(facecolor="#bcbcbc", edgecolor="black", label="Common process (CP)"),
    Patch(facecolor="#404040", edgecolor="black", label="Individual differences (EM)"),
]
true_handles = [Line2D([0], [0], color=COL_TRUE, ls=":", lw=1.2, label="True model")]

leg_fit = axA.legend(handles=fit_handles, loc="lower left",
                     frameon=False, fontsize=7.5, ncol=1,
                     bbox_to_anchor=(0.0, 1),
                     title="Fit type", title_fontsize=8.2,
                     handletextpad=0.5, labelspacing=0.30)
axA.add_artist(leg_fit)
axA.legend(handles=true_handles, loc="upper left",
           frameon=False, fontsize=7.5, ncol=1,
           bbox_to_anchor=(0.005, 0.99),
           handletextpad=0.5, labelspacing=0.30)


# ── Panel B: RSA bar ────────────────────────────────────────────────────────
axB = fig.add_subplot(gs_top_right[0, 0])
means_B = [rsa_df["IDRNN"].mean(), rsa_df["Vanilla"].mean()]
sems_B  = [stats.sem(rsa_df["IDRNN"]), stats.sem(rsa_df["Vanilla"])]
x_B = np.array([0.0, 0.55])     # bring bars closer together
axB.bar(x_B, means_B, color=[COL_IDRNN, COL_VANILLA], alpha=0.92,
        width=0.30, edgecolor="black", linewidth=0.5, zorder=2)
axB.errorbar(x_B, means_B, yerr=sems_B, fmt="none",
             ecolor="#444444", elinewidth=0.7, capsize=0, zorder=3)

# Individual dataset dots (one per dataset)
rng = np.random.default_rng(0)
for i, k in enumerate(["IDRNN", "Vanilla"]):
    v = rsa_df[k].dropna().values
    j = rng.normal(0, 0.035, size=len(v))
    axB.scatter(x_B[i] + j, v, s=DOT_SIZE_AB, **DOT_KW)

# BF bracket
ymax_B = max(means_B) + max(sems_B)
bracket(axB, x_B[0], x_B[1], ymax_B + 0.02, 0.008, fmt_bf(bf_RSA))

axB.set_xticks([])
axB.set_xlim(-0.30, 0.85)        # tighter x range
axB.set_ylabel("RSA r")
# Zoom in on the IDRNN vs Vanilla difference (otherwise both bars look identical)
b_vals = np.concatenate([rsa_df["IDRNN"].values, rsa_df["Vanilla"].values])
axB.set_ylim(max(0.0, b_vals.min() - 0.05), b_vals.max() + 0.08)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)


# ── Panel C: alpha vs PC1 of step-2 latents (dataset 0) ─────────────────────
axC = fig.add_subplot(gs_bot[0, 0])
axC.scatter(ds0_alpha, z_pc1, c=COL_IDRNN,
            s=24, alpha=0.85, edgecolors="black", linewidths=0.3, zorder=2)
slope, intercept = np.polyfit(ds0_alpha, z_pc1, 1)
xs = np.linspace(ds0_alpha.min(), ds0_alpha.max(), 100)
axC.plot(xs, slope*xs + intercept, color="#C44E52", ls="--", lw=1.5, zorder=3)
axC.text(0.96, 0.96,
         f"r = {r_panelC:+.3f}\nBF₁₀={fmt_bf(bf_panelC)}",
         transform=axC.transAxes, va="top", ha="right", fontsize=9,
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
axC.set_xlabel("Ground-truth α")
axC.set_ylabel(f"PC1 ({ev1*100:.1f}% var.)")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)


# ── Panel D (env decoding, training data): top-right (stacked below b) ─────
# Connected-scatter style: one grey line per (dataset, seed) traverses the
# four feature types. Lower accuracy means the feature carries less env
# information — that's what we want for a pure participant signature.
axE = fig.add_subplot(gs_top_right[1, 0])
env_df_full = pd.read_csv("env_decoding_per_seed.csv")
env_train = env_df_full[env_df_full["split"] == "train_data"]

# Four feature types — different layers in different models, so colors map
# to the corresponding "model family" colour where possible. IDRNN z and
# IDRNN h share the blue family (different shades) because both come from
# the IDRNN architecture; the h-state is the decoder's hidden state and is
# conditioned on z, hence its proximity to IDRNN z conceptually.
COL_IDRNN_H = "#6cb1de"   # lighter IDRNN-blue, for IDRNN's decoder h-state
MODELS_E = [
    ("IDRNN z",   "IDRNN_mu",           COL_IDRNN),
    ("CP RNN h",  "common_process_h",   COL_RNN_CP),
    ("Vanilla h", "vanilla_h",          COL_VANILLA),
    ("IDRNN h",   "informed_decoder_h", COL_IDRNN_H),
]
model_keys = [k for _, k, _ in MODELS_E]
model_labs = [l for l, _, _ in MODELS_E]
model_cols = [c for _, _, c in MODELS_E]
xs_E = np.arange(len(MODELS_E))

# Pivot: one row per (dataset, seed) — 50 rows × 4 features
env_pivot = env_train.pivot_table(index=["dataset_id", "seed"],
                                  columns="model",
                                  values="acc_mean").reset_index()

# Grey connecting lines, one per (dataset, seed)
for _, row in env_pivot.iterrows():
    ys = [row[k] for k in model_keys]
    axE.plot(xs_E, ys, color="grey", alpha=0.30, linewidth=0.6, zorder=1)

# Per-feature scatter coloured by model family
for i, key in enumerate(model_keys):
    ys = env_pivot[key].values
    axE.scatter([xs_E[i]] * len(ys), ys, color=model_cols[i],
                edgecolor="black", s=18, linewidth=0.4, alpha=0.85, zorder=3)

# Group medians as wide horizontal marks
for i, key in enumerate(model_keys):
    med = env_pivot[key].median()
    axE.scatter([xs_E[i]], [med], marker="_", s=380,
                color=model_cols[i], linewidth=2.5, zorder=4)

# Chance line for 3-way decoding (matches panel-a true-model line style)
axE.axhline(1/3, ls=":", color=COL_TRUE, lw=1.2, zorder=1)
axE.text(len(MODELS_E) - 0.55, 1/3 + 0.012, "chance",
         ha="right", va="bottom", fontsize=7.5, color=COL_TRUE)

# Paired BFs (IDRNN z vs each other feature) — stacked brackets that sit
# above the accuracy ceiling (y=1). Y-axis tick labels still stop at 1.0;
# brackets live in the headroom above.
bf_env = {}
for key in model_keys[1:]:
    a, b = env_pivot["IDRNN_mu"].values, env_pivot[key].values
    m = np.isfinite(a) & np.isfinite(b)
    t, _ = stats.ttest_rel(a[m], b[m])
    bf_env[key] = float(bayesfactor_ttest(t=abs(t), nx=int(m.sum()), paired=True))
for i, key in enumerate(model_keys[1:]):
    yb = 1.06 + i * 0.13
    bracket(axE, 0, i + 1, yb, 0.018, fmt_bf(bf_env[key]))

axE.set_xticks(xs_E)
axE.set_xticklabels(model_labs)
axE.set_xlim(-0.5, len(MODELS_E) - 0.5)
# Keep tick labels at the accuracy range; let brackets extend above
axE.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
axE.set_ylim(0.20, 1.06 + 0.13 * len(model_keys[1:]) + 0.06)
axE.set_ylabel("Environment decoding accuracy")
axE.spines["top"].set_visible(False)
axE.spines["right"].set_visible(False)


# ── Panel E (three-regression bars): bottom-right ─────────────────────────────
axD = fig.add_subplot(gs_bot[0, 1])
xs_D = np.arange(len(panelD_stats))
heights = [rec["r"] for rec in panelD_stats]
lo_err  = [abs(rec["r"] - rec["ci"][0]) for rec in panelD_stats]
hi_err  = [abs(rec["ci"][1] - rec["r"]) for rec in panelD_stats]
colors_D = [MODEL_COLORS_D[rec["model"]] for rec in panelD_stats]
alphas_D = [REG_ALPHA_D[rec["reg"]]      for rec in panelD_stats]
for xi, h, c, a in zip(xs_D, heights, colors_D, alphas_D):
    axD.bar(xi, h, color=c, alpha=a, edgecolor="black", linewidth=0.5, zorder=2)

# Reasonable selection of bootstrap r values overlaid on each bar
N_BOOT_PTS = 80
rng_pts = np.random.default_rng(7)
for xi, rec in zip(xs_D, panelD_stats):
    sample = rng_pts.choice(rec["boots"], size=N_BOOT_PTS, replace=False)
    j = rng_pts.normal(0, 0.045, size=N_BOOT_PTS)
    axD.scatter(xi + j, sample, s=DOT_SIZE_E, **DOT_KW)

axD.errorbar(xs_D, heights, yerr=[lo_err, hi_err],
             fmt="none", ecolor="#888888", elinewidth=0.7, capsize=0, zorder=5)

# Reserve room around the cloud of bootstrap points + BF text
all_pts = np.concatenate([r["boots"] for r in panelD_stats])
ymin = float(np.percentile(all_pts, 1.0))
ymax = float(np.percentile(all_pts, 99.0))
axD.set_ylim(min(ymin, min(heights)) - 0.10, max(ymax, max(heights)) + 0.08)

# BF annotation — above the bar for positive r, below the error bar for negative
for xi, rec in zip(xs_D, panelD_stats):
    bf_str = f"{rec['bf']:.2g}" if rec["bf"] >= 100 else f"{rec['bf']:.2f}"
    if rec["r"] >= 0:
        yloc = rec["r"] + hi_err[xi] + 0.020
        va = "bottom"
    else:
        yloc = rec["r"] - lo_err[xi] - 0.030
        va = "top"
    axD.text(xi, yloc, bf_str, ha="center", va=va, fontsize=7.8)

axD.axhline(0, color="grey", lw=0.7, ls=":")
axD.set_xticks([])
axD.set_ylabel("Pearson r (regret → true α)")
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)

# Regression legend — only R1 (exact env) and R2 (marg env); ground truth's
# green colour distinguishes it without a separate legend entry.
REG_LEGEND_LABELS = {
    "R1": "actual environment",
    "R2": "marginalization over environments",
}
reg_handles_D = [Patch(facecolor="#444444", edgecolor="black",
                       alpha=REG_ALPHA_D[r], label=REG_LEGEND_LABELS[r])
                 for r in ["R1", "R2"]]
axD.legend(handles=reg_handles_D, loc="lower left",
           frameon=False, fontsize=7.8,
           bbox_to_anchor=(0.0, 0.0),
           title="Regression", title_fontsize=8.2)


# ── Figure-level Model type legend (applies to all four panels) ────────────
# True model intentionally omitted here — it lives in the panel-a legend
# stack (top-left) because it only appears in panel a.
model_handles = [
    Patch(facecolor=COL_Q_EM,    edgecolor="black", label="Ill-specified model"),
    Patch(facecolor=COL_FQ_EM,   edgecolor="black", label="Cog. model"),
    Patch(facecolor=COL_IDRNN,   edgecolor="black", label="IDRNN"),
    Patch(facecolor=COL_VANILLA, edgecolor="black", label="Vanilla RNN"),
    Patch(facecolor=COL_VAN_H,   edgecolor="black", label="Vanilla+h"),
]
fig.legend(handles=model_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.96),
           ncol=5, frameon=False, fontsize=9,
           title="Model type", title_fontsize=10,
           columnspacing=2.0, handletextpad=0.7)


# ── Panel labels (a, b, c, d, e) ───────────────────────────────────────────
# axE (env-decoding, top-right stacked under b) is labelled "c";
# axC (α vs PC1, bottom-left) is labelled "d". axB and axC share the same
# offset so the top-right ("b") and bottom-left ("d") labels align visually.
# axA ("a") and axC ("d") sit in different-width panels (top-left is wider
# than bottom-left), so we back-compute axC's axes-x to match axA's figure-x.
SHARED_BD_OFFSET = (-0.16, 1.04)
axA_pos, axC_pos = axA.get_position(), axC.get_position()
aA_x = -0.08
target_fig_x = axA_pos.x0 + aA_x * axA_pos.width
aC_x_aligned = (target_fig_x - axC_pos.x0) / axC_pos.width

panel_label_pos = {
    axA: (aA_x, 1.06),
    axB: SHARED_BD_OFFSET,
    axE: SHARED_BD_OFFSET,             # panel c (top-right stacked under b)
    axC: (aC_x_aligned, 1.04),         # panel d — aligned with panel a in figure-x
    axD: (-0.10, 1.04),
}
for ax, label in [(axA, "a"), (axB, "b"), (axE, "c"), (axC, "d"), (axD, "e")]:
    x, y = panel_label_pos[ax]
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")


# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(OUT_DIR, "publication_figure.png")
out_pdf = os.path.join(OUT_DIR, "publication_figure.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf,         bbox_inches="tight")
print(f"\nSaved → {out_png}")
print(f"Saved → {out_pdf}")
