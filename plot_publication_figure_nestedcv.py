"""plot_publication_figure_nestedcv.py — nested-CV / z=1 / dimension-matched
recreation of plots/publication_figure.png, reusing aesthetics from the original
plot_publication_figure.py.

Panel data sources:
  a   : cog NLLs from cog_nll_perfold_synthetic.csv (per-fold refit, same held-out
        fold subjects as the RNNs; legacy model_eval_dfvanilla.csv as fallback) (per-session NLL,
        EM-fit on all 200 subjects); IDRNN/Vanilla from
        final_plots/synthetic/nested_cv_z1/dataset{D}/nested_cv_summary.json
        (per-trial NLL × 200 trials); CP RNN from cp_rnn_nll_nestedcv.csv
        (nested-CV held-out, IDRNN with z=0).
  b   : RSA r and LOO R² (dim-matched) from the same nested-CV summaries.
  c   : alpha vs IDRNN canonical z (= PC1 since z_dim=1) for dataset 0.
  d   : env_decoding_canonical.csv  (dim-matched to z_dim=1).
  e   : plots_dataset0/step1_three_regressions.npz  (R1/R2/R3 → true α,
        Pearson r with bootstrap CI; rollouts use the canonical IDRNN+Vanilla).

Outputs (PNG + PDF for everything):
  plots/publication_figure_nestedcv.{png,pdf}     — composite (RSA in panel b)
  plots/publication_figure_nestedcv_r2.{png,pdf}  — composite (LOO R² in panel b)
  plots/nestedcv_panel_{a,b,b_r2,c,d,e}.{png,pdf} — each panel standalone
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import pearsonr
try:
    from pingouin import bayesfactor_ttest, bayesfactor_pearson
except Exception:
    bayesfactor_ttest = bayesfactor_pearson = None

DATASETS = [0, 1, 2, 3, 5, 7, 10, 12, 15, 17]
T_TRIALS = 200    # synthetic sessions are 200 trials each
Z1_BASE  = "final_plots/synthetic/nested_cv_z1"
CANON_DS0 = "final_plots/synthetic_z1/dataset0/canonical/idrnn"
OUT = "plots"; os.makedirs(OUT, exist_ok=True)

# ── Palette (copied from plot_publication_figure.py) ───────────────────────
COL_Q_CP    = "#a19f9f"; COL_Q_EM    = "#545454"
COL_FQ_CP   = "#a4d3a2"; COL_FQ_EM   = "#3ba83b"
COL_RNN_CP  = "#a0c4e8"; COL_IDRNN   = "#2a82c2"; COL_VANILLA = "#e1861f"
COL_VAN_H   = "#8C564B"   # Vanilla+h (time-mean h), matches plot_publication_figure.py
COL_VAN_H_LAST = "#C49A8A"   # Vanilla+h (last-timestep h) — lighter brown
COL_TRUE    = "#222222"; COL_IDRNN_H = "#6cb1de"
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
    "font.size":9,"axes.labelsize":9.5,"axes.titlesize":10,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8,
    "axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
    "pdf.fonttype":42,"ps.fonttype":42,
})

def fmt_bf(bf):
    if bf is None or not np.isfinite(bf): return "—"
    return f"{bf:.1e}" if bf>=1000 else (f"{bf:.1f}" if bf>=10 else f"{bf:.2f}")

def paired_bf(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
    if bayesfactor_ttest is None or m.sum()<3: return float("nan")
    t=stats.ttest_rel(a[m],b[m]).statistic
    return float(bayesfactor_ttest(t, m.sum(), paired=True))

def bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=0.8, c="black")
    ax.text((x1+x2)/2, y + h*1.1, txt, ha="center", va="bottom", fontsize=7.5)


# ── Load per-dataset metrics ─────────────────────────────────────────────────
def fold_mean(rec, arch, key):
    vals=[fv[arch][key] for fk,fv in rec['folds'].items() if 'error' not in fv.get(arch,{})]
    return float(np.mean(vals)) if vals else np.nan

# NLLs in the original-figure unit: NEG-LOG-LIKELIHOOD PER PARTICIPANT
# (= summed NLL over T_TRIALS).  Cog CSV is already per-session NLL.
# RNN nested-CV summaries store per-trial NLL → × T_TRIALS.
# CP RNN comes from cp_rnn_nll_nestedcv.csv (already per-session NLL).
cog_keys = ["Q (common fit)","Q (MAP)","FQ (common fit)","FQ (MAP)","True model"]
nll_df = {k: [] for k in ["Q_CP","Q_EM","FQ_CP","FQ_EM","RNN_CP","IDRNN","Vanilla","True"]}
rsa_idr, rsa_van, r2_idr, r2_van = [], [], [], []

cp_df = pd.read_csv("cp_rnn_nll_nestedcv.csv") if os.path.exists("cp_rnn_nll_nestedcv.csv") else None
# Cognitive-model NLLs from the per-fold refit (compute_synthetic_cog_nll_perfold.py):
# fit on each nested-CV fold's train subjects, evaluated on the SAME held-out fold
# subjects as the RNN NLLs (legacy model_eval_dfvanilla.csv used the separate test
# simulation — a different population draw; kept only as fallback).
cog_pf = (pd.read_csv("cog_nll_perfold_synthetic.csv")
          if os.path.exists("cog_nll_perfold_synthetic.csv") else None)
if cog_pf is None:
    print("WARNING panel a: cog_nll_perfold_synthetic.csv missing -> "
          "legacy test-sim cog NLLs (NOT the same eval subjects as the RNNs)")
for D in DATASETS:
    # cog (per-fold held-out; legacy fallback)
    _cog_pairs = [("Q_CP","Q (common fit)"), ("Q_EM","Q (MAP)"),
                  ("FQ_CP","FQ (common fit)"), ("FQ_EM","FQ (MAP)"),
                  ("True","True model")]
    pf = cog_pf[cog_pf["dataset_id"]==D] if cog_pf is not None else None
    if pf is not None and len(pf):
        for tag, name in _cog_pairs:
            v = pf[pf["model"]==name]["nll_summed"].values
            nll_df[tag].append(float(np.mean(v)) if len(v) else np.nan)
    else:
        if cog_pf is not None:
            print(f"WARNING panel a: dataset {D} missing from per-fold CSV -> legacy fallback")
        ev = pd.read_csv(f"data_dataset{D}/model_eval_dfvanilla.csv")
        for tag, name in _cog_pairs:
            v = ev[ev["model"]==name]["normalized_likelihood"].values
            nll_df[tag].append(float(np.mean(v)) if len(v) else np.nan)
    # RNN nested-CV held-out (per-trial × 200 → per-session)
    rec = json.load(open(f"{Z1_BASE}/dataset{D}/nested_cv_summary.json"))
    s = rec["summary"]
    nll_df["IDRNN"].append(s["nll_idrnn_mean"] * T_TRIALS)
    nll_df["Vanilla"].append(s["nll_vanilla_mean"] * T_TRIALS)
    # CP RNN: mean across this dataset's sessions in cp_rnn_nll_nestedcv
    if cp_df is not None:
        nll_df["RNN_CP"].append(float(cp_df[cp_df["dataset_id"]==D]["nll_summed"].mean()))
    else:
        nll_df["RNN_CP"].append(np.nan)
    rsa_idr.append(fold_mean(rec,"idrnn","rsa_r_mean_across_seeds"))
    rsa_van.append(fold_mean(rec,"vanilla","rsa_r_mean_across_seeds"))
    r2_idr.append(fold_mean(rec,"idrnn","decode_r2_mean_across_seeds"))
    r2_van.append(fold_mean(rec,"vanilla","decode_r2_mean_across_seeds"))

nll_df = pd.DataFrame(nll_df)
rsa_idr=np.array(rsa_idr); rsa_van=np.array(rsa_van)
r2_idr=np.array(r2_idr);   r2_van=np.array(r2_van)

# BFs for panel a brackets (mirror original)
bf_Q  = paired_bf(nll_df["Q_CP"],  nll_df["Q_EM"])
bf_FQ = paired_bf(nll_df["FQ_CP"], nll_df["FQ_EM"])
bf_RNNCP_IDRNN = paired_bf(nll_df["RNN_CP"], nll_df["IDRNN"])
bf_Van_IDRNN   = paired_bf(nll_df["Vanilla"], nll_df["IDRNN"])
bf_rsa = paired_bf(rsa_idr, rsa_van)
bf_r2  = paired_bf(r2_idr,  r2_van)

# ── Panel C: alpha vs z (ds0) ───────────────────────────────────────────────
z0 = np.load(f"{CANON_DS0}/latents_train.npy").reshape(-1)
alpha0 = pd.read_csv("data_dataset0/true_parameter_values.csv")["alphaP_list"].values[:len(z0)]
r_c = float(np.corrcoef(alpha0, z0)[0,1])
bf_c = float(bayesfactor_pearson(r_c, len(z0))) if bayesfactor_pearson else float("nan")
# orient z so the correlation is positive (sign of z is arbitrary)
if r_c < 0:
    z0 = -z0; r_c = -r_c
# % variance explained by PC1 is trivially 100% since z_dim=1
ev1_pct = 100.0

# ── Panel D: env decoding (train split) ────────────────────────────────────
env = pd.read_csv("env_decoding_canonical.csv")
env = env[env["split"]=="train_data"]
ENV_MODELS = [("IDRNN z",  "IDRNN_mu",            COL_IDRNN),
              ("CP RNN h", "common_process_h",    COL_RNN_CP),
              ("Vanilla h","vanilla_h",           COL_VANILLA),
              ("IDRNN h",  "informed_decoder_h",  COL_IDRNN_H)]
env_piv = env.pivot_table(index="dataset_id", columns="model", values="acc_mean")

# ── Panel E: three regressions (ds0) ────────────────────────────────────────
# Layout copied from plot_publication_figure.py: ground-truth + 2 bars per
# RNN family (R1=exact-env, R2=marg-env rollout), with bootstrap clouds + CIs.
tr = np.load("plots_dataset0/step1_three_regressions.npz")
y_tr = tr["true_alpha"]
PANEL_E_BARS = [
    ("ground truth",   "GT", "R1_raw_mean_reward",  COL_TRUE),
    ("IDRNN",          "R1", "R2_idrnn",            COL_IDRNN),
    ("IDRNN",          "R2", "R3_idrnn",            COL_IDRNN),
    ("Vanilla",        "R1", "R2_van0",             COL_VANILLA),
    ("Vanilla",        "R2", "R3_van0",             COL_VANILLA),
    ("Vanilla+h\n(avg)","R1", "R2_vanH",            COL_VAN_H),
    ("Vanilla+h\n(avg)","R2", "R3_vanH",            COL_VAN_H),
    ("Vanilla+h\n(last)","R1","R2_vanH_lastT",      COL_VAN_H_LAST),
    ("Vanilla+h\n(last)","R2","R3_vanH_lastT",      COL_VAN_H_LAST),
]
REG_ALPHA_E = {"GT": 1.00, "R1": 0.50, "R2": 1.00}
def reg_r(key):
    if key not in tr: return np.nan
    x = tr[key]; m = np.isfinite(x) & np.isfinite(y_tr)
    return float(pearsonr(x[m], y_tr[m])[0]) if m.sum() > 3 else np.nan

panel_e_stats = []
for mdl, reg, pkey, col in PANEL_E_BARS:
    x = np.asarray(tr.get(pkey, np.full_like(y_tr, np.nan)))
    m = np.isfinite(x) & np.isfinite(y_tr)
    if m.sum() < 4:
        panel_e_stats.append(dict(model=mdl, reg=reg, color=col, r=np.nan,
                                   ci=(np.nan, np.nan), boots=np.array([]),
                                   bf=np.nan))
        continue
    r0 = float(pearsonr(x[m], y_tr[m])[0])
    # Use the script's own bootstrap distribution if present (faster, matches the
    # n_boot=2000 used in analyze_synthetic_three_regressions).
    bkey = "boots_" + pkey
    if bkey in tr.files:
        boots = np.asarray(tr[bkey], dtype=float)
    else:
        rng = np.random.default_rng(1)
        xv, yv = x[m], y_tr[m]; n = len(xv); boots = np.empty(2000)
        for i in range(len(boots)):
            idx = rng.integers(0, n, size=n)
            boots[i] = np.corrcoef(xv[idx], yv[idx])[0, 1]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    bf = float(bayesfactor_pearson(r0, m.sum())) if bayesfactor_pearson else np.nan
    panel_e_stats.append(dict(model=mdl, reg=reg, color=col, r=r0,
                               ci=(float(lo), float(hi)), boots=boots, bf=bf))


# ── Panel draw functions (used for composite + standalone) ─────────────────
def draw_a(ax):
    labels = ["Ill-spec. CP","Ill-spec. EM","Cog model CP","Cog model EM",
              "CP RNN","IDRNN","Vanilla RNN"]
    keys   = ["Q_CP","Q_EM","FQ_CP","FQ_EM","RNN_CP","IDRNN","Vanilla"]
    cols   = [COL_Q_CP,COL_Q_EM,COL_FQ_CP,COL_FQ_EM,COL_RNN_CP,COL_IDRNN,COL_VANILLA]
    means  = [nll_df[k].mean() for k in keys]
    sems   = [stats.sem(nll_df[k].dropna()) for k in keys]
    xs     = np.arange(len(labels))
    ax.bar(xs, means, color=cols, alpha=0.92, edgecolor="black", linewidth=0.5, zorder=2)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="#444444",
                elinewidth=0.7, capsize=0, zorder=3)
    rng = np.random.default_rng(0)
    for i,k in enumerate(keys):
        v = nll_df[k].dropna().values
        if len(v) == 0: continue
        ax.scatter(xs[i] + rng.normal(0, 0.05, len(v)), v, s=12,
                   c="black", alpha=0.4, zorder=4, edgecolors="none")
    true_mean = nll_df["True"].mean()
    ax.axhline(true_mean, ls=":", color=COL_TRUE, lw=1.2, zorder=1)
    ymax = max(means) + max(sems); h_step = 0.4 * (max(means) - true_mean)
    yb = ymax + h_step
    bracket(ax, 0, 1, yb,                  h_step*0.3, fmt_bf(bf_Q))
    bracket(ax, 2, 3, yb,                  h_step*0.3, fmt_bf(bf_FQ))
    bracket(ax, 4, 5, yb,                  h_step*0.3, fmt_bf(bf_RNNCP_IDRNN))
    bracket(ax, 5, 6, yb + h_step*1.3,     h_step*0.3, fmt_bf(bf_Van_IDRNN))
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("neg. log-likelihood / participant")
    ax.set_ylim(bottom=true_mean - 2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("a", loc="left", fontweight="bold")

def draw_b(ax, metric="rsa"):
    if metric == "r2":
        di, dv, bf, ylab = r2_idr, r2_van, bf_r2, "α-decoding R² (LOO ridge)"
    else:
        di, dv, bf, ylab = rsa_idr, rsa_van, bf_rsa, "RSA r (latent vs α)"
    xs=[0,1]; means=[np.nanmean(di), np.nanmean(dv)]
    sems=[stats.sem(di[np.isfinite(di)]), stats.sem(dv[np.isfinite(dv)])]
    ax.bar(xs, means, yerr=sems, color=[COL_IDRNN, COL_VANILLA],
           edgecolor="black", linewidth=0.5, alpha=0.92, capsize=4, zorder=2)
    rng = np.random.default_rng(1)
    for i,d in enumerate([di, dv]):
        ax.scatter(xs[i] + rng.normal(0, 0.05, len(d)), d, s=12,
                   c="black", alpha=0.4, edgecolors="none", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(["IDRNN z=1","Vanilla"])
    ax.set_ylabel(ylab); ax.axhline(0, color="grey", lw=0.5)
    ymax = max(means) + max(sems)
    bracket(ax, 0, 1, ymax*1.05, ymax*0.05, fmt_bf(bf))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("b", loc="left", fontweight="bold")

def draw_c(ax):
    ax.scatter(alpha0, z0, s=14, color=COL_IDRNN, edgecolor="black",
               linewidth=0.3, alpha=0.85)
    ax.set_xlabel("true α"); ax.set_ylabel(f"IDRNN z  (PC1, {ev1_pct:.0f}% var)")
    ax.text(0.05, 0.95, f"r = {r_c:+.3f}\nBF₁₀={fmt_bf(bf_c)}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("c", loc="left", fontweight="bold")

def draw_d(ax):
    xs = np.arange(len(ENV_MODELS))
    for _, row in env_piv.iterrows():
        ys = [row.get(k, np.nan) for _,k,_ in ENV_MODELS]
        ax.plot(xs, ys, color="grey", alpha=0.3, lw=0.6, zorder=1)
    for i,(lab,k,c) in enumerate(ENV_MODELS):
        if k not in env_piv: continue
        ys = env_piv[k].values
        ax.scatter([i]*len(ys), ys, color=c, edgecolor="black",
                   s=18, linewidth=0.4, alpha=0.85, zorder=3)
        ax.scatter([i], [np.nanmedian(ys)], marker="_", s=380, color=c, linewidth=2.5, zorder=4)
    ax.axhline(1/3, ls=":", color=COL_TRUE, lw=1.2)
    ax.text(len(ENV_MODELS)-0.5, 1/3+0.012, "chance", ha="right",
            va="bottom", fontsize=7.5, color=COL_TRUE)
    # Paired BF brackets: IDRNN_mu vs each other feature (mirror original)
    keys = [k for _,k,_ in ENV_MODELS]
    bfs = {}
    a = env_piv["IDRNN_mu"].values
    for k in keys[1:]:
        b = env_piv[k].values
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or bayesfactor_ttest is None: bfs[k] = np.nan; continue
        t = stats.ttest_rel(a[m], b[m]).statistic
        bfs[k] = float(bayesfactor_ttest(t=abs(t), nx=int(m.sum()), paired=True))
    for i, k in enumerate(keys[1:]):
        yb = 1.06 + i * 0.13
        bracket(ax, 0, i+1, yb, 0.018, fmt_bf(bfs[k]))
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.20, 1.06 + 0.13 * (len(keys)-1) + 0.06)
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in ENV_MODELS], rotation=30, ha="right")
    ax.set_ylabel("Env-decoding accuracy (dim-matched)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("d", loc="left", fontweight="bold")

def draw_e(ax):
    xs = np.arange(len(panel_e_stats))
    heights = [rec["r"] for rec in panel_e_stats]
    lo_err  = [abs(rec["r"] - rec["ci"][0]) if np.isfinite(rec["r"]) else 0
               for rec in panel_e_stats]
    hi_err  = [abs(rec["ci"][1] - rec["r"]) if np.isfinite(rec["r"]) else 0
               for rec in panel_e_stats]
    colors  = [rec["color"] for rec in panel_e_stats]
    alphas  = [REG_ALPHA_E[rec["reg"]] for rec in panel_e_stats]
    for xi, h, c, a in zip(xs, heights, colors, alphas):
        ax.bar(xi, h, color=c, alpha=a, edgecolor="black", linewidth=0.5, zorder=2)
    # Bootstrap cloud: scatter sample of bootstrap r values jittered around each bar
    N_BOOT_PTS = 80
    rng_pts = np.random.default_rng(7)
    for xi, rec in zip(xs, panel_e_stats):
        if len(rec["boots"]) == 0: continue
        sample = rng_pts.choice(rec["boots"], size=min(N_BOOT_PTS, len(rec["boots"])), replace=False)
        j = rng_pts.normal(0, 0.045, size=len(sample))
        ax.scatter(xi + j, sample, s=5, c="black", alpha=0.40, edgecolors="none", zorder=4)
    ax.errorbar(xs, heights, yerr=[lo_err, hi_err],
                fmt="none", ecolor="#888888", elinewidth=0.7, capsize=0, zorder=5)
    # BF text above each bar (per-bar: r vs 0)
    all_pts = np.concatenate([r["boots"] for r in panel_e_stats if len(r["boots"])])
    y_top = (np.nanmax(all_pts) if all_pts.size else max(heights or [0])) + 0.06
    for xi, rec in zip(xs, panel_e_stats):
        if np.isfinite(rec.get("bf", np.nan)):
            ax.text(xi, y_top, fmt_bf(rec["bf"]), ha="center", fontsize=7,
                    color="#333333")
    # Pairwise-difference BFs via paired bootstrap (the boots arrays are
    # paired by resampling index since analyze_synthetic_three_regressions
    # seeds rngs identically across keys).  t = Δr / SE(boots_diff) →
    # bayesfactor_ttest(t, n=N_subjects, paired=True).
    rec_by = {(r["model"], r["reg"]): (i, r) for i, r in enumerate(panel_e_stats)}
    N_SUBJ = len(y_tr)
    def _pair_bf(a, b):
        ia, ra = rec_by[a]; ib, rb = rec_by[b]
        if len(ra["boots"])==0 or len(rb["boots"])==0: return None
        d_boot = np.asarray(ra["boots"]) - np.asarray(rb["boots"])
        se = float(np.nanstd(d_boot, ddof=1))
        delta = float(ra["r"] - rb["r"])
        if se < 1e-12 or bayesfactor_ttest is None: return None
        t = delta / se
        bf = float(bayesfactor_ttest(t=abs(t), nx=N_SUBJ, paired=True))
        return ia, ib, delta, bf
    pairs = [
        (("IDRNN","R2"),         ("Vanilla+h\n(avg)","R2")),
        (("IDRNN","R2"),         ("Vanilla+h\n(last)","R2")),
        (("Vanilla+h\n(avg)","R2"),("Vanilla+h\n(last)","R2")),
    ]
    pair_results = [r for r in (_pair_bf(*p) for p in pairs) if r is not None]
    # Place pairwise brackets BELOW the bars, stacked with explicit spacing so
    # they don't overlap.  Each bracket occupies its own row.
    ymin_data = float(np.nanmin(heights) if heights else -0.6)
    h_step = 0.085   # gap between rows; tuned to clear text + bracket
    yb = ymin_data - 0.06
    for k, (ia, ib, delta, bf) in enumerate(pair_results):
        y = yb - k * h_step
        tick = 0.018
        ax.plot([ia, ia, ib, ib], [y+tick, y, y, y+tick], lw=0.9, c="black")
        ax.text((ia+ib)/2, y - 0.012, f"Δr={delta:+.2f}, BF={fmt_bf(bf)}",
                ha="center", va="top", fontsize=7)
    # Reserve room below for the brackets + their text labels
    ax.set_ylim(bottom=yb - h_step * len(pair_results) - 0.02)
    # Tick labels: model name (which may already contain a newline) + reg label
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['model']}\n{r['reg']}" for r in panel_e_stats],
                       rotation=0, fontsize=7.0)
    ax.axhline(0, color="grey", lw=0.5, zorder=1)
    ax.set_ylabel("Pearson r  (predictor vs α)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("e", loc="left", fontweight="bold")


# ── Composite (mirror original 2-row layout) ───────────────────────────────
def render_composite(metric, suffix):
    fig = plt.figure(figsize=(13.0, 8.8))
    gs_outer = GridSpec(2, 1, figure=fig, hspace=0.45, left=0.07, right=0.985,
                         top=0.92, bottom=0.10)
    gs_top = GridSpecFromSubplotSpec(1, 2, gs_outer[0], width_ratios=[1.45,1.0], wspace=0.30)
    gs_top_right = GridSpecFromSubplotSpec(2, 1, gs_top[0,1], height_ratios=[0.85,1.0], hspace=0.55)
    gs_bot = GridSpecFromSubplotSpec(1, 2, gs_outer[1], width_ratios=[1.0,1.15], wspace=0.30)
    axA = fig.add_subplot(gs_top[0,0]); draw_a(axA)
    axB = fig.add_subplot(gs_top_right[0,0]); draw_b(axB, metric=metric)
    axD = fig.add_subplot(gs_top_right[1,0]); draw_d(axD)
    axC = fig.add_subplot(gs_bot[0,0]); draw_c(axC)
    axE = fig.add_subplot(gs_bot[0,1]); draw_e(axE)
    title_metric = "LOO R²" if metric=="r2" else "RSA"
    fig.suptitle(f"Synthetic study — nested-CV, z=1 anchored, dimension-matched  (panel b: {title_metric})",
                 fontweight="bold", fontsize=12)
    for ext in ("png","pdf"):
        fig.savefig(f"{OUT}/publication_figure_nestedcv{suffix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

render_composite("rsa", "")
render_composite("r2",  "_r2")

# ── Standalone panels ──────────────────────────────────────────────────────
def render_panel(name, draw_fn, size, suffix=""):
    f, ax = plt.subplots(figsize=size)
    draw_fn(ax); f.tight_layout()
    for ext in ("png","pdf"):
        f.savefig(f"{OUT}/nestedcv_panel_{name}{suffix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(f)

render_panel("a", draw_a, (7, 4.5))
render_panel("b", lambda ax: draw_b(ax, metric="rsa"), (4.5, 4.5))
render_panel("b", lambda ax: draw_b(ax, metric="r2"),  (4.5, 4.5), suffix="_r2")
render_panel("c", draw_c, (4.5, 4.5))
render_panel("d", draw_d, (5.5, 4.5))
render_panel("e", draw_e, (7.5, 5.5))

print("Saved 2 composites (RSA + LOO R²) + 6 standalone panels (a,b,b_r2,c,d,e) to plots/")
print(f"  Panel a NLL/participant means:")
for k in ["Q_CP","Q_EM","FQ_CP","FQ_EM","RNN_CP","IDRNN","Vanilla","True"]:
    print(f"    {k:<8s}: {nll_df[k].mean():6.2f}  (sem {stats.sem(nll_df[k].dropna()):.2f})")
print(f"  BFs: Q-pair={fmt_bf(bf_Q)}  FQ-pair={fmt_bf(bf_FQ)}  CP-vs-IDRNN={fmt_bf(bf_RNNCP_IDRNN)}  Van-vs-IDRNN={fmt_bf(bf_Van_IDRNN)}")
print(f"  Panel b RSA: IDRNN={np.nanmean(rsa_idr):+.3f} Vanilla={np.nanmean(rsa_van):+.3f}  BF={fmt_bf(bf_rsa)}")
print(f"  Panel b R² : IDRNN={np.nanmean(r2_idr):+.3f} Vanilla={np.nanmean(r2_van):+.3f}  BF={fmt_bf(bf_r2)}")
print(f"  Panel c ds0 α-vs-z: r={r_c:+.3f}")
print(f"  Panel d env-decode (median): " + ", ".join(f"{m[0]}={np.nanmedian(env_piv[m[1]].values):.2f}" for m in ENV_MODELS if m[1] in env_piv))
print(f"  Panel e r [95% boot CI]:")
for r in panel_e_stats:
    print(f"    {r['model']:>12s} {r['reg']:>3s}: r={r['r']:+.3f}  CI=[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  BF={fmt_bf(r['bf'])}")
