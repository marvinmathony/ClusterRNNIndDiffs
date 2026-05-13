#!/usr/bin/env python3
"""
Decode 6 questionnaire scales (PANAS_PA, PANAS_NA, STICSA, PHQ, CEI, BIG5_open)
from Thalmann cognitive-model parameters (7 params: β0, β1, β2, log_τ, β_ucb,
logit_λ, C_decay).

Mirrors the decoding pipeline in train_and_decode_thalmann_step1.py:
  • In-sample OLS (R²-based multiple-r) + LOO RidgeCV (out-of-sample r)
  • Pearson r per scale, p-values, plot of |r| with significance stars

Each subject's cog params are taken from the fold where they appear as test
(cog_em_state.npz["h_test"]) — analogous to how RNN latents are extracted
in the outer-CV pipeline (option a).

Outputs (plots_thalmann/decode_questionnaires_from_cog_params/):
  cog_params_per_subject.csv
  decoding_results.csv
  decode_questionnaires_from_cog_params.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, f as f_dist

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="data_thalmann")
parser.add_argument("--out_dir", type=str,
                    default="plots_thalmann/decode_questionnaires_from_cog_params")
parser.add_argument("--cog_state_file", type=str, default="cog_em_state.npz",
                    help="Per-fold cog-model EM state file. Use "
                         "illspec_em_state.npz for the 2-param ill-specified "
                         "Q-learning baseline.")
parser.add_argument("--quest_path", type=str,
                    default="data/finalQuestionnaireDataSession1.csv")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR  = args.out_dir
N_FOLDS  = args.folds
os.makedirs(OUT_DIR, exist_ok=True)

# Match the SCALES used in train_and_decode_thalmann_step1.py
SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5\nOpenness"),
}
SCALE_KEYS = list(SCALES.keys())

# Param labels for the cog model
COG_PARAM_LABELS = ["β0", "β1", "β2", "log_τ", "β_ucb", "logit_λ", "C_decay"]
ILLSPEC_PARAM_LABELS = ["logit_α", "log_β"]


# ── Assemble per-subject cog params (test-fold extraction) ────────────────────
def load_cog_params():
    rows = []
    for fold in range(N_FOLDS):
        f_dir = os.path.join(DATA_DIR, f"fold{fold}")
        state_p = os.path.join(f_dir, args.cog_state_file)
        if not os.path.exists(state_p):
            print(f"  fold {fold}: missing {args.cog_state_file}, skipping")
            continue
        em       = np.load(state_p, allow_pickle=True)
        h_test   = em["h_test"]                # (n_test, K)
        test_ids = em["test_ids"].tolist()
        for i, sid in enumerate(test_ids):
            rows.append({
                "subid": int(sid),
                "fold":  fold,
                **{f"p{k}": float(h_test[i, k]) for k in range(h_test.shape[1])}
            })
    df = pd.DataFrame(rows).sort_values("subid").reset_index(drop=True)
    return df


df_cog = load_cog_params()
P_COLS = [c for c in df_cog.columns if c.startswith("p")]
n_params = len(P_COLS)
LABELS = (COG_PARAM_LABELS if args.cog_state_file == "cog_em_state.npz"
          else ILLSPEC_PARAM_LABELS)
print(f"Loaded {len(df_cog)} subjects with cog params  "
      f"({n_params} params): {LABELS[:n_params]}")
df_cog.to_csv(os.path.join(OUT_DIR, "cog_params_per_subject.csv"), index=False)

# ── Load questionnaire ────────────────────────────────────────────────────────
quest = pd.read_csv(args.quest_path).set_index("ID")
for k, (items, _) in SCALES.items():
    # Skip missing items (older participants may have shorter PHQ)
    valid_items = [c for c in items if c in quest.columns]
    quest[k] = quest[valid_items].mean(axis=1)


# ── Regressors (matches train_and_decode_thalmann_step1.py) ──────────────────
def loo_ridge(Z, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    """LOO RidgeCV with per-fold z-scoring of X and y."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(),  y[tr].std()  + 1e-8
        clf = RidgeCV(alphas=list(alphas))
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return float(r), float(p), preds


def insample_ols(Z, y):
    """In-sample OLS multiple correlation + F-test p-value."""
    n, p = Z.shape
    clf = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 and dfd > 0 else np.inf
    p_val = float(f_dist.sf(F, dfn, dfd)) if dfd > 0 else 1.0
    return float(np.sqrt(max(r2, 0.0))), p_val, preds


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ── Run decoding ──────────────────────────────────────────────────────────────
print(f"\n── Decoding: in-sample OLS + LOO RidgeCV ──")
subids = df_cog["subid"].values
X = df_cog[P_COLS].values

decode_rows = []
for scale in SCALE_KEYS:
    y_all = quest.reindex(subids)[scale].values.astype(float)
    mask = np.isfinite(y_all) & np.isfinite(X).all(axis=1)
    if mask.sum() < 30:
        print(f"  {scale}: too few valid (n={int(mask.sum())}), skipping")
        continue
    Xc = X[mask]
    yy = y_all[mask]
    n  = int(mask.sum())

    r_ins, p_ins, _ = insample_ols(Xc, yy)
    r_loo, p_loo, _ = loo_ridge(Xc, yy)
    chance_r = np.sqrt(n_params / max(n - 1, 1))

    print(f"  {scale:<10}  IN-SAMPLE r={r_ins:+.3f} ({p_ins:.3g} {sig_stars(p_ins)})  "
          f"LOO r={r_loo:+.3f} ({p_loo:.3g} {sig_stars(p_loo)})  "
          f"chance r≈ {chance_r:.3f}")
    decode_rows.append({
        "scale": scale, "n": n,
        "r_insample":      r_ins, "p_insample":      p_ins,
        "r_loo":           r_loo, "p_loo":           p_loo,
        "chance_r_insample": chance_r,
    })

decode_df = pd.DataFrame(decode_rows)
decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)


# ── Plot ──────────────────────────────────────────────────────────────────────
labels = [SCALES[k][1] for k in decode_df["scale"]]
x = np.arange(len(labels))
w = 0.55
color = "#55A868"

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Left: in-sample OLS
ax = axes[0]
ax.bar(x, decode_df["r_insample"].abs(), w,
       color=color, edgecolor="k", linewidth=0.5)
for i, row in decode_df.iterrows():
    ax.text(i, abs(row["r_insample"]) + 0.01, sig_stars(row["p_insample"]),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ch = decode_df["chance_r_insample"].mean()
ax.axhline(ch, color="grey", ls="--", lw=1.2, alpha=0.7,
           label=f"chance r (k={n_params}) ≈ {ch:.3f}")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax.set_ylabel("|r|  (decodability)", fontsize=11)
ax.set_title("In-sample OLS (R²-based, k-biased)", fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Right: LOO RidgeCV
ax = axes[1]
ax.bar(x, decode_df["r_loo"].abs(), w,
       color=color, edgecolor="k", linewidth=0.5)
for i, row in decode_df.iterrows():
    ax.text(i, abs(row["r_loo"]) + 0.01, sig_stars(row["p_loo"]),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(0, color="grey", lw=1, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax.set_ylabel("|r|  (decodability)", fontsize=11)
ax.set_title("LOO RidgeCV (unbiased — true signal test)", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[0].set_ylim(0, min(1.0, axes[0].get_ylim()[1] + 0.1))
fig.suptitle(f"Questionnaire decoding from cog params  "
             f"(n_params={n_params}, N≤{len(df_cog)})",
             fontweight="bold")
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "decode_questionnaires_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
print(f"Saved → {os.path.join(OUT_DIR, 'decoding_results.csv')}")
plt.close(fig)
