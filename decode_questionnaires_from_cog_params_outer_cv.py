#!/usr/bin/env python3
"""
Outer-CV decoder of 6 questionnaire scales (PANAS_PA, PANAS_NA, STICSA, PHQ,
CEI, BIG5_open) from Thalmann cognitive-model parameters.

Mirrors the decoding pipeline in analyze_hp_v3_outer_cv.py:
  • For each fold, load h_test from cog_em_state.npz (cog params for the
    fold's test subjects — fit without their data), match to questionnaire
    scores, run LOO RidgeCV with within-fold z-scoring of X and y.
  • Pool predictions across folds, report Pearson r per scale.
  • Also reports the "pooled across folds" LOO ridge for direct comparability
    with the RNN outer-CV pipeline.

Outputs (plots_thalmann/decode_questionnaires_from_cog_params/):
  outer_cv_decoding_results.csv
  outer_cv_decode_questionnaires_from_cog_params.png
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="data_thalmann")
parser.add_argument("--out_dir", type=str,
                    default="plots_thalmann/decode_questionnaires_from_cog_params")
parser.add_argument("--cog_state_file", type=str, default="cog_em_state.npz")
parser.add_argument("--quest_path", type=str,
                    default="data/finalQuestionnaireDataSession1.csv")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR  = args.out_dir
N_FOLDS  = args.folds
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.1, 1, 10, 100, 1000]

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

quest = pd.read_csv(args.quest_path).set_index("ID")
for k, (items, _) in SCALES.items():
    valid = [c for c in items if c in quest.columns]
    quest[k] = quest[valid].mean(axis=1)


# ── LOO ridge utilities (matches analyze_hp_v3_outer_cv.py) ──────────────────
def loo_ridge(Z, y):
    """LOO RidgeCV with z-scoring within each train split."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(),  y[tr].std()  + 1e-8
        clf = RidgeCV(alphas=ALPHAS)
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return float(r), float(p)


def loo_preds_within_fold(Z, y):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(),  y[tr].std()  + 1e-8
        clf = RidgeCV(alphas=ALPHAS)
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    return preds


def decode_within_fold(Z_per_fold, subids_per_fold, scale_key):
    """Within-fold LOO; pool predictions across folds, single Pearson r."""
    y_list, p_list = [], []
    for Z, sids in zip(Z_per_fold, subids_per_fold):
        y = np.array([quest.loc[sid, scale_key] if sid in quest.index else np.nan
                      for sid in sids])
        mask = np.isfinite(y) & np.isfinite(Z).all(axis=1)
        if mask.sum() < 10:
            continue
        y_list.append(y[mask])
        p_list.append(loo_preds_within_fold(Z[mask], y[mask]))
    if not y_list:
        return float("nan"), 1.0
    y_pool = np.concatenate(y_list)
    p_pool = np.concatenate(p_list)
    r, p = pearsonr(y_pool, p_pool)
    return float(r), float(p)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ── Load cog params per fold ─────────────────────────────────────────────────
all_h_per_fold     = []   # list[(n_test, K)] per fold
all_subids_per_fold = []  # list[array of subids]
all_h_pooled  = []
all_subids_pooled = []
n_params = None

for fold in range(N_FOLDS):
    f_dir = os.path.join(DATA_DIR, f"fold{fold}")
    state_p = os.path.join(f_dir, args.cog_state_file)
    if not os.path.exists(state_p):
        print(f"  fold {fold}: missing {args.cog_state_file}, skipping")
        continue
    em       = np.load(state_p, allow_pickle=True)
    h_test   = em["h_test"]               # (n_test, K)
    test_ids = np.array(em["test_ids"].tolist(), dtype=int)
    n_params = h_test.shape[1]
    all_h_per_fold.append(h_test)
    all_subids_per_fold.append(test_ids)
    all_h_pooled.append(h_test)
    all_subids_pooled.append(test_ids)
    print(f"  fold {fold}: {len(test_ids)} test subjects, K={n_params}")

if not all_h_per_fold:
    raise SystemExit("No fold cog-param state files found.")

h_pooled        = np.concatenate(all_h_pooled,      axis=0)
subids_pooled   = np.concatenate(all_subids_pooled, axis=0)
print(f"\nPooled {len(subids_pooled)} subjects across {len(all_h_per_fold)} folds.")


# ── Run decoding ──────────────────────────────────────────────────────────────
rows = []
print(f"\n── Decoding ──")
for key in SCALE_KEYS:
    # Pooled-across-folds LOO (mixes per-fold params; fine because cog
    # parameters are in the same parameter space across folds — but the
    # within-fold variant remains the principled one).
    y_pool = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                       for sid in subids_pooled])
    mask_pool = np.isfinite(y_pool) & np.isfinite(h_pooled).all(axis=1)
    if mask_pool.sum() < 20:
        r_p, p_p = float("nan"), 1.0
    else:
        r_p, p_p = loo_ridge(h_pooled[mask_pool], y_pool[mask_pool])

    # Within-fold LOO
    r_w, p_w = decode_within_fold(all_h_per_fold, all_subids_per_fold, key)

    print(f"  {key:<10}  pooled r={r_p:+.3f} ({p_p:.3g} {sig_stars(p_p)})  "
          f"within-fold r={r_w:+.3f} ({p_w:.3g} {sig_stars(p_w)})")
    rows.append({"scale": key,
                 "r_pooled":   r_p, "p_pooled":   p_p,
                 "r_wf":       r_w, "p_wf":       p_w,
                 "n_pooled":   int(mask_pool.sum())})

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(OUT_DIR, "outer_cv_decoding_results.csv"), index=False)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
x = np.arange(len(SCALE_KEYS))
w = 0.55

for ax, r_col, p_col, title in [
    (axes[0], "r_pooled", "p_pooled", "Pooled-across-folds LOO Ridge"),
    (axes[1], "r_wf",     "p_wf",     "Within-fold LOO Ridge (principled)"),
]:
    bar_cols = ["#55A868" if r >= 0 else "#a05050"
                for r in results_df[r_col]]
    ax.bar(x, results_df[r_col].abs(), w, color=bar_cols, edgecolor="k",
           linewidth=0.5)
    for i, row in results_df.iterrows():
        ax.text(i, abs(row[r_col]) + 0.01, sig_stars(row[p_col]),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="grey", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_LABELS, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("|r|  (LOO Pearson)", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"Questionnaire decoding from cog params — outer {N_FOLDS}-fold CV  "
             f"(n_params={n_params}, N≈{int(results_df['n_pooled'].max())})",
             fontweight="bold")
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "outer_cv_decode_questionnaires_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
print(f"Saved → {os.path.join(OUT_DIR, 'outer_cv_decoding_results.csv')}")
plt.close(fig)
