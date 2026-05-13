#!/usr/bin/env python3
"""
Decode age-group labels from Sloutsky cognitive-model parameters using the
same LOOCV logistic regression pipeline as decode_across_seeds.py
(StandardScaler + LogisticRegressionCV, StratifiedKFold inner CV).

This is the cog-model parameter analogue of decode_across_seeds.py; it
provides a non-RNN baseline using the 5 fitted EM parameters
(theta, w_value, w_uncert, w_lag, w_novelty) as predictors.

Cog params: data_sloutsky/em_results.npz   (h_all_test, participants_test).
Outputs    : plots_sloutsky/decode_age_from_cog_params{_seed_tag}.{png,json}.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sloutsky_cog_model import unpack_params

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--em_path", type=str,
                    default="data_sloutsky/em_results.npz",
                    help="Path to EM-fitted cog-model results (default uses "
                         "the global EM fit on the full split).")
parser.add_argument("--data_dir", type=str, default="data_sloutsky")
parser.add_argument("--plot_dir", type=str, default="plots_sloutsky")
parser.add_argument("--n_bootstrap", type=int, default=0,
                    help="Number of bootstrap resamples (0 = none) over "
                         "participants for SEM bars.")
args = parser.parse_args()

DATA_DIR = args.data_dir
PLOT_DIR = args.plot_dir
INNER_CV = 5
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load test labels ──────────────────────────────────────────────────────────
df_test  = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique = (df_test.drop_duplicates(subset="subid")
                    .sort_values("subid")
                    .reset_index(drop=True))
GROUP_MAP = {"young_child": 0, "old_child": 1, "adult": 2}

# ── Load cog params ───────────────────────────────────────────────────────────
em = np.load(args.em_path, allow_pickle=True)
h_test            = em["h_all_test"]                # (B_test, 5)
print(f"how many test participants?: {h_test.shape}")
participants_test = em["participants_test"].tolist()
B = len(participants_test)
print(f"Loaded cog params for {B} test participants from {args.em_path}")

# Match labels to cog-model row order
sub_to_age = dict(zip(df_unique["subid"].astype(int), df_unique["age"]))
labels = np.array([GROUP_MAP[sub_to_age[int(s)]] for s in participants_test])

# Convert raw EM h-vectors to interpretable params
COG_PARAM_NAMES = ["theta", "b_value_train", "b_uncertain_train",
                   "b_lag_train", "b_novelty_train"]
COG_PARAM_LABELS = ["θ", "w value", "w uncert", "w lag", "w novelty"]

X = np.array([
    [unpack_params(h_test[i])[k] for k in COG_PARAM_NAMES]
    for i in range(B)
])
print(f"Predictor matrix X shape: {X.shape}")
print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")


# ── LOOCV logistic regression ────────────────────────────────────────────────
def logistic_loocv(X, y, inner_cv=INNER_CV):
    """Same logic as decode_across_seeds.logistic_loocv."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=classes.dtype)

    for tr, te in tqdm(loo.split(X), total=len(y), desc="LOOCV", leave=False):
        y_tr = y[tr]
        min_class = np.min(np.bincount(y_tr))
        n_splits  = min(inner_cv, min_class) if min_class >= 2 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        base = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", max_iter=5000, cv=cv,
        )
        clf = make_pipeline(StandardScaler(), base)
        clf.fit(X[tr], y_tr)
        preds[te] = clf.predict(X[te])

    return preds, balanced_accuracy_score(y, preds)


print("\nRunning LOOCV logistic regression on cog params...")
preds, bal_acc = logistic_loocv(X, labels)
print(f"Balanced accuracy = {bal_acc:.4f}  (chance = {1/3:.4f})")

# Per-class correctness
correct = (preds == labels).astype(float)
per_class = {g: float(correct[labels == GROUP_MAP[g]].mean())
             for g in GROUP_MAP}
print(f"Per-class correctness: {per_class}")

# Optional bootstrap SEM
sem = 0.0
boot_accs = []
if args.n_bootstrap > 0:
    rng = np.random.default_rng(0)
    for _ in range(args.n_bootstrap):
        idx = rng.choice(B, B, replace=True)
        boot_accs.append(balanced_accuracy_score(labels[idx], preds[idx]))
    sem = float(np.std(boot_accs))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.2, 4))
ax.bar([0], [bal_acc], color="#55A868", edgecolor="k", width=0.5)
if sem > 0:
    ax.errorbar([0], [bal_acc], yerr=[sem], fmt="none", color="k", capsize=5)
ax.axhline(1/3, color="k", ls=":", lw=0.8, alpha=0.5, label="Chance (1/3)")
ax.set_xticks([0])
ax.set_xticklabels(["Cog params\n(EM)"], fontsize=10)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_ylim(0, 1.0)
title = f"Age-group decoding from cog params\nLOOCV logistic, N={B}"
if args.n_bootstrap > 0:
    title += f"  (SEM from {args.n_bootstrap} bootstraps)"
ax.set_title(title, fontsize=10)
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out_png = os.path.join(PLOT_DIR, "decode_age_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved plot → {out_png}")
plt.close(fig)

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "n_participants": int(B),
    "balanced_accuracy": float(bal_acc),
    "bootstrap_sem": float(sem),
    "per_class_correctness": per_class,
    "predictors": COG_PARAM_NAMES,
    "labels_per_subid": {int(s): int(l) for s, l in zip(participants_test, labels)},
    "preds_per_subid":  {int(s): int(p) for s, p in zip(participants_test, preds)},
    "em_path": args.em_path,
}
out_json = os.path.join(PLOT_DIR, "decode_age_from_cog_params.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {out_json}")
