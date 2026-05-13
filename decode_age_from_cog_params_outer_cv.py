#!/usr/bin/env python3
"""
Outer-CV decoder of age-group from Sloutsky cognitive-model parameters.

Mirrors the decoding pipeline of analyze_outer_cv.py but uses the 5 EM
parameters (theta, w_value, w_uncert, w_lag, w_novelty) as predictors
instead of RNN latents:

  • For each fold, load fold-specific em_results.npz to get cog params
    for fold's training and test subjects.
  • Train RBF-SVM and L2 logistic regression on training-set cog params,
    predict on test-set cog params → per-participant correctness.
  • Each subject appears as test in exactly one fold, so the union covers
    all N participants.
  • Report balanced accuracy, paired Wilcoxon and t-test against chance.

Outputs (plots_sloutsky/):
  outer_cv_age_decoding_from_cog_params.png
  outer_cv_age_decoding_from_cog_params.json
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_1samp, sem as scipy_sem
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sloutsky_cog_model import unpack_params

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds",   type=int, default=3)
parser.add_argument("--data_dir", type=str, default="data_sloutsky")
parser.add_argument("--plot_dir", type=str, default="plots_sloutsky")
args = parser.parse_args()
N_FOLDS  = args.folds
DATA_DIR = args.data_dir
PLOT_DIR = args.plot_dir
os.makedirs(PLOT_DIR, exist_ok=True)

GROUP_ORDER  = ["young_child", "old_child", "adult"]
GROUP_MAP    = {g: i for i, g in enumerate(GROUP_ORDER)}
EXCLUDE_SUBIDS = {59}    # matches analyze_outer_cv.py

C_PARAMS     = [0.01, 0.1, 1, 10, 100]
GAMMA_PARAMS = [0.01, 0.1, 1, "scale", "auto"]

COG_PARAM_NAMES = ["theta", "b_value_train", "b_uncertain_train",
                   "b_lag_train", "b_novelty_train"]


# ── Decoders (matches analyze_outer_cv.py) ───────────────────────────────────
def rbfsvm_traintest(X_train, y_train, X_test):
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    gs = GridSearchCV(pipe, {"svc__C": C_PARAMS, "svc__gamma": GAMMA_PARAMS},
                      cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_.get_params()
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", C=best["svc__C"],
                            gamma=best["svc__gamma"]))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def logreg_traintest(X_train, y_train, X_test):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    gs = GridSearchCV(pipe, {"logisticregression__C": C_PARAMS},
                      cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_C = gs.best_estimator_.get_params()["logisticregression__C"]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=best_C, max_iter=1000))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ── Build per-subject (label, cog params) by fold ─────────────────────────────
all_subid_to_label = {}
fold_test_subids   = {}
fold_train_subids  = {}

for fold in range(N_FOLDS):
    f_dir = os.path.join(DATA_DIR, f"fold{fold}")
    df_te = pd.read_csv(os.path.join(f_dir, "df_test.csv"))
    df_tr = pd.read_csv(os.path.join(f_dir, "df_train.csv"))
    fold_test_subids[fold]  = sorted(df_te["subid"].unique())
    fold_train_subids[fold] = sorted(df_tr["subid"].unique())
    for sid in fold_test_subids[fold]:
        all_subid_to_label[int(sid)] = df_te.loc[df_te["subid"] == sid,
                                                  "age"].iloc[0]

all_subids = sorted(s for s in all_subid_to_label.keys()
                    if s not in EXCLUDE_SUBIDS)
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
N = len(all_subids)
print(f"Total participants: {N} (excluded: {EXCLUDE_SUBIDS})")

y_age = np.array([GROUP_MAP[all_subid_to_label[s]] for s in all_subids])
fold_y_test  = {f: np.array([GROUP_MAP[all_subid_to_label[s]]
                              for s in fold_test_subids[f]
                              if s not in EXCLUDE_SUBIDS])
                for f in range(N_FOLDS)}
fold_y_train = {f: np.array([GROUP_MAP[all_subid_to_label[s]]
                              for s in fold_train_subids[f]
                              if s not in EXCLUDE_SUBIDS])
                for f in range(N_FOLDS)}


def _params_ordered(h_all, participants, ordered_subs):
    """Return (n, 5) cog-param matrix aligned to `ordered_subs`."""
    idx_map = {int(s): i for i, s in enumerate(participants)}
    rows = []
    for s in ordered_subs:
        if s in EXCLUDE_SUBIDS or int(s) not in idx_map:
            rows.append([np.nan] * len(COG_PARAM_NAMES))
            continue
        p = unpack_params(h_all[idx_map[int(s)]])
        rows.append([p[k] for k in COG_PARAM_NAMES])
    return np.array(rows, dtype=float)


# ── Run decoding per fold ─────────────────────────────────────────────────────
def run_decoder(decoder_fn, label):
    correct = np.full(N, np.nan)
    for fold in range(N_FOLDS):
        em_p = os.path.join(DATA_DIR, f"fold{fold}", "em_results.npz")
        if not os.path.exists(em_p):
            print(f"  fold {fold}: missing em_results.npz, skipping")
            continue
        em = np.load(em_p, allow_pickle=True)

        test_subs  = [s for s in fold_test_subids[fold]  if s not in EXCLUDE_SUBIDS]
        train_subs = [s for s in fold_train_subids[fold] if s not in EXCLUDE_SUBIDS]
        X_tr = _params_ordered(em["h_all_train"],
                               em["participants_train"].tolist(), train_subs)
        X_te = _params_ordered(em["h_all_test"],
                               em["participants_test"].tolist(), test_subs)
        y_tr = np.array([GROUP_MAP[all_subid_to_label[s]] for s in train_subs])
        y_te = np.array([GROUP_MAP[all_subid_to_label[s]] for s in test_subs])

        # Drop rows with NaN cog params
        m_tr = np.isfinite(X_tr).all(axis=1)
        m_te = np.isfinite(X_te).all(axis=1)
        if m_tr.sum() < len(GROUP_ORDER) or m_te.sum() == 0:
            print(f"  fold {fold} [{label}]: insufficient cog-param rows, skip")
            continue

        preds = decoder_fn(X_tr[m_tr], y_tr[m_tr], X_te[m_te])
        te_keep = np.array(test_subs)[m_te]
        for i, sid in enumerate(te_keep):
            if sid in subid_to_idx:
                correct[subid_to_idx[sid]] = float(preds[i] == y_te[m_te][i])
        print(f"  fold {fold} [{label}]: train n={m_tr.sum()} → "
              f"test n={m_te.sum()}, fold acc={float((preds == y_te[m_te]).mean()):.3f}")
    n_valid = int(np.isfinite(correct).sum())
    print(f"  [{label}] participants with prediction: {n_valid}/{N}")
    return correct


print("\n── RBF-SVM decoder ──")
correct_svm = run_decoder(rbfsvm_traintest, "SVM")
print("\n── Logistic Regression decoder ──")
correct_lr  = run_decoder(logreg_traintest, "LR")


def balanced_acc_from_correct(correct):
    per_class = [correct[y_age == c].mean() for c in range(len(GROUP_ORDER))
                 if (y_age == c).any()]
    return float(np.nanmean(per_class)) if per_class else float("nan")


# ── Stats and plot ────────────────────────────────────────────────────────────
def _significance(correct, chance=1/3):
    valid = correct[np.isfinite(correct)]
    if len(valid) < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t, p_t = ttest_1samp(valid, chance, alternative="greater")
    try:
        W, p_w = wilcoxon(valid - chance, alternative="greater")
    except ValueError:
        W, p_w = float("nan"), float("nan")
    return float(t), float(p_t), float(W), float(p_w)


ba_svm = balanced_acc_from_correct(correct_svm)
ba_lr  = balanced_acc_from_correct(correct_lr)
t_svm, pt_svm, W_svm, pw_svm = _significance(correct_svm)
t_lr,  pt_lr,  W_lr,  pw_lr  = _significance(correct_lr)

print(f"\nBalanced acc — SVM: {ba_svm:.3f}  (t vs chance: t={t_svm:.2f}, p={pt_svm:.4f})")
print(f"Balanced acc — LR : {ba_lr:.3f}  (t vs chance: t={t_lr:.2f}, p={pt_lr:.4f})")

chance = 1.0 / len(GROUP_ORDER)
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
for ax, ba, corr, title, p_t in [
    (axes[0], ba_svm, correct_svm, "RBF-SVM",            pt_svm),
    (axes[1], ba_lr,  correct_lr,  "Logistic Regression", pt_lr),
]:
    ax.bar([0], [ba], width=0.5, color="#55A868", edgecolor="k", alpha=0.85)
    ax.axhline(chance, color="k", ls=":", lw=1, alpha=0.6,
               label=f"Chance ({chance:.2f})")
    rng = np.random.default_rng(1)
    fin = corr[np.isfinite(corr)]
    jit = rng.uniform(-0.12, 0.12, size=fin.shape[0])
    ax.scatter(np.zeros_like(fin) + jit, fin, alpha=0.4, c="k", s=14, zorder=5,
               edgecolors="none")
    star = ("***" if p_t < 0.001 else "**" if p_t < 0.01
            else "*" if p_t < 0.05 else "n.s.")
    ax.text(0, ba + 0.06, f"p={p_t:.3g} {star}", ha="center", fontsize=9)
    ax.set_xticks([0])
    ax.set_xticklabels(["Cog params"], fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Balanced accuracy", fontsize=10)
fig.suptitle(f"Age-group decoding from cog params\n"
             f"Outer {N_FOLDS}-fold CV  ·  N={int(np.isfinite(correct_svm).sum())} participants",
             fontweight="bold")
fig.tight_layout()
out_png = os.path.join(PLOT_DIR, "outer_cv_age_decoding_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
plt.close(fig)

# ── JSON ──────────────────────────────────────────────────────────────────────
summary = {
    "n_participants":      N,
    "n_folds":             N_FOLDS,
    "predictors":          COG_PARAM_NAMES,
    "rbf_svm": {
        "balanced_accuracy": ba_svm,
        "n_valid":           int(np.isfinite(correct_svm).sum()),
        "t_vs_chance":       t_svm, "p_t": pt_svm,
        "wilcoxon":          W_svm, "p_w": pw_svm,
        "correct_per_subid": {int(s): float(correct_svm[subid_to_idx[s]])
                              for s in all_subids
                              if np.isfinite(correct_svm[subid_to_idx[s]])},
    },
    "logreg": {
        "balanced_accuracy": ba_lr,
        "n_valid":           int(np.isfinite(correct_lr).sum()),
        "t_vs_chance":       t_lr, "p_t": pt_lr,
        "wilcoxon":          W_lr, "p_w": pw_lr,
        "correct_per_subid": {int(s): float(correct_lr[subid_to_idx[s]])
                              for s in all_subids
                              if np.isfinite(correct_lr[subid_to_idx[s]])},
    },
}
out_json = os.path.join(PLOT_DIR, "outer_cv_age_decoding_from_cog_params.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {out_json}")
