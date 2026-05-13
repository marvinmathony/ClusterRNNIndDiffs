#!/usr/bin/env python3
"""
Outer-CV decoder of diagnosis (Healthy/Depression/Bipolar) from Dezfouli
cognitive-model parameters (QLP — 3 params per subject).

Mirrors the decoding pipeline in analyze_outer_cv.py but uses cog params
from each fold's qlp_em_state.npz (h_train, h_test) as predictors instead
of RNN latents.

  • For each fold: train RBF-SVM and L2 logistic regression on h_train
    (with diagnoses from df_train.csv), predict on h_test.
  • Per-participant correctness (each subject appears once across folds).
  • Report overall accuracy + paired Wilcoxon and t-test against chance.

Outputs (plots_dezfouli/):
  outer_cv_diag_decoding_from_cog_params.png
  outer_cv_diag_decoding_from_cog_params.json
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_1samp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="data_dezfouli")
parser.add_argument("--plot_dir", type=str, default="plots_dezfouli")
parser.add_argument("--cog_state_file", type=str, default="qlp_em_state.npz",
                    help="Per-fold cog-model EM state file. "
                         "Use ql_em_state.npz for ill-specified 2-param model.")
args = parser.parse_args()
N_FOLDS  = args.folds
DATA_DIR = args.data_dir
PLOT_DIR = args.plot_dir
os.makedirs(PLOT_DIR, exist_ok=True)

DIAG_LABELS = ["Healthy", "Depression", "Bipolar"]
DIAG_TO_INT = {d: i for i, d in enumerate(DIAG_LABELS)}

C_PARAMS     = [0.01, 0.1, 1, 10, 100]
GAMMA_PARAMS = [0.01, 0.1, 1, "scale", "auto"]


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


# ── Build per-subject metadata, gather subject set ────────────────────────────
all_subid_to_diag = {}
fold_test_subids  = {}
fold_train_subids = {}

for fold in range(N_FOLDS):
    f_dir = os.path.join(DATA_DIR, f"fold{fold}")
    df_te = pd.read_csv(os.path.join(f_dir, "df_test.csv"))
    df_tr = pd.read_csv(os.path.join(f_dir, "df_train.csv"))
    fold_test_subids[fold]  = sorted(df_te["subid"].unique())
    fold_train_subids[fold] = sorted(df_tr["subid"].unique())
    for sid in fold_test_subids[fold]:
        all_subid_to_diag[int(sid)] = df_te.loc[df_te["subid"] == sid,
                                                 "diag"].iloc[0]

all_subids = sorted(all_subid_to_diag.keys())
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
N = len(all_subids)
print(f"Total participants: {N}")
print("Diagnosis counts: " +
      ", ".join(f"{d}={sum(1 for v in all_subid_to_diag.values() if v == d)}"
                for d in DIAG_LABELS))

y_diag = np.array([DIAG_TO_INT[all_subid_to_diag[s]] for s in all_subids])


def _params_ordered(h_all, ids_all, ordered_subs):
    """Return (n, K) cog-param matrix aligned to ordered_subs."""
    idx_map = {int(s): i for i, s in enumerate(ids_all)}
    K = h_all.shape[1]
    rows = []
    for s in ordered_subs:
        if int(s) not in idx_map:
            rows.append([np.nan] * K)
        else:
            rows.append(list(h_all[idx_map[int(s)]]))
    return np.array(rows, dtype=float)


# ── Run decoder per fold ──────────────────────────────────────────────────────
def run_decoder(decoder_fn, label):
    correct = np.full(N, np.nan)
    n_params = None
    for fold in range(N_FOLDS):
        state_p = os.path.join(DATA_DIR, f"fold{fold}", args.cog_state_file)
        if not os.path.exists(state_p):
            print(f"  fold {fold}: missing {args.cog_state_file}, skipping")
            continue
        em       = np.load(state_p, allow_pickle=True)
        h_train  = em["h_train"]
        h_test   = em["h_test"]
        ids_tr   = em["train_ids"].tolist()
        ids_te   = em["test_ids"].tolist()
        n_params = h_train.shape[1]

        train_subs = fold_train_subids[fold]
        test_subs  = fold_test_subids[fold]
        X_tr = _params_ordered(h_train, ids_tr, train_subs)
        X_te = _params_ordered(h_test,  ids_te, test_subs)
        y_tr = np.array([DIAG_TO_INT[all_subid_to_diag[s]] for s in train_subs])
        y_te = np.array([DIAG_TO_INT[all_subid_to_diag[s]] for s in test_subs])

        m_tr = np.isfinite(X_tr).all(axis=1)
        m_te = np.isfinite(X_te).all(axis=1)
        if m_tr.sum() < len(DIAG_LABELS) or m_te.sum() == 0:
            print(f"  fold {fold} [{label}]: insufficient cog-param rows")
            continue

        preds = decoder_fn(X_tr[m_tr], y_tr[m_tr], X_te[m_te])
        te_keep = np.array(test_subs)[m_te]
        for i, sid in enumerate(te_keep):
            correct[subid_to_idx[int(sid)]] = float(preds[i] == y_te[m_te][i])
        print(f"  fold {fold} [{label}]: train n={m_tr.sum()} → "
              f"test n={m_te.sum()}, fold acc={float((preds == y_te[m_te]).mean()):.3f}")
    print(f"  [{label}] participants with prediction: "
          f"{int(np.isfinite(correct).sum())}/{N}")
    return correct, n_params


print(f"\n── RBF-SVM decoder ({args.cog_state_file}) ──")
correct_svm, n_p = run_decoder(rbfsvm_traintest, "SVM")
print("\n── Logistic regression decoder ──")
correct_lr,  _   = run_decoder(logreg_traintest,  "LR")


def overall_acc(corr):
    valid = corr[np.isfinite(corr)]
    return float(valid.mean()) if len(valid) > 0 else float("nan")


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


acc_svm = overall_acc(correct_svm)
acc_lr  = overall_acc(correct_lr)
t_svm, pt_svm, W_svm, pw_svm = _significance(correct_svm)
t_lr,  pt_lr,  W_lr,  pw_lr  = _significance(correct_lr)
print(f"\nOverall acc — SVM: {acc_svm:.3f}  (t vs chance: t={t_svm:.2f}, p={pt_svm:.4f})")
print(f"Overall acc — LR : {acc_lr:.3f}  (t vs chance: t={t_lr:.2f}, p={pt_lr:.4f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
chance = 1.0 / len(DIAG_LABELS)
PAPER_REF = 0.52   # Dezfouli et al. reported overall classification rate
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
for ax, ba, corr, title, p_t in [
    (axes[0], acc_svm, correct_svm, "RBF-SVM",            pt_svm),
    (axes[1], acc_lr,  correct_lr,  "Logistic Regression", pt_lr),
]:
    ax.bar([0], [ba], width=0.5, color="#55A868", edgecolor="k", alpha=0.85)
    ax.axhline(chance,    color="k",         ls=":",  lw=1, alpha=0.6,
               label=f"Chance ({chance:.2f})")
    ax.axhline(PAPER_REF, color="firebrick", ls="--", lw=1.2, alpha=0.7,
               label=f"Paper ({PAPER_REF:.0%})")
    rng = np.random.default_rng(1)
    fin = corr[np.isfinite(corr)]
    jit = rng.uniform(-0.12, 0.12, size=fin.shape[0])
    ax.scatter(np.zeros_like(fin) + jit, fin, alpha=0.4, c="k", s=14, zorder=5,
               edgecolors="none")
    star = ("***" if p_t < 0.001 else "**" if p_t < 0.01
            else "*" if p_t < 0.05 else "n.s.")
    ax.text(0, max(ba, chance) + 0.08, f"p={p_t:.3g} {star}",
            ha="center", fontsize=9)
    ax.set_xticks([0])
    ax.set_xticklabels(["Cog params (QLP)"], fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Overall accuracy", fontsize=10)
fig.suptitle(f"Diagnosis decoding from QLP cog params  ·  "
             f"Outer {N_FOLDS}-fold CV  ·  "
             f"N={int(np.isfinite(correct_svm).sum())} participants",
             fontweight="bold")
fig.tight_layout()
out_png = os.path.join(PLOT_DIR, "outer_cv_diag_decoding_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
plt.close(fig)

# ── JSON ──────────────────────────────────────────────────────────────────────
summary = {
    "n_participants":      N,
    "n_folds":             N_FOLDS,
    "n_cog_params":         int(n_p) if n_p is not None else None,
    "cog_state_file":       args.cog_state_file,
    "rbf_svm": {
        "overall_accuracy": acc_svm,
        "n_valid":          int(np.isfinite(correct_svm).sum()),
        "t_vs_chance":      t_svm, "p_t": pt_svm,
        "wilcoxon":         W_svm, "p_w": pw_svm,
        "correct_per_subid": {int(s): float(correct_svm[subid_to_idx[s]])
                              for s in all_subids
                              if np.isfinite(correct_svm[subid_to_idx[s]])},
    },
    "logreg": {
        "overall_accuracy": acc_lr,
        "n_valid":          int(np.isfinite(correct_lr).sum()),
        "t_vs_chance":      t_lr, "p_t": pt_lr,
        "wilcoxon":         W_lr, "p_w": pw_lr,
        "correct_per_subid": {int(s): float(correct_lr[subid_to_idx[s]])
                              for s in all_subids
                              if np.isfinite(correct_lr[subid_to_idx[s]])},
    },
}
out_json = os.path.join(PLOT_DIR, "outer_cv_diag_decoding_from_cog_params.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {out_json}")
