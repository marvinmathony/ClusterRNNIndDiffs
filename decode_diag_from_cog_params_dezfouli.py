#!/usr/bin/env python3
"""
Decode diagnosis (Healthy / Depression / Bipolar) from Dezfouli cognitive-model
parameters (QLP — 3 params per subject: logit_φ, log_β, κ).

Mirrors the decoding pipeline in train_and_decode_dezfouli_step1.py but uses
the EM-fitted cog params as predictors instead of RNN latents:

  • 3-way multinomial LOO logistic
  • 3 pairwise binary LOO logistic (H-D, H-B, D-B)
  • PCA on cog-param matrix (3 components → all 3 params), per-PC LOO logistic
  • Pearson r between PC scores and per-subject stay probability

Each subject's cog params are taken from the fold where they appear as test
(qlp_em_state.npz["h_test"]), so per-subject params come from a model fit
without that subject's data — analogous to how RNN latents are extracted in
the outer-CV pipeline.

Outputs (plots_dezfouli/decode_diag_from_cog_params/):
  cog_params_per_subject.csv
  decoding_results.csv
  pc_decoding_results.csv
  pc_stay_corr.csv
  decode_diag_from_cog_params.png
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="data_dezfouli")
parser.add_argument("--out_dir", type=str,
                    default="plots_dezfouli/decode_diag_from_cog_params")
parser.add_argument("--cog_state_file", type=str, default="qlp_em_state.npz",
                    help="Per-fold cog-model EM state file. Use "
                         "ql_em_state.npz for the 2-param ill-specified model.")
parser.add_argument("--n_perm", type=int, default=2000,
                    help="Label-permutation reps for null distribution.")
parser.add_argument("--logreg_C", type=float, default=1.0)
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR  = args.out_dir
N_FOLDS  = args.folds
N_PERM   = args.n_perm
os.makedirs(OUT_DIR, exist_ok=True)

DIAG_LABELS = ["Healthy", "Depression", "Bipolar"]
DIAG_TO_INT = {d: i for i, d in enumerate(DIAG_LABELS)}
PAIRWISE = [("Healthy", "Depression"),
            ("Healthy", "Bipolar"),
            ("Depression", "Bipolar")]


# ── Load & assemble per-subject cog params (test-fold extraction) ─────────────
def load_cog_params():
    """
    For each subject, take their cog params from the fold where they appear
    as test (so the EM fit was run without them).  Returns a DataFrame with
    columns subid, diag, p1..pK.
    """
    rows = []
    for fold in range(N_FOLDS):
        f_dir = os.path.join(DATA_DIR, f"fold{fold}")
        state_p = os.path.join(f_dir, args.cog_state_file)
        if not os.path.exists(state_p):
            print(f"  fold {fold}: missing {args.cog_state_file}, skipping")
            continue
        em       = np.load(state_p, allow_pickle=True)
        h_test   = em["h_test"]                     # (n_test, K)
        test_ids = em["test_ids"].tolist()          # subject ids (ints)
        df_te    = pd.read_csv(os.path.join(f_dir, "df_test.csv"))
        sid2diag = (df_te.drop_duplicates("subid")
                         .set_index("subid")["diag"].to_dict())
        for i, sid in enumerate(test_ids):
            sid_int = int(sid)
            if sid_int not in sid2diag:
                continue
            rows.append({
                "subid": sid_int,
                "fold":  fold,
                "diag":  sid2diag[sid_int],
                **{f"p{k}": float(h_test[i, k]) for k in range(h_test.shape[1])}
            })
    df = pd.DataFrame(rows).sort_values("subid").reset_index(drop=True)
    return df


df_cog = load_cog_params()
P_COLS = [c for c in df_cog.columns if c.startswith("p")]
PARAM_LABELS_QLP = {"p0": "logit_φ", "p1": "log_β", "p2": "κ"}
PARAM_LABELS_QL  = {"p0": "logit_φ", "p1": "log_β"}
LABEL_MAP = (PARAM_LABELS_QLP if args.cog_state_file == "qlp_em_state.npz"
             else PARAM_LABELS_QL)

print(f"Loaded {len(df_cog)} subjects with cog params  ({len(P_COLS)} params).")
print(f"Params: {[LABEL_MAP.get(c, c) for c in P_COLS]}")
print(f"Diagnosis counts: " +
      ", ".join(f"{d}={int((df_cog['diag']==d).sum())}" for d in DIAG_LABELS))

df_cog.to_csv(os.path.join(OUT_DIR, "cog_params_per_subject.csv"), index=False)

X = df_cog[P_COLS].values
y = np.array([DIAG_TO_INT[d] for d in df_cog["diag"]])
subids = df_cog["subid"].values


# ── LOO logistic decoding (matches train_and_decode_dezfouli_step1.py) ───────
def _fit_logistic(X, y, multinomial):
    if multinomial:
        return LogisticRegression(
            max_iter=2000, C=args.logreg_C,
            multi_class="multinomial", solver="lbfgs",
        ).fit(X, y)
    return LogisticRegression(max_iter=2000, C=args.logreg_C).fit(X, y)


def loo_logistic(X, y, multinomial):
    n = len(y)
    preds = np.zeros(n, dtype=int)
    probs = (np.zeros((n, len(np.unique(y)))) if multinomial else np.zeros(n))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        scaler = StandardScaler().fit(X[tr])
        clf = _fit_logistic(scaler.transform(X[tr]), y[tr], multinomial)
        Xte = scaler.transform(X[te])
        preds[te] = clf.predict(Xte)[0]
        if multinomial:
            classes = clf.classes_
            full = np.zeros(probs.shape[1])
            full[classes] = clf.predict_proba(Xte)[0]
            probs[te] = full
        else:
            probs[te] = clf.predict_proba(Xte)[0, 1]
    return preds, probs


def decoding_metrics(y, preds, probs, multinomial, n_perm=N_PERM, rng_seed=0):
    bacc = balanced_accuracy_score(y, preds)
    if multinomial:
        try:
            auroc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        except ValueError:
            auroc = np.nan
    else:
        auroc = roc_auc_score(y, probs)

    rng = np.random.default_rng(rng_seed)
    null_bacc = np.empty(n_perm); null_auroc = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_bacc[i] = balanced_accuracy_score(y_perm, preds)
        if multinomial:
            try:
                null_auroc[i] = roc_auc_score(y_perm, probs,
                                              multi_class="ovr", average="macro")
            except ValueError:
                null_auroc[i] = np.nan
        else:
            null_auroc[i] = roc_auc_score(y_perm, probs)
    p_bacc  = float((null_bacc  >= bacc).mean())
    p_auroc = float(np.nanmean(null_auroc >= auroc))
    return dict(bacc=float(bacc), auroc=float(auroc),
                p_bacc=p_bacc, p_auroc=p_auroc)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ── Stay probability — needed for PC × stay-prob correlation ─────────────────
def compute_stay_prob():
    """Per-subject P(c_t == c_{t-1}) over valid trial pairs.
    Pulls c_test from the fold where each subject appears as test."""
    p_per_sid = {}
    for fold in range(N_FOLDS):
        f_dir = os.path.join(DATA_DIR, f"fold{fold}")
        c_te = np.load(os.path.join(f_dir, "c_test.npy"))     # (B, n_blocks, T)
        df_te = pd.read_csv(os.path.join(f_dir, "df_test.csv"))
        sub_order = df_te.drop_duplicates("subid")["subid"].values
        for i, sid in enumerate(sub_order):
            if i >= c_te.shape[0]:
                break
            same, total = 0, 0
            for b in range(c_te.shape[1]):
                seq = c_te[i, b]
                valid = seq[seq != -100]
                if len(valid) >= 2:
                    same  += int((valid[1:] == valid[:-1]).sum())
                    total += len(valid) - 1
            p_per_sid[int(sid)] = same / max(total, 1)
    return np.array([p_per_sid.get(int(s), np.nan) for s in subids])


stay_p = compute_stay_prob()
print(f"Stay-prob: mean={np.nanmean(stay_p):.3f}  "
      f"range=[{np.nanmin(stay_p):.3f}, {np.nanmax(stay_p):.3f}]")


# ── Full-cog-param decoding ──────────────────────────────────────────────────
contrasts = [("3way", None)] + [
    (f"{a}_vs_{b}", (DIAG_TO_INT[a], DIAG_TO_INT[b])) for a, b in PAIRWISE
]

decode_rows = []
print(f"\n── Full-param diagnosis decoding (LOO logistic, n={len(y)}) ──")
for cname, sel in contrasts:
    if sel is None:
        Xc, yc, multi = X, y, True
    else:
        m = np.isin(y, sel)
        Xc = X[m]
        yc = (y[m] == sel[1]).astype(int)
        multi = False
    preds, probs = loo_logistic(Xc, yc, multinomial=multi)
    mets = decoding_metrics(yc, preds, probs, multinomial=multi, rng_seed=0)
    print(f"  CogParams  {cname:<22} bacc={mets['bacc']:.3f} "
          f"({mets['p_bacc']:.3g} {sig_stars(mets['p_bacc'])})  "
          f"AUROC={mets['auroc']:.3f} "
          f"({mets['p_auroc']:.3g} {sig_stars(mets['p_auroc'])})")
    decode_rows.append({"contrast": cname, "n": int(len(yc)), **mets})

decode_df = pd.DataFrame(decode_rows)
decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)


# ── Per-PC decoding & PC × stay-prob ─────────────────────────────────────────
n_pc = X.shape[1]
print(f"\n── Per-PC decoding (PCA on cog params, {n_pc} components) ──")
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=n_pc).fit(Xs)
scores = pca.transform(Xs)
print("  Explained variance: " +
      ", ".join(f"PC{i+1}={pca.explained_variance_ratio_[i]:.2f}"
                for i in range(n_pc)))

pc_decoding_rows, pc_stay_rows = [], []
for k in range(n_pc):
    pc = scores[:, k:k+1]
    for cname, sel in contrasts:
        if sel is None:
            Xc, yc, multi = pc, y, True
        else:
            m = np.isin(y, sel)
            Xc = pc[m]
            yc = (y[m] == sel[1]).astype(int)
            multi = False
        preds, probs = loo_logistic(Xc, yc, multinomial=multi)
        mets = decoding_metrics(yc, preds, probs, multinomial=multi, rng_seed=k)
        pc_decoding_rows.append({
            "pc": k + 1,
            "explained_var": float(pca.explained_variance_ratio_[k]),
            "contrast": cname, "n": int(len(yc)), **mets,
        })
    valid = np.isfinite(stay_p)
    if valid.sum() >= 4:
        r, p = pearsonr(scores[valid, k], stay_p[valid])
        pc_stay_rows.append({
            "pc": k + 1,
            "explained_var": float(pca.explained_variance_ratio_[k]),
            "r": float(r), "p": float(p),
        })

pc_dec_df  = pd.DataFrame(pc_decoding_rows)
pc_stay_df = pd.DataFrame(pc_stay_rows)
pc_dec_df.to_csv(os.path.join(OUT_DIR, "pc_decoding_results.csv"), index=False)
pc_stay_df.to_csv(os.path.join(OUT_DIR, "pc_stay_corr.csv"), index=False)


# ── Plot: AUROC per contrast (full + per-PC), plus stay-prob bars ────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (1) Full-param AUROC per contrast
contrast_order = [c for c, _ in contrasts]
ax = axes[0]
sub = decode_df.set_index("contrast").loc[contrast_order]
x = np.arange(len(contrast_order))
ax.bar(x, sub["auroc"].values, color="#55A868", edgecolor="k", linewidth=0.5)
for j, (cn, row) in enumerate(sub.iterrows()):
    ax.text(j, row["auroc"] + 0.01, sig_stars(row["p_auroc"]),
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(contrast_order, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("AUROC (LOO logistic)")
ax.set_title("Full cog params → diagnosis", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (2) Per-PC AUROC for 3-way contrast
ax = axes[1]
sub = pc_dec_df[pc_dec_df["contrast"] == "3way"].sort_values("pc")
ax.bar(sub["pc"], sub["auroc"], color="#4C72B0", edgecolor="k", linewidth=0.5)
for _, row in sub.iterrows():
    ax.text(row["pc"], row["auroc"] + 0.01, sig_stars(row["p_auroc"]),
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
ax.set_xticks(sub["pc"])
ax.set_xticklabels([f"PC{int(k)}\n({LABEL_MAP.get(P_COLS[int(k)-1],'')})"
                    for k in sub["pc"]], fontsize=8)
ax.set_ylabel("AUROC")
ax.set_title("Per-PC 3-way diagnosis AUROC", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (3) PC × stay-prob correlation
ax = axes[2]
if not pc_stay_df.empty:
    sub = pc_stay_df.sort_values("pc")
    bars = ax.bar(sub["pc"], sub["r"], color="#55A868", edgecolor="k", linewidth=0.5)
    for bar, p in zip(bars, sub["p"]):
        y_ = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                y_ + (0.02 if y_ >= 0 else -0.05),
                sig_stars(p), ha="center",
                va="bottom" if y_ >= 0 else "top",
                fontsize=8, fontweight="bold")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(sub["pc"])
    ax.set_xticklabels([f"PC{int(k)}" for k in sub["pc"]], fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("PC × stay-prob", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
else:
    ax.set_visible(False)

fig.suptitle(f"Diagnosis decoding from QLP cog params  "
             f"(N={len(y)} subjects · per-subject params from test-fold EM)",
             fontweight="bold")
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "decode_diag_from_cog_params.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
plt.close(fig)
print(f"Saved → {os.path.join(OUT_DIR, 'decoding_results.csv')}")
print(f"Saved → {os.path.join(OUT_DIR, 'pc_decoding_results.csv')}")
print(f"Saved → {os.path.join(OUT_DIR, 'pc_stay_corr.csv')}")
