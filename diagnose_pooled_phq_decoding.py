#!/usr/bin/env python3
"""
Diagnose whether the near-perfect pooled PHQ decoding for IDRNN
(|r|~0.95 on the hp_v3 `uw00_lmbd02_eh5_h5_z3` combo) is real or
an artifact of pooling z's from three independently-trained
fold-encoders into one LOO ridge.

Hypothesis to rule out: per-fold encoders produce fold-constant
components in z (essentially a "fold id" signal).  If per-fold
PHQ means also differ, the pooled ridge can trivially decode PHQ
by reading the fold id, even when no subject-level information is
present in z.

Outputs -> plots_thalmann/comparison/
  diagnose_phq_pooled.png   — four diagnostic panels
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, ".")
from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep,
    test_latentrnn_secondstep_causal_posterior_weighting,
)

# ── Config ────────────────────────────────────────────────────────────────────
COMBO      = "uw00_lmbd005_eh5_h5_z10"
HP_VERSION = "hp_v3"
DGP        = "thalmann"
N_FOLDS    = 3
IS_N       = 2       # IS samples; mu_tensor doesn't depend on N, so keep small
# How to reduce the (B, Bk*T, z) posterior tensor to one (B, z) vector:
#   "last" — mu at the last valid trial
#   "mean" — mean of mu across all valid trials
#   int k  — mu at the k-th (1-indexed) valid trial, e.g. 20
LATENT_MODE = "last"
# Extraction path:
#   True  — one encoder forward pass with return_per_timestep=True.
#           Numerically identical to the causal test-function output for
#           continuous_encoder=True models (which zero padding internally),
#           but ~100× faster.  Use for all continuous_encoder=True combos.
#   False — run test_latentrnn_secondstep_causal_posterior_weighting.  Needed
#           only if per-block (continuous_encoder=False) causal extraction
#           diverges from the one-shot encoder call.
FAST_EXTRACT = True
PLOT_DIR   = "plots_thalmann/comparison"
os.makedirs(PLOT_DIR, exist_ok=True)

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
quest["PHQ"] = quest[[f"PHQ_9_{i}" for i in range(10)]].mean(axis=1)


# One random seed per fold, cached so pooled and Procrustes analyses stay
# consistent on the same encoder.
_seed_rng    = np.random.default_rng(0)
_chosen_seed = {}


def _pick_seed_dir(fold, run_base):
    """Return a (cached) randomly chosen seed dir for this fold.
    Only considers seeds with a loadable best checkpoint."""
    if fold in _chosen_seed:
        return _chosen_seed[fold]
    valid = []
    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if os.path.exists(ckpt):
            valid.append(sd)
    if not valid:
        raise RuntimeError(f"No valid seed dirs under {run_base}")
    pick = str(_seed_rng.choice(valid))
    _chosen_seed[fold] = pick
    print(f"  fold {fold}: picked seed '{pick}' (of {len(valid)} valid)")
    return pick


# Cache full LatentRNN_secondstep models by (run_base, sd) so we don't reload
# the same checkpoint twice across the pooled and Procrustes analyses.
_model_cache = {}


def _load_model(run_base, sd):
    """Load the full LatentRNN_secondstep (encoder + frozen decoder + task
    embedding) at its CV-selected best epoch for one seed dir.  Matches the
    model construction used in testing_script.py."""
    cache_key = (run_base, sd)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    cfg_p = os.path.join(run_base, sd, "config.json")
    with open(cfg_p) as f: cfg = json.load(f)
    mc      = cfg["model_config"]
    best_ep = cfg["cv_selected_epoch"]
    ckpt    = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
    state   = torch.load(ckpt, map_location="cpu")

    enc_in_dim   = mc.get("enc_in_dim", cfg["in_dim"])
    enc_hidden   = mc.get("enc_hidden", cfg.get("hidden", 8))
    dec_in_dim   = mc.get("dec_in_dim", cfg["in_dim"])
    task_emb_dim = mc.get("task_emb_dim", 0)
    n_tasks      = mc.get("n_tasks", None)
    reinit       = mc.get("reinit_decoder_per_block", False)

    encoder = IDRNN(in_dim=enc_in_dim, z_dim=mc["z_dim"], hid=enc_hidden,
                    n_tasks=n_tasks, task_emb_dim=task_emb_dim,
                    continuous_encoder=mc.get("continuous_encoder", False))
    decoder = Decoder(in_dim=dec_in_dim, z_dim=mc["z_dim"],
                      hid=cfg["hidden"], A=cfg["A"])
    model = LatentRNN_secondstep(
        encoder=encoder, hid=cfg["hidden"], z_dim=mc["z_dim"],
        in_dim=dec_in_dim, A=cfg["A"], decoder=decoder,
        n_tasks=n_tasks, task_emb_dim=task_emb_dim,
        reinit_decoder_per_block=reinit,
    )
    model.load_state_dict(state)
    model.set_task_ids(task_ids_global)
    model.eval()
    _model_cache[cache_key] = model
    return model


def _causal_mu_tensor(model, xin, c_test):
    """Run the causal importance-weighted test function and return the
    per-timestep posterior-mean tensor (B, Bk*T, z_dim).  Matches what
    testing_script.py saves as `latents_tensor*.pt`.  `c_test` must be shape
    (B, Bk, T) with -100 at padded trials."""
    with torch.no_grad():
        _, _, mu_tensor, _ = test_latentrnn_secondstep_causal_posterior_weighting(
            model=model, blocks=xin, y=c_test, N=IS_N, id_case=True,
            enc_blocks=xin,
        )
    return mu_tensor


def _fast_mu_tensor(model, xin):
    """One-shot encoder pass (return_per_timestep=True), reshaped to
    (B, Bk*T, z_dim) to match the causal-path output.  For continuous_encoder
    models the padding is zeroed inside IDRNN so the per-position mu's at
    valid positions equal what the causal test function stores."""
    with torch.no_grad():
        mu_all, _ = model.encoder(xin, return_per_timestep=True)  # (B, Bk, T, z)
    B, Bk, T, z = mu_all.shape
    return mu_all.reshape(B, Bk * T, z)


def _mu_tensor(model, xin, c_test):
    """Dispatch to fast or causal extraction per FAST_EXTRACT."""
    if FAST_EXTRACT:
        return _fast_mu_tensor(model, xin)
    return _causal_mu_tensor(model, xin, c_test)


def _pick_trial(mu_tensor, c_test, mode=None):
    """Reduce (B, Bk*T, z) causal-posterior tensor to (B, z) per-subject
    latent.  `mode` is `LATENT_MODE` by default:
      "last"  — mu at last valid trial
      "mean"  — mean of mu across valid trials
      int k   — mu at k-th (1-indexed) valid trial"""
    mode = LATENT_MODE if mode is None else mode
    if c_test.dim() == 4:
        c_test = c_test.squeeze(-1)
    B, Bk, T = c_test.shape
    N = Bk * T
    valid_flat = (c_test >= 0).reshape(B, N)
    has_valid  = valid_flat.any(dim=1)

    if mode == "mean":
        mask  = valid_flat.unsqueeze(-1).float()          # (B, N, 1)
        z_sum = (mu_tensor * mask).sum(dim=1)             # (B, z)
        n     = mask.sum(dim=1).clamp(min=1)              # (B, 1)
        return (z_sum / n).numpy()

    if mode == "last":
        idx = (N - 1) - valid_flat.flip(dims=[1]).long().argmax(dim=1)
    elif isinstance(mode, int):
        cumsum  = valid_flat.cumsum(dim=1)
        reached = (cumsum >= mode).any(dim=1)
        idx     = (cumsum >= mode).float().argmax(dim=1)
        idx     = torch.where(reached, idx, torch.zeros_like(idx))
    else:
        raise ValueError(f"LATENT_MODE must be 'last', 'mean', or int; got {mode!r}")

    idx = torch.where(has_valid, idx, torch.zeros_like(idx))
    return mu_tensor[torch.arange(B), idx].numpy()


def load_fold_z(fold):
    """Load z from ONE randomly chosen seed for this fold.  Latent is the
    causal posterior mean (test_latentrnn_secondstep_causal_posterior_weighting),
    reduced per LATENT_MODE."""
    run_base = f"runs_thalmann_{HP_VERSION}_{COMBO}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"
    xin_test = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    c_test   = torch.tensor(np.load(f"{data_dir}/c_test.npy"),   dtype=torch.float32)
    df_test  = pd.read_csv(f"{data_dir}/df_test.csv")
    subids   = (df_test["subid"].values if "subid" in df_test.columns
                else df_test["session"].values)

    sd    = _pick_seed_dir(fold, run_base)
    model = _load_model(run_base, sd)
    mu_tensor = _mu_tensor(model, xin_test, c_test)
    return _pick_trial(mu_tensor, c_test), subids


def _safe_pearsonr(y, preds):
    """pearsonr that returns NaN when either input is constant (undefined r)."""
    y = np.asarray(y); preds = np.asarray(preds)
    if y.size < 2 or np.std(y) < 1e-12 or np.std(preds) < 1e-12:
        return np.nan
    return pearsonr(y, preds)[0]


RIDGE_ALPHAS = np.unique(np.concatenate([
    np.logspace(-4, 2, 30),        # 1e-4 to 100
    np.linspace(100, 1000, 150),   # dense 100 to 1000
    #np.logspace(3, 4, 20),         # 1000 to 1e4  (capped — high α forces
                                    #  intercept-only fit → LOO r → −1 artefact)
]))


def loo_ridge(Z, y, estimator="ridge", return_diag=False):
    """LOO-CV regression.  Returns (r, preds) or (r, preds, diag).
    estimator: 'ridge'  → RidgeCV(alphas=RIDGE_ALPHAS)
               'linear' → LinearRegression (no regularisation).
    diag dict: alphas_picked (ridge only), coef_norms per LOO fit."""
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    alphas, coef_norms = [], []
    for tr, te in loo.split(Z):
        if estimator == "ridge":
            clf = RidgeCV(alphas=RIDGE_ALPHAS)
        elif estimator == "linear":
            clf = LinearRegression()
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        clf.fit(Z[tr], y[tr])
        if estimator == "ridge":
            alphas.append(clf.alpha_)
        coef_norms.append(float(np.linalg.norm(clf.coef_)))
        preds[te] = clf.predict(Z[te])
    r = _safe_pearsonr(y, preds)
    if return_diag:
        diag = {"alphas":     np.asarray(alphas) if alphas else None,
                "coef_norms": np.asarray(coef_norms)}
        return r, preds, diag
    return r, preds


# ── Load z + PHQ for each fold ───────────────────────────────────────────────
print(f"Loading z for combo {COMBO} across {N_FOLDS} folds...")
z_per_fold, sids_per_fold = [], []
for f in range(N_FOLDS):
    z, s = load_fold_z(f)
    z_per_fold.append(z)
    sids_per_fold.append(s)
    print(f"  fold {f}: {z.shape}")

z_pooled    = np.concatenate(z_per_fold)
sids_pooled = np.concatenate(sids_per_fold)
fold_ids    = np.concatenate([np.full(len(s), f) for f, s in enumerate(sids_per_fold)])
phq         = np.array([quest.loc[sid, "PHQ"] if sid in quest.index else np.nan
                        for sid in sids_pooled])
mask        = ~np.isnan(phq)

z_pooled    = z_pooled[mask]
fold_ids    = fold_ids[mask]
phq         = phq[mask]
print(f"Total N with PHQ = {len(phq)}")

# ── 1. Pooled LOO ridge ──────────────────────────────────────────────────────
r_pooled, preds_pooled = loo_ridge(z_pooled, phq)
print(f"Pooled LOO r_PHQ = {r_pooled:+.3f}")

# ── 2. Within-fold LOO ridge ─────────────────────────────────────────────────
preds_wf = np.zeros(len(phq))
preds_wf_linear = np.zeros(len(phq))
r_wf_per_fold = []
r_wf_per_fold_linear = []
for f in range(N_FOLDS):
    idx = fold_ids == f
    if idx.sum() < 10: continue
    r_f, preds_f = loo_ridge(z_pooled[idx], phq[idx])
    r_linear_f, preds_linear_f = loo_ridge(z_pooled[idx], phq[idx], estimator="linear")
    preds_wf[idx] = preds_f
    preds_wf_linear[idx] = preds_linear_f
    r_wf_per_fold.append(r_f)
    r_wf_per_fold_linear.append(r_linear_f)
r_wf_pooled_preds,_ = pearsonr(phq, preds_wf)
r_wf_pooled_preds_linear, _ = pearsonr(phq, preds_wf_linear)
print(f"Within-fold per-fold r = {[f'{r:+.3f}' for r in r_wf_per_fold]}  "
      f"(mean |r|={np.mean(np.abs(r_wf_per_fold)):.3f})"
      f"Within-fold per-fold r linear regression = {[f'{r:+.3f}' for r in r_wf_per_fold_linear]}  "
      f"(mean |r|={np.mean(np.abs(r_wf_per_fold_linear)):.3f})")

print(f"r of pooled WF preds vs y = {r_wf_pooled_preds:+.3f}")
print(f"r of pooled WF preds vs y linear = {r_wf_pooled_preds_linear:+.3f}")

# ── 3. Fold-identity baseline (clean LOO fold-mean predictor) ────────────────
# For each held-out subject, predict = mean(PHQ_train_same_fold).  No ridge,
# no standardisation artefacts — this is the honest null-ceiling for "only
# knowing which fold you're in".
preds_fold_only = np.zeros(len(phq))
for i in range(len(phq)):
    f = fold_ids[i]
    train_mask = (fold_ids == f)
    train_mask[i] = False
    preds_fold_only[i] = phq[train_mask].mean()
r_fold_only = r2_score(phq, preds_fold_only)
print(f"\nFold-identity-ONLY baseline r_PHQ = {r_fold_only:+.3f}   "
      "(clean LOO fold-mean predictor — if close to r_pooled, "
      "pooled decoding is fold-identity artifact)")

# Fraction of variance explained by fold mean
phq_by_fold = [phq[fold_ids == f] for f in range(N_FOLDS)]
fold_means  = [arr.mean() for arr in phq_by_fold]
fold_stds   = [arr.std()  for arr in phq_by_fold]
grand_mean  = phq.mean()
ss_between  = sum(len(arr) * (m - grand_mean)**2
                  for arr, m in zip(phq_by_fold, fold_means))
ss_total    = ((phq - grand_mean)**2).sum()
eta2_fold   = ss_between / ss_total
print(f"Between-fold PHQ means : {[f'{m:.3f}' for m in fold_means]}")
print(f"Within-fold PHQ stds   : {[f'{s:.3f}' for s in fold_stds]}")
print(f"η² (fold-explained PHQ variance) = {eta2_fold:.4f}")

# Fold-mean-of-z (how different are z spaces across folds?)
z_fold_means = np.stack([z_pooled[fold_ids == f].mean(0) for f in range(N_FOLDS)])

# ── 4. PCA of z colored by fold ──────────────────────────────────────────────
pca = PCA(n_components=2).fit(z_pooled)
z_pca = pca.transform(z_pooled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")


# ── Plot: 2x3 diagnostic figure ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fold_colors = ["#4C72B0", "#DD8452", "#55A868"]

# Panel 1: y_true vs pooled preds, colored by fold
ax = axes[0, 0]
for f in range(N_FOLDS):
    idx = fold_ids == f
    ax.scatter(phq[idx], preds_pooled[idx], color=fold_colors[f],
               alpha=0.7, s=25, label=f"Fold {f} (n={idx.sum()})",
               edgecolors="k", linewidths=0.3)
lims = [min(phq.min(), preds_pooled.min()) - 0.05,
        max(phq.max(), preds_pooled.max()) + 0.05]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True PHQ")
ax.set_ylabel("Pooled-LOO predicted PHQ")
ax.set_title(f"Pooled LOO:  r = {r_pooled:+.3f}", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel 2: y_true vs within-fold preds, colored by fold
ax = axes[0, 1]
for f in range(N_FOLDS):
    idx = fold_ids == f
    ax.scatter(phq[idx], preds_wf[idx], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}: r={r_wf_per_fold[f]:+.2f}")
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True PHQ")
ax.set_ylabel("Within-fold LOO predicted PHQ")
ax.set_title(f"Within-fold LOO:  pooled-preds r = {r_wf_pooled_preds:+.3f}",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel 3: fold-identity-only baseline
ax = axes[0, 2]
for f in range(N_FOLDS):
    idx = fold_ids == f
    ax.scatter(phq[idx], preds_fold_only[idx], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}")
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True PHQ")
ax.set_ylabel("Predicted PHQ from fold-id ONLY")
ax.set_title(f"Fold-identity baseline:  r = {r_fold_only:+.3f}\n"
             f"(η² fold = {eta2_fold:.3f})",
             fontweight="bold",
             color=("red" if abs(r_fold_only) > 0.5 else "black"))
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel 4: PHQ distribution by fold (true values)
ax = axes[1, 0]
bp = ax.boxplot(phq_by_fold, patch_artist=True, widths=0.5,
                medianprops=dict(color="k"))
for patch, col in zip(bp["boxes"], fold_colors):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax.set_xticklabels([f"Fold {f}\n(n={len(arr)})" for f, arr in enumerate(phq_by_fold)])
ax.set_ylabel("PHQ")
ax.set_title("True PHQ by fold\n"
             f"means = {[round(m,2) for m in fold_means]}",
             fontweight="bold")
ax.grid(axis="y", alpha=0.2)

# Panel 5: PCA of z colored by fold
ax = axes[1, 1]
for f in range(N_FOLDS):
    idx = fold_ids == f
    ax.scatter(z_pca[idx, 0], z_pca[idx, 1], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("z latent space, coloured by fold\n"
             "(clustering ⇒ fold-identity signal in z)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel 6: summary text
ax = axes[1, 2]
ax.axis("off")
txt = (
    f"Combo: {COMBO}\n"
    f"Subjects with PHQ (pooled): {len(phq)}\n\n"
    f"Pooled LOO r            : {r_pooled:+.3f}\n"
    f"Within-fold LOO r (pooled preds): {r_wf_pooled_preds:+.3f}\n"
    f"Within-fold per-fold r  : {[f'{r:+.2f}' for r in r_wf_per_fold]}\n\n"
    f"Fold-identity ONLY r    : {r_fold_only:+.3f}\n"
    f"η² (fold→PHQ)           : {eta2_fold:.3f}\n\n"
    "Interpretation\n"
    "───────────────\n"
    "If fold-identity r ≈ pooled r   ⇒  pooled decoding is dominated\n"
    "    by fold-mean differences in PHQ × fold-constant components in z,\n"
    "    i.e. the high r is an artefact of pooling independent encoders.\n\n"
    "If pooled r  ≫  fold-identity r ⇒  z carries real subject-level\n"
    "    information about PHQ beyond what fold id can explain."
)
ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace",
        fontsize=9, transform=ax.transAxes)

fig.suptitle(f"Diagnosing pooled PHQ decoding — IDRNN {COMBO}",
             fontweight="bold", fontsize=13)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "diagnose_phq_pooled.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Procrustes-aligned decoding
# ═══════════════════════════════════════════════════════════════════════════
# Idea: per-fold encoders produce non-aligned latent spaces, so pooling z's is
# suspect.  If we can rotate each fold's latent space into a common frame, then
# pool the held-out z's and regress to PHQ, we get an honest test of whether
# subject-level signal survives alignment.
#
# Steps:
#   1.  For every subject, compute z under ALL 3 fold encoders.
#   2.  Orthogonal Procrustes: rotate folds 1 and 2 into fold 0's frame.
#   3.  For each subject, take their HELD-OUT z (fold they were test in),
#       apply that fold's rotation → aligned held-out z.
#   4.  Pooled LOO ridge on aligned held-out z's.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("Procrustes-aligned pooled decoding")
print("═" * 70)


def load_all_subject_inputs():
    """Concatenate xin_test and c_test across folds (each subject appears
    exactly once)."""
    xin_list, c_list, sid_list, fold_list = [], [], [], []
    for f in range(N_FOLDS):
        xin = torch.tensor(np.load(f"data_{DGP}/fold{f}/xin_test.npy"),
                           dtype=torch.float32)
        c   = torch.tensor(np.load(f"data_{DGP}/fold{f}/c_test.npy"),
                           dtype=torch.float32)
        df  = pd.read_csv(f"data_{DGP}/fold{f}/df_test.csv")
        sids = (df["subid"].values if "subid" in df.columns
                else df["session"].values)
        xin_list.append(xin)
        c_list.append(c)
        sid_list.append(sids)
        fold_list.append(np.full(len(sids), f))
    return (torch.cat(xin_list, 0), torch.cat(c_list, 0),
            np.concatenate(sid_list), np.concatenate(fold_list))


def compute_z_all_subjects(fold, xin_all, c_all):
    """Run ONE randomly chosen seed's full model for this fold on xin_all and
    reduce the causal posterior tensor per LATENT_MODE."""
    run_base = f"runs_thalmann_{HP_VERSION}_{COMBO}/fold{fold}"
    sd    = _pick_seed_dir(fold, run_base)
    model = _load_model(run_base, sd)
    mu_tensor = _mu_tensor(model, xin_all, c_all)
    return _pick_trial(mu_tensor, c_all)


# 1. Inputs + each-encoder z for all 236 subjects
xin_all, c_all, sids_all, fold_of_sub = load_all_subject_inputs()
print(f"Loaded {len(sids_all)} subject inputs "
      f"(fold counts = {[int((fold_of_sub == f).sum()) for f in range(N_FOLDS)]})")

Z_per_enc = [compute_z_all_subjects(f, xin_all, c_all) for f in range(N_FOLDS)]
for f, Z in enumerate(Z_per_enc):
    print(f"  encoder fold {f}: Z shape = {Z.shape}")

# Save PHQ + each subject's single held-out z (from the fold where they were
# in the test set) — one z-dim vector per participant.
_phq_for_csv = np.array([quest.loc[sid, "PHQ"] if sid in quest.index else np.nan
                         for sid in sids_all])
_z_dim       = Z_per_enc[0].shape[1]
_Z_heldout   = np.stack([Z_per_enc[int(fold_of_sub[i])][i]
                         for i in range(len(sids_all))])
_csv_cols = {"subid": sids_all,
             "test_fold": fold_of_sub.astype(int),
             "phq": _phq_for_csv}
for j in range(_z_dim):
    _csv_cols[f"z_d{j}"] = _Z_heldout[:, j]
_latent_csv_path = os.path.join(PLOT_DIR,
                                f"phq_latents_{COMBO}_{LATENT_MODE}.csv")
pd.DataFrame(_csv_cols).to_csv(_latent_csv_path, index=False)
print(f"Saved PHQ + per-encoder z's to {_latent_csv_path}")

# 2. Procrustes-align folds 1,2 → fold 0 reference
Z_ref = Z_per_enc[0]
Z_aligned = [Z_ref]
rotations = [np.eye(Z_ref.shape[1])]
for f in range(1, N_FOLDS):
    R, scale = orthogonal_procrustes(Z_per_enc[f], Z_ref)
    Z_aligned.append(Z_per_enc[f] @ R)
    rotations.append(R)
    err_before = np.linalg.norm(Z_per_enc[f] - Z_ref)
    err_after  = np.linalg.norm(Z_per_enc[f] @ R - Z_ref)
    print(f"  fold {f} → fold 0 Procrustes error: "
          f"{err_before:.2f} -> {err_after:.2f}   (scale param={scale:.2f})")

# 3. Per-subject, take held-out (= test-fold) aligned z
Z_heldout_aligned = np.zeros_like(Z_ref)
for i, f in enumerate(fold_of_sub):
    Z_heldout_aligned[i] = Z_aligned[int(f)][i]

# Align PHQ to sids_all
phq_all  = np.array([quest.loc[sid, "PHQ"] if sid in quest.index else np.nan
                     for sid in sids_all])
mask_all = ~np.isnan(phq_all)
Z_aln    = Z_heldout_aligned[mask_all]
phq_aln  = phq_all[mask_all]
fold_aln = fold_of_sub[mask_all]
print(f"N with PHQ (aligned set) = {len(phq_aln)}")

# 4. Pooled LOO ridge on aligned z
r_aligned, preds_aligned = loo_ridge(Z_aln, phq_aln)
print(f"\nAligned pooled LOO r_PHQ       = {r_aligned:+.3f}")
print(f"  (vs unaligned pooled r        = {r_pooled:+.3f})")
print(f"  (vs within-fold pooled-preds r = {r_wf_pooled_preds:+.3f})")
print(f"  (vs fold-id-only baseline r    = {r_fold_only:+.3f})")



# For context: what does pooled LOO look like on UNaligned but same-source z?
#   (i.e. using each subject's held-out z_f without rotation)
Z_heldout_unaligned = np.zeros_like(Z_ref)
for i, f in enumerate(fold_of_sub):
    Z_heldout_unaligned[i] = Z_per_enc[int(f)][i]
Z_unaln = Z_heldout_unaligned[mask_all]
r_unaln, preds_unaln = loo_ridge(Z_unaln, phq_aln)
print(f"  (sanity: unaligned held-out pooled r = {r_unaln:+.3f} — should "
      f"match earlier r_pooled)")

# Per-fold decomposition of aligned preds
r_aln_per_fold = []
for f in range(N_FOLDS):
    idx = fold_aln == f
    if idx.sum() >= 10:
        r_f = r2_score(phq_aln[idx], preds_aligned[idx])
        r_aln_per_fold.append(r_f)
        print(f"  within-fold r (aligned preds) fold {f}: {r_f:+.3f}")

# PCA of aligned z for visualization
pca_aln  = PCA(n_components=2).fit(Z_aln)
z_aln_pc = pca_aln.transform(Z_aln)


# ═══════════════════════════════════════════════════════════════════════════
# Full Procrustes (rotation + uniform scale + translation)
# ═══════════════════════════════════════════════════════════════════════════
# Orthogonal Procrustes alone cannot correct per-fold scale or mean offsets.
# Full Procrustes solves   min_{s,R,t} || s · Z_f · R + t - Z_0 ||²
# with optimal scale  s = tr((Z_f')ᵀ Z_0' R) / ||Z_f'||_F²
# and translation    t = mean(Z_0) - s · mean(Z_f) · R
# where primes denote mean-centred matrices.
print("\n" + "─" * 70)
print("Full Procrustes (rotation + scale + translation)")
print("─" * 70)


def full_procrustes(Z_src, Z_ref):
    """Return (Z_src_aligned, R, s, t) — best fit of s·Z_src·R + t to Z_ref."""
    mu_s = Z_src.mean(0); mu_r = Z_ref.mean(0)
    Zs   = Z_src - mu_s
    Zr   = Z_ref - mu_r
    R, scale_num = orthogonal_procrustes(Zs, Zr)
    s = scale_num / (np.linalg.norm(Zs) ** 2)
    t = mu_r - s * mu_s @ R
    Z_aligned = s * Z_src @ R + t
    return Z_aligned, R, s, t


Z_ref_full   = Z_per_enc[0]
Z_aligned_full = [Z_ref_full]
for f in range(1, N_FOLDS):
    Z_a, Rf, sf, tf = full_procrustes(Z_per_enc[f], Z_ref_full)
    Z_aligned_full.append(Z_a)
    err_before = np.linalg.norm(Z_per_enc[f] - Z_ref_full)
    err_after  = np.linalg.norm(Z_a          - Z_ref_full)
    print(f"  fold {f} → fold 0 full-Proc error: "
          f"{err_before:.2f} -> {err_after:.2f}   (s={sf:.3f}, "
          f"||t||={np.linalg.norm(tf):.3f})")

# Per-subject held-out z in full-aligned frame
Z_heldout_full = np.zeros_like(Z_ref_full)
for i, f in enumerate(fold_of_sub):
    Z_heldout_full[i] = Z_aligned_full[int(f)][i]

Z_full    = Z_heldout_full[mask_all]
r_full, preds_full = loo_ridge(Z_full, phq_aln)
print(f"\nFull-Procrustes pooled LOO r_PHQ = {r_full:+.3f}")
print(f"  (vs orthogonal-Procrustes r    = {r_aligned:+.3f})")
print(f"  (vs unaligned pooled r         = {r_pooled:+.3f})")
print(f"  (vs within-fold pooled-preds r  = {r_wf_pooled_preds:+.3f})")

plt.figure()
plt.scatter(phq_aln, preds_full, alpha=0.4)
plt.xlabel("True PHQ")
plt.ylabel("Predicted PHQ")
plt.savefig("plots_thalmann/comparison/predictions.png", dpi=300, bbox_inches="tight")
plt.close()

r_full_per_fold = []
for f in range(N_FOLDS):
    idx = fold_aln == f
    if idx.sum() >= 10:
        r_f = r2_score(phq_aln[idx], preds_full[idx])
        r_full_per_fold.append(r_f)
        print(f"  within-fold r2 (full-aligned preds) fold {f}: {r_f:+.3f}")

pca_full  = PCA(n_components=2).fit(Z_full)
z_full_pc = pca_full.transform(Z_full)


# ─── Within-fold LOO ridge on full-Procrustes-aligned z ──────────────────────
# Run LOO ridge *within each fold* on the full-Procrustes-aligned held-out z,
# aggregate the per-fold predictions across all subjects, and correlate with
# PHQ.  This regression cannot exploit between-fold PHQ-mean differences (each
# fit only sees one fold's subjects) — if the aggregated Pearson r is
# meaningfully positive, there's genuine subject-level PHQ signal in z.
def within_fold_ridge_agg(Z, y, fold_labels, n_folds, estimator="ridge"):
    """LOO regression within each fold, pool preds, return (r, per_fold_r, preds)."""
    preds = np.zeros_like(y)
    per_fold_r = []
    for f in range(n_folds):
        idx = np.where(fold_labels == f)[0]
        if len(idx) < 10:
            continue
        r_f, preds_f = loo_ridge(Z[idx], y[idx], estimator=estimator)
        preds[idx] = preds_f
        per_fold_r.append(r_f)
    r_agg = _safe_pearsonr(y, preds)
    return r_agg, per_fold_r, preds


r_wf_full, r_wf_full_per_fold, preds_wf_full = within_fold_ridge_agg(
    Z_full, phq_aln, fold_aln, N_FOLDS)
print(f"\nWithin-fold LOO on full-Procrustes z, aggregated:")
print(f"  per-fold r        = {[f'{r:+.3f}' for r in r_wf_full_per_fold]}")
print(f"  aggregated Pearson r vs PHQ = {r_wf_full:+.3f}   "
      f"(positive ⇒ signal survives without cross-fold pooling)")

# Same within-fold regression on UNALIGNED held-out z (no Procrustes at all).
# Since each regression stays within one fold, alignment shouldn't matter in
# principle — differences between this and the aligned version indicate the
# alignment itself introduces or removes within-fold structure.
r_wf_unaln, r_wf_unaln_per_fold, preds_wf_unaln = within_fold_ridge_agg(
    Z_unaln, phq_aln, fold_aln, N_FOLDS)
print(f"Within-fold LOO on UNALIGNED z, aggregated:")
print(f"  per-fold r        = {[f'{r:+.3f}' for r in r_wf_unaln_per_fold]}")
print(f"  aggregated Pearson r vs PHQ = {r_wf_unaln:+.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Leak-free Procrustes + LOO ridge
# ═══════════════════════════════════════════════════════════════════════════
# Previous sections fit the alignment on ALL 236 subjects.  For each LOO
# iteration, the alignment therefore "sees" the held-out subject via the two
# non-test encoders (which trained on it).  To eliminate that leakage we refit
# R (+ s, t) per LOO fold on the n-1 training subjects, then apply the learned
# transform to the held-out subject before decoding.
#
# Cost: 236 Procrustes + 236 RidgeCV fits — still seconds at d=3.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Leak-free LOO Procrustes (alignment refit on n-1 subjects each fold)")
print("─" * 70)


def loo_ridge_leak_free_procrustes(Z_per_enc_, fold_of_sub_, y_all, mask_all,
                                   use_full=True):
    """For each held-out subject, refit Procrustes on the remaining n-1 subjects,
    then run one ridge fit → single prediction for held-out subject.
    Returns (r, preds_on_valid_mask)."""
    N_all = Z_per_enc_[0].shape[0]
    valid = np.where(mask_all)[0]
    preds = np.full(N_all, np.nan)

    for i in valid:
        train = np.setdiff1d(valid, [i])
        Z_ref_tr = Z_per_enc_[0][train]

        # Align each non-reference fold to fold 0 using train subjects only,
        # then apply the learned transform to ALL subjects (including held-out i).
        aligned_by_fold = {0: Z_per_enc_[0]}
        for f in range(1, N_FOLDS):
            if use_full:
                mu_s = Z_per_enc_[f][train].mean(0)
                mu_r = Z_ref_tr.mean(0)
                Zs   = Z_per_enc_[f][train] - mu_s
                Zr   = Z_ref_tr - mu_r
                R, scale_num = orthogonal_procrustes(Zs, Zr)
                s = scale_num / (np.linalg.norm(Zs) ** 2)
                t = mu_r - s * mu_s @ R
                aligned_by_fold[f] = s * Z_per_enc_[f] @ R + t
            else:
                R, _ = orthogonal_procrustes(Z_per_enc_[f][train], Z_ref_tr)
                aligned_by_fold[f] = Z_per_enc_[f] @ R

        Z_heldout_mat = np.stack([
            aligned_by_fold[int(fold_of_sub_[s])][s] for s in range(N_all)
        ])

        Ztr = Z_heldout_mat[train]
        ytr = y_all[train]
        mu_z, sd_z = Ztr.mean(0), Ztr.std(0) + 1e-8
        mu_y, sd_y = ytr.mean(),  ytr.std()  + 1e-8
        clf = RidgeCV(alphas=RIDGE_ALPHAS)
        clf.fit((Ztr - mu_z) / sd_z, (ytr - mu_y) / sd_y)
        pred = clf.predict((Z_heldout_mat[[i]] - mu_z) / sd_z) * sd_y + mu_y
        preds[i] = pred[0]

    r = r2_score(y_all[valid], preds[valid])
    return r, preds[valid]


r_orth_lf, preds_orth_lf = loo_ridge_leak_free_procrustes(
    Z_per_enc, fold_of_sub, phq_all, mask_all, use_full=False)
print(f"Leak-free orthogonal Procrustes r = {r_orth_lf:+.3f}   "
      f"(vs leaky {r_aligned:+.3f})")

r_full_lf, preds_full_lf = loo_ridge_leak_free_procrustes(
    Z_per_enc, fold_of_sub, phq_all, mask_all, use_full=True)
print(f"Leak-free full Procrustes       r = {r_full_lf:+.3f}   "
      f"(vs leaky {r_full:+.3f})")
print(f"actual predictions of leak free full procrustes: {preds_full_lf}")

# Per-fold breakdown for leak-free full Procrustes
r_full_lf_per_fold = []
for f in range(N_FOLDS):
    idx = fold_aln == f
    if idx.sum() >= 10:
        r_f = r2_score(phq_aln[idx], preds_full_lf[idx])
        r_full_lf_per_fold.append(r_f)
        print(f"  within-fold r (leak-free full) fold {f}: {r_f:+.3f}")


# ── Enumerate valid seeds per fold (used by multi-questionnaire analysis) ───
valid_seeds_per_fold = []
for f in range(N_FOLDS):
    run_base_f = f"runs_thalmann_{HP_VERSION}_{COMBO}/fold{f}"
    fold_valid = []
    for sd in sorted(d for d in os.listdir(run_base_f) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base_f, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as _f: cfg_sd = json.load(_f)
        best_ep = cfg_sd.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(run_base_f, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if os.path.exists(ckpt):
            fold_valid.append(sd)
    valid_seeds_per_fold.append(fold_valid)
    print(f"  fold {f}: {len(fold_valid)} valid seeds available")


# ═══════════════════════════════════════════════════════════════════════════
# Fold-id decodability FROM z — is the fold signal still readable after
# alignment?  (Complementary to fold-id-only PHQ baseline: that checks how
# much PHQ variance fold-id alone can explain; this checks how much fold-id
# information remains in z in each space.)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Fold-id decodability from z (chance = 1/3 = 33.3%)")
print("─" * 70)
from sklearn.linear_model import LogisticRegression


def fold_loo_accuracy(Z, fold_labels):
    """Leave-one-out multinomial LR.  Return accuracy."""
    correct = 0
    for i in range(len(fold_labels)):
        tr = np.ones(len(fold_labels), dtype=bool); tr[i] = False
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        clf = LogisticRegression(solver="lbfgs",
                                 C=1.0, max_iter=500)
        clf.fit((Z[tr] - mu) / sd, fold_labels[tr])
        pred = clf.predict((Z[[i]] - mu) / sd)[0]
        correct += (pred == fold_labels[i])
    return correct / len(fold_labels)


# Build held-out-z for each of the three spaces
Z_unaligned_heldout = Z_heldout_unaligned[mask_all]   # = Z_unaln
Z_orth_heldout      = Z_aln                            # orthogonal Procrustes
Z_full_heldout      = Z_full                           # full Procrustes

#acc_unaligned = fold_loo_accuracy(Z_unaligned_heldout, fold_aln)
#acc_orth      = fold_loo_accuracy(Z_orth_heldout,       fold_aln)
acc_full      = fold_loo_accuracy(Z_full_heldout,       fold_aln)
#print(f"Unaligned z   : fold LOO-accuracy = {acc_unaligned:.3f}")
#print(f"Orth-aligned z: fold LOO-accuracy = {acc_orth:.3f}")
print(f"Full-aligned z: fold LOO-accuracy = {acc_full:.3f}")
print(f"Chance level  : {1/N_FOLDS:.3f}   "
      "(if accuracy is far above chance ⇒ fold-id still encoded in z)")

# ── Plot: Procrustes diagnostic (2x3) ────────────────────────────────────────
fig2, ax2 = plt.subplots(2, 3, figsize=(17, 10))

# Shared scatter limits across the two alignment scatters
lims_a = [min(phq_aln.min(), preds_aligned.min(), preds_full.min()) - 0.05,
          max(phq_aln.max(), preds_aligned.max(), preds_full.max()) + 0.05]

# Panel [0,0]: orthogonal-Procrustes pooled preds vs true
ax = ax2[0, 0]
for f in range(N_FOLDS):
    idx = fold_aln == f
    ax.scatter(phq_aln[idx], preds_aligned[idx], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}")
ax.plot(lims_a, lims_a, "k--", lw=0.8, alpha=0.6, label="y = x")
ax.set_xlim(lims_a); ax.set_ylim(lims_a)
ax.set_xlabel("True PHQ")
ax.set_ylabel("Pooled-LOO predicted PHQ")
ax.set_title(f"Orthogonal Procrustes:  r = {r_aligned:+.3f}",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel [0,1]: full-Procrustes pooled preds vs true
ax = ax2[0, 1]
for f in range(N_FOLDS):
    idx = fold_aln == f
    ax.scatter(phq_aln[idx], preds_full[idx], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}")
ax.plot(lims_a, lims_a, "k--", lw=0.8, alpha=0.6, label="y = x")
ax.set_xlim(lims_a); ax.set_ylim(lims_a)
ax.set_xlabel("True PHQ")
ax.set_ylabel("Pooled-LOO predicted PHQ")
ax.set_title(f"Full Procrustes (R + s + t):  r = {r_full:+.3f}",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel [0,2]: comparison bar chart of |r|
ax = ax2[0, 2]
labels = ["Unaligned\npooled", "Within-fold\n(pooled preds)",
          "Fold-id\nonly",
          "Orthogonal\nProcrustes\n(leaky)",
          "Orthogonal\nProcrustes\n(leak-free)",
          "Full\nProcrustes\n(leaky)",
          "Full\nProcrustes\n(leak-free)"]
vals   = [abs(r_pooled), abs(r_wf_pooled_preds), abs(r_fold_only),
          abs(r_aligned), abs(r_orth_lf),
          abs(r_full),    abs(r_full_lf)]
colors = ["#D9534F", "#5CB85C", "#F0AD4E",
          "#337AB7", "#1E5F9E",
          "#764BA2", "#4A2D6C"]
bars = ax.bar(labels, vals, color=colors, edgecolor="k", alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.15)
ax.set_ylabel("|r|  (PHQ decoding)")
ax.set_title("Pooled |r| across decoding strategies", fontweight="bold")
ax.tick_params(axis="x", labelsize=7)
ax.grid(axis="y", alpha=0.2)

# Panel [1,0]: PCA of orthogonal-aligned z
ax = ax2[1, 0]
for f in range(N_FOLDS):
    idx = fold_aln == f
    ax.scatter(z_aln_pc[idx, 0], z_aln_pc[idx, 1], color=fold_colors[f],
               alpha=0.7, s=25, edgecolors="k", linewidths=0.3,
               label=f"Fold {f}")
ax.set_xlabel(f"PC1 ({pca_aln.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_aln.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("z after orthogonal Procrustes\n(fold clusters = residual artefact)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel [1,1]: fold-id decodability from z across spaces
"""ax = ax2[1, 1]
acc_labels = ["Unaligned", "Orthogonal\nProcrustes", "Full\nProcrustes"]
acc_vals   = [acc_unaligned, acc_orth, acc_full]
acc_colors = ["#D9534F", "#337AB7", "#764BA2"]
bars = ax.bar(acc_labels, acc_vals, color=acc_colors, edgecolor="k", alpha=0.85)
for b, v in zip(bars, acc_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(1/N_FOLDS, color="k", ls="--", lw=1,
           label=f"Chance = {1/N_FOLDS:.2f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("LOO accuracy (multinomial LR)")
ax.set_title("Fold-id decodability FROM z\n"
             "(above chance ⇒ fold signal remains in latent space)",
             fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.2)"""

# Panel [1,2]: summary text
ax = ax2[1, 2]
ax.axis("off")
per_fold_orth    = ", ".join(f"{r:+.2f}" for r in r_aln_per_fold)
per_fold_full    = ", ".join(f"{r:+.2f}" for r in r_full_per_fold)
per_fold_full_lf = ", ".join(f"{r:+.2f}" for r in r_full_lf_per_fold)
txt = (
    f"Combo: {COMBO}\n"
    f"N with PHQ: {len(phq_aln)}\n\n"
    f"Orthogonal Procrustes (R only)\n"
    f"  leaky    pooled r   : {r_aligned:+.3f}   [{per_fold_orth}]\n"
    f"  leak-free pooled r  : {r_orth_lf:+.3f}\n\n"
    f"Full Procrustes (R + s + t)\n"
    f"  leaky    pooled r   : {r_full:+.3f}   [{per_fold_full}]\n"
    f"  leak-free pooled r  : {r_full_lf:+.3f}   [{per_fold_full_lf}]\n\n"
    f"Reference comparisons\n"
    f"─────────────────────\n"
    f"Unaligned pooled r     : {r_pooled:+.3f}\n"
    f"Within-fold (pooled)   : {r_wf_pooled_preds:+.3f}\n"
    f"Fold-mean-only (clean) : {r_fold_only:+.3f}\n\n"
    f"Fold-id decodability FROM z\n"
    #f"  unaligned   : {acc_unaligned:.3f}\n"
    #f"  orthogonal  : {acc_orth:.3f}\n"
    f"  full        : {acc_full:.3f}\n"
    f"  chance      : {1/N_FOLDS:.3f}\n\n"
    "Interpretation\n"
    "──────────────\n"
    "• Δ(unaligned − orthogonal) : rotation artefact\n"
    "• Δ(orthogonal − full)      : scale + mean-offset artefact\n"
    "• Δ(leaky − leak-free)      : alignment-leakage inflation\n"
    "• Fold-id acc. > chance     : fold signal still present\n"
)
ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace",
        fontsize=9, transform=ax.transAxes)

fig2.suptitle(
    f"Procrustes-aligned pooled PHQ decoding — IDRNN {COMBO}",
    fontweight="bold", fontsize=13)
fig2.tight_layout()
out2 = os.path.join(PLOT_DIR, "diagnose_phq_procrustes.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"\nSaved {out2}")

# ═══════════════════════════════════════════════════════════════════════════
# Multi-questionnaire decoding
#   For every numeric questionnaire column:
#     (full vs orthogonal Procrustes) × (random seed vs seed-averaged latents)
#     → pooled LOO ridge r, within-fold aggregated Pearson r, per-fold r
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("Multi-questionnaire decoding")
print("═" * 70)


def compute_Z_for_seed(fold, sd):
    """Wrapper: load model, extract per-subject z via LATENT_MODE pick."""
    run_base_f = f"runs_thalmann_{HP_VERSION}_{COMBO}/fold{fold}"
    model = _load_model(run_base_f, sd)
    mu    = _mu_tensor(model, xin_all, c_all)
    return _pick_trial(mu, c_all)


# (a) Random seed combo
mq_rng = np.random.default_rng(2026)
rand_seed_combo = [str(mq_rng.choice(sf)) for sf in valid_seeds_per_fold]
print(f"Random seed combo for single-seed variant: {rand_seed_combo}")
Z_per_enc_rand = [compute_Z_for_seed(f, sd) for f, sd in enumerate(rand_seed_combo)]

# (b) Seed-averaged latents (mean of z across all valid seeds per fold)
Z_per_enc_avg = []
for f, seeds_f in enumerate(valid_seeds_per_fold):
    Zs = np.stack([compute_Z_for_seed(f, sd) for sd in seeds_f])
    Z_per_enc_avg.append(Zs.mean(axis=0))
    print(f"  fold {f}: averaged z across {len(seeds_f)} seeds → {Z_per_enc_avg[f].shape}")


def align_heldout(Z_per_enc_list, alignment):
    """Align folds 1..N to fold 0 using `alignment` ∈ {'full','orth'}, then
    return the per-subject held-out z (each subject from their test-fold's
    aligned space).  Fit on ALL subjects (leaky; matches the pooled analyses)."""
    Z_ref = Z_per_enc_list[0]
    Z_al  = [Z_ref]
    for f in range(1, N_FOLDS):
        if alignment == "full":
            Z_a, _, _, _ = full_procrustes(Z_per_enc_list[f], Z_ref)
        elif alignment == "orth":
            R, _ = orthogonal_procrustes(Z_per_enc_list[f], Z_ref)
            Z_a  = Z_per_enc_list[f] @ R
        else:
            raise ValueError(alignment)
        Z_al.append(Z_a)
    Z_ho = np.zeros_like(Z_ref)
    for i, f in enumerate(fold_of_sub):
        Z_ho[i] = Z_al[int(f)][i]
    return Z_ho


def decode_one(Z_ho_all, y_all_subs, fold_of_all, estimator="ridge"):
    """Pooled LOO + within-fold aggregated + per-fold regression.
    Returns dict with r_pooled, r_wf_agg, r_fold0..r_fold{N-1} and
    (ridge only) the median alpha picked across the pooled LOO fits."""
    mask = ~np.isnan(y_all_subs)
    if mask.sum() < 30:
        return None
    Zh = Z_ho_all[mask]; y = y_all_subs[mask]; fl = fold_of_all[mask]
    r_p, _, diag   = loo_ridge(Zh, y, estimator=estimator, return_diag=True)
    r_wf, r_pf, _  = within_fold_ridge_agg(Zh, y, fl, N_FOLDS, estimator=estimator)
    out = {"r_pooled": r_p, "r_wf_agg": r_wf, "n": int(mask.sum())}
    if estimator == "ridge" and diag["alphas"] is not None:
        out["alpha_med"] = float(np.median(diag["alphas"]))
    for f in range(N_FOLDS):
        out[f"r_fold{f}"] = r_pf[f] if f < len(r_pf) else np.nan
    return out


# Pre-compute the four aligned held-out z matrices
Z_ho_map = {
    ("rand", "full"): align_heldout(Z_per_enc_rand, "full"),
    ("rand", "orth"): align_heldout(Z_per_enc_rand, "orth"),
    ("avg",  "full"): align_heldout(Z_per_enc_avg,  "full"),
    ("avg",  "orth"): align_heldout(Z_per_enc_avg,  "orth"),
}

# Group columns by prefix (strip trailing "_<digits>") and aggregate multi-item
# scales by row-wise mean.  Single-column groups (e.g. motiv_mem_0, age, rt) stay
# as-is but are renamed to their stripped prefix for consistency.
import re as _re
_item_pat = _re.compile(r"^(.+?)_(\d+)$")
_groups = {}
for c in quest.columns:
    if not pd.api.types.is_numeric_dtype(quest[c]):
        continue
    m = _item_pat.match(c)
    prefix = m.group(1) if m else c
    _groups.setdefault(prefix, []).append(c)

quest_scores = pd.DataFrame(index=quest.index)
for prefix, cols in _groups.items():
    quest_scores[prefix] = quest[cols].mean(axis=1) if len(cols) > 1 else quest[cols[0]]
score_cols = list(quest_scores.columns)
print(f"\nDecoding {len(score_cols)} questionnaire scores "
      f"(items aggregated per prefix; scales={ {k: len(v) for k,v in _groups.items() if len(v)>1} })")

ESTIMATORS = ("ridge", "linear")

rows = []
for col in score_cols:
    y_col = np.array([quest_scores.loc[sid, col] if sid in quest_scores.index
                      else np.nan for sid in sids_all])
    row = {"score": col}
    for (source, align), Zh in Z_ho_map.items():
        for est in ESTIMATORS:
            res = decode_one(Zh, y_col, fold_of_sub, estimator=est)
            if res is None:
                continue
            row["n"] = res["n"]
            prefix = f"{source}_{align}_{est}"
            row[f"{prefix}_pooled"] = res["r_pooled"]
            row[f"{prefix}_wf_agg"] = res["r_wf_agg"]
            for f in range(N_FOLDS):
                row[f"{prefix}_fold{f}"] = res[f"r_fold{f}"]
            if est == "ridge" and "alpha_med" in res:
                row[f"{prefix}_alpha_med"] = res["alpha_med"]
    if "n" not in row:
        continue
    rows.append(row)

df_mq = pd.DataFrame(rows)

# Rank by absolute avg_full_ridge_wf_agg for a quick "most decodable" view
rank_key = "avg_full_ridge_wf_agg"
if rank_key in df_mq.columns:
    df_mq = df_mq.sort_values(rank_key, key=lambda s: s.abs(),
                              ascending=False).reset_index(drop=True)

mq_csv = os.path.join(PLOT_DIR, f"multi_quest_decoding_{COMBO}.csv")
df_mq.to_csv(mq_csv, index=False)
print(f"Saved {mq_csv}")


# ─── Shuffled-y null distribution ────────────────────────────────────────────
# Permute y and redo the pooled + within-fold-aggregated regression to get the
# "zero point" for this LOO setup.  LOO-on-an-intercept-only-ish fit biases r
# slightly negative even under the null, so this calibrates interpretation.
print("\nComputing shuffled-y null (avg-full Procrustes z, ridge + linear)...")
N_PERM = 30
perm_rng = np.random.default_rng(7)
Zh_null  = Z_ho_map[("avg", "full")]

# Use PHQ_9 as representative score; fall back to the first score if missing.
null_score = "PHQ_9" if "PHQ_9" in score_cols else score_cols[0]
y_null_src = np.array([quest_scores.loc[sid, null_score]
                       if sid in quest_scores.index else np.nan
                       for sid in sids_all])
mask_null  = ~np.isnan(y_null_src)
Zh_null_m  = Zh_null[mask_null]
y_null_m   = y_null_src[mask_null]
fl_null    = fold_of_sub[mask_null]

null_res = {est: {"pooled": [], "wf_agg": []} for est in ESTIMATORS}
for p in range(N_PERM):
    y_shuf = perm_rng.permutation(y_null_m)
    for est in ESTIMATORS:
        r_p, _ = loo_ridge(Zh_null_m, y_shuf, estimator=est)
        r_wf, _, _ = within_fold_ridge_agg(Zh_null_m, y_shuf, fl_null, N_FOLDS,
                                           estimator=est)
        null_res[est]["pooled"].append(r_p)
        null_res[est]["wf_agg"].append(r_wf)

print(f"Null distributions (score={null_score!r}, n_perm={N_PERM}):")
for est in ESTIMATORS:
    for key in ("pooled", "wf_agg"):
        arr = np.array(null_res[est][key])
        print(f"  {est:<6s}  {key:<7s}  median={np.median(arr):+.3f}  "
              f"mean={arr.mean():+.3f} ± {arr.std():.3f}  "
              f"range=[{arr.min():+.3f}, {arr.max():+.3f}]")


print("\nTop results sorted by |avg_full_ridge_wf_agg|:")
show_cols = ["score", "n",
             "avg_full_ridge_pooled",  "avg_full_linear_pooled",
             "avg_full_ridge_wf_agg",  "avg_full_linear_wf_agg",
             "avg_orth_ridge_pooled",  "avg_orth_linear_pooled",
             "rand_full_ridge_pooled", "rand_full_linear_pooled",
             "avg_full_ridge_alpha_med"]
show_cols = [c for c in show_cols if c in df_mq.columns]
with pd.option_context("display.float_format", "{:+.3f}".format,
                       "display.width", 200,
                       "display.max_rows", 60):
    print(df_mq[show_cols].head(30).to_string(index=False))