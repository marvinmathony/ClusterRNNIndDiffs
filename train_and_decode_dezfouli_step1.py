#!/usr/bin/env python3
"""
Retrain IDRNN (step-1, lookup embeddings) and Vanilla RNN on ALL 101 Dezfouli
subjects, then decode diagnosis (Healthy / Depression / Bipolar) from the
per-subject latents.  Mirrors train_and_decode_thalmann_step1.py.

Design choices
--------------
• IDRNN HP from hp_search_results_dezfouli ranking #1: z_dim=8, hidden=5,
  step1_epochs=3000.  Vanilla hidden is set to z_dim (=8) so both models have
  the same number of decoding predictors.  (lmbd / enc_hidden are step-2
  knobs and do not affect step 1.)
• Inner 3-fold CV over BLOCKS (12 blocks) for IDRNN epoch selection; final
  refit on all blocks for the selected epoch count.
• Vanilla uses the existing 3-fold subject-CV via
  train_ablated_noblocks_palminteri_CV + train_final_ablated_model_after_cv.
• Seed selection over 3 seeds:
    – IDRNN best seed = argmax reconstruction specificity
        specificity = mean(NLL with mismatched z) − mean(NLL with own z)
    – Vanilla best seed = argmin final train NLL.
• Diagnosis decoding from per-subject latents:
    – 3-way multinomial LOO logistic
    – three pairwise binary LOO logistic (H-D, H-B, D-B)
    Reported as macro-AUROC + balanced accuracy with permutation null.
• PCA on the per-subject latent matrix (3 components for both models):
    – Per-PC LOO logistic decoding (3-way + each pairwise)
    – Pearson r between PC score and per-subject stay probability
      P(c_t == c_{t-1}).

Outputs (plots_dezfouli/step1_vs_vanilla/):
  latents_idrnn_step1_bestseed{N}.pt     — (101, z_dim) lookup embeddings
  latents_vanilla_bestseed{N}.pt         — (101, hidden) time-avg hidden
  seed_summary.csv                       — per-seed NLL / specificity
  decoding_results.csv                   — full-latent diagnosis decoding
  pc_decoding_results.csv                — per-PC diagnosis decoding
  pc_stay_corr.csv                       — per-PC × stay-prob correlation
  step1_vs_vanilla_decoding.png
  pc_decoding.png
  pc_stay_corr.png
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr

sys.path.insert(0, ".")
from modelsandtraining import (
    Decoder, LookupEncoderZ, LatentRNNz,
    train_ablated_noblocks_palminteri_CV, train_final_ablated_model_after_cv,
)

# ── CLI (so we can sweep z_dim / hidden / mask without copying the file) ──────
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--z_dim",   type=int, default=8)
_ap.add_argument("--hidden",  type=int, default=None,
                 help="Vanilla hidden + IDRNN decoder hidden. Default = z_dim.")
_ap.add_argument("--out_tag", type=str, default="_v2",
                 help="Suffix for OUT_DIR / CKPT_DIR. Use a unique tag to avoid "
                      "overwriting existing outputs (e.g. '_zsweep_z2').")
_ap.add_argument("--mask_input_dims", type=str, default="",
                 help="Comma-separated 0-indexed input dims to zero out before "
                      "training (e.g. '0' drops prev_choice).")
_ap.add_argument("--seeds", type=str, default=None,
                 help="Comma-separated seed override (default = built-in list of 10).")
_args, _ = _ap.parse_known_args()

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_TAG            = _args.out_tag
DGP                = "dezfouli"
DATA_DIR           = "data_dezfouli"
OUT_DIR            = f"plots_dezfouli/step1_vs_vanilla{RUN_TAG}"
CKPT_DIR           = f"runs_dezfouli_step1_vs_vanilla{RUN_TAG}"

Z_DIM              = _args.z_dim
HIDDEN             = _args.hidden if _args.hidden is not None else Z_DIM
A                  = 2
PCA_N_COMPONENTS   = min(3, Z_DIM)
STEP1_MAX_EPOCHS   = 3000
STEP1_PATIENCE     = 1000
VANILLA_MAX_EPOCHS = 3000
N_INNER_FOLDS      = 3
SEEDS              = ([int(s) for s in _args.seeds.split(",")] if _args.seeds
                      else [42, 56, 85, 100, 162, 200, 300, 400, 500, 600])
MASK_INPUT_DIMS    = ([int(d) for d in _args.mask_input_dims.split(",")]
                      if _args.mask_input_dims else [])
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
N_MISMATCH_SAMPLES = 50
N_PERM             = 2000
LOGREG_C           = 1.0

DIAG_LABELS = ["Healthy", "Depression", "Bipolar"]
DIAG_TO_INT = {d: i for i, d in enumerate(DIAG_LABELS)}
PAIRWISE = [("Healthy", "Depression"), ("Healthy", "Bipolar"), ("Depression", "Bipolar")]

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_subjects():
    """
    Reconstruct the full 101-subject arrays in a deterministic subid order.
    fold0's train+test covers every subject exactly once.

    Returns: xin (101, 12, 202, 2), c (101, 12, 202), subids (101,),
             diag (101,) categorical labels.
    """
    f0_tr_df = pd.read_csv(f"{DATA_DIR}/fold0/df_train.csv")
    f0_te_df = pd.read_csv(f"{DATA_DIR}/fold0/df_test.csv")
    tr_xin = np.load(f"{DATA_DIR}/fold0/xin_train.npy")
    te_xin = np.load(f"{DATA_DIR}/fold0/xin_test.npy")
    tr_c   = np.load(f"{DATA_DIR}/fold0/c_train.npy")
    te_c   = np.load(f"{DATA_DIR}/fold0/c_test.npy")

    all_xin  = np.concatenate([tr_xin, te_xin], axis=0)
    all_c    = np.concatenate([tr_c,   te_c],   axis=0)
    all_sids = np.concatenate([f0_tr_df["subid"].values, f0_te_df["subid"].values])
    all_diag = np.concatenate([f0_tr_df["diag"].values, f0_te_df["diag"].values])

    order  = np.argsort(all_sids)
    xin    = all_xin[order]
    c      = all_c[order]
    subids = all_sids[order]
    diag   = all_diag[order]
    assert xin.shape[0] == 101 and xin.shape[-1] == 2, f"Unexpected xin shape {xin.shape}"
    return xin, c, subids, diag


# ── IDRNN step-1 training with inner-CV epoch selection ───────────────────────
def _build_step1_model(n_participants, in_dim):
    encoder = LookupEncoderZ(n_participants=n_participants, z_dim=Z_DIM)
    decoder = Decoder(in_dim=in_dim, z_dim=Z_DIM, hid=HIDDEN, A=A)
    return LatentRNNz(
        encoder=encoder, decoder=decoder, hid=HIDDEN, z_dim=Z_DIM,
        in_dim=in_dim, A=A, block_structure=True,
    )


def _masked_block_nll(logits_all, c_all, block_mask):
    """Cross-entropy over a subset of blocks (padding -100 ignored)."""
    logits_sel = logits_all[:, block_mask]
    c_sel      = c_all[:,      block_mask]
    return F.cross_entropy(
        logits_sel.reshape(-1, A), c_sel.reshape(-1), ignore_index=-100
    )


def train_idrnn_step1_inner_cv(xin_t, c_t, seed):
    """3-fold CV over the 12 blocks. Returns selected_epoch and the mean
    val-NLL curve clipped to the shortest fold."""
    n_subj   = xin_t.shape[0]
    n_blocks = xin_t.shape[1]
    blocks   = np.arange(n_blocks)

    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    all_val_curves = []

    for fi, (tr_idx, va_idx) in enumerate(kf.split(blocks), start=1):
        train_mask = np.zeros(n_blocks, dtype=bool); train_mask[tr_idx] = True
        val_mask   = np.zeros(n_blocks, dtype=bool); val_mask[va_idx]   = True

        torch.manual_seed(seed * 1000 + fi); np.random.seed(seed * 1000 + fi)
        model = _build_step1_model(n_subj, in_dim=xin_t.shape[-1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        ids_all = torch.arange(n_subj, device=device)
        tm_t = torch.tensor(train_mask, device=device)
        vm_t = torch.tensor(val_mask,   device=device)

        val_curve = []; best_val = float("inf"); no_improve = 0
        for ep in range(1, STEP1_MAX_EPOCHS + 1):
            model.train(); opt.zero_grad()
            logits, _, _ = model(ids_all, xin_t)
            train_nll = _masked_block_nll(logits, c_t, tm_t)
            train_nll.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                logits_ev, _, _ = model(ids_all, xin_t)
                val_nll = _masked_block_nll(logits_ev, c_t, vm_t).item()
            val_curve.append(val_nll)

            if val_nll < best_val - 1e-5:
                best_val = val_nll; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= STEP1_PATIENCE:
                    break
            if ep % 200 == 0:
                print(f"  [seed {seed} fold {fi}] ep {ep}  "
                      f"train_nll={train_nll.item():.4f}  val_nll={val_nll:.4f}")
        all_val_curves.append(np.array(val_curve))
        print(f"  [seed {seed} fold {fi}] stopped at ep {len(val_curve)}, "
              f"best val_nll={best_val:.4f}")

    L = min(len(c) for c in all_val_curves)
    mat = np.stack([c[:L] for c in all_val_curves], axis=0)
    mean_val = mat.mean(axis=0)
    selected_epoch = int(np.argmin(mean_val)) + 1
    print(f"[seed {seed}] CV selected epoch = {selected_epoch}  "
          f"(mean val NLL = {mean_val[selected_epoch-1]:.4f})")
    return selected_epoch, mean_val


def train_idrnn_step1_final(xin_t, c_t, seed, n_epochs):
    """Retrain step 1 from scratch on all blocks for `n_epochs`."""
    n_subj = xin_t.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model = _build_step1_model(n_subj, in_dim=xin_t.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ids_all = torch.arange(n_subj, device=device)

    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        logits, _, _ = model(ids_all, xin_t)
        nll = F.cross_entropy(logits.reshape(-1, A), c_t.reshape(-1).long(),
                              ignore_index=-100)
        nll.backward(); opt.step()
        if ep % 200 == 0:
            print(f"  [seed {seed} final] ep {ep}  nll={nll.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits, _, _ = model(ids_all, xin_t)
        final_nll = F.cross_entropy(logits.reshape(-1, A), c_t.reshape(-1).long(),
                                    ignore_index=-100).item()
    emb = model.encoder.embed.weight.detach().cpu().numpy()
    return model, emb, final_nll


# ── Reconstruction specificity for step-1 lookup embeddings ───────────────────
def compute_specificity_step1(model, xin_t, c_t,
                              n_mismatch=N_MISMATCH_SAMPLES, rng_seed=0):
    """specificity = mismatched_NLL − matched_NLL  (higher → more identity in z)."""
    model.eval()
    n_subj = xin_t.shape[0]
    ids_all = torch.arange(n_subj, device=device)
    mask = (c_t.reshape(-1) >= 0).float()
    denom = mask.sum().clamp(min=1)

    with torch.no_grad():
        logits_m, _, _ = model(ids_all, xin_t)
        log_p = F.log_softmax(logits_m.reshape(-1, A), dim=-1)
        chosen = log_p.gather(-1, c_t.reshape(-1).clamp(min=0).long()
                                    .unsqueeze(-1)).squeeze(-1)
        matched_nll = -(chosen * mask).sum() / denom

        rng = np.random.default_rng(rng_seed)
        mis_vals = []
        for k in range(n_mismatch):
            perm_np = rng.permutation(n_subj)
            same = (perm_np == np.arange(n_subj))
            while same.any():
                perm_np[same] = (perm_np[same] + 1) % n_subj
                same = (perm_np == np.arange(n_subj))
            perm = torch.as_tensor(perm_np, device=device, dtype=torch.long)
            z_mis = model.encoder(perm)

            logits_list = []
            for b in range(xin_t.size(1)):
                block_in = model._append_task_emb(xin_t[:, b], b)
                out, _   = model.decoder(block_in, z_mis)
                logits_list.append(out)
            logits_mis = torch.stack(logits_list, dim=1)
            log_p_mis  = F.log_softmax(logits_mis.reshape(-1, A), dim=-1)
            chosen_mis = log_p_mis.gather(
                -1, c_t.reshape(-1).clamp(min=0).long().unsqueeze(-1)
            ).squeeze(-1)
            mis_vals.append(float(-(chosen_mis * mask).sum() / denom))

    mismatched_nll = float(np.mean(mis_vals))
    matched_val    = float(matched_nll)
    return mismatched_nll - matched_val, matched_val, mismatched_nll


# ── Vanilla training ──────────────────────────────────────────────────────────
def train_vanilla_full(xin_t, c_t, seed):
    """Inner 3-fold subject CV → epoch selection → final refit on all 101."""
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = xin_t.shape[-1]

    cv = train_ablated_noblocks_palminteri_CV(
        hidden=HIDDEN, in_dim=in_dim, A=A,
        X_train=xin_t, y_train=c_t.long(), device=device,
        epoch_nr=VANILLA_MAX_EPOCHS, lr=LR, nr_splits=N_INNER_FOLDS,
    )
    n_epochs = cv["selected_epoch"]
    print(f"  [seed {seed} vanilla] CV selected epoch = {n_epochs}")

    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _, _, _, _ = train_final_ablated_model_after_cv(
        hidden=HIDDEN, in_dim=in_dim, A=A,
        X_train=xin_t, y_train=c_t.long(), device=device,
        n_epochs=n_epochs, lr=LR,
        checkpoint_dir=None, loss_dir=None,
    )
    model.eval()
    with torch.no_grad():
        logits, _, hidden_tr = model(xin_t)        # hidden_tr: (B, n_blocks, T, hid)
        valid = (c_t >= 0).float()
        n_valid = valid.sum(dim=(1, 2)).clamp(min=1)
        h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2)) / n_valid.unsqueeze(-1)
        final_nll = F.cross_entropy(
            logits.reshape(-1, A), c_t.reshape(-1).long(), ignore_index=-100
        ).item()
    return model, h_avg.cpu().numpy(), final_nll, n_epochs


# ── Stay probability ───────────────────────────────────────────────────────────
def compute_stay_prob(c_np):
    """Per-subject P(c_t == c_{t-1}) over valid (non-padding) trial pairs."""
    n_subj = c_np.shape[0]
    p = np.zeros(n_subj)
    for s in range(n_subj):
        same, total = 0, 0
        for b in range(c_np.shape[1]):
            seq = c_np[s, b]
            valid = seq[seq != -100]
            if len(valid) >= 2:
                same  += int((valid[1:] == valid[:-1]).sum())
                total += len(valid) - 1
        p[s] = same / max(total, 1)
    return p


# ── LOO logistic decoding ─────────────────────────────────────────────────────
def _fit_logistic(X, y, multinomial):
    if multinomial:
        return LogisticRegression(
            max_iter=2000, C=LOGREG_C,
            multi_class="multinomial", solver="lbfgs",
        ).fit(X, y)
    return LogisticRegression(max_iter=2000, C=LOGREG_C).fit(X, y)


def loo_logistic(X, y, multinomial):
    """Returns (preds, probs).  probs has shape (n,) for binary, (n, K) multi."""
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
            # Map class probs back to canonical [0..K-1] indexing
            classes = clf.classes_
            full = np.zeros(probs.shape[1])
            full[classes] = clf.predict_proba(Xte)[0]
            probs[te] = full
        else:
            probs[te] = clf.predict_proba(Xte)[0, 1]
    return preds, probs


def decoding_metrics(y, preds, probs, multinomial, n_perm=N_PERM, rng_seed=0):
    """Balanced accuracy, AUROC, and label-permutation p-values."""
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    xin_np, c_np, subids, diag_str = load_all_subjects()
    diag_3way = np.array([DIAG_TO_INT[d] for d in diag_str], dtype=int)
    print(f"  xin {xin_np.shape}, c {c_np.shape}, subids ({len(subids)})")
    print(f"  diagnosis counts: " +
          ", ".join(f"{d}={int((diag_str == d).sum())}" for d in DIAG_LABELS))

    if MASK_INPUT_DIMS:
        # Zero specified input dims (preserves padding rows untouched: their
        # entire vector is -100 so this just leaves -100 in those spots).
        for d in MASK_INPUT_DIMS:
            xin_np[..., d] = np.where(xin_np[..., d] == -100.0,
                                       -100.0, 0.0).astype(xin_np.dtype)
        print(f"  masked input dims (zeroed): {MASK_INPUT_DIMS}")

    print(f"  config: Z_DIM={Z_DIM}, HIDDEN={HIDDEN}, "
          f"PCA_N_COMPONENTS={PCA_N_COMPONENTS}, n_seeds={len(SEEDS)}, "
          f"OUT_DIR={OUT_DIR}")

    xin_t = torch.as_tensor(xin_np, dtype=torch.float32, device=device)
    c_t   = torch.as_tensor(c_np,   dtype=torch.long,    device=device)

    stay_p = compute_stay_prob(c_np)
    print(f"  stay-prob: mean={stay_p.mean():.3f}  range=[{stay_p.min():.3f}, {stay_p.max():.3f}]")

    # ── IDRNN step-1 ──────────────────────────────────────────────────────────
    idrnn_results = {}
    for seed in SEEDS:
        print(f"\n══ IDRNN step-1 seed {seed} ══")
        sel_ep, val_curve = train_idrnn_step1_inner_cv(xin_t, c_t, seed)
        model, emb, final_nll = train_idrnn_step1_final(xin_t, c_t, seed, n_epochs=sel_ep)
        spec, m_nll, mis_nll = compute_specificity_step1(model, xin_t, c_t, rng_seed=seed)
        print(f"  [seed {seed}] final_train_nll={final_nll:.4f}  "
              f"matched={m_nll:.4f}  mismatched={mis_nll:.4f}  specificity={spec:+.4f}")
        idrnn_results[seed] = dict(
            emb=emb, selected_epoch=sel_ep, final_nll=final_nll,
            matched_nll=m_nll, mismatched_nll=mis_nll, specificity=spec,
            val_curve=val_curve,
            model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        )
        torch.save(
            {"emb": emb, "selected_epoch": sel_ep, "seed": seed,
             "specificity": spec,
             "model_state": idrnn_results[seed]["model_state"]},
            os.path.join(CKPT_DIR, f"idrnn_step1_seed{seed}.pt"),
        )

    # ── Vanilla ────────────────────────────────────────────────────────────────
    vanilla_results = {}
    for seed in SEEDS:
        print(f"\n══ Vanilla seed {seed} ══")
        v_model, h_lat, final_nll, n_ep = train_vanilla_full(xin_t, c_t, seed)
        v_state = {k: v.detach().cpu() for k, v in v_model.state_dict().items()}
        print(f"  [seed {seed}] final_train_nll={final_nll:.4f}  n_epochs={n_ep}")
        vanilla_results[seed] = dict(
            h=h_lat, final_nll=final_nll, selected_epoch=n_ep, model_state=v_state,
        )
        torch.save(
            {"h": h_lat, "selected_epoch": n_ep, "seed": seed,
             "final_nll": final_nll, "model_state": v_state},
            os.path.join(CKPT_DIR, f"vanilla_seed{seed}.pt"),
        )

    # ── Seed selection ────────────────────────────────────────────────────────
    best_idrnn_seed   = max(idrnn_results,   key=lambda s: idrnn_results[s]["specificity"])
    best_vanilla_seed = min(vanilla_results, key=lambda s: vanilla_results[s]["final_nll"])
    print(f"\n── Seed selection ──")
    print(f"  IDRNN best seed (by specificity) : {best_idrnn_seed}   "
          f"spec={idrnn_results[best_idrnn_seed]['specificity']:+.4f}")
    print(f"  Vanilla best seed (by train NLL) : {best_vanilla_seed}   "
          f"nll={vanilla_results[best_vanilla_seed]['final_nll']:.4f}")

    rows = []
    for s in SEEDS:
        r = idrnn_results[s]
        rows.append({
            "model": "IDRNN", "seed": s, "selected_epoch": r["selected_epoch"],
            "final_nll": r["final_nll"],
            "matched_nll": r["matched_nll"], "mismatched_nll": r["mismatched_nll"],
            "specificity": r["specificity"], "is_best": s == best_idrnn_seed,
        })
    for s in SEEDS:
        r = vanilla_results[s]
        rows.append({
            "model": "Vanilla", "seed": s, "selected_epoch": r["selected_epoch"],
            "final_nll": r["final_nll"],
            "matched_nll": np.nan, "mismatched_nll": np.nan, "specificity": np.nan,
            "is_best": s == best_vanilla_seed,
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "seed_summary.csv"), index=False)

    z_idrnn   = idrnn_results[best_idrnn_seed]["emb"]        # (101, z_dim)
    h_vanilla = vanilla_results[best_vanilla_seed]["h"]      # (101, hidden)
    torch.save({
        "z": z_idrnn, "subids": subids, "diag": diag_str,
        "stay_prob": stay_p, "seed": best_idrnn_seed,
        "model_state":  idrnn_results[best_idrnn_seed]["model_state"],
        "z_dim": Z_DIM, "hidden": HIDDEN, "A": A,
        "base_in_dim": xin_t.shape[-1],
    }, os.path.join(OUT_DIR, f"latents_idrnn_step1_bestseed{best_idrnn_seed}.pt"))
    torch.save({
        "h": h_vanilla, "subids": subids, "diag": diag_str,
        "stay_prob": stay_p, "seed": best_vanilla_seed,
        "model_state":  vanilla_results[best_vanilla_seed]["model_state"],
        "hidden": HIDDEN, "A": A, "base_in_dim": xin_t.shape[-1],
    }, os.path.join(OUT_DIR, f"latents_vanilla_bestseed{best_vanilla_seed}.pt"))

    # ── Diagnosis decoding (full latent) ──────────────────────────────────────
    print(f"\n── Full-latent diagnosis decoding (LOO logistic, n={len(diag_3way)}) ──")
    feats = {"IDRNN": z_idrnn, "Vanilla": h_vanilla}
    decode_rows = []

    contrasts = [("3way", None)] + [
        (f"{a}_vs_{b}", (DIAG_TO_INT[a], DIAG_TO_INT[b])) for a, b in PAIRWISE
    ]
    for name, X in feats.items():
        for cname, sel in contrasts:
            if sel is None:
                Xc, yc, multi = X, diag_3way, True
            else:
                m = np.isin(diag_3way, sel)
                Xc = X[m]
                yc = (diag_3way[m] == sel[1]).astype(int)
                multi = False
            preds, probs = loo_logistic(Xc, yc, multinomial=multi)
            mets = decoding_metrics(yc, preds, probs, multinomial=multi, rng_seed=0)
            print(f"  {name:<8} {cname:<22} bacc={mets['bacc']:.3f} "
                  f"({mets['p_bacc']:.3g} {sig_stars(mets['p_bacc'])})  "
                  f"AUROC={mets['auroc']:.3f} "
                  f"({mets['p_auroc']:.3g} {sig_stars(mets['p_auroc'])})")
            decode_rows.append(
                {"model": name, "contrast": cname, "n": int(len(yc)), **mets}
            )
    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)

    # ── PCA + per-PC analyses ─────────────────────────────────────────────────
    print(f"\n── PCA + per-PC analyses ──")
    pc_decoding_rows = []
    pc_stay_rows     = []

    pca_objs = {}
    for name, X in feats.items():
        Xs = StandardScaler().fit_transform(X)
        n_comp = min(PCA_N_COMPONENTS, X.shape[1])
        pca = PCA(n_components=n_comp).fit(Xs)
        scores = pca.transform(Xs)                       # (101, n_comp)
        pca_objs[name] = (pca, scores)
        print(f"\n  [{name}] explained variance: " +
              ", ".join(f"PC{i+1}={pca.explained_variance_ratio_[i]:.2f}"
                        for i in range(n_comp)))

        for k in range(n_comp):
            pc = scores[:, k:k+1]
            for cname, sel in contrasts:
                if sel is None:
                    Xc, yc, multi = pc, diag_3way, True
                else:
                    m = np.isin(diag_3way, sel)
                    Xc = pc[m]
                    yc = (diag_3way[m] == sel[1]).astype(int)
                    multi = False
                preds, probs = loo_logistic(Xc, yc, multinomial=multi)
                mets = decoding_metrics(yc, preds, probs, multinomial=multi, rng_seed=k)
                pc_decoding_rows.append({
                    "model": name, "pc": k + 1,
                    "explained_var": float(pca.explained_variance_ratio_[k]),
                    "contrast": cname, "n": int(len(yc)), **mets,
                })

            r, p = pearsonr(scores[:, k], stay_p)
            pc_stay_rows.append({
                "model": name, "pc": k + 1,
                "explained_var": float(pca.explained_variance_ratio_[k]),
                "r": float(r), "p": float(p),
            })

    pc_dec_df  = pd.DataFrame(pc_decoding_rows)
    pc_stay_df = pd.DataFrame(pc_stay_rows)
    pc_dec_df.to_csv(os.path.join(OUT_DIR, "pc_decoding_results.csv"), index=False)
    pc_stay_df.to_csv(os.path.join(OUT_DIR, "pc_stay_corr.csv"), index=False)

    # Print summary of per-PC stay correlations
    print("\n  Per-PC × stay-prob correlation:")
    for _, row in pc_stay_df.iterrows():
        print(f"    {row['model']:<8} PC{int(row['pc'])}  "
              f"r={row['r']:+.3f}  p={row['p']:.3g} {sig_stars(row['p'])}  "
              f"(var={row['explained_var']:.2f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1) Full-latent diagnosis decoding
    df = decode_df[decode_df["contrast"].isin(["3way"] + [c for c, _ in contrasts[1:]])]
    contrast_order = ["3way"] + [c for c, _ in contrasts[1:]]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(contrast_order)); w = 0.35
    c1, c2 = "#4C72B0", "#DD8452"
    for i, m in enumerate(["IDRNN", "Vanilla"]):
        sub = df[df["model"] == m].set_index("contrast").loc[contrast_order]
        offset = (-w/2 if i == 0 else w/2)
        ax.bar(x + offset, sub["auroc"].values, w,
               color=(c1 if i == 0 else c2), edgecolor="k", linewidth=0.5,
               label=f"{m} (seed {best_idrnn_seed if m=='IDRNN' else best_vanilla_seed})")
        for j, (cn, row) in enumerate(sub.iterrows()):
            ax.text(j + offset, row["auroc"] + 0.01, sig_stars(row["p_auroc"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=(c1 if i == 0 else c2))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(contrast_order, fontsize=9)
    ax.set_ylabel("AUROC (LOO logistic)")
    ax.set_title("Diagnosis decoding from per-subject latents")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step1_vs_vanilla_decoding.png"), dpi=150)
    plt.close(fig)

    # 2) Per-PC AUROC for the 3-way contrast (one panel per model)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(axes, ["IDRNN", "Vanilla"]):
        sub = pc_dec_df[(pc_dec_df["model"] == m) & (pc_dec_df["contrast"] == "3way")]
        sub = sub.sort_values("pc")
        ax.bar(sub["pc"], sub["auroc"], color="#4C72B0", edgecolor="k", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.text(row["pc"], row["auroc"] + 0.01, sig_stars(row["p_auroc"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(f"{m} — per-PC 3-way diagnosis AUROC")
        ax.set_xlabel("PC"); ax.set_xticks(sub["pc"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pc_decoding.png"), dpi=150)
    plt.close(fig)

    # 3) Per-PC × stay-prob correlation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(axes, ["IDRNN", "Vanilla"]):
        sub = pc_stay_df[pc_stay_df["model"] == m].sort_values("pc")
        bars = ax.bar(sub["pc"], sub["r"], color="#55A868", edgecolor="k", linewidth=0.5)
        for bar, p in zip(bars, sub["p"]):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (0.02 if y >= 0 else -0.05),
                    sig_stars(p), ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(f"{m} — per-PC × stay-prob")
        ax.set_xlabel("PC"); ax.set_xticks(sub["pc"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pc_stay_corr.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
