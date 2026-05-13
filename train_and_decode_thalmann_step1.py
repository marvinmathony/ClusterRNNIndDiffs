#!/usr/bin/env python3
"""
Retrain IDRNN (step-1 only, lookup embeddings) and Vanilla RNN on ALL 236
Thalmann subjects (no outer CV / train-test split), then decode 6 questionnaire
scales from the latents via LOO ridge.

Key design choices
------------------
• IDRNN HP: z_dim=10, hidden=5, task_emb_dim=2 — mirrors the step-1 settings of
  the `uw05_lmbd005_eh5_h5_z10` combo that already showed PHQ behavioural
  signal in step-1 lookup embeddings (analyze_thalmann_step1_behavior.py).
  lmbd/unif_weight are step-2 knobs and do not affect step 1.
• IDRNN epoch selection: inner 3-fold CV over task-0 blocks. Task-1 block
  (block 30) is always in train (single big block, most informative). At each
  epoch we compute train NLL on train blocks and val NLL on held-out blocks;
  selected_epoch = argmin mean val NLL across folds. Then we retrain from
  scratch on ALL 31 blocks for selected_epoch epochs.
• Vanilla: existing train_ablated_noblocks_palminteri_CV + final refit.
• Seed selection: N_SEEDS seeds each.
  - IDRNN best seed = argmax reconstruction specificity
      specificity = mean(NLL with wrong-subject z) − mean(NLL with own z)
  - Vanilla best seed = argmin final train NLL (no natural specificity analog).
• Decoding: LOO RidgeCV with z-scoring; Pearson r + Steiger z-test for
  IDRNN-vs-Vanilla per scale.

Outputs (plots_thalmann/step1_vs_vanilla/):
  latents_idrnn_step1_bestseed{N}.pt       — (236, z_dim) lookup embeddings
  latents_vanilla_bestseed{N}.pt           — (236, hidden) time-avg hidden
  seed_summary.csv                         — per-seed NLL, specificity
  decoding_results.csv                     — per-scale r, p, Steiger
  step1_vs_vanilla_decoding.png            — bar plot
"""

import os, sys, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
from scipy.stats import pearsonr, norm
from scipy.stats import pearsonr, norm, f as f_dist

sys.path.insert(0, ".")
from modelsandtraining import (
    Decoder, LookupEncoderZ, LatentRNNz, AblatedRNN,
    train_ablated_noblocks_palminteri_CV, train_final_ablated_model_after_cv,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DGP               = "thalmann"
DATA_DIR          = "data_thalmann"
OUT_DIR           = "plots_thalmann/step1_vs_vanilla"
CKPT_DIR          = "runs_thalmann_step1_vs_vanilla"

Z_DIM             = 10
HIDDEN            = 3
TASK_EMB_DIM      = 2
A                 = 4
STEP1_MAX_EPOCHS  = 5000
STEP1_PATIENCE    = 2000      # early stop inner-CV if mean-val-NLL doesn't
                              # improve for this many epochs
VANILLA_MAX_EPOCHS = 5000
N_INNER_FOLDS     = 3
N_SEEDS           = 3
SEEDS             = [42, 56, 85]
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
N_MISMATCH_SAMPLES = 50
STEIGER_N         = 236

SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5\nOpenness"),
    # CFA compound factors (loaded from CFA_compound_questionnaire_factors_s1.csv)
    "AxDep":     (["AxDep"],                                      "CFA\nAxDep"),
    "posMood":   (["posMood"],                                    "CFA\nposMood"),
    "negMood":   (["negMood"],                                    "CFA\nnegMood"),
    "Exp":       (["Exp"],                                        "CFA\nExploration"),
}
SCALE_KEYS = list(SCALES.keys())

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_subjects():
    """
    Reconstruct the full 236-subject arrays in the canonical `shared_ids` order
    used by load_thalmann.py. We rebuild from fold0's train+test which covers
    every subject exactly once.

    Returns: xin (236, 31, 200, 5), c (236, 31, 200), subids (236,),
             task_ids_per_block (31,)
    """
    # Don't import load_thalmann (it regenerates files at import time).
    # Read the fold0 arrays directly, reindex by subid → deterministic order.
    f0_tr_df = pd.read_csv(f"{DATA_DIR}/fold0/df_train.csv")
    f0_te_df = pd.read_csv(f"{DATA_DIR}/fold0/df_test.csv")
    tr_xin = np.load(f"{DATA_DIR}/fold0/xin_train.npy")
    te_xin = np.load(f"{DATA_DIR}/fold0/xin_test.npy")
    tr_c   = np.load(f"{DATA_DIR}/fold0/c_train.npy")
    te_c   = np.load(f"{DATA_DIR}/fold0/c_test.npy")

    all_xin  = np.concatenate([tr_xin, te_xin], axis=0)
    all_c    = np.concatenate([tr_c,   te_c],   axis=0)
    all_sids = np.concatenate([f0_tr_df["subid"].values, f0_te_df["subid"].values])

    # Sort by subid for a deterministic order
    order = np.argsort(all_sids)
    xin    = all_xin[order]
    c      = all_c[order]
    subids = all_sids[order]

    task_ids = np.load(f"{DATA_DIR}/task_ids_per_block.npy")
    assert xin.shape == (236, 31, 200, 5), f"Unexpected xin shape {xin.shape}"
    return xin, c, subids, task_ids


# ── IDRNN step-1 training with inner-CV epoch selection ───────────────────────
def _build_step1_model(n_participants, in_dim):
    """Fresh LatentRNNz (LookupEncoderZ + Decoder) with task embedding."""
    encoder = LookupEncoderZ(n_participants=n_participants, z_dim=Z_DIM)
    dec_in_dim = in_dim + TASK_EMB_DIM
    decoder = Decoder(in_dim=dec_in_dim, z_dim=Z_DIM, hid=HIDDEN, A=A)
    model = LatentRNNz(
        encoder=encoder, decoder=decoder, hid=HIDDEN, z_dim=Z_DIM,
        in_dim=dec_in_dim, A=A, block_structure=True,
        n_tasks=2, task_emb_dim=TASK_EMB_DIM,
    )
    return model


def _masked_block_nll(logits_all, c_all, block_mask):
    """Cross-entropy over a subset of blocks.
    logits_all : (B, 31, T, A)
    c_all      : (B, 31, T) long, -100 = padding
    block_mask : (31,) bool
    """
    logits_sel = logits_all[:, block_mask]           # (B, nb, T, A)
    c_sel      = c_all[:,      block_mask]            # (B, nb, T)
    return F.cross_entropy(
        logits_sel.reshape(-1, A), c_sel.reshape(-1), ignore_index=-100
    )


def train_idrnn_step1_inner_cv(xin_t, c_t, task_ids_tensor, seed):
    """
    3-fold CV over task-0 blocks (task-1 always train). Returns per-fold
    val-NLL curves (clipped to effective length after early stop) and the
    selected epoch = argmin mean val NLL.
    """
    n_subj = xin_t.shape[0]
    task0_blocks = np.where(task_ids_tensor.cpu().numpy() == 0)[0]
    task1_blocks = np.where(task_ids_tensor.cpu().numpy() == 1)[0]

    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    all_val_curves = []

    for fi, (tr_idx, va_idx) in enumerate(kf.split(task0_blocks), start=1):
        tr_blocks = np.concatenate([task0_blocks[tr_idx], task1_blocks])
        va_blocks = task0_blocks[va_idx]
        train_mask = np.zeros(31, dtype=bool); train_mask[tr_blocks] = True
        val_mask   = np.zeros(31, dtype=bool); val_mask[va_blocks]   = True

        torch.manual_seed(seed * 1000 + fi)
        np.random.seed(seed * 1000 + fi)
        model = _build_step1_model(n_subj, in_dim=xin_t.shape[-1]).to(device)
        model.set_task_ids(task_ids_tensor)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        ids_all = torch.arange(n_subj, device=device)
        tm_t = torch.tensor(train_mask, device=device)
        vm_t = torch.tensor(val_mask,   device=device)

        val_curve = []
        best_val = float("inf"); epochs_no_improve = 0
        for ep in range(1, STEP1_MAX_EPOCHS + 1):
            model.train(); opt.zero_grad()
            logits, _, _ = model(ids_all, xin_t)       # (B, 31, T, A)
            train_nll = _masked_block_nll(logits, c_t, tm_t)
            train_nll.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                logits_ev, _, _ = model(ids_all, xin_t)
                val_nll = _masked_block_nll(logits_ev, c_t, vm_t).item()
            val_curve.append(val_nll)

            if val_nll < best_val - 1e-5:
                best_val = val_nll; epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= STEP1_PATIENCE:
                    break
            if ep % 200 == 0:
                print(f"  [seed {seed} fold {fi}] ep {ep}  "
                      f"train_nll={train_nll.item():.4f}  val_nll={val_nll:.4f}")
        all_val_curves.append(np.array(val_curve))
        print(f"  [seed {seed} fold {fi}] stopped at ep {len(val_curve)}, "
              f"best val_nll={best_val:.4f}")

    # Align to the shortest effective length, then average
    L = min(len(c) for c in all_val_curves)
    mat = np.stack([c[:L] for c in all_val_curves], axis=0)      # (folds, L)
    mean_val = mat.mean(axis=0)
    selected_epoch = int(np.argmin(mean_val)) + 1
    print(f"[seed {seed}] CV selected epoch = {selected_epoch}  "
          f"(mean val NLL = {mean_val[selected_epoch-1]:.4f})")
    return selected_epoch, mean_val


def train_idrnn_step1_final(xin_t, c_t, task_ids_tensor, seed, n_epochs):
    """Retrain step 1 from scratch on all 31 blocks for `n_epochs`."""
    n_subj = xin_t.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model = _build_step1_model(n_subj, in_dim=xin_t.shape[-1]).to(device)
    model.set_task_ids(task_ids_tensor)
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
        final_nll = F.cross_entropy(logits.reshape(-1, A),
                                    c_t.reshape(-1).long(),
                                    ignore_index=-100).item()
    emb = model.encoder.embed.weight.detach().cpu().numpy()      # (236, z_dim)
    return model, emb, final_nll


# ── Reconstruction specificity for step-1 lookup embeddings ───────────────────
def compute_specificity_step1(model, xin_t, c_t, n_mismatch=N_MISMATCH_SAMPLES,
                              rng_seed=0):
    """
    Matched NLL:    predict each subject's choices with their own lookup z.
    Mismatched NLL: predict with a randomly permuted z (no self-pairing).
    specificity = mismatched − matched   (higher → more subject-specific).
    Averaged over `n_mismatch` random permutations. Padding (-100) ignored.
    """
    model.eval()
    n_subj = xin_t.shape[0]
    ids_all = torch.arange(n_subj, device=device)
    mask = (c_t.reshape(-1) >= 0).float()
    denom = mask.sum().clamp(min=1)

    with torch.no_grad():
        logits_m, z_own, _ = model(ids_all, xin_t)
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
            z_mis = model.encoder(perm)                     # (B, z_dim)

            # Re-run decoder per-block with z_mis but subject's own inputs
            logits_list = []
            for b in range(xin_t.size(1)):
                block_in = model._append_task_emb(xin_t[:, b], b)
                out, _   = model.decoder(block_in, z_mis)   # (B, T, A)
                logits_list.append(out)
            logits_mis = torch.stack(logits_list, dim=1)    # (B, 31, T, A)
            log_p_mis  = F.log_softmax(logits_mis.reshape(-1, A), dim=-1)
            chosen_mis = log_p_mis.gather(
                -1, c_t.reshape(-1).clamp(min=0).long().unsqueeze(-1)
            ).squeeze(-1)
            mis_vals.append(float(-(chosen_mis * mask).sum() / denom))

    mismatched_nll = float(np.mean(mis_vals))
    matched_val    = float(matched_nll)
    return mismatched_nll - matched_val, matched_val, mismatched_nll


# ── Vanilla training ──────────────────────────────────────────────────────────
def train_vanilla_full(xin_t, c_t, task_ids_tensor, seed):
    """
    Inner-CV epoch selection + final refit on all 236 subjects.
    Returns (model, h_latents (236, HIDDEN), final_train_nll).
    """
    n_subj = xin_t.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)

    in_dim = xin_t.shape[-1]
    dec_in_dim = in_dim + TASK_EMB_DIM

    cv = train_ablated_noblocks_palminteri_CV(
        hidden=HIDDEN, in_dim=in_dim, A=A, X_train=xin_t, y_train=c_t.long(),
        device=device, epoch_nr=VANILLA_MAX_EPOCHS, lr=LR,
        nr_splits=N_INNER_FOLDS,
        dec_in_dim=dec_in_dim, n_tasks=2, task_emb_dim=TASK_EMB_DIM,
        task_ids=task_ids_tensor,
    )
    n_epochs = cv["selected_epoch"]
    print(f"  [seed {seed} vanilla] CV selected epoch = {n_epochs}")

    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _, _, _, _ = train_final_ablated_model_after_cv(
        hidden=HIDDEN, in_dim=in_dim, A=A, X_train=xin_t, y_train=c_t.long(),
        device=device, n_epochs=n_epochs, lr=LR,
        checkpoint_dir=None, loss_dir=None,
        dec_in_dim=dec_in_dim, n_tasks=2, task_emb_dim=TASK_EMB_DIM,
        task_ids=task_ids_tensor,
    )
    model.eval()
    with torch.no_grad():
        logits, _, hidden_tr = model(xin_t)        # hidden_tr: (B, 31, T, hid)
        valid = (c_t >= 0).float()                  # (B, 31, T)
        n_valid = valid.sum(dim=(1, 2)).clamp(min=1)
        h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2)) / n_valid.unsqueeze(-1)
        final_nll = F.cross_entropy(
            logits.reshape(-1, A), c_t.reshape(-1).long(), ignore_index=-100
        ).item()
    return model, h_avg.cpu().numpy(), final_nll, n_epochs


# ── LOO ridge + Steiger ───────────────────────────────────────────────────────
def loo_ridge(Z, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    """Leave-one-out RidgeCV with per-fold inner CV alpha selection.
    Standardizes features and target on each training fold (avoids fold leakage)."""
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
    """
    In-sample OLS: fit on all data, score on all data.
    Returns (r, p, preds) where
      r     = sqrt(R²)              — multiple correlation coefficient (≥ 0)
      p     = F-test p-value         — valid null test for 'any linear combo
                                        of Z predicts y'
      preds = in-sample predictions  — same samples the model was fit on
    """
    n, p = Z.shape
    clf   = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F      = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val  = float(f_dist.sf(F, dfn, dfd))
    return float(np.sqrt(max(r2, 0.0))), p_val, preds


def steiger_test(r12, r13, r23, n):
    r12 = np.clip(r12, -0.9999, 0.9999)
    r13 = np.clip(r13, -0.9999, 0.9999)
    r23 = np.clip(r23, -0.9999, 0.9999)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rsq_bar = (r12**2 + r13**2) / 2
    f = (1 - r23) / (2 * (1 - rsq_bar))
    h = (1 - f * rsq_bar) / (1 - rsq_bar)
    var = (2 * (1 - r23) / (n - 1)) * h
    if var <= 0:
        return 0.0, 1.0
    z = (z12 - z13) / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    xin_np, c_np, subids, task_ids_np = load_all_subjects()
    xin_t = torch.as_tensor(xin_np, dtype=torch.float32, device=device)
    c_t   = torch.as_tensor(c_np,   dtype=torch.long,    device=device)
    task_ids_tensor = torch.as_tensor(task_ids_np, dtype=torch.long, device=device)
    print(f"  xin {tuple(xin_t.shape)}, c {tuple(c_t.shape)}, "
          f"subids ordered ({len(subids)})")

    # Questionnaire (item-level scales + CFA composite factors)
    quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    factors = (pd.read_csv("data/CFA_compound_questionnaire_factors_s1.csv")
                 .set_index("ID")[["AxDep", "posMood", "negMood", "Exp"]])
    quest = quest.join(factors, how="left")
    for k, (items, _) in SCALES.items():
        quest[k] = quest[items].mean(axis=1)

    # ── IDRNN step-1: inner CV + final refit per seed ─────────────────────────
    idrnn_results = {}
    for seed in SEEDS:
        print(f"\n══ IDRNN step-1 seed {seed} ══")
        sel_ep, val_curve = train_idrnn_step1_inner_cv(
            xin_t, c_t, task_ids_tensor, seed
        )
        model, emb, final_nll = train_idrnn_step1_final(
            xin_t, c_t, task_ids_tensor, seed, n_epochs=sel_ep
        )
        spec, m_nll, mis_nll = compute_specificity_step1(
            model, xin_t, c_t, rng_seed=seed
        )
        print(f"  [seed {seed}] final_train_nll={final_nll:.4f}  "
              f"matched={m_nll:.4f}  mismatched={mis_nll:.4f}  "
              f"specificity={spec:+.4f}")
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

    # ── Vanilla: inner CV + final refit per seed ──────────────────────────────
    vanilla_results = {}
    for seed in SEEDS:
        print(f"\n══ Vanilla seed {seed} ══")
        v_model, h_lat, final_nll, n_ep = train_vanilla_full(
            xin_t, c_t, task_ids_tensor, seed
        )
        v_state = {k: v.detach().cpu() for k, v in v_model.state_dict().items()}
        print(f"  [seed {seed}] final_train_nll={final_nll:.4f}  n_epochs={n_ep}")
        vanilla_results[seed] = dict(
            h=h_lat, final_nll=final_nll, selected_epoch=n_ep,
            model_state=v_state,
        )
        torch.save(
            {"h": h_lat, "selected_epoch": n_ep, "seed": seed,
             "final_nll": final_nll, "model_state": v_state},
            os.path.join(CKPT_DIR, f"vanilla_seed{seed}.pt"),
        )

    # ── Seed selection ────────────────────────────────────────────────────────
    best_idrnn_seed = max(idrnn_results, key=lambda s: idrnn_results[s]["specificity"])
    best_vanilla_seed = min(vanilla_results, key=lambda s: vanilla_results[s]["final_nll"])
    print(f"\n── Seed selection ──")
    print(f"  IDRNN best seed (by specificity) : {best_idrnn_seed}   "
          f"spec={idrnn_results[best_idrnn_seed]['specificity']:+.4f}")
    print(f"  Vanilla best seed (by train NLL) : {best_vanilla_seed}   "
          f"nll={vanilla_results[best_vanilla_seed]['final_nll']:.4f}")

    # Save seed summary
    rows = []
    for s in SEEDS:
        r = idrnn_results[s]
        rows.append({
            "model": "IDRNN", "seed": s, "selected_epoch": r["selected_epoch"],
            "final_nll": r["final_nll"],
            "matched_nll": r["matched_nll"],
            "mismatched_nll": r["mismatched_nll"],
            "specificity": r["specificity"],
            "is_best": s == best_idrnn_seed,
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

    # Save selected latents (bundle full model state for downstream rollouts)
    z_idrnn   = idrnn_results[best_idrnn_seed]["emb"]        # (236, z_dim)
    h_vanilla = vanilla_results[best_vanilla_seed]["h"]      # (236, hidden)
    torch.save({
        "z": z_idrnn, "subids": subids, "seed": best_idrnn_seed,
        "model_state":  idrnn_results[best_idrnn_seed]["model_state"],
        "z_dim":        Z_DIM,
        "hidden":       HIDDEN,
        "task_emb_dim": TASK_EMB_DIM,
        "A":            A,
        "base_in_dim":  xin_t.shape[-1],   # e.g. 5 for thalmann
    }, os.path.join(OUT_DIR, f"latents_idrnn_step1_bestseed{best_idrnn_seed}.pt"))
    torch.save({
        "h": h_vanilla, "subids": subids, "seed": best_vanilla_seed,
        "model_state":  vanilla_results[best_vanilla_seed]["model_state"],
        "hidden":       HIDDEN,
        "task_emb_dim": TASK_EMB_DIM,
        "A":            A,
        "base_in_dim":  xin_t.shape[-1],
    }, os.path.join(OUT_DIR, f"latents_vanilla_bestseed{best_vanilla_seed}.pt"))

    # ── Decoding ──────────────────────────────────────────────────────────────
    # Two estimates per (scale, latent) — in-sample OLS multiple-r (biased
    # upward by sqrt(k/(n-1)) under the null), and LOO RidgeCV out-of-sample r
    # (unbiased; positive only if real signal beats overfitting).
    print(f"\n── Decoding: in-sample OLS + LOO RidgeCV ──")
    decode_rows = []
    z_dim_i = z_idrnn.shape[1]
    z_dim_v = h_vanilla.shape[1]
    for scale in SCALE_KEYS:
        y_all = quest.reindex(subids)[scale].values.astype(float)
        mask = ~np.isnan(y_all)
        if mask.sum() < 30:
            print(f"  {scale}: too few valid (n={mask.sum()}), skipping")
            continue
        Zi = z_idrnn[mask];   Hv = h_vanilla[mask];   yy = y_all[mask]
        n = int(mask.sum())

        # In-sample OLS (existing analysis — matches step1_vs_vanilla_decoding.png)
        r_i_ins, p_i_ins, preds_i_ins = insample_ols(Zi, yy)
        r_v_ins, p_v_ins, preds_v_ins = insample_ols(Hv, yy)
        r_iv_ins, _ = pearsonr(preds_i_ins, preds_v_ins)
        z_s_ins, p_s_ins = steiger_test(r_i_ins, r_v_ins, r_iv_ins, n)

        # LOO RidgeCV (out-of-sample test for actual signal)
        r_i_loo, p_i_loo, preds_i_loo = loo_ridge(Zi, yy)
        r_v_loo, p_v_loo, preds_v_loo = loo_ridge(Hv, yy)
        r_iv_loo, _ = pearsonr(preds_i_loo, preds_v_loo)
        z_s_loo, p_s_loo = steiger_test(r_i_loo, r_v_loo, r_iv_loo, n)

        # Chance baseline for in-sample r (multivariate-OLS noise floor)
        chance_i = np.sqrt(z_dim_i / (n - 1))
        chance_v = np.sqrt(z_dim_v / (n - 1))

        print(f"  {scale:<10}  IN-SAMPLE  IDRNN r={r_i_ins:+.3f} ({p_i_ins:.3g} {sig_stars(p_i_ins)})  "
              f"Vanilla r={r_v_ins:+.3f} ({p_v_ins:.3g} {sig_stars(p_v_ins)})")
        print(f"  {' '*10}  LOO Ridge  IDRNN r={r_i_loo:+.3f} ({p_i_loo:.3g} {sig_stars(p_i_loo)})  "
              f"Vanilla r={r_v_loo:+.3f} ({p_v_loo:.3g} {sig_stars(p_v_loo)})  "
              f"chance r≈ {chance_i:.3f}/{chance_v:.3f}")

        decode_rows.append({
            "scale": scale, "n": n,
            # In-sample OLS (matches existing)
            "r_idrnn": r_i_ins, "p_idrnn": p_i_ins,
            "r_vanilla": r_v_ins, "p_vanilla": p_v_ins,
            "r_iv": r_iv_ins, "z_steiger": z_s_ins, "p_steiger": p_s_ins,
            # LOO RidgeCV (new)
            "r_idrnn_loo": r_i_loo, "p_idrnn_loo": p_i_loo,
            "r_vanilla_loo": r_v_loo, "p_vanilla_loo": p_v_loo,
            "r_iv_loo": r_iv_loo, "z_steiger_loo": z_s_loo,
            "p_steiger_loo": p_s_loo,
            # Noise floor for in-sample r
            "chance_r_idrnn": chance_i,
            "chance_r_vanilla": chance_v,
        })
    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Two panels: in-sample OLS (left) and LOO RidgeCV (right).
    # In-sample r is biased by sqrt(k/(n-1)) under H0 — drawn as a chance line.
    # LOO r is unbiased; positive r means real out-of-sample predictive signal.
    labels = [SCALES[k][1] for k in decode_df["scale"]]
    x = np.arange(len(labels))
    w = 0.35
    c1, c2 = "#4C72B0", "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    panels = [
        (axes[0], "r_idrnn",     "r_vanilla",     "p_idrnn",     "p_vanilla",
         "p_steiger",     "In-sample OLS (R²-based, k-biased)"),
        (axes[1], "r_idrnn_loo", "r_vanilla_loo", "p_idrnn_loo", "p_vanilla_loo",
         "p_steiger_loo", "LOO RidgeCV (unbiased — true signal test)"),
    ]
    for ax, ci, cv, pi_, pv_, ps_, title in panels:
        ax.bar(x - w/2, decode_df[ci].abs(), w,
               color=c1, alpha=0.85, edgecolor="k", linewidth=0.5,
               label=f"IDRNN step-1 lookup (seed {best_idrnn_seed})")
        ax.bar(x + w/2, decode_df[cv].abs(), w,
               color=c2, alpha=0.85, edgecolor="k", linewidth=0.5,
               label=f"Vanilla h (seed {best_vanilla_seed})")

        # Significance stars and Steiger bracket
        for i, row in decode_df.iterrows():
            ax.text(i - w/2, abs(row[ci]) + 0.01, sig_stars(row[pi_]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=c1)
            ax.text(i + w/2, abs(row[cv]) + 0.01, sig_stars(row[pv_]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=c2)
            y_max = max(abs(row[ci]), abs(row[cv])) + 0.07
            ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                    [y_max - 0.01, y_max, y_max, y_max - 0.01],
                    color="k", linewidth=0.8)
            stars = sig_stars(row[ps_])
            ax.text(i, y_max + 0.005, stars, ha="center", va="bottom",
                    fontsize=8, color=("k" if stars != "n.s." else "grey"))

        # In-sample noise floor (only meaningful for the left panel)
        if ci == "r_idrnn":
            ch_i = decode_df["chance_r_idrnn"].mean()
            ch_v = decode_df["chance_r_vanilla"].mean()
            ax.axhline(ch_i, color=c1, ls="--", lw=1.2, alpha=0.7,
                       label=f"chance r (k={z_dim_i}) ≈ {ch_i:.3f}")
            ax.axhline(ch_v, color=c2, ls="--", lw=1.2, alpha=0.7,
                       label=f"chance r (k={z_dim_v}) ≈ {ch_v:.3f}")
        else:
            ax.axhline(0, color="grey", lw=1, alpha=0.5)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("|r|  (decodability)", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0, min(1.0, axes[0].get_ylim()[1] + 0.1))
    fig.suptitle(
        f"Questionnaire decoding — step-1 lookup z vs vanilla h  "
        f"(all 236 subjects, no outer CV)",
        fontweight="bold",
    )
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "step1_vs_vanilla_decoding.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png}")
    print(f"Saved {os.path.join(OUT_DIR, 'decoding_results.csv')}")
    print(f"Saved {os.path.join(OUT_DIR, 'seed_summary.csv')}")


if __name__ == "__main__":
    main()
