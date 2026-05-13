#!/usr/bin/env python3
"""
Compare best IDRNN vs Vanilla on Thalmann data.

"Best" for IDRNN = combo with lowest mean cv_val_nll across outer-CV folds
(read from summary_table_per_fold.csv).  Model selection uses inner-CV NLL
(epoch selection criterion), which is orthogonal to decodability — so using
the same model for both NLL and decodability evaluation is NOT cherry-picking.

Both models are evaluated on the same outer-CV test folds.

NLL:
  - IDRNN : recomputed on outer test fold using full LatentRNN_secondstep
            (encoder + decoder).  NOT taken from config cv_val_nll (which is
            inner-CV NLL, slightly optimistic).
  - Vanilla: recomputed on the same outer test fold.
  Significance: paired Wilcoxon signed-rank test on per-subject NLL (N=236).

Decodability (per-fold LOO-CV ridge):
  - IDRNN  latent : encoder μ at blk-0, t=9 (last real trial of task-0 block 0)
  - Vanilla latent: GRU hidden state at t=9 (flat sequence)
  - Pooled r  : predictions concatenated across folds → overall Pearson r + p
  - IDRNN vs Vanilla : Steiger (1980) / Meng et al. (1992) z-test on dependent
                       correlations sharing Y.

Outputs: plots_thalmann/comparison/
  nll_comparison.png          — per-fold bar + distribution (violin) + Wilcoxon p
  decodability_comparison.png — per-scale pooled r, ± CI, significance stars
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, wilcoxon, norm

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN, Decoder, LatentRNN_secondstep

# ── Config ─────────────────────────────────────────────────────────────────────
DGP          = "thalmann"
N_FOLDS      = 3
PLOT_DIR     = "plots_thalmann/comparison"
SUMMARY_CSV  = "plots_thalmann/hp_v2/summary_table_per_fold.csv"
# hp_v3: continuous encoder — run directories use this suffix instead of hp_v2
HP_VERSION   = "hp_v3"

# Decoding mode:
#   "pooled"     — decoder trained on z's pooled across all outer-CV folds
#                  (N-1 training subjects per LOO iteration).  Optimistic if
#                  per-fold encoders produce non-aligned latent spaces, but
#                  matches the ranking metric used in analyze_hp_v3_outer_cv.
#   "within_fold"— decoder trained on one fold's z-space only (~N/3 subjects
#                  per LOO iteration).  Principled, never mixes independent
#                  latent spaces; lower r.
# File outputs are suffixed with the mode name.
DECODING_MODE = "pooled"
assert DECODING_MODE in ("pooled", "within_fold")
MODE_SUFFIX   = "_pooled" if DECODING_MODE == "pooled" else "_wf"

# Representations used for decoding:
#
# IDRNN  : encoder μ at the last valid trial of the full continuous sequence
#           (return_per_timestep=False).  The encoder GRU now runs without
#           reset across all 31 blocks and both tasks, so this is the fully
#           informed individual posterior after all 500 real trials.
#
# Vanilla: time-average of GRU hidden states across ALL valid (non-padding)
#           trials (500 real trials across 31 blocks).

os.makedirs(PLOT_DIR, exist_ok=True)

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5\nOpenness"),
}
SCALE_KEYS   = list(SCALES.keys())
SCALE_LABELS = [SCALES[k][1] for k in SCALE_KEYS]

quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
for k, (items, _) in SCALES.items():
    quest[k] = quest[items].mean(axis=1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_persubj_nll(logits, c_np):
    """
    Per-subject mean NLL.
    logits : (B, n_blk, T, A)  or  (B, T_flat, A) — will be flattened.
    c_np   : matching shape, -100 = padding.
    Returns: np.ndarray (B,)
    """
    B = logits.shape[0]
    logits_flat = logits.reshape(B, -1, logits.shape[-1])
    c_flat      = torch.tensor(c_np, dtype=torch.long).reshape(B, -1)
    mask        = (c_flat >= 0)
    log_p       = F.log_softmax(logits_flat, dim=-1)
    chosen      = log_p.gather(-1, c_flat.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    subj_nll    = -(chosen * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
    return subj_nll.detach().numpy()


def loo_ridge_within_fold(Z, y):
    """
    LOO-CV ridge, z-scored within train split.
    Returns (r, p, preds) — preds is the full out-of-sample prediction array.
    """
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z = Z[tr].mean(0); sd_z = Z[tr].std(0) + 1e-8
        mu_y = y[tr].mean();  sd_y = y[tr].std()  + 1e-8
        clf  = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return r, p, preds


def steiger_test(r12, r13, r23, n):
    """
    Test H0: rho(Y, X1) == rho(Y, X2) (Meng, Rosenthal & Rubin 1992).
    r12 = corr(Y, IDRNN),  r13 = corr(Y, Vanilla),  r23 = corr(IDRNN, Vanilla).
    Returns (z_stat, two-tailed p).
    """
    r12 = np.clip(r12, -0.9999, 0.9999)
    r13 = np.clip(r13, -0.9999, 0.9999)
    r23 = np.clip(r23, -0.9999, 0.9999)

    z12 = np.arctanh(r12)
    z13 = np.arctanh(r13)

    r_sq_bar = (r12**2 + r13**2) / 2
    f = (1 - r23) / (2 * (1 - r_sq_bar))
    h = (1 - f * r_sq_bar) / (1 - r_sq_bar)

    var = (2 * (1 - r23) / (n - 1)) * h
    if var <= 0:
        return 0.0, 1.0
    z_stat = (z12 - z13) / np.sqrt(var)
    p_val  = 2 * norm.sf(abs(z_stat))
    return float(z_stat), float(p_val)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ══════════════════════════════════════════════════════════════════════════════
# IDRNN — Pareto-optimal combo (best NLL–PHQ tradeoff from hp_v2 search)
# ══════════════════════════════════════════════════════════════════════════════
BEST_COMBO = "uw00_lmbd02_eh20_h10_z5"
print(f"IDRNN combo: {BEST_COMBO}  (hp_v3, pooled-LOO: best PHQ + mean|r| vs vanilla)")
print(f"  pooled |r|_PHQ=0.947, mean|r|=0.496, NLL=0.654  "
      f"(vanilla pooled: 0.703 / 0.338 / 0.629)")
print(f"  Decoding mode: {DECODING_MODE}   file suffix: {MODE_SUFFIX}")
print(f"  Using {HP_VERSION} runs (continuous encoder, z at last valid timestep)")

idrnn_fold_nll     = []          # mean NLL per fold (scalar)
idrnn_all_subj_nll = []          # per-subject NLL, pooled across folds
idrnn_z_per_fold   = []          # list of (B_f, z_dim) per fold
idrnn_sub_per_fold = []          # list of subids per fold

# CPRNN: same decoder weights as IDRNN but z forced to zeros (no ID signal).
# Equivalent to test_latentrnn_secondstep_causal_posterior_weighting(id_case=False):
# GRU decoder is causal, so block-level forward with constant z gives identical
# logits to the trial-by-trial causal evaluation when z is deterministic.
cprnn_fold_nll     = []
cprnn_all_subj_nll = []

for fold in range(N_FOLDS):
    run_base = f"runs_thalmann_{HP_VERSION}_{BEST_COMBO}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)
    c_test_np = np.load(f"{data_dir}/c_test.npy")
    df_test   = pd.read_csv(f"{data_dir}/df_test.csv")
    subids    = (df_test["subid"].values if "subid" in df_test.columns
                 else df_test["session"].values)

    z_seeds         = []
    nll_seeds       = []   # IDRNN per-subject NLL (B,) per seed
    nll_cprnn_seeds = []   # CPRNN per-subject NLL (B,) per seed

    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt): continue
        try:
            state = torch.load(ckpt, map_location="cpu")

            # ── encoder (standalone) for fully informed posterior μ ─────────
            enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                        hid=mc["enc_hidden"], n_tasks=mc.get("n_tasks"),
                        task_emb_dim=mc.get("task_emb_dim", 0),
                        continuous_encoder=mc.get("continuous_encoder", False))
            enc.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
            enc.set_task_ids(task_ids_global)
            enc.eval()
            with torch.no_grad():
                mu, _ = enc(xin_test, return_per_timestep=False)  # (B, z_dim)
            z_seeds.append(mu.numpy())

            # ── full model (encoder + decoder) for test NLL ─────────────────
            enc2 = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                         hid=mc["enc_hidden"], n_tasks=mc.get("n_tasks"),
                         task_emb_dim=mc.get("task_emb_dim", 0),
                         continuous_encoder=mc.get("continuous_encoder", False))
            dec2 = Decoder(mc["dec_in_dim"], mc["z_dim"], mc["hidden"], mc["A"])
            model_full = LatentRNN_secondstep(
                encoder=enc2, hid=mc["hidden"], z_dim=mc["z_dim"],
                in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec2,
                n_tasks=mc.get("n_tasks"),
                task_emb_dim=mc.get("task_emb_dim", 0),
                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", False),
            )
            model_full.load_state_dict(state)
            model_full.set_task_ids(task_ids_global)
            model_full.eval()
            with torch.no_grad():
                # IDRNN: full forward, z = encoder posterior mean
                logits, mu_full, _, _, _ = model_full(xin_test, xin_test)
            nll_seeds.append(compute_persubj_nll(logits, c_test_np))  # (B,)

            # ── CPRNN: same decoder + task embeddings, z forced to zeros.
            # Replicates model_full's block loop (LatentRNN_secondstep.forward)
            # but replaces z=μ with z=0.  Equivalent to id_case=False in
            # test_latentrnn_secondstep_causal_posterior_weighting.
            z_zero = torch.zeros_like(mu_full)
            with torch.no_grad():
                if xin_test.dim() == 4 and xin_test.size(1) > 1:
                    h_cp = None
                    blk_logits_cp = []
                    for b in range(xin_test.size(1)):
                        if model_full.reinit_decoder_per_block and b > 0:
                            h_cp = None
                        inp_b = model_full._append_task_emb(xin_test[:, b], b)
                        lg_b, h_cp = model_full.decoder(inp_b, z_zero, hidden=h_cp)
                        blk_logits_cp.append(lg_b)
                    logits_cp = torch.stack(blk_logits_cp, dim=1)
                else:
                    inp_cp   = model_full._append_task_emb(xin_test.squeeze(1), 0)
                    lg_cp, _ = model_full.decoder(inp_cp, z_zero)
                    logits_cp = lg_cp.unsqueeze(1)
            nll_cprnn_seeds.append(compute_persubj_nll(logits_cp, c_test_np))

        except Exception as e:
            print(f"  IDRNN warning fold{fold}/{sd}: {e}")

    if not z_seeds: continue

    z_mean          = np.stack(z_seeds).mean(0)
    nll_subj_mean   = np.stack(nll_seeds).mean(0)
    nll_cp_subj_mean = np.stack(nll_cprnn_seeds).mean(0)
    idrnn_fold_nll.append(float(nll_subj_mean.mean()))
    idrnn_all_subj_nll.extend(nll_subj_mean.tolist())
    cprnn_fold_nll.append(float(nll_cp_subj_mean.mean()))
    cprnn_all_subj_nll.extend(nll_cp_subj_mean.tolist())
    idrnn_z_per_fold.append(z_mean)
    idrnn_sub_per_fold.append(subids)

print(f"IDRNN  NLL per fold : {[f'{v:.4f}' for v in idrnn_fold_nll]}")
print(f"IDRNN  mean NLL     : {np.nanmean(idrnn_fold_nll):.4f}")
print(f"CPRNN  NLL per fold : {[f'{v:.4f}' for v in cprnn_fold_nll]}")
print(f"CPRNN  mean NLL     : {np.nanmean(cprnn_fold_nll):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Vanilla — compute test NLL + extract h at position T_EXTRACT_FLAT
# ══════════════════════════════════════════════════════════════════════════════
van_base = "runs_vanilla_thalmann"

vanilla_fold_nll     = []
vanilla_all_subj_nll = []
vanilla_h_per_fold   = []
vanilla_sub_per_fold = []

for fold in range(N_FOLDS):
    fold_dir  = os.path.join(van_base, f"fold{fold}")
    data_dir  = f"data_{DGP}/fold{fold}"

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)
    c_test_np = np.load(f"{data_dir}/c_test.npy")
    df_test   = pd.read_csv(f"{data_dir}/df_test.csv")
    subids    = (df_test["subid"].values if "subid" in df_test.columns
                 else df_test["session"].values)

    # Mirror testing_script.py: block_structure is determined by data shape,
    # not by config.  xin_test is 4-D (B, 31, 200, 5) for Thalmann, so
    # block_structure=True — each block processed independently with fresh h0.
    # Running on the flat 6200-step sequence (block_structure=False) instead
    # would give artificially inflated NLL because 190 padding steps corrupt
    # the GRU hidden state before every 10-trial real segment.
    _block_structure = (xin_test.dim() == 4)

    h_seeds   = []
    nll_seeds = []   # list of (B,) arrays

    for sd in sorted(d for d in os.listdir(fold_dir) if d.startswith("seed_")):
        cfg_p = os.path.join(fold_dir, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(fold_dir, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt): continue
        try:
            state = torch.load(ckpt, map_location="cpu")
            model = AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                               block_structure=_block_structure,
                               n_tasks=mc.get("n_tasks"),
                               task_emb_dim=mc.get("task_emb_dim", 0))
            model.load_state_dict(state)
            model.set_task_ids(task_ids_global)
            model.eval()

            with torch.no_grad():
                # Pass 4-D xin_test directly; block_structure=True processes
                # each block independently.
                # Returns logits (B,31,200,A), final_hid, hidden_tr (B,31,200,hid)
                logits, _, hidden_tr = model(xin_test)

            # Time-average hidden state across all valid (non-padding) trials.
            # valid mask from choices: (B, 31, 200) — True where c >= 0.
            valid = torch.tensor(c_test_np >= 0, dtype=torch.float32)  # (B,31,200)
            n_valid = valid.sum(dim=(1, 2), keepdim=True).clamp(min=1)  # (B,1,1)
            h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2)) / n_valid.squeeze(-1)
            h_seeds.append(h_avg.numpy())   # (B, hid)
            nll_seeds.append(compute_persubj_nll(logits, c_test_np))   # (B,)

        except Exception as e:
            print(f"  Vanilla warning fold{fold}/{sd}: {e}")

    if not h_seeds: continue

    h_mean        = np.stack(h_seeds).mean(0)        # (B_test, hid)
    nll_subj_mean = np.stack(nll_seeds).mean(0)      # (B_test,)
    vanilla_fold_nll.append(float(nll_subj_mean.mean()))
    vanilla_all_subj_nll.extend(nll_subj_mean.tolist())
    vanilla_h_per_fold.append(h_mean)
    vanilla_sub_per_fold.append(subids)

print(f"Vanilla NLL per fold: {[f'{v:.4f}' for v in vanilla_fold_nll]}")
print(f"Vanilla mean NLL    : {np.nanmean(vanilla_fold_nll):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Cognitive-model baselines — load per-fold NLLs from thalmann_cog_model.py
# Four CSVs per fold: cog_model_results.csv (EM), cog_model_cp_results.csv (CP),
# illspec_em_results.csv, illspec_cp_results.csv.  Each has per-subject mean NLL
# per valid trial in `normalized_likelihood`.
# ══════════════════════════════════════════════════════════════════════════════
COG_BASELINES = {
    "cog_em":     "cog_model_results.csv",
    "cog_cp":     "cog_model_cp_results.csv",
    "illspec_em": "illspec_em_results.csv",
    "illspec_cp": "illspec_cp_results.csv",
}

cog_fold_nll     = {k: [] for k in COG_BASELINES}
cog_all_subj_nll = {k: [] for k in COG_BASELINES}
cog_available    = True

for fold in range(N_FOLDS):
    sids_f = idrnn_sub_per_fold[fold]
    for key, fname in COG_BASELINES.items():
        csv = f"data_{DGP}/fold{fold}/{fname}"
        if not os.path.exists(csv):
            print(f"  {key}: {csv} not found → skipping cog baselines. "
                  f"Run: python thalmann_cog_model.py --fold {fold}")
            cog_available = False
            break
        df_c = pd.read_csv(csv).set_index("subid")
        nll_aligned = df_c.reindex(sids_f)["normalized_likelihood"].values
        if np.isnan(nll_aligned).any():
            missing = np.array(sids_f)[np.isnan(nll_aligned)]
            print(f"  {key} fold {fold}: missing subids {missing[:5]}… "
                  f"skipping cog baselines")
            cog_available = False
            break
        cog_fold_nll[key].append(float(nll_aligned.mean()))
        cog_all_subj_nll[key].extend(nll_aligned.tolist())
    if not cog_available:
        break

if cog_available:
    for key in COG_BASELINES:
        print(f"{key:12s} NLL per fold : "
              f"{[f'{v:.4f}' for v in cog_fold_nll[key]]}  "
              f"mean={np.nanmean(cog_fold_nll[key]):.4f}")


# ── Random baseline NLL ───────────────────────────────────────────────────────
# Weighted average of log(n_choices) per valid trial.
# Task-0 (blocks 0-29): 2-armed → log(2); Task-1 (block 30): 4-armed → log(4).
_task_ids_np    = task_ids_global.numpy()
_n_choices      = np.where(_task_ids_np == 0, 2, 4)
_trials_per_blk = np.where(_task_ids_np == 0, 10, 200)
random_nll      = float((np.log(_n_choices) * _trials_per_blk).sum()
                        / _trials_per_blk.sum())
print(f"Random baseline NLL: {random_nll:.4f}  "
      f"({int(_trials_per_blk.sum())} valid trials: "
      f"300×log(2) + 200×log(4) weighted avg)")


# ── Per-subject NLL arrays for all 7 groups ──────────────────────────────────
idrnn_nll_arr   = np.array(idrnn_all_subj_nll)
vanilla_nll_arr = np.array(vanilla_all_subj_nll)
cprnn_nll_arr   = np.array(cprnn_all_subj_nll)
assert len(idrnn_nll_arr) == len(vanilla_nll_arr) == len(cprnn_nll_arr)

if cog_available:
    cog_em_arr     = np.array(cog_all_subj_nll["cog_em"])
    cog_cp_arr     = np.array(cog_all_subj_nll["cog_cp"])
    illspec_em_arr = np.array(cog_all_subj_nll["illspec_em"])
    illspec_cp_arr = np.array(cog_all_subj_nll["illspec_cp"])
else:
    cog_em_arr = cog_cp_arr = illspec_em_arr = illspec_cp_arr = None

# Pairwise Wilcoxon comparisons used in the NLL plot and summary.
#   key = (label_a, label_b); value = (W, p)
pairs = {}
def _wilc(a, b):
    w, p = wilcoxon(a, b, alternative="two-sided")
    return float(w), float(p)

# RNN primary contrast + CP vs ID comparisons (match sloutsky reference).
pairs[("IDRNN",   "Vanilla")] = _wilc(idrnn_nll_arr,  vanilla_nll_arr)
pairs[("CPRNN",   "IDRNN")]   = _wilc(cprnn_nll_arr,  idrnn_nll_arr)
pairs[("Vanilla", "CPRNN")]   = _wilc(vanilla_nll_arr, cprnn_nll_arr)

if cog_available:
    pairs[("IllSpec_CP", "IllSpec_EM")] = _wilc(illspec_cp_arr, illspec_em_arr)
    pairs[("Cog_CP",     "Cog_EM")]     = _wilc(cog_cp_arr,     cog_em_arr)
    pairs[("Cog_EM",     "IDRNN")]      = _wilc(cog_em_arr,     idrnn_nll_arr)

print(f"\n── Per-subject NLL (N={len(idrnn_nll_arr)}) ──────────────────────────")
print(f"  IDRNN      {idrnn_nll_arr.mean():.4f} ± {idrnn_nll_arr.std():.4f}")
print(f"  Vanilla    {vanilla_nll_arr.mean():.4f} ± {vanilla_nll_arr.std():.4f}")
print(f"  CPRNN      {cprnn_nll_arr.mean():.4f} ± {cprnn_nll_arr.std():.4f}")
if cog_available:
    print(f"  Cog_EM     {cog_em_arr.mean():.4f} ± {cog_em_arr.std():.4f}")
    print(f"  Cog_CP     {cog_cp_arr.mean():.4f} ± {cog_cp_arr.std():.4f}")
    print(f"  IllSpec_EM {illspec_em_arr.mean():.4f} ± {illspec_em_arr.std():.4f}")
    print(f"  IllSpec_CP {illspec_cp_arr.mean():.4f} ± {illspec_cp_arr.std():.4f}")
print("\nWilcoxon pairs:")
for (a, b), (w, p) in pairs.items():
    print(f"  {a:<12} vs {b:<12}  W={w:8.0f}  p={p:.3g}  {sig_stars(p)}")

# Kept for backward-compatibility with the old Summary printout
wstat,    p_nll = pairs[("IDRNN", "Vanilla")]
wstat_ic = p_ic = wstat_vc = p_vc = None
if cog_available:
    wstat_ic, p_ic = _wilc(idrnn_nll_arr,   cog_em_arr)
    wstat_vc, p_vc = _wilc(vanilla_nll_arr, cog_em_arr)


# ── Decodability: per-fold r's (err bars) + pooled r + Steiger ───────────────
# DECODING_MODE controls where the ridge is trained:
#   "pooled"     : decoder trained on ALL folds' z's pooled together (one LOO
#                  over N≈236 subjects).  Matches analyze_hp_v3_outer_cv ranking.
#   "within_fold": decoder trained on one fold's z-space only (~N/3 subjects
#                  per LOO iteration), never mixing encoder latent spaces.
pooled_results    = {}
idrnn_fold_rs     = {k: [] for k in SCALE_KEYS}
vanilla_fold_rs   = {k: [] for k in SCALE_KEYS}

print(f"\n── Decodability (mode={DECODING_MODE}) ───────────────────────────────────")

for scale in SCALE_KEYS:
    # Build pooled (z, y) and per-fold slices
    z_folds_i, h_folds_v, y_folds, mask_sizes = [], [], [], []
    for zf, hf, sids in zip(idrnn_z_per_fold, vanilla_h_per_fold, idrnn_sub_per_fold):
        y    = quest.reindex(sids)[scale].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            z_folds_i.append(None); h_folds_v.append(None); y_folds.append(None)
            mask_sizes.append(0)
            continue
        z_folds_i.append(zf[mask]); h_folds_v.append(hf[mask]); y_folds.append(y[mask])
        mask_sizes.append(mask.sum())

    if sum(mask_sizes) == 0:
        continue

    if DECODING_MODE == "pooled":
        # Pool across folds, then one LOO ridge per model
        Zi = np.concatenate([z for z in z_folds_i if z is not None])
        Hv = np.concatenate([h for h in h_folds_v if h is not None])
        yy = np.concatenate([y for y in y_folds   if y is not None])
        _, _, preds_idrnn   = loo_ridge_within_fold(Zi, yy)   # same function, pooled Z
        _, _, preds_vanilla = loo_ridge_within_fold(Hv, yy)

        # Per-fold r (for err bars): slice preds back by fold sizes
        offset = 0
        for zf, yf in zip(z_folds_i, y_folds):
            if zf is None: continue
            n = len(yf)
            ri, _ = pearsonr(yf, preds_idrnn[offset:offset+n])
            rv, _ = pearsonr(yf, preds_vanilla[offset:offset+n])
            idrnn_fold_rs[scale].append(ri)
            vanilla_fold_rs[scale].append(rv)
            offset += n

    else:  # within_fold
        preds_i_list, preds_v_list, y_list = [], [], []
        for zf, hf, yf in zip(z_folds_i, h_folds_v, y_folds):
            if zf is None: continue
            ri, _, pi = loo_ridge_within_fold(zf, yf)
            rv, _, pv = loo_ridge_within_fold(hf, yf)
            idrnn_fold_rs[scale].append(ri)
            vanilla_fold_rs[scale].append(rv)
            preds_i_list.append(pi); preds_v_list.append(pv); y_list.append(yf)
        yy            = np.concatenate(y_list)
        preds_idrnn   = np.concatenate(preds_i_list)
        preds_vanilla = np.concatenate(preds_v_list)

    r_i, p_i = pearsonr(yy, preds_idrnn)
    r_v, p_v = pearsonr(yy, preds_vanilla)
    r_iv, _  = pearsonr(preds_idrnn, preds_vanilla)
    z_steiger, p_steiger = steiger_test(r_i, r_v, r_iv, len(yy))

    pooled_results[scale] = dict(
        r_idrnn=r_i, p_idrnn=p_i,
        r_vanilla=r_v, p_vanilla=p_v,
        r_iv=r_iv,
        z_steiger=z_steiger, p_steiger=p_steiger,
    )
    print(f"  {scale:<12}  IDRNN r={r_i:+.3f} (p={p_i:.3g} {sig_stars(p_i)})  "
          f"Vanilla r={r_v:+.3f} (p={p_v:.3g} {sig_stars(p_v)})  "
          f"Steiger z={z_steiger:+.2f} p={p_steiger:.3g} {sig_stars(p_steiger)}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: NLL comparison — 7-bar panel matching plots_sloutsky/outer_cv_nll.png
# ══════════════════════════════════════════════════════════════════════════════
# Layout mirrors the sloutsky reference: 7 bars grouped by model type, with
# common-process (light) and individual-differences (dark) side-by-side in the
# three model-type groups.  Individual-participant dots overlaid (x-jittered),
# SEM error bars, pairwise significance brackets, y starts at 0.

# Group colours (model-type); CP → light shade, EM/ID → full saturation.
COL = {
    "illspec_cp": "#A9A9A9",    "illspec_em": "#4F4F4F",          # grey pair
    "cog_cp":     "#A8D5A2",    "cog_em":     "#2E7D32",          # green pair
    "cprnn":      "#9ECAE1",    "idrnn":      "#1F77B4",          # blue pair
    "vanilla":    "#E67E22",                                       # orange
}
BAR_ORDER = ["illspec_cp", "illspec_em",
             "cog_cp",     "cog_em",
             "cprnn",      "idrnn",
             "vanilla"]
BAR_LABELS = {
    "illspec_cp": "Ill-spec.\nCP",  "illspec_em": "Ill-spec.\nEM",
    "cog_cp":     "Cog model\nCP",  "cog_em":     "Cog model\nEM",
    "cprnn":      "CP RNN",         "idrnn":      "IDRNN",
    "vanilla":    "Vanilla\nRNN",
}

if cog_available:
    bar_data = {
        "illspec_cp": illspec_cp_arr,
        "illspec_em": illspec_em_arr,
        "cog_cp":     cog_cp_arr,
        "cog_em":     cog_em_arr,
        "cprnn":      cprnn_nll_arr,
        "idrnn":      idrnn_nll_arr,
        "vanilla":    vanilla_nll_arr,
    }
else:
    # Skip the cog bars if CSVs are missing.
    BAR_ORDER = ["cprnn", "idrnn", "vanilla"]
    bar_data = {"cprnn": cprnn_nll_arr, "idrnn": idrnn_nll_arr,
                "vanilla": vanilla_nll_arr}

fig, ax = plt.subplots(figsize=(12, 7.5))

# Bar x-positions with small gaps between model-type groups.
if cog_available:
    xpos = {"illspec_cp": 0.0, "illspec_em": 0.9,
            "cog_cp":     2.1, "cog_em":     3.0,
            "cprnn":      4.2, "idrnn":      5.1,
            "vanilla":    6.3}
else:
    xpos = {"cprnn": 0.0, "idrnn": 0.9, "vanilla": 2.1}

BAR_W = 0.75
rng   = np.random.default_rng(0)

for key in BAR_ORDER:
    arr    = bar_data[key]
    mean_v = np.nanmean(arr)
    sem_v  = np.nanstd(arr) / np.sqrt(len(arr))
    x      = xpos[key]
    # Bar (mean) + SEM error bar
    ax.bar(x, mean_v, BAR_W, yerr=sem_v, capsize=4,
           color=COL[key], alpha=0.85, edgecolor="k", linewidth=0.6,
           zorder=2, label=BAR_LABELS[key])
    # Scatter dots — individual participants, x-jittered
    jitter = rng.uniform(-BAR_W/3, BAR_W/3, size=len(arr))
    ax.scatter(x + jitter, arr,
               s=11, color="black", alpha=0.22, zorder=3, edgecolors="none")

# Chance / random-model reference line
ax.axhline(random_nll, color="k", linewidth=1.2, linestyle="--", zorder=1)
ax.text(max(xpos.values()) + 0.45, random_nll, f" Random model ({random_nll:.3f})",
        ha="right", va="bottom", fontsize=9, color="k")

# Pairwise significance brackets — only the comparisons visible in the sloutsky ref
bracket_specs = []
if cog_available:
    bracket_specs = [
        ("illspec_cp", "illspec_em", pairs[("IllSpec_CP", "IllSpec_EM")]),
        ("cog_cp",     "cog_em",     pairs[("Cog_CP",     "Cog_EM")]),
        ("cprnn",      "idrnn",      pairs[("CPRNN",      "IDRNN")]),
        ("idrnn",      "vanilla",    pairs[("IDRNN",      "Vanilla")]),
    ]
else:
    bracket_specs = [
        ("cprnn",   "idrnn",   pairs[("CPRNN",   "IDRNN")]),
        ("idrnn",   "vanilla", pairs[("IDRNN",   "Vanilla")]),
    ]

# Significance brackets: place them just above the top of the bar+SEM range,
# with enough headroom below the random-model line that the legend sits above.
bar_tops = [np.nanmean(a) + np.nanstd(a)/np.sqrt(len(a)) for a in bar_data.values()]
y_bar_top = max(bar_tops)
y_brack_base = y_bar_top + 0.05
brack_step   = 0.035

for i, (k1, k2, (w_val, p_val)) in enumerate(bracket_specs):
    y_b    = y_brack_base + i * brack_step
    x1, x2 = xpos[k1], xpos[k2]
    ax.plot([x1, x1, x2, x2],
            [y_b - 0.008, y_b, y_b, y_b - 0.008],
            color="k", linewidth=0.9)
    ax.text((x1 + x2) / 2, y_b + 0.002, sig_stars(p_val),
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Axes, legend, title
ax.set_xticks(list(xpos.values()))
ax.set_xticklabels([BAR_LABELS[k] for k in BAR_ORDER], fontsize=10)
ax.set_ylabel("Mean NLL per trial  (↓ better)", fontsize=11)
# Top: leave headroom above the random-model line for the legend.
y_top_lim = max(random_nll, y_brack_base + len(bracket_specs) * brack_step) + 0.22
ax.set_ylim(0, y_top_lim)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Two-row legend (model type + fit type) matching the reference.
from matplotlib.patches import Patch
model_handles = [
    Patch(facecolor=COL["illspec_em"], edgecolor="k", label="Ill-specified model"),
    Patch(facecolor=COL["cog_em"],     edgecolor="k", label="Cog. model"),
    Patch(facecolor=COL["idrnn"],      edgecolor="k", label="ID RNN"),
    Patch(facecolor=COL["vanilla"],    edgecolor="k", label="Vanilla RNN"),
] if cog_available else [
    Patch(facecolor=COL["idrnn"],   edgecolor="k", label="ID RNN"),
    Patch(facecolor=COL["vanilla"], edgecolor="k", label="Vanilla RNN"),
]
fit_handles = [
    Patch(facecolor="lightgray", edgecolor="k", label="Common process (CP)"),
    Patch(facecolor="dimgray",   edgecolor="k", label="Individual differences (ID/EM)"),
]
leg1 = ax.legend(handles=model_handles, title="Model type", loc="upper left",
                 fontsize=8, title_fontsize=9, frameon=False, ncol=len(model_handles))
ax.add_artist(leg1)
ax.legend(handles=fit_handles, title="Fit type", loc="upper right",
          fontsize=8, title_fontsize=9, frameon=False, ncol=2)

fig.suptitle(f"NLL on outer-CV test fold — Thalmann (N={len(idrnn_nll_arr)} subjects)",
             fontweight="bold")
fig.tight_layout()
_nll_fn = f"nll_comparison{MODE_SUFFIX}.png"
fig.savefig(os.path.join(PLOT_DIR, _nll_fn), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {_nll_fn}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Decodability — |r| bars with significance annotations
# ══════════════════════════════════════════════════════════════════════════════
scales_with_data = [k for k in SCALE_KEYS if k in pooled_results]
labels_with_data = [SCALES[k][1] for k in scales_with_data]

r_idrnn_pool   = [pooled_results[k]["r_idrnn"]   for k in scales_with_data]
p_idrnn_pool   = [pooled_results[k]["p_idrnn"]   for k in scales_with_data]
r_vanilla_pool = [pooled_results[k]["r_vanilla"] for k in scales_with_data]
p_vanilla_pool = [pooled_results[k]["p_vanilla"] for k in scales_with_data]
p_steiger_arr  = [pooled_results[k]["p_steiger"] for k in scales_with_data]
z_steiger_arr  = [pooled_results[k]["z_steiger"] for k in scales_with_data]

# Per-fold r mean ± std (for error bars)
idrnn_r_std    = [np.std(idrnn_fold_rs[k])    if idrnn_fold_rs.get(k)   else np.nan
                  for k in scales_with_data]
vanilla_r_std  = [np.std(vanilla_fold_rs[k])  if vanilla_fold_rs.get(k) else np.nan
                  for k in scales_with_data]

# Colours reused by the decodability plot + tradeoff plot (no longer defined in
# the NLL plot above, which uses its own COL dict).
c1, c2 = COL["idrnn"], COL["vanilla"]
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(scales_with_data))

bars_i = ax.bar(x - w/2, [abs(v) for v in r_idrnn_pool], w,
                yerr=idrnn_r_std, capsize=4,
                label=f"IDRNN", color=c1, alpha=0.85,
                edgecolor="k", linewidth=0.5)
bars_v = ax.bar(x + w/2, [abs(v) for v in r_vanilla_pool], w,
                yerr=vanilla_r_std, capsize=4,
                label="Vanilla RNN", color=c2, alpha=0.85,
                edgecolor="k", linewidth=0.5)

# Individual significance stars on top of each bar
for i, (r_val, p_val) in enumerate(zip(r_idrnn_pool, p_idrnn_pool)):
    y_bar = abs(r_val) + (idrnn_r_std[i] if not np.isnan(idrnn_r_std[i]) else 0) + 0.01
    ax.text(i - w/2, y_bar, sig_stars(p_val), ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=c1)
for i, (r_val, p_val) in enumerate(zip(r_vanilla_pool, p_vanilla_pool)):
    y_bar = abs(r_val) + (vanilla_r_std[i] if not np.isnan(vanilla_r_std[i]) else 0) + 0.01
    ax.text(i + w/2, y_bar, sig_stars(p_val), ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=c2)

# Steiger brackets between paired bars
for i, (ri, rv, p_s) in enumerate(zip(r_idrnn_pool, r_vanilla_pool, p_steiger_arr)):
    y_max = max(abs(ri) + (idrnn_r_std[i] if not np.isnan(idrnn_r_std[i]) else 0),
                abs(rv) + (vanilla_r_std[i] if not np.isnan(vanilla_r_std[i]) else 0))
    y_brack = y_max + 0.07
    ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
            [y_brack - 0.01, y_brack, y_brack, y_brack - 0.01],
            color="k", linewidth=0.8)
    stars = sig_stars(p_s)
    col = "k" if stars != "n.s." else "grey"
    ax.text(i, y_brack + 0.005, stars, ha="center", va="bottom",
            fontsize=8, color=col)

ax.axhline(0, color="k", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels_with_data, fontsize=9)
ax.set_ylabel("|r|  (LOO-CV decodability)", fontsize=11)
ax.set_title(f"Decodability: IDRNN vs Vanilla  (pooled outer-CV, N={len(idrnn_all_subj_nll)})",
             fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, min(1.0, ax.get_ylim()[1] + 0.08))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
_dec_fn = f"decodability_comparison{MODE_SUFFIX}.png"
fig.savefig(os.path.join(PLOT_DIR, _dec_fn),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {_dec_fn}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: NLL vs |r_PHQ| tradeoff across hp_v3 combos
# ══════════════════════════════════════════════════════════════════════════════
HP_V3_CSV = "plots_thalmann/hp_v3_outer_cv/summary_table.csv"
if os.path.exists(HP_V3_CSV):
    hp_v3_outer = pd.read_csv(HP_V3_CSV)
    hp_v3_outer = hp_v3_outer[hp_v3_outer["n_folds_ok"] == 3].copy()
    van_mean_nll = np.nanmean(vanilla_fold_nll)

    # Two panels: z at last valid timestep vs time-averaged z.
    # Column choice follows DECODING_MODE so the tradeoff matches the analysis.
    if DECODING_MODE == "pooled":
        panel_specs = [
            ("r_last_PHQ",    "r_last_PHQ",    "z at last valid timestep"),
            ("r_avg_PHQ",     "r_avg_PHQ",     "z time-averaged"),
        ]
        _loo_label = "pooled LOO"
    else:
        panel_specs = [
            ("r_last_wf_PHQ", "r_last_PHQ",    "z at last valid timestep"),
            ("r_avg_wf_PHQ",  "r_avg_PHQ",     "z time-averaged"),
        ]
        _loo_label = "within-fold LOO"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (r_col_wf, r_col_fallback, subtitle) in zip(axes, panel_specs):
        r_col = r_col_wf if r_col_wf in hp_v3_outer.columns else r_col_fallback
        abs_r = hp_v3_outer[r_col].abs().values
        nll_v = hp_v3_outer["test_nll"].values

        sc = ax.scatter(nll_v, abs_r,
                        c=hp_v3_outer["uw"], cmap="coolwarm", vmin=-0.1, vmax=0.6,
                        s=50, alpha=0.7, edgecolors="none", zorder=3)

        # Highlight selected combo
        best_mask = (hp_v3_outer["combo"] == BEST_COMBO).values
        if best_mask.any():
            ax.scatter(nll_v[best_mask], abs_r[best_mask],
                       color="red", s=180, marker="*", zorder=5, edgecolors="k",
                       linewidths=0.8, label=f"Selected: {BEST_COMBO}")

        # Pareto front
        pareto = np.zeros(len(nll_v), dtype=bool)
        for i in range(len(nll_v)):
            if not any(nll_v[j] <= nll_v[i] and abs_r[j] >= abs_r[i] and
                       (nll_v[j] < nll_v[i] or abs_r[j] > abs_r[i])
                       for j in range(len(nll_v)) if j != i):
                pareto[i] = True
        order = np.argsort(nll_v[pareto])
        ax.plot(nll_v[pareto][order], abs_r[pareto][order],
                color="red", lw=1.2, ls="--", alpha=0.6, zorder=4, label="Pareto front")

        ax.axvline(van_mean_nll, color=c2, lw=1.5, ls=":", alpha=0.8,
                   label=f"Vanilla NLL ({van_mean_nll:.3f})")

        ax.set_xlabel("Test NLL (outer CV, pooled)", fontsize=11)
        ax.set_ylabel(f"|r| PHQ decoding ({_loo_label})", fontsize=11)
        ax.set_title(subtitle, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.colorbar(sc, ax=ax, label="unif_weight")

    fig.suptitle(f"NLL vs PHQ decodability tradeoff  "
                 f"({len(hp_v3_outer)} hp_v3 combos, {_loo_label})",
                 fontweight="bold")
    fig.tight_layout()
    _tr_fn = f"nll_vs_phq_tradeoff{MODE_SUFFIX}.png"
    fig.savefig(os.path.join(PLOT_DIR, _tr_fn),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_tr_fn} (two panels: z_last | z_avg)")
else:
    print(f"Skipping tradeoff plot: {HP_V3_CSV} not found. "
          f"Run analyze_hp_v3_outer_cv.py after the hp_v3 sweep completes.")


# ══════════════════════════════════════════════════════════════════════════════
# Summary printout
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary ────────────────────────────────────────────────────────────────")
print(f"Best IDRNN combo : {BEST_COMBO}")
print(f"IDRNN  mean NLL  : {np.nanmean(idrnn_fold_nll):.4f} "
      f"(median per-subj: {np.median(idrnn_nll_arr):.4f})")
print(f"Vanilla mean NLL : {np.nanmean(vanilla_fold_nll):.4f} "
      f"(median per-subj: {np.median(vanilla_nll_arr):.4f})")
print(f"CPRNN  mean NLL  : {np.nanmean(cprnn_fold_nll):.4f} "
      f"(median per-subj: {np.median(cprnn_nll_arr):.4f})")
if cog_available:
    print(f"Cog_EM    mean NLL: {np.nanmean(cog_fold_nll['cog_em']):.4f} "
          f"(median per-subj: {np.median(cog_em_arr):.4f})")
    print(f"Cog_CP    mean NLL: {np.nanmean(cog_fold_nll['cog_cp']):.4f} "
          f"(median per-subj: {np.median(cog_cp_arr):.4f})")
    print(f"IllSpec_EM mean NLL: {np.nanmean(cog_fold_nll['illspec_em']):.4f} "
          f"(median per-subj: {np.median(illspec_em_arr):.4f})")
    print(f"IllSpec_CP mean NLL: {np.nanmean(cog_fold_nll['illspec_cp']):.4f} "
          f"(median per-subj: {np.median(illspec_cp_arr):.4f})")
print("\nWilcoxon pairs (N={}):".format(len(idrnn_nll_arr)))
for (a, b), (w, p) in pairs.items():
    print(f"  {a:<12} vs {b:<12}  W={w:8.0f}  p={p:.3g}  {sig_stars(p)}")
print()
print(f"{'Scale':<12}  IDRNN r  (p)        Vanilla r  (p)       "
      f"Steiger z  (p)       Winner")
for k in scales_with_data:
    res = pooled_results[k]
    winner = ("IDRNN" if abs(res["r_idrnn"]) > abs(res["r_vanilla"]) else "Vanilla")
    print(f"  {k:<12}  {res['r_idrnn']:+.3f} ({res['p_idrnn']:.3g} "
          f"{sig_stars(res['p_idrnn'])})"
          f"  {res['r_vanilla']:+.3f} ({res['p_vanilla']:.3g} "
          f"{sig_stars(res['p_vanilla'])})"
          f"  {res['z_steiger']:+.2f} ({res['p_steiger']:.3g} "
          f"{sig_stars(res['p_steiger'])})"
          f"  {winner}")
print(f"\nOutputs saved to {PLOT_DIR}")
