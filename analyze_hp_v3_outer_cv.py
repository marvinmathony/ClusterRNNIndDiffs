#!/usr/bin/env python3
"""
Outer-CV analysis for HP search v3 (continuous encoder).

For each HP combo in runs_thalmann_hp_v3_{combo}/fold{k}/seed_*/:
  - Loads IDRNN (continuous encoder) from checkpoints
  - Extracts z in two ways:
      (a) last valid timestep  (fully informed posterior)
      (b) time-average over all valid timesteps
  - Recomputes test NLL via full LatentRNN_secondstep forward pass
  - Pools test-set z across all outer-CV folds (each participant once)
  - LOO-CV ridge regression -> Pearson r per psychometric scale

Also loads Vanilla RNN from runs_vanilla_thalmann/ for comparison:
  - Recomputes test NLL
  - Extracts time-averaged GRU hidden state for decoding

Outputs (plots_thalmann/hp_v3_outer_cv/):
  summary_table.csv
  nll_vs_phq_tradeoff.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from joblib import Parallel, delayed

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN, Decoder, LatentRNN_secondstep

# ── Config ────────────────────────────────────────────────────────────────────
RUNS_BASE = "runs_thalmann"
PLOT_DIR  = "plots_thalmann/hp_v3_outer_cv"
DGP       = "thalmann"
N_FOLDS   = 3
SEEDS     = [42, 123, 456]
os.makedirs(PLOT_DIR, exist_ok=True)

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

# ── Questionnaire scales ─────────────────────────────────────────────────────
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"
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

quest = pd.read_csv(QUEST_PATH).set_index("ID")
for key, (items, _) in SCALES.items():
    quest[key] = quest[items].mean(axis=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def loo_ridge(Z, y):
    """LOO-CV ridge, z-scored within train split."""
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return float(r), float(p)


def loo_preds_within_fold(Z, y):
    """Like loo_ridge but returns the per-subject OOS predictions only."""
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    return preds


def decode_within_fold(Z_per_fold, subids_per_fold, scale_key):
    """
    LOO ridge within each fold (no cross-fold mixing of latent spaces),
    then pool per-fold predictions and compute a single Pearson r vs pooled y.
    """
    y_list, p_list = [], []
    for Z, sids in zip(Z_per_fold, subids_per_fold):
        y = np.array([quest.loc[sid, scale_key] if sid in quest.index else np.nan
                      for sid in sids])
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            continue
        y_list.append(y[mask])
        p_list.append(loo_preds_within_fold(Z[mask], y[mask]))
    if not y_list:
        return float("nan"), 1.0
    y_pool      = np.concatenate(y_list)
    preds_pool  = np.concatenate(p_list)
    r, p = pearsonr(y_pool, preds_pool)
    return float(r), float(p)


def compute_persubj_nll(logits, c_np):
    """Per-subject mean NLL. logits: (B, ..., A); c_np: matching, -100=padding."""
    B = logits.shape[0]
    logits_flat = logits.reshape(B, -1, logits.shape[-1])
    c_flat      = torch.tensor(c_np, dtype=torch.long).reshape(B, -1)
    mask        = (c_flat >= 0)
    log_p       = F.log_softmax(logits_flat, dim=-1)
    chosen      = log_p.gather(-1, c_flat.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    subj_nll    = -(chosen * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
    return subj_nll.detach().numpy()


# ── IDRNN: extract z + NLL for one combo/fold ────────────────────────────────
def get_idrnn_fold(combo, fold):
    """
    Returns (z_last, z_avg, subj_nll, subids) averaged over seeds,
    or (None, None, None, None) if data is missing.

    z_last: encoder mu at last valid timestep (return_per_timestep=False)
    z_avg:  time-average of encoder mu over all valid timesteps
    """
    run_base = f"{RUNS_BASE}_hp_v3_{combo}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    if not os.path.isdir(run_base):
        return None, None, None, None

    seed_dirs = sorted([
        d for d in os.listdir(run_base)
        if d.startswith("seed_") and os.path.isdir(os.path.join(run_base, d))
    ])
    if not seed_dirs:
        return None, None, None, None

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    c_test_np = np.load(f"{data_dir}/c_test.npy")
    df_test   = pd.read_csv(f"{data_dir}/df_test.csv")
    subids    = df_test["subid"].values if "subid" in df_test.columns else df_test["session"].values

    z_last_seeds = []
    z_avg_seeds  = []
    nll_seeds    = []

    for sd in seed_dirs:
        run_dir  = os.path.join(run_base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue

        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]

        try:
            state = torch.load(ckpt_path, map_location="cpu")

            # ── Encoder: last-timestep z ──────────────────────────────────
            enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                        hid=mc["enc_hidden"], n_tasks=mc.get("n_tasks"),
                        task_emb_dim=mc.get("task_emb_dim", 0),
                        continuous_encoder=mc.get("continuous_encoder", False))
            enc.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
            enc.set_task_ids(task_ids_global)
            enc.eval()

            with torch.no_grad():
                mu_last, _ = enc(xin_test, return_per_timestep=False)  # (B, z_dim)
                mu_all, _  = enc(xin_test, return_per_timestep=True)   # (B, n_blk, T, z_dim)

            z_last_seeds.append(mu_last.numpy())

            # Time-average over valid timesteps
            # Valid mask from choices: where c >= 0
            c_tensor = torch.tensor(c_test_np, dtype=torch.long)  # (B, n_blk, T)
            valid = (c_tensor >= 0).unsqueeze(-1).float()         # (B, n_blk, T, 1)
            z_sum = (mu_all * valid).sum(dim=(1, 2))              # (B, z_dim)
            n_valid = valid.sum(dim=(1, 2)).clamp(min=1)          # (B, 1)
            z_avg_seeds.append((z_sum / n_valid).numpy())

            # ── Full model: test NLL ──────────────────────────────────────
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
                logits, _, _, _, _ = model_full(xin_test, xin_test)
            nll_seeds.append(compute_persubj_nll(logits, c_test_np))

        except Exception as e:
            print(f"    Warning ({combo} fold{fold} {sd}): {e}")

    if not z_last_seeds:
        return None, None, None, None

    z_last = np.stack(z_last_seeds).mean(0)      # (B, z_dim)
    z_avg  = np.stack(z_avg_seeds).mean(0)        # (B, z_dim)
    nll    = np.stack(nll_seeds).mean(0)           # (B,)
    return z_last, z_avg, nll, subids


# ── Vanilla: extract hidden + NLL for one fold ───────────────────────────────
def get_vanilla_fold(fold):
    """Returns (h_avg, subj_nll, subids) or (None, None, None)."""
    fold_dir = f"runs_vanilla_thalmann/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    if not os.path.isdir(fold_dir):
        return None, None, None

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    c_test_np = np.load(f"{data_dir}/c_test.npy")
    df_test   = pd.read_csv(f"{data_dir}/df_test.csv")
    subids    = df_test["subid"].values if "subid" in df_test.columns else df_test["session"].values

    h_seeds   = []
    nll_seeds = []

    for sd in sorted(d for d in os.listdir(fold_dir) if d.startswith("seed_")):
        cfg_path = os.path.join(fold_dir, sd, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue
        ckpt = os.path.join(fold_dir, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt):
            continue

        try:
            state = torch.load(ckpt, map_location="cpu")
            _block_structure = (xin_test.dim() == 4)
            model = AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                               block_structure=_block_structure,
                               n_tasks=mc.get("n_tasks"),
                               task_emb_dim=mc.get("task_emb_dim", 0))
            model.load_state_dict(state)
            model.set_task_ids(task_ids_global)
            model.eval()

            with torch.no_grad():
                logits, _, hidden_tr = model(xin_test)

            # Time-average hidden over valid trials
            valid = torch.tensor(c_test_np >= 0, dtype=torch.float32)
            n_valid = valid.sum(dim=(1, 2), keepdim=True).clamp(min=1)
            h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2)) / n_valid.squeeze(-1)
            h_seeds.append(h_avg.numpy())
            nll_seeds.append(compute_persubj_nll(logits, c_test_np))

        except Exception as e:
            print(f"    Vanilla warning fold{fold}/{sd}: {e}")

    if not h_seeds:
        return None, None, None

    return np.stack(h_seeds).mean(0), np.stack(nll_seeds).mean(0), subids


# ── Process one IDRNN combo ──────────────────────────────────────────────────
def process_combo(combo):
    """Full outer-CV pipeline for one HP combo."""
    all_z_last, all_z_avg, all_nll, all_subids = [], [], [], []

    for fold in range(N_FOLDS):
        z_last, z_avg, nll, subids = get_idrnn_fold(combo, fold)
        if z_last is None:
            continue
        all_z_last.append(z_last)
        all_z_avg.append(z_avg)
        all_nll.append(nll)
        all_subids.append(subids)

    if not all_z_last:
        return None

    z_last_pooled = np.concatenate(all_z_last)
    z_avg_pooled  = np.concatenate(all_z_avg)
    nll_pooled    = np.concatenate(all_nll)
    subids_pooled = np.concatenate(all_subids)

    # Parse HP from combo name
    parts = combo.split("_")
    try:
        uw   = {"00": 0.0, "01": 0.1, "05": 0.5}[parts[0][2:]]
        lmbd = {"005": 0.05, "01": 0.1, "02": 0.2}[parts[1][4:]]
        eh   = int(parts[2][2:])
        h    = int(parts[3][1:])
        z    = int(parts[4][1:])
    except Exception:
        return None

    # ── Pooled-across-folds LOO (optimistic: mixes per-fold latent spaces) ──
    decoding_last = {}
    decoding_avg  = {}
    for key in SCALE_KEYS:
        y = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                      for sid in subids_pooled])
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            decoding_last[key] = (np.nan, 1.0)
            decoding_avg[key]  = (np.nan, 1.0)
            continue
        decoding_last[key] = loo_ridge(z_last_pooled[mask], y[mask])
        decoding_avg[key]  = loo_ridge(z_avg_pooled[mask],  y[mask])

    mean_abs_r_last = float(np.nanmean([abs(decoding_last[k][0]) for k in SCALE_KEYS]))
    mean_abs_r_avg  = float(np.nanmean([abs(decoding_avg[k][0])  for k in SCALE_KEYS]))

    # ── Within-fold LOO (principled: no cross-fold latent-space mixing) ────
    decoding_last_wf = {}
    decoding_avg_wf  = {}
    for key in SCALE_KEYS:
        decoding_last_wf[key] = decode_within_fold(all_z_last, all_subids, key)
        decoding_avg_wf[key]  = decode_within_fold(all_z_avg,  all_subids, key)

    mean_abs_r_last_wf = float(np.nanmean([abs(decoding_last_wf[k][0]) for k in SCALE_KEYS]))
    mean_abs_r_avg_wf  = float(np.nanmean([abs(decoding_avg_wf[k][0])  for k in SCALE_KEYS]))

    mean_nll = float(nll_pooled.mean())

    print(f"  {combo}: nll={mean_nll:.4f}  "
          f"pooled |r|_last={mean_abs_r_last:.3f}  "
          f"wf |r|_last={mean_abs_r_last_wf:.3f}  "
          f"wf |r|_PHQ_last={abs(decoding_last_wf['PHQ'][0]):.3f}  "
          f"n_test={len(subids_pooled)}")

    return {
        "combo": combo, "uw": uw, "lmbd": lmbd,
        "enc_hidden": eh, "hidden": h, "z_dim": z,
        "test_nll": mean_nll,
        "mean_abs_r_last":     mean_abs_r_last,
        "mean_abs_r_avg":      mean_abs_r_avg,
        "mean_abs_r_last_wf":  mean_abs_r_last_wf,
        "mean_abs_r_avg_wf":   mean_abs_r_avg_wf,
        "decoding_last":    decoding_last,
        "decoding_avg":     decoding_avg,
        "decoding_last_wf": decoding_last_wf,
        "decoding_avg_wf":  decoding_avg_wf,
        "n_test": len(subids_pooled),
        "n_folds_ok": len(all_z_last),
    }


# ── Discover combos ──────────────────────────────────────────────────────────
combo_dirs = sorted(glob.glob(f"{RUNS_BASE}_hp_v3_*/"))
combos = [os.path.basename(d.rstrip("/")).replace(f"{os.path.basename(RUNS_BASE)}_hp_v3_", "")
          for d in combo_dirs]
print(f"Found {len(combos)} hp_v3 combos.")

if not combos:
    print("No hp_v3 runs found. Run hyperparam_search_thalmann_v3.sbatch first.")
    raise SystemExit(1)

results_raw = Parallel(n_jobs=8)(delayed(process_combo)(c) for c in combos)
results = [r for r in results_raw if r is not None]
print(f"\n{len(results)} combos with complete outer-CV results.")

if not results:
    print("No complete results. Are the training jobs done?")
    raise SystemExit(1)


# ── Vanilla baseline (pooled) ────────────────────────────────────────────────
print("\nProcessing Vanilla baseline...")
van_h_all, van_nll_all, van_subids_all = [], [], []
for fold in range(N_FOLDS):
    h_avg, nll, subids = get_vanilla_fold(fold)
    if h_avg is not None:
        van_h_all.append(h_avg)
        van_nll_all.append(nll)
        van_subids_all.append(subids)

vanilla_results = None
if van_h_all:
    van_h_pooled      = np.concatenate(van_h_all)
    van_nll_pooled    = np.concatenate(van_nll_all)
    van_subids_pooled = np.concatenate(van_subids_all)
    van_mean_nll      = float(van_nll_pooled.mean())

    van_decoding    = {}
    van_decoding_wf = {}
    for key in SCALE_KEYS:
        y = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                      for sid in van_subids_pooled])
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            van_decoding[key] = (np.nan, 1.0)
        else:
            van_decoding[key] = loo_ridge(van_h_pooled[mask], y[mask])
        van_decoding_wf[key] = decode_within_fold(van_h_all, van_subids_all, key)

    van_mean_abs_r    = float(np.nanmean([abs(van_decoding[k][0])    for k in SCALE_KEYS]))
    van_mean_abs_r_wf = float(np.nanmean([abs(van_decoding_wf[k][0]) for k in SCALE_KEYS]))
    vanilla_results = {
        "test_nll": van_mean_nll,
        "mean_abs_r":     van_mean_abs_r,
        "mean_abs_r_wf":  van_mean_abs_r_wf,
        "decoding":    van_decoding,
        "decoding_wf": van_decoding_wf,
        "n_test": len(van_subids_pooled),
    }
    print(f"  Vanilla: nll={van_mean_nll:.4f}  "
          f"pooled |r|={van_mean_abs_r:.3f}  "
          f"wf |r|={van_mean_abs_r_wf:.3f}  "
          f"wf |r|_PHQ={abs(van_decoding_wf['PHQ'][0]):.3f}  "
          f"n_test={len(van_subids_pooled)}")


# ── Save CSV ─────────────────────────────────────────────────────────────────
rows = []
for d in results:
    row = {k: d[k] for k in ["combo", "uw", "lmbd", "enc_hidden", "hidden", "z_dim",
                               "test_nll",
                               "mean_abs_r_last",    "mean_abs_r_avg",
                               "mean_abs_r_last_wf", "mean_abs_r_avg_wf",
                               "n_test", "n_folds_ok"]}
    for k in SCALE_KEYS:
        row[f"r_last_{k}"]    = d["decoding_last"][k][0]
        row[f"p_last_{k}"]    = d["decoding_last"][k][1]
        row[f"r_avg_{k}"]     = d["decoding_avg"][k][0]
        row[f"p_avg_{k}"]     = d["decoding_avg"][k][1]
        row[f"r_last_wf_{k}"] = d["decoding_last_wf"][k][0]
        row[f"p_last_wf_{k}"] = d["decoding_last_wf"][k][1]
        row[f"r_avg_wf_{k}"]  = d["decoding_avg_wf"][k][0]
        row[f"p_avg_wf_{k}"]  = d["decoding_avg_wf"][k][1]
    rows.append(row)

df = pd.DataFrame(rows).sort_values("test_nll")
csv_path = os.path.join(PLOT_DIR, "summary_table.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved -> {csv_path}")

# Print comparison (within-fold metrics — the principled ones)
print("\n── Within-fold LOO (each decoder trained on one fold's z-space only) ──")
print(f"{'combo':<35} {'NLL':>7} {'wf|r|_last':>11} {'wf|r|_avg':>10} "
      f"{'PHQ_last_wf':>12} {'PHQ_avg_wf':>11}")
df_wf = df.sort_values("mean_abs_r_last_wf", ascending=False)
for _, row in df_wf.iterrows():
    print(f"  {row.combo:<33} {row.test_nll:7.4f} "
          f"{row.mean_abs_r_last_wf:11.3f} {row.mean_abs_r_avg_wf:10.3f} "
          f"{abs(row.get('r_last_wf_PHQ', float('nan'))):12.3f} "
          f"{abs(row.get('r_avg_wf_PHQ',  float('nan'))):11.3f}")

if vanilla_results:
    print(f"\n  {'Vanilla':<33} {vanilla_results['test_nll']:7.4f} "
          f"{'':>11} {vanilla_results['mean_abs_r_wf']:10.3f} "
          f"{'':>12} {abs(vanilla_results['decoding_wf']['PHQ'][0]):11.3f}")

# ── Pareto recommendation (within-fold, PHQ) ─────────────────────────────
print("\n── Pareto-optimal combos (within-fold |r|_PHQ_last vs test_nll) ────────")
df["_abs_r_phq_wf"] = df["r_last_wf_PHQ"].abs()
nll_v = df["test_nll"].values
phq_v = df["_abs_r_phq_wf"].values
pareto = np.zeros(len(nll_v), dtype=bool)
for i in range(len(nll_v)):
    if not any(nll_v[j] <= nll_v[i] and phq_v[j] >= phq_v[i] and
               (nll_v[j] < nll_v[i] or phq_v[j] > phq_v[i])
               for j in range(len(nll_v)) if j != i):
        pareto[i] = True
pareto_df = df[pareto].sort_values("test_nll")
for _, row in pareto_df.iterrows():
    print(f"  {row.combo:<33} nll={row.test_nll:.4f} "
          f"|r|_PHQ_wf={row._abs_r_phq_wf:.3f} "
          f"mean|r|_wf={row.mean_abs_r_last_wf:.3f}")
df.drop(columns=["_abs_r_phq_wf"], inplace=True)


# ── Plot: NLL vs |r_PHQ| tradeoff (within-fold LOO — principled) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, r_col, van_r_col, title_suffix in [
    (axes[0], "r_last_wf_PHQ", "decoding_wf", "z at last valid timestep (within-fold LOO)"),
    (axes[1], "r_avg_wf_PHQ",  "decoding_wf", "z time-averaged (within-fold LOO)"),
]:
    df["_abs_r"] = df[r_col].abs()

    sc = ax.scatter(df["test_nll"], df["_abs_r"],
                    c=df["uw"], cmap="coolwarm", vmin=-0.1, vmax=0.6,
                    s=50, alpha=0.7, edgecolors="none", zorder=3)

    # Pareto front
    nll_v = df["test_nll"].values
    phq_v = df["_abs_r"].values
    pareto = np.zeros(len(nll_v), dtype=bool)
    for i in range(len(nll_v)):
        if not any(nll_v[j] <= nll_v[i] and phq_v[j] >= phq_v[i] and
                   (nll_v[j] < nll_v[i] or phq_v[j] > phq_v[i])
                   for j in range(len(nll_v)) if j != i):
            pareto[i] = True
    par = df[pareto].sort_values("test_nll")
    ax.plot(par["test_nll"], par["_abs_r"],
            color="red", lw=1.2, ls="--", alpha=0.6, zorder=4, label="Pareto front")

    # Vanilla reference (within-fold)
    if vanilla_results:
        van_phq_r = abs(vanilla_results[van_r_col]["PHQ"][0])
        ax.axvline(vanilla_results["test_nll"], color="grey", lw=1.5, ls=":",
                   alpha=0.8, label=f"Vanilla NLL ({vanilla_results['test_nll']:.3f})")
        ax.axhline(van_phq_r, color="grey", lw=1, ls=":", alpha=0.5,
                   label=f"Vanilla |r|_PHQ_wf ({van_phq_r:.3f})")

    ax.set_xlabel("Test NLL (outer CV, pooled)", fontsize=11)
    ax.set_ylabel("|r| PHQ decoding (within-fold LOO)", fontsize=11)
    ax.set_title(f"NLL vs PHQ decodability — {title_suffix}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(sc, ax=ax, label="unif_weight")

    df.drop(columns=["_abs_r"], inplace=True)

fig.suptitle("hp_v3 (continuous encoder): NLL vs PHQ decodability tradeoff  "
             "— decoder trained within each fold's latent space",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "nll_vs_phq_tradeoff.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved nll_vs_phq_tradeoff.png (within-fold LOO)")

print(f"\nDone. All outputs in {PLOT_DIR}/")
