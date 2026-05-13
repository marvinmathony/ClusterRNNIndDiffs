#!/usr/bin/env python3
"""
Does step 2.5 capture within-subject drift in perseveration?

Setup
-----
For each (fold, seed) in runs_dezfouli_v3_step25_z1/, we have two checkpoints:
    epoch{N:04d}_pre_step25.pt   ← step-2 weights (frozen-decoder, static μ)
    epoch{N:04d}.pt              ← step-2.5 weights (unfrozen, per-trial μ)

For each subject's held-out test set (assembled across folds so every subject
is out-of-sample exactly once), forward both models and compute per-(subject,
block) model stay-probability, where
    model_stay_block(s, b) = mean over trials of P(c_t == c_{t-1} | model).

We compare to the behavioral block stay-probability and ask:
  • Across-subject:   does subj-mean P(stay) match between model and behavior?
  • Within-subject:   does the per-block residual (block − subj-mean) match?

The hypothesis: step 2.5 specifically improves within-subject tracking.

Each model is run in its training-time inference mode:
    step-2     → return_per_timestep = False  (static, full-session μ)
    step-2.5   → return_per_timestep = True   (per-trial causal μ)

Outputs (plots_perseveration_variance/step25_tracking/):
  per_block_predictions_seed{S}.csv  — long-form (s, b, seed, beh, m2, m25)
  step25_within_subj_summary.csv     — per-seed within/between r for each model
  step25_within_subj_tracking.png    — scatter + bar plot
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sys
sys.path.insert(0, ".")
from modelsandtraining import (
    Decoder, IDRNN, LatentRNN_secondstep,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--run_suffix", default="v3_step25_z1",
                help="Suffix of runs_dezfouli_<suffix> to analyze.")
ap.add_argument("--seeds", default=None,
                help="Comma-separated seeds. Default = all seeds with both ckpts.")
ap.add_argument("--folds", default="0,1,2")
ap.add_argument("--out_dir",
                default="plots_perseveration_variance/step25_tracking")
args = ap.parse_args()

DGP = "dezfouli"
RUN_SUFFIX = args.run_suffix
RUNS_BASE = f"runs_{DGP}_{RUN_SUFFIX}"
DATA_DIR  = f"data_{DGP}"
OUT_DIR   = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
FOLDS = [int(f) for f in args.folds.split(",")]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_fold(fold):
    """Returns dicts with xin (B, n_blocks, T, in_dim), c (B, n_blocks, T),
    subids and diag for both train and test halves of the given fold."""
    out = {}
    for split in ("train", "test"):
        df = pd.read_csv(f"{DATA_DIR}/fold{fold}/df_{split}.csv")
        xin = np.load(f"{DATA_DIR}/fold{fold}/xin_{split}.npy")
        c   = np.load(f"{DATA_DIR}/fold{fold}/c_{split}.npy")
        out[split] = dict(
            xin=xin, c=c,
            subids=df["subid"].values, diag=df["diag"].values,
        )
    return out


def list_seeds(fold):
    """Seeds with both pre- and post-step25 checkpoints in the given fold."""
    fold_dir = os.path.join(RUNS_BASE, f"fold{fold}")
    if not os.path.isdir(fold_dir):
        return []
    seeds = []
    for d in sorted(os.listdir(fold_dir)):
        if not d.startswith("seed_"):
            continue
        seed = int(d.split("_")[1])
        cfg_p = os.path.join(fold_dir, d, "config.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        ep = cfg.get("cv_selected_epoch")
        if ep is None:
            continue
        ckpt_dir = os.path.join(fold_dir, d, "checkpoints")
        pre  = os.path.join(ckpt_dir, f"epoch{ep:04d}_pre_step25.pt")
        post = os.path.join(ckpt_dir, f"epoch{ep:04d}.pt")
        if os.path.exists(pre) and os.path.exists(post):
            seeds.append(seed)
    return seeds


def build_model(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=cfg["z_dim"],
                hid=mc["enc_hidden"])
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=cfg["z_dim"],
                  hid=cfg["hidden"], A=cfg["A"])
    model = LatentRNN_secondstep(
        encoder=enc, hid=cfg["hidden"], z_dim=cfg["z_dim"],
        in_dim=mc["dec_in_dim"], A=cfg["A"], decoder=dec,
    )
    return model.to(device)


# ── Per-block stay metrics ────────────────────────────────────────────────────
def behavioral_per_block_stay_prob(c_arr):
    """(B, n_blocks, T) with -100 padding → (B, n_blocks) stay-prob, NaN if <2 valid."""
    B, K, T = c_arr.shape
    sp = np.full((B, K), np.nan)
    for s in range(B):
        for b in range(K):
            seq = c_arr[s, b]
            v = seq[seq >= 0]
            if len(v) >= 2:
                sp[s, b] = (v[1:] == v[:-1]).mean()
    return sp


def model_per_block_stay_prob(model, xin_t, c_t, return_per_timestep):
    """
    Model's predicted P(c_t == c_{t-1}) averaged over valid trials per (subject, block).
    Returns (B, n_blocks) with NaN for blocks with <1 valid t≥1 trial.
    """
    model.return_per_timestep = return_per_timestep
    model.eval()
    with torch.no_grad():
        logits, *_ = model(xin_t, xin_t, sample_z=False)   # same input to enc/dec
        probs = F.softmax(logits, dim=-1)                  # (B, K, T, A)
    probs_np = probs.cpu().numpy()
    c_np = c_t.cpu().numpy()
    B, K, T, A = probs_np.shape
    sp = np.full((B, K), np.nan)
    for s in range(B):
        for b in range(K):
            seq = c_np[s, b]
            stays = []
            for t in range(1, T):
                if seq[t-1] < 0 or seq[t] < 0:
                    continue
                prev = int(seq[t-1])
                stays.append(probs_np[s, b, t, prev])
            if stays:
                sp[s, b] = float(np.mean(stays))
    return sp


# ── Forward both variants for a (fold, seed) ──────────────────────────────────
def predict_one_seed(fold, seed, fold_data):
    """Returns dict with per-block stay arrays for behavior + step2 + step25,
    over the **test** subjects of this fold (held-out)."""
    run_dir = os.path.join(RUNS_BASE, f"fold{fold}", f"seed_{seed}")
    cfg_p   = os.path.join(run_dir, "config.json")
    with open(cfg_p) as f:
        cfg = json.load(f)
    ep = cfg["cv_selected_epoch"]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    pre_path  = os.path.join(ckpt_dir, f"epoch{ep:04d}_pre_step25.pt")
    post_path = os.path.join(ckpt_dir, f"epoch{ep:04d}.pt")

    xin_te = torch.as_tensor(fold_data["test"]["xin"],
                              dtype=torch.float32, device=device)
    c_te   = torch.as_tensor(fold_data["test"]["c"],
                              dtype=torch.long,    device=device)
    subids = fold_data["test"]["subids"]
    diag   = fold_data["test"]["diag"]

    # Behavioral
    beh = behavioral_per_block_stay_prob(fold_data["test"]["c"])

    # Step-2 model: load pre-step25 weights, run with return_per_timestep=False
    model = build_model(cfg)
    sd2 = torch.load(pre_path, map_location=device)
    model.load_state_dict(sd2)
    m2 = model_per_block_stay_prob(model, xin_te, c_te, return_per_timestep=False)

    # Step-2.5 model: load post-step25 weights, run with return_per_timestep=True
    model2 = build_model(cfg)
    sd25 = torch.load(post_path, map_location=device)
    model2.load_state_dict(sd25)
    m25 = model_per_block_stay_prob(model2, xin_te, c_te, return_per_timestep=True)

    return dict(beh=beh, m2=m2, m25=m25,
                subids=subids, diag=diag, n_blocks=beh.shape[1])


# ── Stat helpers ──────────────────────────────────────────────────────────────
def subj_mean_and_resid(arr):
    """arr: (B, K). Returns (subj_mean (B,), residual (B, K) with NaN preserved)."""
    sm = np.nanmean(arr, axis=1)
    return sm, arr - sm[:, None]


def safe_pearsonr(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    seeds_per_fold = {f: list_seeds(f) for f in FOLDS}
    all_seeds = sorted(set.intersection(*[set(s) for s in seeds_per_fold.values()]))
    if args.seeds:
        wanted = [int(s) for s in args.seeds.split(",")]
        all_seeds = [s for s in all_seeds if s in wanted]
    print(f"Folds: {FOLDS}  Seeds (in all folds): {len(all_seeds)} → {all_seeds}")
    if not all_seeds:
        print("No seeds with both checkpoints in all folds. Exiting.")
        return

    fold_data = {f: load_fold(f) for f in FOLDS}

    # ── Per-seed per-block tables ────────────────────────────────────────────
    rows = []
    seed_summaries = []
    for seed in all_seeds:
        # Concat test subjects across folds (each subject appears once OOS)
        beh_list, m2_list, m25_list, sid_list, diag_list = [], [], [], [], []
        for f in FOLDS:
            r = predict_one_seed(f, seed, fold_data[f])
            beh_list.append(r["beh"]);  m2_list.append(r["m2"])
            m25_list.append(r["m25"])
            sid_list.append(r["subids"]); diag_list.append(r["diag"])
        beh = np.concatenate(beh_list, axis=0)
        m2  = np.concatenate(m2_list,  axis=0)
        m25 = np.concatenate(m25_list, axis=0)
        sids = np.concatenate(sid_list)
        diag = np.concatenate(diag_list)
        # sort by subid for stable output
        order = np.argsort(sids)
        beh = beh[order]; m2 = m2[order]; m25 = m25[order]
        sids = sids[order]; diag = diag[order]
        n_subj, n_blocks = beh.shape
        print(f"\n── seed {seed}: assembled {n_subj} OOS subjects × {n_blocks} blocks ──")

        # Long format dump
        for s in range(n_subj):
            for b in range(n_blocks):
                if np.isnan(beh[s, b]):
                    continue
                rows.append(dict(
                    seed=seed, subid=int(sids[s]), diag=str(diag[s]),
                    block=b,
                    beh=float(beh[s, b]),
                    m2=float(m2[s, b]) if not np.isnan(m2[s, b]) else np.nan,
                    m25=float(m25[s, b]) if not np.isnan(m25[s, b]) else np.nan,
                ))

        # Decompose: subj means and residuals
        beh_sm,  beh_r  = subj_mean_and_resid(beh)
        m2_sm,   m2_r   = subj_mean_and_resid(m2)
        m25_sm,  m25_r  = subj_mean_and_resid(m25)

        # Across-subject correlations (subj means)
        r2_b,  p2_b,  n2_b  = safe_pearsonr(m2_sm,  beh_sm)
        r25_b, p25_b, n25_b = safe_pearsonr(m25_sm, beh_sm)

        # Within-subject correlations (residuals, all (s,b) pooled)
        r2_w,  p2_w,  n2_w  = safe_pearsonr(m2_r.flatten(),  beh_r.flatten())
        r25_w, p25_w, n25_w = safe_pearsonr(m25_r.flatten(), beh_r.flatten())

        # Δ residual: does step25's adjustment track behavior residual?
        delta = m25 - m2
        _, delta_r = subj_mean_and_resid(delta)
        r_dlt, p_dlt, n_dlt = safe_pearsonr(delta_r.flatten(), beh_r.flatten())

        # Variance decomposition of model predictions (mirror beh decomposition)
        def _var_decomp(arr):
            sm = np.nanmean(arr, axis=1)
            wsd = np.nanstd(arr - sm[:, None], axis=1, ddof=1)
            return float(np.nanstd(sm, ddof=1)), float(np.nanmean(wsd))

        beh_bsd, beh_wsd = _var_decomp(beh)
        m2_bsd,  m2_wsd  = _var_decomp(m2)
        m25_bsd, m25_wsd = _var_decomp(m25)

        seed_summaries.append(dict(
            seed=seed,
            n_subj=n_subj, n_blocks=n_blocks,
            # between-subject (subject-mean) correlations
            r_betw_step2=r2_b,   p_betw_step2=p2_b,
            r_betw_step25=r25_b, p_betw_step25=p25_b,
            # within-subject (block-residual) correlations
            r_with_step2=r2_w,   p_with_step2=p2_w,  n_with=n2_w,
            r_with_step25=r25_w, p_with_step25=p25_w,
            # Δ-residual ↔ behaviour-residual
            r_delta_resid=r_dlt, p_delta_resid=p_dlt,
            # variance decomposition
            beh_between_sd=beh_bsd,  beh_within_sd=beh_wsd,
            m2_between_sd=m2_bsd,    m2_within_sd=m2_wsd,
            m25_between_sd=m25_bsd,  m25_within_sd=m25_wsd,
        ))
        print(f"  between-subj r: step2={r2_b:.3f}  step25={r25_b:.3f}")
        print(f"  within-subj  r: step2={r2_w:.3f}  step25={r25_w:.3f}  "
              f"(Δ-resid r={r_dlt:.3f})")
        print(f"  within SD   beh={beh_wsd:.3f}  step2={m2_wsd:.3f}  step25={m25_wsd:.3f}")

    df_long = pd.DataFrame(rows)
    df_long.to_csv(os.path.join(OUT_DIR, "per_block_predictions_long.csv"),
                   index=False)
    df_sum = pd.DataFrame(seed_summaries)
    df_sum.to_csv(os.path.join(OUT_DIR, "step25_within_subj_summary.csv"),
                  index=False)

    # ── Aggregate plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (1) Within-subject correlations: step2 vs step25 across seeds
    ax = axes[0, 0]
    x = df_sum["r_with_step2"].values
    y = df_sum["r_with_step25"].values
    ax.scatter(x, y, alpha=0.7)
    lo, hi = -0.05, max(x.max(), y.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    for i, s in enumerate(df_sum["seed"]):
        ax.annotate(int(s), (x[i], y[i]), fontsize=6, alpha=0.6)
    ax.set_xlabel("Within-subj r — step 2 model")
    ax.set_ylabel("Within-subj r — step 2.5 model")
    n_better = (y > x).sum()
    ax.set_title(f"Within-subj tracking improves with step 2.5\n"
                 f"step25 > step2 in {n_better}/{len(x)} seeds")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # (2) Between-subject correlations: step2 vs step25 across seeds
    ax = axes[0, 1]
    x = df_sum["r_betw_step2"].values
    y = df_sum["r_betw_step25"].values
    ax.scatter(x, y, alpha=0.7)
    lo, hi = min(x.min(), y.min()) - 0.05, 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax.set_xlabel("Between-subj r — step 2 model")
    ax.set_ylabel("Between-subj r — step 2.5 model")
    ax.set_title("Between-subject tracking (subject means)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # (3) Δ-residual ↔ behaviour residual
    ax = axes[0, 2]
    ax.hist(df_sum["r_delta_resid"].values, bins=15,
            color="#b63", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="grey", linestyle="--")
    ax.axvline(df_sum["r_delta_resid"].mean(), color="black", linestyle="-",
               label=f"mean = {df_sum['r_delta_resid'].mean():.3f}")
    ax.set_xlabel("Pearson r — (step25 − step2) residual\nvs. behavioral residual")
    ax.set_ylabel("# seeds")
    ax.set_title("Step 2.5 *adjustment* tracks within-subj drift?")
    ax.legend()

    # (4) Within-subject SD: beh, step2, step25 (per seed)
    ax = axes[1, 0]
    seeds = df_sum["seed"].values
    width = 0.27
    x_pos = np.arange(len(seeds))
    ax.bar(x_pos - width, df_sum["beh_within_sd"].values,
           width, label="behavior", color="#3b6")
    ax.bar(x_pos,         df_sum["m2_within_sd"].values,
           width, label="step 2",   color="#36b")
    ax.bar(x_pos + width, df_sum["m25_within_sd"].values,
           width, label="step 2.5", color="#b63")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seeds, rotation=90, fontsize=7)
    ax.set_ylabel("Within-subject SD of P(stay)")
    ax.set_title("Within-subject prediction variance")
    ax.legend(fontsize=8)

    # (5) Pooled scatter: behavioral residual vs step2 residual (one seed = mean)
    # Use seed-averaged predictions for clarity
    df_long_avg = df_long.groupby(["subid", "block"]).agg(
        beh=("beh", "mean"), m2=("m2", "mean"), m25=("m25", "mean")
    ).reset_index()
    bsm = df_long_avg.groupby("subid")["beh"].transform("mean")
    m2sm = df_long_avg.groupby("subid")["m2"].transform("mean")
    m25sm = df_long_avg.groupby("subid")["m25"].transform("mean")
    df_long_avg["beh_resid"]  = df_long_avg["beh"]  - bsm
    df_long_avg["m2_resid"]   = df_long_avg["m2"]   - m2sm
    df_long_avg["m25_resid"]  = df_long_avg["m25"]  - m25sm

    ax = axes[1, 1]
    ax.scatter(df_long_avg["m2_resid"], df_long_avg["beh_resid"],
               alpha=0.25, s=8, color="#36b")
    r2, _ = pearsonr(df_long_avg["m2_resid"], df_long_avg["beh_resid"])
    ax.set_xlabel("Step-2 model residual (block − subj mean)")
    ax.set_ylabel("Behavior residual")
    ax.set_title(f"Step 2 within-subj scatter\nr = {r2:.3f}  (seed-averaged)")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)

    ax = axes[1, 2]
    ax.scatter(df_long_avg["m25_resid"], df_long_avg["beh_resid"],
               alpha=0.25, s=8, color="#b63")
    r25, _ = pearsonr(df_long_avg["m25_resid"], df_long_avg["beh_resid"])
    ax.set_xlabel("Step-2.5 model residual (block − subj mean)")
    ax.set_ylabel("Behavior residual")
    ax.set_title(f"Step 2.5 within-subj scatter\nr = {r25:.3f}  (seed-averaged)")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step25_within_subj_tracking.png"), dpi=150)
    plt.close(fig)

    # ── Final printout ────────────────────────────────────────────────────────
    print("\n══════════════════════ Aggregate ══════════════════════")
    print(f"n_seeds                = {len(df_sum)}")
    print(f"between-subj r:  step2  = {df_sum['r_betw_step2'].mean():.3f}  "
          f"± {df_sum['r_betw_step2'].std():.3f}")
    print(f"                step25  = {df_sum['r_betw_step25'].mean():.3f}  "
          f"± {df_sum['r_betw_step25'].std():.3f}")
    print(f"within-subj  r:  step2  = {df_sum['r_with_step2'].mean():.3f}  "
          f"± {df_sum['r_with_step2'].std():.3f}")
    print(f"                step25  = {df_sum['r_with_step25'].mean():.3f}  "
          f"± {df_sum['r_with_step25'].std():.3f}")
    print(f"Δ-resid r (step25−step2 vs beh resid):  "
          f"{df_sum['r_delta_resid'].mean():.3f}  "
          f"± {df_sum['r_delta_resid'].std():.3f}")
    n_better = int((df_sum['r_with_step25'] > df_sum['r_with_step2']).sum())
    print(f"step25 > step2 within-subj r in {n_better}/{len(df_sum)} seeds")
    print(f"\nOutputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
