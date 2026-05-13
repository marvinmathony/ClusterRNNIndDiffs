#!/usr/bin/env python3
"""
Within- vs across-subject decomposition of perseveration (stay-prob).

Motivation
----------
We saw that step 2.5 fine-tuning (decoder fine-tune after IDRNN distillation)
helps on dezfouli. One hypothesis: dezfouli has substantial *within*-participant
drift in perseveration, so a single static latent per subject under-fits and
the step-2.5 decoder fine-tune lets the readout adapt to that drift.

This script tests the premise by computing per-(subject, block) stay-prob
P(c_t == c_{t-1}) and decomposing its variance into between-subject
(individual differences) and within-subject (across blocks) components.
A binomial null tells us how much within-subject variance is just
finite-sample noise vs. real drift.

Datasets
--------
- dezfouli: (101 subj, 12 blocks, 202 trials) — primary
- thalmann task0: (236 subj, 30 blocks, 10 trials) — comparison (block 0..29)

Outputs (plots_perseveration_variance/):
  per_block_stay_prob_dezfouli.csv / _thalmann_task0.csv
  variance_decomposition.csv
  perseveration_within_vs_between.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots_perseveration_variance"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Stay-prob per (subject, block) ────────────────────────────────────────────
def per_block_stay_prob(c_arr):
    """
    c_arr: (n_subj, n_blocks, T) with -100 padding.
    Returns:
        sp:  (n_subj, n_blocks) stay probability (NaN if a block has <2 valid trials)
        n:   (n_subj, n_blocks) number of valid (c_t, c_{t-1}) pairs
    """
    n_subj, n_blocks, T = c_arr.shape
    sp = np.full((n_subj, n_blocks), np.nan, dtype=float)
    n  = np.zeros((n_subj, n_blocks), dtype=int)
    for s in range(n_subj):
        for b in range(n_blocks):
            seq = c_arr[s, b]
            valid = seq[seq >= 0]
            if len(valid) >= 2:
                same  = int((valid[1:] == valid[:-1]).sum())
                total = len(valid) - 1
                sp[s, b] = same / total
                n[s, b]  = total
    return sp, n


# ── Variance decomposition ────────────────────────────────────────────────────
def decompose_variance(sp, n_pairs):
    """
    Inputs
    ------
    sp:      (n_subj, n_blocks) stay-prob per block (NaN = missing)
    n_pairs: (n_subj, n_blocks) number of (c_t,c_{t-1}) pairs

    Returns dict with:
      grand_mean, between_var, within_var, total_var,
      icc1                 — ICC(1): between / (between + within)
      binom_within_var     — expected within-subj variance under per-subject
                              binomial null (no true drift)
      excess_within_var    — within_var - binom_within_var (real drift)
      between_sd, within_sd, binom_within_sd, excess_within_sd
      mean_blocks_per_subj
    """
    valid = ~np.isnan(sp)
    # subject means weighted by trial-pair count (so blocks with more pairs count more)
    w = n_pairs.astype(float) * valid
    w_sum = w.sum(axis=1)
    keep = w_sum > 0
    sp_k = sp[keep]
    w_k  = w[keep]
    n_k  = n_pairs[keep]
    n_subj = sp_k.shape[0]

    subj_mean = np.nansum(sp_k * w_k, axis=1) / w_k.sum(axis=1)
    grand_mean = float(np.average(subj_mean, weights=w_k.sum(axis=1)))

    # between-subject variance (unweighted across subjects, treats each subj equally)
    between_var = float(np.var(subj_mean, ddof=1))

    # within-subject variance — pooled across subjects, weighted by # pairs
    resid_sq_w = []
    n_blocks_used = []
    for s in range(n_subj):
        m = ~np.isnan(sp_k[s])
        if m.sum() < 2:
            continue
        diffs = sp_k[s, m] - subj_mean[s]
        # weighted variance with finite-sample correction
        ws = w_k[s, m]
        var_s = np.sum(ws * diffs ** 2) / ws.sum()
        resid_sq_w.append(var_s)
        n_blocks_used.append(m.sum())
    within_var = float(np.mean(resid_sq_w))

    total_var = between_var + within_var
    icc1 = between_var / total_var if total_var > 0 else np.nan

    # Binomial null: each block's stay-prob has variance p(1-p)/n_t around the
    # subject's true rate. Use subject mean as plug-in for p; average expected
    # variance across subjects, weighted by block count.
    expected_per_subj = []
    for s in range(n_subj):
        m = ~np.isnan(sp_k[s]) & (n_k[s] > 0)
        if m.sum() < 2:
            continue
        p = subj_mean[s]
        per_block = p * (1 - p) / n_k[s, m]
        expected_per_subj.append(np.mean(per_block))
    binom_within_var = float(np.mean(expected_per_subj))
    excess_within_var = max(within_var - binom_within_var, 0.0)

    return dict(
        n_subj=n_subj,
        mean_blocks_per_subj=float(np.mean(n_blocks_used)),
        grand_mean=grand_mean,
        between_var=between_var, between_sd=np.sqrt(between_var),
        within_var=within_var,   within_sd=np.sqrt(within_var),
        total_var=total_var,
        icc1=icc1,
        binom_within_var=binom_within_var,
        binom_within_sd=np.sqrt(binom_within_var),
        excess_within_var=excess_within_var,
        excess_within_sd=np.sqrt(excess_within_var),
    )


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_dezfouli():
    """Reconstruct full 101-subject c-array (sorted by subid)."""
    f0_tr_df = pd.read_csv("data_dezfouli/fold0/df_train.csv")
    f0_te_df = pd.read_csv("data_dezfouli/fold0/df_test.csv")
    tr_c = np.load("data_dezfouli/fold0/c_train.npy")
    te_c = np.load("data_dezfouli/fold0/c_test.npy")
    all_c    = np.concatenate([tr_c, te_c], axis=0)
    all_sids = np.concatenate([f0_tr_df["subid"].values, f0_te_df["subid"].values])
    order = np.argsort(all_sids)
    return all_c[order], all_sids[order]


def load_thalmann_task0():
    """thalmann blocks 0..29 are the 2-armed binary task. 10 trials each."""
    tr_c = np.load("data_thalmann/fold0/c_train.npy")
    te_c = np.load("data_thalmann/fold0/c_test.npy")
    all_c = np.concatenate([tr_c, te_c], axis=0)
    return all_c[:, :30, :]  # drop the 4-way block 30


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_overview(sp_d, sp_t, dec_d, dec_t, savepath):
    """4-panel overview comparing dezfouli (top) and thalmann task0 (bottom)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for row, (name, sp, dec) in enumerate(
        [("dezfouli (12 blocks × 202 trials)", sp_d, dec_d),
         ("thalmann task0 (30 blocks × 10 trials)", sp_t, dec_t)]
    ):
        # (1) Per-subject distribution of block stay-prob, sorted by subj mean
        subj_mean = np.nanmean(sp, axis=1)
        order = np.argsort(subj_mean)
        sp_sorted = sp[order]
        ax = axes[row, 0]
        # boxplot per subject
        bp_data = [sp_sorted[s][~np.isnan(sp_sorted[s])] for s in range(sp_sorted.shape[0])]
        ax.boxplot(bp_data, positions=np.arange(len(bp_data)), widths=0.7,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", alpha=0.5, linewidth=0.3),
                   medianprops=dict(color="navy", linewidth=0.5),
                   whiskerprops=dict(linewidth=0.3),
                   capprops=dict(linewidth=0.3))
        ax.plot(np.arange(len(bp_data)), subj_mean[order], "k-", lw=0.6, alpha=0.7)
        ax.set_xlabel("Subject (sorted by mean stay-prob)")
        ax.set_ylabel("Per-block stay-prob")
        ax.set_title(f"{name}\nblock-level stay-prob per subject")
        ax.set_ylim(0, 1)
        if len(bp_data) > 30:
            step = max(1, len(bp_data) // 10)
            ax.set_xticks(np.arange(0, len(bp_data), step))
            ax.set_xticklabels(np.arange(0, len(bp_data), step))

        # (2) Between vs within SD bars (with binomial null)
        ax = axes[row, 1]
        labels = ["between-subject", "within-subject\n(observed)",
                  "within-subject\n(binomial null)", "within-subject\n(excess)"]
        vals = [dec["between_sd"], dec["within_sd"],
                dec["binom_within_sd"], dec["excess_within_sd"]]
        colors = ["#3b6", "#36b", "#aaa", "#b63"]
        ax.bar(range(4), vals, color=colors)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("SD of stay-prob")
        ax.set_title(f"Variance decomposition\nICC(1) = {dec['icc1']:.3f}")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # (3) Within-subject SD distribution across subjects, with binomial-null
        # mean line (per-subject expected SD using subj mean as plug-in p)
        within_sd_per_subj = np.nanstd(sp, axis=1, ddof=1)
        ax = axes[row, 2]
        ax.hist(within_sd_per_subj, bins=25, color="#36b", alpha=0.7, edgecolor="white")
        ax.axvline(dec["binom_within_sd"], color="grey", linestyle="--",
                   label=f"binomial null SD = {dec['binom_within_sd']:.3f}")
        ax.axvline(dec["between_sd"], color="#3b6", linestyle="-",
                   label=f"between-subj SD = {dec['between_sd']:.3f}")
        ax.set_xlabel("Within-subject SD of block stay-prob")
        ax.set_ylabel("# subjects")
        ax.set_title("Within-subject SD distribution")
        ax.legend(fontsize=8)

        # (4) Block-to-block lag-1 autocorrelation across subjects
        # For each subject, corr(block_t, block_{t+1}) is not well defined
        # (single pair). Instead, compute pearson r between stay-prob in
        # block b and block b+1 across subjects, for each adjacent block pair.
        n_blocks = sp.shape[1]
        ax = axes[row, 3]
        rs = []
        for b in range(n_blocks - 1):
            x = sp[:, b]; y = sp[:, b + 1]
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() > 5:
                rs.append(np.corrcoef(x[m], y[m])[0, 1])
            else:
                rs.append(np.nan)
        ax.plot(np.arange(1, n_blocks), rs, "o-", color="#36b")
        ax.axhline(0, color="grey", linestyle=":", lw=0.5)
        ax.set_xlabel("Block transition (b → b+1)")
        ax.set_ylabel("Pearson r across subjects")
        ax.set_title("Lag-1 block correlation\n(across-subject)")
        ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dezfouli...")
    c_d, sids_d = load_dezfouli()
    print(f"  c shape {c_d.shape}, {len(sids_d)} subjects")
    sp_d, n_d = per_block_stay_prob(c_d)
    print(f"  stay-prob: per-block matrix {sp_d.shape}, "
          f"valid blocks per subj: {(~np.isnan(sp_d)).sum(axis=1).min()}–"
          f"{(~np.isnan(sp_d)).sum(axis=1).max()}")
    dec_d = decompose_variance(sp_d, n_d)

    print("\nLoading thalmann task0...")
    c_t = load_thalmann_task0()
    print(f"  c shape {c_t.shape}")
    sp_t, n_t = per_block_stay_prob(c_t)
    print(f"  stay-prob: per-block matrix {sp_t.shape}")
    dec_t = decompose_variance(sp_t, n_t)

    # ── Save per-block stay-prob (long format) ────────────────────────────────
    rows_d = []
    for s in range(sp_d.shape[0]):
        for b in range(sp_d.shape[1]):
            if not np.isnan(sp_d[s, b]):
                rows_d.append({"subid": int(sids_d[s]), "block": b,
                               "stay_prob": float(sp_d[s, b]),
                               "n_pairs": int(n_d[s, b])})
    pd.DataFrame(rows_d).to_csv(
        os.path.join(OUT_DIR, "per_block_stay_prob_dezfouli.csv"), index=False)

    rows_t = []
    for s in range(sp_t.shape[0]):
        for b in range(sp_t.shape[1]):
            if not np.isnan(sp_t[s, b]):
                rows_t.append({"subj_idx": s, "block": b,
                               "stay_prob": float(sp_t[s, b]),
                               "n_pairs": int(n_t[s, b])})
    pd.DataFrame(rows_t).to_csv(
        os.path.join(OUT_DIR, "per_block_stay_prob_thalmann_task0.csv"), index=False)

    # ── Save decomposition summary ────────────────────────────────────────────
    summary = pd.DataFrame([
        {"dataset": "dezfouli",       **dec_d},
        {"dataset": "thalmann_task0", **dec_t},
    ])
    summary.to_csv(os.path.join(OUT_DIR, "variance_decomposition.csv"), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_overview(sp_d, sp_t, dec_d, dec_t,
                  os.path.join(OUT_DIR, "perseveration_within_vs_between.png"))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n══════════════ Variance decomposition ══════════════")
    for name, dec in [("dezfouli", dec_d), ("thalmann_task0", dec_t)]:
        print(f"\n{name}:")
        print(f"  n_subj            = {dec['n_subj']}")
        print(f"  blocks/subj (avg) = {dec['mean_blocks_per_subj']:.1f}")
        print(f"  grand mean P(stay)= {dec['grand_mean']:.3f}")
        print(f"  between-subj SD   = {dec['between_sd']:.4f}")
        print(f"  within-subj SD    = {dec['within_sd']:.4f}")
        print(f"  binomial null SD  = {dec['binom_within_sd']:.4f}")
        print(f"  excess within SD  = {dec['excess_within_sd']:.4f}  "
              f"(real drift after subtracting binomial noise)")
        print(f"  ICC(1)            = {dec['icc1']:.3f}  "
              f"(between / (between + within))")
        ratio = (dec['excess_within_sd'] / dec['between_sd']
                 if dec['between_sd'] > 0 else np.nan)
        print(f"  excess-within / between SD = {ratio:.3f}")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
