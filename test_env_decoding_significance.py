"""
Significance test for IDRNN_mu vs vanilla_h env-decoding accuracy.

Pair structure:
  one observation per (dataset_id, seed) per split → 50 paired differences.
  Diff = vanilla_h_acc - IDRNN_mu_acc (and same for vanilla_h_PCA3 vs IDRNN_mu).

Tests reported:
  1. Wilcoxon signed-rank (paired, non-parametric).
  2. Sign-flip permutation test (10k resamples), distribution-free; respects pairing
     and is robust to seeds-within-dataset clustering.
  3. Cluster-permutation: same as (2) but flip all 5 seeds of a dataset together,
     so the test treats datasets as the independent unit (n=10).
  4. Cohen's d_z on the paired differences (effect-size summary).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def perm_test_signflip(diffs, n_perm=10000, rng=None):
    """One-sample sign-flip permutation, two-sided p-value on mean(diffs) ≠ 0."""
    rng = np.random.default_rng(rng)
    obs = np.mean(diffs)
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        null.append(np.mean(signs * diffs))
    null = np.array(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p, null


def perm_test_cluster(df_pair, n_perm=10000, rng=None):
    """
    Flip the sign of *all 5 seeds within a dataset together*, treating dataset
    as the independent unit. df_pair has columns: dataset_id, seed, diff.
    """
    rng = np.random.default_rng(rng)
    by_ds = df_pair.groupby("dataset_id")["diff"].apply(np.array).to_dict()
    ds_means = {d: arr.mean() for d, arr in by_ds.items()}
    obs = np.mean(list(ds_means.values()))
    ds_ids = list(by_ds.keys())
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(ds_ids))
        flipped = [s * ds_means[d] for s, d in zip(signs, ds_ids)]
        null.append(np.mean(flipped))
    null = np.array(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p, null


def cohen_dz(diffs):
    """Effect size for paired differences."""
    return np.mean(diffs) / np.std(diffs, ddof=1)


def run_pair(df, split, model_a, model_b):
    """Returns dict of test results for accuracy(model_b) - accuracy(model_a)."""
    sub = df[df["split"] == split]
    pivot = sub.pivot_table(index=["dataset_id", "seed"],
                            columns="model", values="acc_mean").reset_index()
    pivot["diff"] = pivot[model_b] - pivot[model_a]
    diffs = pivot["diff"].values

    w_stat, w_p = wilcoxon(diffs, alternative="two-sided")
    obs_mean, perm_p, null_dist = perm_test_signflip(diffs, n_perm=10000, rng=0)
    cl_obs, cl_p, cl_null = perm_test_cluster(pivot[["dataset_id", "seed", "diff"]],
                                              n_perm=10000, rng=0)
    dz = cohen_dz(diffs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    return {
        "split": split, "comparison": f"{model_b} − {model_a}",
        "n_pairs": len(diffs),
        "median_diff": float(np.median(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "cohens_dz": float(dz),
        "wilcoxon_W": float(w_stat),
        "wilcoxon_p": float(w_p),
        "perm_p_signflip": float(perm_p),
        "perm_p_cluster": float(cl_p),
        "n_diff_positive": n_pos,
        "n_diff_negative": n_neg,
        "_diffs": diffs,
        "_null": null_dist,
        "_cluster_null": cl_null,
        "_cluster_obs": cl_obs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="env_decoding_per_seed.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("IDRNN_mu", "vanilla_h"),
        ("IDRNN_mu", "common_process_h"),
        ("IDRNN_mu", "informed_decoder_h"),
        ("informed_decoder_h", "common_process_h"),  # does z=0 beat z=μ?
        ("informed_decoder_h", "vanilla_h"),
    ]
    splits = ["train_data", "test_data"]
    results = []
    for split in splits:
        for a, b in comparisons:
            results.append(run_pair(df, split, a, b))

    # ── Print + save numeric results ─────────────────────────────────────
    rep = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ])
    print("\n=== Significance tests (vanilla vs IDRNN env-decoding) ===")
    print(rep.to_string(index=False))
    rep_path = Path("env_decoding_significance.csv")
    rep.to_csv(rep_path, index=False)
    print(f"\nSaved table → {rep_path}")

    # ── Plot: paired-diff histograms with p-values (one row per comparison) ──
    pretty = {
        "IDRNN_mu":           "IDRNN μ",
        "vanilla_h":          "Vanilla h",
        "common_process_h":   "Common-process h",
        "informed_decoder_h": "Informed IDRNN decoder h",
    }
    n_rows = len(comparisons)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, 2)
    for j, split in enumerate(splits):
        for i, (a, b) in enumerate(comparisons):
            r = next(x for x in results
                     if x["split"] == split and x["comparison"] == f"{b} − {a}")
            ax = axes[i, j]
            diffs = r["_diffs"]
            ax.hist(diffs, bins=18, color="#9467bd", edgecolor="black", alpha=0.85)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(np.mean(diffs), color="red", linewidth=2, label=f"mean = {np.mean(diffs):.3f}")
            ax.axvline(np.median(diffs), color="darkorange", linewidth=2,
                       linestyle=":", label=f"median = {np.median(diffs):.3f}")
            ax.set_title(f"{pretty[b]} − {pretty[a]}  •  {split.replace('_', ' ')}",
                         fontsize=10)
            txt = (f"Wilcoxon p = {r['wilcoxon_p']:.2e}\n"
                   f"sign-flip perm p = {r['perm_p_signflip']:.4f}\n"
                   f"cluster perm p = {r['perm_p_cluster']:.4f}\n"
                   f"Cohen's d_z = {r['cohens_dz']:.2f}\n"
                   f"+ / − : {r['n_diff_positive']} / {r['n_diff_negative']}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            if i == n_rows - 1:
                ax.set_xlabel(f"Δ accuracy  ({pretty[b]} − {pretty[a]})")
            if j == 0:
                ax.set_ylabel("count of cells")
    fig.suptitle("Paired difference in env-decoding accuracy\n"
                 "(50 dataset × seed cells per panel)", fontsize=11)
    fig.tight_layout()
    out1 = Path(args.outdir) / "env_decoding_paired_diffs.png"
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"Saved figure → {out1}")

    # ── Plot: cluster-permutation null distributions (vs IDRNN_mu rows) ──
    headline_pairs = [("IDRNN_mu", "vanilla_h"),
                      ("IDRNN_mu", "common_process_h"),
                      ("IDRNN_mu", "informed_decoder_h"),
                      ("informed_decoder_h", "common_process_h")]
    fig, axes = plt.subplots(len(headline_pairs), 2,
                             figsize=(11, 2.6 * len(headline_pairs)), sharey=False)
    for i, (a, b) in enumerate(headline_pairs):
        for j, split in enumerate(splits):
            r = next(x for x in results
                     if x["split"] == split and x["comparison"] == f"{b} − {a}")
            ax = axes[i, j]
            ax.hist(r["_cluster_null"], bins=40, color="#bbbbbb",
                    edgecolor="black", alpha=0.9)
            ax.axvline(r["_cluster_obs"], color="red", linewidth=2,
                       label=f"observed = {r['_cluster_obs']:.3f}")
            ax.axvline(-r["_cluster_obs"], color="red", linewidth=1,
                       linestyle=":", alpha=0.6)
            ax.set_title(f"{pretty[b]} − {pretty[a]}  •  {split.replace('_',' ')}  "
                         f"(cluster perm p={r['perm_p_cluster']:.4f})",
                         fontsize=9)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            if i == len(headline_pairs) - 1:
                ax.set_xlabel("dataset-mean Δ acc. under sign-flip null")
            if j == 0:
                ax.set_ylabel("# permutations")
    fig.suptitle("Cluster-permutation null: flip all 5 seeds of a dataset together (n=10 datasets)",
                 fontsize=11)
    fig.tight_layout()
    out2 = Path(args.outdir) / "env_decoding_cluster_perm_null.png"
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"Saved figure → {out2}")


if __name__ == "__main__":
    main()
