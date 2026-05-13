#!/usr/bin/env python3
"""
Compare per-subject IDRNN test NLL between v3 (no step 2.5) and v3_step25
(with step 2.5).  Pools across folds; for each subject takes the mean NLL
across the top-K-by-step1-specificity seeds within that subject's fold.

Run after both pipelines complete:
    python compare_v3_vs_v3_step25.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

DATA_DIR = "data_dezfouli"
PLOT_DIR = "plots_dezfouli"
N_FOLDS  = 3
TOP_K    = 5

def per_seed_specificity(suffix):
    """Returns dict[fold] -> dict[seed] -> step1_specificity."""
    out = {}
    for f in range(N_FOLDS):
        out[f] = {}
        d = f"runs_dezfouli_{suffix}/fold{f}"
        if not os.path.isdir(d):
            continue
        for sd in os.listdir(d):
            cfg = f"{d}/{sd}/config.json"
            if not os.path.exists(cfg): continue
            try:
                cf = json.load(open(cfg))
                sp = cf.get("step1_specificity")
                if sp is not None:
                    out[f][int(sd.split("_")[1])] = float(sp)
            except Exception:
                pass
    return out

def per_subj_topk_nll(suffix, top_k=TOP_K):
    """Per-subject IDRNN test NLL averaged across top-K-by-spec seeds in
    that subject's fold."""
    spec = per_seed_specificity(suffix)
    rows = []
    for f in range(N_FOLDS):
        df_test = pd.read_csv(f"{DATA_DIR}/fold{f}/df_test.csv")
        subids  = df_test["subid"].astype(int).values
        # rank seeds by specificity
        ranked = sorted(spec[f].items(), key=lambda kv: -kv[1])[:top_k]
        top_seeds = [s for s, _ in ranked]
        if not top_seeds:
            continue
        per_sub = {sid: [] for sid in subids}
        for sd in top_seeds:
            csv_p = f"{DATA_DIR}/fold{f}/seed_{sd}/{suffix}/rnn_resultslatentmodel.csv"
            if not os.path.exists(csv_p): continue
            nlls = pd.read_csv(csv_p)["normalized_likelihood"].values
            for i, sid in enumerate(subids):
                if i < len(nlls):
                    per_sub[sid].append(nlls[i])
        for sid, vs in per_sub.items():
            if vs:
                rows.append({"subid": int(sid),
                             "fold": f,
                             "nll": float(np.mean(vs)),
                             "n_top_seeds_used": len(vs)})
    return pd.DataFrame(rows)

def main():
    a = per_subj_topk_nll("v3")
    b = per_subj_topk_nll("v3_step25")
    if a.empty or b.empty:
        print(f"v3 rows: {len(a)}, v3_step25 rows: {len(b)} — one or both pipelines incomplete.")
        return
    df = (a.rename(columns={"nll": "nll_v3"})
            .merge(b.rename(columns={"nll": "nll_v3_step25"})[["subid", "nll_v3_step25"]],
                   on="subid", how="inner"))
    print(f"=== Per-subject IDRNN test NLL — top-{TOP_K} by step1 specificity (n={len(df)}) ===")
    print(f"  v3        mean = {df['nll_v3'].mean():.4f}  sd={df['nll_v3'].std():.4f}")
    print(f"  v3_step25 mean = {df['nll_v3_step25'].mean():.4f}  "
          f"sd={df['nll_v3_step25'].std():.4f}")
    print(f"  Δ (v3 − v3_step25) mean = "
          f"{(df['nll_v3'] - df['nll_v3_step25']).mean():+.4f}  "
          f"(positive = step 2.5 better)")
    t, p = ttest_rel(df["nll_v3"], df["nll_v3_step25"])
    print(f"  paired t = {t:.3f}  p = {p:.3g}")

    df.to_csv(os.path.join(PLOT_DIR, "v3_vs_v3_step25_per_subject.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        ax.plot([0, 1], [row["nll_v3"], row["nll_v3_step25"]],
                color="gray", alpha=0.4, linewidth=0.6)
    ax.scatter([0]*len(df), df["nll_v3"], color="#4C72B0", s=42,
                edgecolor="k", linewidth=0.4, label="v3 (no step 2.5)")
    ax.scatter([1]*len(df), df["nll_v3_step25"], color="#DD8452", s=42,
                edgecolor="k", linewidth=0.4, label="v3_step25 (step 2.5)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v3", "v3_step25"])
    ax.set_ylabel("Per-subject test NLL/trial (top-5 by spec)")
    ax.set_title(f"v3 vs v3_step25 — IDRNN paired comparison\n"
                  f"paired t={t:.2f}, p={p:.2g}, n={len(df)}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "v3_vs_v3_step25.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    print(f"Saved → {os.path.join(PLOT_DIR, 'v3_vs_v3_per_subject.csv')}")

if __name__ == "__main__":
    main()
