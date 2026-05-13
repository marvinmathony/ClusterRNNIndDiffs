#!/usr/bin/env python3
"""
Diagnostic for "why doesn't IDRNN beat vanilla on dezfouli?".

Tests the hypothesis that IDRNN's per-subject z is largely redundant with
the perseveration signal vanilla extracts directly from `prev_choice`.
Concretely: per subject, compute (IDRNN test NLL − Vanilla test NLL); if the
gap grows with the subject's stay-probability, vanilla wins most where the
prev_choice signal is strongest.

Inputs (existing v1 outer-CV artifacts; nothing is retrained):
  data_dezfouli/foldF/df_test.csv                       (test subids per fold)
  data_dezfouli/foldF/seed_S/rnn_resultslatentmodel.csv (IDRNN per-subj NLL)
  data_dezfouli/foldF/seed_S/rnn_resultsvanilla.csv     (Vanilla per-subj NLL)
  data/for_plos.csv                                     (raw choices/diag)

Outputs:
  plots_dezfouli/idrnn_failure_per_subject.csv
  plots_dezfouli/idrnn_failure_diagnostic.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

DATA_DIR = "data_dezfouli"
RAW_CSV  = "data/for_plos.csv"
PLOT_DIR = "plots_dezfouli"
N_FOLDS  = 3
os.makedirs(PLOT_DIR, exist_ok=True)


def per_subject_stay_prob():
    df = pd.read_csv(RAW_CSV)
    df["choice_binary"] = df["key"].map({"R1": 0, "R2": 1}).astype(int)
    df["subid"] = df["ID"].astype("category").cat.codes
    out = {}
    diag = {}
    for sid, dfi in df.groupby("subid"):
        same = total = 0
        for _, blk in dfi.groupby("block"):
            seq = blk["choice_binary"].values
            if len(seq) >= 2:
                same  += int((seq[1:] == seq[:-1]).sum())
                total += len(seq) - 1
        out[int(sid)] = same / max(total, 1)
        diag[int(sid)] = dfi["diag"].iloc[0]
    return pd.DataFrame({
        "subid": list(out.keys()),
        "stay_prob": [out[s] for s in out],
        "diag":     [diag[s] for s in out],
    })


def aggregate_perfold_subj_nll(model_csv_name):
    """Average per-subject NLL across seeds within each fold; return long DF
    with columns (subid, normalized_likelihood, fold)."""
    rows = []
    for fold in range(N_FOLDS):
        fold_dir = os.path.join(DATA_DIR, f"fold{fold}")
        df_test  = pd.read_csv(os.path.join(fold_dir, "df_test.csv"))
        # subids in fold-test order
        subids_in_test_order = df_test["subid"].astype(int).values

        seed_dirs = sorted([s for s in os.listdir(fold_dir)
                            if s.startswith("seed_")])
        per_subj = {sid: [] for sid in subids_in_test_order}
        for sd in seed_dirs:
            p = os.path.join(fold_dir, sd, model_csv_name)
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            if "normalized_likelihood" not in df.columns:
                continue
            nlls = df["normalized_likelihood"].values
            for i, sid in enumerate(subids_in_test_order):
                if i < len(nlls):
                    per_subj[sid].append(nlls[i])
        for sid, vs in per_subj.items():
            if vs:
                rows.append({"subid": sid,
                             "normalized_likelihood": float(np.mean(vs)),
                             "fold": fold,
                             "n_seeds": len(vs)})
    return pd.DataFrame(rows)


def main():
    print("Loading per-subject stay probabilities + diagnoses ...")
    stay_df = per_subject_stay_prob()
    print(f"  {len(stay_df)} subjects;  mean stay-prob = "
          f"{stay_df['stay_prob'].mean():.3f}")

    print("Aggregating IDRNN per-subject NLL across folds & seeds ...")
    idrnn_df   = aggregate_perfold_subj_nll("rnn_resultslatentmodel.csv")
    print("Aggregating Vanilla per-subject NLL ...")
    vanilla_df = aggregate_perfold_subj_nll("rnn_resultsvanilla.csv")
    print("Aggregating CP RNN per-subject NLL ...")
    cprnn_df   = aggregate_perfold_subj_nll("rnn_results_common_process.csv")

    df = (idrnn_df.rename(columns={"normalized_likelihood": "nll_idrnn"})
                  .drop(columns="n_seeds")
          .merge(vanilla_df.rename(columns={"normalized_likelihood": "nll_vanilla"})
                           .drop(columns=["n_seeds", "fold"]),
                 on="subid", how="inner")
          .merge(cprnn_df.rename(columns={"normalized_likelihood": "nll_cprnn"})
                           .drop(columns=["n_seeds", "fold"]),
                 on="subid", how="inner")
          .merge(stay_df, on="subid", how="inner"))

    df["gap_idrnn_minus_vanilla"] = df["nll_idrnn"] - df["nll_vanilla"]
    df["gap_idrnn_minus_cprnn"]   = df["nll_idrnn"] - df["nll_cprnn"]
    df["gap_vanilla_minus_cprnn"] = df["nll_vanilla"] - df["nll_cprnn"]

    out_csv = os.path.join(PLOT_DIR, "idrnn_failure_per_subject.csv")
    df.sort_values("stay_prob").to_csv(out_csv, index=False)
    print(f"\nSaved per-subject CSV → {out_csv}\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    n = len(df)
    print(f"=== N = {n} subjects ===")
    for col, label in [("nll_idrnn",   "IDRNN"),
                        ("nll_vanilla", "Vanilla"),
                        ("nll_cprnn",   "CP RNN")]:
        print(f"  {label:<8}: mean NLL/trial = {df[col].mean():.4f} "
              f"± {df[col].std():.4f}")
    print(f"\n  Mean (IDRNN − Vanilla): {df['gap_idrnn_minus_vanilla'].mean():+.4f}")
    print(f"  Mean (IDRNN − CP RNN ): {df['gap_idrnn_minus_cprnn'].mean():+.4f}  "
          f"(< 0 ⇒ z helps over CP)")
    print()

    # Correlation: per-subject gap vs stay-prob
    r1, p1 = pearsonr(df["stay_prob"], df["gap_idrnn_minus_vanilla"])
    rs1, ps1 = spearmanr(df["stay_prob"], df["gap_idrnn_minus_vanilla"])
    print(f"  stay-prob × (IDRNN − Vanilla NLL):  Pearson r = {r1:+.3f} (p={p1:.3g}); "
          f"Spearman ρ = {rs1:+.3f} (p={ps1:.3g})")
    r2, p2 = pearsonr(df["stay_prob"], df["gap_idrnn_minus_cprnn"])
    print(f"  stay-prob × (IDRNN − CP RNN NLL):   Pearson r = {r2:+.3f} (p={p2:.3g})")
    r3, p3 = pearsonr(df["stay_prob"], df["nll_vanilla"])
    print(f"  stay-prob × Vanilla NLL:            Pearson r = {r3:+.3f} (p={p3:.3g})")
    r4, p4 = pearsonr(df["stay_prob"], df["nll_idrnn"])
    print(f"  stay-prob × IDRNN NLL:              Pearson r = {r4:+.3f} (p={p4:.3g})")

    # Quartile breakdown
    df["sp_q"] = pd.qcut(df["stay_prob"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    quart = (df.groupby("sp_q", observed=True)
               .agg(n=("subid", "count"),
                    nll_idrnn=("nll_idrnn", "mean"),
                    nll_vanilla=("nll_vanilla", "mean"),
                    gap=("gap_idrnn_minus_vanilla", "mean"),
                    stay_prob=("stay_prob", "mean")))
    print("\n  Stay-prob quartile breakdown:")
    print(quart.round(4).to_string())

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    diag_colors = {"Healthy":"#3C8DAD", "Depression":"#D7263D", "Bipolar":"#F26430"}

    # (a) NLL vs stay-prob, both architectures
    ax = axes[0, 0]
    for diag, sub in df.groupby("diag"):
        ax.scatter(sub["stay_prob"], sub["nll_idrnn"], alpha=0.55,
                   color=diag_colors.get(diag, "gray"), s=42, marker="o",
                   edgecolor="k", linewidth=0.4,
                   label=f"IDRNN  – {diag}")
        ax.scatter(sub["stay_prob"], sub["nll_vanilla"], alpha=0.55,
                   color=diag_colors.get(diag, "gray"), s=42, marker="^",
                   edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Per-subject stay probability")
    ax.set_ylabel("Test NLL / trial (lower = better)")
    ax.set_title("Per-subject NLL — IDRNN (○) vs Vanilla (△)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, loc="upper right", ncol=1)

    # (b) Gap vs stay-prob (the key diagnostic)
    ax = axes[0, 1]
    for diag, sub in df.groupby("diag"):
        ax.scatter(sub["stay_prob"], sub["gap_idrnn_minus_vanilla"], alpha=0.7,
                   color=diag_colors.get(diag, "gray"), s=46,
                   edgecolor="k", linewidth=0.4, label=diag)
    # OLS line
    from numpy.polynomial.polynomial import Polynomial
    m, b = np.polyfit(df["stay_prob"], df["gap_idrnn_minus_vanilla"], 1)
    xs = np.linspace(df["stay_prob"].min(), df["stay_prob"].max(), 50)
    ax.plot(xs, m*xs + b, "k--", linewidth=1.2,
            label=f"OLS  (r={r1:+.2f}, p={p1:.2g})")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Per-subject stay probability")
    ax.set_ylabel("IDRNN NLL − Vanilla NLL  (>0: IDRNN worse)")
    ax.set_title("Per-subject NLL gap vs stay-prob")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")

    # (c) Quartile bars
    ax = axes[1, 0]
    qmeans = quart["gap"].values
    qsem   = (df.groupby("sp_q", observed=True)["gap_idrnn_minus_vanilla"]
                .sem()).values
    xpos = np.arange(len(qmeans))
    ax.bar(xpos, qmeans, yerr=qsem, capsize=5,
           color=["#5A9BD3","#7CC36F","#F2C14E","#E07A5F"],
           edgecolor="k", linewidth=0.6)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(quart.index)
    ax.set_ylabel("Mean (IDRNN − Vanilla NLL)  ± SEM")
    ax.set_title("Gap by stay-prob quartile")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (d) Group means (IDRNN, CP RNN, Vanilla) for context
    ax = axes[1, 1]
    means = [df["nll_cprnn"].mean(), df["nll_idrnn"].mean(), df["nll_vanilla"].mean()]
    sems  = [df["nll_cprnn"].sem(),  df["nll_idrnn"].sem(),  df["nll_vanilla"].sem()]
    cols  = ["#9ECAE1", "#1F77B4", "#E67E22"]
    labels = ["CP RNN", "IDRNN", "Vanilla"]
    ax.bar(np.arange(3), means, yerr=sems, capsize=5,
           color=cols, edgecolor="k", linewidth=0.6)
    ax.axhline(np.log(2), color="k", linestyle="--", linewidth=0.8,
               label=f"random (log 2 = {np.log(2):.3f})")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean test NLL/trial")
    ax.set_title("Architecture comparison — pooled (v1)")
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Why IDRNN doesn't beat Vanilla on dezfouli — per-subject diagnostic",
                 fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(PLOT_DIR, "idrnn_failure_diagnostic.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure → {out_png}")


if __name__ == "__main__":
    main()
