"""Plot per-seed environment-decoding results from env_decoding_per_seed.csv."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="env_decoding_per_seed.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    model_order = ["IDRNN_mu", "vanilla_h", "common_process_h", "informed_decoder_h"]
    model_labels = {
        "IDRNN_mu":           "IDRNN μ\n(encoder, z=3)",
        "vanilla_h":          "Vanilla h\n(10d)",
        "common_process_h":   "Common process h\n(IDRNN decoder, z=0)",
        "informed_decoder_h": "IDRNN decoder h\n(z=μ, informed)",
    }
    colors = {
        "IDRNN_mu":           "#5B9BD5",
        "vanilla_h":          "#ED7D31",
        "common_process_h":   "#a0c4e8",
        "informed_decoder_h": "#2E75B6",
    }

    splits = ["train_data", "test_data"]
    split_label = {"train_data": "train rollouts", "test_data": "test rollouts"}

    # ── Figure 1: bar + strip plot, one panel per split ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        x_pos = np.arange(len(model_order))
        means = [sub[sub["model"] == m]["acc_mean"].mean() for m in model_order]
        sems = [sub[sub["model"] == m]["acc_mean"].std() / np.sqrt(len(sub[sub["model"] == m]))
                for m in model_order]
        bars = ax.bar(x_pos, means, yerr=sems, capsize=4,
                      color=[colors[m] for m in model_order],
                      edgecolor="black", linewidth=0.8, alpha=0.95)

        # overlay individual (dataset, seed) points
        rng = np.random.default_rng(0)
        for i, m in enumerate(model_order):
            pts = sub[sub["model"] == m]["acc_mean"].values
            jitter = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(np.full_like(pts, i, dtype=float) + jitter, pts,
                       color="black", alpha=0.35, s=12, zorder=3)

        ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1, label="chance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=9)
        ax.set_title(split_label[split])
        ax.set_ylim(0.2, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        for b, m, s in zip(bars, means, sems):
            ax.text(b.get_x() + b.get_width() / 2, m + s + 0.02,
                    f"{m:.3f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Decoding accuracy\n(env: low / normal / high)")
    fig.suptitle("Decoding environment from per-seed RNN latents (10 datasets × 5 seeds)",
                 fontsize=11)
    fig.tight_layout()
    out1 = Path(args.outdir) / "env_decoding_per_seed_bars.png"
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ── Figure 2: per-dataset bars (avg over seeds) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    per_ds = (df.groupby(["dataset_id", "split", "model"])["acc_mean"]
              .mean().reset_index())
    for ax, split in zip(axes, splits):
        sub = per_ds[per_ds["split"] == split]
        pivot = sub.pivot(index="dataset_id", columns="model", values="acc_mean")[model_order]
        x_pos = np.arange(len(pivot.index))
        n = len(model_order)
        width = 0.92 / n
        offsets = (np.arange(n) - (n - 1) / 2) * width
        for i, m in enumerate(model_order):
            ax.bar(x_pos + offsets[i], pivot[m].values, width=width,
                   color=colors[m], edgecolor="black", linewidth=0.5,
                   alpha=0.95, label=model_labels[m].replace("\n", " "))
        ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"ds{d}" for d in pivot.index], fontsize=8)
        ax.set_title(split_label[split])
        ax.set_ylim(0.2, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Decoding accuracy")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Per-dataset env-decoding accuracy (each bar = mean over 5 seeds)",
                 fontsize=11)
    fig.tight_layout()
    out2 = Path(args.outdir) / "env_decoding_per_dataset.png"
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ── Figure 3: per-(dataset, seed) lines connecting the four models ─────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        merged = sub.pivot_table(index=["dataset_id", "seed"],
                                 columns="model", values="acc_mean").reset_index()
        for _, r in merged.iterrows():
            ys = [r[m] for m in model_order]
            xs = list(range(len(model_order)))
            ax.plot(xs, ys, color="grey", alpha=0.3, linewidth=0.7, zorder=1)
            ax.scatter(xs, ys,
                       color=[colors[m] for m in model_order],
                       edgecolor="black", s=22, zorder=3, alpha=0.9)
        # group medians
        for i, m in enumerate(model_order):
            ax.scatter([i], [merged[m].median()], marker="_", s=400,
                       color=colors[m], linewidth=3, zorder=4)
        ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(list(range(len(model_order))))
        ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=9)
        ax.set_xlim(-0.5, len(model_order) - 0.5)
        ax.set_title(split_label[split])
        ax.set_ylim(0.2, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Decoding accuracy")
    fig.suptitle("Within-(dataset, seed) pairing — IDRNN μ vs Vanilla h vs Common-process h vs informed IDRNN decoder",
                 fontsize=11)
    fig.tight_layout()
    out3 = Path(args.outdir) / "env_decoding_paired.png"
    fig.savefig(out3, dpi=180, bbox_inches="tight")
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
