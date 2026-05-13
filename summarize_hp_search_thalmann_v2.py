#!/usr/bin/env python3
"""
Summarize the Thalmann HP search v2.

For each HP constellation (unif_weight, lmbd, enc_hidden, hidden, z_dim):
  1. Plot z_std over training epochs (collapse diagnostic)
  2. Report cv_val_nll
  3. Decode psychometric questionnaire scores via LOO-CV ridge regression

Outputs (in plots_thalmann/hp_v2/):
  z_collapse_grid.png        — all combos' z_std curves in one grid
  nll_heatmap.png            — cv_val_nll across HP dims
  top10_decoding.png         — decoding for top 10 combos by mean |r|
  summary_table.csv          — ranked results
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.patches
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN

# ── Config ────────────────────────────────────────────────────────────────────
HP_DIR   = "hp_search_results_thalmann_v2"
PLOT_DIR = "plots_thalmann/hp_v2"
DGP      = "thalmann"
os.makedirs(PLOT_DIR, exist_ok=True)

VANILLA_NLL = 0.6348   # mean cv_val_loss across all vanilla seeds/folds

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

# Full train set (165 participants) — no outer fold split
xin_all  = torch.tensor(np.load(f"data_{DGP}/xin_train.npy"), dtype=torch.float32)
df_all   = pd.read_csv(f"data_{DGP}/df_train.csv")
subids_all = df_all["subid"].values if "subid" in df_all.columns else df_all["session"].values

# ── Questionnaire scales ───────────────────────────────────────────────────────
QUEST_PATH     = "data/finalQuestionnaireDataSession1.csv"
PANAS_PA_ITEMS = [0, 2, 4, 6, 8, 14, 16, 18]
PANAS_NA_ITEMS = [1, 3, 5, 7, 9, 11, 13, 15]
SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in PANAS_PA_ITEMS], "PANAS Pos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in PANAS_NA_ITEMS], "PANAS Neg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],       "STICSA Anxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],       "PHQ-9 Depression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],        "CEI Curiosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],        "BIG5 Openness"),
}
SCALE_KEYS   = list(SCALES.keys())
SCALE_LABELS = [SCALES[k][1] for k in SCALE_KEYS]

quest = pd.read_csv(QUEST_PATH).set_index("ID")
for key, (items, _) in SCALES.items():
    quest[key] = quest[items].mean(axis=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def loo_ridge(Z, y):
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


def load_combo(combo_dir):
    """
    Load all seeds for one combo.
    Returns: cfg (from first valid seed), z_std_curve (mean across seeds),
             cv_val_nll (mean across seeds), z_final (mean mu across seeds).
    """
    seed_dirs = sorted(glob.glob(os.path.join(combo_dir, "seed_*")))
    if not seed_dirs:
        return None

    z_std_curves, cv_val_nlls, z_finals = [], [], []
    cfg = None

    for sd in seed_dirs:
        cfg_path = os.path.join(sd, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            c = json.load(f)
        if cfg is None:
            cfg = c

        nll = c.get("cv_val_nll", None)
        if nll is not None:
            cv_val_nlls.append(nll)

        curve_path = os.path.join(sd, "z_std_curve.npy")
        if os.path.exists(curve_path):
            z_std_curves.append(np.load(curve_path))

        # Load encoder and get mu for all 165 train participants
        mc = c["model_config"]
        best_ep = c.get("cv_selected_epoch")
        if best_ep is None:
            continue
        ckpt_path = os.path.join(sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt_path):
            # try latest
            ckpts = sorted(glob.glob(os.path.join(sd, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]

        try:
            state = torch.load(ckpt_path, map_location="cpu")
            enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                        n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
            enc_state = {k[len("encoder."):]: v for k, v in state.items()
                         if k.startswith("encoder.")}
            enc.load_state_dict(enc_state)
            enc.eval()
            enc.set_task_ids(task_ids_global)
            with torch.no_grad():
                mu, _ = enc(xin_all)
            if mu.dim() == 4:
                mu = mu[:, -1, -1, :]   # last block, last timestep
            z_finals.append(mu.numpy())
        except Exception as e:
            print(f"  Warning: could not load encoder for {sd}: {e}")

    if not cv_val_nlls:
        return None

    # Average across seeds
    mean_curve = np.stack(z_std_curves, 0).mean(0) if z_std_curves else None
    mean_z     = np.stack(z_finals, 0).mean(0) if z_finals else None

    return {
        "cfg": cfg,
        "cv_val_nll": float(np.mean(cv_val_nlls)),
        "z_std_curve": mean_curve,
        "z_final": mean_z,
        "n_seeds": len(cv_val_nlls),
    }


# ── Collect results ────────────────────────────────────────────────────────────
print("Loading HP search results...")
combos = sorted(os.listdir(HP_DIR))
results = []

for combo in combos:
    combo_dir = os.path.join(HP_DIR, combo)
    if not os.path.isdir(combo_dir):
        continue
    data = load_combo(combo_dir)
    if data is None:
        print(f"  Skipping {combo} (no valid seeds)")
        continue

    # Parse HP from combo name: uw{}_lmbd{}_eh{}_h{}_z{}
    parts = combo.split("_")
    try:
        uw   = float(parts[0][2:].replace("p", "."))
        lmbd = float(parts[1][4:].replace("p", "."))
        eh   = int(parts[2][2:])
        h    = int(parts[3][1:])
        z    = int(parts[4][1:])
    except Exception:
        print(f"  Could not parse combo name: {combo}")
        continue

    data.update({"combo": combo, "uw": uw, "lmbd": lmbd,
                 "enc_hidden": eh, "hidden": h, "z_dim": z})

    # Decode psychometric scores
    if data["z_final"] is not None:
        z_mat = data["z_final"]
        rs = {}
        for key in SCALE_KEYS:
            subid_col = subids_all
            y_scores = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                                 for sid in subid_col])
            mask = ~np.isnan(y_scores)
            if mask.sum() < 20:
                rs[key] = (np.nan, 1.0)
                continue
            r, p = loo_ridge(z_mat[mask], y_scores[mask])
            rs[key] = (r, p)
        data["decoding"] = rs
        data["mean_abs_r"] = float(np.nanmean([abs(rs[k][0]) for k in SCALE_KEYS]))
    else:
        data["decoding"] = {k: (np.nan, 1.0) for k in SCALE_KEYS}
        data["mean_abs_r"] = np.nan

    results.append(data)
    print(f"  {combo}: cv_val_nll={data['cv_val_nll']:.4f}  "
          f"mean|r|={data['mean_abs_r']:.3f}  n_seeds={data['n_seeds']}")

if not results:
    print("No results found. Have jobs completed?")
    raise SystemExit(1)

print(f"\nLoaded {len(results)} combos.")

# ── Save ranked CSV ────────────────────────────────────────────────────────────
rows = []
for d in results:
    row = {"combo": d["combo"], "uw": d["uw"], "lmbd": d["lmbd"],
           "enc_hidden": d["enc_hidden"], "hidden": d["hidden"], "z_dim": d["z_dim"],
           "cv_val_nll": d["cv_val_nll"], "mean_abs_r": d["mean_abs_r"],
           "n_seeds": d["n_seeds"]}
    for k in SCALE_KEYS:
        row[f"r_{k}"] = d["decoding"][k][0]
        row[f"p_{k}"] = d["decoding"][k][1]
    rows.append(row)

df_results = pd.DataFrame(rows).sort_values("mean_abs_r", ascending=False)
csv_path = os.path.join(PLOT_DIR, "summary_table.csv")
df_results.to_csv(csv_path, index=False)
print(f"Saved ranked table → {csv_path}")
print("\nTop 10 by mean |r|:")
print(df_results[["combo", "uw", "lmbd", "enc_hidden", "hidden", "z_dim",
                   "cv_val_nll", "mean_abs_r"]].head(10).to_string(index=False))


# ── Plot 1: z_std collapse curves (grid) ──────────────────────────────────────
print("\nPlotting z_std collapse curves...")
fig_rows = 6   # one row per unif_weight × lmbd combo (3×3=9, but use ~6 rows for layout)
n_with_curves = sum(1 for d in results if d["z_std_curve"] is not None)

UW_VALS   = sorted(set(d["uw"]   for d in results))
LMBD_VALS = sorted(set(d["lmbd"] for d in results))
EH_VALS   = sorted(set(d["enc_hidden"] for d in results))
H_VALS    = sorted(set(d["hidden"] for d in results))
Z_VALS    = sorted(set(d["z_dim"] for d in results))

COLORS = {3: "steelblue", 5: "darkorange", 10: "green"}
LSTYLE = {5: "-", 10: "--"}

n_rows = len(UW_VALS) * len(LMBD_VALS)
n_cols = len(EH_VALS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows),
                         sharex=False, sharey=False)
axes = np.array(axes).reshape(n_rows, n_cols)

for row_i, uw in enumerate(UW_VALS):
    for row_j, lmbd in enumerate(LMBD_VALS):
        ax_row = row_i * len(LMBD_VALS) + row_j
        for col_i, eh in enumerate(EH_VALS):
            ax = axes[ax_row, col_i]
            ax.set_title(f"uw={uw} λ={lmbd} eh={eh}", fontsize=7)
            ax.axhline(0.0, color="gray", lw=0.5, ls=":")
            for d in results:
                if d["uw"] != uw or d["lmbd"] != lmbd or d["enc_hidden"] != eh:
                    continue
                if d["z_std_curve"] is None:
                    continue
                curve = d["z_std_curve"]
                epochs = np.arange(1, len(curve) + 1)
                label = f"h={d['hidden']} z={d['z_dim']}"
                ax.plot(epochs, curve,
                        color=COLORS.get(d["z_dim"], "gray"),
                        ls=LSTYLE.get(d["hidden"], "-"),
                        lw=0.8, alpha=0.85, label=label)
            ax.set_xlabel("epoch", fontsize=6)
            ax.set_ylabel("z_std", fontsize=6)
            ax.tick_params(labelsize=5)

# legend on last axis
handles, labels = axes[0, -1].get_legend_handles_labels()
if handles:
    axes[0, -1].legend(handles, labels, fontsize=5, loc="upper left")

plt.suptitle("z_std across training (collapse diagnostic)\ncolor=z_dim  linestyle=dec_hidden",
             fontsize=10, y=1.01)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "z_collapse_grid.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ── Plot 2: NLL heatmap (uw × lmbd for best enc_hidden/hidden/z_dim combo) ───
print("Plotting NLL heatmap...")
fig, axes = plt.subplots(1, len(Z_VALS), figsize=(5 * len(Z_VALS), 4))
if len(Z_VALS) == 1:
    axes = [axes]

for ax, z in zip(axes, Z_VALS):
    mat = np.full((len(UW_VALS), len(LMBD_VALS)), np.nan)
    for d in results:
        if d["z_dim"] != z:
            continue
        uw_i   = UW_VALS.index(d["uw"])
        lmbd_i = LMBD_VALS.index(d["lmbd"])
        cur = d["cv_val_nll"]
        if np.isnan(mat[uw_i, lmbd_i]) or cur < mat[uw_i, lmbd_i]:
            mat[uw_i, lmbd_i] = cur

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=max(0.60, np.nanmin(mat) - 0.02),
                   vmax=min(0.80, np.nanmax(mat) + 0.02))
    ax.set_xticks(range(len(LMBD_VALS))); ax.set_xticklabels(LMBD_VALS)
    ax.set_yticks(range(len(UW_VALS)));   ax.set_yticklabels(UW_VALS)
    ax.set_xlabel("λ"); ax.set_ylabel("unif_weight")
    ax.set_title(f"z_dim={z}\n(best enc_hidden/hidden)")
    plt.colorbar(im, ax=ax, label="cv_val_nll ↓")
    # annotate cells
    for i in range(len(UW_VALS)):
        for j in range(len(LMBD_VALS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.axhline(0.0 - 0.5, color="red", lw=1.5, ls="--", label=f"vanilla={VANILLA_NLL:.3f}")

plt.suptitle(f"cv_val_nll (flat cross-entropy)  |  vanilla baseline = {VANILLA_NLL:.3f}",
             fontsize=11)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "nll_heatmap.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ── Plot 3: Top-10 decoding bar chart ─────────────────────────────────────────
print("Plotting top-10 decoding...")
TOP_N = 10
top = df_results.head(TOP_N)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()

for ax_i, (_, row) in enumerate(top.iterrows()):
    ax = axes[ax_i]
    rs   = [row[f"r_{k}"] for k in SCALE_KEYS]
    ps   = [row[f"p_{k}"] for k in SCALE_KEYS]
    bars = ax.bar(range(len(SCALE_KEYS)), [abs(r) for r in rs],
                  color=["#e07b7b" if r >= 0 else "#7b9be0" for r in rs])
    for bi, (r, p) in enumerate(zip(rs, ps)):
        if p < 0.05:
            ax.text(bi, abs(r) + 0.005, "*", ha="center", va="bottom", fontsize=9)
    ax.axhline(VANILLA_NLL - VANILLA_NLL, color="gray")   # dummy — add vanilla per-scale if available
    ax.set_xticks(range(len(SCALE_KEYS)))
    ax.set_xticklabels(SCALE_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("|Pearson r|", fontsize=7)
    short = row["combo"].replace("uw", "uw=").replace("lmbd", "λ=").replace("_eh", " eh=").replace("_h", " h=").replace("_z", " z=")
    ax.set_title(f"#{ax_i+1}: {short}\nnll={row['cv_val_nll']:.3f}  mean|r|={row['mean_abs_r']:.3f}",
                 fontsize=6.5)

plt.suptitle(f"Top-{TOP_N} combos by mean |r| (decoding of psychometric scores)\nred=positive r, blue=negative r  *=p<0.05",
             fontsize=10)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "top10_decoding.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ── Plot 4: Mean |r| vs NLL scatter per combo ─────────────────────────────────
print("Plotting decoding vs NLL scatter...")
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap("tab10")
uw_color = {uw: cmap(i) for i, uw in enumerate(UW_VALS)}

for d in results:
    if np.isnan(d["mean_abs_r"]):
        continue
    ax.scatter(d["cv_val_nll"], d["mean_abs_r"],
               color=uw_color[d["uw"]], s=20 + d["z_dim"] * 3,
               alpha=0.7, edgecolors="none")

ax.axvline(VANILLA_NLL, color="red", ls="--", lw=1.5, label=f"vanilla NLL={VANILLA_NLL:.3f}")
ax.set_xlabel("cv_val_nll (lower = better fit)")
ax.set_ylabel("mean |Pearson r| (decoding, higher = better)")
ax.set_title("NLL vs decoding across all HP combos")

legend_patches = [matplotlib.patches.Patch(color=uw_color[uw], label=f"uw={uw}")
                  for uw in UW_VALS]
ax.legend(handles=legend_patches + [
    matplotlib.lines.Line2D([0], [0], color="red", ls="--", label=f"vanilla NLL")
], fontsize=8)

# Annotate top-5 combos
for _, row in df_results.head(5).iterrows():
    ax.annotate(row["combo"].split("_z")[0], xy=(row["cv_val_nll"], row["mean_abs_r"]),
                fontsize=5, ha="left")

plt.tight_layout()
out = os.path.join(PLOT_DIR, "nll_vs_decoding_scatter.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")

print("\nDone. All plots in", PLOT_DIR)
