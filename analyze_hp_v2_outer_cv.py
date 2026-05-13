#!/usr/bin/env python3
"""
Outer-CV decoding analysis for HP search v2.

For each HP combo in hp_search_results_thalmann_v2/:
  - Loads IDRNN encoder from runs_thalmann_hp_v2_{combo}/fold{k}/seed_*/
  - Runs encoder on held-out test subjects (data_thalmann/fold{k}/xin_test.npy)
  - Pools test-set z across all folds (true outer-CV, each participant appears once)
  - LOO-CV ridge regression → Pearson r per psychometric scale

Outputs (plots_thalmann/hp_v2_outer_cv/):
  summary_table.csv
  nll_vs_decoding_scatter.png
  top15_decoding.png
  nll_heatmap_z_dim.png
  best_vs_vanilla_bar.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.patches
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from joblib import Parallel, delayed

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN

# ── Config ────────────────────────────────────────────────────────────────────
HP_DIR    = "hp_search_results_thalmann_v2"
RUNS_BASE = "runs_thalmann"
PLOT_DIR  = "plots_thalmann/hp_v2_outer_cv"
DGP       = "thalmann"
N_FOLDS   = 3
SEEDS     = [42, 123, 456]
os.makedirs(PLOT_DIR, exist_ok=True)

VANILLA_NLL = 0.6348

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

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


def get_z_test_for_fold(combo, fold):
    """
    Load encoder from runs_thalmann_hp_v2_{combo}/fold{fold}/seed_*/,
    run on xin_test, return (z_test, subids_test) averaged over seeds.
    """
    run_base = f"{RUNS_BASE}_hp_v2_{combo}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"

    if not os.path.isdir(run_base):
        return None, None

    seed_dirs = sorted([
        d for d in os.listdir(run_base)
        if d.startswith("seed_") and os.path.isdir(os.path.join(run_base, d))
    ])
    if not seed_dirs:
        return None, None

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)
    df_test   = pd.read_csv(f"{data_dir}/df_test.csv")
    subids    = df_test["subid"].values if "subid" in df_test.columns else df_test["session"].values

    z_seeds = []
    for sd in seed_dirs:
        run_dir  = os.path.join(run_base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc        = cfg["model_config"]
        best_ep   = cfg.get("cv_selected_epoch")
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
            enc   = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                          n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
            enc_state = {k[len("encoder."):]: v for k, v in state.items()
                         if k.startswith("encoder.")}
            enc.load_state_dict(enc_state)
            enc.eval()
            enc.set_task_ids(task_ids_global)
            with torch.no_grad():
                mu, _ = enc(xin_test)
            if mu.dim() == 4:
                mu = mu[:, -1, -1, :]   # last block, last timestep → (B_test, z_dim)
            z_seeds.append(mu.numpy())
        except Exception as e:
            print(f"    Warning ({combo} fold{fold} {sd}): {e}")

    if not z_seeds:
        return None, None

    z_test = np.stack(z_seeds, 0).mean(0)  # average over seeds → (B_test, z_dim)
    return z_test, subids


def process_combo(combo):
    """Full outer-CV decoding pipeline for one HP combo."""
    combo_dir = os.path.join(HP_DIR, combo)
    if not os.path.isdir(combo_dir):
        return None

    # Read cv_val_nll from original HP search (trained on all 165)
    configs = glob.glob(os.path.join(combo_dir, "seed_*/config.json"))
    nlls = []
    z_stds = []
    for p in configs:
        with open(p) as f: c = json.load(f)
        if "cv_val_nll" in c: nlls.append(c["cv_val_nll"])
        if "cv_z_std_final" in c: z_stds.append(c["cv_z_std_final"])

    # Parse HP from combo name
    parts = combo.split("_")
    try:
        uw_raw   = parts[0][2:]
        uw       = {"00": 0.0, "01": 0.1, "05": 0.5}.get(uw_raw, float("nan"))
        lmbd_raw = parts[1][4:]
        lmbd     = {"005": 0.05, "01": 0.1, "02": 0.2}.get(lmbd_raw, float("nan"))
        eh = int(parts[2][2:])
        h  = int(parts[3][1:])
        z  = int(parts[4][1:])
    except Exception:
        return None

    # Collect test-set z across all outer folds
    all_z, all_subids = [], []
    for fold in range(N_FOLDS):
        z_test, subids = get_z_test_for_fold(combo, fold)
        if z_test is None:
            continue
        all_z.append(z_test)
        all_subids.append(subids)

    if not all_z:
        return None

    z_pooled      = np.concatenate(all_z,      axis=0)   # (B_total, z_dim)
    subids_pooled = np.concatenate(all_subids, axis=0)   # (B_total,)

    # Match questionnaire scores
    decoding = {}
    for key in SCALE_KEYS:
        y = np.array([quest.loc[sid, key] if sid in quest.index else np.nan
                      for sid in subids_pooled])
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            decoding[key] = (np.nan, 1.0)
            continue
        r, p = loo_ridge(z_pooled[mask], y[mask])
        decoding[key] = (r, p)

    mean_abs_r = float(np.nanmean([abs(decoding[k][0]) for k in SCALE_KEYS]))

    print(f"  {combo}: nll={np.mean(nlls):.4f}  mean|r|={mean_abs_r:.3f}  "
          f"n_test={len(subids_pooled)}")

    return {
        "combo":       combo,
        "uw":          uw,
        "lmbd":        lmbd,
        "enc_hidden":  eh,
        "hidden":      h,
        "z_dim":       z,
        "cv_val_nll":  float(np.mean(nlls)) if nlls else float("nan"),
        "cv_z_std":    float(np.mean(z_stds)) if z_stds else float("nan"),
        "mean_abs_r":  mean_abs_r,
        "decoding":    decoding,
        "n_test":      len(subids_pooled),
        "n_folds_ok":  len(all_z),
    }


# ── Run in parallel ────────────────────────────────────────────────────────────
combos = sorted(os.listdir(HP_DIR))
print(f"Processing {len(combos)} combos with {N_FOLDS} folds, {len(SEEDS)} seeds each...")

results_raw = Parallel(n_jobs=8)(delayed(process_combo)(c) for c in combos)
results = [r for r in results_raw if r is not None]
print(f"\n{len(results)} combos with outer-CV decoding results.")

if not results:
    print("No results — have outer-CV training jobs finished?")
    raise SystemExit(1)

# ── Save CSV ───────────────────────────────────────────────────────────────────
rows = []
for d in results:
    row = {k: d[k] for k in ["combo","uw","lmbd","enc_hidden","hidden","z_dim",
                               "cv_val_nll","cv_z_std","mean_abs_r","n_test","n_folds_ok"]}
    for k in SCALE_KEYS:
        row[f"r_{k}"] = d["decoding"][k][0]
        row[f"p_{k}"] = d["decoding"][k][1]
    rows.append(row)

df = pd.DataFrame(rows).sort_values("mean_abs_r", ascending=False)
csv_path = os.path.join(PLOT_DIR, "summary_table.csv")
df.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

print("\nTop 20 by mean |r|:")
cols = ["combo","uw","lmbd","enc_hidden","hidden","z_dim","cv_val_nll","mean_abs_r"] + \
       [f"r_{k}" for k in SCALE_KEYS]
print(df[cols].head(20).to_string(index=False))


# ── Plot 1: NLL vs decoding scatter ───────────────────────────────────────────
uw_colors = {0.0: "steelblue", 0.1: "darkorange", 0.5: "green"}
z_sizes   = {3: 30, 5: 60, 10: 120}

fig, ax = plt.subplots(figsize=(9, 6))
for d in results:
    ax.scatter(d["cv_val_nll"], d["mean_abs_r"],
               color=uw_colors.get(d["uw"], "gray"),
               s=z_sizes.get(d["z_dim"], 60),
               alpha=0.75, edgecolors="none")

ax.axvline(VANILLA_NLL, color="red", ls="--", lw=1.5, label=f"vanilla NLL={VANILLA_NLL:.3f}")
ax.set_xlabel("cv_val_nll (↓ better behavioral fit)")
ax.set_ylabel("mean |Pearson r| outer-CV decoding (↑ better)")
ax.set_title("HP search: NLL vs outer-CV psychometric decoding\n(marker size = z_dim; color = unif_weight)")

legend_elems = [matplotlib.patches.Patch(color=c, label=f"uw={uw}") for uw, c in uw_colors.items()]
legend_elems += [matplotlib.lines.Line2D([0],[0], color="red", ls="--", label="vanilla NLL")]
ax.legend(handles=legend_elems, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nll_vs_decoding_scatter.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved nll_vs_decoding_scatter.png")


# ── Plot 2: Top 15 bar chart ───────────────────────────────────────────────────
TOP_N = 15
top = df.head(TOP_N)

ncols = 5
nrows = (TOP_N + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=False)
axes = np.array(axes).flatten()

for ax_i, (_, row) in enumerate(top.iterrows()):
    ax = axes[ax_i]
    rs = [row[f"r_{k}"] for k in SCALE_KEYS]
    ps = [row[f"p_{k}"] for k in SCALE_KEYS]
    colors = ["#c0392b" if r >= 0 else "#2980b9" for r in rs]
    ax.bar(range(len(SCALE_KEYS)), [abs(r) for r in rs], color=colors)
    for bi, (r, p) in enumerate(zip(rs, ps)):
        if not np.isnan(p) and p < 0.05:
            ax.text(bi, abs(r) + 0.005, "*", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(SCALE_KEYS)))
    ax.set_xticklabels(SCALE_LABELS, rotation=40, ha="right", fontsize=6)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("|r|", fontsize=7)
    ax.set_title(
        f"#{ax_i+1} uw={row.uw} λ={row.lmbd}\neh={row.enc_hidden} h={row.hidden} z={row.z_dim}\n"
        f"nll={row.cv_val_nll:.3f}  |r|={row.mean_abs_r:.3f}",
        fontsize=6.5
    )

for ax in axes[TOP_N:]:
    ax.axis("off")

plt.suptitle("Top-15 HP combos by outer-CV mean |r|\nred=positive r  blue=negative r  *=p<0.05",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top15_decoding.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved top15_decoding.png")


# ── Plot 3: Best IDRNN vs vanilla per scale ────────────────────────────────────
# Load vanilla outer-CV decoding (from analyze_unif_sweep_thalmann results for uw=0.0)
# Use the best combo overall and best combo with nll < 0.76
best_overall = df.iloc[0]
best_good_nll = df[df["cv_val_nll"] < 0.76].iloc[0] if len(df[df["cv_val_nll"] < 0.76]) > 0 else None

# Vanilla r values from the unif sweep (uw=0.0 outer CV)
# Load from the sweep plots csv if available, else report without vanilla per-scale
vanilla_r_path = "plots_thalmann/hp_v2_outer_cv/vanilla_r.csv"
vanilla_r = None
if os.path.exists(vanilla_r_path):
    vanilla_r = pd.read_csv(vanilla_r_path, index_col=0)

fig, axes = plt.subplots(1, 2 if best_good_nll is not None else 1,
                          figsize=(14 if best_good_nll is not None else 8, 5))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

titles = ["Best overall (highest |r|)", "Best with NLL < 0.76"]
rows_to_plot = [best_overall]
if best_good_nll is not None:
    rows_to_plot.append(best_good_nll)

for ax, row, title in zip(axes, rows_to_plot, titles):
    rs = [row[f"r_{k}"] for k in SCALE_KEYS]
    ps = [row[f"p_{k}"] for k in SCALE_KEYS]
    x  = np.arange(len(SCALE_KEYS))
    w  = 0.35

    bars = ax.bar(x - w/2, [abs(r) for r in rs], w,
                  color=["#c0392b" if r>=0 else "#2980b9" for r in rs],
                  label=f"IDRNN (uw={row.uw} λ={row.lmbd} h={row.hidden} z={row.z_dim})")

    for bi, (r, p) in enumerate(zip(rs, ps)):
        if not np.isnan(p) and p < 0.05:
            ax.text(x[bi] - w/2, abs(r) + 0.005, "*", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("|Pearson r| (outer-CV)")
    ax.set_title(f"{title}\nNLL={row.cv_val_nll:.3f}  mean|r|={row.mean_abs_r:.3f}")
    ax.legend(fontsize=7)

plt.suptitle("Best IDRNN HP combos: outer-CV psychometric decoding", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "best_vs_vanilla_bar.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved best_vs_vanilla_bar.png")


# ── Plot 4: NLL heatmap by z_dim ──────────────────────────────────────────────
UW_VALS   = sorted(df.uw.dropna().unique())
LMBD_VALS = sorted(df.lmbd.dropna().unique())
Z_VALS    = sorted(df.z_dim.dropna().unique())

fig, axes = plt.subplots(1, len(Z_VALS), figsize=(5 * len(Z_VALS), 4))
for ax, z in zip(axes, Z_VALS):
    sub = df[df.z_dim == z]
    mat_nll = np.full((len(UW_VALS), len(LMBD_VALS)), np.nan)
    mat_r   = np.full((len(UW_VALS), len(LMBD_VALS)), np.nan)
    for _, row in sub.iterrows():
        if row.uw not in UW_VALS or row.lmbd not in LMBD_VALS:
            continue
        ui = UW_VALS.index(row.uw)
        li = LMBD_VALS.index(row.lmbd)
        # Best (lowest) NLL per cell
        if np.isnan(mat_nll[ui, li]) or row.cv_val_nll < mat_nll[ui, li]:
            mat_nll[ui, li] = row.cv_val_nll
            mat_r[ui, li]   = row.mean_abs_r

    im = ax.imshow(mat_r, aspect="auto", cmap="YlGn",
                   vmin=0.0, vmax=max(0.35, np.nanmax(mat_r)))
    ax.set_xticks(range(len(LMBD_VALS))); ax.set_xticklabels(LMBD_VALS, fontsize=8)
    ax.set_yticks(range(len(UW_VALS)));   ax.set_yticklabels(UW_VALS, fontsize=8)
    ax.set_xlabel("λ"); ax.set_ylabel("unif_weight")
    ax.set_title(f"z_dim={z}\n(outer-CV mean |r|, best h/eh per cell)")
    plt.colorbar(im, ax=ax, label="mean |r| ↑")
    for i in range(len(UW_VALS)):
        for j in range(len(LMBD_VALS)):
            if not np.isnan(mat_r[i, j]):
                ax.text(j, i, f"r={mat_r[i,j]:.2f}\nnll={mat_nll[i,j]:.3f}",
                        ha="center", va="center", fontsize=6.5, color="black")

plt.suptitle(f"Outer-CV decoding |r| by HP  |  vanilla NLL={VANILLA_NLL:.3f}", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "decoding_heatmap_by_z.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved decoding_heatmap_by_z.png")

print(f"\nDone. All plots in {PLOT_DIR}/")
