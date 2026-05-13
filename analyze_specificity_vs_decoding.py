#!/usr/bin/env python3
"""
Compute reconstruction specificity for every hp_v2 combo and correlate it
with questionnaire-score decodability (from summary_table_per_fold.csv).

Specificity = mean_subject(mismatched_NLL - matched_NLL)
  matched   : decode subject i's sequence using their own encoder z
  mismatched: decode subject i's sequence using a random other subject's z
              (averaged over N_MISMATCH permutations)

Higher specificity → z carries individual-specific reconstruction-relevant info.
Question: does specificity predict decodability of questionnaire scores?

Outputs: plots_thalmann/hp_v2/
  specificity_table.csv
  specificity_vs_decoding.png
  specificity_vs_nll.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder

# ── Config ─────────────────────────────────────────────────────────────────────
RUNS_BASE   = "runs_thalmann"
PLOT_DIR    = "plots_thalmann/hp_v2"
DGP         = "thalmann"
N_FOLDS     = 3
BLK_USE     = 0
T_USE       = 9      # last real trial of task-0 block 0
N_MISMATCH  = 15     # number of random permutations for mismatched NLL

os.makedirs(PLOT_DIR, exist_ok=True)

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

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


# ── Core computation ───────────────────────────────────────────────────────────

def reconstruction_nll(dec, task_emb_weight, xin_test, c_test_np, z_use):
    """
    Compute mean per-subject NLL using decoder with z_use.

    dec             : Decoder (already eval'd)
    task_emb_weight : (n_tasks, task_emb_dim) tensor
    xin_test        : (B, 31, 200, 5)  decoder base input
    c_test_np       : (B, 31, 200) int array, -100 = padding
    z_use           : (B, z_dim) tensor

    Returns: np.array (B,) per-subject mean NLL over valid timesteps
    """
    B, n_blocks, T, _ = xin_test.shape
    all_log_probs = []
    with torch.no_grad():
        for b in range(n_blocks):
            tid  = task_ids_global[b].long()
            temb = task_emb_weight[tid]  # (task_emb_dim,)
            temb_exp = temb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            inp_b   = torch.cat([xin_test[:, b], temb_exp], dim=-1)  # (B, T, dec_in_dim)
            logits_b, _ = dec(inp_b, z_use, hidden=None)              # reinit from z each block
            all_log_probs.append(F.log_softmax(logits_b, dim=-1))     # (B, T, A)

    log_p  = torch.stack(all_log_probs, dim=1)           # (B, 31, T, A)
    c      = torch.tensor(c_test_np, dtype=torch.long)   # (B, 31, T)
    mask   = (c >= 0)
    c_cl   = c.clamp(min=0)
    chosen = log_p.gather(-1, c_cl.unsqueeze(-1)).squeeze(-1)  # (B, 31, T)
    chosen = chosen * mask.float()
    n_valid = mask.float().sum(dim=[1, 2]).clamp(min=1)
    nll_per_sub = -chosen.sum(dim=[1, 2]) / n_valid      # (B,)
    return nll_per_sub.numpy()


def compute_specificity_for_fold(combo, fold):
    """
    Returns dict with specificity, matched_nll for a combo × fold.
    Uses the first seed that has a valid checkpoint.
    """
    run_base = f"{RUNS_BASE}_hp_v2_{combo}/fold{fold}"
    data_dir = f"data_{DGP}/fold{fold}"
    if not os.path.isdir(run_base):
        return None

    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    c_test_np = np.load(f"{data_dir}/c_test.npy")
    B = xin_test.shape[0]

    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        mc      = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None:
            continue
        ckpt = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt):
            continue

        try:
            state = torch.load(ckpt, map_location="cpu")

            # ── encoder ──────────────────────────────────────────────────────
            enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                        hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                        task_emb_dim=mc["task_emb_dim"])
            enc.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
            enc.eval()
            enc.set_task_ids(task_ids_global)

            # ── decoder ──────────────────────────────────────────────────────
            dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                          hid=mc["hidden"], A=mc["A"])
            dec.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("decoder.")})
            dec.eval()

            # ── task embedding ───────────────────────────────────────────────
            if "task_embedding.weight" not in state:
                continue
            task_emb_w = state["task_embedding.weight"]  # (n_tasks, task_emb_dim)

            # ── encoder z ────────────────────────────────────────────────────
            with torch.no_grad():
                mu, _ = enc(xin_test)              # (B, 31, 200, z_dim)
            z = mu[:, BLK_USE, T_USE, :]           # (B, z_dim)

            # ── matched NLL ──────────────────────────────────────────────────
            matched = reconstruction_nll(dec, task_emb_w, xin_test, c_test_np, z)

            # ── mismatched NLL ───────────────────────────────────────────────
            mismatch_nlls = []
            rng = np.random.RandomState(0)
            for _ in range(N_MISMATCH):
                perm = rng.permutation(B)
                # Avoid self-pairing
                while (perm == np.arange(B)).any():
                    perm = rng.permutation(B)
                z_perm = z[torch.from_numpy(perm)]
                mismatch_nlls.append(
                    reconstruction_nll(dec, task_emb_w, xin_test, c_test_np, z_perm)
                )
            mismatched = np.stack(mismatch_nlls).mean(0)

            specificity = float((mismatched - matched).mean())
            return {
                "specificity":  specificity,
                "matched_nll":  float(matched.mean()),
                "mismatch_nll": float(mismatched.mean()),
            }

        except Exception as e:
            print(f"  Warning ({combo} fold{fold} {sd}): {e}")
            continue

    return None


def parse_combo(combo):
    parts = combo.split("_")
    try:
        uw   = {"00": 0.0, "01": 0.1, "05": 0.5}.get(parts[0][2:], float("nan"))
        lmbd = {"005": 0.05, "01": 0.1, "02": 0.2}.get(parts[1][4:], float("nan"))
        eh   = int(parts[2][2:]); h = int(parts[3][1:]); z = int(parts[4][1:])
        return uw, lmbd, eh, h, z
    except Exception:
        return None, None, None, None, None


def process_combo(combo):
    uw, lmbd, eh, h, z_dim = parse_combo(combo)
    if uw is None:
        return None

    fold_specs = []
    fold_matched = []
    for fold in range(N_FOLDS):
        res = compute_specificity_for_fold(combo, fold)
        if res is not None:
            fold_specs.append(res["specificity"])
            fold_matched.append(res["matched_nll"])

    if not fold_specs:
        return None

    return {
        "combo":       combo,
        "uw":          uw,
        "lmbd":        lmbd,
        "enc_hidden":  eh,
        "hidden":      h,
        "z_dim":       z_dim,
        "specificity": float(np.mean(fold_specs)),
        "specificity_std": float(np.std(fold_specs)),
        "matched_nll": float(np.mean(fold_matched)),
        "n_folds":     len(fold_specs),
    }


# ── Discover combos ─────────────────────────────────────────────────────────────
all_combo_dirs = sorted(
    d for d in os.listdir(".")
    if d.startswith(f"{RUNS_BASE}_hp_v2_uw") and os.path.isdir(d)
)
combos = sorted({d[len(f"{RUNS_BASE}_hp_v2_"):] for d in all_combo_dirs})
print(f"Found {len(combos)} combos")

# ── Compute specificity in parallel ─────────────────────────────────────────────
spec_results = Parallel(n_jobs=8, verbose=5)(
    delayed(process_combo)(c) for c in combos
)
spec_results = [r for r in spec_results if r is not None]
df_spec = pd.DataFrame(spec_results)
print(f"\nComputed specificity for {len(df_spec)} combos.")
print(df_spec[["combo", "specificity", "specificity_std", "matched_nll", "n_folds"]].head(10).to_string())

# ── Merge with decoding results ─────────────────────────────────────────────────
dec_csv = os.path.join(PLOT_DIR, "summary_table_per_fold.csv")
df_dec  = pd.read_csv(dec_csv)
df = pd.merge(df_spec, df_dec[
    ["combo", "cv_val_nll", "mean_abs_r"] +
    [f"r_{k}" for k in SCALE_KEYS] +
    [f"r_std_{k}" for k in SCALE_KEYS]
], on="combo", how="inner")

df.to_csv(os.path.join(PLOT_DIR, "specificity_table.csv"), index=False)
print(f"\nMerged table: {len(df)} rows. Saved specificity_table.csv")

# Correlations between specificity and each scale
print("\nCorrelations: specificity vs |r_scale|")
for k in SCALE_KEYS + ["mean_abs_r"]:
    col = f"r_{k}" if k in SCALE_KEYS else k
    vals = df[col].abs() if k in SCALE_KEYS else df[col]
    r, p = pearsonr(df["specificity"], vals)
    print(f"  {k:15s}  r={r:+.3f}  p={p:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Specificity vs decodability (one panel per scale + mean_abs_r)
# ══════════════════════════════════════════════════════════════════════════════
n_scales = len(SCALE_KEYS)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for j, (sk, label) in enumerate(zip(SCALE_KEYS, SCALE_LABELS)):
    ax = axes[j]
    r_vals = df[f"r_{sk}"].abs()
    sc = ax.scatter(df["specificity"], r_vals,
                    c=df["z_dim"], cmap="viridis", s=40, alpha=0.7,
                    edgecolors="grey", linewidths=0.3)
    r_corr, p_corr = pearsonr(df["specificity"], r_vals)
    ax.set_xlabel("Specificity (Δ NLL mismatch - match)", fontsize=8)
    ax.set_ylabel(f"|r| per-fold LOO-CV", fontsize=8)
    ax.set_title(f"{label}\nr={r_corr:+.2f}  p={p_corr:.3f}", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Extra panel: mean_abs_r
ax = axes[n_scales]
sc = ax.scatter(df["specificity"], df["mean_abs_r"],
                c=df["z_dim"], cmap="viridis", s=40, alpha=0.7,
                edgecolors="grey", linewidths=0.3)
r_corr, p_corr = pearsonr(df["specificity"], df["mean_abs_r"])
ax.set_xlabel("Specificity (Δ NLL)", fontsize=8)
ax.set_ylabel("|r| mean across scales", fontsize=8)
ax.set_title(f"Mean |r| across all scales\nr={r_corr:+.2f}  p={p_corr:.3f}",
             fontsize=9, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.colorbar(sc, ax=axes[n_scales], label="z_dim")

# Hide unused panels
for ax in axes[n_scales+1:]:
    ax.set_visible(False)

fig.suptitle("Reconstruction specificity vs questionnaire-score decodability\n"
             "(each dot = one HP combo; colour = z_dim)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "specificity_vs_decoding.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved specificity_vs_decoding.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Specificity vs NLL (to check whether the two metrics agree)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
for zdim, grp in df.groupby("z_dim"):
    ax.scatter(grp["cv_val_nll"], grp["specificity"],
               label=f"z={zdim}", s=40, alpha=0.7)
r_corr, p_corr = pearsonr(df["cv_val_nll"], df["specificity"])
ax.set_xlabel("cv_val_nll (lower = better fit)")
ax.set_ylabel("Specificity (Δ NLL mismatch - match)")
ax.set_title(f"NLL vs Specificity across HP combos\n"
             f"r={r_corr:+.2f}  p={p_corr:.3f}", fontweight="bold")
ax.legend(fontsize=9, title="z_dim")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "specificity_vs_nll.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved specificity_vs_nll.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: NLL + Specificity joint ranking — Pareto front
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["cv_val_nll"], df["specificity"],
                c=df["mean_abs_r"], cmap="RdYlGn", s=60,
                vmin=df["mean_abs_r"].quantile(0.1),
                vmax=df["mean_abs_r"].quantile(0.9),
                edgecolors="k", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="mean |r| across scales (decodability)")

# Annotate top-5 by mean_abs_r
top5 = df.nlargest(5, "mean_abs_r")
for _, row in top5.iterrows():
    ax.annotate(row["combo"].replace("_", "\n"),
                (row["cv_val_nll"], row["specificity"]),
                fontsize=6, alpha=0.85,
                xytext=(4, 4), textcoords="offset points")

ax.set_xlabel("cv_val_nll  (← better fit)")
ax.set_ylabel("Specificity  (→ more individual-specific z)")
ax.set_title("Joint view: NLL × Specificity × Decodability\n"
             "(colour = mean |r|; ideal combo: low NLL, high specificity, green)",
             fontweight="bold")
ax.invert_xaxis()   # lower NLL = better → point left → rightward = better
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "pareto_nll_specificity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved pareto_nll_specificity.png")

print("\nDone. Outputs in", PLOT_DIR)
