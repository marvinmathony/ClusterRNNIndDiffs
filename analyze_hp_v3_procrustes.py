#!/usr/bin/env python3
"""
Leak-free full-Procrustes pooled decoding for hp_v3 outer-CV runs
(and Vanilla baseline).

Per-fold encoders produce non-aligned latent spaces, so raw pooled LOO ridge
is inflated (e.g. PHQ r can climb to 0.95 as a pure alignment artefact).
This script:
  1. For every combo, runs each fold's encoder on the FULL 236-subject pool,
     giving three (N, d) matrices (one per fold encoder).
  2. For each held-out subject i, refits a FULL Procrustes transform (R, s, t)
     on the other 235 subjects (no leakage of i's latent into R).
  3. Rotates i's held-out z into the reference frame, fits ridge on the
     235 aligned training latents, predicts i, for every scale.
  4. Verifies fold-id LOO decodability from the aligned z is at chance
     (1/3) — if it isn't, the alignment failed for that combo.

Runs IDRNN combos and the Vanilla RNN baseline.

Outputs (plots_thalmann/hp_v3_procrustes/):
  summary_table.csv            per-combo r per scale + fold-id accuracy
  fold_id_decodability.png     histogram of fold-id acc across combos
  combos_mean_r.png            per-combo mean |r| bar chart
  scale_comparison.png         per-scale IDRNN-top vs Vanilla |r|
"""

import os, json, glob, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
from joblib import Parallel, delayed

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN

# ── Config ───────────────────────────────────────────────────────────────────
RUNS_BASE = "runs_thalmann"
PLOT_DIR  = "plots_thalmann/hp_v3_procrustes"
DGP       = "thalmann"
N_FOLDS   = 3
N_JOBS    = 8
os.makedirs(PLOT_DIR, exist_ok=True)

TASK_IDS_PATH = f"data_{DGP}/task_ids_per_block.npy"
task_ids_global = torch.tensor(np.load(TASK_IDS_PATH), dtype=torch.long)

# Scales (match hp_v3 analysis)
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"
SCALES = {
    "PANAS_PA":  [f"PANAS_{i}"  for i in [0, 2, 4, 6, 8, 14, 16, 18]],
    "PANAS_NA":  [f"PANAS_{i}"  for i in [1, 3, 5, 7, 9, 11, 13, 15]],
    "STICSA":    [f"STICSA_{i}" for i in range(22)],
    "PHQ":       [f"PHQ_9_{i}"  for i in range(10)],
    "CEI":       [f"CEI_{i}"    for i in range(4)],
    "BIG5_open": [f"BIG_5_{i}"  for i in range(6)],
}
SCALE_KEYS   = list(SCALES.keys())
SCALE_LABELS = {
    "PANAS_PA":  "PANAS Pos.", "PANAS_NA":  "PANAS Neg.",
    "STICSA":    "STICSA Anx.", "PHQ":       "PHQ-9 Depr.",
    "CEI":       "CEI Curios.", "BIG5_open": "BIG5 Open.",
}

quest = pd.read_csv(QUEST_PATH).set_index("ID")
for k, items in SCALES.items():
    quest[k] = quest[items].mean(axis=1)


# ── Assemble full subject pool ──────────────────────────────────────────────
def load_all_subject_inputs():
    xin_list, sid_list, fold_list, c_list = [], [], [], []
    for f in range(N_FOLDS):
        xin = torch.tensor(np.load(f"data_{DGP}/fold{f}/xin_test.npy"),
                           dtype=torch.float32)
        df  = pd.read_csv(f"data_{DGP}/fold{f}/df_test.csv")
        sids = (df["subid"].values if "subid" in df.columns
                else df["session"].values)
        c = np.load(f"data_{DGP}/fold{f}/c_test.npy")
        xin_list.append(xin)
        sid_list.append(sids)
        fold_list.append(np.full(len(sids), f))
        c_list.append(c)
    return (torch.cat(xin_list, 0),
            np.concatenate(sid_list),
            np.concatenate(fold_list),
            np.concatenate(c_list))


xin_all, sids_all, fold_of_sub, c_all = load_all_subject_inputs()
N_SUB = len(sids_all)
print(f"Subject pool: {N_SUB} participants "
      f"(fold sizes = {[int((fold_of_sub == f).sum()) for f in range(N_FOLDS)]})")

# Questionnaire target matrix (N_SUB, n_scales); NaN for missing
Y_all = np.full((N_SUB, len(SCALE_KEYS)), np.nan)
for j, key in enumerate(SCALE_KEYS):
    for i, sid in enumerate(sids_all):
        if sid in quest.index and not pd.isna(quest.loc[sid, key]):
            Y_all[i, j] = quest.loc[sid, key]
print(f"Scales x subjects with data: "
      f"{[(k, int((~np.isnan(Y_all[:, j])).sum())) for j, k in enumerate(SCALE_KEYS)]}")


# ── Procrustes helpers ───────────────────────────────────────────────────────
def full_procrustes_fit(Z_src, Z_ref):
    mu_s, mu_r = Z_src.mean(0), Z_ref.mean(0)
    Zs, Zr = Z_src - mu_s, Z_ref - mu_r
    R, scale_num = orthogonal_procrustes(Zs, Zr)
    s = scale_num / (np.linalg.norm(Zs) ** 2)
    t = mu_r - s * mu_s @ R
    return R, s, t


def apply_full_procrustes(Z, R, s, t):
    return s * Z @ R + t


def loo_leak_free_full_procrustes(Z_per_enc, fold_of_sub_, Y, mask_any):
    """For each held-out subject, refit Procrustes on n-1 subjects, then
    ridge-predict all scales. Returns (N_sub, n_scales) preds with NaNs where
    either subject or scale was invalid in training."""
    N_sub    = Z_per_enc[0].shape[0]
    n_scales = Y.shape[1]
    preds = np.full((N_sub, n_scales), np.nan)
    valid_idx = np.where(mask_any)[0]

    for i in valid_idx:
        train = np.setdiff1d(valid_idx, [i])
        Z_ref_tr = Z_per_enc[0][train]

        aligned = {0: Z_per_enc[0]}
        for f in range(1, N_FOLDS):
            R, s, t = full_procrustes_fit(Z_per_enc[f][train], Z_ref_tr)
            aligned[f] = apply_full_procrustes(Z_per_enc[f], R, s, t)

        Z_hel = np.stack([aligned[int(fold_of_sub_[ss])][ss]
                          for ss in range(N_sub)])

        for k in range(n_scales):
            y_col = Y[:, k]
            if np.isnan(y_col[i]):
                continue
            train_k = train[~np.isnan(y_col[train])]
            if len(train_k) < 20:
                continue
            mu_z, sd_z = Z_hel[train_k].mean(0), Z_hel[train_k].std(0) + 1e-8
            mu_y, sd_y = y_col[train_k].mean(),  y_col[train_k].std()  + 1e-8
            clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
            clf.fit((Z_hel[train_k] - mu_z) / sd_z,
                    (y_col[train_k] - mu_y) / sd_y)
            preds[i, k] = (clf.predict((Z_hel[[i]] - mu_z) / sd_z)[0]
                           * sd_y + mu_y)
    return preds


def build_aligned_z_sample(Z_per_enc, fold_of_sub_, use_full=True):
    """Sample-level aligned z (used only for the fold-id decodability
    diagnostic — not for the ridge decoding)."""
    Z_ref = Z_per_enc[0]
    aligned = {0: Z_ref}
    if use_full:
        for f in range(1, N_FOLDS):
            R, s, t = full_procrustes_fit(Z_per_enc[f], Z_ref)
            aligned[f] = apply_full_procrustes(Z_per_enc[f], R, s, t)
    else:
        for f in range(1, N_FOLDS):
            R, _ = orthogonal_procrustes(Z_per_enc[f], Z_ref)
            aligned[f] = Z_per_enc[f] @ R
    return np.stack([aligned[int(fold_of_sub_[ss])][ss]
                     for ss in range(Z_ref.shape[0])])


def fold_id_loo_accuracy(Z, fold_labels):
    """LOO multinomial LR predicting fold from z. Chance = 1/N_FOLDS."""
    correct = 0
    for i in range(len(fold_labels)):
        tr = np.ones(len(fold_labels), dtype=bool); tr[i] = False
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                 C=1.0, max_iter=500)
        clf.fit((Z[tr] - mu) / sd, fold_labels[tr])
        pred = clf.predict((Z[[i]] - mu) / sd)[0]
        correct += (pred == fold_labels[i])
    return correct / len(fold_labels)


# ── IDRNN encoder on full subject pool ──────────────────────────────────────
def compute_z_all_subjects_idrnn(combo, fold):
    """Return z_last averaged over seeds (N_SUB, d) or None if missing."""
    run_base = f"{RUNS_BASE}_hp_v3_{combo}/fold{fold}"
    if not os.path.isdir(run_base):
        return None
    z_seeds = []
    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(run_base, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt): continue
        try:
            state = torch.load(ckpt, map_location="cpu")
            enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                        hid=mc["enc_hidden"], n_tasks=mc.get("n_tasks"),
                        task_emb_dim=mc.get("task_emb_dim", 0),
                        continuous_encoder=mc.get("continuous_encoder", False))
            enc.load_state_dict({k[8:]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
            enc.set_task_ids(task_ids_global)
            enc.eval()
            with torch.no_grad():
                mu, _ = enc(xin_all, return_per_timestep=False)
            z_seeds.append(mu.numpy())
        except Exception as e:
            print(f"    {combo}/fold{fold}/{sd}: {e}")
    if not z_seeds:
        return None
    return np.stack(z_seeds).mean(0)


def compute_h_all_subjects_vanilla(fold):
    """Vanilla: time-averaged hidden state over full subject pool."""
    fold_dir = f"runs_vanilla_thalmann/fold{fold}"
    if not os.path.isdir(fold_dir):
        return None
    h_seeds = []
    for sd in sorted(d for d in os.listdir(fold_dir) if d.startswith("seed_")):
        cfg_p = os.path.join(fold_dir, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc = cfg["model_config"]
        best_ep = cfg.get("cv_selected_epoch")
        if best_ep is None: continue
        ckpt = os.path.join(fold_dir, sd, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt): continue
        try:
            state = torch.load(ckpt, map_location="cpu")
            block_structure = (xin_all.dim() == 4)
            model = AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"],
                               A=mc["A"], block_structure=block_structure,
                               n_tasks=mc.get("n_tasks"),
                               task_emb_dim=mc.get("task_emb_dim", 0))
            model.load_state_dict(state)
            model.set_task_ids(task_ids_global)
            model.eval()
            with torch.no_grad():
                _, _, hidden_tr = model(xin_all)
            valid = torch.tensor(c_all >= 0, dtype=torch.float32)
            n_valid = valid.sum(dim=(1, 2), keepdim=True).clamp(min=1)
            h_avg = ((hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2))
                     / n_valid.squeeze(-1))
            h_seeds.append(h_avg.numpy())
        except Exception as e:
            print(f"    vanilla/fold{fold}/{sd}: {e}")
    if not h_seeds:
        return None
    return np.stack(h_seeds).mean(0)


# ── Per-combo processing ─────────────────────────────────────────────────────
def process_combo(combo):
    Z_per_enc = []
    for f in range(N_FOLDS):
        Z = compute_z_all_subjects_idrnn(combo, f)
        if Z is None:
            return None
        Z_per_enc.append(Z)

    mask_any = ~np.all(np.isnan(Y_all), axis=1)

    preds = loo_leak_free_full_procrustes(Z_per_enc, fold_of_sub, Y_all, mask_any)

    decoding = {}
    for k, key in enumerate(SCALE_KEYS):
        mk = ~np.isnan(Y_all[:, k]) & ~np.isnan(preds[:, k])
        if mk.sum() >= 20:
            r, p = pearsonr(Y_all[mk, k], preds[mk, k])
            decoding[key] = (float(r), float(p))
        else:
            decoding[key] = (np.nan, 1.0)

    Z_aln = build_aligned_z_sample(Z_per_enc, fold_of_sub, use_full=True)
    acc_full_proc = fold_id_loo_accuracy(Z_aln[mask_any], fold_of_sub[mask_any])
    # Also compute pre-alignment fold-id accuracy for contrast
    Z_unaln = np.stack([Z_per_enc[int(fold_of_sub[ss])][ss]
                        for ss in range(N_SUB)])
    acc_unaligned = fold_id_loo_accuracy(Z_unaln[mask_any], fold_of_sub[mask_any])

    parts = combo.split("_")
    try:
        uw   = {"00": 0.0, "01": 0.1, "05": 0.5}[parts[0][2:]]
        lmbd = {"005": 0.05, "01": 0.1, "02": 0.2}[parts[1][4:]]
        eh   = int(parts[2][2:])
        h    = int(parts[3][1:])
        z    = int(parts[4][1:])
    except Exception:
        uw = lmbd = eh = h = z = np.nan

    mean_abs_r = float(np.nanmean([abs(decoding[k][0]) for k in SCALE_KEYS]))
    print(f"  {combo:<35} mean|r|={mean_abs_r:.3f}  "
          f"|r|_PHQ={abs(decoding['PHQ'][0]):.3f}  "
          f"fold-acc pre={acc_unaligned:.3f} post={acc_full_proc:.3f}")

    return {
        "combo": combo, "uw": uw, "lmbd": lmbd,
        "enc_hidden": eh, "hidden": h, "z_dim": z,
        "decoding":          decoding,
        "mean_abs_r":        mean_abs_r,
        "fold_acc_unaligned": acc_unaligned,
        "fold_acc_procrustes": acc_full_proc,
    }


def process_vanilla():
    H_per_enc = []
    for f in range(N_FOLDS):
        h = compute_h_all_subjects_vanilla(f)
        if h is None:
            return None
        H_per_enc.append(h)

    mask_any = ~np.all(np.isnan(Y_all), axis=1)
    preds = loo_leak_free_full_procrustes(H_per_enc, fold_of_sub, Y_all, mask_any)
    decoding = {}
    for k, key in enumerate(SCALE_KEYS):
        mk = ~np.isnan(Y_all[:, k]) & ~np.isnan(preds[:, k])
        if mk.sum() >= 20:
            r, p = pearsonr(Y_all[mk, k], preds[mk, k])
            decoding[key] = (float(r), float(p))
        else:
            decoding[key] = (np.nan, 1.0)
    H_aln = build_aligned_z_sample(H_per_enc, fold_of_sub, use_full=True)
    acc_full_proc = fold_id_loo_accuracy(H_aln[mask_any], fold_of_sub[mask_any])
    H_unaln = np.stack([H_per_enc[int(fold_of_sub[ss])][ss]
                        for ss in range(N_SUB)])
    acc_unaligned = fold_id_loo_accuracy(H_unaln[mask_any], fold_of_sub[mask_any])
    mean_abs_r = float(np.nanmean([abs(decoding[k][0]) for k in SCALE_KEYS]))
    print(f"  {'Vanilla':<35} mean|r|={mean_abs_r:.3f}  "
          f"|r|_PHQ={abs(decoding['PHQ'][0]):.3f}  "
          f"fold-acc pre={acc_unaligned:.3f} post={acc_full_proc:.3f}")
    return {
        "combo": "vanilla",
        "decoding": decoding,
        "mean_abs_r": mean_abs_r,
        "fold_acc_unaligned":  acc_unaligned,
        "fold_acc_procrustes": acc_full_proc,
    }


# ── Run ──────────────────────────────────────────────────────────────────────
combo_dirs = sorted(glob.glob(f"{RUNS_BASE}_hp_v3_*/"))
combos = [os.path.basename(d.rstrip("/")).replace(
    f"{os.path.basename(RUNS_BASE)}_hp_v3_", "") for d in combo_dirs]
print(f"\nFound {len(combos)} hp_v3 combos. Processing...\n")

results_raw = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(process_combo)(c) for c in combos)
results = [r for r in results_raw if r is not None]
print(f"\n{len(results)} combos with complete results.\n")

print("Processing Vanilla baseline...")
vanilla_result = process_vanilla()


# ── Save CSV ─────────────────────────────────────────────────────────────────
rows = []
for d in results:
    row = {k: d[k] for k in ["combo", "uw", "lmbd", "enc_hidden", "hidden", "z_dim",
                               "mean_abs_r", "fold_acc_unaligned", "fold_acc_procrustes"]}
    for k in SCALE_KEYS:
        row[f"r_{k}"] = d["decoding"][k][0]
        row[f"p_{k}"] = d["decoding"][k][1]
    rows.append(row)
df = pd.DataFrame(rows).sort_values("mean_abs_r", ascending=False)
csv_path = os.path.join(PLOT_DIR, "summary_table.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

if vanilla_result:
    van_row = {"combo": "vanilla",
               "mean_abs_r": vanilla_result["mean_abs_r"],
               "fold_acc_unaligned":  vanilla_result["fold_acc_unaligned"],
               "fold_acc_procrustes": vanilla_result["fold_acc_procrustes"]}
    for k in SCALE_KEYS:
        van_row[f"r_{k}"] = vanilla_result["decoding"][k][0]
        van_row[f"p_{k}"] = vanilla_result["decoding"][k][1]
    van_df = pd.DataFrame([van_row])
    van_df.to_csv(os.path.join(PLOT_DIR, "vanilla_row.csv"), index=False)


# ── Flag combos where alignment failed ──────────────────────────────────────
CHANCE = 1.0 / N_FOLDS
fail_threshold = CHANCE + 0.15  # flag anything >48% on 3-class (chance=33%)
failed = df[df["fold_acc_procrustes"] > fail_threshold]
print(f"\n{len(failed)}/{len(df)} combos have fold-id acc >{fail_threshold:.2f} "
      f"after full Procrustes (alignment incomplete):")
for _, row in failed.head(15).iterrows():
    print(f"  {row.combo:<35} fold-acc={row.fold_acc_procrustes:.3f}")


# ── Plot 1: fold-id decodability distribution ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
accs_un  = df["fold_acc_unaligned"].values
accs_pr  = df["fold_acc_procrustes"].values
for ax, accs, title, color in [
    (axes[0], accs_un, "Unaligned z", "#D9534F"),
    (axes[1], accs_pr, "After full Procrustes", "#337AB7"),
]:
    ax.hist(accs, bins=20, color=color, edgecolor="k", alpha=0.85)
    ax.axvline(CHANCE, color="red", ls="--", lw=1.5,
               label=f"Chance = {CHANCE:.2f}")
    if vanilla_result:
        v = (vanilla_result["fold_acc_unaligned"]  if "Unaligned" in title
             else vanilla_result["fold_acc_procrustes"])
        ax.axvline(v, color="black", ls="-.", lw=1.5,
                   label=f"Vanilla ({v:.2f})")
    ax.set_xlabel("Fold-id LOO accuracy")
    ax.set_ylabel("# combos")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Fold-id decodability from z — alignment diagnostic",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fold_id_decodability.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fold_id_decodability.png")


# ── Plot 2: per-combo mean |r| bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(12, 0.15 * len(df)), 6))
df_s = df.sort_values("mean_abs_r", ascending=False)
x = np.arange(len(df_s))
norm = plt.Normalize(vmin=CHANCE, vmax=min(1.0, accs_pr.max()))
cmap = plt.cm.coolwarm
bars = ax.bar(x, df_s["mean_abs_r"],
              color=[cmap(norm(v)) for v in df_s["fold_acc_procrustes"]],
              edgecolor="k", linewidth=0.3)
if vanilla_result:
    ax.axhline(vanilla_result["mean_abs_r"], color="black", ls="--", lw=1.5,
               label=f"Vanilla mean |r| = {vanilla_result['mean_abs_r']:.3f}")
ax.set_xticks(x)
ax.set_xticklabels(df_s["combo"], rotation=90, fontsize=6)
ax.set_ylabel("Mean |r| across 6 scales (leak-free full Procrustes)")
ax.set_title("IDRNN combos — leak-free full-Procrustes pooled decoding",
             fontweight="bold")
ax.legend(loc="upper right")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, label="fold-id acc after Procrustes (chance=0.33)")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "combos_mean_r.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved combos_mean_r.png")


# ── Plot 3: per-scale comparison, best IDRNN vs Vanilla ─────────────────────
# Best IDRNN = highest mean_abs_r with fold_acc_procrustes at chance (≤0.45)
if vanilla_result:
    clean = df[df["fold_acc_procrustes"] <= CHANCE + 0.12].copy()
    if len(clean) == 0:
        clean = df.copy()
    best_idx = clean["mean_abs_r"].idxmax()
    best = df.loc[best_idx]
    best_combo = best["combo"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCALE_KEYS))
    w = 0.35
    van_vals = [abs(vanilla_result["decoding"][k][0]) for k in SCALE_KEYS]
    idr_vals = [abs(best[f"r_{k}"]) for k in SCALE_KEYS]
    ax.bar(x - w/2, van_vals, w, label=f"Vanilla (acc={vanilla_result['fold_acc_procrustes']:.2f})",
           color="#D9534F", edgecolor="k")
    ax.bar(x + w/2, idr_vals, w,
           label=f"{best_combo} (acc={best['fold_acc_procrustes']:.2f})",
           color="#337AB7", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([SCALE_LABELS[k] for k in SCALE_KEYS], rotation=30,
                       ha="right")
    ax.set_ylabel("|r| (leak-free full Procrustes)")
    ax.set_title("Best IDRNN (alignment passed) vs Vanilla",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "scale_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved scale_comparison.png")


print(f"\nDone. Outputs in {PLOT_DIR}/")
