#!/usr/bin/env python3
"""
Per-fold LOO-CV ridge decoding + explainability for Thalmann IDRNN.

Key design decisions:
  - Regression run per fold (n≈79 each), z-scored within fold only.
  - r values aggregated across folds (mean ± std).
  - Correct last timestep: t=9 for task-0 blocks (0-29), t=199 for task-1 (block 30).
  - Explainability uses a single fold's model + test data only.

Outputs in plots_thalmann/:
  per_fold_decoding.png        — bar chart of mean |r| per scale, IDRNN vs Vanilla
  per_fold_decoding_table.csv  — full per-fold r values
  explainability_wslf.png      — win-stay/lose-shift for high vs low PHQ
  explainability_reward_sens.png
  explainability_reversal.png
  explainability_pca.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, AblatedRNN, Decoder, LatentRNN_secondstep

# ── Config ─────────────────────────────────────────────────────────────────────
DGP      = "thalmann"
COMBO    = "uw05_lmbd005_eh5_h5_z10"
EXPL_FOLD = 0          # fold used for explainability
N_FOLDS  = 3
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

# Correct last timestep per block type
T_TASK0_LAST = 9    # blocks 0-29: 10 real trials, padding from t=10
T_TASK1_LAST = 199  # block 30: 200 real trials
TASK1_BLK    = 30
A            = 4
COL_HIGH = "#C44E52"
COL_LOW  = "#4C72B0"

# ── Questionnaire ──────────────────────────────────────────────────────────────
quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
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
for k, (items, _) in SCALES.items():
    quest[k] = quest[items].mean(axis=1)

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def loo_ridge(Z, y):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        Z_tr, Z_te, y_tr = Z[tr], Z[te], y[tr]
        mu_z, sd_z = Z_tr.mean(0), Z_tr.std(0) + 1e-8
        mu_y, sd_y = y_tr.mean(), y_tr.std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z_tr - mu_z) / sd_z, (y_tr - mu_y) / sd_y)
        preds[te] = clf.predict((Z_te - mu_z) / sd_z) * sd_y + mu_y
    return pearsonr(y, preds)


def load_idrnn_z(fold, blk, t):
    """Load seed-averaged encoder z for test subjects of given fold at (blk, t)."""
    run_base = f"runs_{DGP}_hp_v2_{COMBO}/fold{fold}"
    data_dir  = f"data_{DGP}/fold{fold}"
    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)
    subids    = pd.read_csv(f"{data_dir}/df_test.csv")["subid"].values
    z_seeds   = []
    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc = cfg["model_config"]
        ckpt_path = os.path.join(run_base, sd, "checkpoints",
                                 f"epoch{cfg['cv_selected_epoch']:04d}.pt")
        if not os.path.exists(ckpt_path): continue
        state = torch.load(ckpt_path, map_location="cpu")
        enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                    n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
        enc.load_state_dict({k[8:]: v for k, v in state.items()
                             if k.startswith("encoder.")})
        enc.eval(); enc.set_task_ids(task_ids_global)
        with torch.no_grad():
            mu, _ = enc(xin_test)   # (B, 31, 200, z_dim)
        z_seeds.append(mu[:, blk, t, :].numpy())
    if not z_seeds: return None, None
    return np.stack(z_seeds).mean(0), subids   # (B_test, z_dim)


def load_vanilla_z(fold):
    """Mean GRU hidden across task-0 blocks for vanilla, last real timestep."""
    run_base = f"runs_vanilla_{DGP}/fold{fold}"
    data_dir  = f"data_{DGP}/fold{fold}"
    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"), dtype=torch.float32)
    subids    = pd.read_csv(f"{data_dir}/df_test.csv")["subid"].values
    z_seeds   = []
    for sd in sorted(d for d in os.listdir(run_base) if d.startswith("seed_")):
        cfg_p = os.path.join(run_base, sd, "config.json")
        if not os.path.exists(cfg_p): continue
        with open(cfg_p) as f: cfg = json.load(f)
        mc = cfg["model_config"]
        ckpt_path = os.path.join(run_base, sd, "checkpoints",
                                 f"epoch{cfg['cv_selected_epoch']:04d}.pt")
        if not os.path.exists(ckpt_path): continue
        van = AblatedRNN(hid=mc["hidden"], in_dim=mc.get("dec_in_dim", mc["in_dim"]),
                         A=mc["A"], block_structure=True,
                         n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0))
        van.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        van.eval()
        if hasattr(van, "set_task_ids"): van.set_task_ids(task_ids_global)
        with torch.no_grad():
            _, hid, _ = van(xin_test)   # hid: (Bk=31, B, hidden)
        # Average hidden state over task-0 blocks only (0-29)
        hid_task0 = hid[:30, :, :]     # (30, B, hidden)
        z_seeds.append(hid_task0.mean(0).numpy())   # (B, hidden)
    if not z_seeds: return None, None
    return np.stack(z_seeds).mean(0), subids

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Per-fold decoding
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 1: Per-fold LOO-CV ridge decoding")
print("=" * 60)

# Choose which (blk, t) to use — last real timestep of block 0 (task0)
BLK_USE, T_USE = 0, T_TASK0_LAST
print(f"Using block={BLK_USE}, t={T_USE} (last real trial of task-0 block 0)\n")

idrnn_rs = {k: [] for k in SCALE_KEYS}   # fold r values
van_rs    = {k: [] for k in SCALE_KEYS}

for fold in range(N_FOLDS):
    z_i, subids_i = load_idrnn_z(fold, BLK_USE, T_USE)
    z_v, subids_v = load_vanilla_z(fold)

    print(f"Fold {fold}  (n_test={len(subids_i)}):")
    print(f"  IDRNN z_std:   {z_i.std(0).round(3)}")
    print(f"  Vanilla z_std: {z_v.std(0).round(3)}")

    for scale in SCALE_KEYS:
        y_i = quest.reindex(subids_i)[scale].values.astype(float)
        y_v = quest.reindex(subids_v)[scale].values.astype(float)
        mask_i = ~np.isnan(y_i)
        mask_v = ~np.isnan(y_v)
        r_i, p_i = loo_ridge(z_i[mask_i], y_i[mask_i])
        r_v, p_v = loo_ridge(z_v[mask_v], y_v[mask_v])
        idrnn_rs[scale].append(r_i)
        van_rs[scale].append(r_v)
        print(f"  {scale:<12s}: IDRNN r={r_i:+.3f}{'*' if p_i<0.05 else ' '} "
              f"  Vanilla r={r_v:+.3f}{'*' if p_v<0.05 else ' '}")
    print()

# Aggregate
print("Aggregated across folds (mean ± std of r):")
rows = []
for scale in SCALE_KEYS:
    ri = np.array(idrnn_rs[scale])
    rv = np.array(van_rs[scale])
    print(f"  {scale:<12s}: IDRNN {ri.mean():+.3f} ± {ri.std():.3f}  "
          f"Vanilla {rv.mean():+.3f} ± {rv.std():.3f}")
    rows.append({"scale": scale,
                 **{f"idrnn_fold{f}": idrnn_rs[scale][f] for f in range(N_FOLDS)},
                 "idrnn_mean": ri.mean(), "idrnn_std": ri.std(),
                 **{f"van_fold{f}": van_rs[scale][f] for f in range(N_FOLDS)},
                 "van_mean": rv.mean(), "van_std": rv.std()})

df_results = pd.DataFrame(rows)
df_results.to_csv(os.path.join(PLOT_DIR, "per_fold_decoding_table.csv"), index=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(SCALE_KEYS))
w = 0.35
ri_means = [abs(df_results.loc[i, "idrnn_mean"]) for i in range(len(SCALE_KEYS))]
rv_means = [abs(df_results.loc[i, "van_mean"])   for i in range(len(SCALE_KEYS))]
ri_stds  = [df_results.loc[i, "idrnn_std"]       for i in range(len(SCALE_KEYS))]
rv_stds  = [df_results.loc[i, "van_std"]         for i in range(len(SCALE_KEYS))]

ax.bar(x - w/2, ri_means, w, yerr=ri_stds, color="#4477AA", alpha=0.85,
       edgecolor="white", capsize=4, label=f"IDRNN ({COMBO})")
ax.bar(x + w/2, rv_means, w, yerr=rv_stds, color="#BBBBBB", alpha=0.85,
       edgecolor="white", capsize=4, label="Vanilla")
ax.set_xticks(x); ax.set_xticklabels(SCALE_LABELS, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("|r|  LOO-CV ridge  (mean ± std across 3 folds)")
ax.set_title(f"Per-fold decoding: block0 last real trial (t={T_USE+1})\n"
             f"IDRNN {COMBO} vs Vanilla", fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim(0, 1)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "per_fold_decoding.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nSaved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Explainability on EXPL_FOLD
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Part 2: Explainability (fold {EXPL_FOLD})")
print("=" * 60)

# Load z for test subjects at last real timestep of block 0
z_expl, subids_expl = load_idrnn_z(EXPL_FOLD, BLK_USE, T_USE)
phq = quest.reindex(subids_expl)["PHQ"].values.astype(float)
valid = ~np.isnan(phq)
z_expl, subids_expl, phq = z_expl[valid], subids_expl[valid], phq[valid]

# Pick single participants with extreme PHQ scores
idx_lowest  = int(np.argmin(phq))
idx_highest = int(np.argmax(phq))
print(f"n={len(phq)}, PHQ range: [{phq.min():.3f}, {phq.max():.3f}]")
print(f"Lowest  PHQ: subj={subids_expl[idx_lowest]}  PHQ={phq[idx_lowest]:.3f}")
print(f"Highest PHQ: subj={subids_expl[idx_highest]} PHQ={phq[idx_highest]:.3f}")
print(f"\nz_std all: {z_expl.std(0).round(3)}")
print(f"z lowest  PHQ: {z_expl[idx_lowest].round(3)}")
print(f"z highest PHQ: {z_expl[idx_highest].round(3)}")
diff = z_expl[idx_highest] - z_expl[idx_lowest]
print(f"diff (high-low): {diff.round(3)}")
print(f"diff/std:        {(diff / (z_expl.std(0)+1e-8)).round(3)}")

z_low  = z_expl[idx_lowest]   # individual z of lowest-PHQ participant
z_high = z_expl[idx_highest]  # individual z of highest-PHQ participant

# Also keep group indices for PCA colouring
q25, q75 = np.percentile(phq, 25), np.percentile(phq, 75)
idx_low  = np.where(phq <= q25)[0]
idx_high = np.where(phq >= q75)[0]

# Load decoder model for EXPL_FOLD (use first seed)
run_base = f"runs_{DGP}_hp_v2_{COMBO}/fold{EXPL_FOLD}"
sd0 = sorted(d for d in os.listdir(run_base) if d.startswith("seed_"))[0]
cfg_path = os.path.join(run_base, sd0, "config.json")
with open(cfg_path) as f: cfg0 = json.load(f)
mc0 = cfg0["model_config"]
ckpt_path = os.path.join(run_base, sd0, "checkpoints",
                         f"epoch{cfg0['cv_selected_epoch']:04d}.pt")
print(f"\nLoading model from {ckpt_path}")

enc0 = IDRNN(in_dim=mc0["enc_in_dim"], z_dim=mc0["z_dim"], hid=mc0["enc_hidden"],
             n_tasks=mc0["n_tasks"], task_emb_dim=mc0["task_emb_dim"])
dec0 = Decoder(in_dim=mc0["dec_in_dim"], z_dim=mc0["z_dim"],
               hid=mc0["hidden"], A=mc0["A"])
full_model = LatentRNN_secondstep(
    encoder=enc0, hid=mc0["hidden"], z_dim=mc0["z_dim"],
    in_dim=mc0["dec_in_dim"], A=mc0["A"], decoder=dec0,
    n_tasks=mc0["n_tasks"], task_emb_dim=mc0["task_emb_dim"],
    reinit_decoder_per_block=mc0["reinit_decoder_per_block"],
)
state = torch.load(ckpt_path, map_location="cpu")
full_model.load_state_dict(state)
full_model.eval()
full_model.set_task_ids(task_ids_global)

decoder   = full_model.decoder
task_emb_w = full_model.task_embedding
with torch.no_grad():
    TEMB1 = task_emb_w(torch.tensor(1))   # task-1 embedding (4,)

Z_DIM = mc0["z_dim"]
print(f"Model: z_dim={Z_DIM}, hidden={mc0['hidden']}, dec_in_dim={mc0['dec_in_dim']}")

# ── Rollout helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout(z_vec, input_seq):
    """Run decoder step-by-step on task 1 with fixed z.
    z_vec: (z_dim,)  input_seq: (T, 5)
    Returns: probs (T,4), hiddens (T,hid)
    """
    z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)
    seq = torch.tensor(input_seq, dtype=torch.float32)
    T   = seq.size(0)
    temb_exp = TEMB1.unsqueeze(0).expand(T, -1)
    seq_in   = torch.cat([seq, temb_exp], dim=-1)   # (T, 9)
    h = decoder.z2h0(z_t).unsqueeze(0)
    probs_list, hid_list = [], []
    for t in range(T):
        step = seq_in[t].unsqueeze(0).unsqueeze(0)
        logits, h = decoder(step, z_t, hidden=h)
        probs_list.append(F.softmax(logits[0, 0], dim=-1).numpy())
        hid_list.append(h[0, 0].numpy())
    return np.stack(probs_list), np.stack(hid_list)

def make_input(arm, reward):
    oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
    return np.append(oh, float(reward))

def zero_input():
    return np.zeros(5, dtype=np.float32)

# ── Analysis 1: Win-Stay / Lose-Shift ─────────────────────────────────────────
print("\nAnalysis 1: Win-Stay / Lose-Shift")
def compute_wsls(z_vec):
    wsls = {}
    for arm in range(A):
        for rew, rl in [(1.0, "win"), (0.0, "lose")]:
            seq = np.stack([zero_input(), make_input(arm, rew), zero_input()])
            probs, _ = rollout(z_vec, seq)
            wsls[(arm, rl)] = probs[2, arm]
    return wsls

wsls_low  = compute_wsls(z_low)
wsls_high = compute_wsls(z_high)
ws_low  = np.mean([wsls_low[(a,"win")]        for a in range(A)])
ls_low  = np.mean([1-wsls_low[(a,"lose")]     for a in range(A)])
ws_high = np.mean([wsls_high[(a,"win")]       for a in range(A)])
ls_high = np.mean([1-wsls_high[(a,"lose")]    for a in range(A)])
print(f"  Lowest  PHQ ({phq[idx_lowest]:.2f}): Win-Stay={ws_low:.3f}, Lose-Shift={ls_low:.3f}")
print(f"  Highest PHQ ({phq[idx_highest]:.2f}): Win-Stay={ws_high:.3f}, Lose-Shift={ls_high:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax = axes[0]
cats = ["Win-Stay", "Lose-Shift"]
x = np.arange(2); w = 0.3
ax.bar(x-w/2, [ws_low,  ls_low],  w, color=COL_LOW,  alpha=0.85, edgecolor="white", label=f"Lowest PHQ ({phq[idx_lowest]:.2f})")
ax.bar(x+w/2, [ws_high, ls_high], w, color=COL_HIGH, alpha=0.85, edgecolor="white", label=f"Highest PHQ ({phq[idx_highest]:.2f})")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=12)
ax.set_ylabel("Probability"); ax.set_ylim(0,1)
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_title("Win-Stay / Lose-Shift\n(avg across arms)", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
x = np.arange(A)
ax.plot(x, [wsls_low[(a,"win")]  for a in range(A)], "o-", color=COL_LOW,  lw=2, label="Low PHQ | win")
ax.plot(x, [wsls_high[(a,"win")] for a in range(A)], "o-", color=COL_HIGH, lw=2, label="High PHQ | win")
ax.plot(x, [wsls_low[(a,"lose")] for a in range(A)], "s--",color=COL_LOW,  lw=2, alpha=0.7, label="Low PHQ | lose")
ax.plot(x, [wsls_high[(a,"lose")]for a in range(A)], "s--",color=COL_HIGH, lw=2, alpha=0.7, label="High PHQ | lose")
ax.set_xticks(x); ax.set_xticklabels([f"Arm {a}" for a in range(A)])
ax.set_ylabel("P(stay)"); ax.set_ylim(0,1); ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_title("P(stay) per arm and outcome", fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"WSLF — High vs Low PHQ (fold {EXPL_FOLD}, group mean z)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "explainability_wslf.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print(f"  Saved → {PLOT_DIR}/explainability_wslf.png")

# ── Analysis 2: Reward Sensitivity ────────────────────────────────────────────
print("Analysis 2: Reward Sensitivity")
reward_levels = np.linspace(0.0, 1.0, 11)
def rew_sens_all_arms(z_vec):
    out = []
    for rew in reward_levels:
        ps = [rollout(z_vec, np.stack([zero_input(), make_input(a,rew), zero_input()]))[0][2,a]
              for a in range(A)]
        out.append(np.mean(ps))
    return np.array(out)

sens_low  = rew_sens_all_arms(z_low)
sens_high = rew_sens_all_arms(z_high)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(reward_levels, sens_low,  "o-", color=COL_LOW,  lw=2, label=f"Lowest PHQ ({phq[idx_lowest]:.2f})")
ax.plot(reward_levels, sens_high, "o-", color=COL_HIGH, lw=2, label=f"Highest PHQ ({phq[idx_highest]:.2f})")
ax.set_xlabel("Previous reward"); ax.set_ylabel("P(stay on same arm)")
ax.set_ylim(0,1); ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_title(f"Reward Sensitivity (fold {EXPL_FOLD}, group mean z)", fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "explainability_reward_sens.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print(f"  Saved → {PLOT_DIR}/explainability_reward_sens.png")

# ── Analysis 3: Reversal Learning ─────────────────────────────────────────────
print("Analysis 3: Reversal Learning")
REVERSAL_T = 200; SWITCH_T = 100

def simulate_reversal(z_vec, seed=42):
    rng = np.random.default_rng(seed)
    z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)
    h   = decoder.z2h0(z_t).unsqueeze(0)
    p0_list, p1_list = [], []
    prev_in = zero_input()
    with torch.no_grad():
        for t in range(REVERSAL_T):
            step_in = torch.tensor(np.concatenate([prev_in, TEMB1.numpy()]),
                                   dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits, h = decoder(step_in, z_t, hidden=h)
            prob = F.softmax(logits[0,0], dim=-1).numpy()
            p0_list.append(prob[0]); p1_list.append(prob[1])
            chosen = int(np.argmax(prob))
            reward = 1.0 if chosen == (0 if t < SWITCH_T else 1) else 0.0
            prev_in = make_input(chosen, reward)
    return np.array(p0_list), np.array(p1_list)

p0_low, p1_low   = simulate_reversal(z_low)
p0_high, p1_high = simulate_reversal(z_high)

def smooth(a, w=10): return np.convolve(a, np.ones(w)/w, mode="same")
t = np.arange(REVERSAL_T)
p_correct_low  = np.where(t < SWITCH_T, p0_low,  p1_low)
p_correct_high = np.where(t < SWITCH_T, p0_high, p1_high)

fig, axes = plt.subplots(1, 2, figsize=(13,5))
ax = axes[0]
ax.plot(t, smooth(p0_low),  color=COL_LOW,  lw=2,    ls="-",  label="Low PHQ P(arm0)")
ax.plot(t, smooth(p0_high), color=COL_HIGH, lw=2,    ls="-",  label="High PHQ P(arm0)")
ax.plot(t, smooth(p1_low),  color=COL_LOW,  lw=2,    ls="--", alpha=0.7, label="Low PHQ P(arm1)")
ax.plot(t, smooth(p1_high), color=COL_HIGH, lw=2,    ls="--", alpha=0.7, label="High PHQ P(arm1)")
ax.axvline(SWITCH_T, color="k", ls=":", lw=1.5, alpha=0.7)
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial"); ax.set_ylabel("Probability"); ax.set_ylim(0,1); ax.set_xlim(0,REVERSAL_T)
ax.legend(fontsize=8, ncol=2); ax.set_title("P(arm 0 / arm 1)", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(t, smooth(p_correct_low),  color=COL_LOW,  lw=2.5, label=f"Lowest PHQ ({phq[idx_lowest]:.2f})")
ax.plot(t, smooth(p_correct_high), color=COL_HIGH, lw=2.5, label=f"Highest PHQ ({phq[idx_highest]:.2f})")
ax.axvline(SWITCH_T, color="k", ls=":", lw=1.5, alpha=0.7)
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial"); ax.set_ylabel("P(correct arm)"); ax.set_ylim(0,1); ax.set_xlim(0,REVERSAL_T)
ax.legend(fontsize=10); ax.set_title("P(correct arm) pre/post reversal", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"Reversal Learning — fold {EXPL_FOLD}, group mean z", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "explainability_reversal.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print(f"  Saved → {PLOT_DIR}/explainability_reversal.png")

# ── Analysis 4: GRU PCA on real task-1 sequences ──────────────────────────────
print("Analysis 4: GRU PCA")
xin_test_fold = np.load(f"data_{DGP}/fold{EXPL_FOLD}/xin_test.npy")  # (B,31,200,5)
XIN_T1 = xin_test_fold[valid, TASK1_BLK, :, :]   # (n, 200, 5)

all_hiddens, all_phq_ts, all_sidx = [], [], []
for i in range(len(z_expl)):
    _, h_i = rollout(z_expl[i], XIN_T1[i])
    all_hiddens.append(h_i)
    all_phq_ts.append(np.full(200, phq[i]))
    all_sidx.append(np.full(200, i))

H_mat  = np.concatenate(all_hiddens)
PHQ_ts = np.concatenate(all_phq_ts)
SIDX   = np.concatenate(all_sidx)

H_sc = StandardScaler().fit_transform(H_mat)
pca  = PCA(n_components=2)
H_pc = pca.fit_transform(H_sc)
ev   = pca.explained_variance_ratio_
print(f"  PCA var: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

subj_pc1 = np.array([H_pc[SIDX==i,0].mean() for i in range(len(z_expl))])
subj_pc2 = np.array([H_pc[SIDX==i,1].mean() for i in range(len(z_expl))])

fig, axes = plt.subplots(1,2, figsize=(13,5))
ax = axes[0]
sc = ax.scatter(subj_pc1, subj_pc2, c=phq, cmap="RdBu_r", s=40, alpha=0.75,
                edgecolors="k", linewidths=0.3)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85); cbar.set_label("PHQ-9", fontsize=10)
for i, col, lbl in [(idx_lowest,  COL_LOW,  f"Lowest PHQ ({phq[idx_lowest]:.2f})"),
                    (idx_highest, COL_HIGH, f"Highest PHQ ({phq[idx_highest]:.2f})")]:
    ax.scatter(subj_pc1[i], subj_pc2[i],
               marker="*", s=400, color=col, edgecolors="k", lw=0.8, zorder=5, label=lbl)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title("Per-subject mean GRU hidden state", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
def grp_traj(grp_idx):
    return np.stack([H_pc[SIDX==i,0] for i in grp_idx])

traj_l = grp_traj(idx_low);  m_l, se_l = traj_l.mean(0), traj_l.std(0)/np.sqrt(len(idx_low))
traj_h = grp_traj(idx_high); m_h, se_h = traj_h.mean(0), traj_h.std(0)/np.sqrt(len(idx_high))
t = np.arange(200)
ax.plot(t, smooth(m_l, 15), color=COL_LOW,  lw=2.5, label=f"Lowest PHQ ({phq[idx_lowest]:.2f})")
ax.plot(t, smooth(m_h, 15), color=COL_HIGH, lw=2.5, label=f"Highest PHQ ({phq[idx_highest]:.2f})")
ax.fill_between(t, smooth(m_l-se_l,15), smooth(m_l+se_l,15), color=COL_LOW,  alpha=0.15)
ax.fill_between(t, smooth(m_h-se_h,15), smooth(m_h+se_h,15), color=COL_HIGH, alpha=0.15)
ax.set_xlabel("Trial"); ax.set_ylabel("PC1 GRU hidden state")
ax.set_title("PC1 trajectory — High vs Low PHQ", fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim(0,200)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"GRU Hidden-State PCA — Task 1, fold {EXPL_FOLD}", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "explainability_pca.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print(f"  Saved → {PLOT_DIR}/explainability_pca.png")

print("\nDone.")
