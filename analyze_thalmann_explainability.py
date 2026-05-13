#!/usr/bin/env python3
"""
Thalmann explainability analysis for IDRNN (unif_weight=0, best PHQ decoding run).

Steps:
  1. Load outer-CV z from runs_thalmann_unif0p0 (all folds × seeds).
  2. Match PHQ-9 Depression scores; split into top / bottom 25%.
  3. Load decoder from fold0/seed_200 checkpoint.
  4. Four counterfactual / PCA analyses on task 1 (restless 4-armed bandit, 200 trials).

Outputs in plots_thalmann/:
  explainability_wslf.png          – win-stay / lose-shift per arm
  explainability_reward_sens.png   – P(stay) vs reward level
  explainability_reversal.png      – reversal-learning trajectory
  explainability_pca.png           – GRU hidden-state PCA on real sequences
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep

# ── Config ─────────────────────────────────────────────────────────────────────
DGP        = "thalmann"
RUN_SUFFIX = "unif0p0"
N_FOLDS    = 3
TASK1_BLK  = 30         # block index for the restless bandit (task 1)
TASK1_ID   = 1          # task id for restless bandit
T_TASK1    = 200        # trials in the restless bandit
A          = 4          # actions
PLOT_DIR   = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

# Palette: high PHQ = warm red, low PHQ = cool blue
COL_HIGH = "#C44E52"
COL_LOW  = "#4C72B0"

# ── Questionnaire ──────────────────────────────────────────────────────────────
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"
PHQ_ITEMS  = [f"PHQ_9_{i}" for i in range(10)]
quest      = pd.read_csv(QUEST_PATH).set_index("ID")
quest["PHQ"] = quest[PHQ_ITEMS].mean(axis=1)

# ── Task-ID tensor (31 blocks: 30 task0 + 1 task1) ────────────────────────────
task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load outer-CV z across all folds × seeds
# ══════════════════════════════════════════════════════════════════════════════
def load_z_fold(fold):
    """Return (z_test, subids_test) for one fold, averaged over seeds."""
    base      = f"runs_{DGP}_{RUN_SUFFIX}/fold{fold}"
    data_dir  = f"data_{DGP}/fold{fold}"
    xin_test  = torch.tensor(np.load(f"{data_dir}/xin_test.npy"),  dtype=torch.float32)
    subids    = pd.read_csv(f"{data_dir}/df_test.csv")["subid"].values

    seed_dirs = sorted(d for d in os.listdir(base)
                       if d.startswith("seed_") and os.path.isdir(os.path.join(base, d)))
    z_seeds = []
    for sd in seed_dirs:
        run_dir  = os.path.join(base, sd)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        mc = cfg["model_config"]

        ckpt_epoch = cfg["cv_selected_epoch"]
        ckpt_path  = os.path.join(run_dir, "checkpoints", f"epoch{ckpt_epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]

        state = torch.load(ckpt_path, map_location="cpu")
        enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                    n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"])
        enc_state = {k[len("encoder."):]: v for k, v in state.items()
                     if k.startswith("encoder.")}
        enc.load_state_dict(enc_state)
        enc.eval()
        enc.set_task_ids(task_ids_global)

        with torch.no_grad():
            mu, _ = enc(xin_test)        # (B, Bk, T, z) per-timestep form
        if mu.dim() == 4:
            mu = mu[:, -1, -1, :]        # last block, last timestep → (B, z_dim)
        z_seeds.append(mu.numpy())

    if not z_seeds:
        return None, None
    z = np.stack(z_seeds, 0).mean(0)    # (B_test, z_dim)
    return z, subids


print("Loading outer-CV z…")
z_all, ids_all = [], []
xin_all = []   # task-1 sequences for PCA (block 30)

for fold in range(N_FOLDS):
    z, subids = load_z_fold(fold)
    if z is None:
        print(f"  fold {fold}: no data, skipping")
        continue
    z_all.append(z)
    ids_all.append(subids)
    xin_fold = np.load(f"data_{DGP}/fold{fold}/xin_test.npy")  # (B,31,200,5)
    xin_all.append(xin_fold[:, TASK1_BLK, :, :])               # (B,200,5)

Z       = np.concatenate(z_all,   axis=0)   # (N, z_dim)
IDS     = np.concatenate(ids_all, axis=0)   # (N,)
XIN_T1  = np.concatenate(xin_all, axis=0)  # (N, 200, 5)

# Match PHQ
phq = quest.reindex(IDS)["PHQ"].values.astype(float)
valid = ~np.isnan(phq)
Z, IDS, XIN_T1, phq = Z[valid], IDS[valid], XIN_T1[valid], phq[valid]
print(f"  Total participants with PHQ: {len(phq)}")

# Top / bottom 25%
q25 = np.percentile(phq, 25)
q75 = np.percentile(phq, 75)
idx_low  = np.where(phq <= q25)[0]
idx_high = np.where(phq >= q75)[0]
print(f"  Low  PHQ ≤{q25:.2f}: n={len(idx_low)},  mean={phq[idx_low].mean():.2f}")
print(f"  High PHQ ≥{q75:.2f}: n={len(idx_high)}, mean={phq[idx_high].mean():.2f}")

z_low  = Z[idx_low].mean(0)    # mean z of bottom-quartile group (z_dim,)
z_high = Z[idx_high].mean(0)   # mean z of top-quartile group

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load decoder model (fold0 / seed_200)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading decoder…")
CKPT_RUN = f"runs_{DGP}_{RUN_SUFFIX}/fold0/seed_200"
with open(os.path.join(CKPT_RUN, "config.json")) as f:
    cfg0 = json.load(f)
mc0 = cfg0["model_config"]
ckpt_ep  = cfg0["cv_selected_epoch"]
ckpt_path = os.path.join(CKPT_RUN, "checkpoints", f"epoch{ckpt_ep:04d}.pt")

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

decoder    = full_model.decoder      # Decoder module
task_emb_w = full_model.task_embedding  # nn.Embedding(2, 4)

Z_DIM = mc0["z_dim"]
HID   = mc0["hidden"]

# Task-1 embedding vector (fixed)
with torch.no_grad():
    TEMB1 = task_emb_w(torch.tensor(TASK1_ID))  # (4,)

print(f"  z_dim={Z_DIM}, hidden={HID}, dec_in_dim={mc0['dec_in_dim']}")

# ══════════════════════════════════════════════════════════════════════════════
# Rollout helper
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def rollout(z_vec, input_seq):
    """
    Run decoder step-by-step with a FIXED z for task 1.

    z_vec   : (z_dim,) numpy array — individual or group-mean latent
    input_seq: (T, 5) numpy array — raw xin for task 1 (base_in_dim=5)

    Returns:
      probs   : (T, A) numpy — softmax probabilities per timestep
      hiddens : (T, hid) numpy — GRU hidden state per timestep
    """
    z_t  = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)   # (1, z_dim)
    seq  = torch.tensor(input_seq, dtype=torch.float32)             # (T, 5)
    T    = seq.size(0)

    # Append task embedding to each step → (T, 9)
    temb_exp = TEMB1.unsqueeze(0).expand(T, -1)                    # (T, 4)
    seq_with_temb = torch.cat([seq, temb_exp], dim=-1)             # (T, 9)

    # Initialize hidden state from z2h0(z) — same as reinit_decoder_per_block
    h = decoder.z2h0(z_t).unsqueeze(0)   # (1, 1, hid)

    probs_list, hid_list = [], []
    for t in range(T):
        step = seq_with_temb[t].unsqueeze(0).unsqueeze(0)   # (1, 1, 9)
        logits, h = decoder(step, z_t, hidden=h)             # logits (1,1,4)
        prob = F.softmax(logits[0, 0], dim=-1).numpy()       # (4,)
        probs_list.append(prob)
        hid_list.append(h[0, 0].numpy())                     # (hid,)

    return np.stack(probs_list), np.stack(hid_list)


def make_input(chosen_arm, reward):
    """Build a single xin timestep: [one_hot(A=4), reward_normed]."""
    oh = np.zeros(A, dtype=np.float32)
    oh[chosen_arm] = 1.0
    return np.append(oh, float(reward))   # (5,)


def zero_input():
    """Neutral xin: no prev choice, no reward (start of block)."""
    return np.zeros(5, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1 — Win-Stay / Lose-Shift
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 1: Win-Stay / Lose-Shift")

def compute_wsls(z_vec):
    """
    For each arm × reward condition, feed one step and read P(same arm).
    Returns dict: {(arm, reward_label) → p_stay}
    """
    wsls = {}
    for arm in range(A):
        for rew, rlabel in [(1.0, "win"), (0.0, "lose")]:
            # Feed: [zero step, arm-chose-rew step]
            # Step 0: neutral (start of block, h0 from z2h0)
            # Step 1: chose `arm`, got `rew`
            # Step 2: P(arm) — "stay"
            seq = np.stack([zero_input(), make_input(arm, rew), zero_input()])
            probs, _ = rollout(z_vec, seq)
            p_stay = probs[2, arm]    # after seeing prev choice=arm, prev rew
            wsls[(arm, rlabel)] = p_stay
    return wsls

wsls_low  = compute_wsls(z_low)
wsls_high = compute_wsls(z_high)

# Average win-stay and lose-shift across arms
win_stay_low   = np.mean([wsls_low[(a, "win")]  for a in range(A)])
lose_shift_low = np.mean([1 - wsls_low[(a, "lose")] for a in range(A)])
win_stay_high  = np.mean([wsls_high[(a, "win")]  for a in range(A)])
lose_shift_high= np.mean([1 - wsls_high[(a, "lose")] for a in range(A)])

print(f"  Low  PHQ — Win-Stay={win_stay_low:.3f}, Lose-Shift={lose_shift_low:.3f}")
print(f"  High PHQ — Win-Stay={win_stay_high:.3f}, Lose-Shift={lose_shift_high:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

# Left: Win-Stay / Lose-Shift summary bars
ax = axes[0]
cats   = ["Win-Stay", "Lose-Shift"]
vals_l = [win_stay_low, lose_shift_low]
vals_h = [win_stay_high, lose_shift_high]
x = np.arange(len(cats))
w = 0.3
ax.bar(x - w/2, vals_l, w, color=COL_LOW,  alpha=0.85, edgecolor="white", label="Low PHQ")
ax.bar(x + w/2, vals_h, w, color=COL_HIGH, alpha=0.85, edgecolor="white", label="High PHQ")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=12)
ax.set_ylabel("Probability", fontsize=11)
ax.set_title("Win-Stay / Lose-Shift\n(averaged across arms)", fontweight="bold")
ax.set_ylim(0, 1); ax.legend(fontsize=10)
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5, label="chance")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Right: Per-arm P(stay | win) and P(stay | lose)
ax = axes[1]
arm_labels = [f"Arm {a}" for a in range(A)]
win_stay_per_arm_low   = [wsls_low[(a, "win")]  for a in range(A)]
win_stay_per_arm_high  = [wsls_high[(a, "win")] for a in range(A)]
lose_stay_per_arm_low  = [wsls_low[(a, "lose")] for a in range(A)]
lose_stay_per_arm_high = [wsls_high[(a, "lose")]for a in range(A)]
x = np.arange(A)
ax.plot(x, win_stay_per_arm_low,  "o-", color=COL_LOW,  lw=2, label="Low PHQ | win")
ax.plot(x, win_stay_per_arm_high, "o-", color=COL_HIGH, lw=2, label="High PHQ | win")
ax.plot(x, lose_stay_per_arm_low,  "s--", color=COL_LOW,  lw=2, alpha=0.7, label="Low PHQ | lose")
ax.plot(x, lose_stay_per_arm_high, "s--", color=COL_HIGH, lw=2, alpha=0.7, label="High PHQ | lose")
ax.set_xticks(x); ax.set_xticklabels(arm_labels)
ax.set_ylabel("P(stay = same arm)"); ax.set_ylim(0, 1)
ax.set_title("P(stay) per arm and outcome", fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("High vs Low PHQ: Win-Stay / Lose-Shift  (task 1 restless bandit, group mean z)",
             fontweight="bold", fontsize=11)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "explainability_wslf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2 — Reward Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 2: Reward Sensitivity")

reward_levels = np.linspace(0.0, 1.0, 11)

def reward_sensitivity(z_vec):
    """P(stay arm 0) after receiving various reward levels on arm 0."""
    p_stay = []
    for rew in reward_levels:
        seq = np.stack([zero_input(), make_input(0, rew), zero_input()])
        probs, _ = rollout(z_vec, seq)
        p_stay.append(probs[2, 0])
    return np.array(p_stay)

sens_low  = reward_sensitivity(z_low)
sens_high = reward_sensitivity(z_high)

# Also compute across all arms (average)
def reward_sensitivity_all_arms(z_vec):
    p_stay = []
    for rew in reward_levels:
        ps = []
        for arm in range(A):
            seq = np.stack([zero_input(), make_input(arm, rew), zero_input()])
            probs, _ = rollout(z_vec, seq)
            ps.append(probs[2, arm])
        p_stay.append(np.mean(ps))
    return np.array(p_stay)

sens_low_avg  = reward_sensitivity_all_arms(z_low)
sens_high_avg = reward_sensitivity_all_arms(z_high)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(reward_levels, sens_low,  "o-", color=COL_LOW,  lw=2, label="Low PHQ")
ax.plot(reward_levels, sens_high, "o-", color=COL_HIGH, lw=2, label="High PHQ")
ax.set_xlabel("Previous reward (arm 0)", fontsize=11)
ax.set_ylabel("P(choose arm 0 next)", fontsize=11)
ax.set_title("Reward Sensitivity — Arm 0", fontweight="bold")
ax.set_ylim(0, 1); ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.legend(fontsize=11); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(reward_levels, sens_low_avg,  "o-", color=COL_LOW,  lw=2, label="Low PHQ")
ax.plot(reward_levels, sens_high_avg, "o-", color=COL_HIGH, lw=2, label="High PHQ")
ax.set_xlabel("Previous reward", fontsize=11)
ax.set_ylabel("P(stay on same arm)", fontsize=11)
ax.set_title("Reward Sensitivity — Avg across arms", fontweight="bold")
ax.set_ylim(0, 1); ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.legend(fontsize=11); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Reward Sensitivity: High vs Low PHQ (task 1, group mean z)",
             fontweight="bold", fontsize=11)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "explainability_reward_sens.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3 — Reversal Learning (closed-loop simulation)
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 3: Reversal Learning")

REVERSAL_T = 200
SWITCH_T   = 100   # arm 0 rewarded for t<SWITCH_T, arm 1 for t>=SWITCH_T

def simulate_reversal(z_vec, seed=42):
    """
    Closed-loop: at each step, model chooses arm = argmax(prob),
    gets reward=1 if correct arm, 0 otherwise.
    Returns:
      prob_arm0  (T,) — P(arm 0) at each step
      prob_arm1  (T,) — P(arm 1) at each step
      choices    (T,) — chosen arm
    """
    rng = np.random.default_rng(seed)
    z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)   # (1, z_dim)
    h   = decoder.z2h0(z_t).unsqueeze(0)                          # (1,1,hid)

    prob0_list, prob1_list, choice_list = [], [], []

    prev_input = zero_input()   # neutral first input
    with torch.no_grad():
        for t in range(REVERSAL_T):
            # Append task embedding
            step_in = np.concatenate([prev_input, TEMB1.numpy()])  # (9,)
            step_t  = torch.tensor(step_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits, h = decoder(step_t, z_t, hidden=h)
            prob = F.softmax(logits[0, 0], dim=-1).numpy()         # (4,)

            prob0_list.append(prob[0])
            prob1_list.append(prob[1])

            # Choose arm (greedy)
            chosen = int(np.argmax(prob))
            choice_list.append(chosen)

            # Determine reward
            correct_arm = 0 if t < SWITCH_T else 1
            reward = 1.0 if chosen == correct_arm else 0.0

            # Build next input
            prev_input = make_input(chosen, reward)

    return np.array(prob0_list), np.array(prob1_list), np.array(choice_list)


p0_low,  p1_low,  ch_low  = simulate_reversal(z_low)
p0_high, p1_high, ch_high = simulate_reversal(z_high)

# Smooth with rolling mean
def smooth(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode="same")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: P(arm 0) and P(arm 1) trajectories
ax = axes[0]
t = np.arange(REVERSAL_T)
ax.plot(t, smooth(p0_low),  color=COL_LOW,  lw=2,   ls="-",  label="Low PHQ — P(arm 0)")
ax.plot(t, smooth(p0_high), color=COL_HIGH, lw=2,   ls="-",  label="High PHQ — P(arm 0)")
ax.plot(t, smooth(p1_low),  color=COL_LOW,  lw=2,   ls="--", label="Low PHQ — P(arm 1)", alpha=0.7)
ax.plot(t, smooth(p1_high), color=COL_HIGH, lw=2,   ls="--", label="High PHQ — P(arm 1)", alpha=0.7)
ax.axvline(SWITCH_T, color="k", ls=":", lw=1.5, alpha=0.7, label=f"Reversal (t={SWITCH_T})")
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial", fontsize=11); ax.set_ylabel("Probability", fontsize=11)
ax.set_ylim(0, 1); ax.set_xlim(0, REVERSAL_T)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_title("Reversal Learning — P(arm 0 / arm 1)", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: P(correct arm) — tracks adaptation to reversal
ax = axes[1]
p_correct_low  = np.where(t < SWITCH_T, p0_low,  p1_low)
p_correct_high = np.where(t < SWITCH_T, p0_high, p1_high)
ax.plot(t, smooth(p_correct_low),  color=COL_LOW,  lw=2.5, label="Low PHQ")
ax.plot(t, smooth(p_correct_high), color=COL_HIGH, lw=2.5, label="High PHQ")
ax.axvline(SWITCH_T, color="k", ls=":", lw=1.5, alpha=0.7)
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial", fontsize=11); ax.set_ylabel("P(correct arm)", fontsize=11)
ax.set_ylim(0, 1); ax.set_xlim(0, REVERSAL_T)
ax.legend(fontsize=10)
ax.set_title("P(correct arm) — pre/post reversal", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Reversal Learning Simulation: High vs Low PHQ (task 1, group mean z)",
             fontweight="bold", fontsize=11)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "explainability_reversal.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4 — GRU Hidden-State PCA on Real Sequences
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 4: GRU PCA on real task-1 sequences")

# Run each test participant with their own z through the decoder (task 1 block)
all_hiddens   = []   # (N * T, hid)
all_phq_vals  = []   # (N * T,) — repeated PHQ per timestep
all_subj_ids  = []   # (N * T,)

for i in range(len(Z)):
    z_i   = Z[i]
    xin_i = XIN_T1[i]          # (200, 5)
    _, h_i = rollout(z_i, xin_i)   # h_i: (200, hid)
    all_hiddens.append(h_i)
    all_phq_vals.append(np.full(T_TASK1, phq[i]))
    all_subj_ids.append(np.full(T_TASK1, i))

H_mat  = np.concatenate(all_hiddens,   axis=0)   # (N*T, hid)
PHQ_ts = np.concatenate(all_phq_vals,  axis=0)   # (N*T,)
SIDX   = np.concatenate(all_subj_ids,  axis=0)   # (N*T,)

# PCA on hidden states
scaler = StandardScaler()
H_sc   = scaler.fit_transform(H_mat)
pca    = PCA(n_components=2)
H_pc   = pca.fit_transform(H_sc)   # (N*T, 2)
ev     = pca.explained_variance_ratio_

print(f"  PCA var explained: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

# Per-subject mean PC trajectory (color by PHQ)
subj_pc1 = np.array([H_pc[SIDX == i, 0].mean() for i in range(len(Z))])
subj_pc2 = np.array([H_pc[SIDX == i, 1].mean() for i in range(len(Z))])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: scatter of per-subject mean hidden state colored by PHQ
ax = axes[0]
sc = ax.scatter(subj_pc1, subj_pc2, c=phq, cmap="RdBu_r", s=40, alpha=0.75,
                edgecolors="k", linewidths=0.3)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("PHQ-9 score", fontsize=10)
# Mark group means
for grp_idx, col, label in [(idx_low, COL_LOW, "Low PHQ"), (idx_high, COL_HIGH, "High PHQ")]:
    valid_grp = np.isin(np.arange(len(Z)), grp_idx)
    ax.scatter(subj_pc1[valid_grp].mean(), subj_pc2[valid_grp].mean(),
               marker="*", s=300, color=col, edgecolors="k", lw=0.8, zorder=5, label=label)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
ax.set_title("Per-subject mean GRU hidden state\n(task 1, colored by PHQ-9)", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: per-trial mean PC1 for high vs low PHQ groups
ax = axes[1]
def group_pc1_trajectory(grp_indices):
    """Mean PC1 across group members for each trial."""
    traj = []
    for i in grp_indices:
        traj.append(H_pc[SIDX == i, 0])
    return np.stack(traj, 0)   # (n_grp, T)

traj_low  = group_pc1_trajectory(idx_low)
traj_high = group_pc1_trajectory(idx_high)
t = np.arange(T_TASK1)

m_low, se_low   = traj_low.mean(0),  traj_low.std(0)  / np.sqrt(len(idx_low))
m_high, se_high = traj_high.mean(0), traj_high.std(0) / np.sqrt(len(idx_high))

ax.plot(t, smooth(m_low,  w=15), color=COL_LOW,  lw=2.5, label="Low PHQ")
ax.plot(t, smooth(m_high, w=15), color=COL_HIGH, lw=2.5, label="High PHQ")
ax.fill_between(t, smooth(m_low-se_low,  w=15), smooth(m_low+se_low,  w=15),
                color=COL_LOW,  alpha=0.15)
ax.fill_between(t, smooth(m_high-se_high,w=15), smooth(m_high+se_high,w=15),
                color=COL_HIGH, alpha=0.15)
ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel(f"PC1 of GRU hidden state", fontsize=11)
ax.set_title("PC1 trajectory across trials — High vs Low PHQ", fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, T_TASK1)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("GRU Hidden-State Dynamics — Task 1 Real Sequences (outer-CV z)",
             fontweight="bold", fontsize=11)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "explainability_pca.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out}")

print("\nDone. All plots saved to", PLOT_DIR)
