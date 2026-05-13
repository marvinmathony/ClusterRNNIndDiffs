#!/usr/bin/env python3
"""
Use step-1 lookup embeddings (not IDRNN encoder z) for the two extreme-PHQ
participants, then run the frozen decoder on a shared actual trial sequence.

Step-1 embeddings are directly optimized by the decoder (no KL, no collapse).
Both participants see the SAME input sequence (subj 8's actual trials)
so any difference in outputs is purely due to z.

Fold 1 is used because subj 8 (lowest PHQ) and subj 21 (highest PHQ)
are both in its training set → both have step-1 embeddings.

Outputs in plots_thalmann/:
  step1_logits_task1.png         — per-arm P over 200 task-1 trials
  step1_logits_task1_summary.png — P(chosen arm) + entropy
  step1_logits_task0.png         — P(arm 0/1) over 30 blocks × 10 trials
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from modelsandtraining import Decoder

# ── Config ─────────────────────────────────────────────────────────────────────
DGP      = "thalmann"
COMBO    = "uw05_lmbd005_eh5_h5_z10"
FOLD     = 1   # subj 8 and 21 are training participants in this fold
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

T_TASK0_LAST = 9
TASK1_BLK    = 30
A = 4

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

# ── Questionnaire ──────────────────────────────────────────────────────────────
quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
quest["PHQ"] = quest[[f"PHQ_9_{i}" for i in range(10)]].mean(1)

# ── Load step-1 frozen decoder checkpoint ─────────────────────────────────────
# Use first seed (they all share the same frozen_decoder after step-1 training)
sd0 = sorted(d for d in os.listdir(f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}")
             if d.startswith("seed_"))[0]
frozen_path = f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}/{sd0}/frozen_decoder/policy_model.pt"
print(f"Loading step-1 model: {frozen_path}")
state = torch.load(frozen_path, map_location="cpu")

# Model config from the regular config.json
with open(f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}/{sd0}/config.json") as f:
    cfg = json.load(f)
mc = cfg["model_config"]
print(f"z_dim={mc['z_dim']}, hidden={mc['hidden']}, dec_in_dim={mc['dec_in_dim']}, task_emb_dim={mc['task_emb_dim']}")

# Build decoder and task embedding
decoder  = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                   hid=mc["hidden"], A=mc["A"])
task_emb_weight = state["task_embedding.weight"]   # (2, task_emb_dim)
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.eval()

# ── Step-1 lookup embeddings ──────────────────────────────────────────────────
emb = state["encoder.embed.weight"].numpy()         # (n_train, z_dim)

# Subject ordering: same as df_train row order
train_df = pd.read_csv(f"data_{DGP}/fold{FOLD}/df_train.csv")
subids_train = train_df["subid"].values              # already in the right order
phq_train    = quest.reindex(subids_train)["PHQ"].values.astype(float)

# Find extreme-PHQ subjects
idx_low  = int(np.argmin(phq_train))
idx_high = int(np.argmax(phq_train))
z_low  = emb[idx_low]
z_high = emb[idx_high]

print(f"\nn_train={len(subids_train)}, PHQ [{phq_train.min():.2f}, {phq_train.max():.2f}]")
print(f"Lowest  PHQ: subj={subids_train[idx_low]},  PHQ={phq_train[idx_low]:.3f}")
print(f"Highest PHQ: subj={subids_train[idx_high]}, PHQ={phq_train[idx_high]:.3f}")
print(f"\nStep-1 embedding std (population): {emb.std(0).round(3)}")
print(f"z_low  (PHQ={phq_train[idx_low]:.2f}): {z_low.round(3)}")
print(f"z_high (PHQ={phq_train[idx_high]:.2f}): {z_high.round(3)}")
print(f"diff/std: {((z_high-z_low)/(emb.std(0)+1e-8)).round(3)}")

COL_LOW  = "#4C72B0"
COL_HIGH = "#C44E52"

# ── Shared sequence: subj 8 (lowest PHQ)'s actual trials ──────────────────────
xin_train = np.load(f"data_{DGP}/fold{FOLD}/xin_train.npy")   # (157, 31, 200, 5)
shared_xin = xin_train[idx_low]                                 # (31, 200, 5)

# ── Decoder rollout ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_decoder(z_vec, seq_block, task_id):
    """
    z_vec:     (z_dim,) numpy
    seq_block: (T, 5) numpy — shared input sequence
    task_id:   int
    Returns probs (T, A)
    """
    T    = seq_block.shape[0]
    z_t  = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)   # (1, z_dim)
    temb = task_emb_weight[task_id]                                  # (task_emb_dim,)
    seq  = torch.tensor(seq_block, dtype=torch.float32)             # (T, 5)
    seq_in = torch.cat([seq, temb.unsqueeze(0).expand(T, -1)], -1) # (T, 5+emb)

    h = decoder.z2h0(z_t).unsqueeze(0)
    probs_list = []
    for t in range(T):
        logits, h = decoder(seq_in[t].unsqueeze(0).unsqueeze(0), z_t, hidden=h)
        probs_list.append(F.softmax(logits[0, 0], dim=-1).numpy())
    return np.stack(probs_list)   # (T, A)

def actual_choices(seq):
    return np.argmax(seq[:, :4], axis=1)

def smooth(a, w=15):
    return np.convolve(a, np.ones(w)/w, mode="same")

# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — restless bandit (block 30, 200 trials)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Task 1 ---")
task1_id = int(task_ids_global[TASK1_BLK].item())
seq_t1   = shared_xin[TASK1_BLK]   # (200, 5)
ch_t1    = actual_choices(seq_t1)

probs_low_t1  = run_decoder(z_low,  seq_t1, task1_id)
probs_high_t1 = run_decoder(z_high, seq_t1, task1_id)

t1 = np.arange(200)
arm_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

# Per-arm probability over time
fig, axes = plt.subplots(A, 1, figsize=(14, 10), sharex=True)
for a in range(A):
    ax = axes[a]
    ax.plot(t1, smooth(probs_low_t1[:,  a], 10), color=COL_LOW,  lw=2,
            label=f"z low  PHQ={phq_train[idx_low]:.2f}")
    ax.plot(t1, smooth(probs_high_t1[:, a], 10), color=COL_HIGH, lw=2,
            label=f"z high PHQ={phq_train[idx_high]:.2f}")
    chose_a = (ch_t1 == a)
    ax.scatter(t1[chose_a], np.full(chose_a.sum(), 0.02),
               marker="|", color=arm_colors[a], s=40, alpha=0.5,
               zorder=3, label="actual choice (shared seq)")
    ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5)
    ax.set_ylim(0, 1); ax.set_ylabel(f"P(arm {a})", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if a == 0:
        ax.legend(fontsize=9, loc="upper right")
axes[-1].set_xlabel("Trial")
fig.suptitle(f"Task 1 (restless) — step-1 embeddings, same sequence\n"
             f"IDRNN {COMBO} fold {FOLD} | "
             f"Shared seq = subj {subids_train[idx_low]} (lowest PHQ={phq_train[idx_low]:.2f})",
             fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_logits_task1.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# Summary: P(chosen arm) + entropy
p_chosen_low  = probs_low_t1[t1, ch_t1]
p_chosen_high = probs_high_t1[t1, ch_t1]
ent_low  = -np.sum(probs_low_t1  * np.log(probs_low_t1  + 1e-9), axis=1)
ent_high = -np.sum(probs_high_t1 * np.log(probs_high_t1 + 1e-9), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(t1, smooth(p_chosen_low),  color=COL_LOW,  lw=2.5,
        label=f"z low  PHQ={phq_train[idx_low]:.2f}  (mean={p_chosen_low.mean():.3f})")
ax.plot(t1, smooth(p_chosen_high), color=COL_HIGH, lw=2.5,
        label=f"z high PHQ={phq_train[idx_high]:.2f}  (mean={p_chosen_high.mean():.3f})")
ax.axhline(0.25, color="grey", ls=":", lw=1, alpha=0.5, label="Chance")
ax.set_xlabel("Trial"); ax.set_ylabel("P(shared sequence's actual choice)")
ax.set_ylim(0, 1); ax.set_xlim(0, 199)
ax.set_title("Model confidence in the shared sequence's choice\n(step-1 embeddings, same trials)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(t1, smooth(ent_low),  color=COL_LOW,  lw=2.5,
        label=f"z low  PHQ={phq_train[idx_low]:.2f}  (mean={ent_low.mean():.3f})")
ax.plot(t1, smooth(ent_high), color=COL_HIGH, lw=2.5,
        label=f"z high PHQ={phq_train[idx_high]:.2f}  (mean={ent_high.mean():.3f})")
ax.axhline(np.log(A), color="grey", ls=":", lw=1, alpha=0.5, label="Max entropy")
ax.set_xlabel("Trial"); ax.set_ylabel("Entropy of P(arm)")
ax.set_ylim(0, np.log(A) * 1.1); ax.set_xlim(0, 199)
ax.set_title("Decoder uncertainty\n(lower = more decisive)", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"Task 1 summary — step-1 embeddings, {COMBO} fold {FOLD}", fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_logits_task1_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Task 0 — 2-armed bandit (blocks 0-29, 10 trials each)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Task 0 ---")
task0_id = int(task_ids_global[0].item())

p0_low_all, p0_high_all, ch0_all = [], [], []
for blk in range(30):
    seq_blk = shared_xin[blk, :T_TASK0_LAST+1, :]
    pl = run_decoder(z_low,  seq_blk, task0_id)
    ph = run_decoder(z_high, seq_blk, task0_id)
    p0_low_all.append(pl[:, 0])
    p0_high_all.append(ph[:, 0])
    ch0_all.append(actual_choices(seq_blk))

p0_low_all  = np.concatenate(p0_low_all)
p0_high_all = np.concatenate(p0_high_all)
ch0_all     = np.concatenate(ch0_all)
t0 = np.arange(300)

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
for ax, (p0, col, lbl) in zip(axes, [
    (p0_low_all,  COL_LOW,  f"z low  PHQ={phq_train[idx_low]:.2f}"),
    (p0_high_all, COL_HIGH, f"z high PHQ={phq_train[idx_high]:.2f}"),
]):
    ax.plot(t0, smooth(p0, 3),   color="#2196F3", lw=2, label="P(arm 0)")
    ax.plot(t0, smooth(1-p0, 3), color="#4CAF50", lw=2, alpha=0.8, label="P(arm 1)")
    ax.scatter(t0[ch0_all==0], np.full((ch0_all==0).sum(), 0.02),
               marker="|", color="#2196F3", s=30, alpha=0.5)
    ax.scatter(t0[ch0_all==1], np.full((ch0_all==1).sum(), 0.98),
               marker="|", color="#4CAF50", s=30, alpha=0.5)
    for b in range(1, 30):
        ax.axvline(b*10, color="grey", ls=":", lw=0.5, alpha=0.35)
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel(lbl + "\nProbability", fontsize=9)
    ax.set_ylim(0, 1); ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[-1].set_xlabel("Trial  (vertical lines = block boundaries)")
fig.suptitle(f"Task 0 (2-armed) — step-1 embeddings, same sequence\n"
             f"{COMBO} fold {FOLD}  |  ticks = actual choices in shared seq",
             fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_logits_task0.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Task 1 — mean P(shared choice):")
print(f"  z_low  (PHQ={phq_train[idx_low]:.2f}):  {p_chosen_low.mean():.4f}")
print(f"  z_high (PHQ={phq_train[idx_high]:.2f}): {p_chosen_high.mean():.4f}")
print(f"Task 1 — mean entropy:")
print(f"  z_low:  {ent_low.mean():.4f}")
print(f"  z_high: {ent_high.mean():.4f}")

p0_chosen_low  = np.where(ch0_all==0, p0_low_all,  1-p0_low_all)
p0_chosen_high = np.where(ch0_all==0, p0_high_all, 1-p0_high_all)
print(f"Task 0 — mean P(shared choice):")
print(f"  z_low:  {p0_chosen_low.mean():.4f}")
print(f"  z_high: {p0_chosen_high.mean():.4f}")

print("\nDone.")
