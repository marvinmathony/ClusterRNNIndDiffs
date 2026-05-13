#!/usr/bin/env python3
"""
On-policy rollout for the two extreme-PHQ participants (step-1 embeddings).
Both face the same environment (same reward schedule), choose on-policy.

Analyses:
  1. P(chosen arm) over trials — learning curve
  2. Win-stay / Lose-shift rates in rolling windows
  3. P(stay on arm A) as a function of number of consecutive rewards
     received on that arm — value update profile

Outputs in plots_thalmann/:
  step1_onpolicy_learning.png   — P(optimal arm) over time
  step1_onpolicy_wsls.png       — rolling win-stay / lose-shift
  step1_onpolicy_value.png      — P(stay) as function of reward streak
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
FOLD     = 1
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

A            = 4
N_TRIALS     = 500     # per rollout
N_ENV_SEEDS  = 50      # average over environments
MAX_STREAK   = 6       # max consecutive reward streak to analyse
WINDOW       = 30      # rolling window for win-stay/lose-shift

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)
TASK1_BLK = 30
COL_LOW   = "#4C72B0"
COL_HIGH  = "#C44E52"

# ── Questionnaire ──────────────────────────────────────────────────────────────
quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
quest["PHQ"] = quest[[f"PHQ_9_{i}" for i in range(10)]].mean(1)

# ── Load model ─────────────────────────────────────────────────────────────────
sd0 = sorted(d for d in os.listdir(f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}")
             if d.startswith("seed_"))[0]
frozen_path = f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}/{sd0}/frozen_decoder/policy_model.pt"
print(f"Loading: {frozen_path}")
state = torch.load(frozen_path, map_location="cpu")

with open(f"runs_{DGP}_hp_v2_{COMBO}/fold{FOLD}/{sd0}/config.json") as f:
    cfg = json.load(f)
mc = cfg["model_config"]

decoder = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                  hid=mc["hidden"], A=mc["A"])
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.eval()

task_emb_w = state["task_embedding.weight"]
emb        = state["encoder.embed.weight"].numpy()

train_df = pd.read_csv(f"data_{DGP}/fold{FOLD}/df_train.csv")
subids   = train_df["subid"].values
phq      = quest.reindex(subids)["PHQ"].values.astype(float)
valid    = ~np.isnan(phq)
phq_v, subids_v = phq[valid], subids[valid]
emb_v    = emb[valid]

idx_low  = int(np.argmin(phq_v))
idx_high = int(np.argmax(phq_v))
z_low    = emb_v[idx_low]
z_high   = emb_v[idx_high]

print(f"Lowest  PHQ: subj={subids_v[idx_low]},  PHQ={phq_v[idx_low]:.3f}")
print(f"Highest PHQ: subj={subids_v[idx_high]}, PHQ={phq_v[idx_high]:.3f}")
print(f"diff/std: {((z_high-z_low)/(emb_v.std(0)+1e-8)).round(2)}")

# ── On-policy rollout ──────────────────────────────────────────────────────────
@torch.no_grad()
def rollout(z_vec, reward_schedule, task_id, rng):
    """
    z_vec:           (z_dim,)
    reward_schedule: (T, A) — p(reward) for each arm at each trial
    Returns: choices (T,), rewards (T,), probs (T, A)
    """
    T    = reward_schedule.shape[0]
    z_t  = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0)
    temb = task_emb_w[task_id].numpy()
    h    = decoder.z2h0(z_t).unsqueeze(0)

    prev = np.concatenate([np.zeros(5, dtype=np.float32), temb])
    choices, rewards, probs_out = [], [], []
    for t in range(T):
        x = torch.tensor(prev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x, z_t, hidden=h)
        p = F.softmax(logits[0, 0], dim=-1).numpy()
        probs_out.append(p.copy())

        arm    = rng.choice(A, p=p)
        reward = float(rng.random() < reward_schedule[t, arm])

        oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [reward], temb])
        choices.append(arm); rewards.append(reward)

    return np.array(choices), np.array(rewards), np.stack(probs_out)

# ── Environment: 2-armed stationary with occasional reversal ──────────────────
def make_env(T, A_active=2, p_high=0.75, n_reversals=2, seed=0):
    rng   = np.random.default_rng(seed)
    sched = np.zeros((T, A))
    # Switch points
    switch_pts = sorted(rng.choice(T, size=n_reversals, replace=False))
    best = rng.integers(0, A_active)
    boundaries = [0] + list(switch_pts) + [T]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i+1]
        sched[s:e, best]   = p_high
        sched[s:e, 1-best] = 1 - p_high
        best = 1 - best   # flip
    return sched

# ── Accumulate rollouts over many environment seeds ────────────────────────────
# Storage: per-trial aggregates
p_opt_low  = np.zeros(N_TRIALS)
p_opt_high = np.zeros(N_TRIALS)

# Win-stay / lose-shift: collect raw trial-level events
ws_events_low  = [[] for _ in range(N_TRIALS)]   # ws_events[t] = list of stay(0/1) after win
ls_events_low  = [[] for _ in range(N_TRIALS)]
ws_events_high = [[] for _ in range(N_TRIALS)]
ls_events_high = [[] for _ in range(N_TRIALS)]

# Streak analysis: streak_stay[n] = list of P(stay) values right after n consecutive rewards
streak_stay_low  = {n: [] for n in range(1, MAX_STREAK+1)}
streak_lose_low  = {n: [] for n in range(1, MAX_STREAK+1)}
streak_stay_high = {n: [] for n in range(1, MAX_STREAK+1)}
streak_lose_high = {n: [] for n in range(1, MAX_STREAK+1)}

task_id = int(task_ids_global[0].item())   # use task-0 embedding (2-armed)

print(f"\nRunning {N_ENV_SEEDS} rollouts...")
for env_seed in range(N_ENV_SEEDS):
    sched = make_env(N_TRIALS, seed=env_seed)
    best_arm_seq = np.argmax(sched, axis=1)   # optimal arm at each trial

    for z_vec, (p_opt, ws_ev, ls_ev, st_st, st_lo), label in [
        (z_low,  (p_opt_low,  ws_events_low,  ls_events_low,
                  streak_stay_low,  streak_lose_low),  "low"),
        (z_high, (p_opt_high, ws_events_high, ls_events_high,
                  streak_stay_high, streak_lose_high), "high"),
    ]:
        rng = np.random.default_rng(env_seed * 10000 + (0 if label=="low" else 1))
        ch, rew, probs = rollout(z_vec, sched, task_id, rng)

        # Learning curve
        p_opt += (ch == best_arm_seq).astype(float) / N_ENV_SEEDS

        # Win-stay / lose-shift per trial
        for t in range(len(ch) - 1):
            stayed = int(ch[t+1] == ch[t])
            if rew[t] == 1:
                ws_ev[t].append(stayed)
            else:
                ls_ev[t].append(1 - stayed)

        # Streak analysis: count consecutive outcomes on chosen arm
        streak_len = 0
        streak_type = None   # 'win' or 'loss'
        last_arm = None
        for t in range(len(ch)):
            arm = ch[t]; r = rew[t]
            outcome = 'win' if r == 1 else 'loss'
            if arm == last_arm and outcome == streak_type:
                streak_len += 1
            else:
                streak_len = 1
                streak_type = outcome
            last_arm = arm

            # Record: after streak_len events on this arm, what's P(same arm next)?
            if t + 1 < len(ch) and streak_len <= MAX_STREAK:
                p_next_same = float(probs[t+1, arm])
                if outcome == 'win':
                    st_st[streak_len].append(p_next_same)
                else:
                    st_lo[streak_len].append(p_next_same)

print("Done.")

# ── Smooth helper ──────────────────────────────────────────────────────────────
def smooth(a, w=15):
    return np.convolve(a, np.ones(w)/w, mode="same")

# ── Rolling win-stay / lose-shift ─────────────────────────────────────────────
def rolling_wsls(ws_events, ls_events, T, window=WINDOW):
    ws = np.full(T, np.nan)
    ls = np.full(T, np.nan)
    for t in range(T):
        lo, hi = max(0, t - window//2), min(T, t + window//2)
        wdata = [v for tt in range(lo, hi) for v in ws_events[tt]]
        ldata = [v for tt in range(lo, hi) for v in ls_events[tt]]
        if wdata: ws[t] = np.mean(wdata)
        if ldata: ls[t] = np.mean(ldata)
    return ws, ls

ws_low,  ls_low  = rolling_wsls(ws_events_low,  ls_events_low,  N_TRIALS)
ws_high, ls_high = rolling_wsls(ws_events_high, ls_events_high, N_TRIALS)

# ── Streak analysis ────────────────────────────────────────────────────────────
def streak_mean(streak_dict):
    ns  = sorted(streak_dict.keys())
    m   = np.array([np.mean(streak_dict[n]) if streak_dict[n] else np.nan for n in ns])
    se  = np.array([np.std(streak_dict[n])/np.sqrt(max(len(streak_dict[n]),1)) for n in ns])
    return np.array(ns), m, se

ns_lo_w, m_lo_w, se_lo_w = streak_mean(streak_stay_low)
ns_hi_w, m_hi_w, se_hi_w = streak_mean(streak_stay_high)
ns_lo_l, m_lo_l, se_lo_l = streak_mean(streak_lose_low)
ns_hi_l, m_hi_l, se_hi_l = streak_mean(streak_lose_high)

t = np.arange(N_TRIALS)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Learning curve
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, smooth(p_opt_low),  color=COL_LOW,  lw=2.5,
        label=f"z low  (PHQ={phq_v[idx_low]:.2f})")
ax.plot(t, smooth(p_opt_high), color=COL_HIGH, lw=2.5,
        label=f"z high (PHQ={phq_v[idx_high]:.2f})")
ax.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5, label="Chance (2-arm)")
ax.set_xlabel("Trial"); ax.set_ylabel("P(optimal arm)")
ax.set_ylim(0, 1); ax.set_xlim(0, N_TRIALS-1)
ax.set_title(f"On-policy learning curve — step-1 embeddings\n"
             f"{COMBO} fold {FOLD}, avg over {N_ENV_SEEDS} environments",
             fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_onpolicy_learning.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Rolling win-stay / lose-shift over time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t, ws_low,  color=COL_LOW,  lw=2.5, label=f"z low  (PHQ={phq_v[idx_low]:.2f})")
ax.plot(t, ws_high, color=COL_HIGH, lw=2.5, label=f"z high (PHQ={phq_v[idx_high]:.2f})")
ax.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial"); ax.set_ylabel("Win-Stay rate")
ax.set_ylim(0, 1); ax.set_xlim(0, N_TRIALS-1)
ax.set_title("Win-Stay (rolling window)", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(t, ls_low,  color=COL_LOW,  lw=2.5, label=f"z low  (PHQ={phq_v[idx_low]:.2f})")
ax.plot(t, ls_high, color=COL_HIGH, lw=2.5, label=f"z high (PHQ={phq_v[idx_high]:.2f})")
ax.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Trial"); ax.set_ylabel("Lose-Shift rate")
ax.set_ylim(0, 1); ax.set_xlim(0, N_TRIALS-1)
ax.set_title("Lose-Shift (rolling window)", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"Perseveration over time — on-policy rollout\n"
             f"{COMBO} fold {FOLD}, avg over {N_ENV_SEEDS} environments",
             fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_onpolicy_wsls.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: P(stay) as function of consecutive wins / losses
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.errorbar(ns_lo_w, m_lo_w, yerr=se_lo_w, color=COL_LOW,  lw=2.5, marker="o",
            capsize=4, label=f"z low  (PHQ={phq_v[idx_low]:.2f})")
ax.errorbar(ns_hi_w, m_hi_w, yerr=se_hi_w, color=COL_HIGH, lw=2.5, marker="o",
            capsize=4, label=f"z high (PHQ={phq_v[idx_high]:.2f})")
ax.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Consecutive wins on same arm"); ax.set_ylabel("P(choose same arm again)")
ax.set_ylim(0, 1); ax.set_xticks(range(1, MAX_STREAK+1))
ax.set_title("Value update: consecutive wins", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.errorbar(ns_lo_l, m_lo_l, yerr=se_lo_l, color=COL_LOW,  lw=2.5, marker="o",
            capsize=4, label=f"z low  (PHQ={phq_v[idx_low]:.2f})")
ax.errorbar(ns_hi_l, m_hi_l, yerr=se_hi_l, color=COL_HIGH, lw=2.5, marker="o",
            capsize=4, label=f"z high (PHQ={phq_v[idx_high]:.2f})")
ax.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Consecutive losses on same arm"); ax.set_ylabel("P(choose same arm again)")
ax.set_ylim(0, 1); ax.set_xticks(range(1, MAX_STREAK+1))
ax.set_title("Perseveration: consecutive losses", fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(f"P(stay) as a function of consecutive reward history\n"
             f"{COMBO} fold {FOLD}, avg over {N_ENV_SEEDS} environments",
             fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_onpolicy_value.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved → {out}")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"\nOverall win-stay:  low={np.nanmean(ws_low):.3f}  high={np.nanmean(ws_high):.3f}")
print(f"Overall lose-shift: low={np.nanmean(ls_low):.3f}  high={np.nanmean(ls_high):.3f}")
print(f"Late P(opt):        low={p_opt_low[400:].mean():.3f}  high={p_opt_high[400:].mean():.3f}")
print("\nStreak analysis:")
for n in range(1, MAX_STREAK+1):
    wl = np.mean(streak_stay_low[n])  if streak_stay_low[n]  else np.nan
    wh = np.mean(streak_stay_high[n]) if streak_stay_high[n] else np.nan
    ll = np.mean(streak_lose_low[n])  if streak_lose_low[n]  else np.nan
    lh = np.mean(streak_lose_high[n]) if streak_lose_high[n] else np.nan
    print(f"  streak={n}  win→stay: low={wl:.3f} high={wh:.3f}  "
          f"loss→stay: low={ll:.3f} high={lh:.3f}")
print("\nDone.")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Per-participant outcome sensitivity vs PHQ
# Outcome sensitivity = P(shift|loss) - P(shift|win)
# High values → behaviour strongly guided by outcomes
# Low values  → similar switching regardless of outcome
# ══════════════════════════════════════════════════════════════════════════════
N_SEEDS_PERSUBJ = 20   # fewer seeds per subject for efficiency

@torch.no_grad()
def compute_outcome_sensitivity(z_vec, n_seeds=N_SEEDS_PERSUBJ, T=N_TRIALS):
    """
    P(shift|loss) - P(shift|win), averaged over n_seeds environment draws.
    """
    shift_after_loss = 0; n_loss = 0
    shift_after_win  = 0; n_win  = 0
    for seed in range(n_seeds):
        sched = make_env(T, seed=seed)
        rng   = np.random.default_rng(seed * 7919)
        ch, rew, _ = rollout(z_vec, sched, task_id, rng)
        for t in range(len(ch) - 1):
            shifted = int(ch[t + 1] != ch[t])
            if rew[t] == 0:
                shift_after_loss += shifted; n_loss += 1
            else:
                shift_after_win += shifted; n_win += 1
    p_shift_loss = shift_after_loss / n_loss if n_loss > 0 else np.nan
    p_shift_win  = shift_after_win  / n_win  if n_win  > 0 else np.nan
    return p_shift_loss - p_shift_win

print(f"\nComputing outcome sensitivity for {valid.sum()} participants "
      f"({N_SEEDS_PERSUBJ} env seeds each)...")

persvs = np.array([compute_outcome_sensitivity(emb_v[i]) for i in range(len(emb_v))])
print("Done.")

# ── Raw behavioural outcome sensitivity (from actual task data) ───────────────
# Rewards are continuous → median-split per participant to define win/loss
raw_df0 = pd.read_csv("data/final2armedBanditSession1.csv")
raw_df1 = pd.read_csv("data/finalRestlessSession1.csv")
raw_df  = pd.concat([
    raw_df0[["ID", "block", "trial", "chosen", "reward"]].assign(task=0),
    raw_df1[["ID", "trial", "chosen", "reward"]].assign(task=1, block=0),
], ignore_index=True).sort_values(["ID", "task", "block", "trial"])

def raw_outcome_sensitivity(grp):
    """P(shift|loss) - P(shift|win) from one participant's raw choices."""
    ch  = grp["chosen"].values
    rew = grp["reward"].values
    med = np.median(rew)
    win = rew >= med
    shift_after_loss = 0; n_loss = 0
    shift_after_win  = 0; n_win  = 0
    for t in range(len(ch) - 1):
        shifted = int(ch[t + 1] != ch[t])
        if win[t]:
            shift_after_win += shifted; n_win += 1
        else:
            shift_after_loss += shifted; n_loss += 1
    p_sl = shift_after_loss / n_loss if n_loss > 0 else np.nan
    p_sw = shift_after_win  / n_win  if n_win  > 0 else np.nan
    return p_sl - p_sw

raw_os = raw_df.groupby("ID").apply(raw_outcome_sensitivity)

# Align raw metric with the same participants used in the model analysis
raw_os_aligned = raw_os.reindex(subids_v).values.astype(float)

# ── Correlations ──────────────────────────────────────────────────────────────
from scipy import stats

# Model-based
mask_m   = np.isfinite(persvs) & np.isfinite(phq_v)
r_sp_m, p_sp_m = stats.spearmanr(persvs[mask_m], phq_v[mask_m])
r_pe_m, p_pe_m = stats.pearsonr(persvs[mask_m],  phq_v[mask_m])

# Raw behavioural
mask_r   = np.isfinite(raw_os_aligned) & np.isfinite(phq_v)
r_sp_r, p_sp_r = stats.spearmanr(raw_os_aligned[mask_r], phq_v[mask_r])
r_pe_r, p_pe_r = stats.pearsonr(raw_os_aligned[mask_r],  phq_v[mask_r])

print(f"\nModel-based  — Spearman r={r_sp_m:.3f} p={p_sp_m:.4f}, "
      f"Pearson r={r_pe_m:.3f} p={p_pe_m:.4f}  (n={mask_m.sum()})")
print(f"Raw behaviour — Spearman r={r_sp_r:.3f} p={p_sp_r:.4f}, "
      f"Pearson r={r_pe_r:.3f} p={p_pe_r:.4f}  (n={mask_r.sum()})")

# ── Side-by-side plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, vals, mask, r_sp, p_sp, r_pe, p_pe, title_prefix in [
    (axes[0], persvs,          mask_m, r_sp_m, p_sp_m, r_pe_m, p_pe_m,
     "On-policy rollout (model)"),
    (axes[1], raw_os_aligned,  mask_r, r_sp_r, p_sp_r, r_pe_r, p_pe_r,
     "Raw behaviour"),
]:
    sc = ax.scatter(
        phq_v[mask], vals[mask],
        c=phq_v[mask], cmap="coolwarm",
        s=50, alpha=0.75, edgecolors="none", zorder=3
    )
    # Regression line
    x_fit  = np.linspace(phq_v[mask].min(), phq_v[mask].max(), 200)
    slope, intercept, *_ = stats.linregress(phq_v[mask], vals[mask])
    ax.plot(x_fit, slope * x_fit + intercept, color="black", lw=1.8,
            ls="--", zorder=4, label="OLS fit")

    # Highlight extreme-PHQ participants
    for idx, col, lbl in [
        (idx_low,  COL_LOW,  f"z low (PHQ={phq_v[idx_low]:.2f})"),
        (idx_high, COL_HIGH, f"z high (PHQ={phq_v[idx_high]:.2f})"),
    ]:
        if mask[idx]:
            ax.scatter(phq_v[idx], vals[idx],
                       color=col, s=120, zorder=5,
                       edgecolors="black", linewidths=1.2, label=lbl)

    ax.set_xlabel("PHQ score (mean item)", fontsize=12)
    ax.set_ylabel("P(shift|loss) − P(shift|win)", fontsize=12)
    ax.set_title(
        f"{title_prefix}\n"
        f"Spearman r={r_sp:.3f}, p={p_sp:.3f}  |  "
        f"Pearson r={r_pe:.3f}, p={p_pe:.3f}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(sc, ax=ax, label="PHQ score")

fig.suptitle(
    f"Outcome sensitivity vs PHQ  —  {COMBO} fold {FOLD}",
    fontsize=13, fontweight="bold"
)
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_outcome_sensitivity_vs_phq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Per-participant perseveration (P(stay)) vs PHQ
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perseveration(z_vec, n_seeds=N_SEEDS_PERSUBJ, T=N_TRIALS):
    """P(stay) = fraction of transitions where participant repeats the same action."""
    stay_count = 0
    total = 0
    for seed in range(n_seeds):
        sched = make_env(T, seed=seed)
        rng   = np.random.default_rng(seed * 7919)
        ch, rew, _ = rollout(z_vec, sched, task_id, rng)
        for t in range(len(ch) - 1):
            stay_count += int(ch[t + 1] == ch[t])
            total += 1
    return stay_count / total if total > 0 else np.nan

print(f"\nComputing perseveration for {valid.sum()} participants "
      f"({N_SEEDS_PERSUBJ} env seeds each)...")
persev_model = np.array([compute_perseveration(emb_v[i]) for i in range(len(emb_v))])
print("Done.")

# Raw behavioural perseveration
def raw_perseveration(grp):
    """P(stay) from one participant's raw choices."""
    ch = grp["chosen"].values
    if len(ch) < 2:
        return np.nan
    stayed = sum(int(ch[t + 1] == ch[t]) for t in range(len(ch) - 1))
    return stayed / (len(ch) - 1)

raw_persev = raw_df.groupby("ID").apply(raw_perseveration)
raw_persev_aligned = raw_persev.reindex(subids_v).values.astype(float)

# Correlations
mask_pm = np.isfinite(persev_model) & np.isfinite(phq_v)
r_sp_pm, p_sp_pm = stats.spearmanr(persev_model[mask_pm], phq_v[mask_pm])
r_pe_pm, p_pe_pm = stats.pearsonr(persev_model[mask_pm],  phq_v[mask_pm])

mask_pr = np.isfinite(raw_persev_aligned) & np.isfinite(phq_v)
r_sp_pr, p_sp_pr = stats.spearmanr(raw_persev_aligned[mask_pr], phq_v[mask_pr])
r_pe_pr, p_pe_pr = stats.pearsonr(raw_persev_aligned[mask_pr],  phq_v[mask_pr])

print(f"\nPerseveration (model)  — Spearman r={r_sp_pm:.3f} p={p_sp_pm:.4f}, "
      f"Pearson r={r_pe_pm:.3f} p={p_pe_pm:.4f}  (n={mask_pm.sum()})")
print(f"Perseveration (raw)    — Spearman r={r_sp_pr:.3f} p={p_sp_pr:.4f}, "
      f"Pearson r={r_pe_pr:.3f} p={p_pe_pr:.4f}  (n={mask_pr.sum()})")

# Side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, vals, mask, r_sp, p_sp, r_pe, p_pe, title_prefix in [
    (axes[0], persev_model,       mask_pm, r_sp_pm, p_sp_pm, r_pe_pm, p_pe_pm,
     "On-policy rollout (model)"),
    (axes[1], raw_persev_aligned, mask_pr, r_sp_pr, p_sp_pr, r_pe_pr, p_pe_pr,
     "Raw behaviour"),
]:
    sc = ax.scatter(
        phq_v[mask], vals[mask],
        c=phq_v[mask], cmap="coolwarm",
        s=50, alpha=0.75, edgecolors="none", zorder=3
    )
    x_fit  = np.linspace(phq_v[mask].min(), phq_v[mask].max(), 200)
    slope, intercept, *_ = stats.linregress(phq_v[mask], vals[mask])
    ax.plot(x_fit, slope * x_fit + intercept, color="black", lw=1.8,
            ls="--", zorder=4, label="OLS fit")

    for idx, col, lbl in [
        (idx_low,  COL_LOW,  f"z low (PHQ={phq_v[idx_low]:.2f})"),
        (idx_high, COL_HIGH, f"z high (PHQ={phq_v[idx_high]:.2f})"),
    ]:
        if mask[idx]:
            ax.scatter(phq_v[idx], vals[idx],
                       color=col, s=120, zorder=5,
                       edgecolors="black", linewidths=1.2, label=lbl)

    ax.set_xlabel("PHQ score (mean item)", fontsize=12)
    ax.set_ylabel("P(stay)", fontsize=12)
    ax.set_title(
        f"{title_prefix}\n"
        f"Spearman r={r_sp:.3f}, p={p_sp:.3f}  |  "
        f"Pearson r={r_pe:.3f}, p={p_pe:.3f}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(sc, ax=ax, label="PHQ score")

fig.suptitle(
    f"Perseveration vs PHQ  —  {COMBO} fold {FOLD}",
    fontsize=13, fontweight="bold"
)
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_perseveration_vs_phq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6: Model-based conditional expectation E[metric | PHQ]
#
# Fit z | PHQ ~ N(A * PHQ + b, Sigma) from the learned embeddings, then for
# a grid of PHQ values sample z's, simulate on-policy, and average.
# This leverages the model's expressivity to denoise the PHQ→behaviour link.
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression

N_PHQ_GRID   = 30     # number of PHQ values to evaluate
N_Z_SAMPLES  = 20     # z samples per PHQ grid point
N_SEEDS_COND = 10     # env seeds per z sample (fewer — already averaging over z)

# Fit p(z | PHQ): linear mean + residual covariance
lr = LinearRegression().fit(phq_v[mask_m, None], emb_v[mask_m])
resid = emb_v[mask_m] - lr.predict(phq_v[mask_m, None])
Sigma = np.cov(resid, rowvar=False)   # (z_dim, z_dim)

phq_grid = np.linspace(phq_v[mask_m].min(), phq_v[mask_m].max(), N_PHQ_GRID)
rng_z    = np.random.default_rng(42)

print(f"\nConditional expectation: {N_PHQ_GRID} PHQ points × "
      f"{N_Z_SAMPLES} z samples × {N_SEEDS_COND} env seeds ...")

os_curve    = np.zeros(N_PHQ_GRID)   # outcome sensitivity
persev_curve = np.zeros(N_PHQ_GRID)  # perseveration

for gi, phq_val in enumerate(phq_grid):
    mu_z = lr.predict([[phq_val]])[0]                        # (z_dim,)
    z_samples = rng_z.multivariate_normal(mu_z, Sigma, size=N_Z_SAMPLES)

    os_vals = []
    pv_vals = []
    for z_vec in z_samples:
        os_vals.append(compute_outcome_sensitivity(z_vec, n_seeds=N_SEEDS_COND))
        pv_vals.append(compute_perseveration(z_vec, n_seeds=N_SEEDS_COND))
    os_curve[gi]    = np.nanmean(os_vals)
    persev_curve[gi] = np.nanmean(pv_vals)

    print(f"  PHQ={phq_val:.2f}  outcome_sens={os_curve[gi]:.4f}  "
          f"persev={persev_curve[gi]:.4f}")

print("Done.")

# Correlations of the grid curves (sanity check — monotonicity)
r_grid_os, p_grid_os = stats.spearmanr(phq_grid, os_curve)
r_grid_pv, p_grid_pv = stats.spearmanr(phq_grid, persev_curve)
print(f"Grid Spearman — outcome_sens: r={r_grid_os:.3f} p={p_grid_os:.4f}, "
      f"persev: r={r_grid_pv:.3f} p={p_grid_pv:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, curve, ylabel, title_label, r_grid, p_grid, \
    per_subj_vals, per_subj_mask in [
    (axes[0], os_curve,     "P(shift|loss) − P(shift|win)",
     "Outcome sensitivity", r_grid_os, p_grid_os,
     persvs, mask_m),
    (axes[1], persev_curve, "P(stay)",
     "Perseveration",       r_grid_pv, p_grid_pv,
     persev_model, mask_pm),
]:
    # Individual participant scatter (faded, for reference)
    ax.scatter(phq_v[per_subj_mask], per_subj_vals[per_subj_mask],
               c="grey", s=20, alpha=0.3, zorder=2, label="Per participant")
    # Conditional expectation curve
    ax.plot(phq_grid, curve, color="#D55E00", lw=3, zorder=4,
            label=r"$\mathbb{E}[\mathrm{metric} \mid \mathrm{PHQ}]$")
    ax.fill_between(phq_grid, curve, alpha=0.15, color="#D55E00", zorder=3)

    ax.set_xlabel("PHQ score (mean item)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{title_label}\n"
        f"Grid Spearman r={r_grid:.3f}, p={p_grid:.3f}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Conditional expectation E[metric | PHQ]  —  {COMBO} fold {FOLD}\n"
    f"z | PHQ ~ N(A·PHQ + b, Σ),  {N_Z_SAMPLES} z samples × "
    f"{N_SEEDS_COND} env seeds per grid point",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_conditional_expectation_vs_phq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
