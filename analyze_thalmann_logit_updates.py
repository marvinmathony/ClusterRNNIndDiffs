#!/usr/bin/env python3
"""
On-policy rollout analyses of the Thalmann all-subjects step-1 IDRNN
(loaded from latents_idrnn_step1_bestseed*.pt) and the matching vanilla
AblatedRNN (latents_vanilla_bestseed*.pt). Both produced by
train_and_decode_thalmann_step1.py.

Rollout setup
-------------
  • 4-armed restless bandit (Daw-style; simulate_restless_bandit.py).
  • Continuous reward divided by REWARD_MAX = 100 is fed into the model
    input feature, matching the training-time normalization used in
    load_thalmann.py.
  • Compressibility analyses use the simulator envs (N_ENV_SEEDS draws).
  • Learning curves use the participant's *actual* counterfactual rewards
    from finalRestlessSession1.csv (reward1..reward4) — apples-to-apples
    comparison: same environment, different policy.

Analyses (each appends new outputs to plots_thalmann/)
------------------------------------------------------
  1. JSD policy update — magnitude of softmax(p_t) → softmax(p_{t+1})
     change per trial, split by reward, bar plot for bottom/top 10% PHQ.
       → step1_jsd_barplot.png

  2. Entropy update — previous vs updated softmax entropy, two panels
     (rewarded / unrewarded), both PHQ groups overlaid; plus a detailed
     mid-range view (prev H ∈ [0.3, 0.55]) and per-participant variants.
       → step1_entropy_updates.png
       → step1_mid_entropy_detail.png
       → step1_mid_entropy_per_participant.png

  3. Per-participant quadratic regression of delta_H on prev H
     (delta_H = β0 + β1·H + β2·H²); scatter of PHQ vs β2 / β1.
       → step1_entropy_regression_vs_phq.png

  4. Mean delta entropy across the full range vs PHQ (per-condition
     scatters with OLS fit and extreme groups highlighted).
       → step1_mean_delta_entropy_vs_phq.png

  5. LZW compressibility score (b_LZW / l_LZW) for Humans, IDRNN,
     Vanilla+h (per-subject hidden init), Vanilla (zero init); humans-vs-
     model scatter with per-subject correlation.
       → step1_compressibility.png
       → step1_compressibility_humans_vs_models.png

  6. Compressibility × all questionnaire scales (incl. motivation),
     FDR-BH corrected, with companion side-by-side panel comparing in-
     sample OLS latent decoding vs compressibility correlations.
       → step1_compressibility_questionnaire_corr.png / .csv
       → step1_latent_decoding_questionnaire_corr.csv
       → step1_latent_vs_compressibility_corr.png

  7. Variance decomposition (in-sample OLS R² and adjusted R²) per scale
     × {IDRNN, Vanilla}: latents only / compressibility only / both,
     plus partial F-tests for whether each adds explanatory power.
       → step1_variance_decomposition.png / .csv

  8. Learning curves under human-environment replay: cumulative reward
     and smoothed per-trial mean reward (Humans / IDRNN / Vanilla+h /
     Vanilla), plus asymptotic-performance boxplot over the last 50
     trials. Raw curves saved for downstream reuse.
       → step1_learning_curves.png
       → step1_learning_curves_boxplot.png
       → step1_learning_curves.npz
"""

import os, json
from functools import lru_cache
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
from simulate_restless_bandit import simulate_restless_bandit, T as SIM_T, K as SIM_K

# ── Config ─────────────────────────────────────────────────────────────────────
DGP      = "thalmann"
# All-subjects step-1 model (no outer CV; covers every participant). Produced by
# train_and_decode_thalmann_step1.py.  Replace the bestseed suffix with whatever
# that script picks.
import glob
_latent_glob = glob.glob(
    "plots_thalmann/step1_vs_vanilla/latents_idrnn_step1_bestseed*.pt"
)
assert _latent_glob, ("No all-subjects step-1 latents found — run "
                      "train_and_decode_thalmann_step1.py first")
ALL_SUBJ_LATENTS = max(_latent_glob, key=os.path.getmtime)
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

A            = 4          # 4-armed restless bandit
assert A == SIM_K, f"A={A} != simulator K={SIM_K}"
N_TRIALS     = SIM_T      # 200 trials per walk (fixed by the simulator)
N_ENV_SEEDS  = 100
PCT          = 10        # percentile cutoff for extreme groups
# Match training-time normalization (load_thalmann.py:42): r_input = r_raw / 100
REWARD_MAX   = 100.0

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
)

COL_LOW   = "#4C72B0"
COL_HIGH  = "#C44E52"

# ── Questionnaire ──────────────────────────────────────────────────────────────
quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
quest["PHQ"] = quest[[f"PHQ_9_{i}" for i in range(10)]].mean(1)

# ── Load all-subjects step-1 model ─────────────────────────────────────────────
# saved["z"] is `model.encoder.embed.weight` after the final step-1 training on
# ALL participants (see train_and_decode_thalmann_step1.py:243 + :491-495), i.e.
# the per-subject step-1 lookup embeddings. saved["model_state"] is the matching
# step-1 IDRNN; the decoder block is reused for on-policy rollouts here.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print(f"Loading: {ALL_SUBJ_LATENTS}")
saved = torch.load(ALL_SUBJ_LATENTS, map_location=DEVICE, weights_only=False)
assert "model_state" in saved, (
    "Latents file does not include full model state — re-run "
    "train_and_decode_thalmann_step1.py after the save-state update."
)
Z_DIM       = saved["z_dim"]
HIDDEN      = saved["hidden"]
TASK_EMB_DIM = saved["task_emb_dim"]
BASE_IN_DIM = saved.get("base_in_dim", 5)
DEC_IN_DIM  = BASE_IN_DIM + TASK_EMB_DIM
BEST_SEED   = saved["seed"]
print(f"  z_dim={Z_DIM}, hidden={HIDDEN}, task_emb_dim={TASK_EMB_DIM}, "
      f"dec_in_dim={DEC_IN_DIM}, best seed={BEST_SEED}")

state = saved["model_state"]
# Sanity check: the saved z must equal encoder.embed.weight from the same
# checkpoint (i.e. step-1 lookup embeddings, not e.g. step-2 inferred latents).
emb_from_state = state["encoder.embed.weight"].detach().cpu().numpy()
assert np.allclose(np.asarray(saved["z"]), emb_from_state, atol=1e-6), (
    "saved['z'] does not match state['encoder.embed.weight'] — these are not "
    "the step-1 lookup embeddings of the bundled model_state."
)
print("  Verified: saved['z'] == state['encoder.embed.weight'] (step-1 embeddings)")

decoder = Decoder(in_dim=DEC_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.to(DEVICE).eval()

task_emb_w = state["task_embedding.weight"].to(DEVICE)
emb        = np.asarray(saved["z"])        # (236, z_dim) step-1 lookup embeddings
subids     = np.asarray(saved["subids"])

phq      = quest.reindex(subids)["PHQ"].values.astype(float)
valid    = ~np.isnan(phq)
phq_v, subids_v = phq[valid], subids[valid]
emb_v    = emb[valid]
print(f"Loaded {emb.shape[0]} participants, {valid.sum()} with valid PHQ")

# ── Select bottom/top 10% PHQ groups ──────────────────────────────────────────
lo_thresh = np.percentile(phq_v, PCT)
hi_thresh = np.percentile(phq_v, 100 - PCT)
idx_low_group  = np.where(phq_v <= lo_thresh)[0]
idx_high_group = np.where(phq_v >= hi_thresh)[0]

print(f"Low  PHQ group (≤{lo_thresh:.3f}): n={len(idx_low_group)}")
print(f"High PHQ group (≥{hi_thresh:.3f}): n={len(idx_high_group)}")

# ── On-policy rollout returning logits ────────────────────────────────────────
@torch.no_grad()
def rollout_logits(z_vec, reward_schedule, task_id, rng):
    """On-policy IDRNN rollout.
    reward_schedule: (T, K) continuous reward array (raw scale, e.g. ~17–83).
    Input feature 4 receives reward / REWARD_MAX, matching training format.
    Returns: choices (T,), rewards_raw (T,), logits_arr (T, A)
    """
    T    = reward_schedule.shape[0]
    z_t  = torch.tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[task_id].detach().cpu().numpy()
    h    = decoder.z2h0(z_t).unsqueeze(0)

    prev = np.concatenate([np.zeros(5, dtype=np.float32), temb])
    choices, rewards_raw, logits_out = [], [], []
    for t in range(T):
        x = torch.tensor(prev, dtype=torch.float32,
                         device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x, z_t, hidden=h)
        logits_np = logits[0, 0].detach().cpu().numpy()
        logits_out.append(logits_np.copy())

        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm        = rng.choice(A, p=p)
        reward_raw = float(reward_schedule[t, arm])
        reward_in  = reward_raw / REWARD_MAX

        oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [reward_in], temb])
        choices.append(arm); rewards_raw.append(reward_raw)

    return np.array(choices), np.array(rewards_raw), np.stack(logits_out)

# ── Environment ───────────────────────────────────────────────────────────────
# Restless 4-armed bandit: per-arm Gaussian random walks around θ=50.  We
# return the *continuous* reward array (T, K) — rollouts index at
# reward_table[t, chosen_arm] and feed reward / REWARD_MAX into the model
# input feature, matching the training-time normalization
# (load_thalmann.py:42).
@lru_cache(maxsize=None)
def make_env(T, seed=0):
    assert T == SIM_T, (
        f"simulator produces T={SIM_T}, analysis requested T={T}"
    )
    return simulate_restless_bandit(seed)["rewards"].astype(np.float32)

# ── Collect JSD between consecutive trials ────────────────────────────────────
# JSD(p_t || p_{t+1}) measures how much the policy changed, regardless of
# direction.  Keyed by rewarded (0/1).

task_id = 1  # restless bandit (task-1); rollouts use the restless task embedding
assert task_id in task_ids_global.tolist(), "restless task id not found in training"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ ANALYSES 1-4 (JSD / entropy update / quadratic regression / mean ΔH      ║
# ║ vs PHQ) ARE COMMENTED OUT — currently focusing on compressibility +      ║
# ║ learning curves. The original code follows inside a raw triple-quoted    ║
# ║ string so Python skips it; remove the wrapper to re-enable.              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_SECTIONS_1_TO_4_DISABLED = r'''
def softmax(logits):
    p = np.exp(logits - logits.max())
    return p / p.sum()

def entropy(p):
    """Shannon entropy in nats."""
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log(p))

def kl_div(p, q):
    """KL(p || q) with clipping to avoid log(0)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return np.sum(p * np.log(p / q))

def jsd(p, q):
    """Jensen-Shannon divergence (not the sqrt distance)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def collect_jsd_and_entropy(participant_indices):
    """Collect entropy-transition tuples (JSD collection disabled).
    ent_bins values are (prev_H, curr_H, trial_idx, participant_idx) tuples."""
    ent_bins = {0: [], 1: []}
    for pi in participant_indices:
        z_vec = emb_v[pi]
        for env_seed in range(N_ENV_SEEDS):
            sched = make_env(N_TRIALS, seed=env_seed)
            rng = np.random.default_rng(env_seed * 10000 + pi)
            ch, rew, logits = rollout_logits(z_vec, sched, task_id, rng)

            for t in range(len(ch) - 1):
                p_prev = softmax(logits[t])
                p_curr = softmax(logits[t + 1])
                rewarded = int(rew[t])
                # jsd_bins[rewarded].append(jsd(p_prev, p_curr))  # JSD disabled
                ent_bins[rewarded].append((entropy(p_prev), entropy(p_curr), t, pi))
    return ent_bins

print(f"\nCollecting entropy transitions for low PHQ group ({len(idx_low_group)} participants)...")
ent_low  = collect_jsd_and_entropy(idx_low_group)
print(f"Collecting entropy transitions for high PHQ group ({len(idx_high_group)} participants)...")
ent_high = collect_jsd_and_entropy(idx_high_group)
print("Done.")

# These are needed by downstream entropy analyses even with JSD disabled
from scipy import stats
REW_NAMES = {0: "Unrewarded", 1: "Rewarded"}

# ══════════════════════════════════════════════════════════════════════════════
# JSD analyses (bar plot, stats tests) — commented out.
# ══════════════════════════════════════════════════════════════════════════════
"""
# ── Compute means and SEMs ───────────────────────────────────────────────────
labels = []
means  = []
sems   = []
colors = []

for rew in [0, 1]:
    for jsd_bins, group_label, col in [
        (jsd_low,  "Low PHQ",  COL_LOW),
        (jsd_high, "High PHQ", COL_HIGH),
    ]:
        vals = np.array(jsd_bins[rew])
        labels.append(f"{group_label}\n{REW_NAMES[rew]}")
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)))
        colors.append(col)

# ── Statistical tests ────────────────────────────────────────────────────────
for rew in [0, 1]:
    vals_lo = np.array(jsd_low[rew])
    vals_hi = np.array(jsd_high[rew])
    t_stat, p_val = stats.ttest_ind(vals_lo, vals_hi, equal_var=False)
    u_stat, p_mwu = stats.mannwhitneyu(vals_lo, vals_hi, alternative="two-sided")
    print(f"\n{REW_NAMES[rew]}:")
    print(f"  Low  PHQ: mean JSD = {vals_lo.mean():.6f} (n={len(vals_lo):,})")
    print(f"  High PHQ: mean JSD = {vals_hi.mean():.6f} (n={len(vals_hi):,})")
    print(f"  Welch t = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"  Mann-Whitney U = {u_stat:.0f}, p = {p_mwu:.2e}")
    print(f"  Cohen's d = {(vals_lo.mean() - vals_hi.mean()) / np.sqrt((vals_lo.var() + vals_hi.var()) / 2):.4f}")

# ── Bar plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=[1.96 * s for s in sems],
              color=colors, capsize=6, edgecolor="black", linewidth=0.8,
              width=0.6, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Mean JSD (policy update magnitude)", fontsize=12)
ax.set_title(
    f"Jensen-Shannon divergence between consecutive trials\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds, "
    f"bottom/top {PCT}% PHQ",
    fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

# Add significance brackets
for rew, x0, x1 in [(0, 0, 1), (1, 2, 3)]:
    vals_lo = np.array(jsd_low[rew])
    vals_hi = np.array(jsd_high[rew])
    _, p_val = stats.ttest_ind(vals_lo, vals_hi, equal_var=False)
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    y_max = max(means[x0] + 1.96 * sems[x0], means[x1] + 1.96 * sems[x1])
    y_bar = y_max * 1.08
    ax.plot([x0, x0, x1, x1], [y_max * 1.02, y_bar, y_bar, y_max * 1.02],
            color="black", lw=1.2)
    ax.text((x0 + x1) / 2, y_bar * 1.01, sig,
            ha="center", va="bottom", fontsize=12, fontweight="bold")

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_jsd_barplot.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")
"""

# ══════════════════════════════════════════════════════════════════════════════
# Entropy update plot: prev entropy (x) vs updated entropy (y),
# 2 panels (Unrewarded / Rewarded), both groups overlaid
# ══════════════════════════════════════════════════════════════════════════════

def bin_entropy(pairs, n_bins=50):
    prev = np.array([p[0] for p in pairs])
    curr = np.array([p[1] for p in pairs])
    edges = np.linspace(np.percentile(prev, 1), np.percentile(prev, 99), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(prev, edges) - 1, 0, n_bins - 1)
    means_out = np.full(n_bins, np.nan)
    sems_out  = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 10:
            means_out[b] = curr[m].mean()
            sems_out[b]  = curr[m].std() / np.sqrt(m.sum())
    return centers, means_out, sems_out

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for col, rewarded in enumerate([0, 1]):
    ax = axes[col]

    for ent_bins, color, label in [
        (ent_low,  COL_LOW,  f"Low PHQ (bottom {PCT}%, n={len(idx_low_group)})"),
        (ent_high, COL_HIGH, f"High PHQ (top {PCT}%, n={len(idx_high_group)})"),
    ]:
        centers, means_e, sems_e = bin_entropy(ent_bins[rewarded])
        ok = np.isfinite(means_e)
        ax.plot(centers[ok], means_e[ok], color=color, lw=2.5, label=label, zorder=5)
        ax.fill_between(centers[ok],
                        means_e[ok] - 1.96 * sems_e[ok],
                        means_e[ok] + 1.96 * sems_e[ok],
                        color=color, alpha=0.15, zorder=4)

    # Identity line
    all_prev = ([p[0] for p in ent_low[rewarded]] +
                [p[0] for p in ent_high[rewarded]])
    lo_lim = np.percentile(all_prev, 1)
    hi_lim = np.percentile(all_prev, 99)
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim],
            color="grey", ls="--", lw=1, alpha=0.6, label="Identity")

    ax.set_xlabel("Previous entropy (nats)", fontsize=12)
    ax.set_ylabel("Updated entropy (nats)", fontsize=12)
    ax.set_title(REW_NAMES[rewarded], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Entropy updates — low vs high PHQ\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds",
    fontsize=14, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_entropy_updates.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Detailed analysis: unrewarded trials in mid-entropy range (0.3 – 0.55)
# ══════════════════════════════════════════════════════════════════════════════
# Rescaled for K=4 (max entropy = ln 4 ≈ 1.386). The original K=2 window
# (0.30, 0.55) covered ~43-80% of ln 2; mapping to K=4 preserves that
# fraction of max entropy.
H_LO, H_HI = 0.60, 1.10

def filter_mid_entropy(ent_list, h_lo=H_LO, h_hi=H_HI):
    """Filter to transitions where previous entropy is in [h_lo, h_hi].
    Returns arrays: prev_H, curr_H, trial_idx, participant_idx."""
    arr = np.array(ent_list)  # (N, 4)
    mask = (arr[:, 0] >= h_lo) & (arr[:, 0] <= h_hi)
    return arr[mask]

mid_low  = filter_mid_entropy(ent_low[0])   # unrewarded only
mid_high = filter_mid_entropy(ent_high[0])

print(f"\n{'='*70}")
print(f"Mid-entropy unrewarded subset (prev H in [{H_LO}, {H_HI}])")
print(f"  Low  PHQ: n = {len(mid_low):,}")
print(f"  High PHQ: n = {len(mid_high):,}")

delta_low  = mid_low[:, 1] - mid_low[:, 0]
delta_high = mid_high[:, 1] - mid_high[:, 0]

print(f"\n  Delta entropy (updated - previous):")
print(f"    Low  PHQ: mean = {delta_low.mean():.5f}, median = {np.median(delta_low):.5f}")
print(f"    High PHQ: mean = {delta_high.mean():.5f}, median = {np.median(delta_high):.5f}")

t_stat, p_val = stats.ttest_ind(delta_low, delta_high, equal_var=False)
u_stat, p_mwu = stats.mannwhitneyu(delta_low, delta_high, alternative="two-sided")
d = (delta_high.mean() - delta_low.mean()) / np.sqrt((delta_low.var() + delta_high.var()) / 2)
print(f"    Welch t = {t_stat:.3f}, p = {p_val:.2e}")
print(f"    Mann-Whitney p = {p_mwu:.2e}")
print(f"    Cohen's d = {d:.4f}")

# ── Figure: 4-panel detailed view ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Zoomed entropy update (finer binning) — unrewarded, mid-range
ax = axes[0, 0]
for data, color, label in [
    (mid_low,  COL_LOW,  f"Low PHQ (n={len(mid_low):,})"),
    (mid_high, COL_HIGH, f"High PHQ (n={len(mid_high):,})"),
]:
    prev, curr = data[:, 0], data[:, 1]
    n_bins = 40
    edges = np.linspace(H_LO, H_HI, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bidx = np.clip(np.digitize(prev, edges) - 1, 0, n_bins - 1)
    bm = np.full(n_bins, np.nan)
    bs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = bidx == b
        if m.sum() > 10:
            bm[b] = curr[m].mean()
            bs[b] = curr[m].std() / np.sqrt(m.sum())
    ok = np.isfinite(bm)
    ax.plot(centers[ok], bm[ok], color=color, lw=2.5, label=label, zorder=5)
    ax.fill_between(centers[ok], bm[ok] - 1.96*bs[ok], bm[ok] + 1.96*bs[ok],
                    color=color, alpha=0.15, zorder=4)
ax.plot([H_LO, H_HI], [H_LO, H_HI], color="grey", ls="--", lw=1, alpha=0.6,
        label="Identity")
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Updated entropy (nats)", fontsize=11)
ax.set_title("A) Zoomed entropy update (unrewarded, mid-range)", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: Delta entropy (updated - previous) as function of previous entropy
ax = axes[0, 1]
for data, color, label in [
    (mid_low,  COL_LOW,  "Low PHQ"),
    (mid_high, COL_HIGH, "High PHQ"),
]:
    prev, curr = data[:, 0], data[:, 1]
    delta = curr - prev
    n_bins = 40
    edges = np.linspace(H_LO, H_HI, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bidx = np.clip(np.digitize(prev, edges) - 1, 0, n_bins - 1)
    bm = np.full(n_bins, np.nan)
    bs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = bidx == b
        if m.sum() > 10:
            bm[b] = delta[m].mean()
            bs[b] = delta[m].std() / np.sqrt(m.sum())
    ok = np.isfinite(bm)
    ax.plot(centers[ok], bm[ok], color=color, lw=2.5, label=label, zorder=5)
    ax.fill_between(centers[ok], bm[ok] - 1.96*bs[ok], bm[ok] + 1.96*bs[ok],
                    color=color, alpha=0.15, zorder=4)
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Delta entropy (nats)", fontsize=11)
ax.set_title("B) Entropy change after unrewarded trial", fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel C: Delta entropy split by early (first 100 trials) vs late (last 100)
ax = axes[1, 0]
EARLY_CUT  = N_TRIALS // 3            # first third
LATE_START = N_TRIALS - N_TRIALS // 3  # last third

for phase_label, t_lo, t_hi, ls in [
    (f"Early (t<{EARLY_CUT})",      0,          EARLY_CUT, "-"),
    (f"Late (t≥{LATE_START})",      LATE_START, N_TRIALS,  "--"),
]:
    for data, color, group_label in [
        (mid_low,  COL_LOW,  "Low"),
        (mid_high, COL_HIGH, "High"),
    ]:
        phase_mask = (data[:, 2] >= t_lo) & (data[:, 2] < t_hi)
        sub = data[phase_mask]
        if len(sub) < 50:
            continue
        prev, curr = sub[:, 0], sub[:, 1]
        delta = curr - prev
        n_bins = 25
        edges = np.linspace(H_LO, H_HI, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bidx = np.clip(np.digitize(prev, edges) - 1, 0, n_bins - 1)
        bm = np.full(n_bins, np.nan)
        bs = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = bidx == b
            if m.sum() > 10:
                bm[b] = delta[m].mean()
                bs[b] = delta[m].std() / np.sqrt(m.sum())
        ok = np.isfinite(bm)
        ax.plot(centers[ok], bm[ok], color=color, lw=2, ls=ls,
                label=f"{group_label} — {phase_label}", zorder=5)
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Delta entropy (nats)", fontsize=11)
ax.set_title("C) Early vs late trials", fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel D: Distribution of delta entropy (histograms)
ax = axes[1, 1]
bins_hist = np.linspace(
    min(np.percentile(delta_low, 1), np.percentile(delta_high, 1)),
    max(np.percentile(delta_low, 99), np.percentile(delta_high, 99)),
    60)
ax.hist(delta_low,  bins=bins_hist, color=COL_LOW,  alpha=0.5, density=True,
        label=f"Low PHQ (mean={delta_low.mean():.4f})")
ax.hist(delta_high, bins=bins_hist, color=COL_HIGH, alpha=0.5, density=True,
        label=f"High PHQ (mean={delta_high.mean():.4f})")
ax.axvline(delta_low.mean(),  color=COL_LOW,  ls="--", lw=2)
ax.axvline(delta_high.mean(), color=COL_HIGH, ls="--", lw=2)
ax.axvline(0, color="grey", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("Delta entropy (nats)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"D) Distribution of entropy change\n"
             f"Welch t={t_stat:.2f}, p={p_val:.2e}, Cohen's d={d:.3f}",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Mid-entropy unrewarded trials (H in [{H_LO}, {H_HI}])\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds, bottom/top {PCT}% PHQ",
    fontsize=13, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_mid_entropy_detail.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Per-bin statistical tests across the mid-entropy range ───────────────────
print(f"\nPer-bin Welch t-tests (delta entropy, {H_LO}–{H_HI}):")
n_test_bins = 10
test_edges = np.linspace(H_LO, H_HI, n_test_bins + 1)
for b in range(n_test_bins):
    lo_e, hi_e = test_edges[b], test_edges[b + 1]
    ml = (mid_low[:, 0] >= lo_e) & (mid_low[:, 0] < hi_e)
    mh = (mid_high[:, 0] >= lo_e) & (mid_high[:, 0] < hi_e)
    dl = mid_low[ml, 1] - mid_low[ml, 0]
    dh = mid_high[mh, 1] - mid_high[mh, 0]
    if len(dl) > 20 and len(dh) > 20:
        t_b, p_b = stats.ttest_ind(dl, dh, equal_var=False)
        d_b = (dh.mean() - dl.mean()) / np.sqrt((dl.var() + dh.var()) / 2)
        print(f"  H=[{lo_e:.3f},{hi_e:.3f}): "
              f"low={dl.mean():+.5f} (n={len(dl):,})  "
              f"high={dh.mean():+.5f} (n={len(dh):,})  "
              f"d={d_b:+.4f}  p={p_b:.3e}")

# ══════════════════════════════════════════════════════════════════════════════
# Per-participant analysis: compute each participant's mean delta-entropy
# curve, then average across participants (proper between-subject SEM).
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("Per-participant mid-entropy analysis")

N_DETAIL_BINS = 40
detail_edges   = np.linspace(H_LO, H_HI, N_DETAIL_BINS + 1)
detail_centers = (detail_edges[:-1] + detail_edges[1:]) / 2

def per_participant_delta_curves(mid_data, participant_indices):
    """For each participant, compute mean delta entropy per bin.
    Returns: (n_participants, n_bins) array with NaN where insufficient data."""
    curves = np.full((len(participant_indices), N_DETAIL_BINS), np.nan)
    for i, pi in enumerate(participant_indices):
        mask = mid_data[:, 3] == pi
        sub = mid_data[mask]
        if len(sub) < 10:
            continue
        prev, curr = sub[:, 0], sub[:, 1]
        delta = curr - prev
        bidx = np.clip(np.digitize(prev, detail_edges) - 1, 0, N_DETAIL_BINS - 1)
        for b in range(N_DETAIL_BINS):
            m = bidx == b
            if m.sum() >= 5:
                curves[i, b] = delta[m].mean()
    return curves

curves_low  = per_participant_delta_curves(mid_low, idx_low_group)
curves_high = per_participant_delta_curves(mid_high, idx_high_group)

# Per-participant overall mean delta entropy (for t-test)
subj_mean_delta_low  = np.array([
    (mid_low[mid_low[:, 3] == pi, 1] - mid_low[mid_low[:, 3] == pi, 0]).mean()
    for pi in idx_low_group
    if (mid_low[:, 3] == pi).sum() > 0
])
subj_mean_delta_high = np.array([
    (mid_high[mid_high[:, 3] == pi, 1] - mid_high[mid_high[:, 3] == pi, 0]).mean()
    for pi in idx_high_group
    if (mid_high[:, 3] == pi).sum() > 0
])

t_subj, p_subj = stats.ttest_ind(subj_mean_delta_low, subj_mean_delta_high, equal_var=False)
d_subj = ((subj_mean_delta_high.mean() - subj_mean_delta_low.mean()) /
          np.sqrt((subj_mean_delta_low.var() + subj_mean_delta_high.var()) / 2))

print(f"  Low  PHQ: n_subj={len(subj_mean_delta_low)}, "
      f"mean delta={subj_mean_delta_low.mean():.5f} ± {subj_mean_delta_low.std():.5f}")
print(f"  High PHQ: n_subj={len(subj_mean_delta_high)}, "
      f"mean delta={subj_mean_delta_high.mean():.5f} ± {subj_mean_delta_high.std():.5f}")
print(f"  Welch t = {t_subj:.3f}, p = {p_subj:.4f}")
print(f"  Cohen's d = {d_subj:.4f}")

# ── Figure: 4-panel per-participant view ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Per-participant delta entropy curves (mean ± between-subject SEM)
ax = axes[0, 0]
for curves, color, label in [
    (curves_low,  COL_LOW,  f"Low PHQ (n={len(idx_low_group)})"),
    (curves_high, COL_HIGH, f"High PHQ (n={len(idx_high_group)})"),
]:
    n_valid = np.sum(np.isfinite(curves), axis=0)
    m = np.nanmean(curves, axis=0)
    s = np.nanstd(curves, axis=0) / np.sqrt(np.maximum(n_valid, 1))
    ok = n_valid >= 3
    ax.plot(detail_centers[ok], m[ok], color=color, lw=2.5, label=label, zorder=5)
    ax.fill_between(detail_centers[ok], m[ok] - 1.96*s[ok], m[ok] + 1.96*s[ok],
                    color=color, alpha=0.15, zorder=4)
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Delta entropy (nats)", fontsize=11)
ax.set_title("A) Mean participant delta-entropy curve\n"
             "(between-subject SEM)", fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: Individual participant curves (spaghetti plot)
ax = axes[0, 1]
for curves, color, label, indices in [
    (curves_low,  COL_LOW,  "Low PHQ",  idx_low_group),
    (curves_high, COL_HIGH, "High PHQ", idx_high_group),
]:
    for i in range(len(indices)):
        ok = np.isfinite(curves[i])
        if ok.sum() >= 5:
            ax.plot(detail_centers[ok], curves[i][ok],
                    color=color, alpha=0.25, lw=0.8)
    # Group mean on top
    n_valid = np.sum(np.isfinite(curves), axis=0)
    m = np.nanmean(curves, axis=0)
    ok = n_valid >= 3
    ax.plot(detail_centers[ok], m[ok], color=color, lw=3, label=label, zorder=5)
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Delta entropy (nats)", fontsize=11)
ax.set_title("B) Individual participant curves + group mean", fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel C: Per-bin between-subject t-tests
ax = axes[1, 0]
p_vals_per_bin = np.full(N_DETAIL_BINS, np.nan)
d_vals_per_bin = np.full(N_DETAIL_BINS, np.nan)
for b in range(N_DETAIL_BINS):
    lo_vals = curves_low[:, b][np.isfinite(curves_low[:, b])]
    hi_vals = curves_high[:, b][np.isfinite(curves_high[:, b])]
    if len(lo_vals) >= 3 and len(hi_vals) >= 3:
        _, p_vals_per_bin[b] = stats.ttest_ind(lo_vals, hi_vals, equal_var=False)
        pooled_sd = np.sqrt((lo_vals.var() + hi_vals.var()) / 2)
        if pooled_sd > 0:
            d_vals_per_bin[b] = (hi_vals.mean() - lo_vals.mean()) / pooled_sd

ok = np.isfinite(d_vals_per_bin)
ax.bar(detail_centers[ok], d_vals_per_bin[ok],
       width=(detail_edges[1] - detail_edges[0]) * 0.85,
       color=np.where(d_vals_per_bin[ok] > 0, COL_HIGH, COL_LOW),
       edgecolor="black", linewidth=0.5, zorder=3)
# Mark significant bins
for b in range(N_DETAIL_BINS):
    if np.isfinite(p_vals_per_bin[b]) and p_vals_per_bin[b] < 0.05:
        ax.text(detail_centers[b], d_vals_per_bin[b] + 0.02 * np.sign(d_vals_per_bin[b]),
                "*", ha="center", va="bottom" if d_vals_per_bin[b] > 0 else "top",
                fontsize=10, fontweight="bold")
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Cohen's d (high − low PHQ)", fontsize=11)
ax.set_title("C) Per-bin effect size (between-subject)\n"
             "* = p < 0.05", fontweight="bold", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel D: Per-participant mean delta entropy (scatter + box)
ax = axes[1, 1]
positions = [0, 1]
bp = ax.boxplot([subj_mean_delta_low, subj_mean_delta_high],
                positions=positions, widths=0.5, patch_artist=True,
                showfliers=False, zorder=2)
for patch, color in zip(bp["boxes"], [COL_LOW, COL_HIGH]):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)
for element in ["whiskers", "caps", "medians"]:
    for line in bp[element]:
        line.set_color("black")

# Overlay individual points
jitter_low  = np.random.default_rng(42).normal(0, 0.05, len(subj_mean_delta_low))
jitter_high = np.random.default_rng(43).normal(0, 0.05, len(subj_mean_delta_high))
ax.scatter(0 + jitter_low,  subj_mean_delta_low,
           color=COL_LOW, s=40, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)
ax.scatter(1 + jitter_high, subj_mean_delta_high,
           color=COL_HIGH, s=40, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels([f"Low PHQ\n(n={len(subj_mean_delta_low)})",
                     f"High PHQ\n(n={len(subj_mean_delta_high)})"], fontsize=11)
ax.set_ylabel("Mean delta entropy per participant", fontsize=11)
ax.set_title(f"D) Per-participant mean delta entropy\n"
             f"Welch t={t_subj:.2f}, p={p_subj:.4f}, Cohen's d={d_subj:.3f}",
             fontweight="bold", fontsize=10)
ax.axhline(0, color="grey", ls=":", lw=1, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Per-participant mid-entropy analysis (H in [{H_LO}, {H_HI}], unrewarded)\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds, bottom/top {PCT}% PHQ",
    fontsize=13, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_mid_entropy_per_participant.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Per-participant quadratic regression:
#   delta_H = beta_0 + beta_1 * H_prev + beta_2 * H_prev^2
# fitted on each participant's unrewarded mid-entropy transitions.
# Then: PHQ vs beta_2 scatter across ALL participants.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("Per-participant quadratic regression (all participants)")
print(f"Running rollouts for all {len(emb_v)} participants...")

# Collect unrewarded (prev_H, delta_H) per participant for ALL participants
from numpy.polynomial import polynomial as P

@torch.no_grad()
def collect_entropy_for_participant(pi):
    """Return list of (prev_H, delta_H, rewarded) tuples — full entropy range,
    both reward conditions."""
    z_vec = emb_v[pi]
    rows = []
    for env_seed in range(N_ENV_SEEDS):
        sched = make_env(N_TRIALS, seed=env_seed)
        rng = np.random.default_rng(env_seed * 10000 + pi)
        ch, rew, logits = rollout_logits(z_vec, sched, task_id, rng)
        for t in range(len(ch) - 1):
            p_prev = softmax(logits[t])
            p_curr = softmax(logits[t + 1])
            h_prev = entropy(p_prev)
            rows.append((h_prev, entropy(p_curr) - h_prev, int(rew[t])))
    return rows

beta0_all = np.full(len(emb_v), np.nan)
beta1_all = np.full(len(emb_v), np.nan)
beta2_all = np.full(len(emb_v), np.nan)

# Per-participant mean delta entropy across the entire entropy range,
# split by reward condition (computed in the same rollout pass).
mean_dH_unrew = np.full(len(emb_v), np.nan)
mean_dH_rew   = np.full(len(emb_v), np.nan)

for pi in range(len(emb_v)):
    rows = collect_entropy_for_participant(pi)

    # Per-condition mean delta entropy (full range)
    arr_full = np.array(rows)  # (N, 3): prev_H, delta_H, rewarded
    if arr_full.size:
        m0 = arr_full[arr_full[:, 2] == 0]
        m1 = arr_full[arr_full[:, 2] == 1]
        if len(m0) >= 5:
            mean_dH_unrew[pi] = m0[:, 1].mean()
        if len(m1) >= 5:
            mean_dH_rew[pi] = m1[:, 1].mean()

    # Quadratic regression: unrewarded, mid-entropy only
    mid = arr_full[(arr_full[:, 2] == 0) &
                   (arr_full[:, 0] >= H_LO) &
                   (arr_full[:, 0] <= H_HI)]
    if len(mid) < 20:
        if (pi + 1) % 50 == 0:
            print(f"  {pi+1}/{len(emb_v)} participants done")
        continue
    h_prev = mid[:, 0]
    delta_h = mid[:, 1]
    # Fit: delta_H = beta_0 + beta_1*H + beta_2*H^2
    X = np.column_stack([np.ones(len(h_prev)), h_prev, h_prev**2])
    betas, res, rank, sv = np.linalg.lstsq(X, delta_h, rcond=None)
    beta0_all[pi] = betas[0]
    beta1_all[pi] = betas[1]
    beta2_all[pi] = betas[2]
    if (pi + 1) % 50 == 0:
        print(f"  {pi+1}/{len(emb_v)} participants done")

print(f"  {len(emb_v)}/{len(emb_v)} participants done")

valid_reg = np.isfinite(beta2_all) & np.isfinite(phq_v)
print(f"  Valid regressions: {valid_reg.sum()}")

# Correlations
r_sp, p_sp = stats.spearmanr(phq_v[valid_reg], beta2_all[valid_reg])
r_pe, p_pe = stats.pearsonr(phq_v[valid_reg], beta2_all[valid_reg])
print(f"  PHQ vs beta_2: Spearman r={r_sp:.4f}, p={p_sp:.4f}")
print(f"                 Pearson  r={r_pe:.4f}, p={p_pe:.4f}")

# Also report beta_1
r_sp1, p_sp1 = stats.spearmanr(phq_v[valid_reg], beta1_all[valid_reg])
r_pe1, p_pe1 = stats.pearsonr(phq_v[valid_reg], beta1_all[valid_reg])
print(f"  PHQ vs beta_1: Spearman r={r_sp1:.4f}, p={p_sp1:.4f}")
print(f"                 Pearson  r={r_pe1:.4f}, p={p_pe1:.4f}")

# ── Figure: regression analysis ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Example quadratic fits for a few low/high PHQ participants
ax = axes[0]
h_grid = np.linspace(0, 2.5, 100)
# Pick a few from each group
for indices, color, label in [
    (idx_low_group,  COL_LOW,  "Low PHQ"),
    (idx_high_group, COL_HIGH, "High PHQ"),
]:
    for i, pi in enumerate(indices[:5]):  # first 5 from each group
        if np.isfinite(beta2_all[pi]):
            y_fit = beta0_all[pi] + beta1_all[pi]*h_grid + beta2_all[pi]*h_grid**2
            ax.plot(h_grid, y_fit, color=color, alpha=0.4, lw=1.2,
                    label=label if i == 0 else None)
    # Group mean fit
    valid_grp = np.array([pi for pi in indices if np.isfinite(beta2_all[pi])])
    if len(valid_grp) > 0:
        m_b0 = beta0_all[valid_grp].mean()
        m_b1 = beta1_all[valid_grp].mean()
        m_b2 = beta2_all[valid_grp].mean()
        y_mean = m_b0 + m_b1*h_grid + m_b2*h_grid**2
        ax.plot(h_grid, y_mean, color=color, lw=3, ls="--",
                label=f"{label} mean", zorder=5)
ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Previous entropy (nats)", fontsize=11)
ax.set_ylabel("Delta entropy (nats)", fontsize=11)
ax.set_title("A) Quadratic fits: delta_H = b0 + b1*H + b2*H²",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: PHQ vs beta_2
ax = axes[1]
sc = ax.scatter(phq_v[valid_reg], beta2_all[valid_reg],
                c=phq_v[valid_reg], cmap="coolwarm",
                s=40, alpha=0.7, edgecolors="none", zorder=3)
# Highlight extreme groups
for indices, color in [(idx_low_group, COL_LOW), (idx_high_group, COL_HIGH)]:
    mask_grp = np.isin(np.arange(len(emb_v)), indices) & valid_reg
    ax.scatter(phq_v[mask_grp], beta2_all[mask_grp],
               edgecolors=color, facecolors="none", s=80, lw=1.5, zorder=4)
# OLS fit line
x_fit = np.linspace(phq_v[valid_reg].min(), phq_v[valid_reg].max(), 200)
slope, intercept, *_ = stats.linregress(phq_v[valid_reg], beta2_all[valid_reg])
ax.plot(x_fit, slope*x_fit + intercept, color="black", lw=2, ls="--", zorder=5)
plt.colorbar(sc, ax=ax, label="PHQ score")
ax.set_xlabel("PHQ score (mean item)", fontsize=11)
ax.set_ylabel("β₂ (quadratic coefficient)", fontsize=11)
ax.set_title(f"B) PHQ vs β₂\n"
             f"Spearman r={r_sp:.3f}, p={p_sp:.4f} | "
             f"Pearson r={r_pe:.3f}, p={p_pe:.4f}",
             fontweight="bold", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel C: PHQ vs beta_1
ax = axes[2]
sc2 = ax.scatter(phq_v[valid_reg], beta1_all[valid_reg],
                 c=phq_v[valid_reg], cmap="coolwarm",
                 s=40, alpha=0.7, edgecolors="none", zorder=3)
for indices, color in [(idx_low_group, COL_LOW), (idx_high_group, COL_HIGH)]:
    mask_grp = np.isin(np.arange(len(emb_v)), indices) & valid_reg
    ax.scatter(phq_v[mask_grp], beta1_all[mask_grp],
               edgecolors=color, facecolors="none", s=80, lw=1.5, zorder=4)
slope1, intercept1, *_ = stats.linregress(phq_v[valid_reg], beta1_all[valid_reg])
ax.plot(x_fit, slope1*x_fit + intercept1, color="black", lw=2, ls="--", zorder=5)
plt.colorbar(sc2, ax=ax, label="PHQ score")
ax.set_xlabel("PHQ score (mean item)", fontsize=11)
ax.set_ylabel("β₁ (linear coefficient)", fontsize=11)
ax.set_title(f"C) PHQ vs β₁\n"
             f"Spearman r={r_sp1:.3f}, p={p_sp1:.4f} | "
             f"Pearson r={r_pe1:.3f}, p={p_pe1:.4f}",
             fontweight="bold", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Quadratic entropy-update regression — PHQ relationship\n"
    f"delta_H = β₀ + β₁·H_prev + β₂·H_prev² (unrewarded, H∈[{H_LO},{H_HI}])\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds, n={valid_reg.sum()} participants",
    fontsize=12, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_entropy_regression_vs_phq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Per-participant mean delta entropy (full entropy range) vs PHQ.
# Separate scatter / correlation for rewarded vs unrewarded trials, all
# participants.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("Mean delta entropy (full range) vs PHQ — all participants")
print("(reusing rollouts from the regression section above)")

# Correlations
mask_u = np.isfinite(mean_dH_unrew) & np.isfinite(phq_v)
mask_r = np.isfinite(mean_dH_rew)   & np.isfinite(phq_v)

r_sp_u, p_sp_u = stats.spearmanr(phq_v[mask_u], mean_dH_unrew[mask_u])
r_pe_u, p_pe_u = stats.pearsonr(phq_v[mask_u],  mean_dH_unrew[mask_u])
r_sp_r, p_sp_r = stats.spearmanr(phq_v[mask_r], mean_dH_rew[mask_r])
r_pe_r, p_pe_r = stats.pearsonr(phq_v[mask_r],  mean_dH_rew[mask_r])

print(f"\nUnrewarded — n={mask_u.sum()}")
print(f"  Spearman r={r_sp_u:.4f}, p={p_sp_u:.4f}")
print(f"  Pearson  r={r_pe_u:.4f}, p={p_pe_u:.4f}")
print(f"Rewarded — n={mask_r.sum()}")
print(f"  Spearman r={r_sp_r:.4f}, p={p_sp_r:.4f}")
print(f"  Pearson  r={r_pe_r:.4f}, p={p_pe_r:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, vals, mask, r_sp, p_sp, r_pe, p_pe, title in [
    (axes[0], mean_dH_unrew, mask_u, r_sp_u, p_sp_u, r_pe_u, p_pe_u, "Unrewarded"),
    (axes[1], mean_dH_rew,   mask_r, r_sp_r, p_sp_r, r_pe_r, p_pe_r, "Rewarded"),
]:
    sc = ax.scatter(phq_v[mask], vals[mask], c=phq_v[mask], cmap="coolwarm",
                    s=40, alpha=0.75, edgecolors="none", zorder=3)
    # OLS fit line
    x_fit = np.linspace(phq_v[mask].min(), phq_v[mask].max(), 200)
    slope, intercept, *_ = stats.linregress(phq_v[mask], vals[mask])
    ax.plot(x_fit, slope*x_fit + intercept, color="black", lw=2, ls="--", zorder=5)
    # Highlight extreme groups
    for indices, color in [(idx_low_group, COL_LOW), (idx_high_group, COL_HIGH)]:
        mask_grp = np.isin(np.arange(len(emb_v)), indices) & mask
        ax.scatter(phq_v[mask_grp], vals[mask_grp],
                   edgecolors=color, facecolors="none", s=80, lw=1.5, zorder=4)
    ax.axhline(0, color="grey", ls=":", lw=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="PHQ score")
    ax.set_xlabel("PHQ score (mean item)", fontsize=11)
    ax.set_ylabel("Mean delta entropy (nats)", fontsize=11)
    ax.set_title(f"{title}\n"
                 f"Spearman r={r_sp:.3f}, p={p_sp:.4f} | "
                 f"Pearson r={r_pe:.3f}, p={p_pe:.4f}",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Mean delta entropy across full range vs PHQ — all participants\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds, "
    f"n={mask_u.sum()} participants",
    fontsize=12, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_mean_delta_entropy_vs_phq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
'''  # end of disabled-sections block (analyses 1-4)

# ══════════════════════════════════════════════════════════════════════════════
# Compressibility (LZW) — humans vs IDRNN vs Vanilla
# Compressibility score = b_LZW / l_LZW, where
#   l_LZW = LZW-compressed length of the choice sequence
#   b_LZW = mean LZW-compressed length of random K-symbol sequences of equal
#           length (baseline expected by chance).
# Higher score → more compressible / more structured.
# ══════════════════════════════════════════════════════════════════════════════
from modelsandtraining import AblatedRNN
from scipy import stats


print(f"\n{'='*70}")
print("Compressibility (LZW): humans vs IDRNN vs Vanilla")

def lzw_length(seq, alphabet_size):
    """LZW compressed length (number of output codes) for an integer sequence."""
    s = [int(x) for x in seq]
    table = {(c,): i for i, c in enumerate(range(alphabet_size))}
    next_code = alphabet_size
    out = 0
    w = ()
    for c in s:
        wc = w + (c,)
        if wc in table:
            w = wc
        else:
            out += 1
            table[wc] = next_code
            next_code += 1
            w = (c,)
    if w:
        out += 1
    return out

def baseline_lzw(length, alphabet_size, n_samples=500, seed=0):
    """Average LZW length for random sequences of given length / alphabet."""
    rng = np.random.default_rng(seed)
    lens = []
    for _ in range(n_samples):
        seq = rng.integers(0, alphabet_size, size=length)
        lens.append(lzw_length(seq, alphabet_size))
    return float(np.mean(lens))

# Baseline once (length matches simulator T = 200 for IDRNN/Vanilla,
# and we'll recompute for the human sequence length as needed)
SIM_LEN = N_TRIALS  # 200
b_sim   = baseline_lzw(SIM_LEN, A, n_samples=500, seed=0)
print(f"  baseline LZW length (random, T={SIM_LEN}, K={A}): "
      f"{b_sim:.2f}")

# ── Human compressibility (restless task choice sequence per participant) ────
print("Loading human restless-task choice sequences...")
raw_restless = pd.read_csv("data/finalRestlessSession1.csv").sort_values(
    ["ID", "trial"])
human_scores = {}
for sid, grp in raw_restless.groupby("ID"):
    ch = grp["chosen"].values.astype(int)
    if len(ch) < 50:
        continue
    l_h = lzw_length(ch, A)
    b_h = baseline_lzw(len(ch), A, n_samples=500, seed=hash(int(sid)) & 0xFFFF)
    human_scores[sid] = b_h / l_h
print(f"  Human participants with restless data: {len(human_scores)}")

human_aligned = np.array([human_scores.get(int(sid), np.nan)
                          for sid in subids_v])

# Cache for per-participant compressibility scores. Loaded if present so we
# can skip the (slow) rollout passes when iterating on downstream stats/plots.
COMPRESS_CACHE = os.path.join(PLOT_DIR, "step1_compressibility_scores.npz")
_compress_cache_hit = False
if os.path.exists(COMPRESS_CACHE):
    try:
        _cache = np.load(COMPRESS_CACHE)
        cached_subids = _cache["subids"]
        if (cached_subids.shape == subids_v.shape
                and np.array_equal(cached_subids, subids_v)
                and int(_cache["n_env_seeds"]) == N_ENV_SEEDS):
            idrnn_scores     = _cache["idrnn_scores"]
            vanilla_scores   = _cache["vanilla_scores"]
            vanilla_h_scores = _cache["vanilla_h_scores"]
            human_aligned    = _cache["human_aligned"]
            _compress_cache_hit = True
            print(f"Loaded cached compressibility scores from {COMPRESS_CACHE} "
                  f"(skipping rollouts).")
        else:
            print(f"Cache exists at {COMPRESS_CACHE} but subids/N_ENV_SEEDS "
                  "differ — recomputing.")
    except Exception as e:
        print(f"Failed to load cache ({e}); recomputing.")

if not _compress_cache_hit:
    # ── IDRNN compressibility (on-policy rollouts) ───────────────────────────
    print("Computing IDRNN compressibility (on-policy rollouts)...")
    idrnn_scores = np.full(len(emb_v), np.nan)
    for pi in range(len(emb_v)):
        z_vec = emb_v[pi]
        rollout_lens = []
        for env_seed in range(N_ENV_SEEDS):
            sched = make_env(N_TRIALS, seed=env_seed)
            rng = np.random.default_rng(env_seed * 10000 + pi)
            ch, _, _ = rollout_logits(z_vec, sched, task_id, rng)
            rollout_lens.append(lzw_length(ch, A))
        mean_l = float(np.mean(rollout_lens))
        idrnn_scores[pi] = b_sim / mean_l
        if (pi + 1) % 50 == 0:
            print(f"  IDRNN: {pi+1}/{len(emb_v)} done")

# ── Vanilla compressibility ──────────────────────────────────────────────────
import glob as _glob
_vanilla_glob = _glob.glob(
    "plots_thalmann/step1_vs_vanilla/latents_vanilla_bestseed*.pt")
assert _vanilla_glob, ("No vanilla latents file found — re-run "
                       "train_and_decode_thalmann_step1.py")
VANILLA_PATH = max(_vanilla_glob, key=os.path.getmtime)
print(f"Loading vanilla model: {VANILLA_PATH}")
v_saved = torch.load(VANILLA_PATH, map_location=DEVICE, weights_only=False)
assert "model_state" in v_saved, (
    "Vanilla latents file lacks model_state — re-run training script after "
    "the save-state update.")
V_HIDDEN       = v_saved["hidden"]
V_TASK_EMB_DIM = v_saved["task_emb_dim"]
V_A            = v_saved["A"]
V_BASE_IN_DIM  = v_saved["base_in_dim"]
V_DEC_IN_DIM   = V_BASE_IN_DIM + V_TASK_EMB_DIM
print(f"  hidden={V_HIDDEN}, task_emb_dim={V_TASK_EMB_DIM}, "
      f"dec_in_dim={V_DEC_IN_DIM}")

vanilla_model = AblatedRNN(
    hid=V_HIDDEN, in_dim=V_DEC_IN_DIM, A=V_A,
    block_structure=False,
    n_tasks=2, task_emb_dim=V_TASK_EMB_DIM,
)
vanilla_model.load_state_dict(v_saved["model_state"])
vanilla_model.to(DEVICE).eval()
v_task_emb_w = v_saved["model_state"]["task_embedding.weight"].to(DEVICE)

v_h_per_subj = np.asarray(v_saved["h"])  # (236, hidden) avg hidden per subj
v_subids     = np.asarray(v_saved["subids"])
assert v_h_per_subj.shape[0] == len(v_subids), "vanilla h vs subids mismatch"

# Reorder vanilla per-subject hidden states to match the IDRNN subids ordering
sid_to_idx = {int(s): i for i, s in enumerate(v_subids)}
v_h_aligned = np.full((len(subids_v), V_HIDDEN), np.nan, dtype=np.float32)
for i, sid in enumerate(subids_v):
    if int(sid) in sid_to_idx:
        v_h_aligned[i] = v_h_per_subj[sid_to_idx[int(sid)]]

@torch.no_grad()
def vanilla_rollout(reward_schedule, task_id, rng, h_init=None):
    """On-policy rollout for vanilla model.
    reward_schedule: (T, K) continuous raw rewards. Input feature 4 receives
    reward / REWARD_MAX (matches training).
    h_init: optional (V_HIDDEN,) — if provided, used as initial hidden state."""
    T = reward_schedule.shape[0]
    temb = v_task_emb_w[task_id].detach().cpu().numpy()
    if h_init is None:
        h = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
    else:
        h = torch.tensor(h_init, dtype=torch.float32,
                         device=DEVICE).reshape(1, 1, V_HIDDEN)
    prev = np.concatenate([np.zeros(V_BASE_IN_DIM, dtype=np.float32), temb])
    choices = []
    for t in range(T):
        x = torch.tensor(prev, dtype=torch.float32,
                         device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h, _ = vanilla_model.dec(x, h0=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm        = rng.choice(V_A, p=p)
        reward_raw = float(reward_schedule[t, arm])
        reward_in  = reward_raw / REWARD_MAX
        oh = np.zeros(V_A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [reward_in], temb])
        choices.append(arm)
    return np.array(choices)

if not _compress_cache_hit:
    print("Computing Vanilla compressibility (on-policy rollouts, zero-init)...")
    # Vanilla zero-init: no subject conditioning, only RNG differs across pi.
    vanilla_scores = np.full(len(emb_v), np.nan)
    # Vanilla per-subject init: hidden state initialized to that participant's
    # averaged training hidden state, otherwise identical to vanilla.
    vanilla_h_scores = np.full(len(emb_v), np.nan)

    for pi in range(len(emb_v)):
        h_init = v_h_aligned[pi] if np.all(np.isfinite(v_h_aligned[pi])) else None
        lens0 = []
        lensH = []
        for env_seed in range(N_ENV_SEEDS):
            sched = make_env(N_TRIALS, seed=env_seed)
            rng0 = np.random.default_rng(env_seed * 10000 + pi)
            chH_rng = np.random.default_rng(env_seed * 10000 + pi + 1)  # decoupled RNG
            ch0 = vanilla_rollout(sched, task_id, rng0, h_init=None)
            lens0.append(lzw_length(ch0, A))
            if h_init is not None:
                chH = vanilla_rollout(sched, task_id, chH_rng, h_init=h_init)
                lensH.append(lzw_length(chH, A))
        vanilla_scores[pi] = b_sim / float(np.mean(lens0))
        if lensH:
            vanilla_h_scores[pi] = b_sim / float(np.mean(lensH))
        if (pi + 1) % 50 == 0:
            print(f"  Vanilla: {pi+1}/{len(emb_v)} done")

    # Save cache so subsequent runs can skip the rollouts above
    np.savez(COMPRESS_CACHE,
             subids=subids_v,
             n_env_seeds=N_ENV_SEEDS,
             idrnn_scores=idrnn_scores,
             vanilla_scores=vanilla_scores,
             vanilla_h_scores=vanilla_h_scores,
             human_aligned=human_aligned)
    print(f"Saved compressibility scores → {COMPRESS_CACHE}")

# ── Stats ────────────────────────────────────────────────────────────────────
def _summary(name, vals):
    v = vals[np.isfinite(vals)]
    print(f"  {name:10s}  n={len(v):3d}  mean={v.mean():.4f}  "
          f"median={np.median(v):.4f}  std={v.std():.4f}")

print("\nCompressibility scores (b_LZW / l_LZW):")
_summary("Humans",    human_aligned)
_summary("IDRNN",     idrnn_scores)
_summary("Vanilla+h", vanilla_h_scores)
_summary("Vanilla",   vanilla_scores)

# Pairwise tests
def _pair(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    return t, p, p_u, d

for n1, v1, n2, v2 in [
    ("Humans",    human_aligned,    "IDRNN",     idrnn_scores),
    ("Humans",    human_aligned,    "Vanilla+h", vanilla_h_scores),
    ("Humans",    human_aligned,    "Vanilla",   vanilla_scores),
    ("IDRNN",     idrnn_scores,     "Vanilla+h", vanilla_h_scores),
    ("IDRNN",     idrnn_scores,     "Vanilla",   vanilla_scores),
    ("Vanilla+h", vanilla_h_scores, "Vanilla",   vanilla_scores),
]:
    t, p, p_u, d = _pair(v1, v2)
    print(f"  {n1:10s} vs {n2:10s}: Welch t={t:.2f}, p={p:.2e}  "
          f"MWU p={p_u:.2e}  Cohen's d={d:.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

groups = ["Humans", "IDRNN", "Vanilla+h", "Vanilla"]
data   = [human_aligned[np.isfinite(human_aligned)],
          idrnn_scores[np.isfinite(idrnn_scores)],
          vanilla_h_scores[np.isfinite(vanilla_h_scores)],
          vanilla_scores[np.isfinite(vanilla_scores)]]
COLORS = ["#666666", "#4C72B0", "#8C564B", "#DD8452"]

bp = ax.boxplot(data, positions=range(len(groups)), widths=0.5,
                patch_artist=True, showfliers=False, zorder=2)
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.4)
for elem in ["whiskers", "caps", "medians"]:
    for line in bp[elem]:
        line.set_color("black")

# Overlay individual points
for i, (vals, c) in enumerate(zip(data, COLORS)):
    jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
    ax.scatter(i + jitter, vals, color=c, s=25, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)

ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.6,
           label="No structure (= baseline)")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(groups, data)],
                    fontsize=11)
ax.set_ylabel("Compressibility score (b_LZW / l_LZW)", fontsize=12)
ax.set_title(
    f"Sequence compressibility — {N_TRIALS}-trial 4-armed restless bandit\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds  |  "
    f"'Vanilla+h' = per-subject avg-train hidden init",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_compressibility.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Per-participant correlation: human compressibility vs model ──────────────
print("\nPer-participant correlations: human vs model compressibility")
corr_results = {}
for label, model_vals in [("IDRNN",     idrnn_scores),
                          ("Vanilla+h", vanilla_h_scores),
                          ("Vanilla",   vanilla_scores)]:
    m = np.isfinite(human_aligned) & np.isfinite(model_vals)
    if m.sum() < 5:
        print(f"  {label}: insufficient data (n={m.sum()})")
        continue
    r_sp, p_sp = stats.spearmanr(human_aligned[m], model_vals[m])
    r_pe, p_pe = stats.pearsonr(human_aligned[m],  model_vals[m])
    corr_results[label] = (model_vals, m, r_sp, p_sp, r_pe, p_pe)
    print(f"  Humans vs {label:10s}  n={m.sum():3d}  "
          f"Spearman r={r_sp:+.3f} p={p_sp:.3e}  |  "
          f"Pearson r={r_pe:+.3f} p={p_pe:.3e}")

# ── Plot scatter of human vs model compressibility ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label, color) in zip(axes,
                              [("IDRNN",     "#4C72B0"),
                               ("Vanilla+h", "#8C564B"),
                               ("Vanilla",   "#DD8452")]):
    if label not in corr_results:
        ax.text(0.5, 0.5, f"No data for {label}",
                transform=ax.transAxes, ha="center", va="center")
        continue
    model_vals, m, r_sp, p_sp, r_pe, p_pe = corr_results[label]
    x = human_aligned[m]; y = model_vals[m]
    ax.scatter(x, y, color=color, s=35, alpha=0.6,
               edgecolors="black", linewidths=0.3, zorder=3)
    # OLS fit
    slope, intercept, *_ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, slope*x_fit + intercept, color="black", lw=2, ls="--",
            zorder=4, label="OLS fit")
    # Identity
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1, alpha=0.5,
            label="Identity")
    ax.set_xlabel("Human compressibility (b_LZW / l_LZW)", fontsize=11)
    ax.set_ylabel(f"{label} compressibility (b_LZW / l_LZW)", fontsize=11)
    ax.set_title(f"Humans vs {label}  (n={m.sum()})\n"
                 f"Spearman r={r_sp:+.3f}, p={p_sp:.3e}\n"
                 f"Pearson  r={r_pe:+.3f}, p={p_pe:.3e}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Per-participant compressibility: humans vs models\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_ENV_SEEDS} env seeds",
    fontsize=12, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_compressibility_humans_vs_models.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Compressibility × all questionnaire scales (per-subject correlations)
# FDR-BH corrected across (scale × model) within each correlation type.
# ══════════════════════════════════════════════════════════════════════════════
from statsmodels.stats.multitest import multipletests

print(f"\n{'='*70}")
print("Compressibility vs all questionnaire scales")

# Define scales — items as in train_and_decode_thalmann_step1.py, plus motivation.
SCALES = {
    "PANAS_PA":   [f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]],
    "PANAS_NA":   [f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]],
    "STICSA":     [f"STICSA_{i}" for i in range(22)],
    "PHQ":        [f"PHQ_9_{i}"  for i in range(10)],
    "CEI":        [f"CEI_{i}"    for i in range(4)],
    "BIG5_open":  [f"BIG_5_{i}"  for i in range(6)],
    "Motiv_slot": ["motiv_slot_0"],
    "Motiv_mem":  ["motiv_mem_0"],
}
for name, items in SCALES.items():
    missing = [c for c in items if c not in quest.columns]
    assert not missing, f"Scale {name}: missing columns {missing}"
    quest[name] = quest[items].mean(axis=1)

scale_names = list(SCALES.keys())

# Score arrays (already aligned to subids_v ordering)
score_arrays = {
    "Humans":    human_aligned,
    "IDRNN":     idrnn_scores,
    "Vanilla+h": vanilla_h_scores,
    "Vanilla":   vanilla_scores,
}

# Compute correlations
records = []  # one per (scale, model, kind)
for scale in scale_names:
    y = quest.reindex(subids_v)[scale].values.astype(float)
    for model_name, scores in score_arrays.items():
        m = np.isfinite(scores) & np.isfinite(y)
        if m.sum() < 10:
            records.append({"scale": scale, "model": model_name,
                            "n": int(m.sum()),
                            "r_sp": np.nan, "p_sp": np.nan,
                            "r_pe": np.nan, "p_pe": np.nan})
            continue
        r_sp, p_sp = stats.spearmanr(scores[m], y[m])
        r_pe, p_pe = stats.pearsonr(scores[m],  y[m])
        records.append({"scale": scale, "model": model_name,
                        "n": int(m.sum()),
                        "r_sp": float(r_sp), "p_sp": float(p_sp),
                        "r_pe": float(r_pe), "p_pe": float(p_pe)})
corr_df = pd.DataFrame(records)

# FDR-BH across (scale × model) per correlation kind, separately
def _fdr(p):
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if valid.any():
        _, p_corr, _, _ = multipletests(p[valid], method="fdr_bh")
        out[valid] = p_corr
    return out

corr_df["p_sp_fdr"] = _fdr(corr_df["p_sp"].values)
corr_df["p_pe_fdr"] = _fdr(corr_df["p_pe"].values)

# Save table
corr_csv = os.path.join(PLOT_DIR, "step1_compressibility_questionnaire_corr.csv")
corr_df.to_csv(corr_csv, index=False)
print(f"Saved table → {corr_csv}")

# Print sorted by absolute Spearman r
print("\nTop correlations (by |Spearman r|):")
hdr = (f"{'Scale':<12}{'Model':<12}{'n':>4}  "
       f"{'r_sp':>7}  {'p_sp':>9}  {'p_sp_FDR':>9}  "
       f"{'r_pe':>7}  {'p_pe':>9}  {'p_pe_FDR':>9}")
print(hdr)
print("-" * len(hdr))
for _, row in corr_df.reindex(
        corr_df["r_sp"].abs().sort_values(ascending=False).index).iterrows():
    star_sp = ("***" if row["p_sp_fdr"] < 0.001
               else "**" if row["p_sp_fdr"] < 0.01
               else "*"  if row["p_sp_fdr"] < 0.05
               else "")
    print(f"{row['scale']:<12}{row['model']:<12}{int(row['n']):>4}  "
          f"{row['r_sp']:>+7.3f}  {row['p_sp']:>9.2e}  "
          f"{row['p_sp_fdr']:>9.2e}{star_sp:<3}  "
          f"{row['r_pe']:>+7.3f}  {row['p_pe']:>9.2e}  "
          f"{row['p_pe_fdr']:>9.2e}")

# ── Heatmap of Spearman r (rows=scales, cols=models), FDR-significant marked ─
models = ["Humans", "IDRNN", "Vanilla+h", "Vanilla"]
heat_r  = np.full((len(scale_names), len(models)), np.nan)
heat_p  = np.full((len(scale_names), len(models)), np.nan)
heat_pf = np.full((len(scale_names), len(models)), np.nan)
for i, s in enumerate(scale_names):
    for j, m in enumerate(models):
        row = corr_df[(corr_df["scale"] == s) & (corr_df["model"] == m)]
        if len(row):
            heat_r[i, j]  = row["r_sp"].iloc[0]
            heat_p[i, j]  = row["p_sp"].iloc[0]
            heat_pf[i, j] = row["p_sp_fdr"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Spearman heatmap
ax = axes[0]
vmax = max(0.001, np.nanmax(np.abs(heat_r)))
im = ax.imshow(heat_r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
for i in range(len(scale_names)):
    for j in range(len(models)):
        if not np.isfinite(heat_r[i, j]):
            continue
        star = ("***" if heat_pf[i, j] < 0.001
                else "**" if heat_pf[i, j] < 0.01
                else "*"  if heat_pf[i, j] < 0.05
                else "")
        ax.text(j, i, f"{heat_r[i, j]:+.2f}{star}",
                ha="center", va="center", fontsize=10,
                color="white" if abs(heat_r[i, j]) > vmax * 0.6 else "black")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_yticks(range(len(scale_names)))
ax.set_yticklabels(scale_names, fontsize=10)
ax.set_title("Spearman r — compressibility × scale\n"
             "* p<0.05, ** p<0.01, *** p<0.001 (FDR-BH within table)",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Spearman r")

# Bar plot grouped by scale
ax = axes[1]
x = np.arange(len(scale_names))
width = 0.2
COLORS_M = {"Humans": "#666666", "IDRNN": "#4C72B0",
            "Vanilla+h": "#8C564B", "Vanilla": "#DD8452"}
for k, m in enumerate(models):
    vals = heat_r[:, k]
    ax.bar(x + (k - 1.5) * width, vals, width=width, color=COLORS_M[m],
           label=m, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scale_names, rotation=30, fontsize=9, ha="right")
ax.set_ylabel("Spearman r", fontsize=11)
ax.set_title("Spearman r per scale × model", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Compressibility correlations with questionnaire scales\n"
    f"all-subjects step-1 seed {BEST_SEED}, n={(np.isfinite(human_aligned)).sum()} "
    f"participants — FDR-BH across {len(scale_names)*len(models)} tests",
    fontsize=12, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_compressibility_questionnaire_corr.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side: latent decoding vs compressibility correlation per scale
#   Latent decoding = Spearman r between LOO ridge predictions and the scale,
#                     using IDRNN z (z_dim) or Vanilla h (hidden_dim) as features.
#   Compressibility = Spearman r between per-subject compressibility score and
#                     the scale (already computed above).
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression
from scipy.stats import f as f_dist

print(f"\n{'='*70}")
print("In-sample OLS latent decoding (matches step1_vs_vanilla_decoding.png) "
      "vs compressibility correlation")

def insample_ols(Z, y):
    """Mirrors train_and_decode_thalmann_step1.py:insample_ols.
    In-sample OLS fit on ALL data, returns
      r     = sqrt(R²)             — multiple correlation coefficient (>=0)
      p     = F-test p-value        — null: no linear combo of Z predicts y
      n     = sample size used
    """
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    Zv, yv = Z[m], y[m]
    n, p = Zv.shape
    if n < p + 2:
        return np.nan, np.nan, int(m.sum())
    clf   = LinearRegression().fit(Zv, yv)
    preds = clf.predict(Zv)
    ss_res = np.sum((yv - preds) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F      = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val  = float(f_dist.sf(F, dfn, dfd))
    return float(np.sqrt(max(r2, 0.0))), p_val, int(m.sum())

# Build feature matrices (aligned to subids_v ordering — same as score arrays)
Z_idrnn   = emb_v                 # (N, z_dim) — saved["z"], already aligned
Z_vanilla = v_h_aligned           # (N, hidden) — vanilla h, aligned to subids_v

dec_records = []
for scale in scale_names:
    y = quest.reindex(subids_v)[scale].values.astype(float)
    for feat_name, Z in [("IDRNN_z", Z_idrnn), ("Vanilla_h", Z_vanilla)]:
        r, p, n = insample_ols(Z, y)
        dec_records.append({
            "scale": scale, "features": feat_name, "n": n,
            "r": r, "p": p,
        })
dec_df = pd.DataFrame(dec_records)
dec_df["p_fdr"] = _fdr(dec_df["p"].values)

dec_csv = os.path.join(PLOT_DIR, "step1_latent_decoding_questionnaire_corr.csv")
dec_df.to_csv(dec_csv, index=False)
print(f"Saved table → {dec_csv}")

# Print summary
print("\nLatent decoding (in-sample OLS → questionnaire):")
hdr = f"{'Scale':<12}{'Feat':<11}{'n':>4}  {'r':>7}  {'p':>9}  {'p_FDR':>9}"
print(hdr); print("-" * len(hdr))
for _, row in dec_df.iterrows():
    s = ("***" if row["p_fdr"] < 0.001
         else "**" if row["p_fdr"] < 0.01
         else "*"  if row["p_fdr"] < 0.05
         else "")
    print(f"{row['scale']:<12}{row['features']:<11}{int(row['n']):>4}  "
          f"{row['r']:>+7.3f}  {row['p']:>9.2e}  "
          f"{row['p_fdr']:>9.2e}{s}")

# ── Side-by-side plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

x = np.arange(len(scale_names))
width = 0.4

# Panel A: Latent decoding (in-sample OLS multiple correlation = sqrt(R²))
ax = axes[0]
for k, (feat, color) in enumerate([("IDRNN_z",   "#4C72B0"),
                                    ("Vanilla_h", "#DD8452")]):
    sub = dec_df[dec_df["features"] == feat].set_index("scale").reindex(scale_names)
    vals = sub["r"].values
    bars = ax.bar(x + (k - 0.5) * width, vals, width=width, color=color,
                  label=feat, edgecolor="black", linewidth=0.5)
    # Annotate FDR-significant bars
    for i, (v, pf) in enumerate(zip(vals, sub["p_fdr"].values)):
        if np.isfinite(pf):
            s = ("***" if pf < 0.001 else "**" if pf < 0.01
                 else "*" if pf < 0.05 else "")
            if s:
                ax.text(x[i] + (k - 0.5) * width,
                        v + 0.01 * np.sign(v if v != 0 else 1),
                        s, ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=10, fontweight="bold")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scale_names, rotation=30, fontsize=9, ha="right")
ax.set_ylabel("Multiple correlation r  (in-sample OLS, =√R²)", fontsize=11)
ax.set_title("A) Latent decoding\n(OLS, all-data fit: latent → questionnaire)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel B: Compressibility correlation (re-using corr_df from above)
ax = axes[1]
for k, (model_name, color) in enumerate([("IDRNN",     "#4C72B0"),
                                          ("Vanilla+h", "#8C564B"),
                                          ("Vanilla",   "#DD8452")]):
    sub = corr_df[corr_df["model"] == model_name].set_index("scale").reindex(scale_names)
    vals = sub["r_sp"].values
    offset = (k - 1) * (width * 0.66)
    ax.bar(x + offset, vals, width=width * 0.66, color=color,
           label=model_name, edgecolor="black", linewidth=0.5)
    for i, (v, pf) in enumerate(zip(vals, sub["p_sp_fdr"].values)):
        if np.isfinite(pf):
            s = ("***" if pf < 0.001 else "**" if pf < 0.01
                 else "*" if pf < 0.05 else "")
            if s:
                ax.text(x[i] + offset,
                        v + 0.01 * np.sign(v if v != 0 else 1),
                        s, ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=9, fontweight="bold")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scale_names, rotation=30, fontsize=9, ha="right")
ax.set_title("B) Compressibility correlation\n(per-subject score vs questionnaire)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Latent decoding vs compressibility correlation, all questionnaire scales\n"
    f"all-subjects step-1 seed {BEST_SEED}, n={(np.isfinite(human_aligned)).sum()} "
    f"participants — FDR-BH within each panel",
    fontsize=12, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_latent_vs_compressibility_corr.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Variance decomposition: predict each scale from (a) latents only,
# (b) compressibility only, (c) latents + compressibility — for IDRNN + Vanilla.
# In-sample OLS R² (and adjusted R²); also F-test for whether adding the
# compressibility column on top of the latents improves the fit.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Variance decomposition: latents vs compressibility vs both")

def ols_r2(Z, y):
    """In-sample OLS. Returns (R2, adj_R2, F-test p, n, p_features)."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    Zv, yv = Z[m], y[m]
    n, p = Zv.shape
    if n < p + 2:
        return np.nan, np.nan, np.nan, int(m.sum()), p
    clf   = LinearRegression().fit(Zv, yv)
    preds = clf.predict(Zv)
    ss_res = np.sum((yv - preds) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    adj    = 1.0 - (1 - r2) * (n - 1) / (n - p - 1)
    dfn, dfd = p, n - p - 1
    F      = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val  = float(f_dist.sf(F, dfn, dfd))
    return float(r2), float(adj), p_val, int(m.sum()), p

def f_test_nested(Z_full, Z_red, y):
    """Partial F-test: does adding the extra cols in Z_full vs Z_red help?"""
    m = (np.isfinite(y) &
         np.all(np.isfinite(Z_full), axis=1) &
         np.all(np.isfinite(Z_red),  axis=1))
    yv = y[m]
    Zf, Zr = Z_full[m], Z_red[m]
    n = len(yv)
    pf, pr = Zf.shape[1], Zr.shape[1]
    rss_full = np.sum((yv - LinearRegression().fit(Zf, yv).predict(Zf)) ** 2)
    rss_red  = np.sum((yv - LinearRegression().fit(Zr, yv).predict(Zr)) ** 2)
    df1 = pf - pr
    df2 = n - pf - 1
    if df1 <= 0 or df2 <= 0 or rss_full <= 0:
        return np.nan, np.nan
    F = ((rss_red - rss_full) / df1) / (rss_full / df2)
    return float(F), float(f_dist.sf(F, df1, df2))

# Pair each model with its corresponding latents and compressibility scores.
# Vanilla pairs h-latents with Vanilla+h compressibility (both use h).
model_specs = [
    ("IDRNN",   Z_idrnn,   idrnn_scores),
    ("Vanilla", Z_vanilla, vanilla_h_scores),
]

vd_records = []
for model_name, Zf, comp in model_specs:
    for scale in scale_names:
        y = quest.reindex(subids_v)[scale].values.astype(float)
        Z_lat  = Zf
        Z_comp = comp.reshape(-1, 1)
        # Drop rows with any NaN in any of the inputs we'll use
        keep = (np.isfinite(y) &
                np.all(np.isfinite(Z_lat), axis=1) &
                np.isfinite(comp))
        if keep.sum() < Z_lat.shape[1] + 3:
            print(f"  {model_name} {scale}: too few valid rows ({keep.sum()})")
            continue
        Z_lat_k  = Z_lat[keep]
        Z_comp_k = Z_comp[keep]
        Z_both_k = np.column_stack([Z_lat_k, Z_comp_k])
        y_k      = y[keep]

        r2_lat,  adj_lat,  p_lat,  n_lat,  _ = ols_r2(Z_lat_k,  y_k)
        r2_comp, adj_comp, p_comp, n_comp, _ = ols_r2(Z_comp_k, y_k)
        r2_both, adj_both, p_both, n_both, _ = ols_r2(Z_both_k, y_k)

        # Incremental F-tests: does each addition improve fit?
        F_inc_comp, p_inc_comp = f_test_nested(Z_both_k, Z_lat_k,  y_k)
        F_inc_lat,  p_inc_lat  = f_test_nested(Z_both_k, Z_comp_k, y_k)

        vd_records.append({
            "model": model_name, "scale": scale, "n": n_both,
            "r2_lat":  r2_lat,  "adj_lat":  adj_lat,  "p_lat":  p_lat,
            "r2_comp": r2_comp, "adj_comp": adj_comp, "p_comp": p_comp,
            "r2_both": r2_both, "adj_both": adj_both, "p_both": p_both,
            "F_inc_comp": F_inc_comp, "p_inc_comp": p_inc_comp,
            "F_inc_lat":  F_inc_lat,  "p_inc_lat":  p_inc_lat,
        })
vd_df = pd.DataFrame(vd_records)

vd_csv = os.path.join(PLOT_DIR, "step1_variance_decomposition.csv")
vd_df.to_csv(vd_csv, index=False)
print(f"Saved table → {vd_csv}")

# Summary print
print("\nVariance decomposition (R² in-sample):")
hdr = (f"{'Model':<8}{'Scale':<12}{'n':>4}  "
       f"{'R²_lat':>7}  {'R²_comp':>8}  {'R²_both':>8}  "
       f"{'adj_lat':>8}  {'adj_comp':>9}  {'adj_both':>9}  "
       f"{'F_inc_comp':>10} {'p_inc_comp':>10}")
print(hdr); print("-" * len(hdr))
for _, row in vd_df.iterrows():
    print(f"{row['model']:<8}{row['scale']:<12}{int(row['n']):>4}  "
          f"{row['r2_lat']:>7.3f}  {row['r2_comp']:>8.3f}  "
          f"{row['r2_both']:>8.3f}  "
          f"{row['adj_lat']:>+8.3f}  {row['adj_comp']:>+9.3f}  "
          f"{row['adj_both']:>+9.3f}  "
          f"{row['F_inc_comp']:>10.2f} {row['p_inc_comp']:>10.2e}")

# ── Plot: grouped bar chart per model ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

x = np.arange(len(scale_names))
width = 0.27
COL_LAT  = "#4C72B0"
COL_COMP = "#DD8452"
COL_BOTH = "#55A868"

for ax_idx, model_name in enumerate(["IDRNN", "Vanilla"]):
    ax = axes[ax_idx]
    sub = vd_df[vd_df["model"] == model_name].set_index("scale").reindex(scale_names)

    # Adjusted R² bars (foreground; transparent overlay marks raw R²)
    for k, (col_adj, col_raw, color, label) in enumerate([
        ("adj_lat",  "r2_lat",  COL_LAT,  "Latents only"),
        ("adj_comp", "r2_comp", COL_COMP, "Compressibility only"),
        ("adj_both", "r2_both", COL_BOTH, "Latents + Compressibility"),
    ]):
        adj_vals = sub[col_adj].values
        raw_vals = sub[col_raw].values
        offset = (k - 1) * width
        # Raw R² as a translucent outline showing the in-sample bias
        ax.bar(x + offset, raw_vals, width=width, color=color, alpha=0.25,
               edgecolor="none", zorder=2)
        # Adjusted R² as the solid bar
        ax.bar(x + offset, adj_vals, width=width, color=color,
               edgecolor="black", linewidth=0.5, label=label, zorder=3)

    # Annotate which "both" bars beat the bigger of the two single-feature
    # bars, and mark significance of incremental compressibility (over latents)
    for i, scale in enumerate(scale_names):
        row = sub.loc[scale]
        p_inc_c = row["p_inc_comp"]
        if np.isfinite(p_inc_c):
            s = ("***" if p_inc_c < 0.001
                 else "**" if p_inc_c < 0.01
                 else "*"  if p_inc_c < 0.05
                 else "")
            if s:
                ax.text(x[i] + width, max(row["adj_both"], row["adj_lat"]) + 0.005,
                        f"+comp{s}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="black")

    ax.axhline(0, color="grey", lw=0.8, zorder=1)
    ax.set_ylabel("Adjusted R²  (raw R² shown faded)", fontsize=11)
    ax.set_title(f"{model_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(scale_names, rotation=30, fontsize=10, ha="right")

fig.suptitle(
    f"Variance decomposition: predicting questionnaire scores from latents "
    f"and / or compressibility\n"
    f"all-subjects step-1 seed {BEST_SEED}, n_max={(np.isfinite(human_aligned)).sum()} "
    f"participants  |  '+comp' stars = partial F-test for adding compressibility "
    f"on top of latents (* p<0.05, ** p<0.01, *** p<0.001)",
    fontsize=11, fontweight="bold")
fig.tight_layout()

out = os.path.join(PLOT_DIR, "step1_variance_decomposition.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Learning curves: cumulative reward over trials.
# Same rollout machinery + simulator as compressibility, but tracking the
# *continuous* reward (not the binarized one fed into the model input).
# Compares Humans (raw reward), IDRNN, Vanilla+h, Vanilla.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Learning curves: cumulative reward over trials")

N_RNG_SEEDS = N_ENV_SEEDS  # rollouts per subject (only RNG varies — env is fixed)

# Build per-participant continuous reward schedules from the actual data
human_envs   = {}    # subid → (T, K) raw counterfactual rewards
human_actual = {}    # subid → (T,)  raw reward the human actually received
for sid, sub in raw_restless.groupby("ID"):
    sub = sub.sort_values("trial")
    arms = sub[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)
    rec  = sub["reward"].values.astype(np.float32)
    if arms.shape[0] < N_TRIALS:
        continue
    human_envs[int(sid)]   = arms[:N_TRIALS]
    human_actual[int(sid)] = rec[:N_TRIALS]

@torch.no_grad()
def rollout_idrnn_reward(z_vec, sched_cont, task_id, rng):
    """Continuous reward sequence from IDRNN on-policy rollout.
    sched_cont: (T, K) raw continuous rewards (e.g. ~17–83)."""
    T = sched_cont.shape[0]
    z_t  = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[task_id].detach().cpu().numpy()
    h    = decoder.z2h0(z_t).unsqueeze(0)
    prev = np.concatenate([np.zeros(5, dtype=np.float32), temb])
    rew_cont = np.zeros(T, dtype=np.float32)
    for t in range(T):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x, z_t, hidden=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = rng.choice(A, p=p)
        rew_cont[t] = float(sched_cont[t, arm])
        oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [rew_cont[t] / REWARD_MAX], temb])
    return rew_cont

@torch.no_grad()
def rollout_vanilla_reward(sched_cont, task_id, rng, h_init=None):
    """Continuous reward sequence from vanilla on-policy rollout."""
    T = sched_cont.shape[0]
    temb = v_task_emb_w[task_id].detach().cpu().numpy()
    if h_init is None:
        h = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
    else:
        h = torch.as_tensor(h_init, dtype=torch.float32,
                            device=DEVICE).reshape(1, 1, V_HIDDEN)
    prev = np.concatenate([np.zeros(V_BASE_IN_DIM, dtype=np.float32), temb])
    rew_cont = np.zeros(T, dtype=np.float32)
    for t in range(T):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h, _ = vanilla_model.dec(x, h0=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = rng.choice(V_A, p=p)
        rew_cont[t] = float(sched_cont[t, arm])
        oh = np.zeros(V_A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [rew_cont[t] / REWARD_MAX], temb])
    return rew_cont

# Per-participant rollouts, replaying that participant's own environment
print("Computing reward curves (human-env replay)...")
idrnn_curves     = np.full((len(emb_v), N_TRIALS), np.nan, dtype=np.float64)
vanilla_curves   = np.full((len(emb_v), N_TRIALS), np.nan, dtype=np.float64)
vanilla_h_curves = np.full((len(emb_v), N_TRIALS), np.nan, dtype=np.float64)
human_curves     = np.full((len(subids_v), N_TRIALS), np.nan, dtype=np.float64)
human_chance_per_trial = np.full((len(subids_v), N_TRIALS), np.nan, dtype=np.float64)

for pi in range(len(emb_v)):
    sid = int(subids_v[pi])
    if sid not in human_envs:
        continue
    sched_cont = human_envs[sid]                                # (T, K) raw
    human_curves[pi] = human_actual[sid]
    human_chance_per_trial[pi] = sched_cont.mean(axis=1)        # arm-mean per trial

    # IDRNN
    z_vec = emb_v[pi]
    seq_sum = np.zeros(N_TRIALS, dtype=np.float64)
    for rng_seed in range(N_RNG_SEEDS):
        rng = np.random.default_rng(rng_seed * 10000 + sid)
        seq_sum += rollout_idrnn_reward(z_vec, sched_cont, task_id, rng)
    idrnn_curves[pi] = seq_sum / N_RNG_SEEDS

    # Vanilla zero-init + per-subject h-init
    h_init = v_h_aligned[pi] if np.all(np.isfinite(v_h_aligned[pi])) else None
    sum0 = np.zeros(N_TRIALS, dtype=np.float64)
    sumH = np.zeros(N_TRIALS, dtype=np.float64)
    for rng_seed in range(N_RNG_SEEDS):
        rng0 = np.random.default_rng(rng_seed * 10000 + sid + 1)
        sum0 += rollout_vanilla_reward(sched_cont, task_id, rng0, h_init=None)
        if h_init is not None:
            rngH = np.random.default_rng(rng_seed * 10000 + sid + 2)
            sumH += rollout_vanilla_reward(sched_cont, task_id, rngH, h_init=h_init)
    vanilla_curves[pi] = sum0 / N_RNG_SEEDS
    if h_init is not None:
        vanilla_h_curves[pi] = sumH / N_RNG_SEEDS

    if (pi + 1) % 50 == 0:
        print(f"  {pi+1}/{len(emb_v)} done")

# Chance baseline (across-arm mean) — single shared curve since envs come
# from the same human task. Average per-participant arm means.
model_chance_per_trial = np.nanmean(human_chance_per_trial, axis=0)

# ── Plot: 2 panels — cumulative reward + smoothed per-trial reward ───────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

trials = np.arange(1, N_TRIALS + 1)
groups = [
    ("Humans",    human_curves,    "#666666"),
    ("IDRNN",     idrnn_curves,    "#4C72B0"),
    ("Vanilla+h", vanilla_h_curves,"#8C564B"),
    ("Vanilla",   vanilla_curves,  "#DD8452"),
]

# Panel A: cumulative reward
ax = axes[0]
for label, curves, color in groups:
    valid = np.isfinite(curves[:, 0])
    if valid.sum() < 5:
        continue
    cum = np.cumsum(curves[valid], axis=1)
    m   = cum.mean(axis=0)
    sem = cum.std(axis=0) / np.sqrt(cum.shape[0])
    ax.plot(trials, m, color=color, lw=2, label=f"{label} (n={cum.shape[0]})")
    ax.fill_between(trials, m - 1.96*sem, m + 1.96*sem,
                    color=color, alpha=0.15)

# Random-play chance baseline: across-arm mean reward at each trial,
# averaged over participants' actual environments
ax.plot(trials, np.cumsum(model_chance_per_trial), color="black",
        lw=1.2, ls=":", alpha=0.6, label="Random play (across-arm mean)")

ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("Cumulative reward", fontsize=11)
ax.set_title("Cumulative reward (mean ± 95% CI)\n"
             "human-replay envs; dotted = random-play baseline",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: smoothed per-trial reward
ax = axes[1]
SMOOTH_W = 20
def _smooth(curves, w):
    """Centered rolling mean with truncating edges (no zero-padding bias).
    At edges the window shrinks to whatever data are available."""
    half = w // 2
    T = curves.shape[1]
    # Pad cumulative sum with a leading zero so c[hi]-c[lo] gives sum [lo..hi)
    c = np.concatenate([np.zeros((curves.shape[0], 1)), np.cumsum(curves, axis=1)], axis=1)
    out = np.empty_like(curves)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out[:, t] = (c[:, hi] - c[:, lo]) / (hi - lo)
    return out

for label, curves, color in groups:
    valid = np.isfinite(curves[:, 0])
    if valid.sum() < 5:
        continue
    sm = _smooth(curves[valid], SMOOTH_W)
    m  = sm.mean(axis=0)
    sem = sm.std(axis=0) / np.sqrt(sm.shape[0])
    ax.plot(trials, m, color=color, lw=2, label=f"{label} (n={sm.shape[0]})")
    ax.fill_between(trials, m - 1.96*sem, m + 1.96*sem,
                    color=color, alpha=0.15)
ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel(f"Mean reward (rolling window={SMOOTH_W})", fontsize=11)
ax.set_title("Per-trial mean reward (smoothed)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Learning curves — restless 4-armed bandit (human-environment replay)\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_RNG_SEEDS} rollouts/subject, "
    f"n_humans={(np.isfinite(human_curves[:,0])).sum()}",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_learning_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Asymptotic-performance boxplot (mean reward over last LAST_W trials) ─────
LAST_W = 50
fig, ax = plt.subplots(figsize=(9, 6))
data, colors, names = [], [], []
for label, curves, color in groups:
    valid = np.isfinite(curves[:, 0])
    if valid.sum() < 5:
        continue
    final = curves[valid, -LAST_W:].mean(axis=1)
    data.append(final); colors.append(color)
    names.append(f"{label}\n(n={len(final)})")

bp = ax.boxplot(data, positions=range(len(data)), widths=0.5,
                patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.4)
for elem in ["whiskers", "caps", "medians"]:
    for line in bp[elem]:
        line.set_color("black")
for i, (vals, c) in enumerate(zip(data, colors)):
    jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
    ax.scatter(i + jitter, vals, color=c, s=25, alpha=0.5,
               edgecolors="black", linewidths=0.3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel(f"Mean reward (last {LAST_W} trials)", fontsize=11)
ax.set_title(f"Asymptotic performance (last {LAST_W} trials)\n"
             f"all-subjects step-1 seed {BEST_SEED}, "
             f"{N_RNG_SEEDS} rollouts/subject (human-replay envs)",
             fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_learning_curves_boxplot.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Individual fit: how well do model curves track each subject's own curve? ─
# Per-participant Pearson r between (smoothed) human curve and each model
# curve. If IDRNN encodes individual differences, r_idrnn should exceed
# r_vanilla on average (Vanilla ignores subject identity once h0 fades).
print("\nIndividual-curve fit: human ↔ model (per-participant Pearson r)")

human_sm    = _smooth(human_curves,    SMOOTH_W)
idrnn_sm    = _smooth(idrnn_curves,    SMOOTH_W)
vanillaH_sm = _smooth(vanilla_h_curves,SMOOTH_W)
vanilla_sm  = _smooth(vanilla_curves,  SMOOTH_W)

def _per_subj_corr(A_, B_):
    out = np.full(A_.shape[0], np.nan)
    for i in range(A_.shape[0]):
        a = A_[i]; b = B_[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 5 and a[m].std() > 0 and b[m].std() > 0:
            out[i] = np.corrcoef(a[m], b[m])[0, 1]
    return out

r_idrnn   = _per_subj_corr(human_sm, idrnn_sm)
r_vanH    = _per_subj_corr(human_sm, vanillaH_sm)
r_vanilla = _per_subj_corr(human_sm, vanilla_sm)

mask_id = np.isfinite(r_idrnn) & np.isfinite(r_vanilla)
mask_vh = np.isfinite(r_idrnn) & np.isfinite(r_vanH)

# Wilcoxon signed-rank (paired) — does IDRNN beat Vanilla per participant?
W_iv, p_iv = stats.wilcoxon(r_idrnn[mask_id], r_vanilla[mask_id],
                             alternative="greater")
W_ih, p_ih = stats.wilcoxon(r_idrnn[mask_vh], r_vanH[mask_vh],
                             alternative="greater")
# Cohen's d on the paired difference
d_iv = ((r_idrnn[mask_id] - r_vanilla[mask_id]).mean() /
         (r_idrnn[mask_id] - r_vanilla[mask_id]).std(ddof=1))
d_ih = ((r_idrnn[mask_vh] - r_vanH[mask_vh]).mean() /
         (r_idrnn[mask_vh] - r_vanH[mask_vh]).std(ddof=1))

print(f"  Mean per-subject Pearson r:")
print(f"    IDRNN   vs human: {np.nanmean(r_idrnn):+.3f}  "
      f"(n={mask_id.sum()})")
print(f"    Vanilla+h vs h:   {np.nanmean(r_vanH):+.3f}")
print(f"    Vanilla zero:     {np.nanmean(r_vanilla):+.3f}")
print(f"  IDRNN > Vanilla    : Wilcoxon W={W_iv:.0f}, p={p_iv:.2e}, d={d_iv:+.3f}")
print(f"  IDRNN > Vanilla+h  : Wilcoxon W={W_ih:.0f}, p={p_ih:.2e}, d={d_ih:+.3f}")

# Fit-quality plot: per-participant scatter + paired-difference distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
ax.scatter(r_vanilla[mask_id], r_idrnn[mask_id], color="#4C72B0", s=30, alpha=0.7,
           edgecolors="black", linewidths=0.3, zorder=3)
lo = min(np.nanmin(r_vanilla), np.nanmin(r_idrnn)) - 0.05
hi = max(np.nanmax(r_vanilla), np.nanmax(r_idrnn)) + 0.05
ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1, alpha=0.6,
        label="y = x (no advantage)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Pearson r (Vanilla vs human)", fontsize=11)
ax.set_ylabel("Pearson r (IDRNN vs human)", fontsize=11)
ax.set_title(f"Per-participant fit quality\n"
             f"Wilcoxon p={p_iv:.2e}, d={d_iv:+.3f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
data = [r_idrnn[mask_id], r_vanH[mask_vh], r_vanilla[mask_id]]
labels = [f"IDRNN\n(n={mask_id.sum()})",
          f"Vanilla+h\n(n={mask_vh.sum()})",
          f"Vanilla\n(n={mask_id.sum()})"]
colors_ = ["#4C72B0", "#8C564B", "#DD8452"]
bp = ax.boxplot(data, positions=range(3), widths=0.5,
                patch_artist=True, showfliers=False, zorder=2)
for patch, c in zip(bp["boxes"], colors_):
    patch.set_facecolor(c); patch.set_alpha(0.4)
for elem in ["whiskers", "caps", "medians"]:
    for line in bp[elem]:
        line.set_color("black")
for i, (vals, c) in enumerate(zip(data, colors_)):
    jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
    ax.scatter(i + jitter, vals, color=c, s=20, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)
ax.axhline(0, color="grey", ls=":", lw=1, alpha=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Pearson r (model vs human, per subject)", fontsize=11)
ax.set_title("Distribution of per-subject fit quality",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_fit_quality.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Sampled-individuals plot: human + IDRNN + Vanilla curves overlaid ────────
N_SAMPLE = 12
rng_sample = np.random.default_rng(0)
# Stratify the sample across the IDRNN-advantage spectrum so we see both
# good and bad fits.
adv = r_idrnn - r_vanilla
ok  = np.where(np.isfinite(adv))[0]
ok_sorted = ok[np.argsort(adv[ok])]
# Take evenly spaced quantiles of the advantage
sample_idx = ok_sorted[np.linspace(0, len(ok_sorted) - 1,
                                    N_SAMPLE, dtype=int)]

cols = 4; rows = (N_SAMPLE + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                          sharex=True, sharey=True)
axes = axes.ravel()
for ax_i, pi in enumerate(sample_idx):
    ax = axes[ax_i]
    ax.plot(trials, human_sm[pi],    color="#666666", lw=1.5,  label="Human")
    ax.plot(trials, idrnn_sm[pi],    color="#4C72B0", lw=1.5,  label="IDRNN")
    ax.plot(trials, vanillaH_sm[pi], color="#8C564B", lw=1.5,  label="Vanilla+h")
    ax.plot(trials, vanilla_sm[pi],  color="#DD8452", lw=1.5,  label="Vanilla")
    sid = int(subids_v[pi])
    ax.set_title(f"sub {sid}  |  r_id={r_idrnn[pi]:+.2f}, "
                 f"r_v={r_vanilla[pi]:+.2f}", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide leftover axes
for j in range(len(sample_idx), len(axes)):
    axes[j].set_visible(False)

axes[0].legend(fontsize=8, loc="lower right")
fig.supxlabel("Trial", fontsize=11)
fig.supylabel(f"Mean reward (rolling window={SMOOTH_W})", fontsize=11)
fig.suptitle(
    f"Sampled individual learning curves (stratified by IDRNN−Vanilla r advantage)\n"
    f"all-subjects step-1 seed {BEST_SEED}, {N_RNG_SEEDS} rollouts/subject "
    f"(human-replay envs)",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_learning_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Sampled-individuals plot stratified along z-space (PCA of IDRNN z) ───────
# Spans the dominant axis of latent variability so we can see whether very
# different latents produce visibly different learning curves.
from sklearn.decomposition import PCA
pca = PCA(n_components=min(3, emb_v.shape[1])).fit(emb_v)
z_scores = pca.transform(emb_v)             # (N, n_components)
ev = pca.explained_variance_ratio_
print(f"\nz-space PCA: explained variance ratio = "
      f"{', '.join(f'{e:.2%}' for e in ev)}")

ok_z   = np.where(np.isfinite(human_curves[:, 0]))[0]
order  = ok_z[np.argsort(z_scores[ok_z, 0])]   # sort by PC1
sample_idx_z = order[np.linspace(0, len(order) - 1, N_SAMPLE, dtype=int)]

cols = 4; rows = (N_SAMPLE + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                          sharex=True, sharey=True)
axes = axes.ravel()
for ax_i, pi in enumerate(sample_idx_z):
    ax = axes[ax_i]
    ax.plot(trials, human_sm[pi],    color="#666666", lw=1.5, label="Human")
    ax.plot(trials, idrnn_sm[pi],    color="#4C72B0", lw=1.5, label="IDRNN")
    ax.plot(trials, vanillaH_sm[pi], color="#8C564B", lw=1.5, label="Vanilla+h")
    ax.plot(trials, vanilla_sm[pi],  color="#DD8452", lw=1.5, label="Vanilla")
    sid = int(subids_v[pi])
    pc1 = z_scores[pi, 0]
    pc2 = z_scores[pi, 1] if z_scores.shape[1] > 1 else np.nan
    ax.set_title(f"sub {sid}  |  PC1={pc1:+.2f}, PC2={pc2:+.2f}\n"
                 f"r_id={r_idrnn[pi]:+.2f}, r_v={r_vanilla[pi]:+.2f}",
                 fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for j in range(len(sample_idx_z), len(axes)):
    axes[j].set_visible(False)
axes[0].legend(fontsize=8, loc="lower right")
fig.supxlabel("Trial", fontsize=11)
fig.supylabel(f"Mean reward (rolling window={SMOOTH_W})", fontsize=11)
fig.suptitle(
    f"Sampled individual learning curves (stratified along z-space PC1)\n"
    f"PC1 explained variance = {ev[0]:.2%}, "
    f"all-subjects step-1 seed {BEST_SEED}, {N_RNG_SEEDS} rollouts/subject",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_individual_learning_curves_zspace.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# Quick scatter of z-PC1 vs PC2, colored by IDRNN-advantage, with sampled
# participants marked — gives an at-a-glance map of the latent space.
fig, ax = plt.subplots(figsize=(8, 6.5))
adv_for_plot = np.where(np.isfinite(adv), adv, np.nan)
sc = ax.scatter(z_scores[:, 0], z_scores[:, 1] if z_scores.shape[1] > 1
                else np.zeros(len(z_scores)),
                c=adv_for_plot, cmap="coolwarm", s=35, alpha=0.7,
                edgecolors="black", linewidths=0.3, zorder=3)
ax.scatter(z_scores[sample_idx_z, 0],
           z_scores[sample_idx_z, 1] if z_scores.shape[1] > 1
           else np.zeros(len(sample_idx_z)),
           facecolors="none", edgecolors="black", s=160, lw=1.5, zorder=4,
           label="sampled (z-PC1 strata)")
ax.scatter(z_scores[sample_idx, 0],
           z_scores[sample_idx, 1] if z_scores.shape[1] > 1
           else np.zeros(len(sample_idx)),
           facecolors="none", edgecolors="goldenrod", s=220, lw=1.5, zorder=5,
           label="sampled (advantage strata)")
plt.colorbar(sc, ax=ax, label="IDRNN advantage (r_idrnn − r_vanilla)")
ax.set_xlabel(f"z-PC1 ({ev[0]:.1%})", fontsize=11)
ax.set_ylabel(f"z-PC2 ({ev[1]:.1%})" if z_scores.shape[1] > 1 else "(constant)",
              fontsize=11)
ax.set_title("z-space PCA, colored by IDRNN advantage",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_zspace_advantage_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# Save curves so future analyses can skip rollouts
np.savez(os.path.join(PLOT_DIR, "step1_learning_curves.npz"),
         subids=subids_v,
         human=human_curves,
         idrnn=idrnn_curves,
         vanilla_h=vanilla_h_curves,
         vanilla=vanilla_curves,
         r_idrnn=r_idrnn,
         r_vanilla_h=r_vanH,
         r_vanilla=r_vanilla)
print(f"Saved → {os.path.join(PLOT_DIR, 'step1_learning_curves.npz')}")

print("\nDone.")
