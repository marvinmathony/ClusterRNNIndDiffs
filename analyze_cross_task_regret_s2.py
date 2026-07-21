#!/usr/bin/env python3
"""
Cross-task regret analysis for the S2-augmented Thalmann pipeline.

A minimal, self-contained port of the cross-task-regret block at the end
of analyze_thalmann_logit_updates.py (lines ~2337-2820): builds task0,
task1 and task3 human-regret arrays from pooled S1+S2 trials, runs
on-policy IDRNN / Vanilla / Vanilla+h rollouts under both exact and
marginalised environments, and writes:

  plots_thalmann_s2/step1_cross_task_regret.npz

Reads the S2-trained models from plots_thalmann_s2/step1_vs_vanilla/.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modelsandtraining import Decoder, AblatedRNN, LatentRNNz, LookupEncoderZ
from simulate_restless_bandit  import simulate_restless_bandit, T as SIM_T, K as SIM_K
from simulate_two_armed_bandit import (
    simulate_two_armed_bandit,
    N_BLOCKS as T0_N_BLOCKS, BLOCK_LEN as T0_BLOCK_LEN,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DGP            = "thalmann_s2"
DATA_DIR       = "data_thalmann_s2"
RUN_SUFFIX     = os.environ.get("S2_RUN_SUFFIX", "")
LATENTS_DIR    = f"plots_thalmann_s2{RUN_SUFFIX}/step1_vs_vanilla"
PLOT_DIR       = f"plots_thalmann_s2{RUN_SUFFIX}"
os.makedirs(PLOT_DIR, exist_ok=True)

A           = 4
assert A == SIM_K
N_TRIALS    = SIM_T       # 200 trials per restless block
N_RNG_SEEDS = 100         # rollouts/envs per condition
REWARD_MAX  = 100.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load step-1 IDRNN ──────────────────────────────────────────────────────────
_glob = glob.glob(os.path.join(LATENTS_DIR, "latents_idrnn_step1_bestseed*.pt"))
assert _glob, f"No IDRNN latents found in {LATENTS_DIR}/"
IDRNN_PATH = max(_glob, key=os.path.getmtime)
print(f"Loading IDRNN: {IDRNN_PATH}")
saved = torch.load(IDRNN_PATH, map_location=DEVICE, weights_only=False)
Z_DIM        = saved["z_dim"]
HIDDEN       = saved["hidden"]
TASK_EMB_DIM = saved["task_emb_dim"]
BASE_IN_DIM  = saved.get("base_in_dim", 5)
DEC_IN_DIM   = BASE_IN_DIM + TASK_EMB_DIM
BEST_SEED    = saved["seed"]
print(f"  z_dim={Z_DIM}, hidden={HIDDEN}, task_emb_dim={TASK_EMB_DIM}, "
      f"dec_in_dim={DEC_IN_DIM}, best seed={BEST_SEED}")

state = saved["model_state"]
decoder = Decoder(in_dim=DEC_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.to(DEVICE).eval()
task_emb_w = state["task_embedding.weight"].to(DEVICE)
emb        = np.asarray(saved["z"])         # (N, z_dim)
subids     = np.asarray(saved["subids"])

# ── Load Vanilla ───────────────────────────────────────────────────────────────
_vglob = glob.glob(os.path.join(LATENTS_DIR, "latents_vanilla_bestseed*.pt"))
assert _vglob, f"No vanilla latents found in {LATENTS_DIR}/"
VANILLA_PATH = max(_vglob, key=os.path.getmtime)
print(f"Loading vanilla: {VANILLA_PATH}")
v_saved = torch.load(VANILLA_PATH, map_location=DEVICE, weights_only=False)
V_HIDDEN       = v_saved["hidden"]
V_TASK_EMB_DIM = v_saved["task_emb_dim"]
V_A            = v_saved["A"]
V_BASE_IN_DIM  = v_saved["base_in_dim"]
V_DEC_IN_DIM   = V_BASE_IN_DIM + V_TASK_EMB_DIM
print(f"  vanilla hidden={V_HIDDEN}, task_emb_dim={V_TASK_EMB_DIM}, "
      f"dec_in_dim={V_DEC_IN_DIM}")

vanilla_model = AblatedRNN(
    hid=V_HIDDEN, in_dim=V_DEC_IN_DIM, A=V_A,
    block_structure=False,
    n_tasks=2, task_emb_dim=V_TASK_EMB_DIM,
)
vanilla_model.load_state_dict(v_saved["model_state"])
vanilla_model.to(DEVICE).eval()
v_task_emb_w = v_saved["model_state"]["task_embedding.weight"].to(DEVICE)

v_h_per_subj = np.asarray(v_saved["h"])
v_subids     = np.asarray(v_saved["subids"])
sid_to_idx_v = {int(s): i for i, s in enumerate(v_subids)}
v_h_aligned = np.full((len(subids), V_HIDDEN), np.nan, dtype=np.float32)
for i, sid in enumerate(subids):
    if int(sid) in sid_to_idx_v:
        v_h_aligned[i] = v_h_per_subj[sid_to_idx_v[int(sid)]]


# ── Rollout-semantics assertion ────────────────────────────────────────────────
# The rollout helpers below re-implement the decoder loop by hand. Verify once,
# teacher-forced, that the hand-rolled stepwise pass reproduces the training
# classes' forward exactly (task-emb concat order, z2h0 / zero-h0 init, step
# semantics) — so a future change to modelsandtraining.py can't silently
# desynchronise this script.
@torch.no_grad()
def _assert_rollouts_match_training_classes(T_chk=7):
    chk_rng = np.random.default_rng(0)

    enc = LookupEncoderZ(n_participants=1, z_dim=Z_DIM).to(DEVICE)
    enc.embed.weight.copy_(torch.as_tensor(emb[:1], dtype=torch.float32))
    ref = LatentRNNz(encoder=enc, decoder=decoder, hid=HIDDEN, z_dim=Z_DIM,
                     in_dim=DEC_IN_DIM, A=A, block_structure=True,
                     n_tasks=2, task_emb_dim=TASK_EMB_DIM).to(DEVICE).eval()
    ref.task_embedding.weight.copy_(task_emb_w)

    for task_id in (0, 1):
        # IDRNN: LatentRNNz.forward vs the stepwise loop used in the rollouts
        base = chk_rng.random((T_chk, BASE_IN_DIM)).astype(np.float32)
        base[0] = 0.0
        ref.set_task_ids(torch.tensor([task_id], device=DEVICE))
        blocks = torch.as_tensor(base, device=DEVICE).view(1, 1, T_chk, BASE_IN_DIM)
        ref_logits, _, _ = ref(torch.zeros(1, dtype=torch.long, device=DEVICE), blocks)

        z_t  = torch.as_tensor(emb[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        temb = task_emb_w[task_id].detach().cpu().numpy()
        h = decoder.z2h0(z_t).unsqueeze(0)
        step = []
        for t in range(T_chk):
            x = torch.as_tensor(np.concatenate([base[t], temb]), dtype=torch.float32,
                                device=DEVICE).view(1, 1, -1)
            lg, h = decoder(x, z_t, hidden=h)
            step.append(lg[0, 0])
        assert torch.allclose(ref_logits[0, 0], torch.stack(step), atol=1e-4), \
            f"IDRNN stepwise rollout diverges from LatentRNNz forward (task {task_id})"

        # Vanilla: AblatedRNN.forward vs the stepwise loop
        base_v = chk_rng.random((T_chk, V_BASE_IN_DIM)).astype(np.float32)
        base_v[0] = 0.0
        vanilla_model.set_task_ids(torch.tensor([task_id], device=DEVICE))
        v_ref, _, _ = vanilla_model(torch.as_tensor(base_v, device=DEVICE).unsqueeze(0))

        temb_v = v_task_emb_w[task_id].detach().cpu().numpy()
        h = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
        step = []
        for t in range(T_chk):
            x = torch.as_tensor(np.concatenate([base_v[t], temb_v]), dtype=torch.float32,
                                device=DEVICE).view(1, 1, -1)
            lg, h, _ = vanilla_model.dec(x, h0=h)
            step.append(lg[0, 0])
        assert torch.allclose(v_ref[0], torch.stack(step), atol=1e-4), \
            f"Vanilla stepwise rollout diverges from AblatedRNN forward (task {task_id})"

    vanilla_model._task_ids = None
    print("Rollout-semantics assertion passed (stepwise == training-class forward).")

_assert_rollouts_match_training_classes()

# ── Pooled human data — task0 + task1 + task3 across S1+S2 ────────────────────
# Per-participant regret = mean over ALL available trials across sessions
# (no within-session weighting — each trial counts equally).

# Task 1 (restless) -------------------------------------------------------------
# Build a LIST of per-session envs per participant. Exact-sim regret averages
# across whichever sessions the participant did; human regret is computed over
# ALL pooled trials (each trial counted once).
raw_rb = pd.concat([
    pd.read_csv("data/finalRestlessSession1.csv"),
    pd.read_csv("data/finalRestlessSession2.csv"),
], ignore_index=True)

human_envs_t1 = {}      # subid → list of (T, K) per-session envs
hum_regret_t1 = {}      # subid → mean regret across ALL trials available
for sid, sub in raw_rb.groupby("ID"):
    sub = sub.sort_values(["session", "trial"])
    arms_all = sub[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)
    rec_all  = sub["reward"].values.astype(np.float32)
    envs = []
    for _, sub_s in sub.groupby("session"):
        sub_s = sub_s.sort_values("trial")
        arms_s = sub_s[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)
        if arms_s.shape[0] >= N_TRIALS:
            envs.append(arms_s[:N_TRIALS])
    if not envs:
        continue
    human_envs_t1[int(sid)] = envs
    hum_regret_t1[int(sid)] = float((arms_all.max(axis=-1) - rec_all).mean())

# Task 0 (2-armed bandit) -------------------------------------------------------
raw_t0 = pd.concat([
    pd.read_csv("data/final2armedBanditSession1.csv"),
    pd.read_csv("data/final2armedBanditSession2.csv"),
], ignore_index=True)
human_envs_t0 = {}      # subid → list of (n_blocks, block_len, 2) per session
hum_regret_t0 = {}      # subid → pooled-trial regret
for sid, sub in raw_t0.groupby("ID"):
    sub = sub.sort_values(["session", "block", "trial"])
    arms_all = sub[["reward1", "reward2"]].values.astype(np.float32)
    rec_all  = sub["reward"].values.astype(np.float32)
    envs = []
    for _, sub_s in sub.groupby("session"):
        sub_s = sub_s.sort_values(["block", "trial"])
        arms_s = sub_s[["reward1", "reward2"]].values.astype(np.float32)
        if arms_s.shape[0] >= T0_N_BLOCKS * T0_BLOCK_LEN:
            envs.append(arms_s[:T0_N_BLOCKS * T0_BLOCK_LEN]
                         .reshape(T0_N_BLOCKS, T0_BLOCK_LEN, 2))
    if not envs:
        continue
    human_envs_t0[int(sid)] = envs
    hum_regret_t0[int(sid)] = float((arms_all.max(axis=-1) - rec_all).mean())

# Task 3 (horizon) --------------------------------------------------------------
raw_h = pd.concat([
    pd.read_csv("data/finalHorizonSession1.csv"),
    pd.read_csv("data/finalHorizonSession2.csv"),
], ignore_index=True).dropna(subset=["chosen"])
hum_regret_task3 = {}
for sid, sub in raw_h.groupby("ID"):
    arms_max = sub[["reward1", "reward2"]].values.max(axis=1)
    regret = arms_max - sub["reward"].values
    h5 = sub["Horizon"].values == 5
    hum_regret_task3[int(sid)] = {
        "all": float(regret.mean()),
        "h5":  float(regret[h5].mean())  if h5.any()  else np.nan,
        "h10": float(regret[~h5].mean()) if (~h5).any() else np.nan,
    }
print(f"Pooled human data: task0 n={len(hum_regret_t0)}, "
      f"task1 n={len(hum_regret_t1)}, task3 n={len(hum_regret_task3)}")

# ── Rollout helpers (lifted from analyze_thalmann_logit_updates.py) ───────────
@torch.no_grad()
def rollout_idrnn_reward(z_vec, sched_cont, task_id, rng):
    """Continuous reward sequence — restless task (4-armed)."""
    T_ = sched_cont.shape[0]
    z_t  = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[task_id].detach().cpu().numpy()
    h    = decoder.z2h0(z_t).unsqueeze(0)
    prev = np.concatenate([np.zeros(BASE_IN_DIM, dtype=np.float32), temb])
    rew_cont = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
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
    T_ = sched_cont.shape[0]
    temb = v_task_emb_w[task_id].detach().cpu().numpy()
    if h_init is None:
        h = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
    else:
        h = torch.as_tensor(h_init, dtype=torch.float32,
                            device=DEVICE).reshape(1, 1, V_HIDDEN)
    prev = np.concatenate([np.zeros(V_BASE_IN_DIM, dtype=np.float32), temb])
    rew_cont = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h, _ = vanilla_model.dec(x, h0=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = rng.choice(V_A, p=p)
        rew_cont[t] = float(sched_cont[t, arm])
        oh = np.zeros(V_A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [rew_cont[t] / REWARD_MAX], temb])
    return rew_cont

@torch.no_grad()
def rollout_idrnn_reward_task0(z_vec, sched_blocks, rng):
    """IDRNN rollout for task0 (2-armed, blocks × trials)."""
    n_blocks, block_len, _ = sched_blocks.shape
    z_t  = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[0].detach().cpu().numpy()
    h_seed = decoder.z2h0(z_t).unsqueeze(0)
    rew_cont = np.zeros((n_blocks, block_len), dtype=np.float32)
    leak_mass = 0.0
    for b in range(n_blocks):
        h = h_seed.clone()
        prev = np.concatenate([np.zeros(BASE_IN_DIM, dtype=np.float32), temb])
        for t in range(block_len):
            x = torch.as_tensor(prev, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0).unsqueeze(0)
            logits, h = decoder(x, z_t, hidden=h)
            full_p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
            leak_mass += float(full_p[2:].sum())
            l01 = logits[0, 0, :2].detach().cpu().numpy()
            p   = np.exp(l01 - l01.max());  p /= p.sum()
            arm = int(rng.choice(2, p=p))
            rew_cont[b, t] = float(sched_blocks[b, t, arm])
            oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
            prev = np.concatenate([oh, [rew_cont[b, t] / REWARD_MAX], temb])
    return rew_cont, leak_mass / (n_blocks * block_len)

@torch.no_grad()
def rollout_vanilla_reward_task0(sched_blocks, h_init, rng):
    """Vanilla rollout for task0."""
    n_blocks, block_len, _ = sched_blocks.shape
    temb = v_task_emb_w[0].detach().cpu().numpy()
    if h_init is None:
        h_seed = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
    else:
        h_seed = torch.as_tensor(h_init, dtype=torch.float32,
                                 device=DEVICE).reshape(1, 1, V_HIDDEN)
    rew_cont = np.zeros((n_blocks, block_len), dtype=np.float32)
    for b in range(n_blocks):
        h = h_seed.clone()
        prev = np.concatenate([np.zeros(V_BASE_IN_DIM, dtype=np.float32), temb])
        for t in range(block_len):
            x = torch.as_tensor(prev, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0).unsqueeze(0)
            logits, h, _ = vanilla_model.dec(x, h0=h)
            l01 = logits[0, 0, :2].detach().cpu().numpy()
            p   = np.exp(l01 - l01.max()); p /= p.sum()
            arm = int(rng.choice(2, p=p))
            rew_cont[b, t] = float(sched_blocks[b, t, arm])
            oh = np.zeros(V_A, dtype=np.float32); oh[arm] = 1.0
            prev = np.concatenate([oh, [rew_cont[b, t] / REWARD_MAX], temb])
    return rew_cont


# ── Marginalised env sets ─────────────────────────────────────────────────────
# Restless sim rewards are unclipped Gaussians (observed range ≈ [-13, 104]
# across 100 seeds) while the real task pays integer points within [0, 100]
# (observed [15, 88]); clip so rollout reward inputs stay inside the trained
# [0, 1] band after /REWARD_MAX. Task-0 generator already clips to [1, 95].
print(f"Sampling {N_RNG_SEEDS} marginal envs for task0 and task1 ...")
MARG_ENVS_T0 = [simulate_two_armed_bandit(s)["rewards"]    for s in range(N_RNG_SEEDS)]
MARG_ENVS_T1 = [np.clip(simulate_restless_bandit(s)["rewards"], 0.0, REWARD_MAX)
                  .astype(np.float32)
                for s in range(N_RNG_SEEDS)]


# ── Per-participant regret arrays ─────────────────────────────────────────────
N = len(subids)
hum_regret_task0       = np.full(N, np.nan)
hum_regret_task1       = np.full(N, np.nan)
hum_regret_task3_all   = np.full(N, np.nan)
hum_regret_task3_h5    = np.full(N, np.nan)
hum_regret_task3_h10   = np.full(N, np.nan)
sim_regret_task0_exact = np.full(N, np.nan)
sim_regret_task1_exact = np.full(N, np.nan)
sim_regret_task0_marg  = np.full(N, np.nan)
sim_regret_task1_marg  = np.full(N, np.nan)
sim_regret_task0_van0_exact = np.full(N, np.nan)
sim_regret_task1_van0_exact = np.full(N, np.nan)
sim_regret_task0_van0_marg  = np.full(N, np.nan)
sim_regret_task1_van0_marg  = np.full(N, np.nan)
sim_regret_task0_vanH_exact = np.full(N, np.nan)
sim_regret_task1_vanH_exact = np.full(N, np.nan)
sim_regret_task0_vanH_marg  = np.full(N, np.nan)
sim_regret_task1_vanH_marg  = np.full(N, np.nan)
leak_mass_accum = []

def _avg_t0_exact(envs, rollout_fn, sid, salt):
    """Mean per-trial regret across (sessions × RNG seeds) for task0.

    Each (session, RNG seed) pair gets a *fresh* call into the rollout
    function — that's how we reset the decoder hidden state between
    sessions (z2h0(z) is recomputed at the start of every rollout, and
    every block-boundary inside a rollout). Sessions also get distinct
    RNG seeds via `sess_idx` so the two sessions don't share noise.
    """
    accum = 0.0
    n = 0
    for sess_idx, env in enumerate(envs):
        arms_max = env.max(axis=-1)
        for s in range(N_RNG_SEEDS):
            rng = np.random.default_rng(
                s * 10000 + sid + salt + 1_000_000 * sess_idx
            )
            rew = rollout_fn(env, rng)
            accum += float((arms_max - rew).mean())
            n += 1
    return accum / max(n, 1)


def _avg_t1_exact(envs, rollout_fn, sid, salt):
    """Mean per-trial regret across (sessions × RNG seeds) for task1.

    Like _avg_t0_exact: each call into the rollout function reseeds the
    decoder hidden state, so the two sessions are treated as independent
    bandit sessions (different env sequences, different RNG seed)."""
    accum = 0.0
    n = 0
    for sess_idx, env in enumerate(envs):
        arms_max = env.max(axis=-1)
        for s in range(N_RNG_SEEDS):
            rng = np.random.default_rng(
                s * 10000 + sid + salt + 1_000_000 * sess_idx
            )
            rew = rollout_fn(env, rng)
            accum += float((arms_max - rew).mean())
            n += 1
    return accum / max(n, 1)


print("Computing regret per participant ...")
for pi in range(N):
    sid = int(subids[pi])
    if (sid not in human_envs_t0) or (sid not in human_envs_t1) or \
       (sid not in hum_regret_task3):
        continue

    # Pooled human regret across all sessions
    hum_regret_task0[pi] = hum_regret_t0[sid]
    hum_regret_task1[pi] = hum_regret_t1[sid]
    hum_regret_task3_all[pi] = hum_regret_task3[sid]["all"]
    hum_regret_task3_h5[pi]  = hum_regret_task3[sid]["h5"]
    hum_regret_task3_h10[pi] = hum_regret_task3[sid]["h10"]

    envs_t0 = human_envs_t0[sid]
    envs_t1 = human_envs_t1[sid]

    # ── R2 exact-env IDRNN regrets ────────────────────────────────────────────
    def _idrnn_t0(env, rng):
        rew, leak = rollout_idrnn_reward_task0(emb[pi], env, rng)
        leak_mass_accum.append(leak)
        return rew
    sim_regret_task0_exact[pi] = _avg_t0_exact(envs_t0, _idrnn_t0, sid, salt=7)

    def _idrnn_t1(env, rng):
        return rollout_idrnn_reward(emb[pi], env, task_id=1, rng=rng)
    sim_regret_task1_exact[pi] = _avg_t1_exact(envs_t1, _idrnn_t1, sid, salt=5)

    # ── R3 marginalised-env model regrets ─────────────────────────────────────
    sum_reg_t0 = 0.0
    for s, env in enumerate(MARG_ENVS_T0):
        rng = np.random.default_rng(s * 10000 + sid + 11)
        rew, _ = rollout_idrnn_reward_task0(emb[pi], env, rng)
        sum_reg_t0 += float((env.max(axis=-1) - rew).mean())
    sim_regret_task0_marg[pi] = sum_reg_t0 / N_RNG_SEEDS

    sum_reg_t1 = 0.0
    for s, env in enumerate(MARG_ENVS_T1):
        rng = np.random.default_rng(s * 10000 + sid + 13)
        rew_t1 = rollout_idrnn_reward(emb[pi], env, task_id=1, rng=rng)
        sum_reg_t1 += float((env.max(axis=-1) - rew_t1).mean())
    sim_regret_task1_marg[pi] = sum_reg_t1 / N_RNG_SEEDS

    # ── Vanilla zero-init ─────────────────────────────────────────────────────
    def _van0_t0(env, rng):
        return rollout_vanilla_reward_task0(env, h_init=None, rng=rng)
    sim_regret_task0_van0_exact[pi] = _avg_t0_exact(envs_t0, _van0_t0, sid, salt=17)

    def _van0_t1(env, rng):
        return rollout_vanilla_reward(env, task_id=1, rng=rng, h_init=None)
    sim_regret_task1_van0_exact[pi] = _avg_t1_exact(envs_t1, _van0_t1, sid, salt=21)

    sum_reg = 0.0
    for s, env in enumerate(MARG_ENVS_T0):
        rng = np.random.default_rng(s * 10000 + sid + 23)
        rew = rollout_vanilla_reward_task0(env, h_init=None, rng=rng)
        sum_reg += float((env.max(axis=-1) - rew).mean())
    sim_regret_task0_van0_marg[pi] = sum_reg / N_RNG_SEEDS
    sum_reg = 0.0
    for s, env in enumerate(MARG_ENVS_T1):
        rng = np.random.default_rng(s * 10000 + sid + 31)
        rew = rollout_vanilla_reward(env, task_id=1, rng=rng, h_init=None)
        sum_reg += float((env.max(axis=-1) - rew).mean())
    sim_regret_task1_van0_marg[pi] = sum_reg / N_RNG_SEEDS

    # ── Vanilla+h (per-subject hidden init) ───────────────────────────────────
    h_init = v_h_aligned[pi] if np.all(np.isfinite(v_h_aligned[pi])) else None
    if h_init is not None:
        def _vanH_t0(env, rng):
            return rollout_vanilla_reward_task0(env, h_init=h_init, rng=rng)
        sim_regret_task0_vanH_exact[pi] = _avg_t0_exact(envs_t0, _vanH_t0, sid, salt=19)

        def _vanH_t1(env, rng):
            return rollout_vanilla_reward(env, task_id=1, rng=rng, h_init=h_init)
        sim_regret_task1_vanH_exact[pi] = _avg_t1_exact(envs_t1, _vanH_t1, sid, salt=25)

        sum_reg = 0.0
        for s, env in enumerate(MARG_ENVS_T0):
            rng = np.random.default_rng(s * 10000 + sid + 29)
            rew = rollout_vanilla_reward_task0(env, h_init=h_init, rng=rng)
            sum_reg += float((env.max(axis=-1) - rew).mean())
        sim_regret_task0_vanH_marg[pi] = sum_reg / N_RNG_SEEDS

        sum_reg = 0.0
        for s, env in enumerate(MARG_ENVS_T1):
            rng = np.random.default_rng(s * 10000 + sid + 37)
            rew = rollout_vanilla_reward(env, task_id=1, rng=rng, h_init=h_init)
            sum_reg += float((env.max(axis=-1) - rew).mean())
        sim_regret_task1_vanH_marg[pi] = sum_reg / N_RNG_SEEDS

    if (pi + 1) % 25 == 0:
        print(f"  {pi+1}/{N} done")

print(f"Mean softmax leakage on arms 2,3 during task0 rollouts: "
      f"{np.mean(leak_mass_accum):.4f}  (should be ≪ 0.1)")

# ── Save NPZ ──────────────────────────────────────────────────────────────────
out_npz = os.path.join(PLOT_DIR, "step1_cross_task_regret.npz")
np.savez(
    out_npz,
    subids=subids,
    hum_regret_task0=hum_regret_task0,
    hum_regret_task1=hum_regret_task1,
    hum_regret_task3=hum_regret_task3_all,
    hum_regret_task3_h5=hum_regret_task3_h5,
    hum_regret_task3_h10=hum_regret_task3_h10,
    sim_regret_task0_exact=sim_regret_task0_exact,
    sim_regret_task1_exact=sim_regret_task1_exact,
    sim_regret_task0_marg=sim_regret_task0_marg,
    sim_regret_task1_marg=sim_regret_task1_marg,
    sim_regret_task0_van0_exact=sim_regret_task0_van0_exact,
    sim_regret_task1_van0_exact=sim_regret_task1_van0_exact,
    sim_regret_task0_van0_marg=sim_regret_task0_van0_marg,
    sim_regret_task1_van0_marg=sim_regret_task1_van0_marg,
    sim_regret_task0_vanH_exact=sim_regret_task0_vanH_exact,
    sim_regret_task1_vanH_exact=sim_regret_task1_vanH_exact,
    sim_regret_task0_vanH_marg=sim_regret_task0_vanH_marg,
    sim_regret_task1_vanH_marg=sim_regret_task1_vanH_marg,
)
print(f"Saved → {out_npz}")
print("\nDone.")
