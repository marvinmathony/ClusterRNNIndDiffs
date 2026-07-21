"""Smoke test for the cross-task regret block of analyze_thalmann_logit_updates.py.

Loads the model + latents directly, runs the new analysis on a SMALL subset of
participants (default 5) without redoing JSD/learning-curves. Verifies shapes,
identities, and that simulate_two_armed_bandit imports cleanly. Not part of the
shipped pipeline — delete after the main script has been validated end-to-end.
"""
import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from modelsandtraining import Decoder, AblatedRNN
from simulate_restless_bandit import simulate_restless_bandit, T as SIM_T, K as SIM_K
from simulate_two_armed_bandit import simulate_two_armed_bandit, N_BLOCKS as T0_N, BLOCK_LEN as T0_L

DGP = "thalmann"
A = 4
N_TRIALS = SIM_T
REWARD_MAX = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ───────────────────────────────────────────────────────────────
LATENTS = max(glob.glob("plots_thalmann/step1_vs_vanilla/latents_idrnn_step1_bestseed*.pt"),
              key=os.path.getmtime)
saved = torch.load(LATENTS, map_location=DEVICE, weights_only=False)
Z_DIM = saved["z_dim"]; HIDDEN = saved["hidden"]; TASK_EMB_DIM = saved["task_emb_dim"]
BASE_IN_DIM = saved.get("base_in_dim", 5); DEC_IN_DIM = BASE_IN_DIM + TASK_EMB_DIM
state = saved["model_state"]
decoder = Decoder(in_dim=DEC_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.to(DEVICE).eval()
task_emb_w = state["task_embedding.weight"].to(DEVICE)
emb = np.asarray(saved["z"]); subids = np.asarray(saved["subids"])

quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
quest["PHQ"] = quest[[f"PHQ_9_{i}" for i in range(10)]].mean(1)
phq = quest.reindex(subids)["PHQ"].values.astype(float)
valid = ~np.isnan(phq)
emb_v, subids_v = emb[valid], subids[valid]

# Cached idrnn/vanilla curves for task1 exact
lc = np.load("plots_thalmann/step1_learning_curves.npz")
idrnn_curves     = lc["idrnn"]
vanilla_curves   = lc["vanilla"]
vanilla_h_curves = lc["vanilla_h"]
print(f"idrnn_curves: shape={idrnn_curves.shape}")

# ── Load vanilla model ───────────────────────────────────────────────────────
import glob
V_PATH = max(glob.glob("plots_thalmann/step1_vs_vanilla/latents_vanilla_bestseed*.pt"),
             key=os.path.getmtime)
v_saved = torch.load(V_PATH, map_location=DEVICE, weights_only=False)
V_HIDDEN = v_saved["hidden"]; V_TASK_EMB_DIM = v_saved["task_emb_dim"]
V_A = v_saved["A"]; V_BASE_IN_DIM = v_saved["base_in_dim"]
V_DEC_IN_DIM = V_BASE_IN_DIM + V_TASK_EMB_DIM
vanilla_model = AblatedRNN(hid=V_HIDDEN, in_dim=V_DEC_IN_DIM, A=V_A,
                           block_structure=False, n_tasks=2,
                           task_emb_dim=V_TASK_EMB_DIM)
vanilla_model.load_state_dict(v_saved["model_state"])
vanilla_model.to(DEVICE).eval()
v_task_emb_w = v_saved["model_state"]["task_embedding.weight"].to(DEVICE)
v_h_per_subj = np.asarray(v_saved["h"])
v_subids = np.asarray(v_saved["subids"])
sid_to_idx = {int(s): i for i, s in enumerate(v_subids)}
v_h_aligned = np.full((len(subids_v), V_HIDDEN), np.nan, dtype=np.float32)
for i, sid in enumerate(subids_v):
    if int(sid) in sid_to_idx:
        v_h_aligned[i] = v_h_per_subj[sid_to_idx[int(sid)]]

# ── Task0/task1 human envs (full participant set) ────────────────────────────
raw_task0 = pd.read_csv("data/final2armedBanditSession1.csv")
human_envs_task0, human_actual_task0 = {}, {}
for sid, sub in raw_task0.groupby("ID"):
    sub = sub.sort_values(["block", "trial"])
    arms = sub[["reward1", "reward2"]].values.astype(np.float32)
    rec  = sub["reward"].values.astype(np.float32)
    if arms.shape[0] < T0_N * T0_L:
        continue
    human_envs_task0[int(sid)]   = arms.reshape(T0_N, T0_L, 2)
    human_actual_task0[int(sid)] = rec.reshape(T0_N, T0_L)

raw_rb = pd.read_csv("data/finalRestlessSession1.csv")
human_envs, human_actual = {}, {}
for sid, sub in raw_rb.groupby("ID"):
    sub = sub.sort_values("trial")
    arms = sub[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)
    rec  = sub["reward"].values.astype(np.float32)
    if arms.shape[0] < N_TRIALS: continue
    human_envs[int(sid)] = arms[:N_TRIALS]
    human_actual[int(sid)] = rec[:N_TRIALS]

raw_h = pd.read_csv("data/finalHorizonSession1.csv").dropna(subset=["chosen"])
hum_regret_task3_dict = {}
for sid, sub in raw_h.groupby("ID"):
    arms_max = sub[["reward1", "reward2"]].values.max(axis=1)
    regret = arms_max - sub["reward"].values
    h5 = sub["Horizon"].values == 5
    hum_regret_task3_dict[int(sid)] = {
        "all": float(regret.mean()),
        "h5":  float(regret[h5].mean())  if h5.any()  else np.nan,
        "h10": float(regret[~h5].mean()) if (~h5).any() else np.nan,
    }
print(f"task3 participants: {len(hum_regret_task3_dict)}")

# ── Rollout functions ────────────────────────────────────────────────────────
@torch.no_grad()
def rollout_idrnn_reward_task0(z_vec, sched_blocks, rng):
    n_blocks, block_len, _ = sched_blocks.shape
    z_t  = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[0].detach().cpu().numpy()
    h_seed = decoder.z2h0(z_t).unsqueeze(0)
    rew_cont = np.zeros((n_blocks, block_len), dtype=np.float32)
    leak = 0.0
    for b in range(n_blocks):
        h = h_seed.clone()
        prev = np.concatenate([np.zeros(5, dtype=np.float32), temb])
        for t in range(block_len):
            x = torch.as_tensor(prev, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0).unsqueeze(0)
            logits, h = decoder(x, z_t, hidden=h)
            full_p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
            leak += float(full_p[2:].sum())
            l01 = logits[0, 0, :2].detach().cpu().numpy()
            p   = np.exp(l01 - l01.max()); p /= p.sum()
            arm = int(rng.choice(2, p=p))
            rew_cont[b, t] = float(sched_blocks[b, t, arm])
            oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
            prev = np.concatenate([oh, [rew_cont[b, t] / REWARD_MAX], temb])
    return rew_cont, leak / (n_blocks * block_len)

@torch.no_grad()
def rollout_vanilla_reward_task0(sched_blocks, h_init, rng):
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

@torch.no_grad()
def rollout_vanilla_reward_t1(sched, h_init, rng):
    T = sched.shape[0]
    temb = v_task_emb_w[1].detach().cpu().numpy()
    if h_init is None:
        h = torch.zeros(1, 1, V_HIDDEN, device=DEVICE)
    else:
        h = torch.as_tensor(h_init, dtype=torch.float32,
                            device=DEVICE).reshape(1, 1, V_HIDDEN)
    prev = np.concatenate([np.zeros(V_BASE_IN_DIM, dtype=np.float32), temb])
    rew = np.zeros(T, dtype=np.float32)
    for t in range(T):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h, _ = vanilla_model.dec(x, h0=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = int(rng.choice(V_A, p=p))
        rew[t] = float(sched[t, arm])
        oh = np.zeros(V_A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [rew[t] / REWARD_MAX], temb])
    return rew

@torch.no_grad()
def rollout_task1(z_vec, sched, rng):
    T = sched.shape[0]
    z_t = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[1].detach().cpu().numpy()
    h = decoder.z2h0(z_t).unsqueeze(0)
    prev = np.concatenate([np.zeros(5, dtype=np.float32), temb])
    rew = np.zeros(T, dtype=np.float32)
    for t in range(T):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x, z_t, hidden=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = int(rng.choice(A, p=p))
        rew[t] = float(sched[t, arm])
        oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [rew[t] / REWARD_MAX], temb])
    return rew

# ── Marg envs (small for smoke test) ────────────────────────────────────────
N_SMOKE_ENVS = 5
MARG_T0 = [simulate_two_armed_bandit(s)["rewards"] for s in range(N_SMOKE_ENVS)]
MARG_T1 = [simulate_restless_bandit(s)["rewards"].astype(np.float32) for s in range(N_SMOKE_ENVS)]
print(f"MARG_T0[0].shape={MARG_T0[0].shape}  MARG_T1[0].shape={MARG_T1[0].shape}")

# ── Run on first 5 participants ─────────────────────────────────────────────
N_SMOKE_SUBS = 5
print(f"\nSmoke run on {N_SMOKE_SUBS} participants with {N_SMOKE_ENVS} envs each:")
leak_all = []
for pi in range(min(N_SMOKE_SUBS, len(emb_v))):
    sid = int(subids_v[pi])
    if sid not in human_envs_task0 or sid not in human_envs or sid not in hum_regret_task3_dict:
        print(f"  pi={pi} sid={sid}: missing data, skip"); continue

    arms_max_t0 = human_envs_task0[sid].max(axis=-1)
    arms_max_t1 = human_envs[sid].max(axis=-1)
    hr_t0 = (arms_max_t0 - human_actual_task0[sid]).mean()
    hr_t1 = (arms_max_t1 - human_actual[sid]).mean()
    hr_t3 = hum_regret_task3_dict[sid]["all"]

    # R2 task0 exact (1 seed for smoke)
    rng = np.random.default_rng(7 + sid)
    rew_t0, leak = rollout_idrnn_reward_task0(emb_v[pi], human_envs_task0[sid], rng)
    leak_all.append(leak)
    sr_t0_exact = (arms_max_t0 - rew_t0).mean()

    # R2 task1 exact: use cached idrnn_curves
    sr_t1_exact = (arms_max_t1 - idrnn_curves[pi]).mean()

    # Math sanity
    lhs = sr_t1_exact + idrnn_curves[pi].mean()
    rhs = arms_max_t1.mean()
    assert abs(lhs - rhs) < 1e-3, (pi, lhs, rhs)

    # R3 task0/task1 marg (N_SMOKE_ENVS envs)
    sr_t0_marg = []
    for env in MARG_T0:
        rng = np.random.default_rng(11 + sid)
        rew, _ = rollout_idrnn_reward_task0(emb_v[pi], env, rng)
        sr_t0_marg.append((env.max(axis=-1) - rew).mean())
    sr_t0_marg = float(np.mean(sr_t0_marg))

    sr_t1_marg = []
    for env in MARG_T1:
        rng = np.random.default_rng(13 + sid)
        rew = rollout_task1(emb_v[pi], env, rng)
        sr_t1_marg.append((env.max(axis=-1) - rew).mean())
    sr_t1_marg = float(np.mean(sr_t1_marg))

    # Vanilla zero-init R2/R3 task0 + R3 task1 + cached R2 task1
    rng = np.random.default_rng(17 + sid)
    rew_v0_t0 = rollout_vanilla_reward_task0(human_envs_task0[sid], None, rng)
    sr_t0_v0_exact = (arms_max_t0 - rew_v0_t0).mean()
    sr_t1_v0_exact = (arms_max_t1 - vanilla_curves[pi]).mean()
    sr_t0_v0_marg = float(np.mean([
        (env.max(axis=-1) - rollout_vanilla_reward_task0(env, None,
            np.random.default_rng(23 + sid + s))).mean()
        for s, env in enumerate(MARG_T0)
    ]))
    sr_t1_v0_marg = float(np.mean([
        (env.max(axis=-1) - rollout_vanilla_reward_t1(env, None,
            np.random.default_rng(31 + sid + s))).mean()
        for s, env in enumerate(MARG_T1)
    ]))

    # Vanilla+h
    h_init = v_h_aligned[pi] if np.all(np.isfinite(v_h_aligned[pi])) else None
    if h_init is not None:
        rng = np.random.default_rng(19 + sid)
        rew_vH_t0 = rollout_vanilla_reward_task0(human_envs_task0[sid], h_init, rng)
        sr_t0_vH_exact = (arms_max_t0 - rew_vH_t0).mean()
        sr_t1_vH_exact = (arms_max_t1 - vanilla_h_curves[pi]).mean()
        sr_t0_vH_marg = float(np.mean([
            (env.max(axis=-1) - rollout_vanilla_reward_task0(env, h_init,
                np.random.default_rng(29 + sid + s))).mean()
            for s, env in enumerate(MARG_T0)
        ]))
        sr_t1_vH_marg = float(np.mean([
            (env.max(axis=-1) - rollout_vanilla_reward_t1(env, h_init,
                np.random.default_rng(37 + sid + s))).mean()
            for s, env in enumerate(MARG_T1)
        ]))
    else:
        sr_t0_vH_exact = sr_t1_vH_exact = sr_t0_vH_marg = sr_t1_vH_marg = np.nan

    print(f"  pi={pi} sid={sid}:")
    print(f"    HUM   t0={hr_t0:.2f}  t1={hr_t1:.2f}  t3={hr_t3:.2f}")
    print(f"    IDRNN exact t0={sr_t0_exact:.2f}  t1={sr_t1_exact:.2f}  | "
          f"marg t0={sr_t0_marg:.2f}  t1={sr_t1_marg:.2f}")
    print(f"    Van+h exact t0={sr_t0_vH_exact:.2f}  t1={sr_t1_vH_exact:.2f}  | "
          f"marg t0={sr_t0_vH_marg:.2f}  t1={sr_t1_vH_marg:.2f}")
    print(f"    Van   exact t0={sr_t0_v0_exact:.2f}  t1={sr_t1_v0_exact:.2f}  | "
          f"marg t0={sr_t0_v0_marg:.2f}  t1={sr_t1_v0_marg:.2f}")

print(f"\nMean softmax leak on arms 2/3 (task0 rollouts): {np.mean(leak_all):.4f}")
print("Math sanity: task1 exact identity holds ✓")
print("Smoke run OK.")
