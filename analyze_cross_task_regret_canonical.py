#!/usr/bin/env python3
"""Panel e — three-regression cross-task regret, Thalmann z=3 CANONICAL model
(IDRNN-focused), VECTORISED over subjects (≈200x faster than per-subject loop).

Four quantities (task-1 restless; task-0 analogues also saved):
  R1 = human regret (observed)
  R2 = exact env, single rng seed (model sees exact env, no marginalisation)
  R3 = exact env, marginalised over rng seeds (N_RNG_SEEDS)
  R4 = marginalised over environments + rng seeds
Held-out transfer target = horizon task (task 3) regret.

All rollouts step ALL subjects through one batched decoder call per timestep
(torch.multinomial samples every subject's arm at once).  The per-subject MEAN
regret (the MC estimate) is what downstream uses, so the batched RNG scheme is
equivalent to the per-subject one in expectation.

Env: THAL_FULL (canonical dir).  Writes {THAL_FULL}/regret/step1_cross_task_regret.npz
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import Decoder
from simulate_restless_bandit import simulate_restless_bandit, T as SIM_T, K as SIM_K
from simulate_two_armed_bandit import (
    simulate_two_armed_bandit, N_BLOCKS as T0_N_BLOCKS, BLOCK_LEN as T0_BLOCK_LEN,
)

_FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full")
PLOT_DIR = f"{_FULL}/regret"
IDRNN_PT = f"{_FULL}/canonical/idrnn/latents_idrnn_canonical.pt"
os.makedirs(PLOT_DIR, exist_ok=True)
A = 4; assert A == SIM_K
N_RNG = int(os.environ.get("N_RNG_SEEDS", "100"))
REWARD_MAX = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  N_RNG={N_RNG}")

saved = torch.load(IDRNN_PT, map_location=DEVICE, weights_only=False)
Z_DIM, HIDDEN, TASK_EMB_DIM = saved["z_dim"], saved["hidden"], saved["task_emb_dim"]
BASE_IN_DIM = saved["base_in_dim"]; DEC_IN_DIM = BASE_IN_DIM + TASK_EMB_DIM
state = saved["model_state"]
decoder = Decoder(in_dim=DEC_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
decoder.to(DEVICE).eval()
task_emb_w = state["task_embedding.weight"].to(DEVICE)         # (n_tasks, te)
emb = torch.as_tensor(np.asarray(saved["z_train"]), dtype=torch.float32, device=DEVICE)  # (N, z)
subids = np.asarray(saved["subids"]).astype(int)
N = len(subids)
print(f"  z{tuple(emb.shape)} seed={saved['seed']} dec_in={DEC_IN_DIM}")
gen = torch.Generator(device=DEVICE)


@torch.no_grad()
def roll_restless(z, env, task_id, seed):
    """z (M,zdim); env (M,T,K) per-subject or (T,K) shared. Returns reward (M,T)."""
    M = z.shape[0]
    if env.dim() == 2:
        env = env.unsqueeze(0).expand(M, -1, -1)
    T = env.shape[1]
    temb = task_emb_w[task_id].unsqueeze(0).expand(M, -1)
    h = decoder.z2h0(z).unsqueeze(0).contiguous()
    prev = torch.cat([torch.zeros(M, BASE_IN_DIM, device=DEVICE), temb], -1)
    rew = torch.empty(M, T, device=DEVICE); ar = torch.arange(M, device=DEVICE)
    gen.manual_seed(seed)
    for t in range(T):
        logits, h = decoder(prev.unsqueeze(1), z, hidden=h)
        arm = torch.multinomial(F.softmax(logits[:, 0], -1), 1, generator=gen).squeeze(1)
        r = env[ar, t, arm]; rew[:, t] = r
        prev = torch.cat([F.one_hot(arm, A).float(), (r/REWARD_MAX).unsqueeze(1), temb], -1)
    return rew


@torch.no_grad()
def roll_t0(z, env, seed):
    """z (M,z); env (M, nb, bl, 2). Returns reward (M, nb, bl) + leak."""
    M, nb, bl, _ = env.shape
    temb = task_emb_w[0].unsqueeze(0).expand(M, -1)
    h_seed = decoder.z2h0(z).unsqueeze(0).contiguous()
    rew = torch.empty(M, nb, bl, device=DEVICE); ar = torch.arange(M, device=DEVICE)
    leak = 0.0; gen.manual_seed(seed)
    for b in range(nb):
        h = h_seed.clone()
        prev = torch.cat([torch.zeros(M, BASE_IN_DIM, device=DEVICE), temb], -1)
        for t in range(bl):
            logits, h = decoder(prev.unsqueeze(1), z, hidden=h)
            fp = F.softmax(logits[:, 0], -1); leak += float(fp[:, 2:].sum(1).mean())
            arm = torch.multinomial(F.softmax(logits[:, 0, :2], -1), 1, generator=gen).squeeze(1)
            r = env[ar, b, t, arm]; rew[:, b, t] = r
            prev = torch.cat([F.one_hot(arm, A).float(), (r/REWARD_MAX).unsqueeze(1), temb], -1)
    return rew, leak / (nb * bl)


# ── Human envs + regret (R1), pooled S1+S2 ─────────────────────────────────────
def _load_t1():
    raw = pd.concat([pd.read_csv("data/finalRestlessSession1.csv"),
                     pd.read_csv("data/finalRestlessSession2.csv")], ignore_index=True)
    envs, reg = {}, {}
    for sid, sub in raw.groupby("ID"):
        sub = sub.sort_values(["session", "trial"])
        arms = sub[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)
        rec = sub["reward"].values.astype(np.float32)
        e = [s.sort_values("trial")[["reward1", "reward2", "reward3", "reward4"]].values.astype(np.float32)[:SIM_T]
             for _, s in sub.groupby("session")
             if s.shape[0] >= SIM_T]
        if e:
            envs[int(sid)] = e; reg[int(sid)] = float((arms.max(-1) - rec).mean())
    return envs, reg


def _load_t0():
    raw = pd.concat([pd.read_csv("data/final2armedBanditSession1.csv"),
                     pd.read_csv("data/final2armedBanditSession2.csv")], ignore_index=True)
    L = T0_N_BLOCKS * T0_BLOCK_LEN
    envs, reg = {}, {}
    for sid, sub in raw.groupby("ID"):
        sub = sub.sort_values(["session", "block", "trial"])
        arms = sub[["reward1", "reward2"]].values.astype(np.float32)
        rec = sub["reward"].values.astype(np.float32)
        e = [s.sort_values(["block", "trial"])[["reward1", "reward2"]].values.astype(np.float32)[:L].reshape(T0_N_BLOCKS, T0_BLOCK_LEN, 2)
             for _, s in sub.groupby("session")
             if s.shape[0] >= L]
        if e:
            envs[int(sid)] = e; reg[int(sid)] = float((arms.max(-1) - rec).mean())
    return envs, reg


def _load_t3():
    raw = pd.concat([pd.read_csv("data/finalHorizonSession1.csv"),
                     pd.read_csv("data/finalHorizonSession2.csv")], ignore_index=True).dropna(subset=["chosen"])
    out = {}
    for sid, sub in raw.groupby("ID"):
        reg = sub[["reward1", "reward2"]].values.max(1) - sub["reward"].values
        h5 = sub["Horizon"].values == 5
        out[int(sid)] = {"all": float(reg.mean()),
                         "h5": float(reg[h5].mean()) if h5.any() else np.nan,
                         "h10": float(reg[~h5].mean()) if (~h5).any() else np.nan}
    return out


envs_t1, reg_t1 = _load_t1(); envs_t0, reg_t0 = _load_t0(); reg_t3 = _load_t3()
print(f"human: t0={len(reg_t0)} t1={len(reg_t1)} t3={len(reg_t3)}")
MARG_T1 = [torch.as_tensor(simulate_restless_bandit(s)["rewards"].astype(np.float32), device=DEVICE) for s in range(N_RNG)]
MARG_T0 = [torch.as_tensor(simulate_two_armed_bandit(s)["rewards"].astype(np.float32), device=DEVICE) for s in range(N_RNG)]

# subject index sets (present in all human tasks)
present = np.array([i for i in range(N) if subids[i] in envs_t0 and subids[i] in envs_t1 and subids[i] in reg_t3])
arr = lambda: np.full(N, np.nan, np.float32)
hum_t0, hum_t1, hum_t3, hum_t3_h5, hum_t3_h10 = arr(), arr(), arr(), arr(), arr()
for pi in present:
    sid = int(subids[pi])
    hum_t0[pi] = reg_t0[sid]; hum_t1[pi] = reg_t1[sid]
    hum_t3[pi] = reg_t3[sid]["all"]; hum_t3_h5[pi] = reg_t3[sid]["h5"]; hum_t3_h10[pi] = reg_t3[sid]["h10"]


def exact_regret(envs, roller, salt, max_T_axis):
    """Batched R2 (single seed) + R3 (mean N_RNG) per subject, averaged over the
    subject's sessions. roller(z_sub, env_batch, seed) -> reward; regret = mean over
    trial axes of (env.max(arms) - reward)."""
    r2_sum = np.zeros(N); r3_sum = np.zeros(N); cnt = np.zeros(N)
    max_sess = max(len(envs[int(subids[pi])]) for pi in present)
    for s_idx in range(max_sess):
        idxs = [pi for pi in present if len(envs[int(subids[pi])]) > s_idx]
        if not idxs:
            continue
        env_b = torch.stack([torch.as_tensor(envs[int(subids[pi])][s_idx], device=DEVICE) for pi in idxs])
        amax = env_b.max(-1).values                      # (M, ...trial dims)
        z_sub = emb[idxs]
        reg_acc = np.zeros(len(idxs)); r2_this = None
        for s in range(N_RNG):
            out = roller(z_sub, env_b, s * 9973 + salt)
            rew = out[0] if isinstance(out, tuple) else out
            reg = (amax - rew).mean(dim=tuple(range(1, rew.dim()))).cpu().numpy()
            reg_acc += reg
            if s == 0:
                r2_this = reg
        r3_this = reg_acc / N_RNG
        for j, pi in enumerate(idxs):
            r2_sum[pi] += r2_this[j]; r3_sum[pi] += r3_this[j]; cnt[pi] += 1
    cnt = np.clip(cnt, 1, None)
    return r2_sum / cnt, r3_sum / cnt


def marg_regret(marg_envs, roller, salt):
    """Batched R4: mean regret over marginal envs (shared across subjects)."""
    z_all = emb[present]; acc = np.zeros(len(present))
    for s, env in enumerate(marg_envs):
        if env.dim() == 3:          # task0 (nb,bl,2): expand to (M,nb,bl,2)
            env_b = env.unsqueeze(0).expand(len(present), -1, -1, -1)
        else:
            env_b = env
        amax = env.max(-1).values
        out = roller(z_all, env_b, s * 9973 + salt)
        rew = out[0] if isinstance(out, tuple) else out
        reg = (amax.unsqueeze(0) - rew).mean(dim=tuple(range(1, rew.dim()))).cpu().numpy()
        acc += reg
    full = np.full(N, np.nan); full[present] = acc / len(marg_envs)
    return full


def _scatter(vals_present):
    full = np.full(N, np.nan); full[present] = vals_present[present]; return full


print("Batched rollouts: task1 ...")
t1_r2, t1_r3 = exact_regret(envs_t1, lambda z, e, s: roll_restless(z, e, 1, s), 5, 1)
t1_r4 = marg_regret(MARG_T1, lambda z, e, s: roll_restless(z, e, 1, s), 13)
print("Batched rollouts: task0 ...")
t0_r2, t0_r3 = exact_regret(envs_t0, lambda z, e, s: roll_t0(z, e, s), 7, 2)
t0_r4 = marg_regret(MARG_T0, lambda z, e, s: roll_t0(z, e, s), 11)

out = os.path.join(PLOT_DIR, "step1_cross_task_regret.npz")
np.savez(out, subids=subids,
         hum_regret_task0=hum_t0, hum_regret_task1=hum_t1, hum_regret_task3=hum_t3,
         hum_regret_task3_h5=hum_t3_h5, hum_regret_task3_h10=hum_t3_h10,
         sim_regret_task0_r2_exact_single=_scatter(t0_r2), sim_regret_task1_r2_exact_single=_scatter(t1_r2),
         sim_regret_task0_r3_exact_margrng=_scatter(t0_r3), sim_regret_task1_r3_exact_margrng=_scatter(t1_r3),
         sim_regret_task0_r4_marg=t0_r4, sim_regret_task1_r4_marg=t1_r4)
print(f"Saved -> {out}")
print(f"sanity: task1 R3 mean={np.nanmean(_scatter(t1_r3)):.2f}  R4 mean={np.nanmean(t1_r4):.2f}  "
      f"human t1 mean={np.nanmean(hum_t1):.2f}")
