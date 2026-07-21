#!/usr/bin/env python3
"""2-armed bandit generator for Thalmann task-0 — Fan et al. / Gershman variant,
with the generative parameters supplied by the authors and the scale calibrated
to reproduce the real schedule's marginals (data/final2armedBanditSession1.csv).

Per Fan et al. (and the Thalmann adaptation):
  • 30 blocks x 10 trials, 2 arms; a NEW pair of arms each block.
  • Block-start generative means reset, drawn from a Gaussian (mean 0, var tau0^2).
    Fan used tau0^2=100; the Thalmann schedule's realised block-mean SD ≈ 22 in
    display units (mean ≈ 47), so we draw baselines N(MU_BASE, SD_BASE) directly
    on the display scale (calibrated, see below).
  • Reward conditions per block (uniform): SS / SF / FS / FF (both stable, one
    stable, both fluctuating).  Arms are NOT labelled as stable/fluctuating.
  • Fluctuating arm: generative mean diffuses as a Gaussian RW with step var
    tau_mean(F)^2 = 4  (SD 2).  Stable arm: mean fixed within block.
  • Reward ~ N(mean, tau^2=1), rounded to nearest integer, clipped to [1, 95].
  • Biased toward an ~8-point initial arm-mean difference; the average arm-mean
    difference over a round is constrained to <= 15 (rejection sampled).

Calibration (display scale) chosen so simulated marginals match the real
schedule: overall mean ≈ 47, SD ≈ 23; stable-arm within-block SD ≈ 1; fluctuating
≈ 5-9; |Δμ| ≤ 15.  Override any constant via env (SIM_T0_SD_BASE, ...) for tuning.
"""
import os
import numpy as np

N_BLOCKS  = 30
BLOCK_LEN = 10
A         = 2

# Paper structure
SIGMA_OBS   = float(os.environ.get("SIM_T0_OBS", "1.0"))      # tau^2 = 1  -> SD 1
SIGMA_WALK  = float(os.environ.get("SIM_T0_WALK", "2.0"))     # tau_mean(F)^2 = 4 -> SD 2
DELTA_MEAN  = 8.0                                              # 8-point bias
DELTA_SD    = 2.0
MAX_DIFF    = 15.0                                            # |Δμ| over round <= 15
# Display-scale baseline (calibrated to the real schedule, not Fan's tau0=10)
MU_BASE     = float(os.environ.get("SIM_T0_MU", "47.0"))
SD_BASE     = float(os.environ.get("SIM_T0_SD_BASE", "24.0"))
REWARD_LO, REWARD_HI = 1.0, 95.0
CONDS = ("SS", "SF", "FS", "FF")


def _sample_block(rng, cond):
    delta = np.clip(rng.normal(DELTA_MEAN, DELTA_SD), 0.5, MAX_DIFF)
    sign  = rng.choice([-1.0, 1.0])
    mu_base = rng.normal(MU_BASE, SD_BASE)
    mu1_init = mu_base
    mu2_init = mu_base + sign * delta
    f1 = cond[0] == "F"; f2 = cond[1] == "F"
    m1 = m2 = None
    for _ in range(100):
        w1 = np.cumsum(rng.normal(0, SIGMA_WALK, BLOCK_LEN)) if f1 else np.zeros(BLOCK_LEN)
        w2 = np.cumsum(rng.normal(0, SIGMA_WALK, BLOCK_LEN)) if f2 else np.zeros(BLOCK_LEN)
        m1 = mu1_init + w1; m2 = mu2_init + w2
        if abs(m1.mean() - m2.mean()) <= MAX_DIFF and \
           m1.min() >= REWARD_LO and m1.max() <= REWARD_HI and \
           m2.min() >= REWARD_LO and m2.max() <= REWARD_HI:
            break
    r1 = np.clip(np.round(m1 + rng.normal(0, SIGMA_OBS, BLOCK_LEN)), REWARD_LO, REWARD_HI)
    r2 = np.clip(np.round(m2 + rng.normal(0, SIGMA_OBS, BLOCK_LEN)), REWARD_LO, REWARD_HI)
    return np.stack([r1, r2], -1).astype(np.float32), np.stack([m1, m2], -1).astype(np.float32)


def simulate_two_armed_bandit(seed):
    """Returns {rewards (30,10,2) float32, true_means (30,10,2), cond (30,)}."""
    rng = np.random.default_rng(seed)
    rewards = np.zeros((N_BLOCKS, BLOCK_LEN, A), np.float32)
    means   = np.zeros((N_BLOCKS, BLOCK_LEN, A), np.float32)
    cond    = np.zeros(N_BLOCKS, np.int32)
    for b in range(N_BLOCKS):
        ci = int(rng.integers(len(CONDS))); cond[b] = ci
        rewards[b], means[b] = _sample_block(rng, CONDS[ci])
    return {"rewards": rewards, "true_means": means, "cond": cond}


if __name__ == "__main__":
    import pandas as pd
    sims = [simulate_two_armed_bandit(s)["rewards"] for s in range(40)]
    S = np.stack(sims)
    print(f"SIM  mean={S.mean():.2f} SD={S.std():.2f} range[{S.min():.0f},{S.max():.0f}]")
    wsd = np.concatenate([s.std(1).ravel() for s in sims])
    print(f"SIM  per-arm within-block SD: median={np.median(wsd):.2f} "
          f"<1.5:{(wsd<1.5).mean():.0%} >4:{(wsd>4).mean():.0%}")
    am = np.concatenate([s.mean(1).ravel() for s in sims])
    print(f"SIM  across-arm-mean SD={am.std():.2f}")
    diffs = np.concatenate([np.abs(s.mean(1)[:, 0] - s.mean(1)[:, 1]) for s in sims])
    print(f"SIM  |Δμ| over round: mean={diffs.mean():.2f} max={diffs.max():.2f}")
    df = pd.read_csv("data/final2armedBanditSession1.csv"); sid = df.ID.iloc[0]
    g = df[df.ID == sid].sort_values(["block", "trial"])
    R = np.stack([gb.sort_values("trial")[["reward1", "reward2"]].values for _, gb in g.groupby("block")])
    print(f"REAL mean={R.mean():.2f} SD={R.std():.2f} range[{R.min():.0f},{R.max():.0f}] "
          f"across-arm-mean SD={R.mean(1).std():.2f} within-SD median={np.median(R.std(1)):.2f}")
