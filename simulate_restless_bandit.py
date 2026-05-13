#!/usr/bin/env python3
"""
Restless 4-armed bandit simulator matching Thalmann task-1 / Daw et al. (2006).

Per-arm latent means follow a Gaussian random walk with decay toward θ=50:
    μ_{i,t+1} = λ μ_{i,t} + (1-λ) θ + v_t,     v_t ~ N(0, σ_innov²)
Parameters from the paper:
    λ            = 0.9836
    θ            = 50            (decay centre)
    σ_innov²     = 7.84          (diffusion, SD ≈ 2.8)
    σ_reward     = 4             (reward observation noise)
    T            = 200 trials
    K            = 4 arms

Rewards are drawn r_{i,t} ~ N(μ_{i,t}, σ_reward²), then binarized per-arm
relative to that arm's empirical mean across the 200 trials:
    reward > arm_mean  →  1   (win)
    reward ≤ arm_mean  →  0   (loss)

Walks are rejection-sampled until both criteria hold:
  (i)  every arm is the best arm on ≥ MIN_FRAC_BEST of trials
  (ii) no arm stays best for more than MAX_RUN_BEST consecutive trials
"""
import os, argparse
import numpy as np


T, K          = 200, 4
LAMBDA        = 0.9836
THETA         = 50.0
SIGMA_INNOV   = np.sqrt(7.84)     # ≈ 2.8
SIGMA_REWARD  = 4.0

MIN_FRAC_BEST = 0.10              # each arm best ≥ 10% of trials
MAX_RUN_BEST  = 40                # no arm best > 40 consecutive trials

# Stationary SD of the AR(1) walk: σ_stat = σ_innov / sqrt(1-λ²)
SIGMA_STAT = SIGMA_INNOV / np.sqrt(1.0 - LAMBDA**2)


def _sample_walks(rng):
    """Draw one set of μ walks from the stationary distribution."""
    mu = np.empty((T, K))
    mu[0] = rng.normal(THETA, SIGMA_STAT, size=K)
    for t in range(1, T):
        v = rng.normal(0.0, SIGMA_INNOV, size=K)
        mu[t] = LAMBDA * mu[t-1] + (1.0 - LAMBDA) * THETA + v
    return mu


def _walks_pass(mu):
    """Check (i) per-arm best-fraction and (ii) max run-length constraints."""
    best = mu.argmax(axis=1)
    frac = np.bincount(best, minlength=K) / T
    if (frac < MIN_FRAC_BEST).any():
        return False
    run, max_run = 1, 1
    for t in range(1, T):
        if best[t] == best[t-1]:
            run += 1; max_run = max(max_run, run)
        else:
            run = 1
    return max_run <= MAX_RUN_BEST


def simulate_restless_bandit(seed, max_attempts=10_000):
    """
    Returns a dict with:
      mu          (T, K)  latent means
      rewards     (T, K)  continuous rewards for every arm per trial
      rewards_bin (T, K)  0/1, thresholded vs each arm's empirical reward mean
      arm_means   (K,)    per-arm empirical reward mean used as threshold
      best_arm    (T,)    argmax_i μ_{i,t}
      attempt     int     how many rejection-sampling attempts it took
    """
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        mu = _sample_walks(rng)
        if _walks_pass(mu):
            break
    else:
        raise RuntimeError(f"No valid walks within {max_attempts} attempts "
                           f"(seed={seed}); loosen MIN_FRAC_BEST / MAX_RUN_BEST.")
    rewards    = rng.normal(mu, SIGMA_REWARD)          # (T, K)
    arm_means  = rewards.mean(axis=0)                  # (K,)
    rewards_bin = (rewards > arm_means[None, :]).astype(np.int8)
    return {
        "mu":          mu,
        "rewards":     rewards,
        "rewards_bin": rewards_bin,
        "arm_means":   arm_means,
        "best_arm":    mu.argmax(axis=1),
        "attempt":     attempt,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--n_runs", type=int, default=1)
    ap.add_argument("--out_dir", default="data_restless_sim")
    ap.add_argument("--plot",   action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for r in range(args.n_runs):
        s   = args.seed + r
        out = simulate_restless_bandit(s)
        f   = os.path.join(args.out_dir, f"run{r}_seed{s}.npz")
        np.savez(f, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        frac = np.bincount(out["best_arm"], minlength=K) / T
        print(f"seed {s}: best-arm fracs {frac.round(2)}  "
              f"arm_means {out['arm_means'].round(2)}  "
              f"attempts={out['attempt']}  → {f}")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            for k in range(K):
                axes[0].plot(out["mu"][:, k], label=f"arm {k}")
            axes[0].axhline(THETA, color="grey", linestyle="--", alpha=0.5)
            axes[0].set_ylabel("μ (latent mean)"); axes[0].legend(ncol=K, fontsize=8)
            axes[0].set_title(f"seed {s} — restless bandit walks")
            axes[1].imshow(out["rewards_bin"].T, aspect="auto",
                           cmap="Greys", interpolation="nearest")
            axes[1].set_yticks(range(K)); axes[1].set_yticklabels([f"arm {k}" for k in range(K)])
            axes[1].set_xlabel("trial"); axes[1].set_ylabel("binarised reward")
            fig.tight_layout()
            fig.savefig(f.replace(".npz", ".png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
