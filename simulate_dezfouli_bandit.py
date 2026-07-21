#!/usr/bin/env python3
"""
Dezfouli 2-armed bandit generative model.

Spec (from the original paper, transcribed):
    The task was divided into 12 different blocks. Within each block one of
    the actions was better than the other in terms of the probability of
    earning a reward but across blocks the action with the higher reward
    probability was varied, i.e., in some of the blocks the left action was
    better while in others the right action was better. The reward
    probability for the better action was 0.25, 0.125, or 0.08 and the
    probability of earning reward from the other action was always 0.05
    (probabilities were fixed within each block); as such, there were six
    pairs of reward probabilities and each was repeated twice.

Block length is NOT fixed in the original task (the empirical data shows
~66-130 trials per (subject, block)). The caller supplies per-block trial
counts; the canonical use is to pass the held-out subject's own empirical
block lengths so the trial budget matches the real session.
"""
import numpy as np


N_BLOCKS   = 12
GOOD_PROBS = (0.25, 0.125, 0.08)   # 3 levels for the better arm
BAD_PROB   = 0.05                  # always 0.05 for the worse arm


def simulate_dezfouli_bandit(seed, block_lengths):
    """
    Parameters
    ----------
    seed : int
        RNG seed (passed to ``np.random.default_rng``).
    block_lengths : array-like of length 12
        Number of trials per block. Pass the held-out subject's empirical
        lengths to keep the trial budget matched.

    Returns
    -------
    rewards : float32 array, shape (12, max_T, 2)
        Bernoulli draws for *both* arms at every trial (so a policy can
        evaluate counterfactual reward). Positions past ``block_lengths[b]``
        are padded with -100.
    good_arm : int array, shape (12,)
        Which arm (0 or 1) was the better one in each block.
    good_prob : float array, shape (12,)
        Reward probability of the better arm in each block.
    """
    block_lengths = np.asarray(block_lengths, dtype=int)
    if block_lengths.shape != (N_BLOCKS,):
        raise ValueError(
            f"block_lengths must have shape ({N_BLOCKS},), got {block_lengths.shape}"
        )

    rng = np.random.default_rng(seed)

    # 6 (arm, good_prob) pairs × 2 repeats = 12 configs; shuffled across blocks.
    configs = [(a, p) for a in (0, 1) for p in GOOD_PROBS] * 2
    rng.shuffle(configs)

    max_T = int(block_lengths.max())
    rewards   = np.full((N_BLOCKS, max_T, 2), -100.0, dtype=np.float32)
    good_arm  = np.empty(N_BLOCKS, dtype=int)
    good_prob = np.empty(N_BLOCKS, dtype=float)

    for b, (arm, p) in enumerate(configs):
        T = int(block_lengths[b])
        probs = [BAD_PROB, BAD_PROB]
        probs[arm] = p
        rewards[b, :T, 0] = (rng.random(T) < probs[0]).astype(np.float32)
        rewards[b, :T, 1] = (rng.random(T) < probs[1]).astype(np.float32)
        good_arm[b]  = arm
        good_prob[b] = p

    return rewards, good_arm, good_prob


if __name__ == "__main__":
    # Smoke check: 1000 envs with fixed block_lengths = 100 each.
    # Verify marginal P(reward | arm=good) matches the schedule.
    n_envs   = 1000
    block_T  = np.full(N_BLOCKS, 100, dtype=int)

    hits_by_prob = {p: [0, 0] for p in GOOD_PROBS}   # [n_rewards, n_trials]
    bad_hits = [0, 0]

    for s in range(n_envs):
        r, ga, gp = simulate_dezfouli_bandit(seed=s, block_lengths=block_T)
        for b in range(N_BLOCKS):
            T = block_T[b]
            good = ga[b]; bad = 1 - good
            hits_by_prob[gp[b]][0] += r[b, :T, good].sum()
            hits_by_prob[gp[b]][1] += T
            bad_hits[0] += r[b, :T, bad].sum()
            bad_hits[1] += T

    print("Schedule sanity check across", n_envs, "envs:")
    for p in GOOD_PROBS:
        n_rew, n_tr = hits_by_prob[p]
        print(f"  good arm @ p={p:.3f}: observed {n_rew/n_tr:.4f}  (expected {p})")
    print(f"  bad arm  @ p={BAD_PROB:.3f}: observed {bad_hits[0]/bad_hits[1]:.4f}  (expected {BAD_PROB})")
