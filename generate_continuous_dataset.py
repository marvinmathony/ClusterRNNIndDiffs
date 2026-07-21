"""
generate_continuous_dataset.py — Build a uniform-α synthetic dataset where
every session faces a *continuous* drifting binary bandit (Gaussian random
walk on per-arm Bernoulli probabilities), not the discrete low/normal/high
mixture used in data_dataset{0..20}.

Output goes to data_continuous_dataset{ID}/ in the same file layout as
data_dataset{ID}, so train_synthetic_step1.py and the three-regression
analyzer can ingest it transparently:
  rewards_train.npy        (N, 2, T)
  xin_train.npy            (N, T, 4)
  c_train.npy              (N, T)
  choice_one_hot_train.npy (N, T, 2)
  df_train.csv
  true_parameter_values.csv  (per-session α)
  params_dict.pkl            (Q_MAP[session] = [α, β])

Usage: python generate_continuous_dataset.py --dataset_id 0
"""
import os, pickle, argparse
import numpy as np
import pandas as pd

import sim_Q_data as sim


N_SESSIONS    = 200
N_TRIALS      = 200
SIGMA_RW      = 0.10     # mean volatility of per-arm Bernoulli probability
SIGMA_PART    = 0.00     # 0 → every session same drift volatility (signal only from α)
INIT_MEAN     = 0.5
ALPHA_LO      = 0.1
ALPHA_HI      = 0.9
BETA          = 3.0


def gen_continuous_envs(n_sessions, n_trials, sigma_rw, sigma_part, init_mean,
                        seed):
    """Drifting binary bandit envs — one per session.
    Returns rewards (N, 2, T) Bernoulli {0, 1}, and the underlying p (N, 2, T)."""
    rng = np.random.default_rng(seed)
    sigmas = np.abs(rng.normal(sigma_rw, sigma_part, n_sessions))
    rewards = np.zeros((n_sessions, 2, n_trials), dtype=np.float32)
    p_true  = np.zeros((n_sessions, 2, n_trials), dtype=np.float32)
    for n in range(n_sessions):
        mu = np.array([init_mean, 1.0 - init_mean], dtype=float)
        for t in range(n_trials):
            p = np.clip(mu, 0.001, 0.999)
            p_true[n, :, t] = p
            rewards[n, 0, t] = float(rng.random() < p[0])
            rewards[n, 1, t] = float(rng.random() < p[1])
            mu = mu + rng.normal(0.0, sigmas[n], size=2)
            mu = np.clip(mu, 0.001, 0.999)
    return rewards, p_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=0)
    args = ap.parse_args()

    OUT = f"data_continuous_dataset{args.dataset_id}"
    PLOT = f"plots_continuous_dataset{args.dataset_id}"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PLOT, exist_ok=True)

    seed_base = 1 + args.dataset_id * 1000
    print(f"Output: {OUT}   seed_base={seed_base}")

    # ── Envs (drifting binary bandit, continuous on per-arm p) ──────────────
    rewardsTrain, p_train = gen_continuous_envs(
        N_SESSIONS, N_TRIALS, SIGMA_RW, SIGMA_PART, INIT_MEAN,
        seed=seed_base,
    )
    rewardsTest, _ = gen_continuous_envs(
        N_SESSIONS, N_TRIALS, SIGMA_RW, SIGMA_PART, INIT_MEAN,
        seed=seed_base + 1,
    )
    np.save(f"{OUT}/rewards_train.npy", rewardsTrain)
    np.save(f"{OUT}/p_true_train.npy",  p_train)   # underlying p, for regret
    np.save(f"{OUT}/rewards_test.npy",  rewardsTest)
    print(f"  envs train: {rewardsTrain.shape}, "
          f"max-p mean across sessions: "
          f"{p_train.max(axis=1).mean(axis=1).mean():.3f} ± "
          f"{p_train.max(axis=1).mean(axis=1).std():.3f}")

    # ── Per-session α from Uniform(0.1, 0.9); β fixed ───────────────────────
    rng = np.random.default_rng(seed_base + 5)
    alphas_train = rng.uniform(ALPHA_LO, ALPHA_HI, N_SESSIONS)
    alphas_test  = rng.uniform(ALPHA_LO, ALPHA_HI, N_SESSIONS)
    betas        = np.full(N_SESSIONS, BETA)
    print(f"  α range = [{alphas_train.min():.3f}, {alphas_train.max():.3f}], "
          f"mean = {alphas_train.mean():.3f}")

    true_param = {
        "alphaP_list": alphas_train.tolist(),
        "alphaN_list": alphas_train.tolist(),
        "alphaF_list": [0.0] * N_SESSIONS,
        "beta_list":   betas.tolist(),
        "phi_list":    [0.0] * N_SESSIONS,
        "tau_list":    [0.0] * N_SESSIONS,
    }
    true_param_test = {
        "alphaP_list": alphas_test.tolist(),
        "alphaN_list": alphas_test.tolist(),
        "alphaF_list": [0.0] * N_SESSIONS,
        "beta_list":   betas.tolist(),
        "phi_list":    [0.0] * N_SESSIONS,
        "tau_list":    [0.0] * N_SESSIONS,
    }
    pd.DataFrame(true_param).to_csv(f"{OUT}/true_parameter_values.csv", index=False)
    pd.DataFrame(true_param_test).to_csv(f"{OUT}/true_test_parameter_values.csv", index=False)

    # ── Simulate Q-learning agents on these envs ────────────────────────────
    print("simulating training data")
    c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, *_ = sim.simulate_Qlearning(
        rewards=rewardsTrain, seed=seed_base + 2,
        n_sessions=N_SESSIONS, n_trials=N_TRIALS,
        alphaP_list=true_param["alphaP_list"],
        alphaN_list=true_param["alphaN_list"],
        beta=true_param["beta_list"],
        alphaF_list=true_param["alphaF_list"],
        phi_list=true_param["phi_list"],
        tau_list=true_param["tau_list"], static=False,
    )
    print("simulating test data")
    _test_out = sim.simulate_Qlearning(
        rewards=rewardsTest, seed=seed_base + 10,
        n_sessions=N_SESSIONS, n_trials=N_TRIALS,
        alphaP_list=true_param_test["alphaP_list"],
        alphaN_list=true_param_test["alphaN_list"],
        beta=true_param_test["beta_list"],
        alphaF_list=true_param_test["alphaF_list"],
        phi_list=true_param_test["phi_list"],
        tau_list=true_param_test["tau_list"], static=False,
    )
    c_test = _test_out[0]
    pA_test = _test_out[2]
    df_test = _test_out[5]
    xin_test = _test_out[6]
    choice_one_hot_test = _test_out[7]

    df_train.to_csv(f"{OUT}/df_train.csv", index=False)
    df_test.to_csv(f"{OUT}/df_test.csv", index=False)
    np.save(f"{OUT}/xin_train.npy", xin_train)
    np.save(f"{OUT}/xin_test.npy",  xin_test)
    np.save(f"{OUT}/choice_one_hot_train.npy", choice_one_hot_train)
    np.save(f"{OUT}/choice_one_hot_test.npy",  choice_one_hot_test)
    np.save(f"{OUT}/c_train.npy", c)
    np.save(f"{OUT}/c_test.npy",  c_test)
    np.save(f"{OUT}/pA_train.npy", pA)
    np.save(f"{OUT}/pA_test.npy",  pA_test)

    # ── params_dict.pkl in same key layout the analyser expects ─────────────
    Q_MAP = {i: [float(alphas_train[i]), float(BETA)] for i in range(N_SESSIONS)}
    Q_common = np.array([float(np.mean(alphas_train)), float(BETA)])
    params_dict = {
        "Q_common":  Q_common,
        "Q_MAP":     Q_MAP,
        "FQ_common": Q_common,
        "FQ_MAP":    Q_MAP,
    }
    with open(f"{OUT}/params_dict.pkl", "wb") as f:
        pickle.dump(params_dict, f)

    print(f"\n✅ Continuous-env dataset saved → {OUT}/")
    print(f"   xin_train: {xin_train.shape}, "
          f"choice_one_hot_train: {choice_one_hot_train.shape}, "
          f"rewards_train: {rewardsTrain.shape}")


if __name__ == "__main__":
    main()
