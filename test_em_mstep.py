"""
Minimal test to verify the EM algorithm with M-step for shared parameters.
Run with: python test_em_mstep.py
"""

import numpy as np
import pandas as pd
from RL_fittingfunctions2 import (
    run_empirical_bayes,
    fit_qlearning_common,
    qlearning_full,
    get_param_indices,
    sigmoid
)

def generate_synthetic_data(n_sessions=10, n_trials=50, true_beta=3.0, true_alphas=None):
    """Generate synthetic Q-learning data with known parameters."""
    np.random.seed(42)

    if true_alphas is None:
        # Generate individual alphas from a distribution
        true_alphas = np.clip(np.random.normal(0.3, 0.1, n_sessions), 0.05, 0.95)

    data = []
    for s in range(n_sessions):
        alpha = true_alphas[s]
        Q = np.array([0.5, 0.5])  # Initial Q-values

        for t in range(n_trials):
            # Softmax choice probabilities
            p0 = np.exp(true_beta * Q[0]) / (np.exp(true_beta * Q[0]) + np.exp(true_beta * Q[1]))
            choice = 0 if np.random.rand() < p0 else 1

            # Reward: option 0 has 70% reward prob, option 1 has 30%
            reward_probs = [0.7, 0.3]
            reward = 1.0 if np.random.rand() < reward_probs[choice] else 0.0

            data.append({
                'session': s,
                'c': choice,
                'r': reward,
                'context': 0
            })

            # Q-learning update
            Q[choice] += alpha * (reward - Q[choice])

    return pd.DataFrame(data), true_alphas, true_beta


def main():
    print("=" * 60)
    print("Testing EM algorithm with M-step for shared parameters")
    print("=" * 60)

    # Generate synthetic data
    n_sessions = 8
    n_trials = 100
    true_beta = 4.0
    true_alphas = np.clip(np.random.normal(0.25, 0.08, n_sessions), 0.05, 0.95)

    print(f"\nTrue parameters:")
    print(f"  True beta: {true_beta}")
    print(f"  True alphas: {true_alphas}")
    print(f"  Mean true alpha: {np.mean(true_alphas):.4f}")

    df, _, _ = generate_synthetic_data(n_sessions, n_trials, true_beta, true_alphas)

    # Model config (simple model: just alpha and beta)
    model_config = {
        "asymmetric_alpha": False,
        "forgetting_type": "none",
        "choice_trace": False,
        "init_Q_free": False
    }

    # Step 1: Fit common parameters (baseline)
    print("\n" + "-" * 60)
    print("Step 1: Fitting common parameters (all participants share same alpha)")
    common_params, common_negll, _ = fit_qlearning_common(df, model_config, n_iter=5)
    idxs = get_param_indices(model_config)
    print(f"  Common fit alpha: {common_params[idxs['alpha']]:.4f}")
    print(f"  Common fit beta:  {common_params[idxs['beta']]:.4f}")
    print(f"  Common fit neg LL: {common_negll:.2f}")

    # Step 2: Run EM with M-step
    print("\n" + "-" * 60)
    print("Step 2: Running EM with M-step (few iterations for speed)")
    m, v, eta_vec, var_vec, m_history, v_history, shared_params = run_empirical_bayes(
        df, model_config, common_params, n_iter=10
    )

    # Convert eta to alpha
    em_alphas = sigmoid(eta_vec)

    print(f"\n  EM results:")
    print(f"    Population mean (m in eta space): {m:.4f}")
    print(f"    Population variance (v): {v:.4f}")
    print(f"    Population mean alpha: {sigmoid(m):.4f}")
    print(f"    EM-fitted shared beta: {shared_params[idxs['beta']]:.4f}")
    print(f"    Individual etas from EM: {eta_vec.flatten()}")
    print(f"    Posterior variances: {var_vec.flatten()}")
    print(f"    Individual alphas from EM: {em_alphas.flatten()}")

    # Step 3: Compare
    print("\n" + "-" * 60)
    print("Step 3: Comparison")
    print(f"  True beta:        {true_beta:.4f}")
    print(f"  Common fit beta:  {common_params[idxs['beta']]:.4f}")
    print(f"  EM M-step beta:   {shared_params[idxs['beta']]:.4f}")
    print()
    print(f"  True mean alpha:     {np.mean(true_alphas):.4f}")
    print(f"  Common fit alpha:    {common_params[idxs['alpha']]:.4f}")
    print(f"  EM mean alpha:       {np.mean(em_alphas):.4f}")

    # Step 4: Verify parameters are different
    print("\n" + "-" * 60)
    print("Step 4: Verify M-step is working (params should differ from common fit)")
    beta_changed = not np.isclose(common_params[idxs['beta']], shared_params[idxs['beta']], atol=1e-3)
    print(f"  Beta changed from common fit: {beta_changed}")
    print(f"    Common: {common_params[idxs['beta']]:.4f} -> EM: {shared_params[idxs['beta']]:.4f}")

    # Step 5: Compute total likelihood with EM params vs common params
    print("\n" + "-" * 60)
    print("Step 5: Compare total negative log-likelihood")

    total_negll_common = 0
    total_negll_em = 0

    for s, (session, group) in enumerate(df.groupby('session')):
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values

        # Common params
        negll_c, _ = qlearning_full(common_params, sessions, choices, rewards, context, model_config)
        total_negll_common += negll_c

        # EM params (individual alpha + shared beta)
        em_param = np.array(shared_params, copy=True)
        em_param[idxs['alpha']] = em_alphas[s]
        negll_em, _ = qlearning_full(em_param, sessions, choices, rewards, context, model_config)
        total_negll_em += negll_em

    print(f"  Total neg LL with common params: {total_negll_common:.2f}")
    print(f"  Total neg LL with EM params:     {total_negll_em:.2f}")
    print(f"  Improvement: {total_negll_common - total_negll_em:.2f}")

    if total_negll_em < total_negll_common:
        print("\n  ✓ EM params give better fit (as expected)")
    else:
        print("\n  ⚠ EM params did not improve fit - may need more iterations")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
