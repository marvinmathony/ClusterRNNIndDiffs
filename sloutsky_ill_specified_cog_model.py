#!/usr/bin/env python3
"""
Ill-specified cognitive model for the Sloutsky data.

Q-learning model: learns values from received reward only (column "r").
No uncertainty, lag, or novelty terms.
Parameters: [log_theta]  (softmax temperature only; alpha fixed to ALPHA_FIXED)

Two fits:
  1. Population (CP): one shared [alpha, theta] across all train participants
     via scipy.optimize.minimize on the joint negative log-likelihood.
     Applied to test participants as-is (no individual fitting).
  2. Individual (MAP via EM): per-participant [alpha, theta] using the
     hierarchical EM machinery from sloutsky_cog_model.py.

Saves:
  data_sloutsky/ill_specified_cp_results.csv   (population NLL on test set)
  data_sloutsky/ill_specified_map_results.csv  (individual MAP NLL on test set)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sloutsky_cog_model import (
    softmax_with_temp, hessian_diag_nd, build_data_arrays,
)

K = 4
N_PARAMS = 1         # [log_theta] only — alpha is fixed
ALPHA_FIXED = 1   # assumed learning rate


# ── Model ─────────────────────────────────────────────────────────────────────
def loglik_ill(pars_vec, choice_p, reward_p):
    """
    Q-learning log-likelihood for one participant.
    Uses only received reward to update Q-values.
    pars_vec: [log_theta]  (alpha is fixed to ALPHA_FIXED)
    """
    alpha = ALPHA_FIXED
    theta = np.exp(pars_vec[0])
    T     = len(choice_p)

    Q      = np.zeros(K)
    sum_ll = 0.0

    for t in range(T):
        probs   = softmax_with_temp(Q, theta)
        c       = int(choice_p[t])
        sum_ll += np.log(probs[c] + 1e-12)
        Q[c]   += alpha*(reward_p[t] - Q[c])   # update only chosen arm

    return sum_ll


# ── Data loading ───────────────────────────────────────────────────────────────
def build_data_with_rewards(df_in):
    """Extend build_data_arrays to also extract per-participant reward sequences."""
    participants = df_in["subid"].unique()
    T = df_in["TrainingTrial"].nunique()

    value_arr  = np.zeros((len(participants), T, K))
    choice_arr = np.zeros((len(participants), T), dtype=int)
    reward_arr = np.zeros((len(participants), T))

    for p_idx, p in enumerate(sorted(participants)):
        df_p = df_in[df_in["subid"] == p].sort_values("TrainingTrial")
        value_arr[p_idx]  = df_p[["Value1", "Value2", "Value3", "Value4"]].values
        choice_arr[p_idx] = df_p["c"].values.astype(int)
        reward_arr[p_idx] = df_p["r"].values

    novelty = np.zeros_like(value_arr)
    lags    = np.zeros_like(value_arr)
    return sorted(participants), value_arr, choice_arr, novelty, lags, reward_arr


# ── Population (CP) fit ───────────────────────────────────────────────────────
def fit_population(value_train, choice_train, reward_train, n_restarts=10):
    """
    Fit one [log_alpha, log_theta] shared across all training participants.
    Returns best pars_vec and the per-participant NLLs evaluated at that point.
    """
    def neg_joint_ll(pars_vec):
        total = 0.0
        for p in range(value_train.shape[0]):
            total -= loglik_ill(pars_vec, choice_train[p], reward_train[p])
        return total

    best_val, best_x = np.inf, np.zeros(N_PARAMS)
    for _ in range(n_restarts):
        x0  = np.random.randn(N_PARAMS) * 0.5
        res = minimize(neg_joint_ll, x0=x0, method="L-BFGS-B",
                       options=dict(maxiter=5000, ftol=1e-9))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()

    theta_hat = np.exp(best_x[0])
    print(f"Population fit: , theta={theta_hat:.4f}  "
          f"(neg joint LL={best_val:.2f})")
    return best_x


# ── EM / individual MAP fit ────────────────────────────────────────────────────
def neg_log_posterior_ill(pars_vec, value_p, choice_p, reward_p, prior_mean, prior_var):
    ll        = loglik_ill(pars_vec, choice_p, reward_p)
    log_prior = np.sum(norm.logpdf(pars_vec, loc=prior_mean, scale=np.sqrt(prior_var)))
    return -(ll + log_prior)


def estimate_individual_ill(prior_mean, prior_var, value_p, choice_p, reward_p):
    obj = lambda pv: neg_log_posterior_ill(pv, value_p, choice_p, reward_p,
                                           prior_mean, prior_var)
    best_val, best_x = np.inf, prior_mean.copy()
    for _ in range(5):
        x0  = prior_mean + np.random.randn(N_PARAMS) * np.sqrt(prior_var)
        res = minimize(obj, x0=x0, method="L-BFGS-B",
                       options=dict(maxiter=2000, ftol=1e-8))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()

    H_diag = hessian_diag_nd(obj, best_x)
    var_i  = np.where(H_diag > 1e-6, 1.0 / H_diag, prior_var)
    var_i  = np.clip(var_i, 1e-4, 10.0)
    return best_x, var_i


def run_em_ill(value_train, choice_train, reward_train, n_iter=50):
    P_loc = value_train.shape[0]
    m = np.zeros(N_PARAMS)
    v = np.ones(N_PARAMS)
    h_all   = np.zeros((P_loc, N_PARAMS))
    var_all = np.zeros((P_loc, N_PARAMS))

    for it in range(n_iter):
        for p in range(P_loc):
            h_all[p], var_all[p] = estimate_individual_ill(
                m, v, value_train[p], choice_train[p], reward_train[p])
        m = np.mean(h_all, axis=0)
        v = np.clip(np.mean(h_all**2 + var_all, axis=0) - m**2, 1e-4, 10.0)
        if it % 5 == 0:
            print(f"EM iter {it:3d}: , "
                  f"theta={np.exp(m[0]):.3f}  v={np.round(v, 4)}")
    return m, v, h_all

def compute_marginal_loglik_logsumexp(m, v, choice_p, reward_p, n_samples=2000, seed=42):
    """
    Monte Carlo estimate of the marginal log-likelihood for one participant:

        log p(y_j | m, v) = log ∫ p(y_j | θ) p(θ | m, v) dθ
                          ≈ logsumexp_s [ loglik(y_j | θ^(s)) ] - log(S)

    where θ^(s) ~ N(m, diag(v))  (prior in unconstrained space).

    Parameters
    ----------
    m, v        : ndarray (D,)  group prior mean and variance (from EM on train)
    n_samples   : int           number of Monte Carlo draws

    Returns
    -------
    marginal_loglik : float   log p(y_j | m, v)
    """
    rng = np.random.default_rng(seed)
    std = np.sqrt(v)
    # Draw samples from the group prior
    theta_samples = rng.normal(loc=m, scale=std, size=(n_samples, len(m)))

    # Evaluate log-likelihood for each sample
    log_liks = np.array([
        loglik_ill(theta_samples[s], choice_p, reward_p)
        for s in range(n_samples)
    ])

    # log p(y | m, v) ≈ logsumexp(log_liks) - log(S)
    marginal_loglik = float(np.logaddexp.reduce(log_liks) - np.log(n_samples))
    return marginal_loglik

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--fold", type=int, default=None,
                         help="Outer CV fold index (0,1,2). If not set, uses "
                              "data_sloutsky/df_train.csv (original behaviour).")
    _args = _parser.parse_args()
    FOLD = _args.fold
    DATA_DIR = f"data_sloutsky/fold{FOLD}" if FOLD is not None else "data_sloutsky"
    if FOLD is not None:
        print(f"=== Outer CV fold {FOLD}: data from {DATA_DIR} ===")

    df_train = pd.read_csv(f"{DATA_DIR}/df_train.csv").sort_values(["subid", "TrainingTrial"])
    df_test  = pd.read_csv(f"{DATA_DIR}/df_test.csv").sort_values(["subid", "TrainingTrial"])

    _, val_tr, ch_tr, _, _, rew_tr = build_data_with_rewards(df_train)
    test_participants, val_te, ch_te, _, _, rew_te = build_data_with_rewards(df_test)
    P_test = len(test_participants)

    # ── 1. Population (CP) fit ────────────────────────────────────────────────
    print("=== Population (CP) fit ===")
    cp_pars = fit_population(val_tr, ch_tr, rew_tr, n_restarts=10)

    cp_nll_test = np.array([
        -loglik_ill(cp_pars, ch_te[p], rew_te[p])
        for p in range(P_test)
    ])
    print(f"Test CP NLL: {cp_nll_test.mean():.3f} ± {cp_nll_test.std():.3f}")
    pd.DataFrame({
        "subid":                 test_participants,
        "normalized_likelihood": cp_nll_test,
        "model":                 "IllSpecified_CP",
    }).to_csv(f"{DATA_DIR}/ill_specified_cp_results.csv", index=False)
    print(f"Saved → {DATA_DIR}/ill_specified_cp_results.csv")

    # ── 2. Individual MAP fit via EM ──────────────────────────────────────────
    # Run EM on training participants to get the population prior (m, v), then
    # compute marginal log-likelihood for each test participant under that prior.
    print("\n=== EM / individual MAP fit ===")
    m, v, _ = run_em_ill(val_tr, ch_tr, rew_tr, n_iter=50)
    print(f"EM population mean: theta={np.exp(m[0]):.3f}")

    marginal_nll_test = np.full(P_test, np.nan)
    for p in range(P_test):
        ch_p  = ch_te[p]
        rew_p = rew_te[p]
        marginal_loglik = compute_marginal_loglik_logsumexp(m, v, ch_p, rew_p,
                                                            n_samples=2000, seed=p)
        marginal_nll_test[p] = -marginal_loglik
        print(f"  P{p:3d} (subid={test_participants[p]}): "
              f"marginal NLL = {marginal_nll_test[p]:.3f}")

    valid = np.isfinite(marginal_nll_test)
    print(f"Mean marginal NLL ({valid.sum()}/{P_test} valid): "
          f"{marginal_nll_test[valid].mean():.3f} ± {marginal_nll_test[valid].std():.3f}")

    pd.DataFrame({
        "subid":                 test_participants,
        "normalized_likelihood": marginal_nll_test,
        "model":                 "IllSpecified_MAP",
    }).to_csv(f"{DATA_DIR}/ill_specified_map_results.csv", index=False)
    print(f"Saved → {DATA_DIR}/ill_specified_map_results.csv")
