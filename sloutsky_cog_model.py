import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

df = pd.read_csv('data_sloutsky/df_train.csv')
df = df.sort_values(["subid", "TrainingTrial"])
participants = df["subid"].unique()
P = len(participants)
print(f"number of participants in train split: {P}")

T = df["TrainingTrial"].nunique()

#options
K = 4

value_train = np.zeros((P, T, K))
choice = np.zeros((P,T), dtype=int)

for p_idx, p in enumerate(participants):
    df_p = df[df["subid"] == p].sort_values("TrainingTrial")

    value_train[p_idx] = df_p[["Value1", "Value2", "Value3", "Value4"]].values
    choice[p_idx] = df_p["c"].values.astype(int)

# Novelty array: option 4 (index 3) is always the novel option
novelty = np.zeros((P, T, K))
novelty[:, :, 3] = 1.0

# Lags array: trials since each option was last chosen
lags = np.zeros((P, T, K))
for p_idx in range(P):
    last_chosen = -np.ones(K)  # trial at which each option was last chosen
    for t in range(T):
        for k in range(K):
            if last_chosen[k] < 0:
                lags[p_idx, t, k] = 0  # never chosen yet
            else:
                lags[p_idx, t, k] = t - last_chosen[k]
        c = choice[p_idx, t]
        last_chosen[c] = t

def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

def softmax_with_temp(v, theta):
    v = np.asarray(v, dtype=float)
    v = theta * v
    v = v - np.max(v)
    ex = np.exp(v)
    return ex / ex.sum()

def normalize(x):
    x = np.asarray(x)
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros_like(x)
    return (x - x.min()) / denom

def update_beliefs_(y, mu, lam, alpha, beta):
    # scalar Normal-Gamma update (one observation)
    lam_new = lam + 1.0
    mu_new = (lam * mu + y) / lam_new
    alpha_new = alpha + 0.5
    beta_new = beta + 0.5 * (lam * (y - mu)**2) / lam_new
    return {"mu": mu_new, "lmbd": lam_new, "alpha": alpha_new, "beta": beta_new}

def update_beliefs(rewards, mu_0, lmbd, alpha, beta): 
    n = 1  
    mean_reward = np.mean(rewards)
    new_lmbd = lmbd+n 
    new_mu = ((lmbd*mu_0)+(n*mean_reward)) / (lmbd+n) 
    new_alpha = alpha + n / 2 
    new_beta = beta + (0.5*n* (np.sum(rewards**2)-mean_reward**2)) + (lmbd*n*(mean_reward-mu_0)**2) / (2*(lmbd+n)) 
    updated_params = { "mu": new_mu, "lmbd": new_lmbd, "alpha": new_alpha, "beta": new_beta} 
    return updated_params

def unpack_params(pars_vec):
    pars_vec = np.asarray(pars_vec, dtype=float)
    theta = np.exp(pars_vec[0])                 # >0
    w_train = softmax(pars_vec[1:5])            # 4 weights sum to 1
    return {
        "theta": theta,
        "b_value_train": w_train[0],
        "b_uncertain_train": w_train[1],
        "b_lag_train": w_train[2],
        "b_novelty_train": w_train[3],
    }

def loglik(pars_vec, value_train, novelty, lags, choice):
    sum_loglik = 0
    #test_slopes = softmax
    params = unpack_params(pars_vec)
    
    K_train = 4
    mu_0 = np.zeros(K_train)
    lmbd = np.ones(K_train)
    alpha = np.ones(K_train)
    beta = np.ones(K_train)
    uncertainty = np.zeros(K_train)

    for i in range(30):
        if i == 0:
            mu_0 = value_train[i,:]
            cur_value = normalize(mu_0)
            choice_value = cur_value * params["b_value_train"]
        else:
            for o in range(4):
                posterior = update_beliefs(value_train[i,o], mu_0[o], lmbd[o], alpha[o], beta[o])
                mu_0[o] = posterior["mu"]
                lmbd[o] = posterior["lmbd"]
                alpha[o] = posterior["alpha"]
                beta[o] = posterior["beta"]

                den = lmbd[o] * max(alpha[o] - 1.0, 1e-12)
                uncertainty[o] = np.sqrt(beta[o] / den)

            u_norm = normalize(uncertainty)
            cur_value = normalize(mu_0)
            choice_value = (
                cur_value * params["b_value_train"]
                + u_norm           * params["b_uncertain_train"]
                + lags[i, :] * params["b_lag_train"]
                + novelty[i,:] * params["b_novelty_train"]
            )
        probs = softmax_with_temp(choice_value, params["theta"])
        sum_loglik += np.log(probs[int(choice[i])] + 1e-12)
    return sum_loglik


# ============================================================
# EM fitting (hierarchical Bayesian) for the Sloutsky model
# ============================================================
N_PARAMS = 5  # pars_vec: [log_theta, w1_raw, w2_raw, w3_raw, w4_raw]


def neg_log_posterior_sloutsky(pars_vec, value_train_p, novelty_p, lags_p,
                               choice_p, prior_mean, prior_var):
    """
    Negative log-posterior for a single participant.
    posterior ∝ likelihood × prior, with independent normal priors on each
    unconstrained parameter.
    """
    ll = loglik(pars_vec, value_train_p, novelty_p, lags_p, choice_p)
    log_prior = np.sum(norm.logpdf(pars_vec, loc=prior_mean,
                                    scale=np.sqrt(prior_var)))
    return -(ll + log_prior)


def hessian_diag_nd(fun, x, eps=1e-5):
    """Diagonal of the Hessian of a scalar function via finite differences."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    f0 = fun(x)
    diag = np.zeros(n)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        diag[j] = (fun(xp) - 2 * f0 + fun(xm)) / (eps ** 2)
    return diag


def estimate_individual_sloutsky(prior_mean, prior_var,
                                  value_train_p, novelty_p, lags_p, choice_p):
    """
    E-step for one participant: find MAP estimate and posterior variance
    (Laplace approximation) for all 5 parameters.

    Returns
    -------
    h_i : ndarray (5,)   MAP estimate in unconstrained space
    var_i : ndarray (5,)  diagonal posterior variances
    """
    obj = lambda pv: neg_log_posterior_sloutsky(
        pv, value_train_p, novelty_p, lags_p, choice_p,
        prior_mean, prior_var)

    best_val = np.inf
    best_x = prior_mean.copy()

    for _ in range(5):
        x0 = prior_mean + np.random.randn(N_PARAMS) * np.sqrt(prior_var)
        res = minimize(obj, x0=x0, method='L-BFGS-B',
                       options=dict(maxiter=2000, ftol=1e-8))
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()

    h_i = best_x

    # Laplace approximation: posterior variance = 1 / diag(Hessian)
    H_diag = hessian_diag_nd(obj, h_i)
    var_i = np.where(H_diag > 1e-6, 1.0 / H_diag, prior_var)
    var_i = np.clip(var_i, 1e-4, 10.0)

    return h_i, var_i


def update_group_prior_nd(h_all, var_all):
    """
    M-step: update the group-level prior from individual posteriors.

    Parameters
    ----------
    h_all   : ndarray (P, D)  MAP estimates for each participant
    var_all : ndarray (P, D)  posterior variances for each participant

    Returns
    -------
    m : ndarray (D,)  updated prior mean
    v : ndarray (D,)  updated prior variance
    """
    m = np.mean(h_all, axis=0)
    v = np.mean(h_all ** 2 + var_all, axis=0) - m ** 2
    v = np.clip(v, 1e-4, 10.0)
    return m, v


def run_em_sloutsky(value_train, choice, novelty, lags, n_iter=50):
    """
    Run the EM algorithm for hierarchical Bayesian fitting of the
    Sloutsky cognitive model.

    Parameters
    ----------
    value_train : ndarray (P, T, K)
    choice      : ndarray (P, T)     integer choices
    novelty     : ndarray (P, T, K)
    lags        : ndarray (P, T, K)
    n_iter      : int

    Returns
    -------
    m           : ndarray (5,)   final group prior mean
    v           : ndarray (5,)   final group prior variance
    h_all       : ndarray (P, 5) individual MAP estimates from last iteration
    m_history   : ndarray (n_iter, 5)
    v_history   : ndarray (n_iter, 5)
    """
    P = value_train.shape[0]

    # Initialise group prior
    m = np.zeros(N_PARAMS)
    v = np.ones(N_PARAMS)

    m_history = []
    v_history = []
    h_all = np.zeros((P, N_PARAMS))
    var_all = np.zeros((P, N_PARAMS))

    for it in range(n_iter):
        # ---------- E-step ----------
        for p in range(P):
            h_all[p], var_all[p] = estimate_individual_sloutsky(
                m, v,
                value_train[p], novelty[p], lags[p], choice[p])

        # ---------- M-step ----------
        m, v = update_group_prior_nd(h_all, var_all)
        m_history.append(m.copy())
        v_history.append(v.copy())

        if it % 5 == 0:
            # Report transformed parameters for interpretability
            theta_pop = np.exp(m[0])
            w_pop = softmax(m[1:5])
            print(f"EM iter {it:3d}: "
                  f"theta={theta_pop:.3f}, "
                  f"w_value={w_pop[0]:.3f}, w_uncert={w_pop[1]:.3f}, "
                  f"w_lag={w_pop[2]:.3f}, w_novel={w_pop[3]:.3f}, "
                  f"v={np.round(v, 4)}")

    return m, v, h_all, np.array(m_history), np.array(v_history)


def compute_marginal_loglik_logsumexp(m, v, value_train_p, novelty_p, lags_p,
                                       choice_p, n_samples=2000, seed=42):
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
        loglik(theta_samples[s], value_train_p, novelty_p, lags_p, choice_p)
        for s in range(n_samples)
    ])

    # log p(y | m, v) ≈ logsumexp(log_liks) - log(S)
    marginal_loglik = float(np.logaddexp.reduce(log_liks) - np.log(n_samples))
    return marginal_loglik


def build_data_arrays(df):
    """Build value_train, choice, novelty, lags arrays from a participant dataframe."""
    participants_loc = df["subid"].unique()
    P_loc = len(participants_loc)
    T_loc = df["TrainingTrial"].nunique()

    value_train_loc = np.zeros((P_loc, T_loc, K))
    choice_loc = np.zeros((P_loc, T_loc), dtype=int)

    for p_idx, p in enumerate(participants_loc):
        df_p = df[df["subid"] == p].sort_values("TrainingTrial")
        value_train_loc[p_idx] = df_p[["Value1", "Value2", "Value3", "Value4"]].values
        choice_loc[p_idx] = df_p["c"].values.astype(int)

    novelty_loc = np.zeros((P_loc, T_loc, K))
    novelty_loc[:, :, 3] = 1.0

    lags_loc = np.zeros((P_loc, T_loc, K))
    for p_idx in range(P_loc):
        last_chosen = -np.ones(K)
        for t in range(T_loc):
            for k in range(K):
                lags_loc[p_idx, t, k] = 0 if last_chosen[k] < 0 else t - last_chosen[k]
            last_chosen[choice_loc[p_idx, t]] = t

    return participants_loc, value_train_loc, choice_loc, novelty_loc, lags_loc


# ============================================================
# Main: run EM fitting
# ============================================================
if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--fold", type=int, default=None,
                         help="Outer CV fold index (0,1,2). If not set, uses "
                              "data_sloutsky/df_train.csv (original behaviour).")
    _args = _parser.parse_args()
    FOLD = _args.fold

    if FOLD is not None:
        DATA_DIR = f"data_sloutsky/fold{FOLD}"
        print(f"=== Outer CV fold {FOLD}: data from {DATA_DIR} ===")
        _df_tr = pd.read_csv(f"{DATA_DIR}/df_train.csv").sort_values(["subid", "TrainingTrial"])
        participants_main, value_train_main, choice_main, novelty_main, lags_main = \
            build_data_arrays(_df_tr)
    else:
        DATA_DIR = "data_sloutsky"
        participants_main = participants
        value_train_main  = value_train
        choice_main       = choice
        novelty_main      = novelty
        lags_main         = lags

    P_main = len(participants_main)

    # ----------------------------------------------------------
    # Step 1: EM on training participants only
    # ----------------------------------------------------------
    print(f"Training participants: {P_main}, Options: {K}")

    m, v, h_all_train, m_history, v_history = run_em_sloutsky(
        value_train_main, choice_main, novelty_main, lags_main, n_iter=50)

    print("\n=== Final group prior (from training participants) ===")
    theta_pop = np.exp(m[0])
    w_pop = softmax(m[1:5])
    print(f"  theta = {theta_pop:.4f}")
    print(f"  w_value     = {w_pop[0]:.4f}")
    print(f"  w_uncertain = {w_pop[1]:.4f}")
    print(f"  w_lag       = {w_pop[2]:.4f}")
    print(f"  w_novelty   = {w_pop[3]:.4f}")
    print(f"  prior variance = {v}")

    # ----------------------------------------------------------
    # Step 2: MAP fit for test participants using the fitted prior
    # ----------------------------------------------------------
    print("\n=== MAP fit for test participants (prior from training EM) ===")
    df_test_raw = pd.read_csv(f'{DATA_DIR}/df_test.csv')
    df_test_raw = df_test_raw.sort_values(["subid", "TrainingTrial"])
    test_participants, value_test, choice_test, novelty_test, lags_test = \
        build_data_arrays(df_test_raw)

    P_test = len(test_participants)
    h_all_test = np.zeros((P_test, N_PARAMS))
    var_all_test = np.zeros((P_test, N_PARAMS))
    ll_test = np.zeros(P_test)

    for p in range(P_test):
        h_all_test[p], var_all_test[p] = estimate_individual_sloutsky(
            m, v,
            value_test[p], novelty_test[p], lags_test[p], choice_test[p])
        ll_test[p] = loglik(
            h_all_test[p], value_test[p], novelty_test[p], lags_test[p], choice_test[p])
        params = unpack_params(h_all_test[p])
        print(f"  P{p:3d} (subid={test_participants[p]}): "
              f"theta={params['theta']:.3f}, "
              f"w=[{params['b_value_train']:.3f}, "
              f"{params['b_uncertain_train']:.3f}, "
              f"{params['b_lag_train']:.3f}, "
              f"{params['b_novelty_train']:.3f}], "
              f"LL={ll_test[p]:.2f}")

    print(f"\nMean test log-likelihood (MAP): {ll_test.mean():.3f} ± {ll_test.std():.3f}")

    # ----------------------------------------------------------
    # Step 2b: Marginal log-likelihood for test participants
    #          via Monte Carlo logsumexp over the group prior
    # ----------------------------------------------------------
    print("\n=== Marginal log-likelihood for test participants (logsumexp) ===")
    marginal_ll_test = np.zeros(P_test)

    for p in range(P_test):
        marginal_ll_test[p] = compute_marginal_loglik_logsumexp(
            m, v,
            value_test[p], novelty_test[p], lags_test[p], choice_test[p],
            n_samples=2000, seed=p)
        print(f"  P{p:3d} (subid={test_participants[p]}): "
              f"marginal LL = {marginal_ll_test[p]:.2f}  "
              f"(MAP LL = {ll_test[p]:.2f})")

    # Marginal NLL (to match the format of rnn_resultslatentmodel.csv)
    marginal_nll_test = -marginal_ll_test
    print(f"\nMean marginal NLL: {marginal_nll_test.mean():.3f} ± {marginal_nll_test.std():.3f}")

    # Save EM (individual MAP via EM prior) NLL — include subid for alignment
    cog_model_df = pd.DataFrame({
        "subid": test_participants,
        "normalized_likelihood": marginal_nll_test,
        "model": "CogModel_EM",
    })
    cog_model_df.to_csv(f"{DATA_DIR}/cog_model_results.csv", index=False)
    print(f"EM NLL saved to {DATA_DIR}/cog_model_results.csv")

    # ----------------------------------------------------------
    # Step 2c: Population (CP) fit — one shared parameter set
    #          for all training participants via joint optimisation
    # ----------------------------------------------------------
    print("\n=== Population (CP) fit — joint optimisation over all train participants ===")

    def neg_joint_ll(pars_vec):
        total = 0.0
        for p_idx in range(P_main):
            total -= loglik(pars_vec, value_train_main[p_idx], novelty_main[p_idx],
                            lags_main[p_idx], choice_main[p_idx])
        return total

    best_val_cp, best_pars_cp = np.inf, np.zeros(N_PARAMS)
    for _ in range(10):
        x0  = np.random.randn(N_PARAMS) * 0.5
        res = minimize(neg_joint_ll, x0=x0, method="L-BFGS-B",
                       options=dict(maxiter=5000, ftol=1e-9))
        if res.fun < best_val_cp:
            best_val_cp, best_pars_cp = res.fun, res.x.copy()

    cp_params = unpack_params(best_pars_cp)
    print(f"  theta={cp_params['theta']:.4f}, "
          f"w_value={cp_params['b_value_train']:.4f}, "
          f"w_uncert={cp_params['b_uncertain_train']:.4f}, "
          f"w_lag={cp_params['b_lag_train']:.4f}, "
          f"w_novelty={cp_params['b_novelty_train']:.4f}")

    cp_nll_test = np.array([
        -loglik(best_pars_cp, value_test[p], novelty_test[p],
                lags_test[p], choice_test[p])
        for p in range(P_test)
    ])
    print(f"  Test CP NLL: {cp_nll_test.mean():.3f} ± {cp_nll_test.std():.3f}")

    pd.DataFrame({
        "subid": test_participants,
        "normalized_likelihood": cp_nll_test,
        "model": "CogModel_CP",
    }).to_csv(f"{DATA_DIR}/cog_model_cp_results.csv", index=False)
    print(f"CP NLL saved to {DATA_DIR}/cog_model_cp_results.csv")

    # ----------------------------------------------------------
    # Step 3: Save everything
    # ----------------------------------------------------------
    np.savez(
        f"{DATA_DIR}/em_results.npz",
        # Group prior (estimated from train participants)
        m=m,
        v=v,
        m_history=m_history,
        v_history=v_history,
        # Train participants MAP estimates
        h_all_train=h_all_train,
        participants_train=participants_main,
        # Test participants MAP estimates (using train-fitted prior)
        h_all_test=h_all_test,
        var_all_test=var_all_test,
        participants_test=test_participants,
        ll_test=ll_test,
        # Combined (for plotting all participants together)
        h_all=np.concatenate([h_all_train, h_all_test], axis=0),
        participants=np.concatenate([participants_main, test_participants]),
    )
    print(f"\nEM results saved to {DATA_DIR}/em_results.npz")
