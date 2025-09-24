
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import torch
from scipy.special import logsumexp
from scipy.stats import truncnorm


# Simulation settings
n_participants = 100
n_blocks = 10
n_trials = 200
total_trials = n_participants * n_blocks * n_trials
noise_variance = 1
innov_variance = 100
reward_sd = np.sqrt(noise_variance)
emp_bayes_iterations = 30
marginal_iterations = 100
with_id = False  # Whether to simulate with individual differences


# Update HybridAgent_opt_sim to use norm.cdf (probit) instead of sigmoid
class HybridAgent_opt_sim_probit:
    def __init__(self, beta, gamma, n_trials):
        self.beta = beta
        self.gamma = gamma
        self.n_trials = n_trials + 1
        self.initial_var = 10
        self.reset()

    def reset(self):
        self.mean = np.zeros((2, self.n_trials))
        self.var = np.ones((2, self.n_trials)) * 10

    def reset_block(self, t):
        self.mean[:, t] = 0
        self.var[:, t] = self.initial_var

    def get_probs(self, t):
        V_t = self.mean[0, t] - self.mean[1, t]
        sigma1 = self.var[0, t]
        sigma2 = self.var[1, t]
        std_dev = np.sqrt(sigma1 + sigma2)
        RU = np.sqrt(sigma1) - np.sqrt(sigma2)
        # Synthetic anchor: 10% of trials where RU is the only signal
        #if np.random.rand() < 0.1:
            #z = self.gamma * RU * 10.0  # Turn off Beta's influence
        #else:
        z = self.beta * V_t / std_dev + self.gamma * RU
        #print(norm.cdf(z))  # DEBUG
        return norm.cdf(z) 

    def step(self, action, reward, t):
        
        #if t % 10 == 0:
        #    self.reset_block(t)  # Reset beliefs every 10 trials
        if action == 0:
            k = self.var[0, t] / (self.var[0, t] + noise_variance + 1e-10)
            self.mean[0, t + 1] = self.mean[0, t] + k * (reward - self.mean[0, t])
            self.var[0, t+1] = np.maximum((1 - k) * self.var[0, t], 0.1)
            self.mean[1, t + 1] = self.mean[1, t]
            self.var[1, t+1] = self.var[1, t]
        else:
            k = self.var[1, t] / (self.var[1, t] + noise_variance)
            self.mean[1, t + 1] = self.mean[1, t] + k * (reward - self.mean[1, t])
            self.var[1, t+1] = np.maximum((1 - k) * self.var[1, t], 0.1)
            self.mean[0, t + 1] = self.mean[0, t]
            self.var[0, t + 1] = self.var[0, t]

def truncated_normal(mean, std, low=0.01, high=np.inf, size=None):
    """Sample from a truncated normal distribution."""
    a = (low - mean) / std  # Lower bound in standard normal space
    b = (high - mean) / std  # Upper bound
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# Re-simulate both datasets using probit-based HybridAgent
# Define simulation function with random walk
def simulate_data_probit_random_walk(with_id=True, tau_sq=4, fixed_beta=1, fixed_gamma=2):
    data = []
    tau = np.sqrt(tau_sq)
    for pid in range(n_participants):
        #beta = truncated_normal(mean=3, std=1.5) if with_id else fixed_beta
        #gamma = truncated_normal(mean=3, std=1.5) if with_id else fixed_gamma
        beta = np.random.normal(loc=3, scale=1) if with_id else 3.0  # truncated/clipped if needed
        gamma = np.random.normal(loc=2, scale=0.7) if with_id else 2.0
        
        agent = HybridAgent_opt_sim_probit(beta, gamma, n_blocks * n_trials + 1)

        t = 0
        for block in range(n_blocks):
            agent.reset_block(t)  # Reset agent beliefs at block start

            mu0 = np.random.normal(0, np.sqrt(innov_variance))  # stable arm
            mu1_list = [np.random.normal(0, np.sqrt(innov_variance))]  # fluctuating arm

            for trial in range(1, n_trials):
                mu1_list.append(np.random.normal(mu1_list[-1], tau))

            for trial in range(n_trials):
                mu1 = mu1_list[trial]
                prob0 = agent.get_probs(t)
                action = np.random.choice([0, 1], p=[prob0, 1 - prob0])
                reward = np.random.normal(mu0 if action == 0 else mu1, reward_sd)

                sigma1 = agent.var[0, t]
                sigma2 = agent.var[1, t]
                RU = np.sqrt(sigma1) - np.sqrt(sigma2)
                TU = np.sqrt(sigma1 + sigma2)
                V_t = agent.mean[0, t] - agent.mean[1, t]

                data.append({
                    "participant": pid,
                    "block": block,
                    "trial": trial,
                    "state": t,
                    "action": action,
                    "reward": reward,
                    "mu0": mu0,
                    "mu1": mu1,
                    "beta_true": beta,
                    "gamma_true": gamma,
                    "posterior_mean_0": agent.mean[0, t],
                    "posterior_mean_1": agent.mean[1, t],
                    "posterior_var_0": sigma1,
                    "posterior_var_1": sigma2,
                    "RU": RU,
                    "TU": TU,
                    "V_t": V_t,
                    "V_over_TU": V_t / (TU + 1e-10),
                    "probability_arm_0": prob0,
                    "probability_arm_1": 1 - prob0
                })

                agent.step(action, reward, t)
                t += 1
    return pd.DataFrame(data)


def simulate_data_probit_random_walk_IDtest(n_participants, n_blocks, with_id=True, tau_sq=4, fixed_beta=1, fixed_gamma=2):
    data = []
    tau = np.sqrt(tau_sq)
    for pid in range(n_participants):
        #beta = truncated_normal(mean=3, std=1.5) if with_id else fixed_beta
        #gamma = truncated_normal(mean=3, std=1.5) if with_id else fixed_gamma
        #beta = np.random.normal(loc=3, scale=1) if with_id else 3.0  # truncated/clipped if needed
        #gamma = np.random.normal(loc=2, scale=0.7) if with_id else 2.0
        beta = np.random.uniform(0.5, 5.0) if with_id else 3.0  # truncated/clipped if needed
        gamma = 2.0 #np.random.uniform(0.5, 5.0) if with_id else 2.0
        agent = HybridAgent_opt_sim_probit(beta, gamma, n_blocks * n_trials + 1)

        t = 0
        for block in range(n_blocks):
            agent.reset_block(t)  # Reset agent beliefs at block start

            mu0 = np.random.normal(0, np.sqrt(innov_variance))  # stable arm
            mu1_list = [np.random.normal(0, np.sqrt(innov_variance))]  # fluctuating arm

            for trial in range(1, n_trials):
                mu1_list.append(np.random.normal(mu1_list[-1], tau))

            for trial in range(n_trials):
                mu1 = mu1_list[trial]
                prob0 = agent.get_probs(t)
                action = np.random.choice([0, 1], p=[prob0, 1 - prob0])
                reward = np.random.normal(mu0 if action == 0 else mu1, reward_sd)

                sigma1 = agent.var[0, t]
                sigma2 = agent.var[1, t]
                TU = np.sqrt(sigma1 + sigma2)
                RU = (np.sqrt(sigma1) - np.sqrt(sigma2)) / TU
                
                V_t = agent.mean[0, t] - agent.mean[1, t]

                data.append({
                    "participant": pid,
                    "block": block,
                    "trial": trial,
                    "state": t,
                    "action": action,
                    "reward": reward,
                    "mu0": mu0,
                    "mu1": mu1,
                    "beta_true": beta,
                    "gamma_true": gamma,
                    "posterior_mean_0": agent.mean[0, t],
                    "posterior_mean_1": agent.mean[1, t],
                    "posterior_var_0": sigma1,
                    "posterior_var_1": sigma2,
                    "RU": RU,
                    "TU": TU,
                    "V_t": V_t,
                    "V_over_TU": V_t / (TU + 1e-10),
                    "probability_arm_0": prob0,
                    "probability_arm_1": 1 - prob0
                })

                agent.step(action, reward, t)
                t += 1
    return pd.DataFrame(data)

def simulate_data_probit_random_walk_individual_participant(with_id=True, fixed_beta=1, fixed_gamma=2, n_participants=1, n_blocks=50, tau_sq=4):
    data = []
    tau = np.sqrt(tau_sq)
    for pid in range(n_participants):
        #beta = truncated_normal(mean=3, std=1.5) if with_id else fixed_beta
        #gamma = truncated_normal(mean=3, std=1.5) if with_id else fixed_gamma
        #beta = np.random.normal(loc=3, scale=1) if with_id else 3.0  # truncated/clipped if needed
        #gamma = np.random.normal(loc=2, scale=0.7) if with_id else 2.0
        beta = np.random.uniform(0.5, 5.0) if with_id else 3.0  # truncated/clipped if needed
        gamma = 2 #np.random.uniform(0.5, 5.0) if with_id else 2.0
        agent = HybridAgent_opt_sim_probit(beta, gamma, n_blocks * n_trials + 1)

        t = 0
        for block in range(n_blocks):
            agent.reset_block(t)  # Reset agent beliefs at block start

            mu0 = np.random.normal(0, np.sqrt(innov_variance))  # stable arm
            mu1_list = [np.random.normal(0, np.sqrt(innov_variance))]  # fluctuating arm

            for trial in range(1, n_trials):
                mu1_list.append(np.random.normal(mu1_list[-1], tau))

            for trial in range(n_trials):
                mu1 = mu1_list[trial]
                prob0 = agent.get_probs(t)
                action = np.random.choice([0, 1], p=[prob0, 1 - prob0])
                reward = np.random.normal(mu0 if action == 0 else mu1, reward_sd)

                sigma1 = agent.var[0, t]
                sigma2 = agent.var[1, t]
                TU = np.sqrt(sigma1 + sigma2)
                RU = (np.sqrt(sigma1) - np.sqrt(sigma2)) / TU
                
                V_t = agent.mean[0, t] - agent.mean[1, t]

                data.append({
                    "participant": pid,
                    "block": block,
                    "trial": trial,
                    "state": t,
                    "action": action,
                    "reward": reward,
                    "mu0": mu0,
                    "mu1": mu1,
                    "beta_true": beta,
                    "gamma_true": gamma,
                    "posterior_mean_0": agent.mean[0, t],
                    "posterior_mean_1": agent.mean[1, t],
                    "posterior_var_0": sigma1,
                    "posterior_var_1": sigma2,
                    "RU": RU,
                    "TU": TU,
                    "V_t": V_t,
                    "V_over_TU": V_t / (TU + 1e-10),
                    "probability_arm_0": prob0,
                    "probability_arm_1": 1 - prob0
                })

                agent.step(action, reward, t)
                t += 1
    return pd.DataFrame(data)

# Log-likelihood
def negative_log_likelihood_IDtest(params, data):
    beta, gamma = params
    agent = HybridAgent_opt_sim_probit(beta, gamma, data.Global_Participant_Trial.max() + 2)
    nll = 0.0
    for row in data.itertuples():
        t = row.Global_Participant_Trial
        if t % 200 == 0:
            agent.reset_block(t)  # Reset beliefs every 10 trials
        choice = row.Action
        reward = row.Reward
        prob0 = agent.get_probs(t)
        prob1 = 1 - prob0
        prob = prob0 if choice == 0 else prob1
        #print(f"t={t}, true_action={row.action}, prob0={prob0:.3f}, prob={prob1:.3f}")  # DEBUG
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        nll -= np.log(prob)
        agent.step(choice, reward, t)
    #print(f"Final NLL: {nll}")  # DEBUG
    return nll

# Log-likelihood
def negative_log_likelihood_IDtest_onlybeta(params, data):
    beta = params
    gamma = 2
    agent = HybridAgent_opt_sim_probit(beta, gamma, data.Global_Participant_Trial.max() + 2)
    nll = 0.0
    for row in data.itertuples():
        t = row.Global_Participant_Trial
        if t % 200 == 0:
            agent.reset_block(t)  # Reset beliefs every n_trials_per_block trials
        choice = row.Action
        reward = row.Reward
        prob0 = agent.get_probs(t)
        prob1 = 1 - prob0
        prob = prob0 if choice == 0 else prob1
        #print(f"t={t}, true_action={row.action}, prob0={prob0:.3f}, prob={prob1:.3f}")  # DEBUG
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        nll -= np.log(prob)
        agent.step(choice, reward, t)
    #print(f"Final NLL: {nll}")  # DEBUG
    return nll

# Log-likelihood
def negative_log_likelihood_onlybeta(params, data):
    beta = params
    gamma = 2
    agent = HybridAgent_opt_sim_probit(beta, gamma, data.state.max() + 2)
    nll = 0.0
    for row in data.itertuples():
        t = row.state
        if t % 200 == 0:
            agent.reset_block(t)  # Reset beliefs every n_trials_per_block trials
        choice = row.action
        reward = row.reward
        prob0 = agent.get_probs(t)
        prob1 = 1 - prob0
        prob = prob0 if choice == 0 else prob1
        #print(f"t={t}, true_action={row.action}, prob0={prob0:.3f}, prob={prob1:.3f}")  # DEBUG
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        nll -= np.log(prob)
        agent.step(choice, reward, t)
    #print(f"Final NLL: {nll}")  # DEBUG
    return nll

def negative_log_likelihood(params, data):
    beta, gamma = params
    agent = HybridAgent_opt_sim_probit(beta, gamma, data.state.max() + 2)
    nll = 0.0
    for row in data.itertuples():
        t = row.state
        if t % 10 == 0:
            agent.reset_block(t)  # Reset beliefs every 10 trials
        choice = row.action
        reward = row.reward
        prob0 = agent.get_probs(t)
        prob1 = 1 - prob0
        prob = prob0 if choice == 0 else prob1
        #print(f"t={t}, true_action={row.action}, prob0={prob0:.3f}, prob={prob1:.3f}")  # DEBUG
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        nll -= np.log(prob)
        agent.step(choice, reward, t)
    #print(f"Final NLL: {nll}")  # DEBUG
    return nll

# MAP log posterior: log likelihood + log prior
def neg_log_posterior(params, data, m, v):
    ll = -negative_log_likelihood(params, data)
    prior = -0.5 * np.sum((params - m)**2 / v)
    #print(f"Params: {params}, NLL: {-ll}, Prior: {prior}")  # Debug print
    return -(ll + prior)

# Estimate h_i and diagonal Hessian (Σ_i)
#todo: make prior an input
def estimate_h_i_and_Sigma_i(data, m, v):
    bounds = None #[(0.01, 7), (0.01, 7)]
    result = minimize(
    fun=neg_log_posterior,
    x0 = np.random.normal(loc=m, scale=np.sqrt(v)), #np.random.uniform(low=[0.5, 0.5], high=[5.0, 5.0]),  # More dispersed initialization
    args=(data, m, v),
    bounds=bounds,
    method='Powell',
    options={
        'maxiter': 2000,  # Increased iterations
        'ftol': 1e-6,     # Tighter tolerance
        'gtol': 1e-6,
    }
) #this finds the MAP estimate of h_i - participant specific estimates of beta and gamma
    h_i = result.x
    eps = np.sqrt(np.finfo(float).eps)
    H = np.zeros((2, 2))
    for j in range(2):
        for k in range(2):
            e_j = np.zeros(2); e_k = np.zeros(2)
            e_j[j] = eps; e_k[k] = eps
            f1 = neg_log_posterior(h_i + e_j + e_k, data, m, v)
            f2 = neg_log_posterior(h_i + e_j - e_k, data, m, v)
            f3 = neg_log_posterior(h_i - e_j + e_k, data, m, v)
            f4 = neg_log_posterior(h_i - e_j - e_k, data, m, v)
            H[j, k] = (f1 - f2 - f3 + f4) / (4 * eps**2)
    Sigma_i = np.linalg.pinv(H)
    return h_i, Sigma_i

# EM updates
def update_group_prior(h_mat, Sigma_stack, min_variance=1e-5):
    m = np.mean(h_mat, axis=0)
    v = np.mean(h_mat**2 + np.diagonal(Sigma_stack, axis1=1, axis2=2), axis=0) - m**2
    v = np.clip(v, [1e-4, 0.4], None)
    return m, v

# EM loop
def run_empirical_bayes(train_dfs, max_iter=100):
    m = np.array([3.0, 2.0])
    v = np.array([1.0**2, 0.7**2])
    m_history = [m.copy()]
    v_history = [v.copy()]
    for _ in range(max_iter):
        h_mat = []
        Sigma_list = []
        for df in train_dfs:
            h_i, Sigma_i = estimate_h_i_and_Sigma_i(df, m, v)
            h_mat.append(h_i)
            Sigma_list.append(Sigma_i)
        h_mat = np.stack(h_mat)
        Sigma_stack = np.stack(Sigma_list)
        m, v = update_group_prior(h_mat, Sigma_stack)
        m_history.append(m.copy())
        v_history.append(v.copy())
    return m, v, h_mat, Sigma_stack, np.array(m_history), np.array(v_history)

def evaluate_log_likelihood(df_test, beta, gamma):
    """
    Evaluate log-likelihood of test data under fitted hybrid agent.
    """
    agent = HybridAgent_opt_sim_probit(beta, gamma, df_test.state.max() + 2)
    log_likelihood = 0.0
    for row in df_test.itertuples():
        t = row.state
        a = row.action
        r = row.reward
        if row.trial == 0:
            agent.reset_block(t)
        prob0 = agent.get_probs(t)
        prob = prob0 if a == 0 else 1 - prob0
        log_likelihood += np.log(np.clip(prob, 1e-10, 1 - 1e-10))
        agent.step(a, r, t)
    return log_likelihood

def evaluate_log_likelihood_agent_marginal(df, testid, n_samples, mu, sd):
    """
    Evaluate log-likelihood of test data under fitted hybrid agent.
    """
    test_pids = df['participant'].unique()[-testid:]
    df_test = df[df['participant'].isin(test_pids)].copy()
    test_dfs = [df for _, df in df_test.groupby("participant")]
    per_part_logL = []
    for df in test_dfs:
        sample_sums = np.empty(n_samples, dtype=float)
        for s in range(n_samples):
            h = np.random.normal(loc=mu, scale=sd)
            beta, gamma = float(h[0]), float(h[1])
            agent = HybridAgent_opt_sim_probit(beta, gamma, df.state.max() + 2)
            ll_sum = 0.0
            for row in df.itertuples():
                t = row.state
                a = row.action
                r = row.reward
                if row.trial == 0:
                    agent.reset_block(t)
                prob0 = agent.get_probs(t)
                prob = prob0 if a == 0 else 1 - prob0
                ll_sum += np.log(np.clip(prob, 1e-10, 1 - 1e-10))
                agent.step(a, r, t)
            sample_sums[s] = ll_sum


        logL_i = logsumexp(sample_sums) - np.log(n_samples)
        per_part_logL.append(logL_i)

    per_part_logL = np.asarray(per_part_logL)
    mean_logL = float(per_part_logL.mean())
    return mean_logL

def evaluate_log_likelihood_agent(df, testid):
    """
    Evaluate log-likelihood of test data under fitted hybrid agent.
    """
    test_pids = df['participant'].unique()[-testid:]
    df_test = df[df['participant'].isin(test_pids)].copy()
    log_likelihood = 0.0
    test_dfs = [df for _, df in df_test.groupby("participant")]
    for df in test_dfs:
        beta = df["beta_true"].iloc[0]
        gamma = df["gamma_true"].iloc[0]
        print(f"Evaluating participant {df['participant'].iloc[0]} with beta={beta}, gamma={gamma}")
        agent = HybridAgent_opt_sim_probit(beta, gamma, df.state.max() + 2)
        
        for row in df.itertuples():
            t = row.state
            a = row.action
            r = row.reward
            if row.trial == 0:
                agent.reset_block(t)
            prob0 = agent.get_probs(t)
            prob = prob0 if a == 0 else 1 - prob0
            log_likelihood += np.log(np.clip(prob, 1e-10, 1 - 1e-10))
            agent.step(a, r, t)
    return log_likelihood / len(test_pids)
        

def estimate_marginal_likelihood(df_test, mu, v, n_samples=100):
    log_liks = []
    for _ in range(n_samples):
        h_sample = np.random.normal(loc=mu, scale=np.sqrt(v))
        beta, gamma = h_sample
        log_lik = evaluate_log_likelihood(df_test, beta, gamma)
        log_liks.append(log_lik)
    return logsumexp(log_liks) - np.log(n_samples)

def test_parameter_recovery(true_beta=1.5, true_gamma=1.25, n_trials=500):
    # Simulate data with known parameters
    agent = HybridAgent_opt_sim_probit(true_beta, true_gamma, n_trials)
    actions, rewards = [], []
    for t in range(n_trials):
        prob0 = agent.get_probs(t)
        action = np.random.choice([0, 1], p=[prob0, 1-prob0])
        reward = np.random.normal(0, 1)  # Simple reward (mean=0)
        agent.step(action, reward, t)
        actions.append(action)
        rewards.append(reward)
    
    # Try to recover parameters
    df_test = pd.DataFrame({"action": actions, "reward": rewards, "state": range(n_trials)})
    result = minimize(
        negative_log_likelihood,
        x0=[1.0, 1.0],
        args=(df_test,),
        bounds=None, #[(0.1, 7), (0.1, 7)],
        method="L-BFGS-B"
    )
    print(f"True: ({true_beta}, {true_gamma}) | Recovered: {result.x}")

def estimate_shared_model(df_train_all):
    """Fit one model to all participants' data."""
    result = minimize(
        negative_log_likelihood,
        x0=[1.0, 1.0],
        args=(df_train_all,),
        bounds= None, #[(0.1, 7), (0.1, 7)],
        method="L-BFGS-B",
        options={
        'maxiter': 1000,  # Increased iterations
        'ftol': 1e-6,     # Tighter tolerance
        'gtol': 1e-6,
    }
    )
    return result.x

if __name__ == "__main__":
    # Simulation loop
    n_runs = 10
    results = []
    recovery_records = []
    eb_histories = []

    for sim in range(n_runs):
        # Simulate data with individual differences
        df_sim = simulate_data_probit_random_walk(with_id=with_id)
        train_pids = df_sim['participant'].unique()[:20]
        test_pids = df_sim['participant'].unique()[20:]
        
        df_train = df_sim[df_sim['participant'].isin(train_pids)].copy()
        df_test = df_sim[df_sim['participant'].isin(test_pids)].copy()

        # Estimate shared model
        beta_shared, gamma_shared = estimate_shared_model(df_train)

        # Run empirical Bayes
        train_dfs = [df for _, df in df_train.groupby("participant")]
        m, v, h_mat, Sigma_stack, m_hist, v_hist = run_empirical_bayes(train_dfs, max_iter=emp_bayes_iterations)
        #save empirical Bayes history
        df_eb_history = pd.DataFrame({
        "iteration": np.arange(len(m_hist)),
        "mean_beta": m_hist[:, 0],
        "mean_gamma": m_hist[:, 1],
        "var_beta": v_hist[:, 0],
        "var_gamma": v_hist[:, 1],
        "run": sim  # optional: track which simulation run this belongs to
        })
        df_eb_history["run"] = sim
        eb_histories.append(df_eb_history)

        # Evaluate log likelihoods on test participants
        sampled_lls = []
        fixed_lls = []

        # Merge true and recovered parameters
        for pid, h in zip(train_pids, h_mat):
            df_participant = df_train[df_train["participant"] == pid]
            true_beta = df_participant["beta_true"].iloc[0]
            true_gamma = df_participant["gamma_true"].iloc[0]
            recovered_beta, recovered_gamma = h

            recovery_records.append({
                "simulation": sim,
                "participant": pid,
                "true_beta": true_beta,
                "true_gamma": true_gamma,
                "recovered_beta": recovered_beta,
                "recovered_gamma": recovered_gamma,

            })

        for pid, df_subj in df_test.groupby("participant"):
            sampled_ll = estimate_marginal_likelihood(df_subj, mu=m, v=v, n_samples= marginal_iterations)
            fixed_ll = evaluate_log_likelihood(df_subj, beta=beta_shared, gamma=gamma_shared)
            sampled_lls.append(sampled_ll)
            fixed_lls.append(fixed_ll)

        avg_sampled_ll = np.mean(sampled_lls)
        avg_fixed_ll = np.mean(fixed_lls)

        results.append({
            "run": sim,
            "with_id": with_id,
            "avg_sampled_ll": avg_sampled_ll,
            "avg_fixed_ll": avg_fixed_ll,
            "model_winner": "sampled" if avg_sampled_ll > avg_fixed_ll else "fixed",
            "m_beta": m[0],
            "m_gamma": m[1],
            "v_beta": v[0],
            "v_gamma": v[1],
            "fixed_beta": beta_shared,
            "fixed_gamma": gamma_shared,
            "emp_bayes_iterations": emp_bayes_iterations,
            "marginal_iterations": marginal_iterations,
        })

    #does estimated group level prior correspond to empirical distribution?
    last_estimated_m = m_hist[-1]
    empirical_m = df_sim[["beta_true", "gamma_true"]].mean().values
    comparison_df = pd.DataFrame([last_estimated_m, empirical_m],
                                columns=["beta", "gamma"],
                                index=["last_estimated", "empirical_mean"])
    comparison_df.to_csv("data/m_comparison.csv")


    pd.concat(eb_histories, ignore_index=True).to_csv("data/eb_all_runs5.csv", index=False)
    df_results = pd.DataFrame(results)
    df_results.to_csv("data/kalman_simulation_results_em30_noID.csv", index=False)
    df_recovery = pd.DataFrame(recovery_records)
    df_recovery.to_csv("data/part_param_recovery_results_em30_noID.csv", index=False)