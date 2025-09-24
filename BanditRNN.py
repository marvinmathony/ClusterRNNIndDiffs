import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from scipy import interpolate
from scipy.special import softmax
from scipy.optimize import minimize



import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.optimize import curve_fit

OPTIMIZE_PER_PARTICIPANT = False

# return action or probability?
# Define two armed bandit environment
class ThompsonAgent:
  """Thompson Sampling agent for the two-armed bandit task.
  """

  def __init__(
      self,
      innov_variance = 100,
      noise_variance = 10,
      n_states = 10,
      n_actions: int = 2,
      trials_per_session = 10.):
    """Update the agent after one step of the task.


    """
    ### my setup ###
    self.innov_variance = innov_variance
    self.noise_variance = noise_variance
    ################
    self._n_actions = n_actions
    self.n_states = n_states
    self.current_trial = 0  # Track the trial number within the task
    self.trials_per_session = trials_per_session

    # Initialize priors
    self.reset_priors()

  def reset_priors(self):
      """Reinitialize the priors for a new session."""
      self.V_t = np.zeros(self.n_states)
      self.P_a0_thompson = np.zeros(self.n_states)
      self.post_mean = np.zeros((self._n_actions, self.n_states))
      self.post_variance = np.ones((self._n_actions, self.n_states)) * 5
      self.kalman_gain = np.zeros((self._n_actions, self.n_states))

  def get_choice_probs(self, state) -> np.ndarray:

    self.V_t[state] = self.post_mean[0][state] - self.post_mean[1][state]
    sigma2_1 = self.post_variance[0][state]  # Variance of arm 1
    sigma2_2 = self.post_variance[1][state]  # Variance of arm 2

    # Compute total uncertainty
    self.std_dev = np.sqrt(sigma2_1 + sigma2_2)

    # Calculate the probability P(a_t = 0)
    self.P_a0_thompson[state] = norm.cdf((self.V_t[state] / self.std_dev))
    #self.P_thompson[1] = 1 - self.P_thompson[0]
    ################
    return self.P_a0_thompson[state]


  def get_choice(self, state) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs(state)
    #choice = np.random.choice(self._n_actions, p=choice_probs)
    choice = 0 if np.random.rand() < choice_probs else 1

    return choice

  def update(self,
              choice: int,
              reward: int,
              state: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent.
    """

    if state < self.n_states - 1:

      for i in range(self._n_actions):
        if choice == i:

          self.kalman_gain[i][state] = self.post_variance[i][state] / (
                    self.post_variance[i][state] + self.noise_variance)
          # self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
          # self.innov_variance + self.noise_variance)
          # else:
          # self.kalman_gain[i][state] = 0
          self.post_variance[i][state + 1] = (1 - self.kalman_gain[i][state]) * self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state] + self.kalman_gain[i][
            state] * (reward - self.post_mean[i][state])

        else:
          self.post_variance[i][state + 1] = self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state]

              # Increment trial counter and check for session reset
      self.current_trial += 1
      if self.current_trial % self.trials_per_session == 0:
          self.reset_priors()


  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    # leave this for now, so that their code still works
    return self.post_mean.copy()

class UCBAgent:
  """Thompson Sampling agent for the two-armed bandit task.
  """

  def __init__(
      self,
      lamda,
      gamma,
      innov_variance = 100,
      noise_variance = 10,
      n_states = 10,
      n_actions: int = 2,
      trials_per_session=10.):
    """Update the agent after one step of the task.


    """
    ### my setup ###
    self.lamda = lamda
    self.gamma = gamma
    self.innov_variance = innov_variance
    self.noise_variance = noise_variance
    self._n_actions = n_actions
    self.n_states = n_states
    self.current_trial = 0  # Track the trial number within the task
    self.trials_per_session = trials_per_session

    # Initialize priors
    self.reset_priors()

  def reset_priors(self):
      """Reinitialize the priors for a new session."""
      self.V_t = np.zeros(self.n_states)
      self.P_a0_ucb = np.zeros(self.n_states)
      self.post_mean = np.zeros((self._n_actions, self.n_states))
      self.post_variance = np.ones((self._n_actions, self.n_states)) * 5
      self.kalman_gain = np.zeros((self._n_actions, self.n_states))

  def get_choice_probs(self, state) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""

    ### my code ###
    self.V_t[state] = self.post_mean[0][state] - self.post_mean[1][state]
    sigma1 = self.post_variance[0][state]  # Variance of arm 1
    sigma2 = self.post_variance[1][state]  # Variance of arm 2

    # Compute ucb probs
    self.P_a0_ucb[state] = norm.cdf((self.V_t[state] + self.gamma * (sigma1 - sigma2)) / self.lamda)

    return self.P_a0_ucb[state]

  def get_choice(self, state) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs(state)
    #choice = np.random.choice(self._n_actions, p=choice_probs)
    choice = 0 if np.random.rand() < choice_probs else 1

    return choice

  def update(self,
              choice: int,
              reward: int,
              state: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent.
    """

    if state < self.n_states - 1:

      for i in range(self._n_actions):
        if choice == i:

          self.kalman_gain[i][state] = self.post_variance[i][state] / (
                    self.post_variance[i][state] + self.noise_variance)
          # self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
          # self.innov_variance + self.noise_variance)
          # else:
          # self.kalman_gain[i][state] = 0
          self.post_variance[i][state + 1] = (1 - self.kalman_gain[i][state]) * self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state] + self.kalman_gain[i][
            state] * (reward - self.post_mean[i][state])

        else:
          self.post_variance[i][state + 1] = self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state]

              # Increment trial counter and check for session reset
      self.current_trial += 1
      if self.current_trial % self.trials_per_session == 0:
          self.reset_priors()

class HybridAgent_opt:

  def __init__(
      self,
      beta,
      gamma,
      n_states,
      innov_variance = 100,
      noise_variance = 10,
      n_actions: int = 2,
      trials_per_session=10):
    
    ### my setup ###
    self.innov_variance = innov_variance
    self.noise_variance = noise_variance
    self._n_actions = n_actions
    self.n_states = n_states
    self.beta = beta
    self.gamma = gamma
    self.current_trial = 0  # Track the trial number within the task
    self.trials_per_session = trials_per_session

    # Initialize priors
    self.reset_priors()

  def reset_priors(self):
      """Reinitialize the priors for a new session."""
      self.V_t = np.zeros(self.n_states)
      self.P_a0_hybrid = np.zeros(self.n_states)
      self.post_mean = np.zeros((self._n_actions, self.n_states))
      self.post_variance = np.ones((self._n_actions, self.n_states)) * 5
      self.kalman_gain = np.zeros((self._n_actions, self.n_states))

  def get_choice_probs(self, state) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""

    ### my code ###
    self.V_t[state] = self.post_mean[0][state] - self.post_mean[1][state]
    sigma1 = self.post_variance[0][state]  # Variance of arm 1
    sigma2 = self.post_variance[1][state]  # Variance of arm 2


    # Thompson
    self.std_dev = np.sqrt(sigma1 + sigma2)

    # Calculate the probability P(a_t = 0)
    self.P_a0_hybrid[state] = norm.cdf(self.beta * self.V_t[state] / self.std_dev + self.gamma * (sigma1 - sigma2))

    return self.P_a0_hybrid[state]

  def get_choice(self, state) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs(state)
    #choice = np.random.choice(self._n_actions, p=choice_probs)
    choice = 0 if np.random.rand() < choice_probs else 1

    return choice

  def update(self,
              choice: int,
              reward: int,
              state: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent.
    """

    if state < self.n_states - 1:

      for i in range(self._n_actions):
        if choice == i:

          self.kalman_gain[i][state] = self.post_variance[i][state] / (
                    self.post_variance[i][state] + self.noise_variance)
          # self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
          # self.innov_variance + self.noise_variance)
          # else:
          # self.kalman_gain[i][state] = 0
          self.post_variance[i][state + 1] = (1 - self.kalman_gain[i][state]) * self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state] + self.kalman_gain[i][
            state] * (reward - self.post_mean[i][state])

        else:
          self.post_variance[i][state + 1] = self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state]

              # Increment trial counter and check for session reset
      self.current_trial += 1
      if self.current_trial % self.trials_per_session == 0:
          self.reset_priors()

# apply Kalman filter to human data
def apply_kalman_filter_to_human_data(df, noise_variance=10, initial_variance=5, trials_per_session=10):
    """
    Applies the Kalman filter update to human behavioral data stored in a DataFrame.
    The priors are reinitialized every `trials_per_session` trials.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing human data. Must have columns 'choice' and 'reward'.
        noise_variance (float): The noise variance in the Kalman update.
        initial_variance (float): The initial posterior variance for each arm.
        trials_per_session (int): The number of trials per session after which the priors are reset.
        
    Returns:
        pd.DataFrame: The DataFrame with additional columns for internal Kalman filter variables.
    """
    
    n_actions = 2
    n_trials = df.shape[0]
    
    # Preallocate arrays for the filter variables
    post_mean = np.zeros((n_actions, n_trials))
    post_variance = np.ones((n_actions, n_trials)) * initial_variance
    kalman_gain = np.zeros((n_actions, n_trials))
    V_t = np.zeros(n_trials)
    
    # Arrays to record computed quantities (to later append to df)
    posterior_std_0 = np.zeros(n_trials)
    posterior_std_1 = np.zeros(n_trials)
    TU = np.zeros(n_trials)
    RU = np.zeros(n_trials)
    
    # Loop through trials sequentially
    for t in range(n_trials - 1):
        # Record current internal state BEFORE updating at trial t.
        V_t[t] = post_mean[0, t] - post_mean[1, t]
        posterior_std_0[t] = post_variance[0, t]
        posterior_std_1[t] = post_variance[1, t]
        TU[t] = np.sqrt(post_variance[0, t] + post_variance[1, t])
        RU[t] = np.sqrt(post_variance[0, t]) - np.sqrt(post_variance[1, t])
        
        # Retrieve human choice and reward for this trial.
        # Assumes that the 'choice' column is coded as 0 or 1.
        row = df.iloc[t]
        choice = int(row['choice'])
        reward = row['reward']
        
        # Update for the chosen arm.
        kalman_gain[choice, t] = post_variance[choice, t] / (post_variance[choice, t] + noise_variance)
        post_mean[choice, t+1] = post_mean[choice, t] + kalman_gain[choice, t] * (reward - post_mean[choice, t])
        post_variance[choice, t+1] = (1 - kalman_gain[choice, t]) * post_variance[choice, t]
        
        # Propagate estimates for the non-chosen arm.
        non_choice = 1 - choice
        post_mean[non_choice, t+1] = post_mean[non_choice, t]
        post_variance[non_choice, t+1] = post_variance[non_choice, t]
        
        # At session boundaries, reset priors for the next trial.
        if (t + 1) % trials_per_session == 0:
            if t + 1 < n_trials:
                post_mean[:, t+1] = 0
                post_variance[:, t+1] = initial_variance

    # Record the final trial's state (no update follows the final trial).
    t = n_trials - 1
    V_t[t] = post_mean[0, t] - post_mean[1, t]
    posterior_std_0[t] = post_variance[0, t]
    posterior_std_1[t] = post_variance[1, t]
    TU[t] = np.sqrt(post_variance[0, t] + post_variance[1, t])
    RU[t] = np.sqrt(post_variance[0, t]) - np.sqrt(post_variance[1, t])
    
    # Append the computed values as new columns in the dataframe.
    df['V_t'] = V_t
    df['posterior_std_0'] = posterior_std_0
    df['posterior_std_1'] = posterior_std_1
    df['poster_mean_0'] = post_mean[0, :]
    df['poster_mean_1'] = post_mean[1, :]
    df['TU'] = TU
    df['RU'] = RU
    
    return df


##### optimization #####
##########################################
# LOAD DATA
##########################################
df = pd.read_csv('human_data.csv')
df['state'] = df.index % 8800  # Ensure state values are within range
df['choice'] = df['choice'].astype(int) - 1  # Convert choice to 0, 1


def negative_log_likelihood_bandit(opt_params, data, fixed_params, model):
    lamda, gamma = opt_params  # Optimized parameters
    n_states = fixed_params['n_states']
    # Create a new model instance with these parameters
    if model == UCBAgent:
        lamda, gamma = opt_params  # Optimized parameters
        model = model(n_states=n_states, lamda=lamda, gamma=gamma)
        print('number of states: ', model.n_states)
        print('lamda: ', model.lamda)
        print('gamma: ', model.gamma)
    elif model == HybridAgent_opt:
        beta, gamma = opt_params  # Optimized parameters
        model = model(n_states=n_states, beta=beta, gamma=gamma)
        print('number of states: ', model.n_states)
        print('beta: ', model.beta)
        print('gamma: ', model.gamma)
    nll = 0

    # Loop over trials in the provided data
    for _, row in data.iterrows():
        choice = int(row['choice'])  # Convert human choice from 1,2 to 0,1
        reward = row['reward']
        state_in_df = row['state']
        # Get model’s probability for choosing action 0 (and action 1)
        action_prob_0 = model.get_choice_probs(state_in_df)
        action_prob_1 = 1 - action_prob_0
        actions_probs = np.array([action_prob_0, action_prob_1])
        action_prob = actions_probs[choice]

        nll -= np.log(action_prob + 1e-10)
        model.update(choice, reward, state_in_df)

    return nll

if OPTIMIZE_PER_PARTICIPANT:
    # Optimize separately for each participant.
    results_per_participant = {}
    for subject, subject_data in df.groupby('subject'):
        print(f"\nOptimizing for subject {subject} ...")
        result = minimize(
            negative_log_likelihood_bandit,
            initial_params,
            args=(subject_data, fixed_params),
            bounds=bounds,
            method='L-BFGS-B'
        )
        results_per_participant[subject] = result
        print(f"Subject: {subject}")
        print(f"  Optimized parameters: {result.x}")
        print(f"  Final negative log-likelihood: {result.fun}\n")
else:
    # Step 1: Optimize Parameters and Store Results
    optimized_params = {}

    models = [UCBAgent, HybridAgent_opt]
    initial_params = [1, 2]  # Starting values for [lamda, gamma] or [beta, gamma]
    bounds = [(0, 10), (0, 10)]

    for model in models:
        if model == UCBAgent:
            fixed_params = {'n_states': 8800}
        elif model == HybridAgent_opt:
            fixed_params = {'n_states': 8800}

        print(f"\nOptimizing {model.__name__} globally for the dataset ...")
        
        global_result = minimize(
            negative_log_likelihood_bandit,
            initial_params,
            args=(df, fixed_params, model),
            bounds=bounds,
            method='L-BFGS-B'
        )

        optimized_params[model.__name__] = global_result.x  # Store optimized parameters

        print(f"  Optimized parameters for {model.__name__}: {global_result.x}")
        print(f"  Final negative log-likelihood: {global_result.fun}\n")

# Step 2: Forward Simulation Using Optimized Parameters
n_participants = 150
n_block_per_p = 20
n_trials_per_block = 10
reward_array = np.empty([2, n_trials_per_block * n_block_per_p * n_participants], dtype=float)

data_ucb = []
data_hybrid = []
data_thompson = []

algorithms = ["ucb", "hybrid", "thompson"]

for algorithm in algorithms:
    state = 0

    for participant in range(n_participants):
        # Retrieve optimized parameters
        if algorithm == "ucb":
            lamda, gamma = [0.1,0]#optimized_params["UCBAgent"]
            agent = UCBAgent(lamda=lamda, gamma=gamma, n_states=n_participants * n_block_per_p * n_trials_per_block)
        elif algorithm == "hybrid":
            beta, gamma = optimized_params["HybridAgent_opt"]
            agent = HybridAgent_opt(beta=beta, gamma=gamma, n_states=n_participants * n_block_per_p * n_trials_per_block)
        else:  # Thompson agent does not need optimization
            agent = ThompsonAgent(n_states=n_participants * n_block_per_p * n_trials_per_block)

        for block in range(n_block_per_p):
            mean_reward_block = np.random.normal(0, np.sqrt(agent.innov_variance), agent._n_actions)
            
            for trial in range(n_trials_per_block):
                # Define the environment
                reward = np.random.normal(mean_reward_block, np.sqrt(agent.noise_variance), agent._n_actions)

                # Run the agent in the environment
                action = agent.get_choice(state)
                reward = reward[action]

                # Append the results to the data list
                data_entry = {
                    'Participant': participant,
                    'Block': block,
                    'Trial': trial,
                    'State': state,
                    'Action': action,
                    'Reward': reward,
                    'V_t': agent.V_t[state],
                    'posterior_std_0': agent.post_variance[0][state],
                    'posterior_std_1': agent.post_variance[1][state],
                    'poster_mean_0': agent.post_mean[0][state],
                    'poster_mean_1': agent.post_mean[1][state],
                    'TU': np.sqrt(agent.post_variance[0][state] + agent.post_variance[1][state]),
                    'RU': np.sqrt(agent.post_variance[0][state]) - np.sqrt(agent.post_variance[1][state])
                }

                if algorithm == "thompson":
                    data_entry['P_a0_thompson'] = agent.get_choice_probs(state)
                    data_thompson.append(data_entry)
                elif algorithm == "ucb":
                    data_entry['P_a0_ucb'] = agent.get_choice_probs(state)
                    data_ucb.append(data_entry)
                else:  # hybrid
                    data_entry['P_a0_hybrid'] = agent.get_choice_probs(state)
                    data_hybrid.append(data_entry)

                agent.update(action, reward, state)
                state += 1

# Apply the Kalman filter updates to the human data.
df = apply_kalman_filter_to_human_data(df, noise_variance=10, initial_variance=5, trials_per_session=10)

# save the updated dataframe.
df.to_csv("kalman_human_data.csv", index=False)

# Convert data to DataFrames
df_ucb = pd.DataFrame(data_ucb)
df_hybrid = pd.DataFrame(data_hybrid)
df_thompson = pd.DataFrame(data_thompson)

# -----------------------------
# Compute Log Likelihood for the True Generating Hybrid Model
# -----------------------------

def compute_log_likelihood(agent_class, data, optimized_params, n_states):
    """
    Compute the average log likelihood for the Hybrid agent
    over the provided data.
    
    Parameters:
        agent_class: The class of the agent (should be HybridAgent_opt).
        data: A DataFrame containing the simulation data with columns 'State', 'Action', and 'Reward'.
        optimized_params: A tuple/list with the optimized parameters (beta, gamma) for the Hybrid agent.
        n_states: Total number of states for the simulation (e.g., n_participants * n_block_per_p * n_trials_per_block).
        
    Returns:
        avg_log_likelihood: The average log likelihood computed over the data.
    """
    beta, gamma = optimized_params
    # Instantiate a new Hybrid agent with the optimized parameters.
    agent = agent_class(n_states=n_states, beta=beta, gamma=gamma)
    
    log_likelihood = 0.0
    log_likelihoods = []

    # Loop sequentially over all trials (rows) in the data.
    for idx, row in data.iterrows():
        state = int(row['State'])
        choice = int(row['Action'])
        # Get the probability of choosing action 0 at the current state.
        prob_0 = agent.get_choice_probs(state)
        # For the chosen action, use prob_0 if action is 0, or (1 - prob_0) if action is 1.
        prob = prob_0 if choice == 0 else 1 - prob_0
        # Accumulate log likelihood (adding a small constant for numerical stability).
        log_likelihood += np.log(prob + 1e-10)
        log_likelihoods.append(log_likelihood)

        
        # Retrieve the reward from data and update the agent.
        reward = row['Reward']
        agent.update(choice, reward, state)
    
    avg_log_likelihood = log_likelihood / len(data)
    return avg_log_likelihood, log_likelihoods

# -----------------------------
# Compute Pseudo R^2 Measure for Each Agent
# -----------------------------
def compute_total_log_likelihood(agent_class, data, optimized_params, n_states):
    """
    Compute the total log likelihood for the provided data given an agent.
    
    Parameters:
        agent_class: The class of the agent (e.g., HybridAgent_opt, UCBAgent, or ThompsonAgent).
        data: A DataFrame containing simulation data. Assumes columns 'State', 'Action', and 'Reward'.
        optimized_params: The optimized parameters for the agent. For ThompsonAgent, pass None.
        n_states: Total number of states for the simulation.
        
    Returns:
        total_log_likelihood: The summed log likelihood over all trials.
    """
    total_log_likelihood = 0.0

    # Instantiate the agent using the optimized parameters if required
    if agent_class == HybridAgent_opt:
        beta, gamma = optimized_params
        agent = agent_class(n_states=n_states, beta=beta, gamma=gamma)
    elif agent_class == UCBAgent:
        lamda, gamma = optimized_params
        agent = agent_class(n_states=n_states, lamda=lamda, gamma=gamma)
    elif agent_class == ThompsonAgent:
        agent = agent_class(n_states=n_states)
    else:
        raise ValueError("Agent class not recognized.")

    # Loop over each trial in the data
    for idx, row in data.iterrows():
        state = int(row['Global_Trial'])  # For simulation data, we use capitalized column names.
        choice = int(row['Action'])
        # Get the probability for action 0 at the current state.
        prob_0 = agent.get_choice_probs(state)
        # If the chosen action is 0, use prob_0; if action is 1, use (1 - prob_0)
        prob = prob_0 if choice == 0 else (1 - prob_0)
        total_log_likelihood += np.log(prob + 1e-10)
        reward = row['Reward']
        agent.update(choice, reward, state)
    
    return total_log_likelihood

def compute_pseudo_r2(agent_class, data, optimized_params, n_states):
    """
    Compute the pseudo R^2 (McFadden's pseudo R^2) for the provided agent and data.
    
    Parameters:
        agent_class: The agent class (HybridAgent_opt, UCBAgent, or ThompsonAgent).
        data: A DataFrame with columns 'State', 'Action', and 'Reward'.
        optimized_params: The optimized parameters for the agent. For ThompsonAgent, pass None.
        n_states: Total number of states for the simulation.
        
    Returns:
        pseudo_r2: The pseudo R^2 value.
    """
    total_ll_model = compute_total_log_likelihood(agent_class, data, optimized_params, n_states)
    # Null model: assumes a constant choice probability of 0.5 for every trial.
    total_ll_null = len(data) * np.log(0.5)
    pseudo_r2 = 1 - (total_ll_model / total_ll_null)
    return pseudo_r2

# -----------------------------
# Compute and Report Pseudo R^2 for Simulation Data
# -----------------------------

# Total number of states in the simulation (as used above)
n_states_sim = n_participants * n_block_per_p * n_trials_per_block

# For HybridAgent_opt, use the optimized parameters from earlier.
pseudo_r2_hybrid = compute_pseudo_r2(HybridAgent_opt, df_hybrid, optimized_params["HybridAgent_opt"], n_states_sim)
print(f"Pseudo R^2 for HybridAgent_opt: {pseudo_r2_hybrid:.4f}")

# For UCBAgent, we used a fixed parameter set in simulation ([lamda, gamma] = [0.1, 0]).
pseudo_r2_ucb = compute_pseudo_r2(UCBAgent, df_ucb, [0.1, 0], n_states_sim)
print(f"Pseudo R^2 for UCBAgent: {pseudo_r2_ucb:.4f}")

# For ThompsonAgent, no optimization parameters are needed.
pseudo_r2_thompson = compute_pseudo_r2(ThompsonAgent, df_thompson, None, n_states_sim)
print(f"Pseudo R^2 for ThompsonAgent: {pseudo_r2_thompson:.4f}")



n_states_sim = n_participants * n_block_per_p * n_trials_per_block

hybrid_optimized_params = optimized_params["HybridAgent_opt"]

hybrid_avg_log_likelihood, trial_log_likelihoods = compute_log_likelihood(HybridAgent_opt, df_hybrid, hybrid_optimized_params, n_states_sim)

print(f"Average Log Likelihood for the True Generating Hybrid Model: {hybrid_avg_log_likelihood:.4f}")

df_hybrid['avglikelihood'] = hybrid_avg_log_likelihood
df_hybrid['Hybrid_Log_Likelihood'] = trial_log_likelihoods


# Define the directory and file paths
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

df_ucb.to_csv(os.path.join(output_dir, "results_ucb.csv"), index=False)
df_hybrid.to_csv(os.path.join(output_dir, "results_hybrid.csv"), index=False)
df_thompson.to_csv(os.path.join(output_dir, "results_thompson.csv"), index=False)

