# by KK
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import random


import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def simulate_Qlearning(
    rewards, seed=1979, n_trials=200, n_sessions=10, 
    alphaP_list=0.5, alphaN_list=0.5, 
    beta=0.3, alphaF_list=0.0, phi_list=0.0, tau_list=0.0, static = False
):
    """
    Simulates behavioral data using an Asymmetric Q-learning model with choice trace (CT),
    and computes session-wise normalized log-likelihood.

    Parameters:
    - rewards (ndarray): A 2D array of reward values for each option (shape: [2, n_trials]).
    - seed (int): Random seed for reproducibility.
    - n_trials (int): Number of trials per session.
    - n_sessions (int): Number of sessions to simulate.
    - alphaP_list (float or list): AlphaP values (or list of values for each session).
    - alphaN_list (float or list): AlphaN values (or list of values for each session).
    - beta (float or list): Beta values (or list of values for each session).
    - alphaF_list (float or list): Forgetting rate values (or list of values for each session).
    - phi_list (float or list): Phi values (or list of values for each session).
    - tau_list (float or list): Tau values (or list of values for each session).

    Returns:
    - c (ndarray): Choices made (shape: [n_sessions, n_trials]).
    - r (ndarray): Rewards received (shape: [n_sessions, n_trials]).
    - p (ndarray): Probability of choosing option 1 (shape: [n_sessions, n_trials]).
    - Q (ndarray): Q-values for each option (shape: [n_sessions, 2, n_trials]).
    - CT (ndarray): Choice trace values (shape: [n_sessions, 2, n_trials]).
    - df (DataFrame): DataFrame containing session, choices, rewards, probabilities, and choice traces.
    - xin (ndarray): Processed input tensor for training (shape: [n_sessions, n_trials, 4]).
    - choice_one_hot (ndarray): One-hot encoded choices (shape: [n_sessions, n_trials, 2]).
    - normalized_LL (float): Overall normalized log-likelihood of simulated choices.
    - session_ll_df (DataFrame): Session-wise normalized log-likelihood.
    - xin_CIR (ndarray): Processed input tensor for training with CIR (shape: [n_sessions, n_trials, 3]).
    """

    np.random.seed(seed)

    def to_list(value, n_sessions):
        return [value] * n_sessions if not isinstance(value, list) else value

    alphaP_list = to_list(alphaP_list, n_sessions)
    alphaN_list = to_list(alphaN_list, n_sessions)
    beta_list = to_list(beta, n_sessions)
    alphaF_list = to_list(alphaF_list, n_sessions)
    phi_list = to_list(phi_list, n_sessions)
    tau_list = to_list(tau_list, n_sessions)

    # Initialize arrays
    Q = np.zeros((n_sessions, 2, n_trials))
    CT = np.zeros((n_sessions, 2, n_trials))  
    c = np.zeros((n_sessions, n_trials), dtype=int)  
    r = np.zeros((n_sessions, n_trials))  
    p = np.zeros((n_sessions, n_trials))  

    dataframes = []

    for session in range(n_sessions):
        alphaP = alphaP_list[session]
        alphaN = alphaN_list[session]
        beta = beta_list[session]
        alphaF = alphaF_list[session]
        phi = phi_list[session]
        tau = tau_list[session]

        for t in range(n_trials):
            # Compute choice probability using softmax
            p[session, t] = np.exp(beta * Q[session, 0, t] + phi * CT[session, 0, t]) / \
                            np.exp(beta * Q[session, :, t] + phi * CT[session, :, t]).sum()

            # Make a choice: 0 or 1
            c[session, t] = np.random.choice([0, 1], p=[p[session, t], 1 - p[session, t]])

            # Assign reward
            r[session, t] = rewards[c[session, t], t] if static else rewards[session, c[session, t], t]

            # Update Q-values and choice traces
            if t < n_trials - 1:
                Q[session, :, t + 1] = (1 - alphaF) * Q[session, :, t]
                delta = r[session, t] - Q[session, c[session, t], t]
                #Q[session, c[session, t], t + 1] += (alphaP if delta > 0 else alphaN) * delta
                Q[session, c[session, t], t + 1] = Q[session, c[session, t], t] + (alphaP if delta > 0 else alphaN) * delta

                CT[session, c[session, t], t + 1] = CT[session, c[session, t], t] + tau * (1 - CT[session, c[session, t], t])
                CT[session, 1 - c[session, t], t + 1] = CT[session, 1 - c[session, t], t] + tau * (0 - CT[session, 1 - c[session, t], t])

        # Store session data in a DataFrame
        df_q_learning = pd.DataFrame({
            'session': np.repeat(session, n_trials),
            'trial': np.arange(n_trials),
            'c': c[session],
            'r': r[session],
            'p': p[session],
            'CT_0': CT[session, 0],
            'CT_1': CT[session, 1],
            'model': 'Q-learning'
        })
        dataframes.append(df_q_learning)

    # Combine session DataFrames into one
    df = pd.concat(dataframes, ignore_index=True)

    # Generate xin and choice_one_hot
    def generate_xin(c, r):
        s, t = c.shape
        xin = np.zeros((s, t, 4))

        cond1 = (c == 0) & (r == 1)
        cond2 = (c == 1) & (r == 1)
        cond3 = (c == 0) & (r == 0)
        cond4 = (c == 1) & (r == 0)

        xin[:, 1:, 0][cond1[:, :-1]] = 1
        xin[:, 1:, 2][cond1[:, :-1]] = 1

        xin[:, 1:, 1][cond2[:, :-1]] = 1
        xin[:, 1:, 3][cond2[:, :-1]] = 1

        xin[:, 1:, 2][cond3[:, :-1]] = 1
        xin[:, 1:, 3][cond4[:, :-1]] = 1

        return xin

    xin = generate_xin(c, r)
    choice_one_hot = to_categorical(c, num_classes=2)

    # Generate xin_CIR
    def generate_xin_CIR(c, r):
        s, t = c.shape
        xin = np.zeros((s, t, 3))

        xin[:, 1:, 0] = r[:, :-1]
        xin[:, 1:, 1][c[:, :-1] == 0] = 1
        xin[:, 1:, 2][c[:, :-1] == 1] = 1

        return xin

    xin_CIR = generate_xin_CIR(c, r)

    # Compute session-wise normalized log-likelihood
    session_ll_list = []
    for session in range(n_sessions):
        ll = np.sum(np.log(np.where(c[session] == 0, p[session], 1 - p[session])))
        normalized_ll = np.exp(ll / n_trials)
        session_ll_list.append({"session": session, "normalized_likelihood": normalized_ll, "model": "True model"})

    session_ll_df = pd.DataFrame(session_ll_list)

    # Compute overall normalized log-likelihood
    log_likelihoods = np.log(np.where(c == 0, p, 1 - p))
    normalized_LL = np.exp(np.sum(log_likelihoods) / c.size)

    return c, r, p, Q, CT, df, xin, choice_one_hot, normalized_LL, session_ll_df, xin_CIR

import numpy as np
import pandas as pd

def simulate_Qlearning_with_context(
    rewards, context_array, seed=1979, n_trials=200, n_sessions=10, 
    alphaP_list=0.5, alphaN_list=0.5, 
    beta=0.3, alphaF_list=0.0, phi_list=0.0, tau_list=0.0
):
    """
    Simulates behavioral data using an Asymmetric Q-learning model with choice trace (CT),
    and incorporates context-based Q-value resetting.

    Parameters:
    - rewards (ndarray): A 2D array of reward values for each option (shape: [2, n_trials]).
    - context_array (ndarray): A 1D array specifying context for each trial (shape: [n_trials]).
    - seed (int): Random seed for reproducibility.
    - n_trials (int): Number of trials per session.
    - n_sessions (int): Number of sessions to simulate.
    - alphaP_list (float or list): AlphaP values (or list of values for each session).
    - alphaN_list (float or list): AlphaN values (or list of values for each session).
    - beta (float or list): Beta values (or list of values for each session).
    - alphaF_list (float or list): Forgetting rate values (or list of values for each session).
    - phi_list (float or list): Phi values (or list of values for each session).
    - tau_list (float or list): Tau values (or list of values for each session).

    Returns:
    - c (ndarray): Choices made (shape: [n_sessions, n_trials]).
    - r (ndarray): Rewards received (shape: [n_sessions, n_trials]).
    - p (ndarray): Probability of choosing option 1 (shape: [n_sessions, n_trials]).
    - Q (ndarray): Q-values for each option (shape: [n_sessions, 2, n_trials]).
    - CT (ndarray): Choice trace values (shape: [n_sessions, 2, n_trials]).
    - df (DataFrame): DataFrame containing session, trial, choices, rewards, probabilities, choice traces, and context.
    """

    np.random.seed(seed)

    def to_list(value, n_sessions):
        return [value] * n_sessions if not isinstance(value, list) else value

    alphaP_list = to_list(alphaP_list, n_sessions)
    alphaN_list = to_list(alphaN_list, n_sessions)
    beta_list = to_list(beta, n_sessions)
    alphaF_list = to_list(alphaF_list, n_sessions)
    phi_list = to_list(phi_list, n_sessions)
    tau_list = to_list(tau_list, n_sessions)

    # Initialize arrays
    Q = np.zeros((n_sessions, 2, n_trials))
    CT = np.zeros((n_sessions, 2, n_trials))  
    c = np.zeros((n_sessions, n_trials), dtype=int)  
    r = np.zeros((n_sessions, n_trials))  
    p = np.zeros((n_sessions, n_trials))  

    dataframes = []

    for session in range(n_sessions):
        alphaP = alphaP_list[session]
        alphaN = alphaN_list[session]
        beta = beta_list[session]
        alphaF = alphaF_list[session]
        phi = phi_list[session]
        tau = tau_list[session]
        
        context = context_array
        
        for t in range(n_trials):
            # Reset Q-values when context changes
            if t > 0 and context[t] != context[t - 1]:
                Q[session, :, t] = 0  # Reset Q-values
                CT[session, :, t] = 0  # Reset choice trace
            # elif t > 0:
            #     Q[session, :, t] = (1 - alphaF) * Q[session, :, t - 1]
            #     CT[session, :, t] = CT[session, :, t - 1]
            
            # Compute choice probability using softmax
            p[session, t] = np.exp(beta * Q[session, 0, t] + phi * CT[session, 0, t]) / \
                            np.exp(beta * Q[session, :, t] + phi * CT[session, :, t]).sum()
            
            # Make a choice: 0 or 1
            c[session, t] = np.random.choice([0, 1], p=[p[session, t], 1 - p[session, t]])
            
            # Assign reward
            r[session, t] = rewards[c[session, t], t]
            
            # Update Q-values and choice traces
            if t < n_trials - 1:
                Q[session, :, t + 1] = (1 - alphaF) * Q[session, :, t]
                delta = r[session, t] - Q[session, c[session, t], t]
                Q[session, c[session, t], t + 1] = Q[session, c[session, t], t] + (alphaP if delta > 0 else alphaN) * delta

                CT[session, c[session, t], t + 1] = CT[session, c[session, t], t] + tau * (1 - CT[session, c[session, t], t])
                CT[session, 1 - c[session, t], t + 1] = CT[session, 1 - c[session, t], t] + tau * (0 - CT[session, 1 - c[session, t], t])

            # # Update Q-values and choice traces
            # delta = r[session, t] - Q[session, c[session, t], t]
            # Q[session, c[session, t], t] += (alphaP if delta > 0 else alphaN) * delta
            
            # CT[session, c[session, t], t] += tau * (1 - CT[session, c[session, t], t])
            # CT[session, 1 - c[session, t], t] += tau * (0 - CT[session, 1 - c[session, t], t])

        # Store session data in a DataFrame
        df_q_learning = pd.DataFrame({
            'session': np.repeat(session, n_trials),
            'subject': np.repeat(session, n_trials),
            'trial': np.arange(n_trials),
            'c': c[session],
            'r': r[session],
            'p': p[session],
            'CT_0': CT[session, 0],
            'CT_1': CT[session, 1],
            'context': context,
            'model': 'Q-learning'
        })
        dataframes.append(df_q_learning)

    # Combine session DataFrames into one
    df = pd.concat(dataframes, ignore_index=True)

    # Compute session-wise normalized log-likelihood
    session_ll_list = []
    for session in range(n_sessions):
        ll = np.sum(np.log(np.where(c[session] == 0, p[session], 1 - p[session])))
        normalized_ll = np.exp(ll / n_trials)
        session_ll_list.append({"session": session, "normalized_likelihood": normalized_ll, "model": "True model"})

    session_ll_df = pd.DataFrame(session_ll_list)

    return c, r, p, Q, CT, df, session_ll_df

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

def simulate_Qlearning_with_variable_params(
    rewards, seed=1979, n_trials=200, n_sessions=10, 
    alphaP_array=None, alphaN_array=None,  
    alphaF_array=None, phi_array=None, tau_array=None, beta_array=None
):
    """
    Simulates behavioral data using an Asymmetric Q-learning model with choice trace (CT),
    allowing trial-by-trial parameter updates.

    Parameters:
    - rewards (ndarray): A 2D array of reward values for each option (shape: [2, n_trials]).
    - seed (int): Random seed for reproducibility.
    - n_trials (int): Number of trials per session.
    - n_sessions (int): Number of sessions to simulate.
    - alphaP_array (ndarray): Array specifying alphaP values per trial (shape: [n_sessions, n_trials]).
    - alphaN_array (ndarray): Array specifying alphaN values per trial (shape: [n_sessions, n_trials]).
    - beta (float): Softmax inverse temperature parameter.
    - alphaF_array (ndarray): Array specifying forgetting rate per trial (shape: [n_sessions, n_trials]).
    - phi_array (ndarray): Array specifying phi per trial (shape: [n_sessions, n_trials]).
    - tau_array (ndarray): Array specifying tau per trial (shape: [n_sessions, n_trials]).

    Returns:
    - c (ndarray): Choices made (shape: [n_sessions, n_trials]).
    - r (ndarray): Rewards received (shape: [n_sessions, n_trials]).
    - p (ndarray): Probability of choosing option 1 (shape: [n_sessions, n_trials]).
    - Q (ndarray): Q-values for each option (shape: [n_sessions, 2, n_trials]).
    - CT (ndarray): Choice trace values (shape: [n_sessions, 2, n_trials]).
    - df (DataFrame): DataFrame containing session, choices, rewards, probabilities, and choice traces.
    - xin (ndarray): Processed input tensor for training (shape: [n_sessions, n_trials, 4]).
    - choice_one_hot (ndarray): One-hot encoded choices (shape: [n_sessions, n_trials, 2]).
    - normalized_LL (float): Overall normalized log-likelihood of simulated choices.
    - session_ll_df (DataFrame): Session-wise normalized log-likelihood.
    - xin_CIR (ndarray): Processed input tensor for training with CIR (shape: [n_sessions, n_trials, 3]).
    """

    np.random.seed(seed)

    # Initialize arrays
    Q = np.zeros((n_sessions, 2, n_trials))
    CT = np.zeros((n_sessions, 2, n_trials))  
    c = np.zeros((n_sessions, n_trials), dtype=int)  
    r = np.zeros((n_sessions, n_trials))  
    p = np.zeros((n_sessions, n_trials))  

    # Default to constant values if arrays are not provided
    def default_array(value):
        return np.full((n_sessions, n_trials), value) if value is not None else np.zeros((n_sessions, n_trials))

    alphaP_array = alphaP_array if alphaP_array is not None else default_array(0.5)
    alphaN_array = alphaN_array if alphaN_array is not None else default_array(0.5)
    alphaF_array = alphaF_array if alphaF_array is not None else default_array(0.0)
    phi_array = phi_array if phi_array is not None else default_array(0.0)
    tau_array = tau_array if tau_array is not None else default_array(0.0)
    beta_array = beta_array if beta_array is not None else default_array(3.0)

    dataframes = []

    for session in range(n_sessions):
        for t in range(n_trials):
            # Extract trial-specific parameters
            alphaP = alphaP_array[session, t]
            alphaN = alphaN_array[session, t]
            alphaF = alphaF_array[session, t]
            phi = phi_array[session, t]
            tau = tau_array[session, t]

            # Compute choice probability using softmax
            p[session, t] = np.exp(beta_array[session, t] * Q[session, 0, t] + phi * CT[session, 0, t]) / \
                            np.exp(beta_array[session, t] * Q[session, :, t] + phi * CT[session, :, t]).sum()

            # Make a choice: 0 or 1
            c[session, t] = np.random.choice([0, 1], p=[p[session, t], 1 - p[session, t]])

            # Assign reward
            r[session, t] = rewards[c[session, t], t]

            # Update Q-values and choice traces
            if t < n_trials - 1:
                Q[session, :, t + 1] = (1 - alphaF) * Q[session, :, t]
                delta = r[session, t] - Q[session, c[session, t], t]
                Q[session, c[session, t], t + 1] = Q[session, c[session, t], t] + (alphaP if delta > 0 else alphaN) * delta

                CT[session, c[session, t], t + 1] = CT[session, c[session, t], t] + tau * (1 - CT[session, c[session, t], t])
                CT[session, 1 - c[session, t], t + 1] = CT[session, 1 - c[session, t], t] + tau * (0 - CT[session, 1 - c[session, t], t])

        # Store session data in a DataFrame
        df_q_learning = pd.DataFrame({
            'session': np.repeat(session, n_trials),
            'trial': np.arange(n_trials),
            'c': c[session],
            'r': r[session],
            'p': p[session],
            'CT_0': CT[session, 0],
            'CT_1': CT[session, 1],
            'model': 'Q-learning'
        })
        dataframes.append(df_q_learning)

    # Combine session DataFrames into one
    df = pd.concat(dataframes, ignore_index=True)

    # Generate xin and choice_one_hot
    def generate_xin(c, r):
        s, t = c.shape
        xin = np.zeros((s, t, 4))

        cond1 = (c == 0) & (r == 1)
        cond2 = (c == 1) & (r == 1)
        cond3 = (c == 0) & (r == 0)
        cond4 = (c == 1) & (r == 0)

        xin[:, 1:, 0][cond1[:, :-1]] = 1
        xin[:, 1:, 2][cond1[:, :-1]] = 1

        xin[:, 1:, 1][cond2[:, :-1]] = 1
        xin[:, 1:, 3][cond2[:, :-1]] = 1

        xin[:, 1:, 2][cond3[:, :-1]] = 1
        xin[:, 1:, 3][cond4[:, :-1]] = 1

        return xin

    xin = generate_xin(c, r)
    choice_one_hot = to_categorical(c, num_classes=2)

    # Generate xin_CIR
    def generate_xin_CIR(c, r):
        s, t = c.shape
        xin = np.zeros((s, t, 3))

        xin[:, 1:, 0] = r[:, :-1]
        xin[:, 1:, 1][c[:, :-1] == 0] = 1
        xin[:, 1:, 2][c[:, :-1] == 1] = 1

        return xin

    xin_CIR = generate_xin_CIR(c, r)

    # Compute overall normalized log-likelihood
    log_likelihoods = np.log(np.where(c == 0, p, 1 - p))
    normalized_LL = np.exp(np.sum(log_likelihoods) / c.size)

    return c, r, p, Q, CT, df, xin, choice_one_hot, normalized_LL, xin_CIR



def gen_reward_seq(seed=1979, T = 5000, interval = 50, N=None):

    np.random.seed(seed)
    dummy_array = [1,1,1]
    p_groups = np.random.dirichlet(dummy_array)
    environments = ["low", "normal", "high"]
    
    # Create the rewards array
    # Assign rewards to the array, flipping every 50 trials
    if N is not None:
        rewards = np.zeros((N, 2, T))
        
        for n in range(N):
            # assign group A, B and C here (one bad group, one normal group, one good group)
            # randomly sampe interval between 20 to 100 per participant
            #interval = random.randrange(1, 80)
            selected_env = np.random.choice(environments)
            print(f"selected environment: {selected_env}")
            if selected_env == "low":
                pHigh = random.uniform(0.1, 0.3)
                gap_low   = random.uniform(0.05, 0.15)
                pLow  = pHigh - gap_low
            elif selected_env == "normal":
                pHigh = 0.8
                pLow = 0.2
            else:
                pHigh = random.uniform(0.7, 0.95)
                gap_high   = random.uniform(0.2, 0.4)
                pLow  = max(0.05, pHigh - gap_high)

            for t in range(T):
                if t % (interval * 2) < interval:
                    rewards[n, 0, t] = 1 if np.random.rand() < pHigh else 0
                    rewards[n, 1, t] = 1 if np.random.rand() < pLow else 0
                else:
                    rewards[n, 1, t] = 1 if np.random.rand() < pHigh else 0
                    rewards[n, 0, t] = 1 if np.random.rand() < pLow else 0
        print("generating rewards for participants individually")
    else:
        rewards = np.zeros((2, T))
        for t in range(T):
            if t % (interval * 2) < interval:
                rewards[0, t] = np.random.choice([0, 1], p=[pHigh, pLow])
                rewards[1, t] = np.random.choice([0, 1], p=[pLow, pHigh])
            else:
                rewards[0, t] = np.random.choice([0, 1], p=[pLow, pHigh])
                rewards[1, t] = np.random.choice([0, 1], p=[pHigh, pLow])

    
    

    return rewards

def gen_reward_seq_og(seed=1979, T = 5000, pHigh = 0.8, pLow = 0.2, interval = 50, N=None):

    np.random.seed(seed)

    # Create the rewards array
    # Assign rewards to the array, flipping every 50 trials
    if N is not None:
        rewards = np.zeros((N, 2, T))
        for n in range(N):
            pHigh = random.uniform(0.5, 1.0)
            pLow = 1-pHigh
            for t in range(T):
                if t % (interval * 2) < interval:
                    rewards[n,0, t] = np.random.choice([0, 1], p=[pHigh, pLow])
                    rewards[n,1, t] = np.random.choice([0, 1], p=[pLow, pHigh])
                else:
                    rewards[n,0, t] = np.random.choice([0, 1], p=[pLow, pHigh])
                    rewards[n,1, t] = np.random.choice([0, 1], p=[pHigh, pLow])
        print("generating rewards for participants individually")
    else:
        rewards = np.zeros((2, T))
        for t in range(T):
            if t % (interval * 2) < interval:
                rewards[0, t] = np.random.choice([0, 1], p=[pHigh, pLow])
                rewards[1, t] = np.random.choice([0, 1], p=[pLow, pHigh])
            else:
                rewards[0, t] = np.random.choice([0, 1], p=[pLow, pHigh])
                rewards[1, t] = np.random.choice([0, 1], p=[pHigh, pLow])

    
    

    return rewards

def generate_drifting_binary_bandit(
        N_participants=200,
        T=200,                  # trials per block
        Bk=10,                  # blocks per participant
        sigma_rw=0.10,          # volatility of reward drift
        sigma_participant=0.05, # participant-specific variability in drift
        init_mean=0.5,          # initial reward mean
    ):
    """
    Generates a 2-armed drifting bandit environment with:
        - binary rewards
        - drifting reward probabilities
        - per-participant volatility σ_rw
    Output:
        rewards: shape (N, Bk, T, 2)
        params: dict with α, β, σ_rw per participant
    """

    rewards = np.zeros((N_participants, Bk, T, 2))

    # --------------------------
    # 1. Sample participant traits
    # --------------------------
    sigmas = np.abs(np.random.normal(sigma_rw, sigma_participant, N_participants))

    # --------------------------
    # 2. Loop participants
    # --------------------------
    for n in range(N_participants):
        for b in range(Bk):

            # drifting reward logits for both arms
            mu = np.array([init_mean, 1-init_mean])  # starting means for 2 arms

            for t in range(T):

                # convert means to probabilities
                p = np.clip(mu, 0.001, 0.999)

                # sample rewards
                r0 = np.random.rand() < p[0]
                r1 = np.random.rand() < p[1]

                rewards[n,b,t,0] = int(r0)
                rewards[n,b,t,1] = int(r1)

                # drift both means independently
                mu += np.random.normal(0, sigmas[n], size=2)

                # clamp to valid range
                mu = np.clip(mu, 0.001, 0.999)

    params = {
        "sigma_rw": sigmas
    }

    return rewards, params

def gen_data_AsymmetricQlearning_variable_alpha(rewards, seed=1979, n_trials=200, n_sessions=10, 
                                                alphaP_list=None, alphaN_list=None, 
                                                beta=0.3, alphaF=0.5):
    """
    Generates simulated data using an Asymmetric Q-learning model where alphaP and alphaN can vary by session.

    Parameters:
    - rewards (ndarray): A 2D array of reward values for each option (shape: [2, n_trials]).
    - seed (int): Random seed for reproducibility.
    - n_trials (int): Number of trials per session.
    - n_sessions (int): Number of sessions to simulate.
    - alphaP_list (list): List of alphaP values for each session. If None, defaults to a constant value of 0.8.
    - alphaN_list (list): List of alphaN values for each session. If None, defaults to a constant value of 0.2.
    - beta (float): Inverse temperature parameter controlling exploration vs. exploitation.
    - alphaF (float): Forgetting rate applied to all Q-values after each trial.

    Returns:
    - c (ndarray): Array of choices made in each session and trial (shape: [n_sessions, n_trials]).
    - r (ndarray): Array of rewards received for each choice (shape: [n_sessions, n_trials]).
    - p (ndarray): Probability of choosing option 1 in each session and trial (shape: [n_sessions, n_trials]).
    - Q (ndarray): Q-values for each option in each session and trial (shape: [n_sessions, 2, n_trials]).
    """

    np.random.seed(seed)

    # Initialize numpy arrays to store Q-values, choices, rewards, and probabilities
    Q = np.zeros((n_sessions, 2, n_trials))
    c = np.zeros((n_sessions, n_trials), dtype=int)  # choice
    r = np.zeros((n_sessions, n_trials))  # reward
    p = np.zeros((n_sessions, n_trials))  # probability of choosing option 1

    # If alphaP_list or alphaN_list is not provided, use default constant values
    if alphaP_list is None:
        alphaP_list = [0.8] * n_sessions
    if alphaN_list is None:
        alphaN_list = [0.2] * n_sessions

    for session in range(n_sessions):
        alphaP = alphaP_list[session]
        alphaN = alphaN_list[session]
        
        for t in range(n_trials):
            if t % 50 == 0:  # At every 50th trial, switch the reward probabilities
                pr = np.array([0.7, 0.3]) if t % 100 == 0 else np.array([0.3, 0.7])
            
            # Calculate the probability of choosing option 0
            p[session, t] = np.exp(beta * Q[session, 0, t]) / np.exp(beta * Q[session, :, t]).sum()
            
            # Determine the choice: option 0 or 1
            c[session, t] = np.random.choice([0, 1], p=[p[session, t], 1 - p[session, t]])

            # Determine the outcome: 0 (no reward) or 1 (reward)
            r[session, t] = rewards[c[session, t], t]

            # Update Q-values for the next trial, applying the learning rates alphaP and alphaN
            if t < n_trials - 1:
                Q[session, :, t + 1] = (1 - alphaF) * Q[session, :, t]
                if r[session, t] - Q[session, c[session, t], t] > 0:
                    Q[session, c[session, t], t + 1] = Q[session, c[session, t], t] + alphaP * (r[session, t] - Q[session, c[session, t], t])
                else:
                    Q[session, c[session, t], t + 1] = Q[session, c[session, t], t] + alphaN * (r[session, t] - Q[session, c[session, t], t])

    return c, r, p, Q



def generate_parameter_lists(true_model, ind_diff_type, Delta_alpha=None, nSession=100):
    """
    Generate parameter lists based on model type and individual difference type.
    
    Parameters:
    - true_model (str): The type of true model (e.g., 'Q', 'FQ', etc.).
    - ind_diff_type (str): Type of individual difference ('continuous_all', 'continuous_alpha_only', 'None', 'discrete_alpha_only', 'full_range').
    - Delta_alpha (float or None): Used only when ind_diff_type is 'discrete_alpha_only'.
    - nSession (int): Number of sessions (default: 100).
    
    Returns:
    - Dictionary containing parameter lists.
    """
    

    if "A" in true_model:
        if ind_diff_type in ['continuous_all', 'continuous_alpha_only']:
            alphaP_list = np.random.uniform(0.4, 0.9, nSession).tolist()
            alphaP_list.sort()
            alphaN_list = np.random.uniform(0.1, 0.6, nSession).tolist()
        elif ind_diff_type == 'None':
            alphaP_list = [0.8] * nSession
            alphaN_list = [0.2] * nSession
        else:
            raise ValueError(f"Unknown ind_diff_type: {ind_diff_type}")
    else:
        if ind_diff_type in ['continuous_all', 'continuous_alpha_only', 'full_range']:
            alphaP_list = np.random.uniform(0.1, 0.9, nSession).tolist()
            alphaP_list.sort()
        elif ind_diff_type == 'None':
            alphaP_list = [0.3] * nSession
        elif ind_diff_type == 'discrete_alpha_only':
            if Delta_alpha is None:
                raise ValueError("Delta_alpha must be provided for discrete_alpha_only.")
            alphaP_list = [0.5 - Delta_alpha * 0.5] * (nSession // 2) + [0.5 + Delta_alpha * 0.5] * (nSession // 2)
        elif ind_diff_type == 'uniform':
            alphaP_list = np.random.uniform(0.1, 0.9, nSession).tolist()
        else:
            raise ValueError(f"Unknown ind_diff_type: {ind_diff_type}")
        
        alphaN_list = alphaP_list.copy()

    alphaF_list = alphaP_list.copy() if true_model == 'FQ' else [0.0] * nSession
    
    #beta_list = np.random.uniform(1.0, 4.0, nSession).tolist() if ind_diff_type == 'continuous_all' else [3.0] * nSession
    beta_list = np.random.uniform(1.0, 4.0, nSession).tolist() if ind_diff_type in ['continuous_all', 'full_range'] else [3.0] * nSession
    if "C" in true_model:
        phi_list = [0] * nSession 
        tau_list = [0] * nSession 
        if ind_diff_type == 'continuous_all':
            phi_list = np.random.uniform(0.5, 2.5, nSession).tolist()
            tau_list = np.random.uniform(0.1, 0.9, nSession).tolist()
        elif ind_diff_type == 'full_range':
            phi_list = np.random.uniform(-2.5, 2.5, nSession).tolist()
            tau_list = np.random.uniform(0.1, 0.9, nSession).tolist()
        else: 
            phi_list = [1.0] * nSession 
            tau_list = [0.3] * nSession 
    else:
        phi_list = [0] * nSession 
        tau_list = [0] * nSession 
    
    # Print the first 5 values for confirmation
    print("alphaP_list (first 5, last 5):", [round(x, 2) for x in alphaP_list[:5]], [round(x, 2) for x in alphaP_list[-5:]])
    print("alphaN_list (first 5, last 5):", [round(x, 2) for x in alphaN_list[:5]], [round(x, 2) for x in alphaN_list[-5:]])
    print("alphaF_list (first 5, last 5):", [round(x, 2) for x in alphaF_list[:5]], [round(x, 2) for x in alphaF_list[-5:]])
    print("beta_list (first 5, last 5):", [round(x, 2) for x in beta_list[:5]], [round(x, 2) for x in beta_list[-5:]])
    print("phi_list (first 5, last 5):", [round(x, 2) for x in phi_list[:5]], [round(x, 2) for x in phi_list[-5:]])
    print("tau_list (first 5, last 5):", [round(x, 2) for x in tau_list[:5]], [round(x, 2) for x in tau_list[-5:]])
    
    return {
        "alphaP_list": alphaP_list,
        "alphaN_list": alphaN_list,
        "alphaF_list": alphaF_list,
        "beta_list": beta_list,
        "phi_list": phi_list,
        "tau_list": tau_list
    }

def plot_kl_and_train_loss(train_loss, kl_loss, sim_nametag):
    # Ensure same length
    assert len(train_loss) == len(kl_loss)
    print(type(train_loss))

    epochs = list(range(1, len(train_loss) + 1))
    train_loss_values = [loss.detach().cpu().item() for loss in train_loss]
    kl_loss_values = [loss for loss in kl_loss]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # === Plot loss (left y-axis) ===
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss', color=color1)
    ax1.plot(epochs, train_loss_values, label='Train Loss', color='tab:red', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1)

    # === Create second y-axis for KL ===
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('KL Divergence', color=color2)
    ax2.plot(epochs, kl_loss_values, label='KL Divergence', color=color2, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color2)

    # === Add legend and title ===
    fig.suptitle("RNN Training: Loss & KL per Epoch", fontsize=14)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.grid(True)
    plt.show()
    plt.savefig(f"plots/vanillaRNNtraining{sim_nametag}.png")
    print(f"training plotted under: plots/vanillaRNNtraining{sim_nametag}.png ")
    figure = plt.gcf()
    return figure
    