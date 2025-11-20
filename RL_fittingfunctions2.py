import numpy as np
import pandas as pd
from scipy.optimize import minimize

# def qlearning_full(param, sessions, choices, rewards, context, model_config):
#     """
#     Q-learning function for computing log-likelihood across all sessions.

#     Parameters:
#     - param: Model parameters to estimate.
#     - sessions: Array of session identifiers.
#     - choices: Array of choices made (can include -1 for missed responses).
#     - rewards: Array of rewards received.
#     - context: Array of context identifiers (or None).
#     - model_config: Dictionary specifying the model configuration.

#     Returns:
#     - Negative log-likelihood.
#     - p1_series: Probability of choosing option 1 for each trial.
#     """
#     # Extract model configuration
#     asymmetric_alpha = model_config.get("asymmetric_alpha", False)
#     forgetting_type = model_config.get("forgetting_type", "none")
#     choice_trace = model_config.get("choice_trace", False)

#     # Extract parameters
#     idx = 0
#     if asymmetric_alpha:
#         alphaP = param[idx]
#         idx += 1
#         alphaN = param[idx]
#         idx += 1
#     else:
#         alpha = param[idx]
#         idx += 1

#     if forgetting_type == "free":
#         alphaF = param[idx]
#         idx += 1
#     elif forgetting_type == "fixed":
#         alphaF = alpha if not asymmetric_alpha else (alphaP + alphaN) * 0.5  
#     else:
#         alphaF = 0.0

#     beta = param[idx]
#     idx += 1

#     if choice_trace:
#         phi = param[idx]
#         idx += 1
#         tau = param[idx]
#         idx += 1
#     else:
#         phi = 0.0
#         tau = 0.0

#     # Initialize variables
#     ll = 0
#     unique_sessions = np.unique(sessions)
#     p1_series = np.zeros(len(sessions))

#     # loop over participants (sessions)
#     for session in unique_sessions:
#         session_indices = np.where(sessions == session)[0]
#         choice = choices[session_indices]
#         reward = rewards[session_indices]
#         context_session = context[session_indices] if context is not None else None

#         T = len(choice)
#         Q = np.zeros((2, T))
#         CT = np.zeros((2, T))  # Choice trace
#         p1 = np.zeros(T)

#         for t in range(T):
#             if t > 0:  # Skip this for the first trial
#                 if context_session is not None:
#                     if context_session[t] != context_session[t - 1]:
#                         Q[:, t] = 0  # Reset Q-values on context change
#                         CT[:, t] = 0  # Reset choice trace on context change    

#             if choice[t] == -1:  # Skip trials with missed responses
#                 if t < T - 1:
#                     Q[:, t + 1] = Q[:, t]  # Carry forward previous Q-values
#                     CT[:, t + 1] = CT[:, t]  # Carry forward previous CT values
#                 continue

#             # Compute choice probabilities
#             p1[t] = np.exp(beta * Q[0, t] + phi * CT[0, t]) / np.exp(beta * Q[:, t] + phi * CT[:, t]).sum()
#             # print(f"t: {t}, choice: {choice[t]}, p1: {p1[t]:.4f}, Q-values: {Q[:, t]}, alphaF: {alphaF}")
#             likelihood = np.clip((choice[t] == 0) * p1[t] + (choice[t] == 1) * (1 - p1[t]), 0.00001, 0.99999)
#             ll += np.log(likelihood)

#             # Update Q-values and choice trace for the next trial
#             if t < T - 1:
#                 Q[:, t + 1] = (1 - alphaF) * Q[:, t]  # Forgetting term
#                 delta = reward[t] - Q[choice[t], t]
#                 Q[choice[t], t + 1] = Q[choice[t], t] + (
#                     alphaP if asymmetric_alpha and delta > 0 else
#                     alphaN if asymmetric_alpha else
#                     alpha
#                 ) * delta
#                 # print(f"t: {t}, choice: {choice[t]}, Q(t): {Q[:, t]}, Q(t+1): {Q[:, t+1]}")

#                 # Update choice trace
#                 CT[choice[t], t + 1] = CT[choice[t], t] + tau * (1 - CT[choice[t], t])
#                 CT[1 - choice[t], t + 1] = CT[1 - choice[t], t] + tau * (0 - CT[1 - choice[t], t])

#         p1_series[session_indices] = p1

#     return -ll, p1_series

def get_q_values(param, df, model_config, trials = 200):
    """
    Q-learning function for computing log-likelihood across all sessions.

    Parameters:
    - param: Model parameters to estimate.
    - sessions: Array of session identifiers.
    - choices: Array of choices made (can include -1 for missed responses).
    - rewards: Array of rewards received.
    - context: Array of context identifiers (or None).
    - model_config: Dictionary specifying the model configuration.

    Returns:
    - Negative log-likelihood.
    - p1_series: Probability of choosing option 1 for each trial.
    """
    print(f"getting q values of common fit model")
    # Extract data once
    sessions = df['session'].values
    choices  = df['c'].values
    rewards  = df['r'].values
    context  = df['context'].values if 'context' in df.columns else None

    # Unique session list
    unique_sessions = np.unique(sessions)
    session_to_idx  = {s: i for i, s in enumerate(unique_sessions)}

    # Prepare output array
    n_sessions = len(unique_sessions)
    Q_all = np.zeros((n_sessions, 2, trials)) 

    # Extract model configuration
    asymmetric_alpha = model_config.get("asymmetric_alpha", False)
    forgetting_type = model_config.get("forgetting_type", "none")
    choice_trace = model_config.get("choice_trace", False)
    init_Q_free = model_config.get("init_Q_free", False)  

    

    # Extract parameters
    idx = 0
    if init_Q_free:
        Q_init_0 = param[idx]  
        idx += 1
        Q_init_1 = param[idx]  
        idx += 1
    else:
        Q_init_0 = 0.0
        Q_init_1 = 0.0

    if asymmetric_alpha:
        alphaP = param[idx]
        idx += 1
        alphaN = param[idx]
        idx += 1
    else:
        alpha = param[idx]
        idx += 1

    if forgetting_type == "free":
        alphaF = param[idx]
        idx += 1
    elif forgetting_type == "fixed":
        alphaF = alpha if not asymmetric_alpha else (alphaP + alphaN) * 0.5  
    else:
        alphaF = 0.0

    beta = param[idx]
    idx += 1

    if choice_trace:
        phi = param[idx]
        idx += 1
        tau = param[idx]
        idx += 1
    else:
        phi = 0.0
        tau = 0.0

    # Initialize variables
    ll = 0
    p1_series = np.zeros(len(sessions))
    
    # loop over participants (sessions)
    for session in unique_sessions:
        s_idx = session_to_idx[session]
        session_indices = np.where(sessions == session)[0]
        choice = choices[session_indices]
        reward = rewards[session_indices]
        context_session = context[session_indices] if context is not None else None        

        T = len(choice)
        Q = np.zeros((2, T))
        CT = np.zeros((2, T))  # Choice trace
        p1 = np.zeros(T)

        Q[0, 0] = Q_init_0
        Q[1, 0] = Q_init_1

        for t in range(T):

            if t >= len(choice):
                print(f"Warning: t={t} exceeds choice array length {len(choice)} in session {session}")
                continue
            if choice[t] not in [-1, 0, 1]:
                print(f"Error: Invalid choice value {choice[t]} at t={t} in session {session}")
                continue

            if t > 0:  
                if context_session is not None and context_session[t] != context_session[t - 1]:
                    Q[:, t] = [Q_init_0, Q_init_1]  # Reset Q-values on context change
                    CT[:, t] = 0  # Reset choice trace on context change  

            # store Q-values *at decision time t* for this trial
            Q_all[s_idx, :, t] = Q[:, t]

            if choice[t] == -1:  # Skip trials with missed responses
                if t < T - 1:
                    Q[:, t + 1] = Q[:, t]  # Carry forward previous Q-values
                    CT[:, t + 1] = CT[:, t]  # Carry forward previous CT values
                continue

            # Compute choice probabilities
            p1[t] = np.exp(beta * Q[0, t] + phi * CT[0, t]) / np.exp(beta * Q[:, t] + phi * CT[:, t]).sum()
            likelihood = np.clip((choice[t] == 0) * p1[t] + (choice[t] == 1) * (1 - p1[t]), 0.00001, 0.99999)
            ll += np.log(likelihood)

            # Update Q-values and choice trace for the next trial
            if t < T - 1:
                Q[:, t + 1] = (1 - alphaF) * Q[:, t]  # Forgetting term
                delta = reward[t] - Q[choice[t], t]
                Q[choice[t], t + 1] = Q[choice[t], t] + (
                    alphaP if asymmetric_alpha and delta > 0 else
                    alphaN if asymmetric_alpha else
                    alpha
                ) * delta

                # Update choice trace
                CT[choice[t], t + 1] = CT[choice[t], t] + tau * (1 - CT[choice[t], t])
                CT[1 - choice[t], t + 1] = CT[1 - choice[t], t] + tau * (0 - CT[1 - choice[t], t])

        p1_series[session_indices] = p1

    return Q_all


def qlearning_full(param, sessions, choices, rewards, context, model_config):
    """
    Q-learning function for computing log-likelihood across all sessions.

    Parameters:
    - param: Model parameters to estimate.
    - sessions: Array of session identifiers.
    - choices: Array of choices made (can include -1 for missed responses).
    - rewards: Array of rewards received.
    - context: Array of context identifiers (or None).
    - model_config: Dictionary specifying the model configuration.

    Returns:
    - Negative log-likelihood.
    - p1_series: Probability of choosing option 1 for each trial.
    """
    # Extract model configuration
    asymmetric_alpha = model_config.get("asymmetric_alpha", False)
    forgetting_type = model_config.get("forgetting_type", "none")
    choice_trace = model_config.get("choice_trace", False)
    init_Q_free = model_config.get("init_Q_free", False)  

    # Extract parameters
    idx = 0
    if init_Q_free:
        Q_init_0 = param[idx]  
        idx += 1
        Q_init_1 = param[idx]  
        idx += 1
    else:
        Q_init_0 = 0.0
        Q_init_1 = 0.0

    if asymmetric_alpha:
        alphaP = param[idx]
        idx += 1
        alphaN = param[idx]
        idx += 1
    else:
        alpha = param[idx]
        idx += 1

    if forgetting_type == "free":
        alphaF = param[idx]
        idx += 1
    elif forgetting_type == "fixed":
        alphaF = alpha if not asymmetric_alpha else (alphaP + alphaN) * 0.5  
    else:
        alphaF = 0.0

    beta = param[idx]
    idx += 1

    if choice_trace:
        phi = param[idx]
        idx += 1
        tau = param[idx]
        idx += 1
    else:
        phi = 0.0
        tau = 0.0

    # Initialize variables
    ll = 0
    unique_sessions = np.unique(sessions)
    p1_series = np.zeros(len(sessions))
    
    # loop over participants (sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        choice = choices[session_indices]
        reward = rewards[session_indices]
        context_session = context[session_indices] if context is not None else None        

        T = len(choice)
        Q = np.zeros((2, T))
        CT = np.zeros((2, T))  # Choice trace
        p1 = np.zeros(T)

        Q[0, 0] = Q_init_0
        Q[1, 0] = Q_init_1

        for t in range(T):

            if t >= len(choice):
                print(f"Warning: t={t} exceeds choice array length {len(choice)} in session {session}")
                continue
            if choice[t] not in [-1, 0, 1]:
                print(f"Error: Invalid choice value {choice[t]} at t={t} in session {session}")
                continue

            if t > 0:  
                if context_session is not None and context_session[t] != context_session[t - 1]:
                    Q[:, t] = [Q_init_0, Q_init_1]  # Reset Q-values on context change
                    CT[:, t] = 0  # Reset choice trace on context change  


            if choice[t] == -1:  # Skip trials with missed responses
                if t < T - 1:
                    Q[:, t + 1] = Q[:, t]  # Carry forward previous Q-values
                    CT[:, t + 1] = CT[:, t]  # Carry forward previous CT values
                continue

            # Compute choice probabilities
            p1[t] = np.exp(beta * Q[0, t] + phi * CT[0, t]) / np.exp(beta * Q[:, t] + phi * CT[:, t]).sum()
            likelihood = np.clip((choice[t] == 0) * p1[t] + (choice[t] == 1) * (1 - p1[t]), 0.00001, 0.99999)
            ll += np.log(likelihood)

            # Update Q-values and choice trace for the next trial
            if t < T - 1:
                Q[:, t + 1] = (1 - alphaF) * Q[:, t]  # Forgetting term
                delta = reward[t] - Q[choice[t], t]
                Q[choice[t], t + 1] = Q[choice[t], t] + (
                    alphaP if asymmetric_alpha and delta > 0 else
                    alphaN if asymmetric_alpha else
                    alpha
                ) * delta

                # Update choice trace
                CT[choice[t], t + 1] = CT[choice[t], t] + tau * (1 - CT[choice[t], t])
                CT[1 - choice[t], t + 1] = CT[1 - choice[t], t] + tau * (0 - CT[1 - choice[t], t])

        p1_series[session_indices] = p1

    return -ll, p1_series


def parse_model_config(model_config):
    """
    Validate and ensure model_config is a dictionary.

    Parameters:
    - model_config: A dictionary specifying the model configuration.

    Returns:
    - dict: The validated configuration dictionary.
    """
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary.")
    required_keys = {"asymmetric_alpha", "forgetting_type", "choice_trace"}
    if not required_keys.issubset(model_config.keys()):
        raise ValueError(f"model_config must contain the keys: {required_keys}")
    return model_config


def fit_qlearning_model(opt_function, sessions, choices, rewards, context=None, model_config=None, n_iter=10):
    """
    Fit the Q-learning model using repeated optimization to find the best parameters.

    Parameters:
    - opt_function: Function to compute log-likelihood.
    - sessions: Array of session identifiers.
    - choices: Array of choices made.
    - rewards: Array of rewards received.
    - context: Array of context identifiers or None (default: None).
    - model_config: Dictionary specifying the model configuration.
    - n_iter: Number of optimization iterations to find the best result.

    Returns:
    - best_params: Best estimated parameters.
    - best_neg_ll: Best negative log-likelihood.
    - best_p1: Probability of choosing option 1 for all trials.
    """
    # Validate model_config
    config = parse_model_config(model_config) if model_config else {}
    asymmetric_alpha = config.get("asymmetric_alpha", False)
    forgetting_type = config.get("forgetting_type", "none")
    choice_trace = config.get("choice_trace", False)
    init_Q_free = model_config.get("init_Q_free", False)  

    idx = 0
    # Initialize lists
    lblist = []
    ublist = []

    if init_Q_free:
        lblist += [0, 0]
        ublist += [1, 1] 

    # Define bounds for optimization
    lblist += [0]  # alpha or alphaP
    ublist += [1]  # alpha or alphaP
    if asymmetric_alpha:
        lblist += [0]  # alphaN
        ublist += [1]  # alphaN
    if forgetting_type == "free":
        lblist += [0]  # alphaF
        ublist += [1]  # alphaF
    lblist += [0]  # beta
    ublist += [20]  # beta

    if choice_trace:
        lblist += [-10, 0]  # phi, tau
        ublist += [10, 1]  # phi, tau

    # Optimization
    best_neg_ll = np.inf
    best_params = None
    best_p1 = None

    for _ in range(n_iter):
        param_ini = np.random.uniform(0, 0.9, len(lblist))

        res = minimize(
            fun=lambda param: opt_function(param, sessions, choices, rewards, context, config)[0],  # contextを追加
            x0=param_ini,
            method='SLSQP',  # 'L-BFGS-B', # 'TNC'
            bounds=list(zip(lblist, ublist)),
            tol=1e-4
        )

        neg_ll, p1 = opt_function(res.x, sessions, choices, rewards, context, config)
        if neg_ll < best_neg_ll:
            best_neg_ll = neg_ll
            best_params = res.x
            best_p1 = p1

    neg_ll, p1 = opt_function(best_params, sessions, choices, rewards, context, config)
    return best_params, neg_ll, p1


def fit_qlearning_common(df, model_config, n_iter=10):
    """
    Fit common parameters across all sessions (participants) for the Q-learning model.

    Parameters:
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_config: Dictionary specifying the model configuration.
    - n_iter: Number of optimization iterations to find the best result.

    Returns:
    - best_params: Best estimated parameters.
    - best_neg_ll: Best negative log-likelihood.
    - p1_common: Probability of choosing option 1 for all trials.
    """
    sessions = df['session'].values
    choices = df['c'].values
    rewards = df['r'].values
    context = df['context'].values if 'context' in df.columns else None

    print(choices)
    print(rewards)
    print(context)
    print(n_iter)

    best_params, best_neg_ll, p1_common = fit_qlearning_model(
        qlearning_full, sessions, choices, rewards, context, model_config, n_iter
    )
    
    return best_params, best_neg_ll, p1_common


def fit_qlearning_by_session(df, model_config, n_iter=10):
    """
    Fit parameters for each session independently for the Q-learning model.

    Parameters:
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_config: Dictionary specifying the model configuration.
    - n_iter: Number of optimization iterations to find the best result.

    Returns:
    - session_results: DataFrame with session-specific parameters and log-likelihoods.
    - p1_individual: Probability of choosing option 1 for all trials, per session.
    """
    session_results = []
    p1_individual = np.zeros(len(df))

    for session, group in df.groupby('session'):
        session_indices = group.index
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values

        context = group['context'].values if 'context' in group.columns else None

        best_params, best_neg_ll, p1 = fit_qlearning_model(
            qlearning_full, sessions, choices, rewards, context, model_config, n_iter
        )
        p1_individual[session_indices] = p1

        session_results.append({
            'session': session,
            'params': best_params,
            'neg_log_likelihood': best_neg_ll
        })

    return pd.DataFrame(session_results), p1_individual


def compute_negll_and_p1(opt_function, param, sessions, choices, rewards, context=None, model_fit=None):
    """
    Compute the negative log-likelihood and probabilities using given parameters.

    Parameters:
    - opt_function: Function to compute log-likelihood and probabilities.
    - param: Parameters to evaluate the model.
    - sessions: Array of session identifiers.
    - choices: Array of choices made.
    - rewards: Array of rewards received.
    - context: Array of context identifiers (optional).
    - model_fit: String indicating which parameters are included in the model.

    Returns:
    - neg_ll: Negative log-likelihood.
    - p1: Probability of choosing option 1 for all trials.
    """
    neg_ll, p1 = opt_function(param, sessions, choices, rewards, context, model_fit)
    return neg_ll, p1


def compute_negll_and_p1_individual_fit(opt_function, params_per_session, df, model_fit):
    """
    Compute negative log-likelihood and probabilities for each session using given parameters.

    Parameters:
    - opt_function: Function to compute log-likelihood and probabilities.
    - params_per_session: Dictionary mapping session IDs to parameter sets.
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_fit: String indicating which parameters are included in the model.

    Returns:
    - total_neg_ll: Total negative log-likelihood across all sessions.
    - p1_all: Continuous series of probabilities for all trials across sessions.
    """
    total_neg_ll = 0
    p1_all = np.zeros(len(df))

    for session, group in df.groupby('session'):
        session_indices = group.index
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values if 'context' in group.columns else None

        param = params_per_session.get(session)
        if param is None:
            raise ValueError(f"Parameters for session {session} are not provided.")

        neg_ll, p1 = opt_function(param, sessions, choices, rewards, context, model_fit)
        total_neg_ll += neg_ll
        p1_all[session_indices] = p1

    return total_neg_ll, p1_all


def compute_negll_and_p1_common_fit(opt_function, common_param, df, model_fit):
    """
    Compute negative log-likelihood and probabilities across all sessions using a common parameter set.

    Parameters:
    - opt_function: Function to compute log-likelihood and probabilities.
    - common_param: Common parameter set to evaluate the model.
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_fit: String indicating which parameters are included in the model.

    Returns:
    - neg_ll: Negative log-likelihood across all sessions.
    - p1_all: Continuous series of probabilities for all trials across sessions.
    """
    sessions = df['session'].values
    choices = df['c'].values
    rewards = df['r'].values
    context = df['context'].values if 'context' in df.columns else None

    neg_ll, p1_all = opt_function(common_param, sessions, choices, rewards, context, model_fit)
    return neg_ll, p1_all



def compute_negll_and_normalized_ll_per_session_individual_fit(opt_function, params_per_session, df, model_fit):
    """
    Compute negative log-likelihood and normalized likelihood for each session using given parameters.
    The function calls `opt_function` with all trials (including missed responses, choice == -1),
    but the normalized likelihood is computed using only valid trials.
    """
    session_results = []

    for session, group in df.groupby('session'):
        # Extract data (keep all trials for opt_function)
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values if 'context' in group.columns else None

        # Count only valid trials (choice != -1) for normalization
        num_valid_trials = np.sum(choices != -1)  # Count valid trials

        if num_valid_trials == 0:
            print(f"Skipping session {session} (all trials are missing responses)")
            continue

        param = params_per_session.get(session)
        if param is None:
            raise ValueError(f"Parameters for session {session} are not provided.")

        # Compute negative log-likelihood with ALL trials
        neg_ll, _ = opt_function(param, sessions, choices, rewards, context, model_fit)

        # Normalize using only valid trials
        normalized_ll = np.exp(-neg_ll / num_valid_trials)  # Only count valid trials

        print(f"Session: {session}, neg_ll: {neg_ll}, num_valid_trials: {num_valid_trials}, normalized_ll: {normalized_ll}")

        session_results.append({
            "session": session,
            "neg_log_likelihood": neg_ll,
            "normalized_likelihood": normalized_ll
        })

    return pd.DataFrame(session_results)


def compute_negll_and_normalized_ll_per_session_common_fit(opt_function, common_params, df, model_fit):
    """
    Compute negative log-likelihood and normalized likelihood for each session using common parameters.
    The function calls `opt_function` with all trials (including missed responses, choice == -1),
    but the normalized likelihood is computed using only valid trials.
    """
    session_results = []

    for session, group in df.groupby('session'):
        # Extract data (keep all trials for opt_function)
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values if 'context' in group.columns else None

        # Count only valid trials (choice != -1) for normalization
        num_valid_trials = np.sum(choices != -1)  # Count valid trials

        if num_valid_trials == 0:
            print(f"Skipping session {session} (all trials are missing responses)")
            continue

        # Compute negative log-likelihood with ALL trials
        neg_ll, _ = opt_function(common_params, sessions, choices, rewards, context, model_fit)

        # Normalize using only valid trials
        normalized_ll = np.exp(-neg_ll / num_valid_trials)  # Only count valid trials

        print(f"Session: {session}, neg_ll: {neg_ll}, num_valid_trials: {num_valid_trials}, normalized_ll: {normalized_ll}")

        session_results.append({
            "session": session,
            "neg_log_likelihood": neg_ll,
            "normalized_likelihood": normalized_ll
        })

    return pd.DataFrame(session_results)


# for MAP ------------------------------------------------------------------------------------ #

from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm



from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def qlearning_full_with_prior(param, sessions, choices, rewards, context=None, model_config=None):
    """
    Q-learning function with priors for computing log-posterior (MAP).
    Uses `qlearning_full` for value updating and adds prior probability.

    Parameters:
    - param: Model parameters to estimate.
    - sessions: Array of session identifiers.
    - choices: Array of choices made.
    - rewards: Array of rewards received.
    - context: Array of context identifiers (optional).
    - model_config (dict): Configuration dictionary.

    Returns:
    - Negative log-posterior (MAP).
    - p1_series: Probability of choosing option 1 for each trial.
    """
    # Extract model configuration
    asymmetric_alpha = model_config.get("asymmetric_alpha", False)
    forgetting_type = model_config.get("forgetting_type", "none")
    choice_trace = model_config.get("choice_trace", False)
    init_Q_free = model_config.get("init_Q_free", False)  

    # Extract parameters and apply priors
    idx = 0
    prior_log_prob = 0  # Initialize prior log-probability

    if init_Q_free:
        idx += 2

    # Alpha parameters
    if asymmetric_alpha:
        alphaP = np.clip(param[idx], 1e-3, 1 - 1e-3)
        prior_log_prob += beta_dist.logpdf(alphaP, a=1.1, b=1.1)
        idx += 1
        alphaN = np.clip(param[idx], 1e-3, 1 - 1e-3)
        prior_log_prob += beta_dist.logpdf(alphaN, a=1.1, b=1.1)
        idx += 1
    else:
        alpha = np.clip(param[idx], 1e-3, 1 - 1e-3)
        prior_log_prob += beta_dist.logpdf(alpha, a=1.1, b=1.1)
        idx += 1

    # Forgetting rate
    if forgetting_type == "free":
        alphaF = np.clip(param[idx], 1e-3, 1 - 1e-3)
        prior_log_prob += beta_dist.logpdf(alphaF, a=1.1, b=1.1)
        idx += 1
    elif forgetting_type == "fixed":
        alphaF = alpha if not asymmetric_alpha else (alphaP + alphaN) * 0.5
    else:
        alphaF = 0.0

    # Beta parameter (inverse temperature)
    beta = np.clip(param[idx], 1e-3, 20)
    prior_log_prob += gamma_dist.logpdf(beta, a=1.2, scale=5.0)
    idx += 1

    # Choice trace parameters
    if choice_trace:
        phi = param[idx]
        prior_log_prob += norm.logpdf(phi, loc=0, scale=np.sqrt(5))
        idx += 1
        tau = np.clip(param[idx], 1e-3, 1 - 1e-3)
        prior_log_prob += beta_dist.logpdf(tau, a=1.1, b=1.1)
        idx += 1
    else:
        phi = 0.0
        tau = 0.0

    # Compute log-likelihood using `qlearning_full`
    neg_ll, p1_series = qlearning_full(param, sessions, choices, rewards, context, model_config)

    # Return negative log-posterior (log-likelihood + log-prior)
    return neg_ll - prior_log_prob, p1_series


def fit_qlearning_by_session_MAP(df, model_config, n_iter=10):
    """
    Fit parameters for each session independently using MAP estimation.

    Parameters:
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_config: Dictionary specifying the model configuration.
    - n_iter: Number of optimization iterations.

    Returns:
    - session_results: DataFrame with session-specific parameters and log-likelihoods.
    - p1_individual: Probability of choosing option 1 for all trials, per session.
    """
    session_results = []
    p1_individual = np.zeros(len(df))

    for session, group in df.groupby('session'):
        session_indices = group.index
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values if 'context' in group.columns else None

        best_params, best_neg_ll, p1 = fit_qlearning_model(
            qlearning_full_with_prior, sessions, choices, rewards, context, model_config, n_iter
        )
        p1_individual[session_indices] = p1

        session_results.append({
            'session': session,
            'params': best_params,
            'neg_log_likelihood': best_neg_ll
        })

    return pd.DataFrame(session_results), p1_individual


def fit_qlearning_by_session_MAP2(df, model_config, n_iter=10):
    """
    Fit parameters for each session independently using MAP estimation.

    Parameters:
    - df: DataFrame containing session, choices, rewards, and optionally context.
    - model_config: Dictionary specifying the model configuration.
    - n_iter: Number of optimization iterations.

    Returns:
    - df: Updated DataFrame with probability of choosing option 1 (p1).
    - session_results: DataFrame with session-specific parameters and log-likelihoods.
    """
    session_results = []
    p1_individual = np.zeros(len(df))

    for session, group in df.groupby('session'):
        session_indices = group.index
        sessions = np.full_like(group['c'].values, session)
        choices = group['c'].values
        rewards = group['r'].values
        context = group['context'].values if 'context' in group.columns else None

        best_params, best_neg_ll, p1 = fit_qlearning_model(
            qlearning_full_with_prior, sessions, choices, rewards, context, model_config, n_iter
        )
        p1_individual[session_indices] = p1

        session_results.append({
            'session': session,
            'params': best_params,
            'neg_log_likelihood': best_neg_ll
        })

    # Add p1 to the DataFrame
    df['p1'] = p1_individual
    
    return df, pd.DataFrame(session_results)

# modified in RL_fitting_functions_v2_1.py

# def fit_all_models(model_configs, df_train, df_test, n_iter):
#     """
#     Fit multiple reinforcement learning models based on the given configurations.

#     Parameters:
#     - model_configs (dict): Dictionary of model names and their configurations.
#     - df_train (DataFrame): Training dataset.
#     - df_test (DataFrame): Test dataset.
#     - n_iter (int): Number of iterations for optimization.

#     Returns:
#     - final_results (DataFrame): Combined results of all models.
#     - params_dict (dict): Dictionary of fitted parameters for all models.
#     - p1_common_dict (dict): Dictionary of p1 values for the common fit of each model.
#     - p1_ML_dict (dict): Dictionary of p1 values for the ML fit of each model.
#     - p1_MAP_dict (dict): Dictionary of p1 values for the MAP fit of each model.
#     """

#     model_results_dict = {}
#     params_dict = {}
#     p1_common_dict = {}
#     p1_ML_dict = {}
#     p1_MAP_dict = {}

#     for model_name, model_config in model_configs.items():
#         print(f"Fitting {model_name}...")
#         print(model_config)

#         # Fit common parameters
#         print("  Fit common parameters")
#         common_params, common_neg_ll, p1_common = fit_qlearning_common(
#             df_train, model_config=model_config, n_iter=n_iter
#         )
#         p1_common_dict[model_name] = p1_common
#         params_dict[f"{model_name}_common"] = common_params

#         # Fit individual parameters for each session (ML)
#         print("  Fit individual parameters (ML)")
#         session_results_df, p1_ML = fit_qlearning_by_session(
#             df_train, model_config=model_config, n_iter=n_iter
#         )
#         params_per_session = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
#         params_dict[f"{model_name}_individual"] = params_per_session
#         p1_ML_dict[model_name] = p1_ML

#         # Fit individual parameters for each session (MAP)
#         print("  Fit individual parameters (MAP)")
#         session_results_df, p1_MAP = fit_qlearning_by_session_MAP(
#             df_train, model_config=model_config, n_iter=n_iter
#         )
#         params_per_session_MAP = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
#         params_dict[f"{model_name}_MAP"] = params_per_session_MAP
#         p1_MAP_dict[model_name] = p1_MAP

#         # Compute normalized likelihoods for common fit on test data
#         print("  Compute results for common fit")
#         results_common = compute_negll_and_normalized_ll_per_session_common_fit(
#             qlearning_full, common_params, df_test, model_config
#         )
#         results_common['model'] = f"{model_name} (common fit)"

#         # Compute normalized likelihoods for individual fit on test data (ML)
#         print("  Compute results for individual fit (ML)")
#         results_individual = compute_negll_and_normalized_ll_per_session_individual_fit(
#             qlearning_full, params_per_session, df_test, model_config
#         )
#         results_individual['model'] = f"{model_name} (individual fit)"

#         # Compute normalized likelihoods for individual fit on test data (MAP)
#         print("  Compute results for individual fit (MAP)")
#         results_individual_MAP = compute_negll_and_normalized_ll_per_session_individual_fit(
#             qlearning_full, params_per_session_MAP, df_test, model_config
#         )
#         results_individual_MAP['model'] = f"{model_name} (MAP)"

#         # Combine results for common and individual fits
#         model_results = pd.concat(
#             [results_common[['session', 'normalized_likelihood', 'model']], 
#              results_individual[['session', 'normalized_likelihood', 'model']],
#              results_individual_MAP[['session', 'normalized_likelihood', 'model']]],
#             ignore_index=True
#         )
#         model_results_dict[model_name] = model_results

#     # Combine results for all models into a single DataFrame
#     final_results = pd.concat(model_results_dict.values(), ignore_index=True)

#     return final_results, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict

def fit_all_models(model_configs, df_train, df_test, n_iter, fit_common=True, fit_ML=True, fit_MAP=True):
    """
    Fit multiple reinforcement learning models based on the given configurations,
    with options to enable/disable common fit, ML fit, and MAP fit.

    Parameters:
    - model_configs (dict): Dictionary of model names and their configurations.
    - df_train (DataFrame): Training dataset.
    - df_test (DataFrame): Test dataset.
    - n_iter (int): Number of iterations for optimization.
    - fit_common (bool): Whether to fit common parameters.
    - fit_ML (bool): Whether to fit Maximum Likelihood (ML) parameters.
    - fit_MAP (bool): Whether to fit Maximum A Posteriori (MAP) parameters.

    Returns:
    - final_results (DataFrame): Combined results of all models.
    - params_dict (dict): Dictionary of fitted parameters for all models.
    - p1_common_dict (dict): Dictionary of p1 values for the common fit of each model.
    - p1_ML_dict (dict): Dictionary of p1 values for the ML fit of each model.
    - p1_MAP_dict (dict): Dictionary of p1 values for the MAP fit of each model.
    """

    model_results_dict = {}
    params_dict = {}
    p1_common_dict = {}
    p1_ML_dict = {}
    p1_MAP_dict = {}
    Q_common_dict = {}

    for model_name, model_config in model_configs.items():
        print(f"Fitting {model_name}...")
        print(model_config)

        model_results = []

        # Fit common parameters
        if fit_common:
            print("  Fit common parameters")
            common_params, common_neg_ll, p1_common = fit_qlearning_common(
                df_train, model_config=model_config, n_iter=n_iter
            )

            p1_common_dict[model_name] = p1_common
            params_dict[f"{model_name}_common"] = common_params

            # --- NEW: compute and store Q-values for common fit ---
            Q_train = get_q_values(common_params, df_train, model_config)
            Q_test  = get_q_values(common_params, df_test,  model_config)
            Q_common_dict[model_name] = {
                "train": Q_train,   # shape (n_sessions, 2, n_train_trials)
                "test":  Q_test     # shape (n_sessions, 2, n_test_trials)
            }

            # Compute normalized likelihoods for common fit on test data
            print("  Compute results for common fit")
            results_common = compute_negll_and_normalized_ll_per_session_common_fit(
                qlearning_full, common_params, df_test, model_config
            )
            results_common['model'] = f"{model_name} (common fit)"
            model_results.append(results_common[['session', 'normalized_likelihood', 'model']])

        # Fit individual parameters for each session (ML)
        if fit_ML:
            print("  Fit individual parameters (ML)")
            session_results_df, p1_ML = fit_qlearning_by_session(
                df_train, model_config=model_config, n_iter=n_iter
            )
            params_per_session = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
            params_dict[f"{model_name}_individual"] = params_per_session
            p1_ML_dict[model_name] = p1_ML

            # Compute normalized likelihoods for individual fit on test data (ML)
            print("  Compute results for individual fit (ML)")
            results_individual = compute_negll_and_normalized_ll_per_session_individual_fit(
                qlearning_full, params_per_session, df_test, model_config
            )
            results_individual['model'] = f"{model_name} (individual fit)"
            model_results.append(results_individual[['session', 'normalized_likelihood', 'model']])

        # Fit individual parameters for each session (MAP)
        if fit_MAP:
            print("  Fit individual parameters (MAP)")
            session_results_df, p1_MAP = fit_qlearning_by_session_MAP(
                df_train, model_config=model_config, n_iter=n_iter
            )
            params_per_session_MAP = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
            params_dict[f"{model_name}_MAP"] = params_per_session_MAP
            p1_MAP_dict[model_name] = p1_MAP

            # Compute normalized likelihoods for individual fit on test data (MAP)
            print("  Compute results for individual fit (MAP)")
            results_individual_MAP = compute_negll_and_normalized_ll_per_session_individual_fit(
                qlearning_full, params_per_session_MAP, df_test, model_config
            )
            results_individual_MAP['model'] = f"{model_name} (MAP)"
            model_results.append(results_individual_MAP[['session', 'normalized_likelihood', 'model']])

        # Combine results for this model
        if model_results:
            model_results_dict[model_name] = pd.concat(model_results, ignore_index=True)

    # Combine results for all models into a single DataFrame
    final_results = pd.concat(model_results_dict.values(), ignore_index=True) if model_results_dict else pd.DataFrame()

    return final_results, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict, Q_common_dict
