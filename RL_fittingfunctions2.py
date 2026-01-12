import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import truncnorm

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
        
        idxs = get_param_indices(model_fit)   # careful: here you pass model_fit = model_config
        alpha_idx = idxs["alpha"]
        print(f"[DEBUG] Session {session}: alpha used in LL = {param[alpha_idx]}")

        # Compute negative log-likelihood with ALL trials
        neg_ll, _ = opt_function(param, sessions, choices, rewards, context, model_fit)

        # Normalize using only valid trials
        normalized_ll = np.exp(-neg_ll / num_valid_trials)  # Only count valid trials

        print(f"Session: {session}, neg_ll: {neg_ll}, num_valid_trials: {num_valid_trials}, normalized_ll: {normalized_ll}")

        session_results.append({
            "session": session,
            "neg_log_likelihood": neg_ll,
            "normalized_likelihood": neg_ll ##changed back to normalized ll if need be
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
            "normalized_likelihood": neg_ll # change back to normalized ll if need be
        })

    return pd.DataFrame(session_results)


# for MAP ------------------------------------------------------------------------------------ #

from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm



from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm
import numpy as np
import pandas as pd
from scipy.optimize import minimize





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
            print(f"common params: {common_params}")

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

            ### fitting part
            #should work
            m, v, eta_vec, var_vec, m_history, v_history = run_empirical_bayes(df_train, model_config, common_params, n_iter=50)
            print(f"eta vector: {eta_vec}")
            print(f"population mean history: {m}")

            params_per_session_MAP = infer_params_for_test_EM(df_test, m, v, common_params, model_config)
            #print(f"after fitting MAP, individual params for session 1 are {params_per_session_MAP[1]}")
            #sanity checks
            test_session = df_test['session'].unique()[0]
            group = df_test[df_test['session'] == test_session]

            sessions = np.full_like(group['c'].values, test_session)
            choices  = group['c'].values
            rewards  = group['r'].values
            context  = group['context'].values if 'context' in group.columns else None

            param_common = common_params
            param_EM     = params_per_session_MAP[test_session]

            neg_ll_common, _ = qlearning_full(param_common, sessions, choices, rewards, context, model_config)
            neg_ll_EM, _     = qlearning_full(param_EM,     sessions, choices, rewards, context, model_config)

            print(f"\nSession {test_session}:")
            print(f"  common fit neg LL = {neg_ll_common}")
            print(f"  EM α only  neg LL = {neg_ll_EM}")
            print(f"  difference        = {neg_ll_common - neg_ll_EM}")

            print("Type of params_per_session_MAP:", type(params_per_session_MAP))
            first_key = list(params_per_session_MAP.keys())[0]
            print("Example session key:", first_key)
            print("Example param vector:", params_per_session_MAP[first_key])
            print("Shape:", np.array(params_per_session_MAP[first_key]).shape)
            idxs = get_param_indices(model_config)

            print("Param index mapping:", idxs)
            alpha_idx = idxs["alpha"]
            print("Alpha index:", alpha_idx)

            print("Common params:", common_params)
            print("Common alpha:", common_params[alpha_idx])

            for s, p in list(params_per_session_MAP.items())[:5]:
                print(f"Session {s}: alpha = {p[alpha_idx]}")

            alphas = np.array([p[alpha_idx] for p in params_per_session_MAP.values()])
            print("Unique alphas (rounded):", np.unique(np.round(alphas, 3)))
            betas = np.array([p[1] for p in params_per_session_MAP.values()])
            print("Unique betas (rounded):", np.unique(np.round(betas, 3)))
            for p in params_per_session_MAP.values():
                if len(p) > 2:
                    FQ = True
                    phis = np.array(p[2])
                    taus = np.array(p[3])
                else:
                    FQ=False
            if FQ:
                print("Unique phis (rounded):", np.unique(np.round(phis, 3)))
                print("Unique taus (rounded):", np.unique(np.round(taus, 3)))

            
            params_dict[f"{model_name}_MAP"] = dict(params_per_session_MAP)
            
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



def log_truncnorm_pdf_vectorized(x, mu, var, lower=0.01, upper=1):
    """Vectorized log-pdf of independent truncated normal variables."""
    std = np.sqrt(var)
    a = (lower - mu) / std
    b = (upper - mu) / std
    z = (x - mu) / std

    log_pdf = (
        -np.log(std)
        + norm.logpdf(z)
        - np.log(norm.cdf(b) - norm.cdf(a))
    )
    return np.sum(log_pdf)

def get_param_indices(model_config):
    """
    Return a dict with the index positions of parameters in the param vector,
    consistent with qlearning_full.
    """
    asymmetric_alpha = model_config.get("asymmetric_alpha", False)
    forgetting_type  = model_config.get("forgetting_type", "none")
    choice_trace     = model_config.get("choice_trace", False)
    init_Q_free      = model_config.get("init_Q_free", False)

    idx = 0
    indices = {}

    if init_Q_free:
        indices["Q_init_0"] = idx; idx += 1
        indices["Q_init_1"] = idx; idx += 1

    if asymmetric_alpha:
        indices["alphaP"] = idx; idx += 1
        indices["alphaN"] = idx; idx += 1
    else:
        indices["alpha"] = idx; idx += 1

    if forgetting_type == "free":
        indices["alphaF"] = idx; idx += 1

    indices["beta"] = idx; idx += 1

    if choice_trace:
        indices["phi"] = idx; idx += 1
        indices["tau"] = idx; idx += 1

    return indices

def set_alpha_in_param(base_param, alpha, model_config):
    """
    Return a copy of base_param with the (symmetric) alpha replaced by `alpha`.
    """
    param = np.array(base_param, copy=True)
    idxs = get_param_indices(model_config)

    if "alpha" not in idxs:
        raise ValueError("Model is not symmetric-alpha (asymmetric_alpha=True). "
                         "set_alpha_in_param currently only supports symmetric alpha.")

    param[idxs["alpha"]] = alpha
    return param

def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# MAP log posterior: log likelihood + log prior
def neg_log_posterior(eta, m, v, base_param, sessions, choices, rewards, context, model_config):
    alpha = sigmoid(eta)
    param = set_alpha_in_param(base_param, alpha, model_config)
    #param['alpha'] = alpha #pseudocode for now!!!
    neg_ll, _ = qlearning_full(param, sessions, choices, rewards, context, model_config) #qlearning full gives -ll
    log_prior = norm.logpdf(eta, loc=m, scale=np.sqrt(v))
    #print(f"Params: {params}, NLL: {-ll}, Prior: {prior}")  # Debug print
    return neg_ll - log_prior

# MAP log posterior: log likelihood + log prior
def neg_loglik_fit_individual(eta, m, v, param, sessions, choices, rewards, context, model_config):
    alpha = sigmoid(eta)
    param['alpha'] = alpha #pseudocode for now!!!
    ll = qlearning_full(param, sessions, choices, rewards, context, model_config) #qlearning full gives -ll
    return -ll

def hessian_1d(fun, theta, eps=1e-5):
    f_plus  = fun(theta + eps)
    f_0     = fun(theta)
    f_minus = fun(theta - eps)
    H = (f_plus - 2 * f_0 + f_minus) / (eps ** 2)
    return H

def hessian3d(data, m, v, eps, h_i):
    H = np.zeros((3, 3))  # Hessian matrix for 3 parameters (w1, w2, w3)
    for j in range(3):
        for k in range(3):
            e_j = np.zeros(3); e_k = np.zeros(3)
            e_j[j] = eps; e_k[k] = eps
            f1 = neg_log_posterior(h_i + e_j + e_k, data, m, v)
            f2 = neg_log_posterior(h_i + e_j - e_k, data, m, v)
            f3 = neg_log_posterior(h_i - e_j + e_k, data, m, v)
            f4 = neg_log_posterior(h_i - e_j - e_k, data, m, v)
            H[j, k] = (f1 - f2 - f3 + f4) / (4 * eps**2)
    Sigma_i = np.linalg.pinv(H)
    return H, Sigma_i

# Estimate h_i and diagonal Hessian (Σ_i)
#todo: make prior an input
def estimate_h_i_and_Sigma_i(m, v, base_params, sessions, choices, rewards, context, model_config, count=0):
    # Unbounded optimization for eta
    alpha_min, alpha_max = 1e-3, 1 - 1e-3
    eta_min = logit(alpha_min)
    eta_max = logit(alpha_max)
    result = minimize(
        fun=lambda eta: neg_log_posterior(eta, m, v, base_params, sessions, choices, rewards, context, model_config),
        x0=np.random.randn(),              # random initialization in R
        bounds=[(eta_min, eta_max)],                       # eta is unbounded
        method="L-BFGS-B",#'Powell',
        options=dict(maxiter=5000, ftol=1e-6)
    )
    #this finds the MAP estimate of h_i - participant specific estimates of eta (log(alpha))
    eta = result.x
    if count < 5:
        print(f"eta value that is passed to hessian (should be only one value): {eta}")
    eps = np.sqrt(np.finfo(float).eps)
    H = hessian_1d(
        lambda eta: neg_log_posterior(eta, m, v, base_params, sessions, choices, rewards, context, model_config),
        eta
    )
    var_theta = -1.0 / H
    
    return eta, var_theta

# EM updates
def update_group_prior(h, Sigma, min_variance=1e-5):
    """
    h:      (N,) or (N,D)     posterior means
    Sigma:  (N,) or (N,D,D)   posterior variances/covariances
    """
    h = np.asarray(h)
    Sigma = np.asarray(Sigma)

    if h.ndim == 1:
        # 1D case
        m = np.mean(h)
        v = np.mean(h**2 + Sigma) - m**2
        v = float(np.clip(v, 1e-4, 0.4))
    else:
        # D-dimensional case (your original)
        m = np.mean(h, axis=0)
        v = np.mean(h**2 + np.diagonal(Sigma, axis1=1, axis2=2), axis=0) - m**2
        v = np.clip(v, [1e-4, 0.4], None)

    return m, v

def update_group_prior1D(h, Sigma, min_variance=1e-5):
    """
    1D case only: h and Sigma are (N,) arrays of posterior means and variances in eta-space.
    """
    h = np.asarray(h).reshape(-1)
    Sigma = np.asarray(Sigma).reshape(-1)

    m = np.mean(h)
    v = np.mean(h**2 + Sigma) - m**2
    v = float(np.clip(v, 1e-4, 0.4))  # clamp to sensible range

    return m, v

# EM loop
def run_empirical_bayes(df, model_config, base_params, n_iter=20):
    m = 0.5  # Initial group prior means for alpha
    v = 0.2  # Initial group prior variances for alpha
    m_history = []
    v_history = []
    session_results = []
    for i in range(n_iter):
        eta_list = []
        var_list = []
        for session, group in df.groupby('session'):
            count = 0
            session_indices = group.index
            sessions = np.full_like(group['c'].values, session)
            choices = group['c'].values
            rewards = group['r'].values
            context = group['context'].values if 'context' in group.columns else None
            eta_i, var_eta_i = estimate_h_i_and_Sigma_i(m, v, base_params, sessions, choices, rewards, context, model_config, count)
            count += 1
            eta_list.append(eta_i) # remember that you're saving the unbounded parameter here
            var_list.append(var_eta_i)
            if i == n_iter:
                session_results.append({
                    'session': session,
                    'params': base_params,

                    })
        eta_vec = np.array(eta_list)
        var_vec = np.array(var_list)
        m, v = update_group_prior1D(eta_vec, var_vec)
        m_history.append(m)
        v_history.append(v)
    
    return m, v, eta_vec, var_vec, np.array(m_history), np.array(v_history)

def infer_params_for_test_EM(df_test, m, v, base_param, model_config):
    """
    For each session in df_test:
    - find eta_j^MAP given fixed prior N(m,v) and base_param
    - build full param vector (alpha_j inserted into base_param)
    Returns: dict session -> param_j
    """
    params_per_session = {}

    for session, group in df_test.groupby("session"):
        sessions = np.full_like(group["c"].values, session)
        choices  = group["c"].values
        rewards  = group["r"].values
        context  = group["context"].values if "context" in group.columns else None

        eta_j, var_eta_j = estimate_h_i_and_Sigma_i(
            m, v,
            base_param,
            sessions, choices, rewards, context,
            model_config
        )
        alpha_j = sigmoid(eta_j)
        param_j = set_alpha_in_param(base_param, alpha_j, model_config)

        params_per_session[session] = param_j

    return params_per_session


# def test_parameter_recovery(true_w1=1.8, true_w2=0.5, true_w3=0.5, n_trials=500):
#     # Simulate data with known parameters
#     agent = HybridAgent_opt_sim_probit(true_w1, true_w2, true_w3, n_trials)
#     actions, rewards = [], []
#     for t in range(n_trials):
#         prob0 = agent.get_probs(t)
#         action = np.random.choice([0, 1], p=[prob0, 1-prob0])
#         reward = np.random.normal(0, 1)  # Simple reward (mean=0)
#         agent.step(action, reward, t)
#         actions.append(action)
#         rewards.append(reward)
    
#     # Try to recover parameters
#     df_test = pd.DataFrame({"action": actions, "reward": rewards, "state": range(n_trials)})
#     result = minimize(
#         negative_log_likelihood,
#         x0=[1.0, 1.0, 1.0],
#         args=(df_test,),
#         bounds=[(0.1, 100), (0.1, 100), (0.1, 100)],
#         method="L-BFGS-B"
#     )
#     print(f"True: ({true_beta}, {true_gamma}) | Recovered: {result.x}")

# def estimate_shared_model(df_train_all):
#     """Fit one model to all participants' data."""
#     result = minimize(
#         negative_log_likelihood,
#         x0=[1.0, 1.0, 1.0],
#         args=(df_train_all,),
#         bounds=[(0.1, 100), (0.1, 100), (0.1, 100)],
#         method="L-BFGS-B",
#         options={
#         'maxiter': 1000,  # Increased iterations
#         'ftol': 1e-6,     # Tighter tolerance
#         'gtol': 1e-6,
#     }
#     )
#     return result.x

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


