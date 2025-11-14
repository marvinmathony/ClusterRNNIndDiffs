import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from tqdm import tqdm
import RL_fitting_functions as fit


def update_rnn_state_off_policy2(model_sim, xin):
    """
    PyTorch version: Get final hidden state for each session after off-policy simulation.
    Args:
        model_sim: trained AblatedRNN model.
        xin: Input tensor [B, T, D]
    Returns:
        session_states: List of tensors [1, hidden_dim] per session
    """
    # take care that model is initialized with block_structure = False
    model_sim.eval()
    #print(f"is model_sim block structure true or false? : {model_sim.block_structure}")
    with torch.no_grad():
        #print(f"xin shape: {xin.shape}")
        logits, final_hid, _ = model_sim(xin)  # final_hid: [B, hidden_dim]
        print(f"final hidden state: {final_hid}, {final_hid.shape}")
        session_states = final_hid#[1, N, 10] #[h.unsqueeze(0) for h in final_hid]  # make each [1, hidden_dim]

    return session_states


def simulate_on_policy(model_sim, model_name, rewardsSim, n_sessions=None, session_states=None, device="cpu"):
    """
    Perform on-policy simulation using a trained RNN model for multiple sessions.

    Parameters:
    - model_sim: Trained RNN model.
    - model_name: Name of the model.
    - rewardsSim: Reward sequence for simulation (shape: [2, nTrial], shared across sessions).
    - session_states: List of initial RNN states for each session.

    Returns:
    - DataFrame containing the simulation results.
    """
    print(f"session_states shape: {session_states.shape}")
    if n_sessions is None:
        n_sessions = session_states.shape[1]
        print(f"Number of sessions: {n_sessions}")

    # Identify the RNN layer and number of hidden units
    rnn_layer = next(
    (layer for layer in model_sim.modules() 
     if isinstance(layer, (nn.GRU, nn.LSTM, nn.RNN))), 
    None
    )

    if rnn_layer is None:
        raise ValueError("No RNN, LSTM, or GRU layer found in the model.")
    


    hidden_size = rnn_layer.hidden_size
    T = rewardsSim.shape[1]
    print(f"T: {T}")
    input_dim = model_sim.in_dim
    
    dataframes = []

    for idx_session in tqdm(range(n_sessions), desc=f"Simulating {model_name} with state updates"):
        # Initialize RNN state
        
        if session_states is not None:
            rnn_state = session_states[:,idx_session,:].unsqueeze(1)
            #rnn_state = rnn_state.unsqueeze(1)
            
            print(f"rnn state shape: {rnn_state.shape}")

        # Initialize variables for storing choices and rewards
        choices = np.zeros(T, dtype=int)
        rewards = np.zeros(T, dtype=int)
        probs = np.zeros(T, dtype=float)

        # **Preallocate input tensor (instead of recreating every trial)**
        x = torch.zeros(1, 1, input_dim)
        x = x.to(rnn_state.device)
        #rnn_state = final_hid[:, i, :].unsqueeze(1)

        for t in range(T):
            
            logits, rnn_state,_ = model_sim(x, rnn_state)
            p = F.softmax(logits, dim=-1)  # shape: (1, 1, 2)
            p_np = p.detach().cpu().numpy().squeeze()
            c = np.random.choice([0, 1], p=p_np)
            r = rewardsSim[c, t]
            #rnn_state = rnn_state[:, 0, :]

            choices[t] = c
            rewards[t] = r
            probs[t] = p_np[0]  # probability of choosing arm 0

            x.fill_(0.0)  # Reset values
            if input_dim == 4:
                x[0, 0, 0] = 1 if r == 1 else 0
                x[0, 0, 1] = 1 if r == 0 else 0
                x[0, 0, 2] = 1 if c == 0 else 0
                x[0, 0, 3] = 1 if c == 1 else 0
            #elif input_dim == 3:
            #    xin_values[0, 0, 0] = r_session[t]
            #    xin_values[0, 0, 1] = 1 if c_session[t] == 0 else 0
            #   xin_values[0, 0, 2] = 1 if c_session[t] == 1 else 0
            # not triggered for now


        # Store session results
        dataframes.append(pd.DataFrame({
            'model': model_name,
            'session': idx_session,
            'trial': np.arange(T),
            'c': choices,
            'r': rewards,
            'p_RNN': probs
        }))
    
    return pd.concat(dataframes, ignore_index=True)

def simulate_on_policy_multiple_epochs4(training_dict, sim_model_GRU, 
                                        xin_pre, rewardsSim,
                                        target_epochs=[1500, 2000], 
                                        model_configs=None, n_iter=2, 
                                        file_id="learning_rate_hist", device="cpu"):
    """
    Perform on-policy simulation for multiple epochs and analyze parameter distributions 
    by fitting RL models.

    Parameters:
    - ret: Dictionary containing training results (from train_rnn_with_checkpoints).
    - model_GRU: The trained GRU model.
    - sim_model_GRU: The simulation model.
    - xin_pre: Previous input data.
    - rewardsSim: Simulated rewards.
    - target_epochs: List of epochs to analyze (default: [1500, 2000]).
    - model_configs: Dictionary of RL models and their configurations.
    - n_iter: Number of iterations for RL fitting.
    - file_id: Identifier for the saved histogram file.

    Returns:
    - fig: The Matplotlib figure object containing histograms of parameters.
    - param_df: A DataFrame containing parameter values for all epochs and sessions.
    """

    if model_configs is None:
        model_configs = {
            "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": False, "init_Q_free": True}
        }

    # Define parameter limits based on type
    param_limits = {
        "alphaP": (0, 1),
        "alphaN": (0, 1),
        "alphaF": (0, 1),
        "beta": (0, 5),
        "phi": (-3, 3),
        "tau": (0, 1)
    }

    # Storage for estimated parameter values
    param_values_dict = {}
    param_records = []  # List to store DataFrame rows

    for target_epoch in target_epochs:
        print(f"\nProcessing epoch {target_epoch}...")

        # Load model weights for target epoch
        if target_epoch == training_dict["best_epoch"]:
            sim_model_GRU.load_state_dict(training_dict["weights"])
            print(f"Model weights have been set to epoch {target_epoch}.")
        else:
            print(f"Epoch {target_epoch} not available — only best_epoch={training_dict['best_epoch']} was saved.")
            continue

        # Generate session states (off-policy update)
        session_states = update_rnn_state_off_policy2(model_sim=sim_model_GRU, xin=xin_pre)
        print(f"session state 0: {session_states[:,0,:]}")
        print(f"session state 1: {session_states[:,1,:]}")
        print(f"session state 2: {session_states[:,2,:]}")



        # Perform on-policy simulation
        df_simulated = simulate_on_policy(
            model_sim=sim_model_GRU, 
            model_name="GRU",
            rewardsSim=rewardsSim,
            session_states=session_states
        )
        df_simulated = df_simulated.rename(columns={'c_sim': 'c', 'r_sim': 'r'})
        print(f"df simulated: {df_simulated.head()}")
        print(f"Fitting model on {len(df_simulated['session'].unique())} sessions with {len(df_simulated)} total trials.")
        df_simulated.to_csv('data/df_simulated.csv')

        # Initialize result storage
        params_dict = {}

        # Fit RL models
        for model_name, model_config in model_configs.items():
            print(f"Fitting {model_name} model...")

            df_result_tmp, session_results_df = fit.fit_qlearning_by_session_MAP2(
                df_simulated, model_config=model_config, n_iter=n_iter
            )

            # Store estimated parameters
            params_per_session_MAP = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
            params_dict[f"{model_name}_MAP"] = params_per_session_MAP

        # Identify the correct _MAP model dynamically
        map_model_name = next((k for k in params_dict.keys() if k.endswith("_MAP")), None)
        if map_model_name is None:
            print(f"No MAP model found for epoch {target_epoch}. Skipping...")
            continue
        print(f"fitted model name is {map_model_name}")

        # Extract parameter names from model_config
        config = fit.parse_model_config(model_config) if model_config else {}
        asymmetric_alpha = config.get("asymmetric_alpha", False)
        forgetting_type = config.get("forgetting_type", "none")
        choice_trace = config.get("choice_trace", False)
        init_Q_free = config.get("init_Q_free", False)

        # Dynamically determine the parameter names based on config
        param_names = []
        if init_Q_free:
            skip_params = 2  # Ignore first two parameters (Q-initial values)
        else:
            skip_params = 0

        param_names.append("alphaP")
        if asymmetric_alpha:
            param_names.append("alphaN")
        if forgetting_type == "free":
            param_names.append("alphaF")
        param_names.append("beta")
        if choice_trace:
            param_names.extend(["phi", "tau"])

        # Extract parameter values for histogram, ignoring Q-init if needed
        param_values = list(params_dict[map_model_name].values())
        param_values_dict[target_epoch] = {param_names[i]: [v[i + skip_params] for v in param_values] for i in range(len(param_names))}

        # Store parameter values in DataFrame format
        for session, values in params_dict[map_model_name].items():
            record = {"epoch": target_epoch, "session": session}
            record.update({param_names[i]: values[i + skip_params] for i in range(len(param_names))})
            param_records.append(record)

    # Convert collected parameter data into a DataFrame
    param_df = pd.DataFrame(param_records)

    # Determine subplot arrangement based on number of parameters
    num_params = len(param_names)  # Number of parameters
    num_epochs = len(target_epochs)
    fig, axes = plt.subplots(num_epochs, num_params, figsize=(4*num_params, 4*num_epochs), sharey=True)

    if num_epochs == 1:
        axes = [axes]  # Ensure iterable when only one row
    if num_params == 1:
        axes = [[ax] for ax in axes]  # Ensure iterable when only one column

    # Plot histograms for all parameters and target epochs
    for i, target_epoch in enumerate(target_epochs):
        for j, param_name in enumerate(param_names):
            ax = axes[i][j]
            param_data = param_values_dict[target_epoch][param_name]

            # Set bin ranges adaptively (20 bins between predefined range)
            bins = np.linspace(param_limits[param_name][0], param_limits[param_name][1], 21)

            ax.hist(param_data, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel(param_name)  # Use actual parameter name
            ax.set_ylabel("Frequency")
            ax.set_title(f"Epoch {target_epoch} - {param_name}")

            # Set x-axis limits based on predefined ranges
            ax.set_xlim(param_limits[param_name])

    plt.tight_layout()

    # Save figure as EPS
    eps_filename = f"plots/{file_id}.png"
    fig.savefig(eps_filename, format='png', dpi=300)
    print(f"Histogram saved as {eps_filename}")

    return fig, param_df


def fit_rl_data(df_simulated, model_configs, target_epochs=[1500, 2000], file_id = "fitting_ground_truth", n_iter=2):
    # Define parameter limits based on type
    param_limits = {
        "alphaP": (0, 1),
        "alphaN": (0, 1),
        "alphaF": (0, 1),
        "beta": (0, 5),
        "phi": (-3, 3),
        "tau": (0, 1)
    }
    # Fit RL models
    params_dict = {}
    # Storage for estimated parameter values
    param_values_dict = {}
    param_records = []  # List to store DataFrame rows
    for target_epoch in target_epochs:

        for model_name, model_config in model_configs.items():
            print(f"Fitting {model_name} model...")

            df_result_tmp, session_results_df = fit.fit_qlearning_by_session_MAP2(
                df_simulated, model_config=model_config, n_iter=n_iter
            )

            # Store estimated parameters
            params_per_session_MAP = {row['session']: row['params'] for _, row in session_results_df.iterrows()}
            params_dict[f"{model_name}_MAP"] = params_per_session_MAP

        # Identify the correct _MAP model dynamically
        map_model_name = next((k for k in params_dict.keys() if k.endswith("_MAP")), None)
        if map_model_name is None:
            print(f"No MAP model found for epoch {target_epoch}. Skipping...")


        # Extract parameter names from model_config
        config = fit.parse_model_config(model_config) if model_config else {}
        asymmetric_alpha = config.get("asymmetric_alpha", False)
        forgetting_type = config.get("forgetting_type", "none")
        choice_trace = config.get("choice_trace", False)
        init_Q_free = config.get("init_Q_free", False)

        # Dynamically determine the parameter names based on config
        param_names = []
        if init_Q_free:
            skip_params = 2  # Ignore first two parameters (Q-initial values)
        else:
            skip_params = 0

        param_names.append("alphaP")
        if asymmetric_alpha:
            param_names.append("alphaN")
        if forgetting_type == "free":
            param_names.append("alphaF")
        param_names.append("beta")
        if choice_trace:
            param_names.extend(["phi", "tau"])

        # Extract parameter values for histogram, ignoring Q-init if needed
        param_values = list(params_dict[map_model_name].values())
        param_values_dict[target_epoch] = {param_names[i]: [v[i + skip_params] for v in param_values] for i in range(len(param_names))}

        # Store parameter values in DataFrame format
        for session, values in params_dict[map_model_name].items():
            record = {"epoch": target_epoch, "session": session}
            record.update({param_names[i]: values[i + skip_params] for i in range(len(param_names))})
            param_records.append(record)

        # Convert collected parameter data into a DataFrame
        param_df = pd.DataFrame(param_records)

        # Determine subplot arrangement based on number of parameters
        num_params = len(param_names)  # Number of parameters
        num_epochs = len(target_epochs)
        fig, axes = plt.subplots(num_epochs, num_params, figsize=(4*num_params, 4*num_epochs), sharey=True)

        if num_epochs == 1:
            axes = [axes]  # Ensure iterable when only one row
        if num_params == 1:
            axes = [[ax] for ax in axes]  # Ensure iterable when only one column

        # Plot histograms for all parameters and target epochs
        for i, target_epoch in enumerate(target_epochs):
            for j, param_name in enumerate(param_names):
                ax = axes[i][j]
                param_data = param_values_dict[target_epoch][param_name]

                # Set bin ranges adaptively (20 bins between predefined range)
                bins = np.linspace(param_limits[param_name][0], param_limits[param_name][1], 21)

                ax.hist(param_data, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel(param_name)  # Use actual parameter name
                ax.set_ylabel("Frequency")
                ax.set_title(f"Epoch {target_epoch} - {param_name}")

                # Set x-axis limits based on predefined ranges
                ax.set_xlim(param_limits[param_name])

        plt.tight_layout()

        # Save figure as EPS
        eps_filename = f"plots/{file_id}.png"
        fig.savefig(eps_filename, format='png', dpi=300)
        print(f"Histogram saved as {eps_filename}")
    return fig, param_df