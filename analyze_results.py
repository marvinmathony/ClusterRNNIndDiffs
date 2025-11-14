import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_functions as plf
import sim_Q_data as sim
import on_policy_sims as op
import os
from compare_models import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#latent = True
latent_nametag = "latentmodel" #loading

conditions = [True, False]
for latent in conditions:

    data_tag = "latentmodel" if latent else "vanilla" #saving

    #### Load Data ####
    # Always available / shared
    df_train = pd.read_csv("data/df_train.csv")
    p1_common_dict = dict(np.load("data/p1_common_dict.npz", allow_pickle=True))
    rewardsTrain = np.load("data/rewards_train.npy")

    # Depending on model
    if latent:
        model_eval_df = pd.read_csv(f"data/model_eval_df{latent_nametag}.csv")
        rnn_df = pd.read_csv(f"data/rnn_results{latent_nametag}.csv")
        pA_rnn = dict(np.load(f"data/pA_rnn_dict{latent_nametag}.npz", allow_pickle=True))
        training_dict = np.load(f"data/training_dict{latent_nametag}.npy", allow_pickle=True).item()
        latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}.pt")
    else:
        model_eval_df = pd.read_csv("data/model_eval_df.csv")
        rnn_df = pd.read_csv("data/rnn_results.csv")
        pA_rnn = dict(np.load("data/pA_rnn_dict.npz", allow_pickle=True))
        training_dict = np.load("data/training_dict.npy", allow_pickle=True).item()
        latent_tensor = torch.load("data/latents_tensor.pt")

    best_epoch = training_dict["best_epoch"]



    for key in pA_rnn:
        print(key)

    os.makedirs("plots", exist_ok=True)
    # Plot p1 for a single session
    selected_session = 0
    plt.figure(figsize=(10, 6))
    for model_name, p1_common in p1_common_dict.items():
        session_indices = df_train['session'] == selected_session
        trials = range(session_indices.sum())
        plt.plot(trials, p1_common[session_indices], label=model_name)

    plt.xlabel("Trial")
    plt.ylabel("p1 (Prob choose option 1)")
    plt.title(f"Comparison of p1 across models (Session {selected_session})")
    plt.legend()
    plt.savefig(f"plots/p1_comparison{data_tag}.png", dpi=150)
    plt.show()

    for model_name, p1_common in p1_common_dict.items():
        # Extract the p1 values for the selected session
        session_indices = df_train['session'] == selected_session
        trials = range(session_indices.sum())  # Number of trials in the selected session
        p1_common_selected = p1_common[session_indices]

        plt.plot(trials, p1_common_selected, label=model_name)

    plt.xlabel('Trial')
    plt.ylabel('Probability of choosing option 1 (p1)')
    plt.title(f'Comparison of p1_common across models (Session {selected_session})')
    plt.legend()
    plt.savefig(f"cog_models_p0_common{data_tag}.png", dpi=150)
    plt.show()
    print("saved p1 plot")


    # Compare model performance
    # change name of GRU to RNN and IDRNN
    # Find all indices where model == "GRU"
    gru_indices = model_eval_df.index[model_eval_df["model"] == "GRU"]

    # Sort to ensure order (top to bottom in file)
    gru_indices = gru_indices.sort_values()

    # Split into the first and last 200
    common_process_idx = gru_indices[-200:]  # second-to-last 200
    idrnn_idx = gru_indices[-400:-200]              # last 200

    # Rename them
    model_eval_df.loc[common_process_idx, "model"] = "common_process_RNN"
    model_eval_df.loc[idrnn_idx, "model"] = "IDRNN"
    print(f"model_eval_df last 400: {model_eval_df.tail(400)}")
    models = model_eval_df["model"].unique()
    fig = plf.compare_model_performance(
        model_eval_df, models, 
        file_id="model_comparison",
        save_dir="plots/",
        ylim=[0.5, 0.64],
        plot_individual_ML=False,
        fig_width=7,
        x_tick_fontsize=18
    )
    fig.savefig(f"plots/model_comparison{data_tag}.png")
    print("✅ Plots saved to plots/")

    
    p0_epoch = pA_rnn[str(best_epoch)] # look up best epoch - this has to be handled automatically in the future 
    idxSub = 0 
    df_sub = df_train[df_train['session'] == idxSub].copy() 
    pA = df_sub['p'].to_numpy()
    c_0 = df_sub['c'] 
    session_indices = df_train['session'] == idxSub 

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    p1_common_selected = p1_common_dict['FQ'][session_indices] 
    if latent:
        pRNN = [p0_epoch[idxSub, :]]
    else:
        pRNN = [p0_epoch[idxSub, :, 0]]
    plf.plot_prediction_RNN_RL([pA], rewardsTrain, c_0.copy().values.reshape(1, -1), 
                            axes[0], maxTrial=200, title=f"Iteration 1300, Subject {idxSub+1}", 
                            p_RL=[p1_common_selected], p_RNN = pRNN)#p_RNN = [p0_epoch[idxSub, :, 0]]) 
    idxSub = 9 
    df_sub = df_train[df_train['session'] == idxSub].copy() 
    pA = df_sub['p'].to_numpy()
    print(f"probabilities of choosing 0 for second participant: {pA}")
    print(f"probabilities of choosing 0 from RNN: {pRNN}")
    c_50 = df_sub['c'] 
    session_indices = df_train['session'] == idxSub 
    p1_common_selected = p1_common_dict['FQ'][session_indices] 
    plf.plot_prediction_RNN_RL([pA], rewardsTrain, c_50.copy().values.reshape(1, -1), 
                            axes[1], maxTrial=200, title=f"Iteration 1300, Subject {idxSub+1}", 
                            p_RL=[p1_common_selected], p_RNN = pRNN) #p_RNN = [p0_epoch[idxSub,:, 0]]) 
    # Adjust layout and save the figure 
    plt.tight_layout() 
    plt.savefig(f'plots/prediction_time_series_for_different_steps{data_tag}.png') 
    plt.show()

    plf.plot_latents(dim_reduction="mds", latent_tensor=latent_tensor, avg=True, n_components=2, name=data_tag)
    plf.plot_latents(dim_reduction="mds", latent_tensor=latent_tensor, avg=False, n_components=2, name=data_tag)

    #latent regression
    param_df = pd.read_csv('data/true_parameter_values.csv')
    modes = ["avg", "final", "all"]
    X_full = latent_tensor.detach().cpu().numpy()  # shape (n_participants, timesteps, hid_dim)
    y = param_df["alphaP_list"].values
    y_class = (y > 0.5).astype(int)
    results = {}
    for mode in modes:
        if mode == "avg":
            X = X_full.mean(axis=1)
        elif mode == "final":
            X = X_full[:,-1,:]
        elif mode == "all":
            X = X_full.reshape(X_full.shape[0], -1) # shape (N, T*hid_dim)
        else:
            raise ValueError(f"Unknown mode {mode}")
        clf = LogisticRegression()
        acc = cross_val_score(clf, X, y_class, cv=5, scoring='accuracy').mean()
        print(f"Mean cross-validated accuracy = {acc:.3f}")
        results[mode] = acc
        """r2_cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean()
        print(f"Mean cross-validated R² = {r2_cv:.3f}")
        results[mode] = r2_cv"""
    # Convert to DataFrame and save
    r2_df = pd.DataFrame([results])
    r2_df.to_csv(f"data/latent_regression_results_{data_tag}.csv", index=False)
    print(f"\nSaved results to data/latent_regression_results_{data_tag}.csv")

    ###############################
    #### quantify the geometry ####
    ###############################

    # X: (n_participants, latent_dim)
    # y_class: (n_participants,) with 0 and 1 labels
    if latent:

        X = X_full[:,-1,:] #X_full.mean(axis=1)
        name = "last_latent"
        plf.latent_euclidian_distance(X, y_class, data_tag, name)
        for mode in modes:
            best_gmm, bics, best_k, latents_used = plf.fit_gmm_bic(latent_tensor = latent_tensor, model_name="latentmodel", mode=mode)
    else:
        X_list = [X_full[:,-1,:], X_full.mean(axis=1)]
        names = ["last_latent", "mean_latent"]
        for X, name in zip(X_list, names):
            plf.latent_euclidian_distance(X, y_class, data_tag, name)
        for mode in modes:
            best_gmm, bics, best_k, latents_used = plf.fit_gmm_bic(latent_tensor=latent_tensor, model_name="vanillamodel", mode=mode)

# plot best number of gaussians over time
latent_tensor_latentmodel = torch.load(f"data/latents_tensor{latent_nametag}.pt")
name_latent = "latentmodel"
latent_tensor_vanillamodel = torch.load(f"data/latents_tensor.pt")
name_vanilla = "vanillamodel"

bics1, bics2, k1, k2 = plf.plot_bic_trajectories(latent_tensor_latentmodel, name_latent, latent_tensor_vanillamodel, name_vanilla)




"""#################################
########## OFF POLICY ###########
#################################

# off-policy state update
nTrialPre = 50
flg_use_train_data_for_pre = True 
flg_on_policy_check = True
xin_train = np.load("data/xinput_train.npy")
xin_train = torch.from_numpy(xin_train).float().to(device)
device = xin_train.device
nTrial_sim = 250

#load model
# Load dictionary
checkpoint = torch.load("checkpoints/ablated_rnn_best.pt", map_location=torch.device("cpu"))  # or "cuda"

# Recreate the model
model_config = checkpoint["model_config"]
model = AblatedRNN(**model_config)

# Load weights
model.load_state_dict(checkpoint["weights"])
model.eval()
print(f"is block structure true or false? : {model.block_structure}")
model = model.to(device)

#################################
########## ON POLICY ###########
#################################


if flg_on_policy_check: 
    if flg_use_train_data_for_pre:
        xin_pre = xin_train[:,:nTrialPre,:]
        #xin_pre = torch.from_numpy(xin_pre).float().to(device)
        
    else:
        nSession = 100

        rewardsPre = sim.gen_reward_seq(seed=1, T = nTrialPre, pHigh = 0.7, pLow = 0.3, interval = 50)

        seed_base = 11

        true_param_for_check = sim.generate_parameter_lists(true_model = 'FQ', ind_diff_type = 'full_range', 
                                        Delta_alpha=None, nSession=nSession)

         #Generate data for training
        c_pre, r_pre, _, _, _, df_pre, xin_pre, _, _, _, _ = sim.simulate_Qlearning(
            rewards=rewardsPre, seed=seed_base + 100, n_sessions=nSession, n_trials=nTrialPre, 
            alphaP_list=true_param_for_check['alphaP_list'],
            alphaN_list=true_param_for_check['alphaN_list'], 
            beta=true_param_for_check['beta_list'], alphaF_list=true_param_for_check['alphaF_list'], 
            phi_list=true_param_for_check['phi_list'], tau_list=true_param_for_check['tau_list']
        )
        xin_pre = torch.from_numpy(xin_pre).float().to(device)

    # Generating the reward sequence
    rewardsSim = sim.gen_reward_seq(seed=1, T = nTrial_sim, pHigh = 0.7, pLow = 0.3, interval = 50)

    target_epochs = [best_epoch]
    file_id = '500trainingsessions_test_250trials_2iters_model2000epochs_initialnotlearned_10hidden_50starters_manual_opt_nocontextchange_optim'

    fig, param_df = op.fit_rl_data(df_pre, model_configs={"FQ": {"asymmetric_alpha": False, "forgetting_type":"fixed", "choice_trace": False, "init_Q_free": True}},
                                   target_epochs=target_epochs)

    fig, param_df = op.simulate_on_policy_multiple_epochs4(
        training_dict, model, 
        xin_pre, rewardsSim, 
        target_epochs=target_epochs, 
        model_configs={"FQ": {"asymmetric_alpha": False, "forgetting_type":"fixed", "choice_trace": False, "init_Q_free": False}},#{"Q+A": {"asymmetric_alpha": True, "forgetting_type":"None", "choice_trace": False, "init_Q_free": True}},  
        n_iter=2, file_id=f"IDT_check_{file_id}", device=device
    )"""