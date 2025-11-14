import sim_Q_data as sim
import RL_fittingfunctions2 as fit
import plot_functions as plf
import random
import numpy as np
import tensorflow as tf
import torch
from compare_models import *
import copy

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 42
random.seed(seed_value)           
np.random.seed(seed_value)        
tf.random.set_seed(seed_value)    

# Parameters and Variables
sim_nametag = "Katahira_setup"
latent_nametag = "latentmodel"
latent = False
palminteri = False
sloutsky = True

n_fit_iter = 5 # iteration for RL fitting (from random initialization)
flg_on_policy_check = 1 # True # True: do on-policy check
nTrial_sim = 5000 # nTrial for on-policy check
file_id = 'scenario1_FQ_discrete'  

# load data
if palminteri:
    df_train = pd.read_csv("data/train_df_palminteri.csv")
    df_test = pd.read_csv("data/test_df_palminteri.csv")
    df_val = pd.read_csv("data/val_df_palminteri.csv")
    df_train = df_train.rename(columns={'sub_across_groups': 'session'})
    df_test = df_test.rename(columns={'sub_across_groups': 'session'})
    df_val = df_val.rename(columns={'sub_across_groups': 'session'})
    # ---- load training arrays ----
    xin_train = np.load("data/train_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_train = np.load("data/train_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_train = np.load("data/c_train_palminteri.npy", allow_pickle=True)

    # ---- load validation arrays ----
    xin_val = np.load("data/val_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_val = np.load("data/val_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_val = np.load("data/c_val_palminteri.npy", allow_pickle=True)

    # ---- load test arrays ----
    xin_test = np.load("data/test_xin_palminteri.npy", allow_pickle=True)
    choice_one_hot_test = np.load("data/test_choice_one_hot_palminteri.npy", allow_pickle=True)
    c_test = np.load("data/c_test_palminteri.npy", allow_pickle=True)
    pA = np.load("data/pA_train.npy") #not relevant here, but just so that we don't have to adapt script too much
    pA_test = np.load("data/pA_test.npy") #not relevant here, but just so that we don't have to adapt script too much
    c_train = torch.from_numpy(c_train).float().to(device)
    xin_val = torch.from_numpy(xin_val).float().to(device)
    choice_one_hot_val = torch.from_numpy(choice_one_hot_val).float().to(device)

elif sloutsky:
    df_train = pd.read_csv("data/train_df_sloutsky.csv")
    df_test = pd.read_csv("data/test_df_sloutsky.csv")
    df_val = pd.read_csv("data/val_df_sloutsky.csv")
    df_train = df_train.rename(columns={'subid': 'session'})
    df_test = df_test.rename(columns={'subid': 'session'})
    df_val = df_val.rename(columns={'subid': 'session'})
    # ---- load training arrays ----
    xin_train = np.load("data/train_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_train = np.load("data/train_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_train = np.load("data/c_train_sloutsky.npy", allow_pickle=True)

    # ---- load validation arrays ----
    xin_val = np.load("data/val_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_val = np.load("data/val_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_val = np.load("data/c_val_sloutsky.npy", allow_pickle=True)

    # ---- load test arrays ----
    xin_test = np.load("data/test_xin_sloutsky.npy", allow_pickle=True)
    choice_one_hot_test = np.load("data/test_choice_one_hot_sloutsky.npy", allow_pickle=True)
    c_test = np.load("data/c_test_sloutsky.npy", allow_pickle=True)
    pA = np.load("data/pA_train.npy") #not relevant here, but just so that we don't have to adapt script too much
    pA_test = np.load("data/pA_test.npy") #not relevant here, but just so that we don't have to adapt script too much
    c_train = torch.from_numpy(c_train).float().to(device)
    c_val = torch.from_numpy(c_val).float().to(device)
    xin_val = torch.from_numpy(xin_val).float().to(device)
    choice_one_hot_val = torch.from_numpy(choice_one_hot_val).float().to(device)


else:
    # --- Load CSVs ---
    df_train = pd.read_csv("data/df_train.csv")
    df_test = pd.read_csv("data/df_test.csv")
    session_ll_df_test = pd.read_csv("data/session_ll_df_test.csv")

    # --- Load NumPy arrays ---
    xin_train = np.load("data/xin_train.npy")
    xin_test = np.load("data/xin_test.npy")
    choice_one_hot_train = np.load("data/choice_one_hot_train.npy")
    choice_one_hot_test = np.load("data/choice_one_hot_test.npy")
    c_test = np.load("data/c_test.npy")
    pA = np.load("data/pA_train.npy")
    pA_test = np.load("data/pA_test.npy")



# Convert to tensors
xin_train = torch.from_numpy(xin_train).float().to(device)
choice_one_hot_train = torch.from_numpy(choice_one_hot_train).float().to(device)
pA = torch.from_numpy(pA).float().to(device)

#print(f"xin_train shape: {xin_train.shape}")

xin_test = torch.from_numpy(xin_test).float().to(device)
c_test = torch.from_numpy(c_test).float().to(device)



B, T, in_dim = xin_train.shape
z_dim = 3
A = 4

# --- Train RNN ---
hidden = 10
epochs = 5000
if latent:
    
    ids = torch.arange(B)
    encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
    model = LatentRNNz(encoder=encoder, z_dim=z_dim, in_dim=in_dim, hid=hidden, A=A, block_structure=False).to(device)
    if palminteri or sloutsky:
        B_val,_,_ = xin_val.shape
        val_ids = torch.arange(B_val)
        model, train_loss, val_loss, pA_dict, training_dict = train_latentrnn_noblocks_palminteri(model=model, ids_train=ids, X_train=xin_train,
        y_onehot=c_train, ids_val=val_ids, X_val=xin_val, y_val_onehot=c_val, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
    else:

        model, train_loss, kl_vals, pA_dict, training_dict = train_latentrnn_noblocks(model=model, ids_train=ids, X_train=xin_train,
        y_onehot=choice_one_hot_train, p_target=pA, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
    z_lookup = training_dict["z"]
    
    #second step
    encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
    frozen_decoder = copy.deepcopy(model.decoder)
    for p in frozen_decoder.parameters():
        p.requires_grad = False
    latent_secondstep = LatentRNN_secondstep(encoder=encoder_IDRNN, z_dim=z_dim,in_dim=in_dim, hid=hidden, A=4,
    decoder=frozen_decoder)
    latent_secondstep.name = "GRU" # "LatentDistillation"
    latent_secondstep.to(device)
    xenc_train = xin_train.unsqueeze(1)                                # (B, 1, T, 4)
    y_train    = torch.argmax(choice_one_hot_train, dim=-1).unsqueeze(1)  # (B, 1, T)
    lookup_z   = z_lookup.to(device)
    
    if palminteri or sloutsky:
        z_val_lookup = training_dict["z_val"]
        lookup_z_val = z_val_lookup.to(device)
        xenc_val = xin_val.unsqueeze(1)
        y_val = torch.argmax(choice_one_hot_val, dim=-1).unsqueeze(1)
        model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN_palminteri(model=latent_secondstep, xenc=xenc_train,
        blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_val, y_val=y_val, z_val_lookup=lookup_z_val, epochs=5000, patience=600, lr=1e-3)
    else:
        xenc_val = xenc_train
        y_val    = y_train
        model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN(model=latent_secondstep, xenc=xenc_train,
        blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_val, y_val=y_val, p_target= pA, epochs=1300, patience=600, lr=1e-3)

else:
    vanilla_nametag = "vanilla"

    model = AblatedRNN(in_dim=in_dim, hid=hidden, A=A, block_structure=False).to(device)
    if palminteri or sloutsky:
        y_train    = torch.argmax(choice_one_hot_train, dim=-1)
        y_val = torch.argmax(choice_one_hot_val, dim=-1)
        model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks_palminteri(
        model, xin_train, y_train, xin_val, y_val, epochs = epochs, lr = 1e-3
        )
    else:

        model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks(
        model, xin_train, choice_one_hot_train, p_target=pA, epochs = epochs, lr = 1e-3
        )
# save model
# Save model configuration for reloading
model_config = {"in_dim": in_dim, "hid": hidden, "block_structure": False}

# Save the relevant parts

if sloutsky or palminteri:
    save_dict = {
    "weights": training_dict["weights"],
    "best_epoch": training_dict["best_epoch"],
    "model_config": model_config
}
else:
    save_dict = {
        "weights": training_dict["weights"],
        "best_epoch": training_dict["best_epoch"],
        "best_kl": training_dict["best_kl"],
        "model_config": model_config
    }

pA_rnn_dict = {k: v.detach().cpu().numpy() for k, v in pA_rnn_dict.items()}



# --- Compute RNN likelihoods ---
def compute_rnn_likelihoods_torch(model, test_xin, choice_test, train_xin, choice_train=None, latent = latent, N=500, id=True):
    rnn_results = []
    B, T = choice_test.shape
    with torch.no_grad():
        if latent:
            if test_xin.ndim == 3:
                test_xin = test_xin.unsqueeze(1)
                choice_test = choice_test.unsqueeze(1)
                train_xin = train_xin.unsqueeze(1)
                if choice_train is not None:
                    choice_train = choice_train.unsqueeze(1)
                else:
                    latent_tensor_train = None
            mean_ll, ll_per_participant, latent_tensor = test_latentrnn_secondstep_causal_posterior_weighting(
                model=model, blocks=test_xin, y=choice_test, N=N, id_case=id
            )
            mean_ll_train, ll_per_participant_train, latent_tensor_train = test_latentrnn_secondstep_causal_posterior_weighting(
                model=model, blocks=train_xin, y=choice_train, N=N, id_case=True
            )

            #probably do linear probe here - or in the test function
            print(f"ll_per_participant: {ll_per_participant}")
            norm_ll = (ll_per_participant / T).exp()  # Normalize and convert to probabilities
            print(f"normalized ll_per_participant: {norm_ll}")
            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": norm_ll.cpu().numpy(),
                "model": "GRU" #"LatentRNN_causalIW" - change code to implement other names as well
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train
        else:
            logits_test, final_hidden, latent_tensor = model(test_xin)
            #also access train tensor
            _, _, latent_tensor_train = model(xin_train)

            probs_test = F.softmax(logits_test, dim=-1)

            choices = choice_test.reshape(-1)
            probs_flat = probs_test.reshape(-1, probs_test.size(-1))
            chosen_probs = probs_flat[torch.arange(B*T), choices.long()].view(B, T)

            log_likelihoods = chosen_probs.clamp(min=1e-8).log()
            log_ll_per_session = log_likelihoods.sum(dim=1)
            norm_ll = (log_ll_per_session / T).exp()

            rnn_results.append(pd.DataFrame({
                "session": np.arange(B),
                "normalized_likelihood": norm_ll.cpu().numpy(),
                "model": "GRU"
            }))
            return pd.concat(rnn_results, ignore_index=True), latent_tensor, latent_tensor_train

rnn_df, latent_tensor, latent_tensor_train = compute_rnn_likelihoods_torch(model, xin_test, c_test, xin_train, choice_train=c_train, latent=latent, id=True)
#rnn_no_id, _, _ = compute_rnn_likelihoods_torch(model, xin_test, c_test, xin_train, choice_train=None, latent=latent, id=False)

# --- Fit Cognitive Models ---
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}
if not sloutsky:

    model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict = fit.fit_all_models(
        model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
    )

    print(f"model_eval_Q_df: {model_eval_Q_df}")
    print(f"session_ll_df_test: {session_ll_df_test}")
    print(f"rnn_df: {rnn_df}")
    print(f"rnn_df: {rnn_no_id}")
    ### run ID and noID and append both here. This can then be used for plotting
    model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test, rnn_df, rnn_no_id], ignore_index=True)


    # Save p1 values (for later plotting)
    np.savez("data/p1_common_dict.npz", **p1_common_dict)

if latent:
    rnn_df.to_csv(f"data/rnn_results{latent_nametag}.csv", index=False)
    if palminteri:
        torch.save(latent_tensor, f"data/latents_tensor{latent_nametag}_palminteri.pt")
        torch.save(latent_tensor_train, f"data/latents_tensor{latent_nametag}_palminteri_traindata.pt")
        print(f"latent tensor with dim {latent_tensor.shape} saved under data/latents_tensor_palminteri.pt")
        model_eval_df.to_csv(f"data/model_eval_df{latent_nametag}.csv", index=False)
    elif sloutsky:
        torch.save(latent_tensor, f"data/latents_tensor{latent_nametag}_sloutsky.pt")
        torch.save(latent_tensor_train, f"data/latents_tensor{latent_nametag}_sloutsky_traindata.pt")
        print(f"latent tensor with dim {latent_tensor.shape} saved under data/latents_tensor_sloutsky.pt")
    else:
        torch.save(latent_tensor, f"data/latents_tensor{latent_nametag}.pt")
        torch.save(latent_tensor_train, f"data/latents_tensor{latent_nametag}_sloutsky_traindata.pt")
        model_eval_df.to_csv(f"data/model_eval_df{latent_nametag}.csv", index=False)

    torch.save(save_dict, f"checkpoints/ablated_rnn_best{latent_nametag}.pt")
    torch.save(model.state_dict(), f"checkpoints/ablated_rnn{latent_nametag}.pt")
    np.savez(f"data/pA_rnn_dict{latent_nametag}.npz", **pA_rnn_dict)
    np.save(f"data/training_dict{latent_nametag}.npy", training_dict)
else:
    rnn_df.to_csv("data/rnn_results.csv", index=False)
    if palminteri:

        torch.save(latent_tensor, "data/latents_tensor_palminteri.pt")
        model_eval_df.to_csv("data/model_eval_df.csv", index=False)
    elif sloutsky:
        torch.save(latent_tensor, f"data/latents_tensor{vanilla_nametag}_sloutsky.pt")
        torch.save(latent_tensor_train, f"data/latents_tensor{vanilla_nametag}_sloutsky_traindata.pt")
    else:
        torch.save(latent_tensor, "data/latents_tensor.pt")
        model_eval_df.to_csv("data/model_eval_df.csv", index=False)

        
    torch.save(save_dict, "checkpoints/ablated_rnn_best.pt")
    torch.save(model.state_dict(), "checkpoints/ablated_rnn.pt")
    np.savez("data/pA_rnn_dict.npz", **pA_rnn_dict)
    np.save("data/training_dict.npy", training_dict)


print("✅ Training and fitting done. Results saved to data/ and checkpoints/")

"""# Generate reward sequences for training and testing
rewardsTrain = sim.gen_reward_seq(seed=seed_base, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
rewardsTest = sim.gen_reward_seq(seed=seed_base + 1, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)

# true_param = sim.generate_parameter_lists(true_model = 'FQ', ind_diff_type = 'discrete_alpha_only', 
#                                       Delta_alpha=0.8, nSession=nSession)
true_param = sim.generate_parameter_lists(true_model = 'Q+A', ind_diff_type = 'None', 
                                      Delta_alpha=None, nSession=nSession)
# ind_diff_type (str): Type of individual difference ('continuous_all', 'continuous_alpha_only', 'None', 'discrete_alpha_only', 'full_range'

# Generate data for training
c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, _, _, _ = sim.simulate_Qlearning(
    rewards=rewardsTrain, seed=seed_base + 2, n_sessions=nSession, n_trials=nTrial, 
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'], 
    beta=true_param['beta_list'], alphaF_list=true_param['alphaF_list'], 
    phi_list=true_param['phi_list'], tau_list=true_param['tau_list']
)

c_test, r_test, pA_test, _, _, df_test, xin_test, choice_one_hot_test, normalized_LL_test, session_ll_df_test, _ = sim.simulate_Qlearning(
    rewards=rewardsTest, seed=seed_base + 10, n_sessions=nSession, n_trials=nTrial, 
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'], 
    beta=true_param['beta_list'], alphaF_list=true_param['alphaF_list'], 
    phi_list=true_param['phi_list'], tau_list=true_param['tau_list']
)

print(f"df_train: {df_train.head(50)}")
print(f"xin_train: {xin_train}")
print(f"xin shape: {xin_train.shape}")
print(f"onehot encoded choices: {choice_one_hot_train}")
print(f"onehot encoded choices shape: {choice_one_hot_train.shape}")
xin_train = torch.from_numpy(xin_train).to(dtype=torch.float32)
choice_one_hot_train = torch.from_numpy(choice_one_hot_train).to(dtype=torch.float32)
pA = torch.from_numpy(pA).to(dtype=torch.float32)

xin_train = xin_train.to(device)
choice_one_hot_train = choice_one_hot_train.to(device)
pA = pA.to(device)


xin_test = torch.from_numpy(xin_test).to(dtype=torch.float32)
c_test = torch.from_numpy(c_test).to(dtype=torch.float32)

xin_test = xin_test.to(device)
c_test = c_test.to(device)


#xin shape: (100, 200, 4)

#### define model ####
#hyperparams
hidden = 10

m_abl = AblatedRNN(in_dim=4, hid=hidden, block_structure = False).to(device)

m_abl, train_loss, val_loss, kl_loss = train_ablated_noblocks(m_abl, xin_train, choice_one_hot_train, p_target=pA)

# plot loss
figure = sim.plot_kl_and_train_loss(train_loss, kl_loss, sim_nametag)


def compute_rnn_likelihoods_torch(model, test_xin, c_test, train_xin, test_mask=None):
    #later it might be useful to reintroduce the initial states argument when using my setup
    rnn_results = []

    # there is going to be a 3D case with changing context, but for now we don't bother
    B, T = c_test.shape  # Assume 2D case: (sessions, trials) 
    device = test_xin.device


    with torch.no_grad():
        # === Test Predictions ===
        logits_test, _, _ = model(test_xin)
        probs_test = F.softmax(logits_test, dim=-1)  # [B, T, A]
        p0_test = probs_test[:,:,0] #should be shape [B,T]

        # === Train Predictions ===
        logits_train, _, _ = model(train_xin)
        probs_train = F.softmax(logits_train, dim=-1)
        p0_train = probs_train[:,:,0] #should be shape [B,T]

        # === Log-likelihoods ===
        choices = c_test.view(-1)  # shape [B*T]
        probs_flat = probs_test.view(-1, probs_test.size(-1))  # [B*T, A]
        chosen_probs = probs_flat[torch.arange(B*T), choices.long()].view(B, T)

        log_likelihoods = chosen_probs.clamp(min=1e-8).log()  # log(p(a_t))

        if test_mask is not None:
            log_likelihoods = log_likelihoods * test_mask  # mask invalid trials
            valid_trials = test_mask.sum(dim=1)
        else:
            valid_trials = torch.tensor([T] * B, dtype=torch.float32, device=device)

        log_ll_per_session = log_likelihoods.sum(dim=1)
        norm_ll = (log_ll_per_session / valid_trials).exp()  # normalized LL

        # Add to dataframe
        rnn_results.append(pd.DataFrame({
            'session': np.arange(B),
            'normalized_likelihood': norm_ll.detach().cpu().numpy(),
            'model': "RNN"
        }))

    rnn_df = pd.concat(rnn_results, ignore_index=True)
    return rnn_df, p0_test, p0_train

rnn_df, p0_test, p0_train = compute_rnn_likelihoods_torch(
    m_abl, xin_test, c_test, xin_train
)

print(f"RNN df: {rnn_df.head(50)}")

# Define model configurations
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    #"Q+A": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": False},
    #"Q+C": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": True},
    #"Q+CA": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": True},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}

# Model fitting
model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict = fit.fit_all_models(
    model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
)

# Append to the model_eval_df DataFrame
model_eval_df = pd.concat([model_eval_Q_df, rnn_df], ignore_index=True)

model_eval_df = pd.concat([model_eval_df, session_ll_df_test], ignore_index=True)

print(f"model_eval_df: {model_eval_df.head(50)}")

# Select the session to plot (e.g., session 1)
selected_session = 0

# Plot p1_common for the selected session across models
plt.figure(figsize=(10, 6))

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
plt.savefig("cog_models_p0_common.png", dpi=150)
plt.show()
print("saved p1 plot")


# Combine model evaluation results
model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test], ignore_index=True)    
model_eval_df = pd.concat([model_eval_df, rnn_df], ignore_index=True)
model_eval_df.to_csv('data/model_eval_df.csv')

# Get unique models
models = model_eval_df['model'].unique()

fig = plf.compare_model_performance(
        model_eval_df, models, 
        file_id=f'model_comparison', 
        save_dir='plots/', ylim=[0.5, 0.64], plot_individual_ML=False, 
        fig_width=7, x_tick_fontsize=18
    )
# Add title with epoch information
fig.suptitle(f'Model Comparison', fontsize=16)
fig.savefig(f'plots/model_comparison_{sim_nametag}.png')
print(f"Figure comparing models (RNN and cog) saved.")"""

