import sim_Q_data as sim
import os
import pandas as pd
import RL_fittingfunctions2 as fit
import plot_functions as plf
import random
import numpy as np
import tensorflow as tf
import torch
#from compare_models import *
import copy
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modelsandtraining import *
import json


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 42
random.seed(seed_value)           
np.random.seed(seed_value)        
tf.random.set_seed(seed_value)    

#### Parameters and Variables ####
sim_nametag = "Katahira_setup"
latent_nametag = "latentmodel"
latent = True
palminteri = False
sloutsky = False
model_fitting = False
animation = False

n_fit_iter = 5 # iteration for RL fitting (from random initialization)
flg_on_policy_check = 1 # True # True: do on-policy check
nTrial_sim = 5000 # nTrial for on-policy check
file_id = 'scenario1_FQ_discrete'  

#### LOAD DATA ####
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
    c_train = np.load("data/c_train.npy")
    c_train = torch.from_numpy(c_train).float().to(device)



# Convert to tensors
xin_train = torch.from_numpy(xin_train).float().to(device)
choice_one_hot_train = torch.from_numpy(choice_one_hot_train).float().to(device)
choice_one_hot_test = torch.from_numpy(choice_one_hot_test).float().to(device)
pA = torch.from_numpy(pA).float().to(device)
pA_test = torch.from_numpy(pA_test).float().to(device)
xin_test = torch.from_numpy(xin_test).float().to(device)
c_test = torch.from_numpy(c_test).float().to(device)
B, T, in_dim = xin_train.shape
B_test, T_test, in_dim_test = xin_test.shape

# HYPERPARAMS
z_dim = 3
A = 2
hidden = 10
epochs = 5000


#### TRAINING #####

if latent:
    
    ids = torch.arange(B)
    idstest = torch.arange(B_test)
    encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
    decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden)
    model = LatentRNNz(encoder=encoder, decoder = decoder, hid=hidden, z_dim=z_dim, in_dim=in_dim, A=A, block_structure=False).to(device)
    if palminteri or sloutsky:
        B_val,_,_ = xin_val.shape
        val_ids = torch.arange(B_val)
        model, train_loss, val_loss, pA_dict, training_dict = train_latentrnn_noblocks_palminteri(model=model, ids_train=ids, X_train=xin_train,
        y_onehot=c_train, ids_val=val_ids, X_val=xin_val, y_val_onehot=c_val, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
    else:
        model, train_loss, kl_vals, pA_dict, training_dict = train_latentrnn_noblocks(model=model, ids_train=ids, ids_test=idstest, X_train=xin_train,
        y_onehot=choice_one_hot_train, X_test=xin_test, y_test_onehot=choice_one_hot_test, p_target=pA, p_test_target=pA_test, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=device)
    z_lookup = training_dict["z"]
    
    #second step
    encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=10)
    frozen_decoder = copy.deepcopy(model.decoder)
    for p in frozen_decoder.parameters():
        p.requires_grad = False
    latent_secondstep = LatentRNN_secondstep(encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,in_dim=in_dim, A=A,
    decoder=frozen_decoder)
    latent_secondstep.name = "GRU" # "LatentDistillation"
    latent_secondstep.to(device)
    xenc_train = xin_train.unsqueeze(1)                                # (B, 1, T, 4)
    y_train    = torch.argmax(choice_one_hot_train, dim=-1).unsqueeze(1)  # (B, 1, T)

    lookup_z   = z_lookup.to(device)
    #y_test = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)

    if palminteri or sloutsky:
        z_val_lookup = training_dict["z_val"]
        lookup_z_val = z_val_lookup.to(device)
        xenc_val = xin_val.unsqueeze(1)
        y_val = torch.argmax(choice_one_hot_val, dim=-1).unsqueeze(1)
        model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN_palminteri(model=latent_secondstep, xenc=xenc_train,
        blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_val, y_val=y_val, z_val_lookup=lookup_z_val, epochs=5000, patience=600, lr=1e-3)
    else:
        #think about whether I want to implement test data here too
        xenc_test = xin_test.unsqueeze(1)
        y_test    = torch.argmax(choice_one_hot_test, dim=-1).unsqueeze(1)
        model, best_mu, best_lv, train_elbos, val_elbos, training_dict, pA_rnn_dict = train_latentrnn_IDRNN(model=latent_secondstep, xenc=xenc_train,
        blocks=xenc_train, y=y_train, lookup_z=lookup_z, xenc_val=xenc_test, y_val=y_test, p_target= pA, device=device, epochs=epochs, patience=epochs, lr=1e-3)

else:
    vanilla_nametag = "vanilla"
    model = AblatedRNN(hid=hidden, in_dim=in_dim, A=A, block_structure=False).to(device)
    if palminteri or sloutsky:
        y_train    = torch.argmax(choice_one_hot_train, dim=-1)
        y_val = torch.argmax(choice_one_hot_val, dim=-1)
        model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks_palminteri(
        model, xin_train, y_train, xin_val, y_val, epochs = epochs, lr = 1e-3
        )
    else:
        print(f"xin test shape: {xin_test.shape}")
        print(f"choice one hot test shape: {choice_one_hot_test.shape}")
        model, train_loss, val_loss, kl_loss, pA_rnn_dict, training_dict = train_ablated_noblocks(
        model, xin_train, choice_one_hot_train, xin_test, choice_one_hot_test, device=device, p_target=pA, p_test=pA_test, epochs = epochs, lr = 1e-3
        )

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

#### LIKELIHOOD FIT ####

if animation:
    checkpoint_files = sorted(glob.glob("checkpoints/*.pt"))

    latent_list = []
    for ckpt in checkpoint_files:
        print("Loading:", ckpt)
        model.load_state_dict(torch.load(ckpt))
        _, latent_tensor, _, normalized_ll_test_IDRNN = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train, latent=latent, choice_train=c_train, id=True)
        latents = latent_tensor[:, -1, :].detach().cpu().numpy()
        latent_list.append(latents)
    all_latents = np.concatenate(latent_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_latents)
    latent_2d_over_epochs = [pca.transform(lat) for lat in latent_list]

    colors = [0 if i < 100 else 1 for i in range(200)]
    group_colors = colors
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        xy = latent_2d_over_epochs[frame]
        ax.scatter(xy[:, 0], xy[:, 1], c=group_colors, cmap="coolwarm")
        ax.set_title(f"Latent space — Epoch {frame*100}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

    

    anim = FuncAnimation(fig, update, frames=len(latent_2d_over_epochs), interval=400)

    anim.save("plots/latent_evolution_04diff_ID10_common02.gif", writer="pillow")
    print("Saved latent_evolution.gif")


rnn_df, latent_tensor, latent_tensor_train, geometric_mean_prob_rnn = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train, latent=latent, choice_train=c_train, id=True)
rnn_no_id, _, _, geometric_mean_prob_common_process  = compute_rnn_likelihoods_torch(test_latentrnn_secondstep_causal_posterior_weighting, model, xin_test, c_test, xin_train, latent=latent, choice_train=c_train, id=False)

if latent:
    print(f"normalized_ll IDRNN: {geometric_mean_prob_rnn}")
    print(f"normalized_ll common process: {geometric_mean_prob_common_process}")

    df = pd.DataFrame([{
        "model": "IDRNN",
        "geometric_mean_prob_rnn": geometric_mean_prob_rnn,
        "geometric_mean_prob_common_process": geometric_mean_prob_common_process
    }])

    out_path = "data/latent_results.csv"

else:
    print(f"normalized_ll vanilla: {geometric_mean_prob_rnn}")

    df = pd.DataFrame([{
        "model": "vanillaRNN",
        "geometric_mean_prob_rnn": geometric_mean_prob_rnn
    }])

    out_path = "data/vanilla_results.csv"


# --- Append to CSV or create it if missing ---
if not os.path.exists(out_path):
    df.to_csv(out_path, index=False)
else:
    df.to_csv(out_path, mode='a', header=False, index=False)

print(f"Saved results in {out_path}")


# --- Fit Cognitive Models ---
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}
if model_fitting:

    model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict, q_common_dict = fit.fit_all_models(
        model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
    )
    
    np.savez(f"data/params_dict.npz", **params_dict)
    print(f"parameter dict saved under data/params_dict.npz ✅")

    np.savez(f"data/q_common_dict.npz", **q_common_dict)
    print(f"q value dict saved under data/q_common_dict.npz ✅")

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
    rnn_no_id.to_csv(f"data/rnn_results_common_process.csv", index=False)
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
        torch.save(latent_tensor_train, f"data/latents_tensor{latent_nametag}_traindata.pt")
        if model_fitting:

            model_eval_df.to_csv(f"data/model_eval_df{latent_nametag}.csv", index=False)

    #torch.save(save_dict, f"checkpoints/ablated_rnn_best{latent_nametag}.pt")
    #torch.save(model.state_dict(), f"checkpoints/ablated_rnn{latent_nametag}.pt")
    np.savez(f"data/pA_rnn_dict{latent_nametag}.npz", **pA_rnn_dict)
    np.save(f"data/training_dict{latent_nametag}.npy", training_dict)
else:
    rnn_df.to_csv(f"data/rnn_results{vanilla_nametag}.csv", index=False)
    if palminteri:

        torch.save(latent_tensor, "data/latents_tensor_palminteri.pt")
        model_eval_df.to_csv("data/model_eval_df.csv", index=False)
    elif sloutsky:
        torch.save(latent_tensor, f"data/latents_tensor{vanilla_nametag}_sloutsky.pt")
        torch.save(latent_tensor_train, f"data/latents_tensor{vanilla_nametag}_sloutsky_traindata.pt")
    else:
        torch.save(latent_tensor, f"data/latents_tensor{vanilla_nametag}.pt")
        print(f"latent saved under data/latents_tensor{vanilla_nametag}.pt")
        if model_fitting:

            model_eval_df.to_csv(f"data/model_eval_df{vanilla_nametag}.csv", index=False)

        
    torch.save(save_dict, "checkpoints/ablated_rnn_best.pt")
    torch.save(model.state_dict(), "checkpoints/ablated_rnn.pt")
    np.savez("data/pA_rnn_dict.npz", **pA_rnn_dict)
    np.save("data/training_dict.npy", training_dict)


print("✅ Training and fitting done. Results saved to data/ and checkpoints/")


