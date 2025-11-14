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
palminteri = True

conditions = [True]
for latent in conditions:

    data_tag = "latentmodel_palminteri" if latent else "vanilla_palminteri" #saving

    #### Load Data ####
    # Always available / shared
    df_train = pd.read_csv("data/train_df_palminteri.csv")
    df_test = pd.read_csv("data/test_df_palminteri.csv")
    p1_common_dict = dict(np.load("data/p1_common_dict.npz", allow_pickle=True))
    rewardsTrain = np.load("data/rewards_train.npy")

    # Depending on model
    if latent:
        model_eval_df = pd.read_csv(f"data/model_eval_df{latent_nametag}.csv")
        rnn_df = pd.read_csv(f"data/rnn_results{latent_nametag}.csv")
        pA_rnn = dict(np.load(f"data/pA_rnn_dict{latent_nametag}.npz", allow_pickle=True))
        training_dict = np.load(f"data/training_dict{latent_nametag}.npy", allow_pickle=True).item()
        if palminteri:
            latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}_palminteri.pt")
            latent_tensor_train = torch.load(f"data/latents_tensor{latent_nametag}_palminteri_traindata.pt")
            print(f"latent tensor train dim: {latent_tensor_train.shape}")
        else:
            latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}.pt")
    """else:
        model_eval_df = pd.read_csv("data/model_eval_df.csv")
        rnn_df = pd.read_csv("data/rnn_results.csv")
        pA_rnn = dict(np.load("data/pA_rnn_dict.npz", allow_pickle=True))
        training_dict = np.load("data/training_dict.npy", allow_pickle=True).item()
        latent_tensor = torch.load("data/latents_tensor.pt")"""

    best_epoch = training_dict["best_epoch"]
    print(f"latent tensor shape: {latent_tensor.shape}")

plf.plot_palminteri_LDA(latent_tensor, latent_tensor_train, df_test, df_train)

#plf.plot_latents(dim_reduction="lds", latent_tensor=latent_tensor, avg=False, n_components=2, name=data_tag, df=df_test, df_train = df_train, group_col="group", latent_tensor_train = latent_tensor_train)
#plf.plot_latents(dim_reduction="lds", latent_tensor=latent_tensor, avg=True, n_components=2, name=data_tag, df=df_test, df_train = df_train, group_col="group", latent_tensor_train = latent_tensor_train)
