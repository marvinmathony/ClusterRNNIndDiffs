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
model_fitting = False
vanilla_nametag = "vanilla"

#### Load Data ####
# Always available / shared
df_train = pd.read_csv("data/df_train.csv")
df_test = pd.read_csv("data/df_test.csv")

p1_common_dict = dict(np.load("data/p1_common_dict.npz", allow_pickle=True))
rewardsTrain = np.load("data/rewards_train.npy")


rnn_df = pd.read_csv("data/rnn_results.csv")
pA_rnn = dict(np.load("data/pA_rnn_dict.npz", allow_pickle=True))
training_dict = np.load("data/training_dict.npy", allow_pickle=True).item()
latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}.pt")
best_epoch = training_dict["best_epoch"]

if model_fitting:
    model_eval_df = pd.read_csv("data/model_eval_df.csv")


plf.plot_latents(dim_reduction="mds", latent_tensor=latent_tensor, avg=False, n_components=2, name="simulation_data", df=None, df_train=None, group_col="game", subject_col = "subid", latent_tensor_train = None)
