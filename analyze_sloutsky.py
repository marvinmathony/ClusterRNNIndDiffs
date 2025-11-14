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
vanilla_nametag = "vanilla"
latent = True



#### Load Data ####
# Always available / shared
df_train = pd.read_csv("data/train_df_sloutsky.csv")
df_test = pd.read_csv("data/test_df_sloutsky.csv")

if latent:

    latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}_sloutsky.pt")
    latent_tensor_train = torch.load(f"data/latents_tensor{latent_nametag}_sloutsky_traindata.pt")
else:
    latent_tensor = torch.load(f"data/latents_tensor{vanilla_nametag}_sloutsky.pt")
    latent_tensor_train = torch.load(f"data/latents_tensor{vanilla_nametag}_sloutsky_traindata.pt")



plf.plot_LDA(latent_tensor, latent_tensor_train, df_test, df_train, dataset_tag = "sloutsky", group_col = "game")
plf.plot_latents(dim_reduction="mds", latent_tensor = latent_tensor, avg=False, n_components=2, name="sloutsky", df=df_test, df_train=None, group_col="game", subject_col = "subid", latent_tensor_train = None)



