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
from sklearn.manifold import TSNE, MDS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#latent = True
latent_nametag = "latentmodel" #loading
model_fitting = True
vanilla_nametag = "vanilla"
latent = True

#### Load Data ####
# Always available / shared
df_train = pd.read_csv("data/df_train.csv")
df_test = pd.read_csv("data/df_test.csv")

p1_common_dict = dict(np.load("data/p1_common_dict.npz", allow_pickle=True))
rewardsTrain = np.load("data/rewards_train.npy")


rnn_df = pd.read_csv("data/rnn_results.csv")
pA_rnn = dict(np.load("data/pA_rnn_dict.npz", allow_pickle=True))
training_dict = np.load("data/training_dict.npy", allow_pickle=True).item()
if latent:

    latent_tensor = torch.load(f"data/latents_tensor{latent_nametag}.pt")
    print(f"loaded latents_tensor{latent_nametag}.pt")
    print(f"dim of latent tensor: {latent_tensor.shape}")
else:
    latent_tensor = torch.load(f"data/latents_tensor{vanilla_nametag}.pt")
    print(f"loaded latents_tensor{vanilla_nametag}.pt")

best_epoch = training_dict["best_epoch"]

if model_fitting:
    npz = np.load("data/q_common_dict.npz", allow_pickle=True)
    q_values_dict = {key: npz[key].item() for key in npz.files}
    for key, value in q_values_dict.items():
        print(f"key: {key}, value: {value}")
    fq_common_test = q_values_dict["FQ"]["test"]
    print(f"fq_common_test shape: {fq_common_test.shape}")
    fq_common_test_flat = fq_common_test.reshape(fq_common_test.shape[0], -1)
    colors = [0 if i < 100 else 1 for i in range(fq_common_test.shape[0])]
    """mean_Q = fq_common_test.mean(axis=2)        # shape (N, 2)
    mean_Q0 = mean_Q[:, 0]             # Q-value for action 0 per P
    mean_Q1 = mean_Q[:, 1]             # Q-value for action 1 per P

    last_Q0 = fq_common_test[:,0,-1]
    last_Q1 = fq_common_test[:,1,-1]

    plt.figure(figsize=(6,6))
    plt.scatter(
    last_Q0, 
    last_Q1, 
    c=colors,           # <--- integrate your colormap
    alpha=0.7)
    plt.xlabel("last Q1")
    plt.ylabel("last Q2")
    plt.title("last Q-values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/dim_reduction_last_FQ_values.png")
    plt.show()"""
    """Z = PCA(n_components=2).fit_transform(fq_common_test_flat)
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=colors, cmap='coolwarm')
    plt.title(f"FQ model common fit Q values")
    plt.xlabel(f"tsne 1")
    plt.ylabel(f"tsne 2")
    plt.grid(True)
    plt.savefig(f"plots/dim_reduction_FQ_values.png")
    plt.show()
    print(f"plot saved under plots/dim_reduction_FQ_values.png")"""

    Z = PCA(n_components=1).fit_transform(fq_common_test_flat)   # shape (N, 1)

    plt.figure(figsize=(8, 3))
    plt.scatter(
        np.arange(Z.shape[0]),   # x-axis = participant index
        Z[:, 0],                 # y-axis = PC1 value
        c=colors,
        cmap='coolwarm'
    )
    plt.title("FQ model common fit Q values (1D PCA)")
    plt.xlabel("Participant")
    plt.ylabel("PC1")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots/dim_reduction_FQ_values.png")
    plt.show()

    print("plot saved under plots/dim_reduction_FQ_values.png")

    """participants = sorted(fq_common.keys()) 
    param_array = np.vstack([q_common[p] for p in participants])
    print(param_array.shape)"""

    if latent:

        model_eval_df = pd.read_csv(f"data/model_eval_df{latent_nametag}.csv")
        models = model_eval_df['model'].unique()
        print(f"models: {models}")
        plf.compare_model_performance(model_eval_df, models, file_id='model_comparison_z6_com2_ID10', save_dir='./plots/',
                                ylim=[0.5, 0.7],
                                plot_individual_ML=False, fig_width=8, x_tick_fontsize=4)
    else:
        model_eval_df = pd.read_csv(f"data/model_eval_df{vanilla_nametag}.csv")
        models = model_eval_df['model'].unique()
        print(f"models: {models}")
        plf.compare_model_performance_original(model_eval_df, models, file_id='model_comparison_z6_com2_ID10', save_dir='./plots/',
                                ylim=[0.5, 0.7],
                                plot_individual_ML=False, fig_width=8, x_tick_fontsize=4)

    


#plf.plot_latents(dim_reduction="pca", latent_tensor=fq_common_test, avg=True, n_components=2, name="FQ", df=None, df_train=None, group_col="game", subject_col = "subid", latent_tensor_train = None)
