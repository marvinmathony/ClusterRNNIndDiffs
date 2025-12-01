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
from scipy.stats import pearsonr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#latent = True
latent_nametag = "latentmodel" #loading
model_fitting = True
vanilla_nametag = "vanilla"
latent = True
analyze_Q = False

#### Load Data ####
# Always available / shared
df_train = pd.read_csv("data/df_train.csv")
df_test = pd.read_csv("data/df_test.csv")
parameter_df = pd.read_csv("data/true_parameter_values.csv")
test_parameter_df = pd.read_csv("data/true_test_parameter_values.csv")
fitted = np.load("data/params_dict.npz", allow_pickle=True)

fitted_params_dict = {}

for key in fitted.files:
    arr = fitted[key]
    
    if arr.dtype == object and arr.size == 1:
        # only unwrap if truly a scalar object
        fitted_params_dict[key] = arr.item()
    else:
        # otherwise keep the array as-is
        fitted_params_dict[key] = arr

#for key, value in fitted_params_dict.items():
#    print(f"key: {key}, value: {value}")
fq_map = fitted_params_dict['FQ_MAP']

fitted_FQ_params = [v[0] for k, v in fq_map.items()]
alphaP = parameter_df["alphaP_list"].to_numpy()
alphaP_test = test_parameter_df["alphaP_list"].to_numpy()
r, p = pearsonr(fitted_FQ_params, alphaP)
print("Correlation r =", r)
print("p-value =", p)

"""#look at only subset of participants
lower_bound = 0.4
upper_bound = 0.6
mask = (alphaP >= lower_bound) & (alphaP <= upper_bound)
alphaP_filtered = alphaP[mask]
fitted_FQ_params = np.array(fitted_FQ_params)
fitted_filtered = fitted_FQ_params[mask]
r_filtered, p_filtered = pearsonr(fitted_filtered, alphaP_filtered)
print(f"Correlation of subset between {lower_bound} and {upper_bound}: r =", r_filtered)
print(f"how many parameters are within bounds: {fitted_filtered.size}")
print("p-value of subset =", p_filtered)"""




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

if analyze_Q:
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

else:
    params = test_parameter_df["alphaP_list"].values
    #latent_tensor = latent_tensor
    plf.plot_latents(dim_reduction="mds", latent_tensor = latent_tensor, avg=False, n_components=2, name="simulation_latent_uniform", df=df_test, param_array = params)
    distance_matrix_latents = plf.rsa_latents(latents = latent_tensor, metric="cosine",title=latent_nametag, reduction="last", plot=True, original_data=False)
    distance_matrix_params = plf.rsa_latents(latents = alphaP_test, metric="cosine",title="test_params", reduction="entire", plot=True, original_data=True)
    #calculate correlation coefficient between actual alpha values and RSA
    def vectorize_rsa(mat):
    # Take only the upper triangle, excluding diagonal
        return mat[np.triu_indices_from(mat, k=1)]
    vec_params  = vectorize_rsa(distance_matrix_params)
    vec_latents = vectorize_rsa(distance_matrix_latents)
    corr_matrix = np.corrcoef(vec_params,vec_latents)[0,1]
    print(f"correlation between latent and actual parameters: {corr_matrix}")


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
        """model_eval_df = pd.read_csv(f"data/model_eval_df{vanilla_nametag}.csv")
        models = model_eval_df['model'].unique()
        print(f"models: {models}")
        plf.compare_model_performance_original(model_eval_df, models, file_id='model_comparison_z6_com2_ID10', save_dir='./plots/',
                                ylim=[0.5, 0.7],
                                plot_individual_ML=False, fig_width=8, x_tick_fontsize=4)"""

    


#plf.plot_latents(dim_reduction="pca", latent_tensor=fq_common_test, avg=True, n_components=2, name="FQ", df=None, df_train=None, group_col="game", subject_col = "subid", latent_tensor_train = None)
