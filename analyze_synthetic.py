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
from scipy.stats import pearsonr, ttest_rel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#latent = True
latent_nametag = "latentmodel" #loading
model_fitting = False
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
    RSA_title = latent_nametag if latent else vanilla_nametag
    params = test_parameter_df["alphaP_list"].values
    #latent_tensor = latent_tensor
    plf.plot_latents(dim_reduction="mds", latent_tensor = latent_tensor, avg=True, n_components=2, name="simulation_vanilla_uniform_new", df=df_test, param_array = params)
    distance_matrix_latents = plf.rsa_latents(latents = latent_tensor, metric="euclidean",title=RSA_title, reduction="avg", plot=True, original_data=False)
    distance_matrix_params = plf.rsa_latents(latents = alphaP_test, metric="euclidean",title="test_params", reduction="entire", plot=True, original_data=True)
    #calculate correlation coefficient between actual alpha values and RSA
    def vectorize_rsa(mat):
    # Take only the upper triangle, excluding diagonal
        return mat[np.triu_indices_from(mat, k=1)]
    vec_params  = vectorize_rsa(distance_matrix_params)
    vec_latents = vectorize_rsa(distance_matrix_latents)
    corr_matrix = np.corrcoef(vec_params,vec_latents)[0,1]
    print(f"correlation between latent and actual parameters: {corr_matrix}")

    

    model_eval_df = pd.read_csv(f"data/model_eval_df{latent_nametag}.csv")
    rnn_ID_df = pd.read_csv(f"data/rnn_resultslatentmodel.csv")
    rnn_common_process_df = pd.read_csv(f"data/rnn_results_common_process.csv")
    rnn_vanilla_df = pd.read_csv(f"data/rnn_resultsvanilla.csv")
    models = model_eval_df['model'].unique()

    # Masks + selection for model_eval_df
    Q_common_fit_mask   = model_eval_df["model"] == "Q (common fit)"
    Q_common_fit        = model_eval_df[Q_common_fit_mask]["normalized_likelihood"]

    Q_MAP_mask          = model_eval_df["model"] == "Q (MAP)"
    Q_map               = model_eval_df[Q_MAP_mask]["normalized_likelihood"]

    FQ_common_fit_mask  = model_eval_df["model"] == "FQ (common fit)"
    FQ_common_fit       = model_eval_df[FQ_common_fit_mask]["normalized_likelihood"]

    FQ_MAP_mask         = model_eval_df["model"] == "FQ (MAP)"
    FQ_map              = model_eval_df[FQ_MAP_mask]["normalized_likelihood"]

    True_model_mask     = model_eval_df["model"] == "True model"
    True_model          = model_eval_df[True_model_mask]["normalized_likelihood"]
    True_model_likelihood = np.mean(True_model)

    # Masks + selection for RNN model frames
    vanilla_rnn_mask    = rnn_vanilla_df["model"] == "vanillaRNN"
    rnn_vanilla         = rnn_vanilla_df[vanilla_rnn_mask]["normalized_likelihood"]

    rnn_ID_mask         = rnn_ID_df["model"] == "IDRNN"
    rnn_ID              = rnn_ID_df[rnn_ID_mask]["normalized_likelihood"]

    rnn_common_process_mask = rnn_common_process_df["model"] == "common_process_RNN"
    rnn_common_process      = rnn_common_process_df[rnn_common_process_mask]["normalized_likelihood"]

    # t-tests
    def print_ttest(name1, data1, name2, data2):
        t, p = ttest_rel(data1, data2)
        print(f"T-test: {name1} vs {name2}")
        print(f"t = {t:.4f}, p = {p:.4e}")
        print()
        return t, p
    
    def p_to_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "n.s."

    def add_sig(ax, x1, x2, y, h, p):
        """Draw significance bars from x1 to x2 at height y with annotation."""
        stars = p_to_stars(p)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
        ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', color='black')

    # Replace with the actual column name you want to test
    col = "normalized_likelihood"

    # Run + print
    t_q, p_q = print_ttest("Q (common fit)", Q_common_fit, "Q (MAP)", Q_map)
    t_fq, p_fq = print_ttest("FQ (common fit)", FQ_common_fit, "FQ (MAP)", FQ_map)
    t_rnnID, p_rnnID = print_ttest("common_process_RNN", rnn_common_process, "IDRNN", rnn_ID)
    t_rnn_vanilla_ID, p_rnn_vanilla_ID = print_ttest("vanillaRNN", rnn_vanilla, "IDRNN", rnn_ID)

    fig, ax = plt.subplots()

    models = ['Q CP', 'Q MAP', 'FQ CP', 'FQ MAP', 'RNNCP', 'RNNID', 'VanillaRNN']
    likelihoods = [np.mean(Q_common_fit), np.mean(Q_map), np.mean(FQ_common_fit), np.mean(FQ_map),np.mean(rnn_common_process), np.mean(rnn_ID), np.mean(rnn_vanilla)]
    bar_labels = ['common fit', 'individual differences', '_common fit', '_individual differences', '_common fit', '_individual differences', 'vanilla NN']
    bar_colors = ['tab:green', 'tab:blue', 'tab:green', 'tab:blue', 'tab:green', 'tab:blue','tab:orange' ]

    ax.bar(models, likelihoods, label=bar_labels, color=bar_colors)
    ax.axhline(True_model_likelihood, linestyle='--', color='black', linewidth=1.5, label='True model')
    ax.set_ylim(bottom=0.5)
    ax.set_ylabel('geometric mean probability per P')
    ax.set_title('Mean Probability of choosing correctly per Participant')
    ax.legend(title='Bar colors', loc='lower left')
    # ---- Add significance lines ----
    ymax = True_model_likelihood
    h = ymax * 0.01  # height of brackets

    add_sig(ax, 0, 1, ymax + h*2, h, p_q)                   # Q_common vs Q_MAP
    add_sig(ax, 2, 3, ymax + h*2, h, p_fq)            # FQ_common vs FQ_MAP
    add_sig(ax, 4, 5, ymax + h*2, h, p_rnnID)         # RNN_common vs RNNID
    add_sig(ax, 6, 5, ymax + h*2.3, h, p_rnn_vanilla_ID) # vanillaRNN vs IDRNN
    plt.savefig(f"plots/bar_plot_model_likelihoods.png")
    plt.show()

"""    plf.compare_model_performance(model_eval_df, models, file_id='model_comparison_z6_com2_ID10', save_dir='./plots/',
                            ylim=[0.5, 0.7],
                            plot_individual_ML=False, fig_width=8, x_tick_fontsize=4)"""
    


    


#plf.plot_latents(dim_reduction="pca", latent_tensor=fq_common_test, avg=True, n_components=2, name="FQ", df=None, df_train=None, group_col="game", subject_col = "subid", latent_tensor_train = None)
