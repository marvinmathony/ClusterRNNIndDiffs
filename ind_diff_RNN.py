
import pandas as pd
import torch
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from data_utils import load_preprocess_data_VAE
from kalman_simulation import simulate_data_probit_random_walk
from compare_models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Setup
zdim = 10
epochs = 1700
repetitions = 1
data_tag = "testing_mutual_info_10latents"  # for saving files

results = []

def run_comparison(Xs, Ys, Xs_test, Ys_test, Xs_val, Ys_val, label="with_id"):
    participant_ids = Xs.shape[0]
    m_emb = LatentRNN(LookupEncoder(participant_ids, zdim), zdim).to(device)
    #m_emb_nocat = LatentRNN_nocat(LookupEncoder(participant_ids, zdim), zdim).to(device)
    m_abl = AblatedRNN().to(device)
    #m_global = GlobalLatentRNN(zdim).to(device)
    #m_psy = LatentRNN(PsychometricEncoder(2, zdim), zdim).to(device); 

    print("LatentRNN (embedding):", count_trainable_params(m_emb))
    print("AblatedRNN (actual):", count_trainable_params(m_abl))
    #print("LatentRNN (emb no cat):", count_trainable_params(m_emb_nocat))
    #print("GlobalLatentRNN:", count_trainable_params(m_global))
    #print("LatentRNN (psy):", count_trainable_params(m_psy))

    # Assign names for clarity
    m_emb.name = "LatentRNN"
    m_abl.name = "AblatedRNN"
    #m_emb_nocat.name = "NoCat"
    #m_global.name = "Global"
    #m_psy.name= "psy"

    # Train models
    val_index = int(0.8 * Xs.size(0))
    xenc_train = torch.arange(Xs.size(0)).to(device)
    xenc_val   = torch.arange(Xs_val.size(0)).to(device)
    xenc = torch.arange(Xs.size(0) + Xs_val.size(0)).to(device)
    #x_psy = torch.ones(Xs.size(0), 2).to(device).float()
    #print(f"xenc_train shape: {xenc_train.shape}, xenc_val shape: {xenc_val.shape}")
    #print(f"Xs shape: {Xs.shape}, Xs_val shape: {Xs_val.shape}, Ys shape: {Ys.shape}, Ys_val shape: {Ys_val.shape}")

    m_emb, mu_emb, lv_emb, _, _, h0 = train_latentrnn_early_stopping(m_emb, xenc_train, Xs, Ys, Xs_val, Ys_val, epochs=epochs)
    print(f"mu embedding shape: {mu_emb.shape}, lv embedding shape: {lv_emb.shape}")
    m_abl, _, _ = train_ablated_early_stopping(m_abl, Xs, Ys, Xs_val, Ys_val, epochs=epochs)
    #m_emb_nocat, mu_nocat, lv_nocat, _, _ = train_latentrnn_early_stopping(m_emb_nocat, xenc_train, Xs, Ys, Xs_val, Ys_val, epochs=epochs)
    #m_global, _ = train_global_latent(m_global, Xs, Ys, Xs_val, Ys_val, epochs=epochs)
    #m_psy,mu_psy, lv_psy, _, _ = train_latentrnn_early_stopping(m_psy, x_psy, Xs, Ys, Xs_val, Ys_val, epochs=epochs)

    # Evaluate models
    mean_ll_emb, _ = test_latentrnn(m_emb, mu_emb, lv_emb, Xs_test, Ys_test)
    mean_ll_abl, _, _ = test_ablated(m_abl, Xs_test, Ys_test)
    #mean_ll_nocat, _ = test_latentrnn(m_emb_nocat, mu_nocat, lv_nocat, Xs_test, Ys_test)
    #mean_ll_global, _ = test_global_latent(m_global, Xs_test, Ys_test)
    #mean_ll_psy, _ = test_latentrnn(m_psy, mu_psy, lv_psy, Xs_test, Ys_test)

    scores = {
        "LatentRNN": mean_ll_emb,
        "AblatedRNN": mean_ll_abl,
        #"NoCat": mean_ll_nocat,
        #"Global": mean_ll_global,
        #"psy": mean_ll_psy,
    }

    return scores, mu_emb, h0


for i in range(repetitions):
    print(f"\n==== Repetition {i+1}/{repetitions} ====")

    for with_id in [True]:
        label = "with_id" if with_id else "no_id"

        # Simulate
        df = simulate_data_probit_random_walk(with_id=with_id)
        fname = f"data/sim2_{label}_{i}.csv"
        df.to_csv(fname, index=False)

        # Load
        X, Y, Xtest, Ytest, pid = load_preprocess_data_VAE(fname, n_blocks_pp=30)
        X, Y = X.to(device), Y.to(device)
        print(f"Data shape: X={X.shape}, Y={Y.shape}, Xtest={Xtest.shape}, Ytest={Ytest.shape}, pid={len(pid)}")
        """val_index = int(X.shape[0] * 0.8)
        Xval, Yval = X[int(val_index):], Y[int(val_index):]
        Xtrain, Ytrain = X[:int(val_index)], Y[:int(val_index)]
        Xval, Yval = Xval.to(device), Yval.to(device)
        Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)"""


        # new arrays for train and val
        Xs_train, Ys_train, Xs_val, Ys_val = [], [], [], []

        for b in range(X.shape[0]):   # loop over participants
            X_blocks = X[b]   # (Bk, T, in_dim)
            Y_blocks = Y[b]   # (Bk, T)

            # split blocks for this participant
            Xb_train, Xb_val, Yb_train, Yb_val = train_test_split(
                X_blocks, Y_blocks, test_size=0.2, random_state=42
            )

            Xs_train.append(Xb_train); Ys_train.append(Yb_train)
            Xs_val.append(Xb_val);     Ys_val.append(Yb_val)

        # stack back to tensors
        Xtrain = torch.stack(Xs_train).to(device)  # (B, Bk_train, T, in_dim)
        Ytrain = torch.stack(Ys_train).to(device)  # (B, Bk_train, T)
        Xval   = torch.stack(Xs_val).to(device)    # (B, Bk_val, T, in_dim)
        Yval   = torch.stack(Ys_val).to(device)    # (B, Bk_val, T)
        print(f"Train shape: X={Xtrain.shape}, Y={Ytrain.shape}, Val shape: X={Xval.shape}, Y={Yval.shape}")
        
        #print(f"Data shape: X={X.shape}, Y={Y.shape}, Xtest={Xtest.shape}, Ytest={Ytest.shape}, pid={len(pid)}")
        Xtest, Ytest = Xtest.to(device), Ytest.to(device)

        # Run models
        logliks, mu_emb, h0 = run_comparison(Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, label=label)

        # Find winner
        best_model = max(logliks, key=logliks.get)

        results.append({
            "run": i,
            "generative_process": label,
            **logliks,
            "winner": best_model
        })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(f"data/model_comparison_results_{data_tag}.csv", index=False)
print(df_results.head())

# Count wins
conf_matrix = pd.crosstab(
    index=df_results["generative_process"],
    columns=df_results["winner"]
)

# Plot
plt.figure(figsize=(8, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Model Recovery Confusion Matrix")
plt.xlabel("Winning Model")
plt.ylabel("Generative Process")
plt.tight_layout()
plt.savefig(f"plots/model_wins_matrix_{data_tag}.png")
plt.show()

#### behavior analysis ####

behavior_df = pd.read_csv(fname)
def compute_behavioral_metrics(df):
    participants = df["participant"].unique()
    metrics = []

    for pid in participants:
        sub_df = df[df["participant"] == pid].copy()
        sub_df = sub_df.sort_values(["block", "trial"])

        actions = sub_df["action"].values
        rewards = sub_df["reward"].values
        RU = sub_df["RU"].values
        V_over_TU = sub_df["V_over_TU"].values
        beta_true = sub_df["beta_true"].values
        gamma_true = sub_df["gamma_true"].values

        # Compute switches (difference between consecutive actions)
        switch = (actions[1:] != actions[:-1]).sum()
        #compute perseverance
        n_switches = len(actions) - 1

        metrics.append({
            "participant": pid,
            "prop_left": (actions == 0).mean(),
            "prop_right": (actions == 1).mean(),
            "prop_switch": switch / n_switches if n_switches > 0 else 0,
            "mean_reward": rewards.mean(),
            "mean_RU": RU.mean(),
            "mean_V_over_TU": V_over_TU.mean(),
            "std_action": actions.std(),
            "beta_true": beta_true.mean(),
            "gamma_true": gamma_true.mean()
        })

    return pd.DataFrame(metrics)

# Compute the metrics
z = mu_emb #h0 #
behavioral_df = compute_behavioral_metrics(behavior_df)
behavioral_df =  behavioral_df[:z.shape[0]]#behavioral_df[:z.shape[0]]  # ensure same number of participants
print(f"behavioral_df: {behavioral_df.head()}")

metric_cols = [
    "prop_left", "prop_right", "prop_switch",
    "mean_reward", "mean_RU", "mean_V_over_TU",
    "std_action", "beta_true", "gamma_true"
]
metrics_df = behavioral_df.sort_values("participant").reset_index(drop=True)
behavior_tensor = torch.from_numpy(metrics_df[metric_cols].to_numpy(dtype=np.float32))
# for safekeeping
participant_ids = metrics_df["participant"].to_list()
feature_names = metric_cols
print(f"z shape: {z.shape}, behavior_tensor shape: {behavior_tensor.shape}")


# z: participant latents (B, z_dim)
# behavior: behavioral summaries (B, M)
B, z_dim, M = behavior_tensor.shape[0], z.shape[1], behavior_tensor.shape[1]
torch.manual_seed(42)

# Convert to numpy
z_np = z.cpu().detach().numpy()
behavior_np = behavior_tensor.cpu().detach().numpy()

# 1. Mutual Information heatmap
mi_matrix = np.zeros((M, z_dim))
for i in range(M):
    for j in range(z_dim):
        mi_matrix[i, j] = mutual_info_regression(z_np[:, [j]], behavior_np[:, i], discrete_features=False)

df_mi = pd.DataFrame(
    mi_matrix,
    columns=[f"z{i}" for i in range(z_dim)],
    index=metric_cols   # <-- use actual metric names here
)

plt.figure(figsize=(8, 6))
sns.heatmap(df_mi, annot=False, cmap="viridis", cbar_kws={'label': 'Mutual Information'})
plt.title("Mutual Information: Latents vs Behavioral Metrics")
plt.xlabel("Latent Dimensions")
plt.ylabel("Behavioral Metrics")
plt.tight_layout()
plt.savefig(f"plots/mutual_info{data_tag}.png")
plt.show()

# Cluster participants
kmeans = KMeans(n_clusters=3, random_state=42).fit(z_np)
clusters = kmeans.labels_
print(f"count of participants per cluster: {pd.Series(clusters).value_counts().sort_index()}")

# Build behavior dataframe with actual names
df_behavior = pd.DataFrame(behavior_np, columns=metric_cols)  # <-- use metric_cols
df_behavior["cluster"] = clusters

# Plot
plt.figure(figsize=(12, 4))
for i, metric in enumerate(metric_cols):
    plt.subplot(1, M, i + 1)
    sns.boxplot(x="cluster", y=metric, data=df_behavior, palette="Set2")
    #plt.title(f"{metric}")
plt.tight_layout()
plt.savefig(f"plots/clustering{data_tag}.png")
plt.show()

ridge = RidgeCV(alphas=np.logspace(-6, 6, 13)).fit(z_np, behavior_np)
ridge_r2 = r2_score(behavior_np, ridge.predict(z_np), multioutput='raw_values')

tree = DecisionTreeRegressor(max_depth=3).fit(z_np, behavior_np)
tree_r2 = r2_score(behavior_np, tree.predict(z_np), multioutput='raw_values')

# Collect R² scores with real metric names
df_r2 = pd.DataFrame({
    "metric": metric_cols,   # <-- use real names
    "ridge_r2": ridge_r2,
    "tree_r2": tree_r2
})

r2_mat = (
    df_r2.replace([np.inf, -np.inf], np.nan)
         .dropna()
         .set_index("metric")
         .loc[metric_cols, ["ridge_r2", "tree_r2"]]  # keep column order
)

plt.figure(figsize=(6, 0.6*len(metric_cols) + 2))
ax = sns.heatmap(
    r2_mat,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    vmin=-0.5, vmax=1.0,              # R² ranges; adjust vmin if you expect lower
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "R²"}
)
ax.set_title("R² by Metric and Model")
ax.set_xlabel("Model")
ax.set_ylabel("Metric")
plt.tight_layout()
plt.savefig(f"plots/r2_heatmap{data_tag}.png", dpi=200)
plt.show()


"""# shuffling decision tree
def cv_tree_r2(X, y, n_splits=5, depths=(2,3,4,5,6), rng=None):
    """"""
    Cross-validated R^2 with a decision tree.
    Depth is tuned on each train fold via inner CV (simple holdout).
    """"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0 if rng is None else rng.randint(10_000))
    fold_r2 = []

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # simple inner tuning by holdout split of the train fold
        inner = KFold(n_splits=3, shuffle=True, random_state=1 if rng is None else rng.randint(10_000))
        best_depth, best_inner = None, -np.inf
        for d in depths:
            inner_scores = []
            for itr, ival in inner.split(Xtr):
                dt = DecisionTreeRegressor(max_depth=d, random_state=123)
                dt.fit(Xtr[itr], ytr[itr])
                inner_scores.append(r2_score(ytr[ival], dt.predict(Xtr[ival])))
            mean_inner = np.mean(inner_scores)
            if mean_inner > best_inner:
                best_inner, best_depth = mean_inner, d

        # train on full train fold with best depth and evaluate on test fold
        dt_best = DecisionTreeRegressor(max_depth=best_depth, random_state=123)
        dt_best.fit(Xtr, ytr)
        fold_r2.append(r2_score(yte, dt_best.predict(Xte)))

    return float(np.mean(fold_r2)), np.array(fold_r2)

def shuffle_test_tree(
    Z, beh_df, metrics, n_splits=5, n_shuffles=1000, depths=(2,3,4,5,6), seed=42, make_plots=True
):
    """"""
    For each metric:
      - compute CV R^2 (real)
      - compute null distribution by shuffling y across participants
      - return empirical p-value and z-score
    """"""
    rng = np.random.default_rng(seed)
    X = np.asarray(Z)

    rows = []
    for m in metrics:
        y = beh_df[m].to_numpy().astype(float)

        # real score
        real_mean_r2, real_folds = cv_tree_r2(X, y, n_splits=n_splits, depths=depths, rng=rng)

        # null by shuffling labels across participants
        null_r2 = np.empty(n_shuffles)
        for s in range(n_shuffles):
            y_shuf = y.copy()
            rng.shuffle(y_shuf)
            null_r2[s], _ = cv_tree_r2(X, y_shuf, n_splits=n_splits, depths=depths, rng=rng)

        # empirical p-value (greater or equal under the null)
        p = (1 + np.sum(null_r2 >= real_mean_r2)) / (n_shuffles + 1)
        # z vs. null
        z = (real_mean_r2 - null_r2.mean()) / (null_r2.std(ddof=1) + 1e-12)

        rows.append({
            "metric": m,
            "real_r2": real_mean_r2,
            "null_mean": null_r2.mean(),
            "null_std": null_r2.std(ddof=1),
            "p_value": p,
            "z_score": z
        })

        if make_plots:
            plt.figure(figsize=(5,3))
            plt.hist(null_r2, bins=30, alpha=0.8)
            plt.axvline(real_mean_r2, linestyle="--", linewidth=2)
            plt.title(f"Shuffle test: {m}\nreal R²={real_mean_r2:.2f}, p={p:.3g}, z={z:.2f}")
            plt.xlabel("R² (null)")
            plt.ylabel("count")
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(rows)

# --- Example usage ---
# metrics = ["prop_left","prop_right","prop_switch","mean_reward","mean_RU",
#            "mean_V_over_TU","std_action","beta_true","gamma_true"]
# results_df = shuffle_test_tree(Z, beh_df, metrics, n_splits=5, n_shuffles=500)
# print(results_df.sort_values("p_value"))"""