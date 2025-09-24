# === Likelihood-vs-latent-dim sweep & plotting ===
import os, numpy as np, pandas as pd
import torch, seaborn as sns, matplotlib.pyplot as plt

# re-use your imports/utilities
from data_utils import load_preprocess_data_VAE
from kalman_simulation import simulate_data_probit_random_walk, evaluate_log_likelihood_agent, evaluate_log_likelihood_agent_marginal
from compare_models import *
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- USER CONFIG --------------------
zdim_grid      = [1,2,3,4,5,7,9,13,16,20]#[1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]   # latent sizes to try
hidden_grid   = [1,2,3,5,7,9,10]  # hidden sizes to try
epochs         = 1700
repetitions    = 5                      # do >1 to get CIs over seeds/sims
data_tag       = "zdim_sweep_sameData_hiddensweep"           # used in saved file names
outdir_plots   = "plots"
outdir_data    = "data"
os.makedirs(outdir_plots, exist_ok=True)
os.makedirs(outdir_data, exist_ok=True)
# -----------------------------------------------------

def run_one_training(Xs, Ys, Xs_test, Ys_test, Xs_val, Ys_val, zdim, label):
    """
    Trains LatentRNN (LookupEncoder) and AblatedRNN for a given zdim and returns test mean LLs.
    IMPORTANT: This code assumes your train_* functions perform early stopping and return
    the model restored at the BEST validation score. If they currently return the LAST state,
    modify those functions to (a) track best state_dict, (b) load it before returning,
    and (c) (optionally) return any 'best_mu, best_lv' that correspond to that best model.
    """
    participant_ids = Xs.shape[0]

    m_emb = LatentRNN(LookupEncoder(participant_ids, zdim), zdim).to(device)
    m_abl = AblatedRNN().to(device)

    m_emb.name = f"LatentRNN(z={zdim})"
    m_abl.name = "AblatedRNN"

    # train with early stopping (assumed to return BEST models/latents)
    m_emb, mu_emb, lv_emb, _, _, _ = train_latentrnn_early_stopping(
        m_emb,
        torch.arange(Xs.size(0)).to(device),  # participant indices for LookupEncoder
        Xs, Ys, Xs_val, Ys_val,
        epochs=epochs
    )
    m_abl, _, _ = train_ablated_early_stopping(
        m_abl, Xs, Ys, Xs_val, Ys_val, epochs=epochs
    )

    # evaluate
    mean_ll_emb, _ = test_latentrnn(m_emb, mu_emb, lv_emb, Xs_test, Ys_test)
    mean_ll_abl, _, _ = test_ablated(m_abl, Xs_test, Ys_test)

    return {
        "LatentRNN": float(mean_ll_emb),
        "AblatedRNN": float(mean_ll_abl),
        "Delta_LL": float(mean_ll_emb - mean_ll_abl),
        "zdim": int(zdim),
        "label": label
    }

def split_train_val_per_participant(X, Y, val_frac=0.2, seed=42):
    """Split each participant's blocks into train/val, then stack back."""
    Xs_train, Ys_train, Xs_val, Ys_val = [], [], [], []
    rng = np.random.RandomState(seed)
    for b in range(X.shape[0]):     # participants
        X_blocks = X[b]             # (Bk, T, in_dim)
        Y_blocks = Y[b]             # (Bk, T)
        # scikit shuffle is deterministic given random_state; we pass a fixed one
        Xb_tr, Xb_va, Yb_tr, Yb_va = train_test_split(
            X_blocks, Y_blocks, test_size=val_frac, random_state=seed
        )
        Xs_train.append(Xb_tr); Ys_train.append(Yb_tr)
        Xs_val.append(Xb_va);   Ys_val.append(Yb_va)

    Xtrain = torch.stack(Xs_train).to(device)
    Ytrain = torch.stack(Ys_train).to(device)
    Xval   = torch.stack(Xs_val).to(device)
    Yval   = torch.stack(Ys_val).to(device)
    return Xtrain, Ytrain, Xval, Yval

all_rows = []
df_withID = simulate_data_probit_random_walk(with_id=True)
df_noID   = simulate_data_probit_random_walk(with_id=False)
datasets = [("with_id", df_withID), ("no_id", df_noID)]
# --- simulate & load
for label, dataframe in datasets:
    df = dataframe.copy()
    fname = os.path.join(outdir_data, f"sim_{label}.csv")
    df.to_csv(fname, index=False) # can be done outside the loop

    X, Y, Xtest, Ytest, pid = load_preprocess_data_VAE(fname, n_blocks_pp=30)
    X, Y, Xtest, Ytest = X.to(device), Y.to(device), Xtest.to(device), Ytest.to(device)
    Xtrain, Ytrain, Xval, Yval = split_train_val_per_participant(X, Y, val_frac=0.2, seed=42)
    participant_ids = Xtrain.shape[0]
    test_ids = Xtest.shape[0]
    #ll of ground truth model on test data
    if label == "with_id":
        agent_ll = evaluate_log_likelihood_agent_marginal(df, test_ids, n_samples=100, mu = [3, 2], sd=[1, 0.7])
    else:
        agent_ll = evaluate_log_likelihood_agent(df, test_ids)
    print(f"Ground truth agent test LL ({label}): {agent_ll:.4f}")
    
    for hidden in hidden_grid:
        print(f"\n=== Hidden size {hidden} ===")
        m_abl = AblatedRNN(hid=hidden).to(device)
        m_abl.name = "AblatedRNN"
        model_abl, _, _  = train_ablated_early_stopping(
            m_abl, Xtrain, Ytrain, Xval, Yval, epochs=epochs
        )
        mean_ll_abl, _, _ = test_ablated(model_abl, Xtest, Ytest)
        # save ll

        for rep in range(repetitions):
            print(f"\n==== Repetition {rep+1}/{repetitions} ====")

            for zdim in zdim_grid:
                print(f"[rep {rep}] {label} — training z={zdim}")
                m_emb = LatentRNN(LookupEncoder(participant_ids, zdim), z_dim=zdim, hid=hidden).to(device)
                m_emb.name = f"LatentRNN(z={zdim})"
                m_emb, mu_emb, lv_emb, _, _, _ = train_latentrnn_early_stopping(
                    m_emb,
                    torch.arange(Xtrain.size(0)).to(device),  # participant indices for LookupEncoder
                    Xtrain, Ytrain, Xval, Yval,
                    epochs=epochs
                )
                mean_ll_emb, _ = test_latentrnn(m_emb, mu_emb, lv_emb, Xtest, Ytest)

                """res = run_one_training(
                    Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, zdim=zdim, label=label
                )"""
                res = {
                    "LatentRNN": float(mean_ll_emb),
                    "AblatedRNN": float(mean_ll_abl),
                    "ground_truth_agent": float(agent_ll),
                    "Delta_LL": float(mean_ll_emb - mean_ll_abl),
                    "zdim": int(zdim),
                    "label": label,
                    "hidden": int(hidden)}
                
                res["rep"] = rep
                all_rows.append(res)

            """# --- per-dim plots (and saves)
            # bar plot of test LLs per model
            df_bar = pd.DataFrame({
                "model": ["LatentRNN", "AblatedRNN"],
                "mean_test_LL": [res["LatentRNN"], res["AblatedRNN"]]
            })"""
            """plt.figure(figsize=(5,4))
            sns.barplot(x="model", y="mean_test_LL", data=df_bar)
            plt.title(f"Test log-likelihood (z={zdim}, {label}, rep={rep})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir_plots, f"ll_by_model_z{zdim}_{label}_rep{rep}_{data_tag}.png"), dpi=200)
            plt.close()

            # delta plot
            plt.figure(figsize=(4.2,4))
            plt.axhline(0, linestyle="--")
            plt.scatter([zdim], [res["Delta_LL"]])
            plt.title(f"ΔLL (Latent − Ablated) | z={zdim}, {label}, rep={rep}")
            plt.xlabel("zdim"); plt.ylabel("Δ Log-Likelihood")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir_plots, f"delta_ll_z{zdim}_{label}_rep{rep}_{data_tag}.png"), dpi=200)
            plt.close()"""

# --- aggregate & save raw table
df_sweep = pd.DataFrame(all_rows)
csv_path = os.path.join(outdir_data, f"zdim_sweep_{data_tag}.csv")
df_sweep.to_csv(csv_path, index=False)
print("Saved:", csv_path)
print(df_sweep.head())

g = df_sweep.groupby(["label", "zdim"])["Delta_LL"]
summary = g.agg(
    mean="mean",
    std=lambda s: s.std(ddof=1),
    n="count",
).reset_index()
summary["ci"] = 1.96 * summary["std"] / np.sqrt(summary["n"].clip(lower=1))

# Optional wide tables (now these columns exist)
wide_mean = summary.pivot(index="zdim", columns="label", values="mean")
wide_ci   = summary.pivot(index="zdim", columns="label", values="ci")

plt.figure(figsize=(7, 4.5))
for label in ["with_id", "no_id"]:
    sub = (summary[summary["label"] == label]
           .sort_values("zdim"))
    plt.errorbar(
        sub["zdim"], sub["mean"], yerr=sub["ci"],
        marker="o", linestyle="-", capsize=4, linewidth=2,
        label=label.replace("_", " ")
    )
plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
plt.title("Δ Log-Likelihood (Latent − Ablated) vs z-dim")
plt.xlabel("Latent dimension (z)")
plt.ylabel("Δ Log-Likelihood (test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir_plots, f"delta_ll_vs_z_{data_tag}.png"), dpi=200)
plt.close()

# --- NEW: overlay plot of model LLs for with_id vs no_id on one figure ---
# long-form table of per-repetition LLs for each model
df_long = df_sweep.melt(
    id_vars=["label", "zdim", "rep", "hidden"],
    value_vars=["LatentRNN", "AblatedRNN", "ground_truth_agent"],
    var_name="model",
    value_name="ll",
)

# aggregate across repetitions to get mean +/- CI for each (model, label, zdim)
model_summary = (
    df_long.groupby(["model", "label", "zdim", "hidden"])
           .agg(mean=("ll", "mean"),
                std =("ll", lambda s: s.std(ddof=1)), # shouldn't this be sem?
                n   =("ll", "count"))
           .reset_index()
)
model_summary["ci"] = 1.96 * model_summary["std"] / np.sqrt(model_summary["n"].clip(lower=1))


# marker map for hidden sizes (cycle if needed)
marker_cycle = ['o','s','D','^','v','P','X','*','h','<','>']
hidden_levels = sorted(model_summary["hidden"].unique())
marker_map = {h: marker_cycle[i % len(marker_cycle)] for i, h in enumerate(hidden_levels)}

plt.figure(figsize=(9, 6))
palette   = dict(zip(["LatentRNN","AblatedRNN","ground_truth_agent"], sns.color_palette("tab10", 3)))
linestyle = {"with_id": "-", "no_id": ":"}

for h in hidden_levels:
    sub_h = model_summary[model_summary["hidden"] == h]
    for model in ["LatentRNN","AblatedRNN","ground_truth_agent"]:
        for lbl in ["with_id","no_id"]:
            sub = (sub_h[(sub_h["model"]==model) & (sub_h["label"]==lbl)]
                   .sort_values("zdim"))
            if sub.empty: 
                continue
            plt.plot(sub["zdim"], sub["mean"],
                     linestyle=linestyle[lbl], marker=marker_map[h], linewidth=2,
                     color=palette[model], label=f"{model} ({lbl.replace('_',' ')}, h={h})")
            plt.fill_between(sub["zdim"], sub["mean"]-sub["ci"], sub["mean"]+sub["ci"],
                             alpha=0.10, step=None, color=palette[model])

plt.title("Test log-likelihood vs. latent dimension\nColor=model, line=with/no ID, marker=hidden")
plt.xlabel("Latent dimension (z)")
plt.ylabel("Mean test log-likelihood")
# build compact legend: one for model/label and one for hidden markers
from matplotlib.lines import Line2D
model_lines = [Line2D([0],[0], color=palette[m], lw=3) for m in ["LatentRNN","AblatedRNN","ground_truth_agent"]]
model_labels = ["LatentRNN","AblatedRNN","ground truth"]
cond_lines  = [Line2D([0],[0], color="black", lw=2, linestyle=ls) for ls in ["-",
":"]]
cond_labels = ["with_id","no_id"]
hidden_marks = [Line2D([0],[0], color="black", marker=marker_map[h], lw=0, markersize=8) for h in hidden_levels]
hidden_labels = [f"h={h}" for h in hidden_levels]

leg1 = plt.legend(model_lines, model_labels, title="Model", loc="upper left")
leg2 = plt.legend(cond_lines,  cond_labels,  title="Data",  loc="lower left")
plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)
plt.legend(hidden_marks, hidden_labels, title="Hidden size", loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(outdir_plots, f"ll_models_by_condition_vs_z_markers_hidden_{data_tag}.png"), dpi=200, bbox_inches="tight")
plt.close()

# Which z gives the biggest average benefit?
best_by_label = (
    df_sweep.groupby(["label", "zdim"])["Delta_LL"].mean()
    .reset_index()
    .sort_values(["label", "Delta_LL"], ascending=[True, False])
    .groupby("label").first().reset_index()
)
best_by_label.to_csv(os.path.join(outdir_data, f"best_z_by_condition_{data_tag}.csv"), index=False)
print("Best z per condition:\n", best_by_label)

# Optional: small table plot of avg LLs by model and z for each condition
for label in ["with_id", "no_id"]:
    df_avg = (
        df_sweep[df_sweep["label"] == label]
        .groupby("zdim")[["LatentRNN", "AblatedRNN", "Delta_LL"]]
        .mean().reset_index()
    )
    plt.figure(figsize=(6.5, 3.8))
    df_melt = df_avg.melt(id_vars="zdim", value_vars=["LatentRNN", "AblatedRNN"], var_name="model", value_name="mean_LL")
    sns.lineplot(data=df_melt, x="zdim", y="mean_LL", hue="model", marker="o")
    plt.title(f"Mean test LL by z (condition: {label})")
    plt.xlabel("zdim"); plt.ylabel("Mean test log-likelihood")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_plots, f"mean_ll_by_z_{label}_{data_tag}.png"), dpi=200)
    plt.close()
