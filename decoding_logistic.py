"""
Age-group decoding from IDRNN vs Vanilla latents, aggregated across CV-pipeline seeds.

For each seed:
  - Loads latents from data_sloutsky/seed_{seed}/
  - Runs LOOCV multinomial logistic regression (LOOCV outer, StratifiedKFold inner)
  - Records balanced accuracy

Statistical comparison (paired across seeds):
  - Wilcoxon signed-rank test (primary; non-parametric, robust for n=11)
  - Paired t-test (secondary)

Output: plots_sloutsky/logistic_decoding_cv_seeds.png
"""

import os
import glob
import json
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data_sloutsky"
PLOT_DIR  = "plots_sloutsky"
TEST_CSV  = "data/test_df_sloutsky.csv"
TRAIN_CSV = "data/train_df_sloutsky.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Labels (same for every seed) ─────────────────────────────────────────────
test_df    = pd.read_csv(TEST_CSV)
df_unique  = test_df.drop_duplicates(subset="subid").sort_values("subid").reset_index(drop=True)
group_map  = {"young_child": 0, "old_child": 1, "adult": 2}
labels     = np.array([group_map[g] for g in df_unique["age"]])
N          = len(labels)

# -- train labels --------
train_df    = pd.read_csv(TRAIN_CSV)
df_unique_train  = train_df.drop_duplicates(subset="subid").sort_values("subid").reset_index(drop=True)
group_map_train  = {"young_child": 0, "old_child": 1, "adult": 2}
labels_train     = np.array([group_map_train[g] for g in df_unique_train["age"]])
N_train          = len(labels_train)

# -- cognitive parameters -------
def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

def unpack_params(pars_vec):
    pars_vec = np.asarray(pars_vec, dtype=float)
    theta = np.exp(pars_vec[0])                 # >0
    w_train = softmax(pars_vec[1:5])            # 4 weights sum to 1
    return {
        "theta": theta,
        "b_value_train": w_train[0],
        "b_uncertain_train": w_train[1],
        "b_lag_train": w_train[2],
        "b_novelty_train": w_train[3],
    }

em = np.load("data_sloutsky/em_results.npz", allow_pickle=True)
h_all_test        = em["h_all_test"]
participants_test = em["participants_test"]
B_test  = N

_cog = [unpack_params(h_all_test[i]) for i in range(B_test)]
COG_VARS = {
    "theta":    np.array([p["theta"]             for p in _cog]),
    "w_value":  np.array([p["b_value_train"]     for p in _cog]),
    "w_uncert": np.array([p["b_uncertain_train"] for p in _cog]),
    "w_lag":    np.array([p["b_lag_train"]       for p in _cog]),
    "w_novelty":np.array([p["b_novelty_train"]   for p in _cog]),
}
true_nov = COG_VARS["w_novelty"]
cog_params = np.stack((COG_VARS["w_novelty"], COG_VARS["w_lag"], COG_VARS["w_uncert"], COG_VARS["w_value"], COG_VARS["theta"]), axis=1)

# ── LOOCV decoding ────────────────────────────────────────────────────────────
def loocv_bal_acc(X, y, inner_cv=5):
    """LOOCV with inner StratifiedKFold for C-selection. Returns balanced accuracy."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    loo  = LeaveOneOut()
    preds = np.empty(len(y), dtype=classes.dtype)
    for tr, te in loo.split(X):
        y_tr = y[tr]
        n_splits = min(inner_cv, np.min(np.bincount(y_tr)))
        cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=3),
            LogisticRegressionCV(
                penalty="l2",
                solver="lbfgs", 
                max_iter=5000, cv=cv,
            )
        )
        clf.fit(X[tr], y_tr)
        preds[te] = clf.predict(X[te])
    return balanced_accuracy_score(y, preds), preds


def logistic_regression_traindata(X,y, inner_cv):
  X = np.asarray(X)
  y = np.asarray(y)
  classes = np.unique(y)
  K = len(classes)
  #loo = LeaveOneOut()
  preds = np.empty(len(y), dtype=classes.dtype)
  probs = np.zeros((len(y), K), dtype=float)
  #for tr, te in tqdm(loo.split(X), total=len(y), desc="LOOCV"):
  min_class = np.min(np.bincount(y))
  if min_class < 2:
    raise ValueError("Not enough samples in the smallest class for inner CV.")
  n_splits = min(inner_cv, min_class)
  cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
  base = LogisticRegressionCV(
              penalty="l2",
              solver="lbfgs",          # supports multinomial
              max_iter=5000,
              cv=cv,
          )
  clf = make_pipeline(StandardScaler(), base)
  #clf = base
  clf.fit(X, y)

  preds = clf.predict(X)
  probs[:, :] = clf.predict_proba(X)[0]

  bal_acc = balanced_accuracy_score(y, preds)
  # multiclass AUC; requires probs for all classes
  auc = 0#roc_auc_score(y, probs, multi_class="ovr", average="macro")

  return preds, probs, bal_acc, auc, clf

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

#RBF-SVM analysis
def rbfsvm_analysis_grid_search(X, y, C_param_list, gamma_param_list):
    param_grid = {'svc__C': C_param_list, 'svc__gamma': gamma_param_list}
    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    grid_search = GridSearchCV(pipe, param_grid, cv=3)
    grid_search.fit(X, y)
    best_params = {k.replace('svc__', ''): v for k, v in grid_search.best_estimator_.get_params().items()
                   if k in ('svc__C', 'svc__gamma')}
    return best_params, grid_search.best_score_ * 100

def run_rbfsvm_analysis(grid_search_fun, X, y, C_params, gamma_params, n_repeats=20):
    best_params, _ = grid_search_fun(X, y, C_params, gamma_params)
    accs = []
    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=i)
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma']))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accs.append(balanced_accuracy_score(y_test, y_pred))
    return np.mean(accs)

def run_rbfsvm_analysis_traintest(grid_search_fun, X_train, y_train, X_test, y_test, C_params, gamma_params):
    best_params, _ = grid_search_fun(X_train, y_train, C_params, gamma_params)
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma']))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)



    




# ── Iterate seeds ─────────────────────────────────────────────────────────────
seed_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "seed_*")))
if not seed_dirs:
    raise FileNotFoundError(f"No seed_* directories found in {DATA_DIR}")

bal_accs_idrnn   = []
bal_accs_vanilla = []
seeds_used       = []

bal_accs_idrnn_traindata = []
bal_accs_vanilla_traindata = []

bal_accs_rbf_svm_idrnn = []
bal_accs_rbf_svm_vanilla = []
bal_accs_rbf_svm_traintest_idrnn = []
bal_accs_rbf_svm_traintest_vanilla = []
C_params     = [0.01, 0.1, 1, 10, 100]
gamma_params = [0.01, 0.1, 1, 'scale', 'auto']


for sd in seed_dirs:
    seed = os.path.basename(sd)
    idrnn_path   = os.path.join(sd, "latents_tensorlatentmodel.pt")
    vanilla_path = os.path.join(sd, "latents_tensorvanilla.pt")
    idrnn_train_path = os.path.join(sd, "latents_tensorlatentmodel_traindata.pt")
    vanilla_train_path = os.path.join(sd, "latents_tensorvanilla_traindata.pt")




    if not os.path.exists(idrnn_path) or not os.path.exists(vanilla_path):
        print(f"  Skipping {seed}: missing latent file(s)")
        continue

    lat_idrnn   = torch.load(idrnn_path,   map_location="cpu").numpy().mean(axis=1)#[:, -1, :]  # last timestep
    lat_vanilla = torch.load(vanilla_path, map_location="cpu").numpy().mean(axis=1)  # time-avg

    lat_idrnn_train = torch.load(idrnn_train_path, map_location="cpu").numpy().mean(axis=1)#[:, -1, :]
    lat_vanilla_train = torch.load(vanilla_train_path, map_location="cpu").numpy().mean(axis=1)

    if lat_idrnn.shape[0] != N or lat_vanilla.shape[0] != N:
        print(f"  Skipping {seed}: latent n={lat_idrnn.shape[0]} != label n={N}")
        continue

    # pca = PCA(n_components=2)
    # pca_latents = pca.fit_transform(lat_idrnn)
    # lat_idrnn = pca_latents
    # pca_latents = pca.fit_transform(lat_vanilla)
    # lat_vanilla = pca_latents
    # print("------ training loo on test tensors -------")
    # print(f"  {seed}: decoding IDRNN ...", flush=True)
    # ba_idrnn, _ = loocv_bal_acc(lat_idrnn, labels)
    # print(f"  {seed}: decoding Vanilla ...", flush=True)
    # ba_van, _   = loocv_bal_acc(lat_vanilla, labels)

    # bal_accs_idrnn.append(ba_idrnn)
    # bal_accs_vanilla.append(ba_van)
    seeds_used.append(seed)
    # print(f"  {seed}: IDRNN={ba_idrnn:.3f}  Vanilla={ba_van:.3f}")

    # # --- logistic on train data and tensors
    # print("----- training logistic regression on train tensors, testing on test tensors -----")
    # predictions_idrnn, probs_idrnn, bal_acc_idrnn_train, auc_idrnn_train, lo_idrnn_train = logistic_regression_traindata(lat_idrnn_train, labels_train, inner_cv=30)
    # predictions_vanilla, probs_vanilla, bal_acc_vanilla_train, auc_vanilla_train, lo_vanilla_train = logistic_regression_traindata(lat_vanilla_train, labels_train, inner_cv=30)

    # print(f"IDRNN logistic regression: balanced accuracy: {bal_acc_idrnn_train}")
    # print(f"Vanilla logistic regression: balanced accuracy: {bal_acc_vanilla_train}")

    # #use the same regression to fit test data
    # predictions_idrnn_test = lo_idrnn_train.predict(lat_idrnn)
    # bal_acc_idrnn_test = balanced_accuracy_score(labels, predictions_idrnn_test)
    # print(f"IDRNN logistic regression on test data. Fitted on train data. Balanced accuracy: {bal_acc_idrnn_test}")
    # bal_accs_idrnn_traindata.append(bal_acc_idrnn_test)

    # predictions_vanilla_test = lo_vanilla_train.predict(lat_vanilla)
    # bal_acc_vanilla_test = balanced_accuracy_score(labels, predictions_vanilla_test)
    # print(f"Vanilla logistic regression on test data. Fitted on train data. Balanced accuracy: {bal_acc_vanilla_test}")
    # bal_accs_vanilla_traindata.append(bal_acc_vanilla_test)

    # rbf-svm analysis
    bal_acc_rbfsvm_idrnn = run_rbfsvm_analysis(rbfsvm_analysis_grid_search, lat_idrnn, labels, C_params, gamma_params, n_repeats=20)
    bal_acc_rbfsvm_vanilla = run_rbfsvm_analysis(rbfsvm_analysis_grid_search, lat_vanilla, labels, C_params, gamma_params, n_repeats=20)
    bal_accs_rbf_svm_idrnn.append(bal_acc_rbfsvm_idrnn)
    bal_accs_rbf_svm_vanilla.append(bal_acc_rbfsvm_vanilla)
    print(f"----- rbfsvm analysis seed {seed} -----\n",
          f"balanced acc idrnn: {bal_acc_rbfsvm_idrnn}, balanced acc vanilla: {bal_acc_rbfsvm_vanilla}")

    # rbf-svm train→test analysis
    bal_acc_rbfsvm_traintest_idrnn = run_rbfsvm_analysis_traintest(rbfsvm_analysis_grid_search, lat_idrnn_train, labels_train, lat_idrnn, labels, C_params, gamma_params)
    bal_acc_rbfsvm_traintest_vanilla = run_rbfsvm_analysis_traintest(rbfsvm_analysis_grid_search, lat_vanilla_train, labels_train, lat_vanilla, labels, C_params, gamma_params)
    bal_accs_rbf_svm_traintest_idrnn.append(bal_acc_rbfsvm_traintest_idrnn)
    bal_accs_rbf_svm_traintest_vanilla.append(bal_acc_rbfsvm_traintest_vanilla)
    print(f"----- rbfsvm train→test analysis seed {seed} -----\n",
          f"balanced acc idrnn: {bal_acc_rbfsvm_traintest_idrnn}, balanced acc vanilla: {bal_acc_rbfsvm_traintest_vanilla}")


n_seeds = len(seeds_used)
print(f"\n{n_seeds} seeds decoded.")

# bal_accs_idrnn   = np.array(bal_accs_idrnn)
# bal_accs_vanilla = np.array(bal_accs_vanilla)
# bal_accs_idrnn_traindata = np.array(bal_accs_idrnn_traindata)
# bal_accs_vanilla_traindata = np.array(bal_accs_vanilla_traindata)

# mean_trainregression_idrnn = bal_accs_idrnn_traindata.mean()
# sem_trainregression_idrnn = bal_accs_idrnn_traindata.std(ddof=1) / np.sqrt(n_seeds)
# mean_trainregression_vanilla = bal_accs_vanilla_traindata.mean()
# sem_trainregression_vanilla = bal_accs_vanilla_traindata.std(ddof=1) / np.sqrt(n_seeds)

# stat_t_train, p_ttest_train = stats.ttest_rel(bal_accs_idrnn_traindata, bal_accs_vanilla_traindata,
#                                    alternative="greater")
# print(f"mean of regression trained on train set: idrnn: {mean_trainregression_idrnn}, vanilla: {mean_trainregression_vanilla}, one sided t-test (idrnn mean greater than vanilla mean): {p_ttest_train}")

# # --- logistic on cog model params ----
# ba_cog_params, _ = loocv_bal_acc(cog_params, labels)
# print(f"ceiling decodability of age: {ba_cog_params}")

# --- rbfsvm ----
bal_acc_rbfsvm_cog = run_rbfsvm_analysis(rbfsvm_analysis_grid_search, cog_params, labels, C_params, gamma_params, n_repeats=20)

bal_accs_rbf_svm_idrnn   = np.array(bal_accs_rbf_svm_idrnn)
mean_rbfsvm_idrnn = bal_accs_rbf_svm_idrnn.mean()
sem_rbfsvm_idrnn = bal_accs_rbf_svm_idrnn.std(ddof=1) / np.sqrt(n_seeds)

bal_accs_rbf_svm_vanilla   = np.array(bal_accs_rbf_svm_vanilla)
mean_rbfsvm_vanilla = bal_accs_rbf_svm_vanilla.mean()
sem_rbfsvm_vanilla = bal_accs_rbf_svm_vanilla.std(ddof=1) / np.sqrt(n_seeds)

stat_t_rbfsvm, p_ttest_rbfsvm = stats.ttest_rel(bal_accs_rbf_svm_idrnn, bal_accs_rbf_svm_vanilla,
                                   alternative="greater")

print(f"rbf svm analysis: cog model balanced acc: {bal_acc_rbfsvm_cog}, idrnn balanced acc: {mean_rbfsvm_idrnn}, sem: {sem_rbfsvm_idrnn}\n",
      f"vanilla balanced acc: {mean_rbfsvm_vanilla}, sem: {sem_rbfsvm_vanilla}, one-sided paired t-test: {p_ttest_rbfsvm}")

# --- rbfsvm train→test ----

bal_accs_rbf_svm_traintest_idrnn   = np.array(bal_accs_rbf_svm_traintest_idrnn)
mean_rbfsvm_traintest_idrnn = bal_accs_rbf_svm_traintest_idrnn.mean()
sem_rbfsvm_traintest_idrnn  = bal_accs_rbf_svm_traintest_idrnn.std(ddof=1) / np.sqrt(n_seeds)

bal_accs_rbf_svm_traintest_vanilla   = np.array(bal_accs_rbf_svm_traintest_vanilla)
mean_rbfsvm_traintest_vanilla = bal_accs_rbf_svm_traintest_vanilla.mean()
sem_rbfsvm_traintest_vanilla  = bal_accs_rbf_svm_traintest_vanilla.std(ddof=1) / np.sqrt(n_seeds)

stat_t_rbfsvm_traintest, p_ttest_rbfsvm_traintest = stats.ttest_rel(bal_accs_rbf_svm_traintest_idrnn, bal_accs_rbf_svm_traintest_vanilla,
                                   alternative="greater")

print(f"rbf svm train→test: cog model balanced acc: {bal_acc_rbfsvm_cog}, idrnn balanced acc: {mean_rbfsvm_traintest_idrnn}, sem: {sem_rbfsvm_traintest_idrnn}\n",
      f"vanilla balanced acc: {mean_rbfsvm_traintest_vanilla}, sem: {sem_rbfsvm_traintest_vanilla}, one-sided paired t-test: {p_ttest_rbfsvm_traintest}")



# # ── Statistics ────────────────────────────────────────────────────────────────
# # Primary: Wilcoxon signed-rank (non-parametric paired test, robust for small n)
# diffs = bal_accs_idrnn - bal_accs_vanilla
# stat_w, p_wilcoxon = stats.wilcoxon(bal_accs_idrnn, bal_accs_vanilla,
#                                      alternative="greater")

# # Secondary: paired t-test
# stat_t, p_ttest = stats.ttest_rel(bal_accs_idrnn, bal_accs_vanilla,
#                                    alternative="greater")

# mean_idrnn   = bal_accs_idrnn.mean()
# sem_idrnn    = bal_accs_idrnn.std(ddof=1) / np.sqrt(n_seeds)
# mean_vanilla = bal_accs_vanilla.mean()
# sem_vanilla  = bal_accs_vanilla.std(ddof=1) / np.sqrt(n_seeds)
# chance       = 1.0 / len(np.unique(labels))

# print(f"\nIDRNN   : {mean_idrnn:.3f} ± {sem_idrnn:.3f} (SEM)")
# print(f"Vanilla: {mean_vanilla:.3f} ± {sem_vanilla:.3f} (SEM)")
# print(f"Wilcoxon (one-sided IDRNN>Vanilla): W={stat_w:.1f}, p={p_wilcoxon:.4f}")
# print(f"Paired t-test (one-sided):           t={stat_t:.2f}, p={p_ttest:.4f}")

# # generalizing logistic regression bar plot 

# x = [0,1]
# y = [mean_trainregression_idrnn, mean_trainregression_vanilla]
# c = [sem_trainregression_idrnn, sem_trainregression_vanilla]
# cols = ["#4C72B0", "#DD8452"]

# fig, ax = plt.subplots()
# ax.bar(x, y, yerr=c, color=cols, edgecolor="k", capsize=5, width=0.5)
# ax.set_xticks(x)
# ax.set_xticklabels(["IDRNN", "Vanilla"])
# ax.set_title(f"one-sided t-test: {p_ttest_train}", fontsize=11)
# out = os.path.join(PLOT_DIR, "logistic_decoding_cv_seeds_traindata.png")
# plt.savefig(out, dpi=150, bbox_inches="tight")

# # ── Plot ──────────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(4.5, 5))

# x    = np.array([0, 1])
# cols = ["#4C72B0", "#DD8452"]
# labels_bar = ["IDRNN", "Vanilla"]
# means = [mean_idrnn, mean_vanilla]
# sems  = [sem_idrnn,  sem_vanilla]

# bars = ax.bar(x, means, yerr=sems, color=cols, edgecolor="k",
#               width=0.5, capsize=5, zorder=2)

# # Individual seed dots connected by lines
# rng = np.random.default_rng(0)
# jitter = rng.uniform(-0.06, 0.06, n_seeds)
# for i, (ba_i, ba_v) in enumerate(zip(bal_accs_idrnn, bal_accs_vanilla)):
#     xi = x[0] + jitter[i]
#     xv = x[1] + jitter[i]
#     ax.plot([xi, xv], [ba_i, ba_v], color="gray", lw=0.8, alpha=0.5, zorder=3)
#     ax.scatter([xi], [ba_i], color=cols[0], s=30, zorder=4, edgecolors="k", linewidths=0.4)
#     ax.scatter([xv], [ba_v], color=cols[1], s=30, zorder=4, edgecolors="k", linewidths=0.4)

# # Chance line
# ax.axhline(chance, color="k", ls=":", lw=1, alpha=0.6, label=f"Chance ({chance:.2f})")

# # Significance bracket
# p_use = p_wilcoxon
# star  = "***" if p_use < 0.001 else "**" if p_use < 0.01 else "*" if p_use < 0.05 else "n.s."
# y_top = max(mean_idrnn + sem_idrnn, mean_vanilla + sem_vanilla) + 0.04
# ax.plot([x[0], x[0], x[1], x[1]], [y_top - 0.01, y_top, y_top, y_top - 0.01],
#         color="k", lw=1)
# ax.text(0.5, y_top + 0.005, f"{star}\nWilcoxon p={p_wilcoxon:.3f}",
#         ha="center", va="bottom", fontsize=9)

# ax.set_xticks(x)
# ax.set_xticklabels(labels_bar, fontsize=11)
# ax.set_ylabel("Balanced Accuracy (LOOCV)", fontsize=11)
# ax.set_title(f"Age-group decoding from latents\n(n={n_seeds} seeds)", fontsize=11)
# ax.set_ylim(0, y_top + 0.07)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.legend(fontsize=9, loc="upper right")

# fig.tight_layout()
# out = os.path.join(PLOT_DIR, "logistic_decoding_cv_seeds.png")
# fig.savefig(out, dpi=150, bbox_inches="tight")
# print(f"\nSaved → {out}")

# ── Train→test RBF-SVM plot ───────────────────────────────────────────────────
chance = 1.0 / len(np.unique(labels))
x      = np.array([0, 1])
cols   = ["#4C72B0", "#DD8452"]
means  = [mean_rbfsvm_traintest_idrnn, mean_rbfsvm_traintest_vanilla]
sems   = [sem_rbfsvm_traintest_idrnn,  sem_rbfsvm_traintest_vanilla]

fig, ax = plt.subplots(figsize=(4.5, 5))

ax.bar(x, means, yerr=sems, color=cols, edgecolor="k",
       width=0.5, capsize=5, zorder=2)

rng    = np.random.default_rng(0)
jitter = rng.uniform(-0.06, 0.06, n_seeds)
for i, (ba_i, ba_v) in enumerate(zip(bal_accs_rbf_svm_traintest_idrnn, bal_accs_rbf_svm_traintest_vanilla)):
    xi = x[0] + jitter[i]
    xv = x[1] + jitter[i]
    ax.plot([xi, xv], [ba_i, ba_v], color="gray", lw=0.8, alpha=0.5, zorder=3)
    ax.scatter([xi], [ba_i], color=cols[0], s=30, zorder=4, edgecolors="k", linewidths=0.4)
    ax.scatter([xv], [ba_v], color=cols[1], s=30, zorder=4, edgecolors="k", linewidths=0.4)

ax.axhline(chance, color="k", ls=":", lw=1, alpha=0.6, label=f"Chance ({chance:.2f})")

star  = "***" if p_ttest_rbfsvm_traintest < 0.001 else "**" if p_ttest_rbfsvm_traintest < 0.01 else "*" if p_ttest_rbfsvm_traintest < 0.05 else "n.s."
y_top = max(means[0] + sems[0], means[1] + sems[1]) + 0.04
ax.plot([x[0], x[0], x[1], x[1]], [y_top - 0.01, y_top, y_top, y_top - 0.01], color="k", lw=1)
ax.text(0.5, y_top + 0.005, f"{star}\nt-test p={p_ttest_rbfsvm_traintest:.3f}",
        ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["IDRNN", "Vanilla"], fontsize=11)
ax.set_ylabel("Balanced Accuracy (train→test)", fontsize=11)
ax.set_title(f"Age-group decoding from latents\n(n={n_seeds} seeds)", fontsize=11)
ax.set_ylim(0, y_top + 0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, loc="upper right")

fig.tight_layout()
out = os.path.join(PLOT_DIR, "rbfsvm_traintest_decoding_cv_seeds.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
