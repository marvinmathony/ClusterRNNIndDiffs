# select_best_epoch_by_rsa.py
import os
import numpy as np
from itertools import combinations
import json
import argparse

# Epoch window configuration (must match analyze_synthetic_multi_dataset.py)
MAX_EPOCH = 3000  # Epoch cutoff to prevent overtraining; set to None for no cutoff
MIN_EPOCH = 1000  # Minimum epoch to consider (allow some initial training)

parser = argparse.ArgumentParser(description="Select best epoch by RSA")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True, help="latent or vanilla modeling")
parser.add_argument('--dataset_id', type=int, default=0, help="dataset ID for multi-dataset experiments")
parser.add_argument('--min_epoch', type=int, default=MIN_EPOCH, help="minimum epoch to consider")
parser.add_argument('--max_epoch', type=int, default=MAX_EPOCH, help="maximum epoch to consider")
parser.add_argument('--dgp', type=str, default=None,
                    help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
args = parser.parse_args()

latent = args.latent
DATASET_ID = args.dataset_id
MIN_EPOCH = args.min_epoch
MAX_EPOCH = args.max_epoch
DGP = args.dgp

# Determine if this is human data (sloutsky/palminteri/spatial_bandit)
is_human_data = DGP in ("sloutsky", "palminteri", "spatial_bandit")

# Build runs directory with optional DGP prefix
# Human data (sloutsky/palminteri) doesn't use dataset IDs
if is_human_data:
    BASE_DIR = f"runs_{DGP}" if latent else f"runs_vanilla_{DGP}"
elif DGP:
    BASE_DIR = f"runs_{DGP}_dataset{DATASET_ID}" if latent else f"runs_vanilla_{DGP}_dataset{DATASET_ID}"
else:
    BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"
SEEDS = [12,50,76,100,142]  # or read from args

print(f"Selecting best epoch in window [{MIN_EPOCH}, {MAX_EPOCH}]")

def list_epochs_for_seed(seed):
    rsa_dir = os.path.join(BASE_DIR, f"seed_{seed}", "rsa")
    files = [f for f in os.listdir(rsa_dir) if f.startswith("epoch_") and f.endswith(".npy")]
    epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
    return sorted(epochs)

# 1. Find common epochs across all seeds within the specified window
epochs_per_seed = {seed: set(list_epochs_for_seed(seed)) for seed in SEEDS}
common_epochs = sorted(set.intersection(*epochs_per_seed.values()))

# Filter to epoch window
common_epochs = [e for e in common_epochs if e >= MIN_EPOCH and (MAX_EPOCH is None or e <= MAX_EPOCH)]
print(f"Found {len(common_epochs)} common epochs in window [{MIN_EPOCH}, {MAX_EPOCH}]")

if not common_epochs:
    raise RuntimeError(f"No common epochs across seeds in window [{MIN_EPOCH}, {MAX_EPOCH}] – check your runs.")

def load_rsa_vector(seed, epoch):
    rsa_path = os.path.join(BASE_DIR, f"seed_{seed}", "rsa", f"epoch_{epoch:04d}.npy")
    return np.load(rsa_path)

def load_loss_vector(seed, epoch):
    loss_path = os.path.join(BASE_DIR, f"seed_{seed}", "loss", f"epoch_{epoch:04d}.npy")
    return np.load(loss_path)

def mean_pairwise_corr(vectors):
    """
    vectors: list of 1D arrays, one per seed
    """
    corrs = []
    for (v1, v2) in combinations(vectors, 2):
        # Protect against NaNs
        if np.std(v1) == 0 or np.std(v2) == 0:
            continue
        c = np.corrcoef(v1, v2)[0, 1]
        corrs.append(c)
    return np.mean(corrs) if corrs else np.nan



reliability_per_epoch = {}
mean_loss_per_epoch = {}

for epoch in common_epochs:
    rsa_vecs = [load_rsa_vector(seed, epoch) for seed in SEEDS]
    rel = mean_pairwise_corr(rsa_vecs)
    reliability_per_epoch[epoch] = rel
    print(f"Epoch {epoch:04d}: mean cross-seed RSA corr = {rel:.3f}")
    losses = [load_loss_vector(seed, epoch) for seed in SEEDS]
    #now I have losses across seeds for a given epoch
    mean_loss = np.mean(losses)
    mean_loss_per_epoch[epoch] = mean_loss
    print(f"Epoch {epoch:04d}: mean loss = {mean_loss:.3f}")




##### sort first based on RSA, then on loss #####
# Sort epochs by RSA (descending) - epochs already filtered to window
top5_epochs_dict = sorted(
    reliability_per_epoch.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

top5_epochs_dict = dict(top5_epochs_dict)
top5_epochs_only = list(top5_epochs_dict.keys())
top5_rsa_values  = list(top5_epochs_dict.values())

print(f"top 5 epochs sorted by rsa in descending order: {top5_rsa_values}")

mean_loss_per_epoch_top5 = {}
# select epoch with lowest loss of those
for epoch in top5_epochs_only:
    losses_top5_per_epoch = [load_loss_vector(seed, epoch) for seed in SEEDS] # losses across seed for given epoch
    mean_loss_top5 = np.mean(losses_top5_per_epoch)
    mean_loss_per_epoch_top5[epoch] = mean_loss_top5
# minumum of the 5
best_epoch = min(mean_loss_per_epoch_top5, key=mean_loss_per_epoch_top5.get)
best_reliability = top5_epochs_dict[best_epoch]



##### sort first based on loss, then RSA #####
# choose top 5 epochs based on train loss
# top_5_best_loss_epochs = []
# mean_loss_per_epoch_copy = mean_loss_per_epoch.copy()
# for i in range(5):
#     best_loss_epoch = min(mean_loss_per_epoch_copy, key=mean_loss_per_epoch_copy.get)
#     top_5_best_loss_epochs.append(best_loss_epoch)
#     mean_loss_per_epoch_copy[best_loss_epoch] = np.inf
#     #top_5_best_loss_epochs[i.toString()] = mean_loss_per_epoch_copy[best_loss_epoch]

# # 2. Choose the epoch with max reliability
# reliability_after_500 = {e: r for e, r in reliability_per_epoch.items() if e > 1}
# threshold = 1
# reliability_top5 = {e: reliability_per_epoch[e]
#                     for e in top_5_best_loss_epochs
#                     if e in reliability_per_epoch and e > threshold}
# best_epoch = max(reliability_top5, key=reliability_top5.get)
# best_reliability = reliability_top5[best_epoch]


# now I want the exact seed that we should use
min_loss = np.inf
for seed in SEEDS:
    loss = load_loss_vector(seed, best_epoch)
    if loss < min_loss:
        min_loss = loss
        best_seed = seed
print(f"\nBest epoch by loss and RSA reliability: {best_epoch:04d} (mean corr = {best_reliability:.3f}, min loss = {min_loss} in seed: {best_seed})")

# save result to a small JSON or text file
out_path = os.path.join(BASE_DIR, "best_epoch_by_rsa.json")
with open(out_path, "w") as f:
    json.dump({"best_epoch": best_epoch,
               "best_reliability": best_reliability,
               "seeds": SEEDS,
               "best_seed": best_seed}, f, indent=2)

print(f"Saved selection to {out_path}")
