#!/usr/bin/env python3
"""
Select best epoch using validation/test loss only.

Simple selection criterion for vanilla models or when ground truth is unavailable.
Selects the epoch with the lowest mean loss across seeds.
"""

import os
import numpy as np
import json
import argparse

# Epoch window configuration
MAX_EPOCH = 1000  # Epoch cutoff to prevent overtraining
MIN_EPOCH = 100  # Minimum epoch to consider

parser = argparse.ArgumentParser(description="Select best epoch by loss")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                    help="latent or vanilla modeling")
parser.add_argument('--dataset_id', type=int, default=0,
                    help="dataset ID for multi-dataset experiments")
parser.add_argument('--min_epoch', type=int, default=MIN_EPOCH,
                    help="minimum epoch to consider")
parser.add_argument('--max_epoch', type=int, default=MAX_EPOCH,
                    help="maximum epoch to consider")
parser.add_argument('--dgp', type=str, default=None,
                    help="Data generating process type (e.g., 'sloutsky', 'uniform').")
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

# Detect available seeds by looking at which seed_* directories exist
available_seeds = []
if os.path.exists(BASE_DIR):
    for d in os.listdir(BASE_DIR):
        if d.startswith("seed_"):
            try:
                seed_num = int(d.split("_")[1])
                available_seeds.append(seed_num)
            except ValueError:
                pass
SEEDS = sorted(available_seeds)
if not SEEDS:
    raise RuntimeError(f"No seed directories found in {BASE_DIR}")

model_type = "IDRNN" if latent else "Vanilla"
print(f"\n{'='*60}")
print(f"EPOCH SELECTION BY LOSS")
print(f"Model: {model_type}")
print(f"Dataset: {DATASET_ID}")
print(f"DGP: {DGP if DGP else 'default'}")
print(f"Epoch window: [{MIN_EPOCH}, {MAX_EPOCH}]")
print(f"{'='*60}\n")


def list_epochs_for_seed(seed):
    loss_dir = os.path.join(BASE_DIR, f"seed_{seed}", "loss")
    if not os.path.exists(loss_dir):
        return []
    files = [f for f in os.listdir(loss_dir) if f.startswith("epoch_") and f.endswith(".npy")]
    epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
    return sorted(epochs)


def load_loss_vector(seed, epoch):
    loss_path = os.path.join(BASE_DIR, f"seed_{seed}", "loss", f"epoch_{epoch:04d}.npy")
    return np.load(loss_path)


# Find common epochs across all seeds within the specified window
epochs_per_seed = {seed: set(list_epochs_for_seed(seed)) for seed in SEEDS}
common_epochs = sorted(set.intersection(*epochs_per_seed.values()))
common_epochs = [e for e in common_epochs if e >= MIN_EPOCH and (MAX_EPOCH is None or e <= MAX_EPOCH)]

print(f"Found {len(common_epochs)} common epochs in window [{MIN_EPOCH}, {MAX_EPOCH}]")

if not common_epochs:
    raise RuntimeError(f"No common epochs across seeds in window [{MIN_EPOCH}, {MAX_EPOCH}] – check your runs.")

# Compute mean loss for each epoch
mean_loss_per_epoch = {}
seed_losses = {}

for epoch in common_epochs:
    losses = [load_loss_vector(seed, epoch) for seed in SEEDS]
    mean_loss = np.mean(losses)
    mean_loss_per_epoch[epoch] = mean_loss
    seed_losses[epoch] = {seed: float(losses[i]) for i, seed in enumerate(SEEDS)}
    print(f"Epoch {epoch:04d}: mean loss = {mean_loss:.4f}")

# Find best epoch (lowest loss)
best_epoch = min(mean_loss_per_epoch, key=mean_loss_per_epoch.get)
best_loss = mean_loss_per_epoch[best_epoch]

# Find best seed for this epoch
best_seed_for_epoch = min(seed_losses[best_epoch], key=seed_losses[best_epoch].get)

print(f"\n{'='*60}")
print(f"BEST EPOCH: {best_epoch}")
print(f"Mean Loss: {best_loss:.4f}")
print(f"Best Seed: {best_seed_for_epoch}")
print(f"{'='*60}\n")

# Save results
result = {
    "best_epoch": best_epoch,
    "best_seed": best_seed_for_epoch,
    "best_loss": float(best_loss),
    "selection_method": "loss",
    "epoch_window": [MIN_EPOCH, MAX_EPOCH],
    "all_epochs": {str(e): float(l) for e, l in mean_loss_per_epoch.items()},
    "seed_details": {str(e): v for e, v in seed_losses.items()}
}

output_path = os.path.join(BASE_DIR, "best_epoch_by_loss.json")
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved results to {output_path}")
