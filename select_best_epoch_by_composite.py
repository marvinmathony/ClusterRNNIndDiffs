# select_best_epoch_by_composite.py
"""
Enhanced epoch selection using multiple unsupervised quality metrics.
This version doesn't require ground truth and should be more robust.
"""

import os
import numpy as np
from itertools import combinations
import json
import argparse
import torch

# Epoch window configuration (must match analyze_synthetic_multi_dataset.py)
DEFAULT_MIN_EPOCH = 1000  # Minimum epoch to consider
DEFAULT_MAX_EPOCH = 3000  # Maximum epoch to consider (prevents overtraining)

parser = argparse.ArgumentParser(description="Select best epoch by composite score")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True, help="latent or vanilla modeling")
parser.add_argument('--dataset_id', type=int, default=0, help="dataset ID for multi-dataset experiments")
parser.add_argument('--min_epoch', type=int, default=DEFAULT_MIN_EPOCH, help="minimum epoch to consider")
parser.add_argument('--max_epoch', type=int, default=DEFAULT_MAX_EPOCH, help="maximum epoch to consider")
args = parser.parse_args()

latent = args.latent
DATASET_ID = args.dataset_id
MIN_EPOCH = args.min_epoch
MAX_EPOCH = args.max_epoch

BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"
SEEDS = [12, 50, 76, 100, 142]

print(f"\n{'='*60}")
print(f"EPOCH SELECTION BY COMPOSITE SCORE")
print(f"Model: {'IDRNN' if latent else 'Vanilla'}")
print(f"Dataset: {DATASET_ID}")
print(f"Epoch window: [{MIN_EPOCH}, {MAX_EPOCH}]")
print(f"{'='*60}\n")

def list_epochs_for_seed(seed):
    rsa_dir = os.path.join(BASE_DIR, f"seed_{seed}", "rsa")
    files = [f for f in os.listdir(rsa_dir) if f.startswith("epoch_") and f.endswith(".npy")]
    epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
    return sorted(epochs)

def load_rsa_vector(seed, epoch):
    rsa_path = os.path.join(BASE_DIR, f"seed_{seed}", "rsa", f"epoch_{epoch:04d}.npy")
    return np.load(rsa_path)

def load_loss_vector(seed, epoch):
    loss_path = os.path.join(BASE_DIR, f"seed_{seed}", "loss", f"epoch_{epoch:04d}.npy")
    return np.load(loss_path)

def mean_pairwise_corr(vectors):
    """Mean pairwise correlation across seeds."""
    corrs = []
    for (v1, v2) in combinations(vectors, 2):
        if np.std(v1) == 0 or np.std(v2) == 0:
            continue
        c = np.corrcoef(v1, v2)[0, 1]
        corrs.append(c)
    return np.mean(corrs) if corrs else np.nan

def compute_geometry_stability(seed, epochs, window_size=50):
    """
    Measures how stable the RSA geometry is in late training.
    Returns mean correlation between consecutive epochs in the late window.
    """
    if len(epochs) < window_size:
        window_size = len(epochs) // 2

    late_epochs = sorted(epochs)[-window_size:]

    try:
        rsa_vectors = [load_rsa_vector(seed, ep) for ep in late_epochs]
    except:
        return 0.0

    # Compute consecutive correlations
    stability_scores = []
    for i in range(len(rsa_vectors) - 1):
        if np.std(rsa_vectors[i]) > 0 and np.std(rsa_vectors[i+1]) > 0:
            corr = np.corrcoef(rsa_vectors[i], rsa_vectors[i+1])[0, 1]
            stability_scores.append(corr)

    return np.mean(stability_scores) if stability_scores else 0.0

def compute_seed_centrality(seed, epoch, all_seeds):
    """
    How similar is this seed's geometry to all other seeds?
    Higher = more canonical/representative.
    """
    rsa_i = load_rsa_vector(seed, epoch)

    correlations = []
    for other_seed in all_seeds:
        if other_seed == seed:
            continue
        try:
            rsa_j = load_rsa_vector(other_seed, epoch)
            if np.std(rsa_i) > 0 and np.std(rsa_j) > 0:
                corr = np.corrcoef(rsa_i, rsa_j)[0, 1]
                correlations.append(corr)
        except:
            continue

    return np.mean(correlations) if correlations else 0.0

# Find common epochs
epochs_per_seed = {seed: set(list_epochs_for_seed(seed)) for seed in SEEDS}
common_epochs = sorted(set.intersection(*epochs_per_seed.values()))

# Apply epoch window filter
common_epochs = [e for e in common_epochs if e >= MIN_EPOCH and e <= MAX_EPOCH]

if not common_epochs:
    raise RuntimeError(f"No common epochs across seeds in window [{MIN_EPOCH}, {MAX_EPOCH}] – check your runs.")

print(f"Found {len(common_epochs)} common epochs in window [{MIN_EPOCH}, {MAX_EPOCH}]")

# Compute metrics for each epoch
scores_per_epoch = {}

for epoch in common_epochs:
    # 1. Cross-seed RSA reliability
    rsa_vecs = [load_rsa_vector(seed, epoch) for seed in SEEDS]
    rsa_reliability = mean_pairwise_corr(rsa_vecs)

    # 2. Mean loss across seeds
    losses = [load_loss_vector(seed, epoch) for seed in SEEDS]
    mean_loss = np.mean(losses)

    # 3. Geometry stability (average across seeds)
    stabilities = []
    for seed in SEEDS:
        stability = compute_geometry_stability(seed, common_epochs)
        stabilities.append(stability)
    mean_stability = np.mean(stabilities)

    scores_per_epoch[epoch] = {
        'rsa_reliability': rsa_reliability,
        'mean_loss': mean_loss,
        'stability': mean_stability
    }

    print(f"Epoch {epoch:04d}: RSA_rel={rsa_reliability:.3f}, "
          f"loss={mean_loss:.3f}, stability={mean_stability:.3f}")

# Normalize metrics (0-1 scale)
all_rsa = [v['rsa_reliability'] for v in scores_per_epoch.values()]
all_loss = [v['mean_loss'] for v in scores_per_epoch.values()]
all_stab = [v['stability'] for v in scores_per_epoch.values()]

min_rsa, max_rsa = np.min(all_rsa), np.max(all_rsa)
min_loss, max_loss = np.min(all_loss), np.max(all_loss)
min_stab, max_stab = np.min(all_stab), np.max(all_stab)

composite_scores = {}
for epoch, metrics in scores_per_epoch.items():
    # Normalize (higher is better for all)
    norm_rsa = (metrics['rsa_reliability'] - min_rsa) / (max_rsa - min_rsa + 1e-8)
    norm_loss = 1.0 - (metrics['mean_loss'] - min_loss) / (max_loss - min_loss + 1e-8)  # Invert
    norm_stab = (metrics['stability'] - min_stab) / (max_stab - min_stab + 1e-8)

    # Weighted combination (you can tune these)
    composite = (
        0.4 * norm_rsa +      # Cross-seed consistency
        0.3 * norm_loss +     # Training quality
        0.3 * norm_stab       # Geometry stability
    )

    composite_scores[epoch] = composite

# Select best epoch by composite score
best_epoch = max(composite_scores, key=composite_scores.get)
best_composite_score = composite_scores[best_epoch]
best_metrics = scores_per_epoch[best_epoch]

print(f"\n{'='*60}")
print(f"Best epoch by composite score: {best_epoch:04d}")
print(f"Composite score: {best_composite_score:.3f}")
print(f"  RSA reliability: {best_metrics['rsa_reliability']:.3f}")
print(f"  Mean loss: {best_metrics['mean_loss']:.3f}")
print(f"  Stability: {best_metrics['stability']:.3f}")
print(f"{'='*60}\n")

# Now select best seed for this epoch
print("Selecting best seed...")
seed_scores = {}
for seed in SEEDS:
    # Centrality: how similar to other seeds
    centrality = compute_seed_centrality(seed, best_epoch, SEEDS)

    # Loss for this seed
    seed_loss = load_loss_vector(seed, best_epoch)

    # Combine
    seed_score = 0.7 * centrality + 0.3 * (1.0 / (1.0 + seed_loss))
    seed_scores[seed] = {
        'score': seed_score,
        'centrality': centrality,
        'loss': seed_loss
    }

    print(f"  Seed {seed}: centrality={centrality:.3f}, loss={seed_loss:.3f}, score={seed_score:.3f}")

best_seed = max(seed_scores, key=lambda s: seed_scores[s]['score'])
print(f"\nBest seed: {best_seed} (most canonical representation)")

# Save results
out_path = os.path.join(BASE_DIR, "best_epoch_by_composite.json")
with open(out_path, "w") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "composite_score": float(best_composite_score),
        "metrics": {
            "rsa_reliability": float(best_metrics['rsa_reliability']),
            "mean_loss": float(best_metrics['mean_loss']),
            "stability": float(best_metrics['stability'])
        },
        "seed_scores": {
            seed: {
                'centrality': float(seed_scores[seed]['centrality']),
                'loss': float(seed_scores[seed]['loss'])
            } for seed in SEEDS
        },
        "min_epoch": MIN_EPOCH,
        "max_epoch": MAX_EPOCH,
        "seeds": SEEDS
    }, f, indent=2)

print(f"\nSaved selection to {out_path}")

# Also save to the standard location for compatibility
out_path_standard = os.path.join(BASE_DIR, "best_epoch_by_rsa.json")
with open(out_path_standard, "w") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "best_reliability": float(best_metrics['rsa_reliability']),
        "seeds": SEEDS
    }, f, indent=2)

print(f"Saved to standard location: {out_path_standard}")

# For IDRNN models, also save to best_epoch_by_specificity.json for testing_script.py compatibility
if latent:
    out_path_specificity = os.path.join(BASE_DIR, "best_epoch_by_specificity.json")
    with open(out_path_specificity, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_seed": best_seed,
            "best_specificity": float(best_composite_score),
            "composite_score": float(best_composite_score),
            "metrics": {
                "rsa_reliability": float(best_metrics['rsa_reliability']),
                "mean_loss": float(best_metrics['mean_loss']),
                "stability": float(best_metrics['stability'])
            },
            "seeds": SEEDS
        }, f, indent=2)
    print(f"Saved IDRNN selection to: {out_path_specificity}")
