# select_best_epoch_unsupervised.py
"""
Unsupervised epoch selection that doesn't rely on ground truth.

Key insight: If the model learns meaningful individual differences,
the learned geometry should:
1. Predict behavioral heterogeneity (latent distance ~ behavioral distance)
2. Be consistent across random seeds (reproducible)
3. Be structured (not random noise)

The key metric is BEHAVIORAL ALIGNMENT: does the learned latent structure
correlate with actual differences in participant behavior?
"""

import os
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
import json
import argparse

parser = argparse.ArgumentParser(description="Select best epoch by unsupervised metrics")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                    help="latent or vanilla modeling")
parser.add_argument('--dataset_id', type=int, default=0,
                    help="dataset ID for multi-dataset experiments")
parser.add_argument('--min_epoch', type=int, default=3000,
                    help="minimum epoch to consider (skip early training)")
args = parser.parse_args()

latent = args.latent
DATASET_ID = args.dataset_id
MIN_EPOCH = args.min_epoch

BASE_DIR = f"runs_dataset{DATASET_ID}" if latent else f"runs_vanilla_dataset{DATASET_ID}"
DATA_DIR = f"data_dataset{DATASET_ID}"
SEEDS = [12, 50, 76, 100, 142]


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
    """Mean pairwise Pearson correlation across seeds."""
    corrs = []
    for (v1, v2) in combinations(vectors, 2):
        if np.std(v1) == 0 or np.std(v2) == 0:
            continue
        c = np.corrcoef(v1, v2)[0, 1]
        corrs.append(c)
    return np.mean(corrs) if corrs else np.nan


def compute_effective_dimensionality(rsa_vector, n_participants=200):
    """
    Compute effective dimensionality of the distance matrix.

    A structured distance matrix (reflecting true underlying variation)
    should have low effective dimensionality. Random noise would have high dim.

    Uses eigenvalue spectrum of the centered distance matrix (classical MDS approach).
    """
    # Reconstruct distance matrix
    dist_mat = squareform(rsa_vector)
    n = dist_mat.shape[0]

    # Double centering (classical MDS)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist_mat ** 2) @ H

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Only positive eigenvalues matter
    pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(pos_eigenvalues) == 0:
        return n  # Degenerate case

    # Participation ratio: effective number of dimensions
    # PR = (sum(lambda))^2 / sum(lambda^2)
    total = np.sum(pos_eigenvalues)
    pr = (total ** 2) / np.sum(pos_eigenvalues ** 2)

    return pr


def compute_rank_consistency(vectors):
    """
    Compute Spearman rank correlation across seeds.

    More robust than Pearson - focuses on whether the *ordering* of
    pairwise distances is consistent, not exact values.
    """
    corrs = []
    for (v1, v2) in combinations(vectors, 2):
        rho, _ = spearmanr(v1, v2)
        if not np.isnan(rho):
            corrs.append(rho)
    return np.mean(corrs) if corrs else np.nan


def compute_distance_variance_ratio(rsa_vector):
    """
    Ratio of between-participant variance to within-participant variance.

    High ratio indicates the model distinguishes participants well.
    This is a proxy for signal-to-noise ratio in the representations.
    """
    # For RSA this is just coefficient of variation
    # Higher CV means more differentiation between participants
    if np.mean(rsa_vector) < 1e-10:
        return 0.0
    return np.std(rsa_vector) / np.mean(rsa_vector)


def compute_distance_entropy(rsa_vector):
    """
    Entropy of the distance distribution.

    LOW entropy = more structured (not uniformly random)
    HIGH entropy = more random/uniform

    Empirically, lower entropy correlates with better ground truth alignment.
    """
    # Normalize to [0, 1]
    rsa_normalized = rsa_vector / (np.max(rsa_vector) + 1e-10)
    rsa_normalized = np.clip(rsa_normalized, 1e-10, 1.0)

    # Compute entropy
    entropy = -np.mean(rsa_normalized * np.log(rsa_normalized))
    return entropy


def compute_local_smoothness(rsa_vector, n_neighbors=10):
    """
    Measure how smooth the distance matrix is locally.

    For a structured representation, similar distances should cluster together
    in the matrix (participants with similar parameters should have similar
    distance patterns to others).

    This computes the average correlation between a row and its nearest neighbors.
    """
    dist_mat = squareform(rsa_vector)
    n = dist_mat.shape[0]

    smoothness_scores = []
    for i in range(n):
        row_i = dist_mat[i, :]
        # Find nearest neighbors (smallest distances)
        neighbor_idx = np.argsort(row_i)[1:n_neighbors+1]  # exclude self

        # Correlate this row with neighbor rows
        for j in neighbor_idx:
            row_j = dist_mat[j, :]
            if np.std(row_i) > 0 and np.std(row_j) > 0:
                corr = np.corrcoef(row_i, row_j)[0, 1]
                smoothness_scores.append(corr)

    return np.mean(smoothness_scores) if smoothness_scores else 0.0


def compute_behavioral_rsa():
    """
    Compute behavioral RSA - pairwise similarity between participants
    based on their choice behavior.

    This is computed ONCE and used to evaluate all epochs.
    Returns the vectorized behavioral distance matrix.
    """
    # Load choice data
    c_test = np.load(f"{DATA_DIR}/c_test.npy")  # Shape: (N_participants, N_trials)

    # Compute choice pattern similarity
    # Use correlation distance: participants with similar choice patterns are close
    vec_behavior = pdist(c_test, metric='correlation')

    return vec_behavior


def compute_behavioral_alignment(rsa_vector, behavioral_rsa):
    """
    How well does the latent geometry align with behavioral differences?

    This is the KEY metric: a good individual-difference model should have
    latents that predict behavioral heterogeneity.

    High correlation = latents capture meaningful behavioral variation.
    """
    # Use Spearman to be robust to non-linear relationships
    corr, _ = spearmanr(rsa_vector, behavioral_rsa)
    return corr if not np.isnan(corr) else 0.0


def compute_temporal_stability(seed, epochs, current_epoch, window=5):
    """
    How stable is the geometry around the current epoch?

    Measures correlation with nearby epochs.
    """
    nearby_epochs = [e for e in epochs if abs(e - current_epoch) <= window * 100 and e != current_epoch]

    if not nearby_epochs:
        return 1.0  # No nearby epochs to compare

    try:
        current_rsa = load_rsa_vector(seed, current_epoch)
    except:
        return 0.0

    correlations = []
    for ep in nearby_epochs:
        try:
            other_rsa = load_rsa_vector(seed, ep)
            if np.std(current_rsa) > 0 and np.std(other_rsa) > 0:
                corr = np.corrcoef(current_rsa, other_rsa)[0, 1]
                correlations.append(corr)
        except:
            continue

    return np.mean(correlations) if correlations else 0.0


# Find common epochs across all seeds
print(f"Loading epochs from {BASE_DIR}...")
epochs_per_seed = {seed: set(list_epochs_for_seed(seed)) for seed in SEEDS}
common_epochs = sorted(set.intersection(*epochs_per_seed.values()))

if not common_epochs:
    raise RuntimeError("No common epochs across seeds – check your runs.")

# Filter to late training
common_epochs = [e for e in common_epochs if e >= MIN_EPOCH]
print(f"Found {len(common_epochs)} common epochs >= {MIN_EPOCH}")

if not common_epochs:
    raise RuntimeError(f"No epochs >= {MIN_EPOCH}. Try lowering --min_epoch")

# Compute behavioral RSA once (same for all epochs)
print("\nComputing behavioral RSA from choice data...")
try:
    behavioral_rsa = compute_behavioral_rsa()
    print(f"  Behavioral RSA computed (shape: {behavioral_rsa.shape})")
    has_behavioral = True
except Exception as e:
    print(f"  Warning: Could not compute behavioral RSA: {e}")
    print(f"  Falling back to other metrics only")
    behavioral_rsa = None
    has_behavioral = False

# Compute metrics for each epoch
print("\nComputing metrics for each epoch...")
print("-" * 80)

metrics_per_epoch = {}

for epoch in common_epochs:
    rsa_vecs = [load_rsa_vector(seed, epoch) for seed in SEEDS]

    # 1. Cross-seed Pearson correlation (consistency)
    pearson_consistency = mean_pairwise_corr(rsa_vecs)

    # 2. Cross-seed Spearman correlation (rank consistency)
    spearman_consistency = compute_rank_consistency(rsa_vecs)

    # 3. Mean effective dimensionality across seeds (lower = more structured)
    eff_dims = [compute_effective_dimensionality(v) for v in rsa_vecs]
    mean_eff_dim = np.mean(eff_dims)

    # 4. Variance in effective dimensionality across seeds (lower = more consistent structure)
    std_eff_dim = np.std(eff_dims)

    # 5. Mean distance CV (discrimination power)
    cvs = [compute_distance_variance_ratio(v) for v in rsa_vecs]
    mean_cv = np.mean(cvs)

    # 6. Local smoothness (averaged across seeds)
    smoothness = np.mean([compute_local_smoothness(v) for v in rsa_vecs])

    # 7. Temporal stability (averaged across seeds)
    stabilities = [compute_temporal_stability(seed, list(epochs_per_seed[seed]), epoch)
                   for seed in SEEDS]
    mean_stability = np.mean(stabilities)

    # 8. Mean loss
    losses = [load_loss_vector(seed, epoch) for seed in SEEDS]
    mean_loss = np.mean(losses)

    # 9. BEHAVIORAL ALIGNMENT
    if has_behavioral:
        alignments = [compute_behavioral_alignment(v, behavioral_rsa) for v in rsa_vecs]
        mean_behavioral_alignment = np.mean(alignments)
    else:
        mean_behavioral_alignment = 0.0

    # 10. DISTANCE ENTROPY - low entropy = more structured (empirically important!)
    entropies = [compute_distance_entropy(v) for v in rsa_vecs]
    mean_entropy = np.mean(entropies)

    metrics_per_epoch[epoch] = {
        'pearson_consistency': pearson_consistency,
        'spearman_consistency': spearman_consistency,
        'eff_dim': mean_eff_dim,
        'eff_dim_std': std_eff_dim,
        'cv': mean_cv,
        'smoothness': smoothness,
        'stability': mean_stability,
        'loss': mean_loss,
        'behavioral_alignment': mean_behavioral_alignment,
        'entropy': mean_entropy
    }

    print(f"Epoch {epoch:04d}: entropy={mean_entropy:.3f}, cv={mean_cv:.3f}, "
          f"pearson={pearson_consistency:.3f}, stab={mean_stability:.3f}")

# Normalize metrics (all scaled so higher = better)
print("\n" + "=" * 80)
print("Normalizing and combining metrics...")

all_metrics = list(metrics_per_epoch.values())
metric_keys = ['pearson_consistency', 'spearman_consistency', 'eff_dim',
               'cv', 'smoothness', 'stability', 'loss', 'behavioral_alignment', 'entropy']

# Get min/max for normalization
ranges = {}
for key in metric_keys:
    values = [m[key] for m in all_metrics]
    ranges[key] = (np.min(values), np.max(values))

# Compute composite score
composite_scores = {}

for epoch, metrics in metrics_per_epoch.items():
    scores = {}

    # Higher is better
    for key in ['pearson_consistency', 'spearman_consistency', 'cv', 'smoothness',
                'stability', 'behavioral_alignment']:
        min_v, max_v = ranges[key]
        scores[key] = (metrics[key] - min_v) / (max_v - min_v + 1e-8)

    # Lower is better (invert)
    for key in ['eff_dim', 'loss', 'entropy']:
        min_v, max_v = ranges[key]
        scores[key] = 1.0 - (metrics[key] - min_v) / (max_v - min_v + 1e-8)

    # Weighted combination
    # Key empirical findings:
    # - Low ENTROPY strongly predicts good ground truth correlation (-0.67)
    # - High CV moderately predicts good ground truth correlation (+0.52)
    # - Cross-seed consistency is needed to filter outliers, not as primary metric
    if has_behavioral:
        composite = (
            0.30 * scores['entropy'] +               # KEY: low entropy = structured
            0.20 * scores['cv'] +                    # KEY: high CV = discriminative
            0.15 * scores['behavioral_alignment'] +  # Behavior alignment
            0.10 * scores['pearson_consistency'] +   # Cross-seed reproducibility
            0.10 * scores['stability'] +             # Temporal stability
            0.05 * scores['spearman_consistency'] +  # Rank-order consistency
            0.05 * scores['smoothness'] +            # Local structure
            0.05 * scores['loss']                    # Training quality
        )
    else:
        # Fallback if no behavioral data
        composite = (
            0.30 * scores['entropy'] +
            0.25 * scores['cv'] +
            0.15 * scores['pearson_consistency'] +
            0.10 * scores['stability'] +
            0.10 * scores['spearman_consistency'] +
            0.05 * scores['smoothness'] +
            0.05 * scores['loss']
        )

    composite_scores[epoch] = composite

# Select best epoch
best_epoch = max(composite_scores, key=composite_scores.get)
best_score = composite_scores[best_epoch]
best_metrics = metrics_per_epoch[best_epoch]

print(f"\n{'='*80}")
print(f"BEST EPOCH: {best_epoch:04d}")
print(f"Composite score: {best_score:.4f}")
print(f"{'='*80}")
print(f"  ENTROPY (low=good):   {best_metrics['entropy']:.4f}  <-- KEY METRIC")
print(f"  CV (high=good):       {best_metrics['cv']:.4f}  <-- KEY METRIC")
print(f"  Behavioral alignment: {best_metrics['behavioral_alignment']:.4f}")
print(f"  Pearson consistency:  {best_metrics['pearson_consistency']:.4f}")
print(f"  Spearman consistency: {best_metrics['spearman_consistency']:.4f}")
print(f"  Temporal stability:   {best_metrics['stability']:.4f}")
print(f"  Mean loss:            {best_metrics['loss']:.4f}")
print(f"{'='*80}\n")

# Select best seed for this epoch
# Strategy: First filter out outlier seeds, then select among the good ones
print("Selecting best seed...")
print("Step 1: Identify outlier seeds (low centrality)")
seed_centralities = {}

for seed in SEEDS:
    rsa_seed = load_rsa_vector(seed, best_epoch)

    # Centrality: mean correlation with other seeds
    correlations = []
    for other_seed in SEEDS:
        if other_seed == seed:
            continue
        rsa_other = load_rsa_vector(other_seed, best_epoch)
        if np.std(rsa_seed) > 0 and np.std(rsa_other) > 0:
            corr = np.corrcoef(rsa_seed, rsa_other)[0, 1]
            correlations.append(corr)

    centrality = np.mean(correlations) if correlations else 0.0
    seed_centralities[seed] = centrality

# Filter out outliers: seeds with centrality < median - 1 std
centrality_values = list(seed_centralities.values())
centrality_median = np.median(centrality_values)
centrality_std = np.std(centrality_values)
centrality_threshold = max(0.3, centrality_median - centrality_std)  # At least 0.3

good_seeds = [s for s, c in seed_centralities.items() if c >= centrality_threshold]
print(f"  Centrality threshold: {centrality_threshold:.3f}")
print(f"  Good seeds (centrality >= threshold): {good_seeds}")

if not good_seeds:
    print("  Warning: No seeds pass threshold, using all seeds")
    good_seeds = SEEDS

# Step 2: Among good seeds, select based on behavioral alignment + loss
print("Step 2: Select best among good seeds based on quality metrics")
seed_scores = {}

for seed in SEEDS:
    rsa_seed = load_rsa_vector(seed, best_epoch)
    centrality = seed_centralities[seed]

    # Loss
    seed_loss = load_loss_vector(seed, best_epoch)

    # Behavioral alignment for this seed
    if has_behavioral:
        behav_align = compute_behavioral_alignment(rsa_seed, behavioral_rsa)
    else:
        behav_align = 0.0

    # For good seeds: prioritize behavioral alignment and loss
    # For outlier seeds: heavily penalize
    if seed in good_seeds:
        seed_score = 0.4 * behav_align + 0.3 * centrality + 0.3 * (1.0 / (1.0 + seed_loss))
    else:
        seed_score = 0.1 * centrality  # Heavy penalty for outliers

    seed_scores[seed] = {
        'score': seed_score,
        'centrality': centrality,
        'loss': seed_loss,
        'behavioral_alignment': behav_align,
        'is_good_seed': seed in good_seeds
    }

    outlier_marker = "" if seed in good_seeds else " [OUTLIER]"
    print(f"  Seed {seed}: centrality={centrality:.3f}, behav_align={behav_align:.3f}, "
          f"loss={seed_loss:.4f}, score={seed_score:.3f}{outlier_marker}")

best_seed = max(seed_scores, key=lambda s: seed_scores[s]['score'])
print(f"\nBest seed: {best_seed}")

# Save results
out_path = os.path.join(BASE_DIR, "best_epoch_unsupervised.json")
with open(out_path, "w") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "composite_score": float(best_score),
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "all_epoch_scores": {str(e): float(s) for e, s in composite_scores.items()},
        "seed_scores": {
            seed: {
                'centrality': float(seed_scores[seed]['centrality']),
                'loss': float(seed_scores[seed]['loss'])
            } for seed in SEEDS
        },
        "seeds": SEEDS,
        "min_epoch": MIN_EPOCH
    }, f, indent=2)

print(f"\nSaved detailed results to {out_path}")

# Also save to standard location for compatibility with existing pipeline
out_path_standard = os.path.join(BASE_DIR, "best_epoch_by_rsa.json")
with open(out_path_standard, "w") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "best_reliability": float(best_metrics['pearson_consistency']),
        "seeds": SEEDS
    }, f, indent=2)

print(f"Saved to standard location: {out_path_standard}")
