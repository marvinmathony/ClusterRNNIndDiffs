#!/usr/bin/env python3
"""
Compute reconstruction specificity metric for unsupervised model selection.

Key idea: If latent representations are specific to input sequences, using a
RANDOM latent to reconstruct a sequence should produce higher loss than using
the MATCHED latent.

The "reconstruction specificity" is defined as:
    specificity = mean(loss_mismatched) - mean(loss_matched)

During training, if the model learns meaningful individual-specific representations,
this specificity should INCREASE (i.e., using wrong latents hurts more).

This metric can be computed without ground truth and could serve as an
unsupervised selection criterion.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import json
import copy

# Import model classes
from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN,
    LookupEncoderZ, LatentRNNz, vectorize_rsa
)
import plot_functions as plf


def load_model_config(run_dir):
    """Load model configuration from config.json."""
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Return model_config if it exists, otherwise extract from top-level
        if "model_config" in config:
            return config["model_config"]
        else:
            # Fallback for older configs without nested model_config
            return {
                "in_dim": config.get("in_dim", 4),
                "z_dim": config.get("z_dim", 1),
                "hidden": config.get("hidden", 10),
                "A": config.get("A", 2),
                "block_structure": False,
                "model_type": "IDRNN" if config.get("latent", True) else "Vanilla"
            }
    return None


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_model_from_config(model_config, n_participants, device, frozen_decoder_path=None):
    """
    Create model architecture from config.

    Args:
        model_config: dict with model architecture parameters
        n_participants: number of participants (for lookup encoder)
        device: torch device
        frozen_decoder_path: path to frozen decoder weights (for IDRNN)

    Returns:
        model: the constructed model
    """
    in_dim = model_config["in_dim"]
    z_dim = model_config["z_dim"]
    hidden = model_config["hidden"]
    A = model_config["A"]
    block_structure = model_config.get("block_structure", False)
    model_type = model_config.get("model_type", "IDRNN")

    if model_type == "IDRNN" or model_type == "latent":
        if frozen_decoder_path is None:
            raise ValueError("frozen_decoder_path required for IDRNN model")

        # Load the frozen decoder from the policy model
        temp_encoder = LookupEncoderZ(n_participants=n_participants, z_dim=z_dim)
        temp_decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A)
        temp_model = LatentRNNz(
            encoder=temp_encoder,
            decoder=temp_decoder,
            hid=hidden,
            z_dim=z_dim,
            in_dim=in_dim,
            A=A,
            block_structure=block_structure
        )
        temp_model.load_state_dict(torch.load(frozen_decoder_path, map_location=device))

        # Create IDRNN model with frozen decoder
        encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
        frozen_decoder = copy.deepcopy(temp_model.decoder)
        for p in frozen_decoder.parameters():
            p.requires_grad = False

        model = LatentRNN_secondstep(
            encoder=encoder_IDRNN,
            hid=hidden,
            z_dim=z_dim,
            in_dim=in_dim,
            A=A,
            decoder=frozen_decoder
        )
    else:
        # Vanilla model
        model = AblatedRNN(hid=hidden, in_dim=in_dim, A=A, block_structure=block_structure)

    return model.to(device)


def compute_reconstruction_loss(model, x_input, targets, z_latent=None, is_latent_model=True):
    """
    Compute cross-entropy reconstruction loss for given inputs.

    Args:
        model: The model (LatentRNN_secondstep for latent, AblatedRNN for vanilla)
        x_input: Input sequences (B, T, in_dim)
        targets: Target choices (B, T)
        z_latent: Latent vectors to use (B, z_dim) - only for latent model
        is_latent_model: Whether this is the latent model

    Returns:
        Per-participant reconstruction losses (B,)
    """
    B, T = targets.shape[:2]

    with torch.no_grad():
        if is_latent_model:
            # For latent model, we directly use decoder with given z
            # Need to expand z for all timesteps
            logits, _ = model.decoder(x_input, z_latent)  # (B, T, A)
        else:
            # For vanilla model
            logits, _, _ = model(x_input)  # (B, T, A)

        # Compute per-participant cross-entropy loss
        probs = F.softmax(logits, dim=-1)

        # Get probability of actual choices
        targets_long = targets.long()
        chosen_probs = probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)

        # Compute negative log-likelihood per participant
        log_probs = torch.log(chosen_probs.clamp(min=1e-8))
        loss_per_participant = -log_probs.sum(dim=1)  # Sum over time

    return loss_per_participant


def compute_matched_and_mismatched_loss(
    model,
    x_input,
    targets,
    n_random_samples=50,
    is_latent_model=True,
    device='cpu'
):
    """
    Compute matched and mismatched reconstruction losses.

    For latent model:
    - Matched: encode participant i's data, decode participant i's data
    - Mismatched: encode participant j's data, decode participant i's data

    For vanilla model:
    - We shuffle hidden states between participants (conceptually similar)

    Args:
        model: The trained model
        x_input: Input sequences (B, T, in_dim)
        targets: Target choices (B, T)
        n_random_samples: Number of random pairings to average
        is_latent_model: Whether this is the latent model
        device: torch device

    Returns:
        dict with 'matched_loss', 'mismatched_loss', 'specificity', 'per_participant'
    """
    B = x_input.shape[0]

    if is_latent_model:
        # Get the encoder (IDRNN)
        # First, encode all participants to get their latents
        x_enc = x_input.unsqueeze(1)  # (B, 1, T, in_dim)
        with torch.no_grad():
            mu, logvar = model.encoder(x_enc, return_per_timestep=False)
            z_matched = mu  # (B, z_dim) - use mean for matched

        # Compute matched loss: use own latent
        matched_loss = compute_reconstruction_loss(
            model, x_input, targets, z_matched, is_latent_model=True
        )

        # Compute mismatched loss: randomly pair latents with wrong sequences
        mismatched_losses = []
        for _ in range(n_random_samples):
            # Create random permutation (ensuring no self-pairing)
            perm = torch.randperm(B, device=device)
            # Make sure no one is paired with themselves
            same_idx = (perm == torch.arange(B, device=device))
            while same_idx.any():
                # Shift indices where there's self-pairing
                shift_idx = same_idx.nonzero(as_tuple=True)[0]
                perm[shift_idx] = (perm[shift_idx] + 1) % B
                same_idx = (perm == torch.arange(B, device=device))

            z_mismatched = z_matched[perm]  # Use someone else's latent
            mismatch_loss = compute_reconstruction_loss(
                model, x_input, targets, z_mismatched, is_latent_model=True
            )
            mismatched_losses.append(mismatch_loss)

        mismatched_loss = torch.stack(mismatched_losses).mean(dim=0)

    else:
        # For vanilla model, there's no explicit latent
        # We can still compute how much forcing different hidden states hurts
        # But the primary metric is just the reconstruction loss itself
        matched_loss = compute_reconstruction_loss(
            model, x_input, targets, None, is_latent_model=False
        )

        # For vanilla, mismatched doesn't quite apply the same way
        # We'll just use matched loss as both (no specificity expected)
        mismatched_loss = matched_loss.clone()

    specificity = mismatched_loss - matched_loss

    return {
        'matched_loss': matched_loss.mean().item(),
        'mismatched_loss': mismatched_loss.mean().item(),
        'specificity': specificity.mean().item(),
        'specificity_std': specificity.std().item(),
        'per_participant_specificity': specificity.cpu().numpy(),
        'per_participant_matched': matched_loss.cpu().numpy(),
        'per_participant_mismatched': mismatched_loss.cpu().numpy()
    }


def compute_metrics_for_epoch(
    epoch,
    seed,
    base_dir,
    data_dir,
    is_latent_model,
    device
):
    """
    Compute reconstruction specificity metrics for a given epoch/seed.
    """
    run_dir = os.path.join(base_dir, f"seed_{seed}")
    checkpoint_path = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")

    if not os.path.exists(checkpoint_path):
        return None

    # Load data
    xin_test = np.load(f"{data_dir}/xin_test.npy")
    c_test = np.load(f"{data_dir}/c_test.npy")

    xin_test = torch.from_numpy(xin_test).float().to(device)
    c_test = torch.from_numpy(c_test).float().to(device)

    B, T, in_dim = xin_test.shape

    # Load model config from run directory
    model_config = load_model_config(run_dir)

    if is_latent_model:
        frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")

        if not os.path.exists(frozen_decoder_path):
            print(f"  Warning: No frozen decoder at {frozen_decoder_path}")
            return None

        if model_config is not None:
            # Use config to create model
            model = create_model_from_config(
                model_config, n_participants=B, device=device,
                frozen_decoder_path=frozen_decoder_path
            )
        else:
            # Fallback to hardcoded defaults for older runs without config
            print(f"  Warning: No config found at {run_dir}, using defaults")
            z_dim = 1
            hidden = 10
            temp_encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
            temp_decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden)
            temp_model = LatentRNNz(encoder=temp_encoder, decoder=temp_decoder,
                                     hid=hidden, z_dim=z_dim, in_dim=in_dim, A=2,
                                     block_structure=False)
            temp_model.load_state_dict(torch.load(frozen_decoder_path, map_location=device))

            encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
            frozen_decoder = copy.deepcopy(temp_model.decoder)
            for p in frozen_decoder.parameters():
                p.requires_grad = False

            model = LatentRNN_secondstep(
                encoder=encoder_IDRNN,
                hid=hidden,
                z_dim=z_dim,
                in_dim=in_dim,
                A=2,
                decoder=frozen_decoder
            )
            model.to(device)

        model = load_model_checkpoint(checkpoint_path, model, device)

    else:
        # Vanilla model
        if model_config is not None:
            model = create_model_from_config(
                model_config, n_participants=B, device=device
            )
        else:
            # Fallback for older runs
            hidden = 10
            model = AblatedRNN(hid=hidden, in_dim=in_dim, A=2, block_structure=False)
            model.to(device)

        model = load_model_checkpoint(checkpoint_path, model, device)

    # Compute metrics
    metrics = compute_matched_and_mismatched_loss(
        model=model,
        x_input=xin_test,
        targets=c_test,
        n_random_samples=50,
        is_latent_model=is_latent_model,
        device=device
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute reconstruction specificity metrics")
    parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                        help="latent or vanilla modeling")
    parser.add_argument('--dataset_id', type=int, default=0,
                        help="dataset ID for multi-dataset experiments")
    parser.add_argument('--min_epoch', type=int, default=1000,
                        help="minimum epoch to consider")
    parser.add_argument('--max_epoch', type=int, default=None,
                        help="maximum epoch to consider")
    parser.add_argument('--step', type=int, default=1,
                        help="sample every N epochs (1 = all epochs)")
    parser.add_argument('--analyze_correlation', action='store_true',
                        help="Analyze correlation with ground truth RSA")
    parser.add_argument('--dgp', type=str, default=None,
                        help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_latent = args.latent
    DATASET_ID = args.dataset_id
    MIN_EPOCH = args.min_epoch
    DGP = args.dgp

    # Build directory names with optional DGP prefix
    if DGP:
        BASE_DIR = f"runs_{DGP}_dataset{DATASET_ID}" if is_latent else f"runs_vanilla_{DGP}_dataset{DATASET_ID}"
        DATA_DIR = f"data_{DGP}_dataset{DATASET_ID}"
    else:
        BASE_DIR = f"runs_dataset{DATASET_ID}" if is_latent else f"runs_vanilla_dataset{DATASET_ID}"
        DATA_DIR = f"data_dataset{DATASET_ID}"
    SEEDS = [12, 50, 76, 100, 142]

    model_type = "IDRNN" if is_latent else "Vanilla"
    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION SPECIFICITY ANALYSIS")
    print(f"Model: {model_type}")
    print(f"Dataset: {DATASET_ID}")
    print(f"{'='*80}\n")

    # Find available epochs
    def list_epochs_for_seed(seed):
        ckpt_dir = os.path.join(BASE_DIR, f"seed_{seed}", "checkpoints")
        if not os.path.exists(ckpt_dir):
            return []
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch") and f.endswith(".pt")]
        epochs = [int(f.replace("epoch", "").replace(".pt", "")) for f in files]
        return sorted(epochs)

    epochs_per_seed = {seed: set(list_epochs_for_seed(seed)) for seed in SEEDS}
    common_epochs = sorted(set.intersection(*epochs_per_seed.values()))
    common_epochs = [e for e in common_epochs if e >= MIN_EPOCH]

    if args.max_epoch is not None:
        common_epochs = [e for e in common_epochs if e <= args.max_epoch]

    if args.step > 1:
        common_epochs = common_epochs[::args.step]

    print(f"Found {len(common_epochs)} epochs to analyze (step={args.step})")

    if not common_epochs:
        print("No common epochs found. Exiting.")
        return

    # Load ground truth for correlation analysis
    if args.analyze_correlation:
        import pandas as pd
        test_params = pd.read_csv(f"{DATA_DIR}/true_test_parameter_values.csv")
        params = test_params["alphaP_list"].values
        distance_matrix_params, _ = plf.rsa_latents(
            latents=params, metric="euclidean", title="params",
            reduction="entire", plot=False, original_data=True, cluster_order=False
        )
        vec_params = vectorize_rsa(distance_matrix_params)

    # Compute metrics for each epoch and seed
    all_results = {}

    for epoch in common_epochs:
        epoch_results = {}
        specificities = []
        matched_losses = []
        mismatched_losses = []

        for seed in SEEDS:
            print(f"Processing epoch {epoch}, seed {seed}...")

            metrics = compute_metrics_for_epoch(
                epoch=epoch,
                seed=seed,
                base_dir=BASE_DIR,
                data_dir=DATA_DIR,
                is_latent_model=is_latent,
                device=device
            )

            if metrics is not None:
                epoch_results[seed] = metrics
                specificities.append(metrics['specificity'])
                matched_losses.append(metrics['matched_loss'])
                mismatched_losses.append(metrics['mismatched_loss'])

        if specificities:
            all_results[epoch] = {
                'per_seed': epoch_results,
                'mean_specificity': np.mean(specificities),
                'std_specificity': np.std(specificities),
                'mean_matched_loss': np.mean(matched_losses),
                'mean_mismatched_loss': np.mean(mismatched_losses)
            }

            print(f"  Epoch {epoch}: specificity = {np.mean(specificities):.4f} +/- {np.std(specificities):.4f}")
            print(f"              matched_loss = {np.mean(matched_losses):.4f}")
            print(f"              mismatched_loss = {np.mean(mismatched_losses):.4f}")

    # Analyze trend over epochs
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    epochs_sorted = sorted(all_results.keys())
    specificities_over_time = [all_results[e]['mean_specificity'] for e in epochs_sorted]
    matched_over_time = [all_results[e]['mean_matched_loss'] for e in epochs_sorted]
    mismatched_over_time = [all_results[e]['mean_mismatched_loss'] for e in epochs_sorted]

    print(f"\nSpecificity over training (should increase for good IDRNN):")
    for e in epochs_sorted:
        print(f"  Epoch {e:5d}: {all_results[e]['mean_specificity']:+.4f}")

    # Trend analysis
    if len(epochs_sorted) > 2:
        corr_with_epoch, p_val = spearmanr(epochs_sorted, specificities_over_time)
        print(f"\nSpecificity trend: r={corr_with_epoch:.3f} (p={p_val:.4f})")

        if corr_with_epoch > 0:
            print("  -> POSITIVE: Specificity increases over training (good sign)")
        else:
            print("  -> NEGATIVE: Specificity decreases over training (unexpected)")

    # Correlation with ground truth RSA
    if args.analyze_correlation:
        print("\n" + "-"*80)
        print("CORRELATION WITH GROUND TRUTH RSA")
        print("-"*80)

        gt_correlations = []
        specificity_values = []

        for epoch in epochs_sorted:
            # Load RSA vectors for this epoch
            epoch_gt_corrs = []
            for seed in SEEDS:
                rsa_path = os.path.join(BASE_DIR, f"seed_{seed}", "rsa", f"epoch_{epoch:04d}.npy")
                if os.path.exists(rsa_path):
                    rsa_vec = np.load(rsa_path)
                    gt_corr = np.corrcoef(vec_params, rsa_vec)[0, 1]
                    epoch_gt_corrs.append(gt_corr)

            if epoch_gt_corrs:
                mean_gt_corr = np.mean(epoch_gt_corrs)
                gt_correlations.append(mean_gt_corr)
                specificity_values.append(all_results[epoch]['mean_specificity'])
                print(f"  Epoch {epoch}: GT_corr={mean_gt_corr:.4f}, Specificity={all_results[epoch]['mean_specificity']:.4f}")

        if len(gt_correlations) > 2:
            r, p = pearsonr(gt_correlations, specificity_values)
            print(f"\nCorrelation between GT_RSA and Specificity: r={r:.3f} (p={p:.4f})")

            if r > 0.3:
                print("  -> POSITIVE correlation: Higher specificity predicts better GT alignment")
            elif r < -0.3:
                print("  -> NEGATIVE correlation: Higher specificity predicts worse GT alignment")
            else:
                print("  -> WEAK correlation: Specificity is not a strong predictor")

    # Save results
    output_path = os.path.join(BASE_DIR, "reconstruction_specificity.json")

    # Convert to JSON-serializable format
    results_to_save = {
        "epochs": epochs_sorted,
        "mean_specificity": specificities_over_time,
        "mean_matched_loss": matched_over_time,
        "mean_mismatched_loss": mismatched_over_time,
        "per_epoch": {
            str(e): {
                'mean_specificity': all_results[e]['mean_specificity'],
                'std_specificity': all_results[e]['std_specificity'],
                'mean_matched_loss': all_results[e]['mean_matched_loss'],
                'mean_mismatched_loss': all_results[e]['mean_mismatched_loss']
            }
            for e in epochs_sorted
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Identify best epoch by specificity
    if specificities_over_time:
        best_idx = np.argmax(specificities_over_time)
        best_epoch = epochs_sorted[best_idx]
        print(f"\nBest epoch by specificity: {best_epoch} (specificity = {specificities_over_time[best_idx]:.4f})")


if __name__ == "__main__":
    main()
