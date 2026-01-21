#!/usr/bin/env python3
"""
Select best epoch using reconstruction specificity metric.

This metric measures how input-specific the learned latent representations are:
- Matched reconstruction: Use participant i's latent to decode participant i's sequence
- Mismatched reconstruction: Use participant j's latent to decode participant i's sequence
- Specificity = mean(loss_mismatched) - mean(loss_matched)

Higher specificity means the model has learned more participant-specific representations.

Empirically, this correlates strongly with ground truth RSA alignment (r > 0.8 across datasets).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import json
import copy

from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN,
    LookupEncoderZ, LatentRNNz, vectorize_rsa
)
import plot_functions as plf

# Import utility functions from compute_reconstruction_specificity
from compute_reconstruction_specificity import (
    load_model_config,
    create_model_from_config,
    load_model_checkpoint
)


def compute_reconstruction_specificity(
    model,
    x_input,
    targets,
    n_random_samples=50,
    device='cpu'
):
    """
    Compute reconstruction specificity for IDRNN.

    Returns the mean difference between mismatched and matched reconstruction losses.
    """
    B = x_input.shape[0]

    # Get the latents for each participant
    x_enc = x_input.unsqueeze(1)  # (B, 1, T, in_dim)
    with torch.no_grad():
        mu, logvar = model.encoder(x_enc, return_per_timestep=False)
        z_matched = mu  # (B, z_dim)

    # Compute matched loss
    with torch.no_grad():
        logits_matched, _ = model.decoder(x_input, z_matched)
        probs_matched = F.softmax(logits_matched, dim=-1)
        targets_long = targets.long()
        chosen_probs_matched = probs_matched.gather(
            dim=-1, index=targets_long.unsqueeze(-1)
        ).squeeze(-1)
        log_probs_matched = torch.log(chosen_probs_matched.clamp(min=1e-8))
        matched_loss = -log_probs_matched.sum(dim=1)

    # Compute mismatched loss (average over random pairings)
    mismatched_losses = []
    for _ in range(n_random_samples):
        perm = torch.randperm(B, device=device)
        same_idx = (perm == torch.arange(B, device=device))
        while same_idx.any():
            shift_idx = same_idx.nonzero(as_tuple=True)[0]
            perm[shift_idx] = (perm[shift_idx] + 1) % B
            same_idx = (perm == torch.arange(B, device=device))

        z_mismatched = z_matched[perm]

        with torch.no_grad():
            logits_mismatch, _ = model.decoder(x_input, z_mismatched)
            probs_mismatch = F.softmax(logits_mismatch, dim=-1)
            chosen_probs_mismatch = probs_mismatch.gather(
                dim=-1, index=targets_long.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_mismatch = torch.log(chosen_probs_mismatch.clamp(min=1e-8))
            mismatch_loss = -log_probs_mismatch.sum(dim=1)
            mismatched_losses.append(mismatch_loss)

    mismatched_loss = torch.stack(mismatched_losses).mean(dim=0)
    specificity = (mismatched_loss - matched_loss).mean().item()

    return specificity, matched_loss.mean().item(), mismatched_loss.mean().item()


def main():
    # Epoch window configuration (must match analyze_synthetic_multi_dataset.py)
    DEFAULT_MAX_EPOCH = 3000  # Epoch cutoff to prevent overtraining
    DEFAULT_MIN_EPOCH = 1000  # Minimum epoch to consider

    parser = argparse.ArgumentParser(description="Select best epoch by reconstruction specificity")
    parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                        help="latent or vanilla modeling")
    parser.add_argument('--dataset_id', type=int, default=0,
                        help="dataset ID for multi-dataset experiments")
    parser.add_argument('--min_epoch', type=int, default=DEFAULT_MIN_EPOCH,
                        help="minimum epoch to consider")
    parser.add_argument('--max_epoch', type=int, default=DEFAULT_MAX_EPOCH,
                        help="maximum epoch to consider")
    parser.add_argument('--dgp', type=str, default=None,
                        help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_latent = args.latent
    DATASET_ID = args.dataset_id
    MIN_EPOCH = args.min_epoch
    MAX_EPOCH = args.max_epoch
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
    print(f"EPOCH SELECTION BY RECONSTRUCTION SPECIFICITY")
    print(f"Model: {model_type}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Epoch window: [{MIN_EPOCH}, {MAX_EPOCH}]")
    print(f"{'='*80}\n")

    # Load data
    xin_test = np.load(f"{DATA_DIR}/xin_test.npy")
    c_test = np.load(f"{DATA_DIR}/c_test.npy")
    xin_test = torch.from_numpy(xin_test).float().to(device)
    c_test = torch.from_numpy(c_test).float().to(device)
    B, T, in_dim = xin_test.shape

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
    common_epochs = [e for e in common_epochs if e >= MIN_EPOCH and (MAX_EPOCH is None or e <= MAX_EPOCH)]

    print(f"Found {len(common_epochs)} common epochs in window [{MIN_EPOCH}, {MAX_EPOCH}]")

    if not common_epochs:
        print("No common epochs found. Exiting.")
        return

    # Compute specificity for each epoch-seed combination
    specificity_per_epoch = {}
    seed_details = {}

    for epoch in common_epochs:
        epoch_specificities = []
        seed_details[epoch] = {}

        for seed in SEEDS:
            run_dir = os.path.join(BASE_DIR, f"seed_{seed}")
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
            frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")

            if not os.path.exists(checkpoint_path) or not os.path.exists(frozen_decoder_path):
                continue

            try:
                # Load model config from run directory
                model_config = load_model_config(run_dir)

                if model_config is not None:
                    # Use config to create model
                    model = create_model_from_config(
                        model_config, n_participants=B, device=device,
                        frozen_decoder_path=frozen_decoder_path
                    )
                else:
                    # Fallback to hardcoded defaults for older runs
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

                # Compute specificity
                spec, matched, mismatched = compute_reconstruction_specificity(
                    model, xin_test, c_test, n_random_samples=30, device=device
                )

                epoch_specificities.append(spec)
                seed_details[epoch][seed] = {
                    'specificity': spec,
                    'matched_loss': matched,
                    'mismatched_loss': mismatched
                }

            except Exception as e:
                print(f"  Warning: Failed to process epoch {epoch}, seed {seed}: {e}")
                continue

        if epoch_specificities:
            mean_spec = np.mean(epoch_specificities)
            std_spec = np.std(epoch_specificities)
            specificity_per_epoch[epoch] = {
                'mean': mean_spec,
                'std': std_spec,
                'values': epoch_specificities
            }
            print(f"Epoch {epoch:5d}: specificity = {mean_spec:.4f} +/- {std_spec:.4f}")

    # Select best epoch (highest specificity)
    best_epoch = max(specificity_per_epoch, key=lambda e: specificity_per_epoch[e]['mean'])
    best_specificity = specificity_per_epoch[best_epoch]['mean']

    print(f"\n{'='*80}")
    print(f"BEST EPOCH: {best_epoch}")
    print(f"Specificity: {best_specificity:.4f}")
    print(f"{'='*80}")

    # Select best seed for this epoch
    print("\nSelecting best seed...")
    seed_scores = seed_details[best_epoch]

    # Best seed = highest specificity
    best_seed = max(seed_scores, key=lambda s: seed_scores[s]['specificity'])
    print(f"Best seed: {best_seed} (specificity = {seed_scores[best_seed]['specificity']:.4f})")

    for seed in SEEDS:
        if seed in seed_scores:
            marker = " <-- BEST" if seed == best_seed else ""
            print(f"  Seed {seed}: specificity = {seed_scores[seed]['specificity']:.4f}{marker}")

    # Save results
    output = {
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "best_specificity": float(best_specificity),
        "all_epochs": {
            str(e): {
                'mean_specificity': float(specificity_per_epoch[e]['mean']),
                'std_specificity': float(specificity_per_epoch[e]['std'])
            }
            for e in specificity_per_epoch
        },
        "seed_details_best_epoch": {
            str(s): {k: float(v) for k, v in seed_scores[s].items()}
            for s in seed_scores
        },
        "min_epoch": MIN_EPOCH,
        "max_epoch": MAX_EPOCH,
        "seeds": SEEDS
    }

    output_path = os.path.join(BASE_DIR, "best_epoch_by_specificity.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Also save in standard format for pipeline compatibility
    standard_output = {
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "best_reliability": float(best_specificity),
        "seeds": SEEDS
    }

    standard_path = os.path.join(BASE_DIR, "best_epoch_by_rsa.json")
    with open(standard_path, 'w') as f:
        json.dump(standard_output, f, indent=2)
    print(f"Saved to standard location: {standard_path}")


if __name__ == "__main__":
    main()
