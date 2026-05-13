#!/usr/bin/env python3
"""
Evaluation script for spatial bandit hyperparameter search.
Uses reconstruction specificity as the metric (no ground truth for RSA).
Mirrors hyperparam_eval_sloutsky.py but points to data_spatial_bandit/.
"""

import os
import json
import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import copy

from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep,
    LookupEncoderZ, LatentRNNz
)
from compute_reconstruction_specificity import (
    load_model_config,
    create_model_from_config,
    load_model_checkpoint
)

DGP = "spatial_bandit"
DATA_DIR = f"data_{DGP}"


def compute_reconstruction_specificity(
    model,
    x_input,
    targets,
    n_random_samples=30,
    device='cpu'
):
    """
    Reconstruction specificity: mean(mismatched_loss - matched_loss).
    Higher is better (model is more participant-specific).
    """
    B = x_input.shape[0]
    x_enc = x_input.unsqueeze(1)
    with torch.no_grad():
        mu, logvar = model.encoder(x_enc, return_per_timestep=False)
        z_matched = mu

    # Matched loss
    with torch.no_grad():
        logits_matched, _ = model.decoder(x_input, z_matched)
        probs_matched = F.softmax(logits_matched, dim=-1)
        targets_long = targets.long()
        chosen_probs_matched = probs_matched.gather(
            dim=-1, index=targets_long.unsqueeze(-1)
        ).squeeze(-1)
        log_probs_matched = torch.log(chosen_probs_matched.clamp(min=1e-8))
        matched_loss = -log_probs_matched.sum(dim=1)

    # Mismatched loss (random permutations)
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


def load_loss_trajectory(run_dir):
    loss_dir = os.path.join(run_dir, "loss")
    if not os.path.exists(loss_dir):
        return None
    loss_files = sorted(glob.glob(os.path.join(loss_dir, "epoch_*.npy")))
    losses = {}
    for lf in loss_files:
        epoch = int(os.path.basename(lf).replace("epoch_", "").replace(".npy", ""))
        losses[epoch] = np.load(lf).item()
    return losses


def format_lmbd_for_path(lmbd):
    if lmbd == int(lmbd):
        return str(int(lmbd))
    return str(lmbd)


def evaluate_hyperparam_combo(lmbd, z_dim, seeds, min_epoch=1000, max_epoch=3000):
    """Evaluate one (lambda, z_dim) combo across seeds using reconstruction specificity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xin_test = np.load(f"{DATA_DIR}/xin_test.npy")
    c_test   = np.load(f"{DATA_DIR}/c_test.npy")
    xin_test = torch.from_numpy(xin_test).float().to(device)
    c_test   = torch.from_numpy(c_test).float().to(device)
    B, T, in_dim = xin_test.shape

    lmbd_str = format_lmbd_for_path(lmbd)
    results = {
        "lmbd": lmbd, "z_dim": z_dim, "dgp": DGP,
        "seeds": seeds, "per_seed": {}
    }

    all_best_specs, all_final_specs = [], []

    for seed in seeds:
        run_dir = f"hp_search_runs_{DGP}/lmbd_{lmbd_str}_z_{z_dim}/seed_{seed}"
        if not os.path.exists(run_dir):
            print(f"Warning: Run directory not found: {run_dir}")
            continue

        frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
        if not os.path.exists(frozen_decoder_path):
            print(f"Warning: Frozen decoder not found for seed {seed}")
            continue

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            print(f"Warning: No checkpoints found for seed {seed}")
            continue

        ckpt_files = [f for f in os.listdir(ckpt_dir)
                      if f.startswith("epoch") and f.endswith(".pt")]
        epochs = sorted([int(f.replace("epoch", "").replace(".pt", "")) for f in ckpt_files])
        epochs_in_window = [e for e in epochs if min_epoch <= e <= max_epoch]

        if not epochs_in_window:
            print(f"Warning: No checkpoints in epoch window [{min_epoch}, {max_epoch}] "
                  f"for seed {seed}")
            continue

        specificities = {}
        for epoch in epochs_in_window:
            checkpoint_path = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
            try:
                model_config = load_model_config(run_dir)
                if model_config is not None:
                    model = create_model_from_config(
                        model_config, n_participants=B, device=device,
                        frozen_decoder_path=frozen_decoder_path
                    )
                else:
                    # Fallback
                    hidden = 10
                    # Determine A from c_test
                    A = int(c_test.max().item()) + 1
                    temp_encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
                    temp_decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A)
                    temp_model = LatentRNNz(
                        encoder=temp_encoder, decoder=temp_decoder,
                        hid=hidden, z_dim=z_dim, in_dim=in_dim, A=A,
                        block_structure=False
                    )
                    temp_model.load_state_dict(
                        torch.load(frozen_decoder_path, map_location=device), strict=False
                    )
                    encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
                    frozen_decoder = copy.deepcopy(temp_model.decoder)
                    for p in frozen_decoder.parameters():
                        p.requires_grad = False
                    model = LatentRNN_secondstep(
                        encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,
                        in_dim=in_dim, A=A, decoder=frozen_decoder
                    )
                    model.to(device)

                model = load_model_checkpoint(checkpoint_path, model, device)
                spec, matched, mismatched = compute_reconstruction_specificity(
                    model, xin_test, c_test, n_random_samples=30, device=device
                )
                specificities[epoch] = spec

            except Exception as e:
                print(f"  Warning: Failed epoch {epoch}, seed {seed}: {e}")
                continue

        if not specificities:
            print(f"Warning: No valid epochs for seed {seed}")
            continue

        best_epoch   = max(specificities, key=specificities.get)
        best_spec    = specificities[best_epoch]
        final_epoch  = max(specificities.keys())
        final_spec   = specificities[final_epoch]

        losses     = load_loss_trajectory(run_dir)
        final_loss = losses.get(final_epoch) if losses else None

        results["per_seed"][seed] = {
            "best_epoch": best_epoch,
            "best_specificity": best_spec,
            "final_epoch": final_epoch,
            "final_specificity": final_spec,
            "final_loss": final_loss,
            "n_epochs_evaluated": len(specificities),
            "all_specificities": {str(e): float(s) for e, s in specificities.items()}
        }
        all_best_specs.append(best_spec)
        all_final_specs.append(final_spec)

    if all_best_specs:
        results["aggregate"] = {
            "mean_best_specificity":  float(np.mean(all_best_specs)),
            "std_best_specificity":   float(np.std(all_best_specs)),
            "mean_final_specificity": float(np.mean(all_final_specs)),
            "std_final_specificity":  float(np.std(all_final_specs)),
            "min_best_specificity":   float(np.min(all_best_specs)),
            "max_best_specificity":   float(np.max(all_best_specs)),
            "n_seeds_evaluated":      len(all_best_specs)
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial bandit hyperparameter search results")
    parser.add_argument("--lmbd",      type=float, required=True)
    parser.add_argument("--z",         type=int,   required=True)
    parser.add_argument("--seeds",     type=str,   default="12 50")
    parser.add_argument("--output",    type=str,   default=None)
    parser.add_argument("--min_epoch", type=int,   default=1000)
    parser.add_argument("--max_epoch", type=int,   default=3000)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    results = evaluate_hyperparam_combo(
        args.lmbd, args.z, seeds,
        min_epoch=args.min_epoch, max_epoch=args.max_epoch
    )

    print("\n" + "=" * 60)
    print(f"Hyperparameter Evaluation ({DGP}): lambda={args.lmbd}, z={args.z}")
    print("=" * 60)
    if "aggregate" in results:
        agg = results["aggregate"]
        print(f"Mean best specificity:  {agg['mean_best_specificity']:.4f} "
              f"+/- {agg['std_best_specificity']:.4f}")
        print(f"Mean final specificity: {agg['mean_final_specificity']:.4f} "
              f"+/- {agg['std_final_specificity']:.4f}")
        print(f"Seeds evaluated: {agg['n_seeds_evaluated']}")

    for seed, sr in results["per_seed"].items():
        print(f"\nSeed {seed}:")
        print(f"  Best:  {sr['best_specificity']:.4f} @ epoch {sr['best_epoch']}")
        print(f"  Final: {sr['final_specificity']:.4f} @ epoch {sr['final_epoch']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
