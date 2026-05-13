#!/usr/bin/env python3
"""
Select best epoch using NLL (negative log-likelihood) per participant on validation data.

For IDRNN: encode sequence → z (mu) → decode → NLL
For Vanilla: forward pass → NLL

The epoch with the lowest mean NLL across seeds is selected.
Val data is used by default; falls back to test if val is unavailable.
"""

import os
import argparse
import numpy as np
import torch
import json
import copy

from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN,
    LookupEncoderZ, LatentRNNz
)
from compute_reconstruction_specificity import (
    load_model_config,
    create_model_from_config,
    load_model_checkpoint,
    compute_reconstruction_loss
)


def compute_nll(model, x_input, targets, is_latent_model, device, x_enc_input=None):
    """
    Compute mean NLL per participant.

    For IDRNN (is_latent_model=True): encode x to get z (mu), then decode.
    For Vanilla (is_latent_model=False): full forward pass.

    Returns
    -------
    mean_nll : float
    per_participant_nll : np.ndarray (B,)
    """
    if is_latent_model:
        x_for_enc = x_enc_input if x_enc_input is not None else x_input
        x_enc = x_for_enc.unsqueeze(1)  # (B, 1, T, enc_in_dim)
        with torch.no_grad():
            mu, _ = model.encoder(x_enc, return_per_timestep=False)
        loss_per_participant = compute_reconstruction_loss(
            model, x_input, targets, z_latent=mu, is_latent_model=True
        )
    else:
        loss_per_participant = compute_reconstruction_loss(
            model, x_input, targets, is_latent_model=False
        )

    return loss_per_participant.mean().item(), loss_per_participant.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Select best epoch by NLL per participant on validation data"
    )
    parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                        help="latent (IDRNN) or vanilla model")
    parser.add_argument('--dataset_id', type=int, default=0,
                        help="dataset ID for multi-dataset experiments")
    parser.add_argument('--min_epoch', type=int, default=100,
                        help="minimum epoch to consider")
    parser.add_argument('--max_epoch', type=int, default=1500,
                        help="maximum epoch to consider")
    parser.add_argument('--dgp', type=str, default=None,
                        help="Data generating process (e.g. 'sloutsky', 'spatial_bandit')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_latent = args.latent
    DATASET_ID = args.dataset_id
    MIN_EPOCH = args.min_epoch
    MAX_EPOCH = args.max_epoch
    DGP = args.dgp

    # Determine if this is human data (no dataset_id suffix in directory names)
    is_human_data = DGP in ("sloutsky", "palminteri", "spatial_bandit")

    if is_human_data:
        BASE_DIR = f"runs_{DGP}" if is_latent else f"runs_vanilla_{DGP}"
        DATA_DIR = f"data_{DGP}"
    elif DGP:
        BASE_DIR = (f"runs_{DGP}_dataset{DATASET_ID}" if is_latent
                    else f"runs_vanilla_{DGP}_dataset{DATASET_ID}")
        DATA_DIR = f"data_{DGP}_dataset{DATASET_ID}"
    else:
        BASE_DIR = (f"runs_dataset{DATASET_ID}" if is_latent
                    else f"runs_vanilla_dataset{DATASET_ID}")
        DATA_DIR = f"data_dataset{DATASET_ID}"

    # Detect available seeds
    available_seeds = []
    if os.path.exists(BASE_DIR):
        for d in os.listdir(BASE_DIR):
            if d.startswith("seed_"):
                try:
                    available_seeds.append(int(d.split("_")[1]))
                except ValueError:
                    pass
    SEEDS = sorted(available_seeds)
    if not SEEDS:
        raise RuntimeError(f"No seed directories found in {BASE_DIR}")

    model_type = "IDRNN" if is_latent else "Vanilla"
    print(f"\n{'='*80}")
    print(f"EPOCH SELECTION BY NLL")
    print(f"Model: {model_type} | DGP: {DGP} | Epoch window: [{MIN_EPOCH}, {MAX_EPOCH}]")
    print(f"{'='*80}\n")

    # Load val data, fall back to test
    data_split = None
    for split in ("val", "test"):
        xin_path = os.path.join(DATA_DIR, f"xin_{split}.npy")
        c_path = os.path.join(DATA_DIR, f"c_{split}.npy")
        if os.path.exists(xin_path) and os.path.exists(c_path):
            data_split = split
            break
    if data_split is None:
        raise RuntimeError(f"No val or test data found in {DATA_DIR}")

    xin_data = (torch.from_numpy(np.load(os.path.join(DATA_DIR, f"xin_{data_split}.npy")))
                .float().to(device))
    c_data = (torch.from_numpy(np.load(os.path.join(DATA_DIR, f"c_{data_split}.npy")))
              .float().to(device))
    B, T, in_dim = xin_data.shape
    print(f"Using {data_split} data: {B} sequences, T={T}, in_dim={in_dim}")

    # Optional encoder-specific input (for split encoder/decoder architectures)
    enc_path = os.path.join(DATA_DIR, f"xin_enc_{data_split}.npy")
    xin_enc = (torch.from_numpy(np.load(enc_path)).float().to(device)
               if os.path.exists(enc_path) else None)

    # Find common checkpoint epochs within window
    def list_epochs(seed):
        ckpt_dir = os.path.join(BASE_DIR, f"seed_{seed}", "checkpoints")
        if not os.path.exists(ckpt_dir):
            return []
        return sorted([
            int(f.replace("epoch", "").replace(".pt", ""))
            for f in os.listdir(ckpt_dir)
            if f.startswith("epoch") and f.endswith(".pt")
        ])

    epochs_per_seed = {s: set(list_epochs(s)) for s in SEEDS}
    common_epochs = sorted(set.intersection(*epochs_per_seed.values()))
    common_epochs = [e for e in common_epochs if MIN_EPOCH <= e <= MAX_EPOCH]

    print(f"Found {len(common_epochs)} common epochs in [{MIN_EPOCH}, {MAX_EPOCH}]")
    if not common_epochs:
        print("No common epochs found. Exiting.")
        return

    # Compute NLL for each epoch across seeds
    nll_per_epoch = {}
    seed_details = {}

    for epoch in common_epochs:
        epoch_nlls = []
        seed_details[epoch] = {}

        for seed in SEEDS:
            run_dir = os.path.join(BASE_DIR, f"seed_{seed}")
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
            frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")

            if not os.path.exists(checkpoint_path):
                continue
            if is_latent and not os.path.exists(frozen_decoder_path):
                continue

            try:
                model_config = load_model_config(run_dir)
                if model_config is not None:
                    model = create_model_from_config(
                        model_config, n_participants=B, device=device,
                        frozen_decoder_path=frozen_decoder_path if is_latent else None
                    )
                else:
                    # Fallback: hardcoded defaults for runs without config.json
                    hidden = 10
                    A = int(c_data.max().item()) + 1
                    if is_latent:
                        z_dim = 1
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
                    else:
                        model = AblatedRNN(hid=hidden, in_dim=in_dim, A=A, block_structure=False)
                    model.to(device)

                model = load_model_checkpoint(checkpoint_path, model, device)
                mean_nll, _ = compute_nll(
                    model, xin_data, c_data,
                    is_latent_model=is_latent, device=device, x_enc_input=xin_enc
                )
                epoch_nlls.append(mean_nll)
                seed_details[epoch][seed] = {'nll': mean_nll}

            except Exception as e:
                print(f"  Warning: Failed epoch {epoch}, seed {seed}: {e}")
                continue

        if epoch_nlls:
            mean_nll = float(np.mean(epoch_nlls))
            std_nll = float(np.std(epoch_nlls))
            nll_per_epoch[epoch] = {'mean': mean_nll, 'std': std_nll}
            print(f"Epoch {epoch:5d}: NLL = {mean_nll:.4f} +/- {std_nll:.4f}")

    if not nll_per_epoch:
        print("No valid epochs found. Exiting.")
        return

    # Select best epoch: lowest mean NLL
    best_epoch = min(nll_per_epoch, key=lambda e: nll_per_epoch[e]['mean'])
    best_nll = nll_per_epoch[best_epoch]['mean']

    print(f"\n{'='*80}")
    print(f"BEST EPOCH: {best_epoch}")
    print(f"Mean NLL:   {best_nll:.4f}")
    print(f"{'='*80}")

    # Best seed = lowest NLL at the best epoch
    best_seed = min(seed_details[best_epoch], key=lambda s: seed_details[best_epoch][s]['nll'])
    print(f"\nBest seed: {best_seed} (NLL = {seed_details[best_epoch][best_seed]['nll']:.4f})")
    for s in SEEDS:
        if s in seed_details[best_epoch]:
            marker = " <-- BEST" if s == best_seed else ""
            print(f"  Seed {s}: NLL = {seed_details[best_epoch][s]['nll']:.4f}{marker}")

    # Save results
    output = {
        "best_epoch": best_epoch,
        "best_seed": best_seed,
        "best_nll": best_nll,
        "data_split_used": data_split,
        "selection_method": "nll",
        "min_epoch": MIN_EPOCH,
        "max_epoch": MAX_EPOCH,
        "seeds": SEEDS,
        "all_epochs": {
            str(e): {
                "mean_nll": float(nll_per_epoch[e]['mean']),
                "std_nll": float(nll_per_epoch[e]['std'])
            }
            for e in nll_per_epoch
        },
        "seed_details_best_epoch": {
            str(s): {"nll": float(seed_details[best_epoch][s]['nll'])}
            for s in seed_details[best_epoch]
        }
    }

    output_path = os.path.join(BASE_DIR, "best_epoch_by_nll.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
