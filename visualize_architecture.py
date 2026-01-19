#!/usr/bin/env python3
"""
Generate publication-quality figures for IDRNN architecture and training process.

Figures:
1. Two-Step Training Overview
2. Detailed Architecture Diagram
3. Training Dynamics / Loss Components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
import os

# Set up publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette
COLORS = {
    'encoder': '#4ECDC4',      # Teal
    'decoder': '#FF6B6B',      # Coral
    'latent': '#FFE66D',       # Yellow
    'input': '#95E1D3',        # Light green
    'output': '#F38181',       # Light red
    'frozen': '#CCCCCC',       # Gray (frozen)
    'arrow': '#2C3E50',        # Dark blue-gray
    'text': '#2C3E50',
    'highlight': '#3498DB',    # Blue
    'loss_ce': '#E74C3C',      # Red
    'loss_kl': '#3498DB',      # Blue
    'loss_total': '#2ECC71',   # Green
}


def draw_rounded_box(ax, x, y, width, height, label, color, fontsize=9,
                     text_color='black', alpha=0.8, bold=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)
    return box


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        mutation_scale=15,
        color=color,
        linewidth=1.5
    )
    ax.add_patch(arrow)
    return arrow


def figure1_two_step_training(output_dir):
    """
    Figure 1: Two-Step Training Overview
    Shows Step 1 (decoder pretraining) and Step 2 (encoder training with frozen decoder)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== STEP 1 ====================
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 1: Pre-train Decoder\n(Learn per-participant embeddings)',
                 fontsize=12, fontweight='bold', pad=10)

    # Participant ID input
    draw_rounded_box(ax, 2, 8, 2.5, 0.8, 'Participant\nID: i', COLORS['input'], fontsize=8)

    # Lookup Encoder
    draw_rounded_box(ax, 5, 8, 2.5, 0.8, 'Lookup\nEmbedding', COLORS['encoder'], fontsize=8, bold=True)
    draw_arrow(ax, (3.3, 8), (3.7, 8))

    # Latent z
    draw_rounded_box(ax, 8, 8, 1.5, 0.8, 'zᵢ', COLORS['latent'], fontsize=10, bold=True)
    draw_arrow(ax, (6.3, 8), (7.2, 8))

    # Trial history input
    draw_rounded_box(ax, 2, 5, 2.5, 1.2, 'Trial History\nsₜ, aₜ₋₁, rₜ₋₁', COLORS['input'], fontsize=8)

    # Concat operation
    ax.add_patch(Circle((5, 5.5), 0.4, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(5, 5.5, '⊕', ha='center', va='center', fontsize=14)
    draw_arrow(ax, (3.3, 5), (4.6, 5.5))
    draw_arrow(ax, (8, 7.5), (8, 6.2))
    draw_arrow(ax, (7.6, 6), (5.4, 5.7), connectionstyle='arc3,rad=-0.2')

    # Decoder GRU
    draw_rounded_box(ax, 5, 4, 2, 0.8, 'GRU', COLORS['decoder'], fontsize=9, bold=True)
    draw_arrow(ax, (5, 5.1), (5, 4.5))

    # h0 initialization
    ax.text(6.5, 4.7, 'h₀ = f(z)', fontsize=8, style='italic', color=COLORS['text'])

    # Linear output
    draw_rounded_box(ax, 5, 2.8, 2, 0.7, 'Linear', COLORS['decoder'], fontsize=9)
    draw_arrow(ax, (5, 3.55), (5, 3.2))

    # Action output
    draw_rounded_box(ax, 5, 1.5, 1.5, 0.7, 'âₜ', COLORS['output'], fontsize=10, bold=True)
    draw_arrow(ax, (5, 2.4), (5, 1.9))

    # Loss annotation
    ax.text(5, 0.5, 'Loss: CE(âₜ, aₜ)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['loss_ce'], linewidth=2))

    # ==================== STEP 2 ====================
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Step 2: Train IDRNN Encoder\n(Decoder frozen)',
                 fontsize=12, fontweight='bold', pad=10)

    # Trial history input (encoder)
    draw_rounded_box(ax, 2, 8.5, 2.5, 1, 'Trial History\nx₁:ₜ', COLORS['input'], fontsize=8)

    # IDRNN Encoder
    draw_rounded_box(ax, 5, 8.5, 2, 0.8, 'GRU', COLORS['encoder'], fontsize=9, bold=True)
    draw_arrow(ax, (3.3, 8.5), (3.95, 8.5))

    # mu and sigma heads
    draw_rounded_box(ax, 4, 7, 1.2, 0.6, 'μ-head', COLORS['encoder'], fontsize=8)
    draw_rounded_box(ax, 6, 7, 1.2, 0.6, 'σ-head', COLORS['encoder'], fontsize=8)
    draw_arrow(ax, (4.5, 8), (4.2, 7.35))
    draw_arrow(ax, (5.5, 8), (5.8, 7.35))

    # mu and logvar outputs
    draw_rounded_box(ax, 4, 6, 1, 0.5, 'μ(t)', COLORS['latent'], fontsize=8)
    draw_rounded_box(ax, 6, 6, 1, 0.5, 'log σ²', COLORS['latent'], fontsize=8)
    draw_arrow(ax, (4, 6.65), (4, 6.3))
    draw_arrow(ax, (6, 6.65), (6, 6.3))

    # Reparameterization
    draw_rounded_box(ax, 5, 5, 2.2, 0.6, 'z = μ + σ·ε', COLORS['latent'], fontsize=8, bold=True)
    draw_arrow(ax, (4, 5.7), (4.3, 5.35))
    draw_arrow(ax, (6, 5.7), (5.7, 5.35))

    # Target z (lookup)
    draw_rounded_box(ax, 8.5, 5, 1.5, 0.6, 'z_lookup', COLORS['frozen'], fontsize=8)
    ax.text(8.5, 4.3, '(from Step 1)', fontsize=7, ha='center', style='italic')

    # KL loss arrow
    draw_arrow(ax, (6.15, 5), (7.7, 5), color=COLORS['loss_kl'], style='->')
    ax.text(6.9, 5.25, 'KL', fontsize=8, color=COLORS['loss_kl'], fontweight='bold')

    # Trial history (decoder input)
    draw_rounded_box(ax, 2, 3, 2.5, 0.8, 'Trial History', COLORS['input'], fontsize=8)

    # Frozen Decoder box
    decoder_box = FancyBboxPatch(
        (3.5, 1.2), 3.5, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['frozen'], edgecolor='black', linewidth=2,
        alpha=0.5, linestyle='--'
    )
    ax.add_patch(decoder_box)
    ax.text(5.25, 3.3, 'Frozen Decoder', fontsize=9, ha='center', fontweight='bold',
            color=COLORS['text'])

    # Decoder internals
    draw_rounded_box(ax, 5.25, 2.5, 1.5, 0.5, 'GRU', COLORS['frozen'], fontsize=8, alpha=0.7)
    draw_rounded_box(ax, 5.25, 1.7, 1.5, 0.5, 'Linear', COLORS['frozen'], fontsize=8, alpha=0.7)

    draw_arrow(ax, (3.3, 3), (4, 2.5))
    draw_arrow(ax, (5, 4.65), (5.25, 3.5))
    draw_arrow(ax, (5.25, 2.2), (5.25, 2))

    # Output
    draw_rounded_box(ax, 5.25, 0.7, 1, 0.5, 'âₜ', COLORS['output'], fontsize=9, bold=True)
    draw_arrow(ax, (5.25, 1.4), (5.25, 1))

    # Loss annotation
    ax.text(5, -0.3, 'Loss: λ·‖μ - z_lookup‖² + (1-λ)·CE(âₜ, aₜ)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['loss_total'], linewidth=2))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure1_two_step_training.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def figure2_detailed_architecture(output_dir):
    """
    Figure 2: Detailed Architecture Diagram
    Shows the full IDRNN architecture with dimensions annotated
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('IDRNN Architecture (Inference)', fontsize=14, fontweight='bold', pad=15)

    # ========== ENCODER SECTION ==========
    # Encoder boundary box
    encoder_box = FancyBboxPatch(
        (0.5, 6.5), 5, 5,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['encoder'], edgecolor='black', linewidth=2, alpha=0.15
    )
    ax.add_patch(encoder_box)
    ax.text(3, 11.2, 'IDRNN Encoder', fontsize=11, ha='center', fontweight='bold',
            color=COLORS['text'])

    # Input
    draw_rounded_box(ax, 1.5, 10, 1.8, 0.8, 'x₁:ₜ\n(B,T,4)', COLORS['input'], fontsize=8)

    # Encoder GRU
    draw_rounded_box(ax, 3, 10, 1.5, 0.8, 'GRU\n(hid=10)', COLORS['encoder'], fontsize=8, bold=True)
    draw_arrow(ax, (2.4, 10), (2.2, 10))

    # Hidden states
    draw_rounded_box(ax, 4.5, 10, 1.5, 0.8, 'hₜ\n(B,T,10)', COLORS['encoder'], fontsize=8)
    draw_arrow(ax, (3.8, 10), (3.7, 10))

    # Split to mu and sigma heads
    draw_rounded_box(ax, 2.5, 8.2, 1.5, 0.7, 'μ-head\nLinear(10→z)', COLORS['encoder'], fontsize=7)
    draw_rounded_box(ax, 4.5, 8.2, 1.5, 0.7, 'σ-head\nLinear(10→z)', COLORS['encoder'], fontsize=7)

    draw_arrow(ax, (4, 9.5), (2.8, 8.6), connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (4.8, 9.5), (4.5, 8.6), connectionstyle='arc3,rad=-0.2')

    # Mu and logvar
    draw_rounded_box(ax, 2.5, 7.2, 1.3, 0.6, 'μ(t)\n(B,T,z)', COLORS['latent'], fontsize=7)
    draw_rounded_box(ax, 4.5, 7.2, 1.3, 0.6, 'log σ²(t)\n(B,T,z)', COLORS['latent'], fontsize=7)
    draw_arrow(ax, (2.5, 7.8), (2.5, 7.55))
    draw_arrow(ax, (4.5, 7.8), (4.5, 7.55))

    # ========== REPARAMETERIZATION ==========
    draw_rounded_box(ax, 3.5, 5.8, 3, 0.8, 'Reparameterization\nz = μ + σ · ε,  ε ~ N(0,1)',
                     COLORS['latent'], fontsize=8, bold=True)
    draw_arrow(ax, (2.5, 6.85), (2.8, 6.25), connectionstyle='arc3,rad=0.1')
    draw_arrow(ax, (4.5, 6.85), (4.2, 6.25), connectionstyle='arc3,rad=-0.1')

    # Latent z output
    draw_rounded_box(ax, 3.5, 4.7, 1.5, 0.7, 'z\n(B, z_dim)', COLORS['latent'], fontsize=8, bold=True)
    draw_arrow(ax, (3.5, 5.35), (3.5, 5.1))

    # ========== DECODER SECTION ==========
    # Decoder boundary box
    decoder_box = FancyBboxPatch(
        (6, 0.5), 5.5, 10.5,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['decoder'], edgecolor='black', linewidth=2, alpha=0.15
    )
    ax.add_patch(decoder_box)
    ax.text(8.75, 10.7, 'Policy Decoder', fontsize=11, ha='center', fontweight='bold',
            color=COLORS['text'])

    # z to h0 transformation
    draw_rounded_box(ax, 8.75, 9.5, 2, 0.8, 'z → h₀\nLinear(z→10)', COLORS['decoder'], fontsize=8)
    draw_arrow(ax, (4.3, 4.7), (6.5, 9.5), connectionstyle='arc3,rad=-0.3')

    # Initial hidden state
    draw_rounded_box(ax, 8.75, 8.3, 1.5, 0.7, 'h₀\n(1,B,10)', COLORS['decoder'], fontsize=8)
    draw_arrow(ax, (8.75, 9.05), (8.75, 8.7))

    # Trial input
    draw_rounded_box(ax, 6.8, 6.5, 1.5, 0.8, 'xₜ\n(B,T,4)', COLORS['input'], fontsize=8)

    # Concatenation
    ax.add_patch(Circle((8.75, 6.5), 0.35, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(8.75, 6.5, '⊕', ha='center', va='center', fontsize=12)
    draw_arrow(ax, (7.6, 6.5), (8.35, 6.5))

    # z broadcast
    ax.text(10.2, 5.8, 'z broadcast\nto all T', fontsize=7, ha='center', style='italic')
    draw_arrow(ax, (4.3, 4.5), (6.5, 4.5), connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (7, 4.5), (7, 5.5))
    draw_rounded_box(ax, 7, 5.2, 1, 0.5, 'z', COLORS['latent'], fontsize=8)
    draw_arrow(ax, (7.5, 5.4), (8.4, 6.2), connectionstyle='arc3,rad=-0.2')

    # Concatenated input
    draw_rounded_box(ax, 10.2, 6.5, 1.8, 0.7, '[xₜ; z]\n(B,T,4+z)', COLORS['input'], fontsize=7)
    draw_arrow(ax, (9.15, 6.5), (9.25, 6.5))

    # Decoder GRU
    draw_rounded_box(ax, 8.75, 4.5, 2, 0.9, 'GRU\n(in=4+z, hid=10)', COLORS['decoder'], fontsize=8, bold=True)
    draw_arrow(ax, (10.2, 6.1), (10.2, 5.5))
    draw_arrow(ax, (10.2, 5.2), (9.5, 4.7), connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (8.75, 7.9), (8.75, 5))

    # GRU output
    draw_rounded_box(ax, 8.75, 3.2, 1.8, 0.7, 'oₜ\n(B,T,10)', COLORS['decoder'], fontsize=8)
    draw_arrow(ax, (8.75, 4), (8.75, 3.6))

    # Linear layer
    draw_rounded_box(ax, 8.75, 2, 2, 0.7, 'Linear\n(10 → 2)', COLORS['decoder'], fontsize=8)
    draw_arrow(ax, (8.75, 2.8), (8.75, 2.4))

    # Logits
    draw_rounded_box(ax, 8.75, 1, 1.8, 0.7, 'logits\n(B,T,2)', COLORS['output'], fontsize=8, bold=True)
    draw_arrow(ax, (8.75, 1.6), (8.75, 1.4))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['encoder'], edgecolor='black', label='Encoder', alpha=0.6),
        mpatches.Patch(facecolor=COLORS['decoder'], edgecolor='black', label='Decoder', alpha=0.6),
        mpatches.Patch(facecolor=COLORS['latent'], edgecolor='black', label='Latent Space', alpha=0.8),
        mpatches.Patch(facecolor=COLORS['input'], edgecolor='black', label='Input', alpha=0.8),
        mpatches.Patch(facecolor=COLORS['output'], edgecolor='black', label='Output', alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

    # Dimension annotations
    ax.text(0.3, 0.5, 'Dimensions:\nB = batch (participants)\nT = trials\nz = latent dim (1-5)\nhid = 10',
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure2_detailed_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def load_loss_data(dataset_id=0, seeds=[12, 50, 76, 100, 142]):
    """Load actual loss data from training runs."""
    import glob

    all_epochs = []
    all_losses = []

    for seed in seeds:
        loss_dir = f"runs_dataset{dataset_id}/seed_{seed}/loss"
        if not os.path.exists(loss_dir):
            continue

        loss_files = sorted(glob.glob(os.path.join(loss_dir, "epoch_*.npy")))
        epochs = []
        losses = []

        for lf in loss_files:
            epoch = int(os.path.basename(lf).replace("epoch_", "").replace(".npy", ""))
            loss_val = np.load(lf).item()
            epochs.append(epoch)
            losses.append(loss_val)

        if epochs:
            all_epochs.append(np.array(epochs))
            all_losses.append(np.array(losses))

    return all_epochs, all_losses


def load_rsa_trajectory(dataset_id=0, seeds=[12, 50, 76, 100, 142]):
    """Load RSA correlation trajectory over epochs."""
    import glob
    import pandas as pd
    from scipy.spatial.distance import pdist

    # Load ground truth parameters
    data_dir = f"data_dataset{dataset_id}"
    test_params = pd.read_csv(f"{data_dir}/true_test_parameter_values.csv")
    params = test_params["alphaP_list"].values.reshape(-1, 1)
    params_dist = pdist(params, metric="euclidean")

    all_epochs = []
    all_rsa_corrs = []

    for seed in seeds:
        rsa_dir = f"runs_dataset{dataset_id}/seed_{seed}/rsa"
        if not os.path.exists(rsa_dir):
            continue

        rsa_files = sorted(glob.glob(os.path.join(rsa_dir, "epoch_*.npy")))
        epochs = []
        rsa_corrs = []

        for rf in rsa_files:
            epoch = int(os.path.basename(rf).replace("epoch_", "").replace(".npy", ""))
            vec_latents = np.load(rf)
            corr = np.corrcoef(params_dist, vec_latents)[0, 1]
            epochs.append(epoch)
            rsa_corrs.append(corr)

        if epochs:
            all_epochs.append(np.array(epochs))
            all_rsa_corrs.append(np.array(rsa_corrs))

    return all_epochs, all_rsa_corrs


def load_hp_search_data(csv_path="hp_search_summary.csv"):
    """Load hyperparameter search results from CSV."""
    import pandas as pd

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    return df


def run_causal_inference_example(dataset_id=0, seed=12, participant_idx=0):
    """
    Run causal inference on a single participant to get z(t) trajectory.
    Returns mu(t), sigma(t), and ground truth parameter.
    """
    import torch
    import json
    import pandas as pd
    from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best epoch/seed from specificity criterion
    best_epoch_path = f"runs_dataset{dataset_id}/best_epoch_by_specificity.json"
    if not os.path.exists(best_epoch_path):
        print(f"Best epoch file not found: {best_epoch_path}")
        return None, None, None, None

    with open(best_epoch_path, 'r') as f:
        best_info = json.load(f)

    best_epoch = best_info['best_epoch']
    best_seed = best_info['best_seed']
    best_specificity = best_info['best_specificity']

    # Load config from best seed's run
    run_dir = f"runs_dataset{dataset_id}/seed_{best_seed}"
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return None, None, None, None

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get model parameters
    in_dim = config.get('in_dim', 4)
    z_dim = config.get('z_dim', 1)
    hidden = config.get('hidden', 10)
    A = config.get('A', 2)
    lmbd = config.get('lmbd', 'N/A')

    # Prepare model info for annotation
    model_info = {
        'dataset_id': dataset_id,
        'seed': best_seed,
        'epoch': best_epoch,
        'z_dim': z_dim,
        'lmbd': lmbd,
        'specificity': best_specificity
    }

    # Load data
    data_dir = f"data_dataset{dataset_id}"
    xin_test = np.load(f"{data_dir}/xin_test.npy")
    test_params = pd.read_csv(f"{data_dir}/true_test_parameter_values.csv")
    true_alpha = test_params["alphaP_list"].values[participant_idx]

    # Get one participant's data
    x_participant = torch.from_numpy(xin_test[participant_idx:participant_idx+1]).float().to(device)
    T = x_participant.shape[1]

    # Build model
    encoder = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=hidden)
    decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A)
    model = LatentRNN_secondstep(encoder=encoder, hid=hidden, z_dim=z_dim, in_dim=in_dim, A=A, decoder=decoder)

    # Load best checkpoint by specificity
    ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_epoch}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Best checkpoint not found: {ckpt_path}")
        return None, None, None, None
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Run causal inference: get z estimate at each time step
    mu_trajectory = []
    sigma_trajectory = []

    with torch.no_grad():
        for t in range(1, T + 1):
            # Only use data up to time t
            x_up_to_t = x_participant[:, :t, :]  # (1, t, in_dim)
            x_batched = x_up_to_t.unsqueeze(1)    # (1, 1, t, in_dim)

            mu_t, logvar_t = model.encoder(x_batched, return_per_timestep=True)
            # Get the estimate at the last time step
            mu_final = mu_t[0, 0, -1].cpu().numpy()  # (z_dim,)
            sigma_final = np.exp(0.5 * logvar_t[0, 0, -1].cpu().numpy())

            # For z_dim > 1, take the first dimension or norm
            if z_dim > 1:
                mu_trajectory.append(np.linalg.norm(mu_final))
                sigma_trajectory.append(np.mean(sigma_final))
            else:
                mu_trajectory.append(mu_final.item() if hasattr(mu_final, 'item') else mu_final[0])
                sigma_trajectory.append(sigma_final.item() if hasattr(sigma_final, 'item') else sigma_final[0])

    return np.array(mu_trajectory), np.array(sigma_trajectory), true_alpha, T


def figure3_training_dynamics(output_dir):
    """
    Figure 3: Training Dynamics
    Uses ACTUAL data from training runs and hyperparameter search.
    """
    fig = plt.figure(figsize=(14, 8))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ========== Panel A: Training Pipeline Timeline ==========
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.set_xlim(0, 10)
    ax_timeline.set_ylim(0, 3)
    ax_timeline.axis('off')
    ax_timeline.set_title('A. Two-Step Training Pipeline', fontsize=12, fontweight='bold', loc='left')

    # Step 1 box
    step1_box = FancyBboxPatch(
        (0.2, 1), 3, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['encoder'], edgecolor='black', linewidth=2, alpha=0.3
    )
    ax_timeline.add_patch(step1_box)
    ax_timeline.text(1.7, 2.3, 'Step 1', fontsize=11, ha='center', fontweight='bold')
    ax_timeline.text(1.7, 1.8, 'Pre-train Decoder', fontsize=9, ha='center')
    ax_timeline.text(1.7, 1.4, '• Learn zᵢ embeddings', fontsize=8, ha='center')
    ax_timeline.text(1.7, 1.1, '• Train policy decoder', fontsize=8, ha='center')

    # Arrow
    draw_arrow(ax_timeline, (3.3, 1.75), (3.7, 1.75), style='-|>', color=COLORS['arrow'])

    # Step 2 box
    step2_box = FancyBboxPatch(
        (3.8, 1), 3.2, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['decoder'], edgecolor='black', linewidth=2, alpha=0.3
    )
    ax_timeline.add_patch(step2_box)
    ax_timeline.text(5.4, 2.3, 'Step 2', fontsize=11, ha='center', fontweight='bold')
    ax_timeline.text(5.4, 1.8, 'Train IDRNN Encoder', fontsize=9, ha='center')
    ax_timeline.text(5.4, 1.4, '• Freeze decoder', fontsize=8, ha='center')
    ax_timeline.text(5.4, 1.1, '• Learn μ(t), σ(t) from history', fontsize=8, ha='center')

    # Arrow
    draw_arrow(ax_timeline, (7.1, 1.75), (7.5, 1.75), style='-|>', color=COLORS['arrow'])

    # Inference box
    infer_box = FancyBboxPatch(
        (7.6, 1), 2.2, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['latent'], edgecolor='black', linewidth=2, alpha=0.3
    )
    ax_timeline.add_patch(infer_box)
    ax_timeline.text(8.7, 2.3, 'Inference', fontsize=11, ha='center', fontweight='bold')
    ax_timeline.text(8.7, 1.8, 'Causal Posterior', fontsize=9, ha='center')
    ax_timeline.text(8.7, 1.4, '• Use only x₁:ₜ', fontsize=8, ha='center')
    ax_timeline.text(8.7, 1.1, '• Sample z ~ q(z|x₁:ₜ)', fontsize=8, ha='center')

    # ========== Panel B: RSA Correlation over Training (REAL DATA) ==========
    ax_loss = fig.add_subplot(gs[1, 0])

    # Load actual RSA trajectory
    all_epochs, all_rsa_corrs = load_rsa_trajectory(dataset_id=0, seeds=[12, 50, 76, 100, 142])

    if all_epochs and all_rsa_corrs:
        # Plot individual seeds with transparency
        for i, (epochs, rsa_corrs) in enumerate(zip(all_epochs, all_rsa_corrs)):
            sorted_idx = np.argsort(epochs)
            ax_loss.plot(epochs[sorted_idx], rsa_corrs[sorted_idx],
                        color=COLORS['encoder'], alpha=0.3, linewidth=1)

        # Compute and plot mean trajectory
        # Interpolate all seeds to common epoch grid
        common_epochs = np.arange(100, 10001, 100)
        interpolated = []
        for epochs, rsa_corrs in zip(all_epochs, all_rsa_corrs):
            sorted_idx = np.argsort(epochs)
            interp_vals = np.interp(common_epochs, epochs[sorted_idx], rsa_corrs[sorted_idx])
            interpolated.append(interp_vals)

        if interpolated:
            mean_rsa = np.mean(interpolated, axis=0)
            std_rsa = np.std(interpolated, axis=0)
            ax_loss.fill_between(common_epochs, mean_rsa - std_rsa, mean_rsa + std_rsa,
                                color=COLORS['encoder'], alpha=0.2)
            ax_loss.plot(common_epochs, mean_rsa, color=COLORS['encoder'],
                        linewidth=2.5, label='Mean RSA (±1 SD)')

        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('RSA Correlation with GT')
        ax_loss.set_title('B. RSA Correlation During Training', fontsize=11, fontweight='bold', loc='left')
        ax_loss.legend(loc='lower right', fontsize=8)
        ax_loss.set_xlim(0, 10000)
        ax_loss.text(0.95, 0.05, f'N={len(all_epochs)} seeds', transform=ax_loss.transAxes,
                    fontsize=8, ha='right', va='bottom', style='italic')

        # Add annotation clarifying this is IDRNN-inferred latents
        # Load config to get model details
        import json
        config_path = "runs_dataset0/seed_12/config.json"
        model_info = "IDRNN-inferred z (Step 2)"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            lmbd = config.get('lmbd', 'N/A')
            z_dim = config.get('z_dim', 'N/A')
            model_info += f"\nDataset 0, λ={lmbd}, z_dim={z_dim}"

        ax_loss.text(0.03, 0.97, model_info, transform=ax_loss.transAxes,
                    fontsize=7, ha='left', va='top', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.8))
    else:
        ax_loss.text(0.5, 0.5, 'No training data found', transform=ax_loss.transAxes,
                    ha='center', va='center', fontsize=10)
        ax_loss.set_title('B. RSA Correlation During Training', fontsize=11, fontweight='bold', loc='left')

    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)

    # ========== Panel C: Lambda Trade-off (REAL DATA from hp_search_summary.csv) ==========
    ax_lambda = fig.add_subplot(gs[1, 1])

    hp_df = load_hp_search_data("hp_search_summary.csv")

    if hp_df is not None and len(hp_df) > 0:
        # Plot for each z_dim
        z_dims = sorted(hp_df['z_dim'].unique())
        colors_z = {1: COLORS['loss_total'], 2: COLORS['highlight'], 3: COLORS['loss_ce']}
        markers_z = {1: 'o', 2: 's', 3: '^'}

        for z_dim in z_dims:
            subset = hp_df[hp_df['z_dim'] == z_dim].sort_values('lambda')
            color = colors_z.get(z_dim, 'gray')
            marker = markers_z.get(z_dim, 'o')

            ax_lambda.errorbar(subset['lambda'], subset['mean_best_rsa'],
                              yerr=subset['std_best_rsa'],
                              fmt=f'{marker}-', color=color, linewidth=2,
                              markersize=6, capsize=3, label=f'z_dim={z_dim}')

        # Mark optimal region
        best_row = hp_df.loc[hp_df['mean_best_rsa'].idxmax()]
        ax_lambda.axvline(x=best_row['lambda'], color='gray', linestyle='--', alpha=0.5)
        ax_lambda.annotate(f"Best: λ={best_row['lambda']}, z={int(best_row['z_dim'])}\nRSA={best_row['mean_best_rsa']:.3f}",
                          xy=(best_row['lambda'], best_row['mean_best_rsa']),
                          xytext=(best_row['lambda'] + 0.15, best_row['mean_best_rsa'] - 0.05),
                          fontsize=7, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

        ax_lambda.set_xlabel('λ (KL weight)')
        ax_lambda.set_ylabel('Best RSA Correlation')
        ax_lambda.set_title('C. Hyperparameter Search Results', fontsize=11, fontweight='bold', loc='left')
        ax_lambda.legend(loc='lower left', fontsize=8)
        ax_lambda.set_xlim(0, 1)
    else:
        ax_lambda.text(0.5, 0.5, 'No HP search data found', transform=ax_lambda.transAxes,
                      ha='center', va='center', fontsize=10)
        ax_lambda.set_title('C. Hyperparameter Search Results', fontsize=11, fontweight='bold', loc='left')

    ax_lambda.spines['top'].set_visible(False)
    ax_lambda.spines['right'].set_visible(False)

    # ========== Panel D: Causal Inference - z(t) Evolution (REAL DATA) ==========
    ax_causal = fig.add_subplot(gs[1, 2])

    # Try to run causal inference on a real participant
    try:
        mu_traj, sigma_traj, true_alpha, T = run_causal_inference_example(
            dataset_id=0, seed=12, participant_idx=100  # Pick a participant
        )

        if mu_traj is not None:
            trials = np.arange(1, T + 1)

            # Plot model's uncertainty (σ) as shaded region
            ax_causal.fill_between(trials, mu_traj - sigma_traj, mu_traj + sigma_traj,
                                   color=COLORS['latent'], alpha=0.3, label='σ(t)')
            # Plot mean trajectory
            ax_causal.plot(trials, mu_traj, color=COLORS['encoder'], linewidth=2, label='μ(t)')

            ax_causal.set_xlabel('Trial t')
            ax_causal.set_ylabel('Inferred z')
            ax_causal.set_title(f'D. Causal Inference (Participant, α={true_alpha:.2f})',
                               fontsize=11, fontweight='bold', loc='left')
            ax_causal.legend(loc='upper right', fontsize=8)
            ax_causal.set_xlim(0, T)
        else:
            raise ValueError("Could not load model")

    except Exception as e:
        print(f"Could not run causal inference: {e}")
        # Fallback to simulated data with clear label
        np.random.seed(42)
        T = 100
        trials = np.arange(T)
        true_z = 0.7

        mu_t = true_z + 0.5 * np.exp(-trials/20) + 0.1 * np.random.randn(T) * np.exp(-trials/30)
        sigma_t = 0.3 * np.exp(-trials/25) + 0.05

        ax_causal.fill_between(trials, mu_t - sigma_t, mu_t + sigma_t,
                               color=COLORS['latent'], alpha=0.3, label='σ(t)')
        ax_causal.plot(trials, mu_t, color=COLORS['encoder'], linewidth=2, label='μ(t)')

        ax_causal.set_xlabel('Trial t')
        ax_causal.set_ylabel('Inferred z')
        ax_causal.set_title('D. Causal Inference: z(t) Evolution (Simulated)',
                           fontsize=11, fontweight='bold', loc='left')
        ax_causal.legend(loc='upper right', fontsize=8)
        ax_causal.set_xlim(0, T)

    ax_causal.spines['top'].set_visible(False)
    ax_causal.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure3_training_dynamics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all figures."""
    output_dir = "plots/architecture"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating IDRNN architecture figures...")
    print("=" * 50)

    figure1_two_step_training(output_dir)
    figure2_detailed_architecture(output_dir)
    figure3_training_dynamics(output_dir)

    print("=" * 50)
    print(f"All figures saved to: {output_dir}/")
    print("  - figure1_two_step_training.png/pdf")
    print("  - figure2_detailed_architecture.png/pdf")
    print("  - figure3_training_dynamics.png/pdf")


if __name__ == "__main__":
    main()
