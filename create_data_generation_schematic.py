"""
Create a schematic diagram for the data generation process.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_schematic():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    color_bandit = '#E8F4FD'      # Light blue
    color_params = '#FFF3E0'      # Light orange
    color_qlearn = '#E8F5E9'      # Light green
    color_output = '#F3E5F5'      # Light purple
    border_color = '#333333'
    arrow_color = '#666666'

    # =========================================================================
    # Title
    # =========================================================================
    ax.text(7, 9.5, 'Data Generation Process', fontsize=18, fontweight='bold',
            ha='center', va='center')

    # =========================================================================
    # Box 1: Two-Armed Bandit Environment
    # =========================================================================
    bandit_box = FancyBboxPatch((0.5, 6.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                                 facecolor=color_bandit, edgecolor=border_color, linewidth=2)
    ax.add_patch(bandit_box)

    ax.text(2.25, 8.7, 'Two-Armed Bandit', fontsize=12, fontweight='bold',
            ha='center', va='center')
    ax.text(2.25, 8.2, 'Environment', fontsize=12, fontweight='bold',
            ha='center', va='center')

    # Bandit details
    ax.text(2.25, 7.6, 'Reversing reward probabilities', fontsize=9, ha='center', style='italic')
    ax.text(2.25, 7.2, r'$p_{high}$ vs $p_{low}$ (flip every 50 trials)', fontsize=9, ha='center')

    # Draw simple bandit arms
    ax.plot([1.3, 1.3], [6.7, 7.0], color='#2196F3', linewidth=8, solid_capstyle='round')
    ax.plot([3.2, 3.2], [6.7, 7.0], color='#FF9800', linewidth=8, solid_capstyle='round')
    ax.text(1.3, 6.55, 'A', fontsize=8, ha='center', fontweight='bold')
    ax.text(3.2, 6.55, 'B', fontsize=8, ha='center', fontweight='bold')

    # =========================================================================
    # Box 2: Individual Differences (Alpha sampling)
    # =========================================================================
    params_box = FancyBboxPatch((5, 6.5), 4, 2.5, boxstyle="round,pad=0.05",
                                 facecolor=color_params, edgecolor=border_color, linewidth=2)
    ax.add_patch(params_box)

    ax.text(7, 8.7, 'Individual Differences', fontsize=12, fontweight='bold',
            ha='center', va='center')

    ax.text(7, 8.0, r'Learning rate $\alpha_i \sim \mathrm{Uniform}(0.1, 0.9)$',
            fontsize=10, ha='center')
    ax.text(7, 7.4, r'$\beta = 3.0$ (fixed)', fontsize=10, ha='center')
    ax.text(7, 6.85, r'$N = 200$ simulated participants', fontsize=9, ha='center', style='italic')

    # =========================================================================
    # Box 3: Q-Learning Model
    # =========================================================================
    qlearn_box = FancyBboxPatch((10, 6.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                                 facecolor=color_qlearn, edgecolor=border_color, linewidth=2)
    ax.add_patch(qlearn_box)

    ax.text(11.75, 8.7, 'Q-Learning Model', fontsize=12, fontweight='bold',
            ha='center', va='center')

    # Q-learning equations
    ax.text(11.75, 8.0, r'$Q_{t+1} = Q_t + \alpha \cdot \delta_t$', fontsize=10, ha='center')
    ax.text(11.75, 7.5, r'$\delta_t = r_t - Q_t$', fontsize=10, ha='center')
    ax.text(11.75, 6.9, r'$P(a) = \frac{e^{\beta Q_a}}{\sum_j e^{\beta Q_j}}$', fontsize=10, ha='center')

    # =========================================================================
    # Arrows connecting top boxes
    # =========================================================================
    # Bandit -> Q-Learning
    ax.annotate('', xy=(10, 7.75), xytext=(4, 7.75),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    ax.text(4.5, 8.1, 'rewards', fontsize=9, ha='left', style='italic', color=arrow_color)

    # Params -> Q-Learning
    ax.annotate('', xy=(10, 7.75), xytext=(9, 7.75),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    ax.text(9.2, 8.1, r'$\alpha_i, \beta$', fontsize=9, ha='left', style='italic', color=arrow_color)

    # =========================================================================
    # Box 4: Choice Generation (center, below)
    # =========================================================================
    choice_box = FancyBboxPatch((4.5, 3.5), 5, 2),
    choice_box = FancyBboxPatch((4.5, 3.5), 5, 2, boxstyle="round,pad=0.05",
                                 facecolor='#FFFDE7', edgecolor=border_color, linewidth=2)
    ax.add_patch(choice_box)

    ax.text(7, 5.2, 'Choice Generation', fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 4.5, r'$c_t \sim \mathrm{Categorical}(P(a=0), P(a=1))$', fontsize=11, ha='center')
    ax.text(7, 3.9, '200 trials per participant', fontsize=9, ha='center', style='italic')

    # Arrow from Q-Learning to Choice
    ax.annotate('', xy=(7, 5.5), xytext=(11.75, 6.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # =========================================================================
    # Box 5: Output Data
    # =========================================================================
    output_box = FancyBboxPatch((2, 0.5), 10, 2.3, boxstyle="round,pad=0.05",
                                 facecolor=color_output, edgecolor=border_color, linewidth=2)
    ax.add_patch(output_box)

    ax.text(7, 2.5, 'Output: Behavioral Data', fontsize=12, fontweight='bold', ha='center')

    # Output details in columns
    ax.text(4, 1.8, r'$\mathbf{X}_{in}$: [choice, reward]', fontsize=10, ha='center')
    ax.text(4, 1.3, 'shape: (N, T, 4)', fontsize=9, ha='center', style='italic')

    ax.text(7, 1.8, r'$\mathbf{c}$: choices', fontsize=10, ha='center')
    ax.text(7, 1.3, 'shape: (N, T)', fontsize=9, ha='center', style='italic')

    ax.text(10, 1.8, r'$\alpha_i$: ground truth', fontsize=10, ha='center')
    ax.text(10, 1.3, 'individual differences', fontsize=9, ha='center', style='italic')

    # Arrow from Choice to Output
    ax.annotate('', xy=(7, 2.8), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # =========================================================================
    # Legend / Key insight box
    # =========================================================================
    insight_box = FancyBboxPatch((0.5, 0.5), 1.3, 2.3, boxstyle="round,pad=0.03",
                                  facecolor='#FAFAFA', edgecolor='#999999', linewidth=1)
    ax.add_patch(insight_box)
    ax.text(1.15, 2.5, 'Key:', fontsize=9, fontweight='bold', ha='center')
    ax.text(1.15, 2.0, r'$\alpha_i$ varies', fontsize=8, ha='center')
    ax.text(1.15, 1.6, 'across', fontsize=8, ha='center')
    ax.text(1.15, 1.2, 'participants', fontsize=8, ha='center')
    ax.text(1.15, 0.8, r'$\rightarrow$ ID', fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig('plots/data_generation_schematic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('plots/data_generation_schematic.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved schematic to plots/data_generation_schematic.png and .pdf")
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    create_schematic()
