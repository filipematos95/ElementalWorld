import matplotlib.pyplot as plt
import numpy as np

def plot_state(grid, timepoint='', save_path=None):
    """
    Comprehensive visualization for the 2-Species Interaction Model.
    Grid Shape: (H, W, 18)

    Layout:
      Row 1: Soil Pools (C, O, N, P)
      Row 2: Species A (Pioneer) - Biomass + Elemental Pools
      Row 3: Species B (Conservative) - Biomass + Elemental Pools
    """
    if hasattr(grid, 'numpy'):
        data = grid.numpy()
    else:
        data = grid

    # Create 3 rows x 5 columns
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # === Define What to Plot ===
    # Format: (Row, Col, Channel_Index, Title, Colormap)

    # --- ROW 1: SOIL ---
    plots = [
        (0, 0, 0, 'Soil Carbon', 'YlOrBr'),
        (0, 1, 1, 'Soil Oxygen', 'Oranges'),
        (0, 2, 2, 'Soil Nitrogen', 'Greens'),
        (0, 3, 3, 'Soil Phosphorus', 'Blues'),
        (0, 4, -1, 'Total Biomass (A+B)', 'viridis'), # Calculated on fly
    ]

    # --- ROW 2: SPECIES A (Pioneer) ---
    # Indices 4-8 are A_C, A_O, A_N, A_P, A_Other
    # Index 14 is A_Biomass
    plots += [
        (1, 0, 14, 'Spp A Biomass (Pioneer)', 'Greens'),
        (1, 1, 4,  'Spp A Carbon', 'YlGn'),
        (1, 2, 6,  'Spp A Nitrogen', 'GnBu'),
        (1, 3, 7,  'Spp A Phosphorus', 'PuBu'),
        (1, 4, 8,  'Spp A Other', 'Purples'),
    ]

    # --- ROW 3: SPECIES B (Conservative) ---
    # Indices 9-13 are B_C, B_O, B_N, B_P, B_Other
    # Index 15 is B_Biomass
    plots += [
        (2, 0, 15, 'Spp B Biomass (Conserv)', 'Blues'),
        (2, 1, 9,  'Spp B Carbon', 'PuBu'),
        (2, 2, 11, 'Spp B Nitrogen', 'YlGnBu'),
        (2, 3, 12, 'Spp B Phosphorus', 'BuPu'),
        (2, 4, 13, 'Spp B Other', 'RdPu'),
    ]

    # === RENDER PLOTS ===
    for (row, col, ch_idx, title, cmap) in plots:
        ax = axes[row, col]

        # Get Channel Data
        if ch_idx == -1:
            # Special Case: Sum of Biomasses (A + B)
            val = data[:, :, 14] + data[:, :, 15]
        else:
            val = data[:, :, ch_idx]

        # --- Dynamic Scaling (Fixes the "Black Screen" bug) ---
        max_val = val.max()
        if max_val < 0.1:
            # For trace elements (N, P), scale to data max
            v_max = max_val + 1e-6
        else:
            # For Biomass/Soil, scale to 1.0 (or slightly higher if overcrowded)
            v_max = max(1.0, max_val)

        im = ax.imshow(val, cmap=cmap, vmin=0, vmax=v_max)

        # Formatting
        ax.set_title(f"{title}\nMax: {max_val:.4f}", fontsize=10, fontweight='bold')
        ax.axis('off')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Stats Overlay
        ax.text(0.02, 0.98, f'μ={val.mean():.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.suptitle(f'Ecosystem State {timepoint}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_history(history):
    """
    Plots the time-series competition between A and B.
    Expects history list: [(mean_A, mean_B), ...]
    """
    A_vals = [x[0] for x in history]
    B_vals = [x[1] for x in history]
    steps = np.arange(len(history)) * 10 # Assuming 10-step intervals

    plt.figure(figsize=(12, 6))

    # Plot curves
    plt.plot(steps, A_vals, label='Species A (Pioneer)', color='green', linewidth=2.5)
    plt.plot(steps, B_vals, label='Species B (Conservative)', color='blue', linewidth=2.5)

    # Fill under curves for "Dominance Area" effect
    plt.fill_between(steps, A_vals, alpha=0.1, color='green')
    plt.fill_between(steps, B_vals, alpha=0.1, color='blue')

    plt.title("Competitive Dynamics: Pioneer vs Conservative", fontsize=14)
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Mean Grid Biomass", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.show()