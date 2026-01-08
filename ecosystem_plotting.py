import matplotlib.pyplot as plt
import numpy as np

def plot_ecosystem_aggregate(grid, timepoint='', save_path=None):
    """
    Plots Soil Pools vs. Total Biota Pools (Aggregated across all species).
    Works for N-Species model.
    """
    if hasattr(grid, 'numpy'):
        data = grid.numpy()
    else:
        data = grid

    # Determine N_Species based on channel count
    # Total = 4 + 5*N + N
    total_ch = data.shape[2]
    n_spp = (total_ch - 4) // 6
    
    # === AGGREGATE BIOTA POOLS ===
    # Pools start at index 4.
    # We have N blocks of 5 channels.
    # We want to sum every 0th (C), 1st (O), 2nd (N), 3rd (P), 4th (Other) across blocks.
    
    pool_start = 4
    pool_end = 4 + (5 * n_spp)
    pools_flat = data[:, :, pool_start:pool_end]
    
    # Reshape to (H, W, N_Species, 5_Elements)
    pools_3d = pools_flat.reshape(data.shape[0], data.shape[1], n_spp, 5)
    
    # Sum across Species Axis (axis 2)
    total_biota_pools = np.sum(pools_3d, axis=2) # Result: (H, W, 5) [C, O, N, P, Other]
    
    # Total Biomass (Sum of biomass metrics at the end)
    bio_start = pool_end
    total_biomass = np.sum(data[:, :, bio_start:], axis=-1)

    # === PLOTTING ===
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Soil
    soil_plots = [
        (0, 'Soil Carbon', 'YlOrBr'),
        (1, 'Soil Oxygen', 'Oranges'),
        (2, 'Soil Nitrogen', 'Greens'),
        (3, 'Soil Phosphorus', 'Blues')
    ]
    
    # Row 2: Total Biota (Sum of all species)
    biota_plots = [
        (0, 'Total Biota Carbon', 'YlGn'),
        (1, 'Total Biota Oxygen', 'Reds'), # O is often red/orange
        (2, 'Total Biota Nitrogen', 'GnBu'),
        (3, 'Total Biota Phos.', 'PuBu')
    ]

    # Render Soil
    for idx, title, cmap in soil_plots:
        ax = axes[0, idx]
        val = data[:, :, idx]
        im = ax.imshow(val, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
    # Render Biota
    for idx, title, cmap in biota_plots:
        ax = axes[1, idx]
        val = total_biota_pools[:, :, idx] # 0=C, 1=O, 2=N, 3=p
        
        # Dynamic Scaling
        vmax = max(0.1, val.max()) # Auto-scale but min 0.1
        
        im = ax.imshow(val, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nMax: {val.max():.4f}", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Ecosystem Aggregates {timepoint}\n(Sum of {n_spp} Species)", fontsize=16, y=0.98)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()


def plot_species_abundance(grid, timepoint='', save_path=None):
    """
    Plots the biomass distribution of each species separately.
    """
    if hasattr(grid, 'numpy'):
        data = grid.numpy()
    else:
        data = grid
        
    total_ch = data.shape[2]
    n_spp = (total_ch - 4) // 6
    
    # Calculate grid dimensions for subplots (e.g. 2x3 for 5 species)
    cols = min(n_spp, 4)
    rows = (n_spp + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1: axes = [axes] # Handle N=1 case
    axes = np.array(axes).flatten()
    
    # Biomass channels start after the pools
    bio_start = 4 + (5 * n_spp)
    
    # Find Global Max for unified scale comparison
    # (Optional: set vmax=None for independent scaling)
    global_max = data[:, :, bio_start:].max()
    
    for i in range(n_spp):
        ax = axes[i]
        ch_idx = bio_start + i
        val = data[:, :, ch_idx]
        
        # Use a distinct colormap for each species cyclically
        cmaps = ['Greens', 'Blues', 'Purples', 'Reds', 'Oranges']
        cmap = cmaps[i % len(cmaps)]
        
        im = ax.imshow(val, cmap=cmap, vmin=0, vmax=global_max)
        ax.set_title(f"Species {i}\nMax: {val.max():.3f}", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
    # Hide empty subplots
    for j in range(n_spp, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"Species Distributions {timepoint}", fontsize=16, y=0.99)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_community_map(grid, timepoint='', save_path=None):
    """
    Creates a single RGB map where each species maps to a color.
    Ideal for N=3 species (RGB), but works for N=1 to N=6.
    """
    if hasattr(grid, 'numpy'):
        data = grid.numpy()
    else:
        data = grid
        
    total_ch = data.shape[2]
    n_spp = (total_ch - 4) // 6
    
    # Locate Biomass Channels
    bio_start = 4 + (5 * n_spp)
    biomass_stack = data[:, :, bio_start:] # Shape (H, W, N)
    
    # Normalize biomass to 0-1 range for color intensity
    # We assume max biomass per cell ~1.2 (Carrying Capacity)
    norm_biomass = np.clip(biomass_stack / 1.0, 0, 1)
    H, W, _ = norm_biomass.shape
    
    # Create Output RGB Image (H, W, 3)
    rgb_map = np.zeros((H, W, 3))
    
    # Define Color Vectors for up to 6 species
    # Spp0=Red, Spp1=Green, Spp2=Blue, Spp3=Yellow, Spp4=Cyan, Spp5=Magenta
    colors = [
            np.array([1.0, 0.0, 0.0]), # 0: Red
            np.array([0.0, 1.0, 0.0]), # 1: Green
            np.array([0.0, 0.0, 1.0]), # 2: Blue
            np.array([1.0, 1.0, 0.0]), # 3: Yellow
            np.array([0.0, 1.0, 1.0]), # 4: Cyan
            np.array([1.0, 0.0, 1.0]), # 5: Magenta
            np.array([1.0, 0.5, 0.0]), # 6: Orange
            np.array([0.5, 1.0, 0.0]), # 7: Lime
            np.array([0.5, 0.0, 0.5]), # 8: Purple
            np.array([0.0, 0.5, 0.5]), # 9: Teal
        ]
        
    # Additive Mixing: Sum(Biomass * Color)
    for i in range(min(n_spp, 10)):
        # Extract species i biomass map (H, W)
        spp_map = norm_biomass[:, :, i]
        
        # Reshape to (H, W, 1) for broadcasting
        spp_map_exp = spp_map[:, :, np.newaxis]
        
        # Add its color contribution
        rgb_map += spp_map_exp * colors[i]
        
    # Clip to valid RGB range (in case of overlap > 1.0)
    rgb_map = np.clip(rgb_map, 0, 1)
    
    # PLOT
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_map)
    plt.title(f"Community Composition {timepoint}", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f'Species {i}') 
        for i in range(min(n_spp, 6))
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()