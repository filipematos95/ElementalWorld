import tensorflow as tf
import numpy as np

class EcosystemModel:
    def __init__(self, n_species=3, height=50, width=50):
        self.H = height
        self.W = width
        self.N_SPP = n_species
        self.N_CHANNELS = 4 + (5 * self.N_SPP) + self.N_SPP
        
        # === SPECIES TRAITS (Dynamic Initialization) ===
        
        # 1. Create default values based on N_SPP
        # (Linear gradient from fast to slow as default)
        default_growth = np.linspace(0.25, 0.05, self.N_SPP).astype(np.float32)
        default_mort = np.linspace(0.02, 0.005, self.N_SPP).astype(np.float32)
        default_seed = np.linspace(0.05, 0.01, self.N_SPP).astype(np.float32)
        
        # 2. Create Variables with CORRECT SHAPE (N,)
        self.growth_rates = tf.Variable(default_growth, dtype=tf.float32)
        self.mort_rates = tf.Variable(default_mort, dtype=tf.float32)
        self.seed_probs = tf.Variable(default_seed, dtype=tf.float32)
        
        # 3. Stoichiometry (Dynamic)
        # Gradient of C:N from 15 (wasteful) to 60 (efficient)
        cn_ratios = np.linspace(15.0, 60.0, self.N_SPP)
        cp_ratios = np.linspace(100.0, 600.0, self.N_SPP)
        
        stoich_list = []
        for i in range(self.N_SPP):
            f_C = 0.45
            f_N = f_C / cn_ratios[i]
            f_P = f_C / cp_ratios[i]
            f_O = 0.42
            f_Other = 1.0 - (f_C + f_N + f_P + f_O)
            stoich_list.append([f_C, f_O, f_N, f_P, f_Other])
            
        self.stoich = tf.Variable(stoich_list, dtype=tf.float32) # Shape (N, 5)

        # Global Params
        self.params = {
            'K': 1.2,
            'input_N': 0.002,
            'input_P': 0.001,
            'leach_N': 0.01,
            'leach_P': 0.005
        }
        
        self.kernel = tf.ones([3, 3, self.N_CHANNELS, 1], dtype=tf.float32)
        self.grid = None

    def initialize_grid(self):
        # Soil
        soil_C = tf.random.uniform((self.H, self.W), 0.2, 0.4)
        soil_O = tf.random.uniform((self.H, self.W), 0.4, 0.6)
        soil_N = tf.random.uniform((self.H, self.W), 0.8, 1.0)
        soil_P = tf.random.uniform((self.H, self.W), 0.6, 0.9)
        
        # Everything else zero
        zeros = tf.zeros((self.H, self.W, self.N_CHANNELS - 4))
        
        init_tensor = tf.concat([
            tf.stack([soil_C, soil_O, soil_N, soil_P], axis=-1),
            zeros
        ], axis=-1)
        
        self.grid = tf.Variable(init_tensor, trainable=False)
        print(f"Initialized N={self.N_SPP} model.")

    @tf.function
    def step(self):
        # 1. Unpack Grid
        current_grid = self.grid
        
        # Soil: (H, W, 4)
        soil = current_grid[:, :, :4]
        soil_C, soil_O, soil_N, soil_P = tf.unstack(soil, axis=-1)
        
        # Species Pools: Reshape to (H, W, N, 5)
        # Slice from index 4 to (4 + 5*N)
        pool_end = 4 + (5 * self.N_SPP)
        spp_flat = current_grid[:, :, 4:pool_end]
        spp_pools = tf.reshape(spp_flat, (self.H, self.W, self.N_SPP, 5))
        
        # C is index 0 in the last dim (..., 0)
        spp_C = spp_pools[:, :, :, 0]
        
        # 2. Calculate Biomass (H, W, N)
        # stoich[:, 0] is f_C for each species
        # We need to broadcast f_C (shape N) over (H, W, N)
        f_C_vec = self.stoich[:, 0] 
        biomass = spp_C / f_C_vec
        
        total_biomass = tf.reduce_sum(biomass, axis=-1) # (H, W)
        
        # 3. Interactions (Space)
        space_factor = tf.maximum(0.0, 1.0 - (total_biomass / self.params['K']))
        space_factor = tf.expand_dims(space_factor, -1) # (H, W, 1) for broadcasting
        
        # 4. Colonization (Seed Rain)
        rand = tf.random.uniform(tf.shape(spp_C))
        # Seed if random < prob AND current biomass is near zero
        is_seed = tf.cast(rand < self.seed_probs, tf.float32) * tf.cast(spp_C < 0.001, tf.float32)
        spp_C_seeded = spp_C + (is_seed * 0.05 * f_C_vec)
        
        # 5. Growth (Vectorized)
        # Growth = rate * C * space
        growth = self.growth_rates * spp_C_seeded * space_factor
        spp_C_new = spp_C_seeded + growth
        
        # Update Pools temporarily (assuming balanced growth for C)
        # We need to update O, N, P, Other proportionally to C growth
        # Or easier: Re-calculate all pools from C based on stoichiometry
        # But we need to track independent N/P limitation. 
        
        # --- NUTRIENT LOGIC ---
        # Target Biomass based on new C
        target_B = spp_C_new / f_C_vec
        
        # Calculate Demands (H, W, N)
        # demand_N = (target_B * f_N) - current_N
        current_N = spp_pools[:, :, :, 2]
        current_P = spp_pools[:, :, :, 3]
        
        f_N_vec = self.stoich[:, 2]
        f_P_vec = self.stoich[:, 3]
        
        dem_N = (target_B * f_N_vec) - current_N
        dem_P = (target_B * f_P_vec) - current_P
        
        # Sum demands across all species (H, W)
        total_dem_N = tf.reduce_sum(tf.maximum(dem_N, 0.), axis=-1)
        total_dem_P = tf.reduce_sum(tf.maximum(dem_P, 0.), axis=-1)
        
        # Scaling Factors (Soil arbitration)
        avail_N = soil_N * 0.5
        avail_P = soil_P * 0.5
        
        scale_N = tf.minimum(1.0, avail_N / (total_dem_N + 1e-9))
        scale_P = tf.minimum(1.0, avail_P / (total_dem_P + 1e-9))
        
        # Broadcast scale factors (H, W, 1)
        scale_N = tf.expand_dims(scale_N, -1)
        scale_P = tf.expand_dims(scale_P, -1)
        
        # Apply Uptake
        uptake_N = tf.maximum(dem_N, 0.) * scale_N
        uptake_P = tf.maximum(dem_P, 0.) * scale_P
        
        new_N = current_N + uptake_N
        new_P = current_P + uptake_P
        
        # 6. Stoichiometry (Liebig's Law Vectorized)
        # Potential biomass from C, N, P
        pot_from_C = spp_C_new / f_C_vec
        pot_from_N = new_N / f_N_vec
        pot_from_P = new_P / f_P_vec
        
        # Min across potential resources
        max_biomass = tf.minimum(tf.minimum(pot_from_C, pot_from_N), pot_from_P)
        
        # 7. Mortality
        # loss = 1 - mortality - respiration
        loss_rates = 1.0 - self.mort_rates - 0.01
        max_biomass = max_biomass * loss_rates
        
        # 8. Reconstruct Pools (H, W, N, 5)
        # New pools = max_biomass * stoichiometry
        # Reshape max_biomass to (H, W, N, 1) to broadcast against stoich (N, 5)
        mb_expanded = tf.expand_dims(max_biomass, -1)
        # stoich_expanded = tf.reshape(self.stoich, (1, 1, self.N_SPP, 5))
        
        new_pools = mb_expanded * self.stoich
        
        # 9. Soil Update
        # Sum uptake across species
        total_up_N = tf.reduce_sum(uptake_N, axis=-1)
        total_up_P = tf.reduce_sum(uptake_P, axis=-1)
        
        # Leaching
        loss_N = self.params['leach_N'] * soil_N
        loss_P = self.params['leach_P'] * soil_P
        
        soil_N_new = tf.clip_by_value(soil_N + self.params['input_N'] - total_up_N - loss_N, 0., 1.)
        soil_P_new = tf.clip_by_value(soil_P + self.params['input_P'] - total_up_P - loss_P, 0., 1.)
        
        # 10. Pack Grid
        # Flatten new_pools back to (H, W, 5*N)
        new_pools_flat = tf.reshape(new_pools, (self.H, self.W, 5*self.N_SPP))
        
        # Metrics: Just biomass (H, W, N)
        # We use max_biomass derived earlier
        
        final_stack = tf.concat([
            tf.stack([soil_C, soil_O, soil_N_new, soil_P_new], axis=-1),
            new_pools_flat,
            max_biomass
        ], axis=-1)
        
        self.grid.assign(final_stack)
        return self.grid
        
    def get_state(self):
        return self.grid.numpy()