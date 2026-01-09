import tensorflow as tf
import numpy as np


class EcosystemModel:
    def __init__(self, 
                 n_species, 
                 height, 
                 width, 
                 growth_rates,
                 mort_rates,
                 seed_probs,
                 niche_centers,
                 niche_widths, 
                 niche_weights, 
                 params):
        
        # Grid and channels
        self.H = height
        self.W = width
        self.N_SPP = n_species
        # 4 soil + 5 pools per species + 1 biomass per species
        self.N_CHANNELS = 4 + (5 * self.N_SPP) + self.N_SPP
        
        # Life-history traits
        self.growth_rates = tf.Variable(growth_rates, dtype=tf.float32, name="Growth_Rates")
        self.mort_rates   = tf.Variable(mort_rates,   dtype=tf.float32, name="Mortality_Rates")
        self.seed_probs   = tf.Variable(seed_probs,   dtype=tf.float32, name="Seed_Probs")
        
        # Biogeochemical niche (in element space)
        self.niche_optimals = tf.Variable(niche_centers, dtype=tf.float32, name="Niche_Centers")
        self.niche_widths   = tf.Variable(niche_widths, dtype=tf.float32, name="Niche_Widths")
        self.niche_weights  = tf.constant(niche_weights, dtype=tf.float32, name="Niche_Weights")

        # Species mean elementome derived from niche centers
        # Enforce a minimum C component, then normalize rows to sum to 1
        raw = self.niche_optimals
        C_col = tf.maximum(raw[:, 0:1], 0.3)
        raw   = tf.concat([C_col, raw[:, 1:]], axis=1)
        row_sums = tf.reduce_sum(raw, axis=1, keepdims=True)
        self.mean_elementome = tf.Variable(raw / row_sums,
                                           dtype=tf.float32,
                                           name="Mean_Elementome")

        # Global parameters
        self.params = params
        
        # 3x3 neighborhood kernel for dispersal
        self.kernel = tf.ones([3, 3, self.N_CHANNELS, 1], dtype=tf.float32)
        self.grid = None


    def initialize_grid(self):
        """Initialize soil with random values and species pools with zeros."""
        soil_C = tf.random.uniform((self.H, self.W), 0.1, 0.9)
        soil_O = tf.random.uniform((self.H, self.W), 0.1, 0.9)
        soil_N = tf.random.uniform((self.H, self.W), 0.1, 0.9)
        soil_P = tf.random.uniform((self.H, self.W), 0.1, 0.9)
        
        zeros = tf.zeros((self.H, self.W, self.N_CHANNELS - 4), dtype=tf.float32)
        
        init_tensor = tf.concat(
            [tf.stack([soil_C, soil_O, soil_N, soil_P], axis=-1), zeros],
            axis=-1
        )
        
        self.grid = tf.Variable(init_tensor, trainable=False, name="Grid")
        print(f"Initialized N={self.N_SPP} model.")


    @tf.function
    def step(self):
        # 1. Neighborhood convolution
        current_grid = self.grid
        padded = tf.pad(current_grid, [[1, 1], [1, 1], [0, 0]])
        neighbors = tf.nn.depthwise_conv2d(
            padded[tf.newaxis, ...],
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )[0]
        
        # 2. Unpack soil and species pools
        soil      = current_grid[:, :, :4]
        soil_C, soil_O, soil_N, soil_P = tf.unstack(soil, axis=-1)
        
        pool_end  = 4 + (5 * self.N_SPP)
        spp_flat  = current_grid[:, :, 4:pool_end]
        spp_pools = tf.reshape(spp_flat, (self.H, self.W, self.N_SPP, 5))

        neigh_flat  = neighbors[:, :, 4:pool_end]
        neigh_pools = tf.reshape(neigh_flat, (self.H, self.W, self.N_SPP, 5))
        
        spp_C   = spp_pools[:, :, :, 0]
        neigh_C = neigh_pools[:, :, :, 0]
        
        # 3. Species mean element fractions (for biomass & demand)
        f_C_vec = self.mean_elementome[:, 0]  # (N,)
        f_N_vec = self.mean_elementome[:, 2]
        f_P_vec = self.mean_elementome[:, 3]
        
        # Biomass from C pool
        biomass = spp_C / (f_C_vec + 1e-9)
        total_biomass = tf.reduce_sum(biomass, axis=-1)
        
        # 4. Space limitation
        space_factor = tf.maximum(0.0, 1.0 - total_biomass / self.params["K"])
        space_factor = space_factor[..., tf.newaxis]
        
        # 5. Colonization: nutrient-limited seed with individual elementome
        rand    = tf.random.uniform(tf.shape(spp_C))
        is_seed = tf.cast(rand < self.seed_probs, tf.float32) * tf.cast(spp_C < 0.001, tf.float32)
        
        target_seed_biomass = 0.02
        
        # 5a. Sample individual elementomes around niche center for seeds
        mu      = self.mean_elementome                      # (N,5)
        mu_grid = mu[tf.newaxis, tf.newaxis, :, :]          # (1,1,N,5)
        
        sigma = (self.niche_widths * 0.2)[tf.newaxis, tf.newaxis, :, tf.newaxis]
        noise = tf.random.normal((self.H, self.W, self.N_SPP, 5))
        
        elem_raw = tf.nn.relu(mu_grid + sigma * noise)
        row_sums = tf.reduce_sum(elem_raw, axis=-1, keepdims=True) + 1e-9
        elem_individual = elem_raw / row_sums               # (H,W,N,5)
        
        fC_seed = elem_individual[..., 0]
        fN_seed = elem_individual[..., 2]
        fP_seed = elem_individual[..., 3]
        
        # available soil C,N,P for seeding
        avail_C = soil_C * 0.3
        avail_N = soil_N * 0.3
        avail_P = soil_P * 0.3
        
        eps = 1e-9
        Bmax_C = avail_C[:, :, tf.newaxis] / (fC_seed + eps)
        Bmax_N = avail_N[:, :, tf.newaxis] / (fN_seed + eps)
        Bmax_P = avail_P[:, :, tf.newaxis] / (fP_seed + eps)
        Bmax   = tf.minimum(tf.minimum(Bmax_C, Bmax_N), Bmax_P)
        
        B_new = tf.minimum(target_seed_biomass, Bmax) * is_seed
        
        # 5b. Element amounts for new individuals (all 5 elements)
        add_pools = B_new[..., tf.newaxis] * elem_individual   # (H,W,N,5)
        add_C     = add_pools[..., 0]
        add_N     = add_pools[..., 2]
        add_P     = add_pools[..., 3]
        
        spp_pools_seeded = spp_pools + add_pools
        spp_C_seeded     = spp_pools_seeded[..., 0]
        
        # subtract seed uptake from soil
        soil_C = tf.maximum(0.0, soil_C - tf.reduce_sum(add_C, axis=-1))
        soil_N = tf.maximum(0.0, soil_N - tf.reduce_sum(add_N, axis=-1))
        soil_P = tf.maximum(0.0, soil_P - tf.reduce_sum(add_P, axis=-1))
        
        # 6. Growth (C pool)
        growth   = self.growth_rates * spp_C_seeded * space_factor
        # 7. Dispersal
        dispersal = (neigh_C / 9.0) * 0.01 * space_factor
        spp_C_new = spp_C_seeded + growth + dispersal
        
        # 8. Nutrient demands for growth
        target_B  = spp_C_new / (f_C_vec + 1e-9)
        current_N = spp_pools_seeded[..., 2]
        current_P = spp_pools_seeded[..., 3]
        
        dem_N = target_B * f_N_vec - current_N
        dem_P = target_B * f_P_vec - current_P
        
        total_dem_N = tf.reduce_sum(tf.maximum(dem_N, 0.0), axis=-1)
        total_dem_P = tf.reduce_sum(tf.maximum(dem_P, 0.0), axis=-1)
        
        avail_N = soil_N * 0.5
        avail_P = soil_P * 0.5
        
        scale_N = tf.minimum(1.0, avail_N / (total_dem_N + 1e-9))
        scale_P = tf.minimum(1.0, avail_P / (total_dem_P + 1e-9))
        
        uptake_N = tf.maximum(dem_N, 0.0) * scale_N[..., tf.newaxis]
        uptake_P = tf.maximum(dem_P, 0.0) * scale_P[..., tf.newaxis]
        
        new_N = current_N + uptake_N
        new_P = current_P + uptake_P
        
        # 9. Liebig’s law
        pot_from_C = spp_C_new / (f_C_vec + 1e-9)
        pot_from_N = new_N      / (f_N_vec + 1e-9)
        pot_from_P = new_P      / (f_P_vec + 1e-9)
        max_biomass = tf.minimum(tf.minimum(pot_from_C, pot_from_N), pot_from_P)
        
        # 10. Niche mortality
        soil_other = tf.ones_like(soil_C) * 0.5
        env_stack = tf.stack([soil_C, soil_O, soil_N, soil_P, soil_other], axis=-1)
        env_grid  = env_stack[:, :, tf.newaxis, :]   # (H,W,1,5)
        optimum_grid = tf.reshape(self.niche_optimals, (1, 1, self.N_SPP, 5))
        
        diff    = env_grid - optimum_grid
        sq_diff = tf.square(diff)
        w_grid  = tf.reshape(self.niche_weights, (1, 1, 1, 5))
        dist_sq = tf.reduce_sum(sq_diff * w_grid, axis=-1)
        sigmas  = tf.reshape(self.niche_widths, (1, 1, self.N_SPP))
        
        prob_survival  = tf.exp(-dist_sq / (2.0 * tf.square(sigmas)))
        niche_mortality = 1.0 - prob_survival
        
        # 11. Apply limitation and mortality directly to pools
        C_after_seed = spp_pools_seeded[..., 0]
        biomass_after_seed = C_after_seed / (f_C_vec + 1e-9)
        
        L = tf.minimum(1.0, max_biomass / (biomass_after_seed + 1e-9))
        
        total_death_rate = self.mort_rates + niche_mortality + 0.01
        loss_rates = tf.clip_by_value(1.0 - total_death_rate, 0.0, 1.0)
        
        total_scale = L * loss_rates
        new_pools = spp_pools_seeded * total_scale[..., tf.newaxis]
        
        # 12. Soil update: uptake + leaching + inputs
        total_up_N = tf.reduce_sum(uptake_N, axis=-1)
        total_up_P = tf.reduce_sum(uptake_P, axis=-1)
        
        loss_N = self.params["leach_N"] * soil_N
        loss_P = self.params["leach_P"] * soil_P
        
        soil_N_new = tf.clip_by_value(soil_N + self.params["input_N"] - total_up_N - loss_N, 0.0, 1.0)
        soil_P_new = tf.clip_by_value(soil_P + self.params["input_P"] - total_up_P - loss_P, 0.0, 1.0)
        
        # 13. Pack new state
        new_pools_flat = tf.reshape(new_pools, (self.H, self.W, 5 * self.N_SPP))
        new_C = new_pools[..., 0]
        new_biomass = new_C / (f_C_vec + 1e-9)
        
        final_stack = tf.concat(
            [
                tf.stack([soil_C, soil_O, soil_N_new, soil_P_new], axis=-1),
                new_pools_flat,
                new_biomass,
            ],
            axis=-1,
        )
        
        self.grid.assign(final_stack)
        return self.grid


    def get_state(self):
        return self.grid.numpy()