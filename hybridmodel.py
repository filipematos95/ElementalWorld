import tensorflow as tf
import numpy as np
"TEST"
class HybridEcosystem:
    def __init__(self, height, width, max_agents, niche_centers, niche_covariances, growth_rate=0.7,
                 respiration_rate=0.02, turnover_rate=0.02, mineralization_rate=0.05, seed_cost=0.3, seed_mass=0.05,
                 K_biomass=1.5, soil_base_ratio=None, soil_pool_mean=1.0, soil_pool_std=0.01, soil_ratio_noise=0.05,
                 soil_input_rate=0.2, soil_availability_rate=[1.0, 1.0, 1.0, 1.0]):
        """
        Initialize the Ecosystem.

        Args:
            height, width: Grid dimensions.
            max_agents: Max size of agent buffer.
            niche_centers: (N_spp, 5) array of optimal [C, N, P, K, O].
            niche_covariances: (N_spp, 5, 5) array of covariance matrices.
        """
        self.H = height
        self.W = width
        self.MAX_AGENTS = max_agents

        # --- 0. NICHE DEFINITIONS ---
        self.niche_centers = tf.constant(niche_centers, dtype=tf.float32, name="Niche_Centers")
        self.N_spp = self.niche_centers.shape[0]

        # Load Covariances instead of left/right limits
        self.tolerance_cov = tf.constant(niche_covariances, dtype=tf.float32, name="Niche_Covariances")

        # Precompute the inverse covariance matrices for Mahalanobis
        self.tolerance_inv = tf.linalg.inv(self.tolerance_cov)

        self.soil_availability_rate = tf.constant(soil_availability_rate, dtype=tf.float32)
        self.soil_availability_rate = tf.reshape(self.soil_availability_rate, [1, 1, 4])

        # --- 1. SOIL SYSTEM (8 Channels) ---
        # [0:4] Inorganic N, P, K, O (Available)
        # [4:8] Organic   N, P, K, O (Litter/Locked)
        self.soil_input_rate = soil_input_rate
        # Base ratio (same across all pixels, with small variation)
        if soil_base_ratio is None:
            self.soil_base_ratio = np.array([0.4, 0.3, 0.2, 0.1])  # [N, P, K, O]
        else:
            self.soil_base_ratio = soil_base_ratio

        # Create spatial variation: small perturbations around base ratio
        # Shape: (H, W, 4)
        noise = tf.random.normal((self.H, self.W, 4), mean=0.0, stddev=soil_ratio_noise)
        base_tiled = tf.constant(soil_base_ratio, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        base_tiled = tf.tile(base_tiled, [self.H, self.W, 1])

        ratio_raw = base_tiled + noise
        ratio_raw = tf.maximum(0.001, ratio_raw)  # Prevent negatives
        init_inorg_ratio = ratio_raw / tf.reduce_sum(ratio_raw, axis=2, keepdims=True)  # Normalize to sum=1

        # Pool size varies spatially (some areas richer than others)
        pool_size = tf.random.normal((self.H, self.W, 1), mean=soil_pool_mean, stddev=soil_pool_std)
        pool_size = tf.maximum(0.5, pool_size)  # Minimum pool

        # Final inorganic concentrations = ratio × pool
        init_inorg = init_inorg_ratio * pool_size
        init_org = tf.zeros((self.H, self.W, 4))
        self.soil = tf.Variable(tf.concat([init_inorg, init_org], axis=-1), name="Soil")

        # Diffusion Kernel
        k = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)
        self.diff_kernel = tf.constant(np.repeat(k[:, :, np.newaxis], 4, axis=2)[:, :, :, np.newaxis])

        # --- 2. AGENTS ---
        # [0:y, 1:x, 2:spp, 3:mass, 4:C, 5:N, 6:P, 7:K, 8:O, 9:alive]
        self.agents = tf.Variable(tf.zeros((self.MAX_AGENTS, 10), dtype=tf.float32), name="Agent_Buffer")
        self.n_agents = tf.Variable(0, dtype=tf.int32)

        # --- 3. PARAMETERS ---
        self.growth_rate         = growth_rate
        self.respiration_rate    = respiration_rate
        self.turnover_rate       = turnover_rate
        self.mineralization_rate = mineralization_rate
        self.seed_cost           = seed_cost
        self.seed_mass           = seed_mass
        self.K_biomass           = K_biomass

    def add_initial_seeds(self, count=50, species_id=0):

        y = tf.random.uniform((count,), 0, self.H)
        x = tf.random.uniform((count,), 0, self.W)
        spp = tf.ones((count,)) * float(species_id)
        mass = tf.ones((count,)) * 0.1
        center = self.niche_centers[species_id] # Shape (5,)

        # 1. Proportional Noise (10%) to keep relative ratios sane
        noise_scale = 0.1
        multiplicative_noise = tf.random.normal((count, 5), mean=1.0, stddev=noise_scale)
        raw = center[tf.newaxis, :] * multiplicative_noise

        # 2. Normalize to sum=1 (initial stoichiometry)
        stoich = raw / tf.reduce_sum(raw, axis=1, keepdims=True)

        # NOTE: Removed the safety clamp because it relied on left/right borders.
        # The proportional noise above is small enough that seeds shouldn't be born completely dead.

        alive = tf.ones((count,))

        new_data = tf.concat([
            tf.stack([y, x, spp, mass], axis=1),
            stoich,
            alive[:, tf.newaxis]
        ], axis=1)

        curr = self.n_agents.value()
        idx = tf.range(curr, curr+count)[:, tf.newaxis]
        self.agents.scatter_nd_update(idx, new_data)
        self.n_agents.assign_add(count)
        print(f"Added {count} seeds of Spp {species_id}.")

    @tf.function
    def step(self, fitness_metric="mahalanobis"):
        # ------------------------------------------
        # PHASE 1: SOIL PHYSICS
        # ------------------------------------------
        soil_curr = self.soil
        inorg_curr = soil_curr[:, :, 0:4]
        org   = soil_curr[:, :, 4:8]

        inorg_in = inorg_curr[tf.newaxis, ...]
        inorg_padded = tf.pad(inorg_in, [[0,0], [1,1], [1,1], [0,0]], mode='SYMMETRIC')
        inorg_diff = tf.nn.depthwise_conv2d(inorg_padded, self.diff_kernel, [1,1,1,1], "VALID")[0]

        # External Input (rain/deposition) matching the species niche to prevent drift
        input_ratio = tf.constant(self.soil_base_ratio, dtype=tf.float32)
        input_ratio = input_ratio / tf.reduce_sum(input_ratio) # Normalize to sum=1

        # Add small consistent input
        inorg_new = inorg_diff + (input_ratio * self.soil_input_rate)
        inorg_available = inorg_new * self.soil_availability_rate

        # ------------------------------------------
        # PHASE 2: AGENT BIOLOGY
        # ------------------------------------------
        active_mask = self.agents[:, 9] > 0.5
        active_idx  = tf.where(active_mask)

        # Early exit
        if tf.shape(active_idx)[0] == 0:
            flux = org * self.mineralization_rate
            self.soil.assign(tf.concat([inorg_curr + flux, org - flux], axis=-1))
            return self.n_agents

        active_data = tf.gather_nd(self.agents, active_idx)
        spp_ids = tf.cast(active_data[:, 2], tf.int32)
        coords = tf.cast(active_data[:, 0:2], tf.int32)
        mass   = active_data[:, 3]

        # Current Internal Status (elementome ratios)
        curr_C = active_data[:, 4]
        curr_N = active_data[:, 5]
        curr_P = active_data[:, 6]
        curr_K = active_data[:, 7]
        curr_O = active_data[:, 8]

        # --- A. BIOGEOCHEMICAL NICHE FITNESS ---
        curr_elementome = active_data[:, 4:9]  # [n_agents, 5] = [C, N, P, K, O]
        my_centers = tf.gather(self.niche_centers, spp_ids)

        # Replaced the old metric function with the pure Mahalanobis function
        niche_fitness = self._compute_niche_fitness_mahalanobis(curr_elementome, my_centers, spp_ids)

        # --- B. NEW UPTAKE LOGIC ---
        desired_growth = niche_fitness * self.growth_rate * mass

        c_uptake_potential = desired_growth * curr_C

        remaining = desired_growth - c_uptake_potential
        my_niche_pref = tf.gather(self.niche_centers[:, 1:], spp_ids)  # Take [N, P, K, O]
        niche_sum = tf.reduce_sum(my_niche_pref, axis=1, keepdims=True) + 1e-9
        niche_norm = my_niche_pref / niche_sum

        desired_npko = remaining[:, tf.newaxis] * niche_norm
        available_npko = tf.gather_nd(inorg_available, coords)

        K_m = 0.1
        uptake_limit_N = available_npko[:,0] / (available_npko[:,0] + K_m)
        uptake_limit_P = available_npko[:,1] / (available_npko[:,1] + K_m)
        uptake_limit_K = available_npko[:,2] / (available_npko[:,2] + K_m)
        uptake_limit_O = available_npko[:,3] / (available_npko[:,3] + K_m)

        nutrient_limits = tf.stack([uptake_limit_N, uptake_limit_P, uptake_limit_K, uptake_limit_O], axis=1)
        nutrient_limit = tf.reduce_min(nutrient_limits, axis=1)
        scale_factor = nutrient_limit[:, tf.newaxis]

        actual_growth = desired_growth * nutrient_limit
        c_uptake = c_uptake_potential * nutrient_limit
        up_N = desired_npko[:, 0:1] * scale_factor
        up_P = desired_npko[:, 1:2] * scale_factor
        up_K = desired_npko[:, 2:3] * scale_factor
        up_O = desired_npko[:, 3:4] * scale_factor

        pool_C = (mass * curr_C) + c_uptake
        pool_N = (mass * curr_N) + up_N[:, 0]
        pool_P = (mass * curr_P) + up_P[:, 0]
        pool_K = (mass * curr_K) + up_K[:, 0]
        pool_O = (mass * curr_O) + up_O[:, 0]

        # --- C. GROWTH (Quota + Space + Niche) ---
        MIN_QUOTA = 0.01

        Q_C = pool_C / (mass + 1e-9)
        Q_N = pool_N / (mass + 1e-9)
        Q_P = pool_P / (mass + 1e-9)
        Q_K = pool_K / (mass + 1e-9)
        Q_O = pool_O / (mass + 1e-9)

        g_C = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_C + 1e-9)))
        g_N = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_N + 1e-9)))
        g_P = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_P + 1e-9)))
        g_K = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_K + 1e-9)))
        g_O = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_O + 1e-9)))
        limit_g = tf.minimum(tf.minimum(g_C, g_N), tf.minimum(tf.minimum(g_P, g_K), g_O))

        grid_bio = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, mass)
        loc_bio  = tf.gather_nd(grid_bio, coords)
        space_f  = tf.maximum(0.0, 1.0 - (loc_bio / self.K_biomass))

        realized_growth = actual_growth * limit_g * space_f
        maint = mass * self.respiration_rate
        fin_mass = mass + realized_growth - maint

        alive = tf.cast(fin_mass > 0.01, tf.float32)

        for s in range(self.N_spp):
            mask = (spp_ids == s)
            if tf.reduce_any(mask):
                tf.print("Spp", s,
                         "fit:", tf.reduce_mean(niche_fitness[mask]),
                         "grow:", tf.reduce_mean(actual_growth[mask]),
                        "resp:", tf.reduce_mean(maint[mask]))

        # --- D. RECYCLING ---
        fin_mass_pos = tf.maximum(0.0, fin_mass)
        dead = fin_mass_pos * (1.0 - alive)
        turn = fin_mass_pos * self.turnover_rate * alive

        loss_C = (dead + turn) * Q_C
        loss_N = (dead + turn) * Q_N
        loss_P = (dead + turn) * Q_P
        loss_K = (dead + turn) * Q_K
        loss_O = (dead + turn) * Q_O

        fp_C = pool_C - loss_C
        fp_N = pool_N - loss_N
        fp_P = pool_P - loss_P
        fp_K = pool_K - loss_K
        fp_O = pool_O - loss_O

        fin_mass_alive = (fin_mass - turn) * alive
        fr_C = fp_C / (fin_mass_alive + 1e-9)
        fr_N = fp_N / (fin_mass_alive + 1e-9)
        fr_P = fp_P / (fin_mass_alive + 1e-9)
        fr_K = fp_K / (fin_mass_alive + 1e-9)
        fr_O = fp_O / (fin_mass_alive + 1e-9)

        fr_C = tf.clip_by_value(fr_C, 0.01, 0.99)
        fr_N = tf.clip_by_value(fr_N, 0.01, 0.99)
        fr_P = tf.clip_by_value(fr_P, 0.01, 0.99)
        fr_K = tf.clip_by_value(fr_K, 0.01, 0.99)
        fr_O = tf.clip_by_value(fr_O, 0.01, 0.99)

        ratio_sum = fr_C + fr_N + fr_P + fr_K + fr_O + 1e-9
        fr_C = fr_C / ratio_sum
        fr_N = fr_N / ratio_sum
        fr_P = fr_P / ratio_sum
        fr_K = fr_K / ratio_sum
        fr_O = fr_O / ratio_sum

        g_rec_C = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_C)
        g_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_N)
        g_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_P)
        g_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_K)
        g_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_O)

        g_up_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_N[:, 0])
        g_up_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_P[:, 0])
        g_up_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_K[:, 0])
        g_up_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_O[:, 0])

        fresh = tf.stack([g_rec_N, g_rec_P, g_rec_K, g_rec_O], axis=-1)
        org_tot = org + fresh
        flux = org_tot * self.mineralization_rate
        org_fin = org_tot - flux
        up_st = tf.stack([g_up_N, g_up_P, g_up_K, g_up_O], axis=-1)

        inorg_fin = tf.maximum(0.0, inorg_curr + flux - up_st)
        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        # --- E. REPRODUCTION ---
        is_fertile = (fin_mass_alive > self.seed_cost)
        do_seed    = tf.random.uniform(tf.shape(fin_mass_alive)) < 0.1
        parents    = is_fertile & do_seed
        fin_mass_alive   = tf.where(parents, fin_mass_alive - self.seed_cost, fin_mass_alive)

        up_rows = tf.concat([
            active_data[:, 0:3],
            fin_mass_alive[:, tf.newaxis],
            fr_C[:, tf.newaxis],
            fr_N[:, tf.newaxis], fr_P[:, tf.newaxis], fr_K[:, tf.newaxis], fr_O[:, tf.newaxis],
            alive[:, tf.newaxis]
        ], axis=1)
        self.agents.scatter_nd_update(active_idx, up_rows)

        current_agents = self.agents.read_value()
        valid_range_mask = tf.range(self.MAX_AGENTS) < self.n_agents
        is_alive = current_agents[:, 9] > 0.5
        keep_mask = tf.logical_and(valid_range_mask, is_alive)

        living_agents = tf.boolean_mask(current_agents, keep_mask)
        new_count = tf.shape(living_agents)[0]

        padding = tf.zeros((self.MAX_AGENTS - new_count, 10), dtype=tf.float32)
        new_tensor_state = tf.concat([living_agents, padding], axis=0)

        self.agents.assign(new_tensor_state)
        self.n_agents.assign(new_count)

        # Spawn Seeds
        p_idx = tf.where(parents)[:, 0]
        n_s = tf.shape(p_idx)[0]
        if n_s > 0:
            p_dat = tf.gather(up_rows, p_idx)
            dy = tf.random.uniform((n_s,), -1, 2, dtype=tf.int32)
            dx = tf.random.uniform((n_s,), -1, 2, dtype=tf.int32)
            ny = (tf.cast(p_dat[:,0], tf.int32) + dy) % self.H
            nx = (tf.cast(p_dat[:,1], tf.int32) + dx) % self.W

            c_rows = tf.concat([
                tf.cast(ny, tf.float32)[:, tf.newaxis],
                tf.cast(nx, tf.float32)[:, tf.newaxis],
                p_dat[:, 2:3],
                tf.ones((n_s, 1)) * self.seed_mass,
                p_dat[:, 4:9],
                tf.ones((n_s, 1))
            ], axis=1)

            st = self.n_agents.value()
            safe = tf.minimum(n_s, self.MAX_AGENTS - st)
            if safe > 0:
                self.agents.scatter_nd_update(
                    tf.range(st, st+safe)[:, tf.newaxis],
                    c_rows[:safe]
                )
                self.n_agents.assign_add(safe)

        if tf.shape(active_idx)[0] > 0:
            #tf.print("DIAG -> Growth:", tf.reduce_mean(actual_growth),
            #         "Resp:", tf.reduce_mean(maint),
            #         "SpaceF:", tf.reduce_mean(space_f),
            #         "Fitness:", tf.reduce_mean(niche_fitness),
            #         "Soil_N:", tf.reduce_mean(inorg_new[:, :, 0]))

        return self.n_agents

    def get_species_biomass(self, species_id):
        active_mask = (self.agents[:, 9] > 0.5) & (self.agents[:, 2] == float(species_id))
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        mass = data[:, 3]
        grid = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, mass)
        return grid.numpy()

    def get_biomass_grid(self):
        active_mask = self.agents[:, 9] > 0.5
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        mass = data[:, 3]
        grid = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, mass)
        return grid.numpy()

    def get_element_pools(self):
        active_mask = self.agents[:, 9] > 0.5
        idx = tf.where(active_mask)
        if tf.shape(idx)[0] == 0: return [0.0, 0.0, 0.0, 0.0, 0.0]
        data = tf.gather_nd(self.agents, idx)
        mass = data[:, 3]
        stoich = data[:, 4:9]
        element_masses = stoich * mass[:, tf.newaxis]
        totals = tf.reduce_sum(element_masses, axis=0)
        return totals.numpy()

    def _compute_niche_fitness_mahalanobis(self, elementome_vals, my_centers, spp_ids):
        n_agents = tf.shape(elementome_vals)[0]

        delta = elementome_vals - my_centers

        if spp_ids is None or tf.shape(spp_ids)[0] != n_agents:
            inv_cov = self.tolerance_inv[0:1]
        else:
            inv_cov = tf.gather(self.tolerance_inv, spp_ids)

        delta_weighted = tf.einsum('ni,nij->nj', delta, inv_cov)
        mahal_sq = tf.reduce_sum(delta * delta_weighted, axis=1)

        mahal_dist = tf.sqrt(mahal_sq)

        # 3 Sigma boundary
        SIGMA_THRESHOLD = 3.0
        normalized_dist = mahal_dist / SIGMA_THRESHOLD

        niche_fitness = 1.0 - tf.square(normalized_dist)

        return tf.clip_by_value(niche_fitness, 0.0, 1.0)


    def get_species_mean_fitness(self, species_id):
        """Calculates the current average Mahalanobis fitness for a specific species."""
        active_mask = (self.agents[:, 9] > 0.5) & (self.agents[:, 2] == float(species_id))
        active_idx = tf.where(active_mask)

        # If the species is extinct/not present, its fitness is 0
        if tf.shape(active_idx)[0] == 0:
            return 0.0

        data = tf.gather_nd(self.agents, active_idx)
        elementome = data[:, 4:9]
        spp_ids = tf.cast(data[:, 2], tf.int32)
        centers = tf.gather(self.niche_centers, spp_ids)

        # Call the internal Mahalanobis function
        fitness = self._compute_niche_fitness_mahalanobis(elementome, centers, spp_ids)

        return float(tf.reduce_mean(fitness).numpy())
