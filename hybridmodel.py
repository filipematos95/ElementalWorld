import tensorflow as tf
import numpy as np

class HybridEcosystem:
    def __init__(self, height, width, max_agents, niche_centers, niche_left, niche_right, growth_rate=0.7, respiration_rate=0.02, turnover_rate=0.02,
                 mineralization_rate=0.05, seed_cost=0.3, seed_mass=0.05, K_biomass=1.5, soil_base_ratio=None, soil_pool_mean=1.0, soil_pool_std=0.01,soil_ratio_noise=0.05):
        """
        Initialize the Ecosystem.

        Args:
            height, width: Grid dimensions.
            max_agents: Max size of agent buffer.
            niche_centers: (N_spp, 5) array of optimal [C, N, P, K, O].
            niche_left: (N_spp, 5) array of left-side tolerances.
            niche_right: (N_spp, 5) array of right-side tolerances.
        """
        self.H = height
        self.W = width
        self.MAX_AGENTS = max_agents

        # --- 0. NICHE DEFINITIONS ---
        self.niche_centers = tf.constant(niche_centers, dtype=tf.float32, name="Niche_Centers")
        self.niche_left    = tf.constant(niche_left,    dtype=tf.float32, name="Niche_Left")
        self.niche_right   = tf.constant(niche_right,   dtype=tf.float32, name="Niche_Right")

        # --- 1. SOIL SYSTEM (8 Channels) ---
        # [0:4] Inorganic N, P, K, O (Available)
        # [4:8] Organic   N, P, K, O (Litter/Locked)

        # Base ratio (same across all pixels, with small variation)
        if soil_base_ratio is None:
            soil_base_ratio = np.array([0.4, 0.3, 0.2, 0.1])  # [N, P, K, O]

        # Create spatial variation: small perturbations around base ratio
        # Shape: (H, W, 4)
        noise = tf.random.normal((self.H, self.W, 4), mean=0.0, stddev=soil_ratio_noise)
        base_tiled = tf.constant(soil_base_ratio, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        base_tiled = tf.tile(base_tiled, [self.H, self.W, 1])

        ratio_raw = base_tiled + noise
        ratio_raw = tf.maximum(0.01, ratio_raw)  # Prevent negatives
        init_inorg_ratio = ratio_raw / tf.reduce_sum(ratio_raw, axis=2, keepdims=True)  # Normalize to sum=1

        # Pool size varies spatially (some areas richer than others)
        pool_size = tf.random.normal((self.H, self.W, 1), mean=soil_pool_mean, stddev=soil_pool_std)
        pool_size = tf.maximum(0.1, pool_size)  # Minimum pool

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

    # --- BETTER INITIALIZATION ---
        # 1. Proportional Noise (10%) to keep relative ratios sane
        # This works better than additive noise for trace elements (like K=0.05)
        noise_scale = 0.1
        multiplicative_noise = tf.random.normal((count, 5), mean=1.0, stddev=noise_scale)
        raw = center[tf.newaxis, :] * multiplicative_noise

        # 2. Normalize to sum=1 (initial stoichiometry)
        stoich_noisy = raw / tf.reduce_sum(raw, axis=1, keepdims=True)

        # 3. SAFETY CLAMP: Force seeds to be within their niche tolerance
        # (This prevents "dead-on-arrival" seeds due to bad RNG luck)
        my_left = self.niche_left[species_id]
        my_right = self.niche_right[species_id]

        # Define safe bounds (90% of tolerance)
        min_safe = center - (my_left * 0.9)
        max_safe = center + (my_right * 0.9)

        # Clip the randomized values to stay inside the survivable niche
        stoich_safe = tf.clip_by_value(stoich_noisy, min_safe, max_safe)

        # 4. Final Re-normalization (clipping might shift sum slightly, so normalize again)
        stoich = stoich_safe / tf.reduce_sum(stoich_safe, axis=1, keepdims=True)


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
    def step(self):
        # ------------------------------------------
        # PHASE 1: SOIL PHYSICS
        # ------------------------------------------
        soil_curr = self.soil
        inorg = soil_curr[:, :, 0:4]
        org   = soil_curr[:, :, 4:8]

        inorg_in = inorg[tf.newaxis, ...]
        inorg_padded = tf.pad(inorg_in, [[0,0], [1,1], [1,1], [0,0]], mode='SYMMETRIC')
        inorg_diff = tf.nn.depthwise_conv2d(inorg_padded, self.diff_kernel, [1,1,1,1], "VALID")[0]

        # External Input (rain/deposition) matching the species niche to prevent drift
        input_ratio = tf.constant([0.2, 0.1, 0.05, 0.05], dtype=tf.float32)
        input_ratio = input_ratio / tf.reduce_sum(input_ratio) # Normalize to sum=1

        # Add small consistent input
        inorg_new = inorg_diff + (input_ratio * 0.008)

        # ------------------------------------------
        # PHASE 2: AGENT BIOLOGY
        # ------------------------------------------
        active_mask = self.agents[:, 9] > 0.5
        active_idx  = tf.where(active_mask)

        # Early exit
        if tf.shape(active_idx)[0] == 0:
            flux = org * self.mineralization_rate
            self.soil.assign(tf.concat([inorg_new + flux, org - flux], axis=-1))
            return self.n_agents

        active_data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(active_data[:, 0:2], tf.int32)
        spp_ids = tf.cast(active_data[:, 2], tf.int32)
        mass   = active_data[:, 3]

        # Current Internal Status (elementome ratios)
        curr_C = active_data[:, 4]
        curr_N = active_data[:, 5]
        curr_P = active_data[:, 6]
        curr_K = active_data[:, 7]
        curr_O = active_data[:, 8]

        # --- A. BIOGEOCHEMICAL NICHE FITNESS ---
        # Current agent elementome
        curr_elementome = active_data[:, 4:9]  # [n_agents, 5] = [C, N, P, K, O]

        # Get species niche parameters
        my_centers = tf.gather(self.niche_centers, spp_ids)
        my_left    = tf.gather(self.niche_left,    spp_ids)
        my_right   = tf.gather(self.niche_right,   spp_ids)


        #if tf.shape(spp_ids)[0] > 0:
        #   tf.print("\n--- FITNESS DEBUG ---")
        #    tf.print("Agent 0 Elementome:", curr_elementome[0])
        #    tf.print("Agent 0 Center:    ", my_centers[0])
        #    tf.print("Agent 0 Tolerances:", my_left[0]) # Assuming symmetric for quick check

        #    diff = curr_elementome[0] - my_centers[0]
        #    tf.print("Difference:        ", diff)

        #    # Check the normalization shift
        #    tf.print("Sum of Center:     ", tf.reduce_sum(my_centers[0]))
        #    tf.print("Sum of Agent:      ", tf.reduce_sum(curr_elementome[0]))


        niche_fitness = self._compute_niche_fitness(curr_elementome, my_centers, my_left, my_right)

       # tf.print("Fitness Stats -> Min:", tf.reduce_min(niche_fitness),"Mean:", tf.reduce_mean(niche_fitness),"Max:", tf.reduce_max(niche_fitness))

        # --- B. NEW UPTAKE LOGIC ---
        # Desired growth = fitness * growth_rate * mass
        desired_growth = niche_fitness * self.growth_rate * mass

        # STEP 1: Carbon from atmosphere (unlimited) — but only if growth actually happens
        c_uptake = desired_growth * curr_C

        # STEP 2: Remaining biomass to fill from soil at soil ratio
        remaining = desired_growth - c_uptake

        soil_ratio_npko = tf.gather_nd(inorg_new, coords)  # [N,P,K,O]
        soil_sum = tf.reduce_sum(soil_ratio_npko, axis=1, keepdims=True) + 1e-9
        soil_ratio_norm = soil_ratio_npko / soil_sum

        desired_npko = remaining[:, tf.newaxis] * soil_ratio_norm
        available_npko = tf.gather_nd(inorg_new, coords)

        can_grow = tf.reduce_all(available_npko >= desired_npko, axis=1)

        # Gate ALL uptake (including C) on can_grow
        actual_growth = tf.where(can_grow, desired_growth, 0.0)
        c_uptake      = tf.where(can_grow, c_uptake, 0.0)

        up_N = tf.where(can_grow[:, tf.newaxis], desired_npko[:, 0:1], 0.0)
        up_P = tf.where(can_grow[:, tf.newaxis], desired_npko[:, 1:2], 0.0)
        up_K = tf.where(can_grow[:, tf.newaxis], desired_npko[:, 2:3], 0.0)
        up_O = tf.where(can_grow[:, tf.newaxis], desired_npko[:, 3:4], 0.0)

        # Actual growth (only if can_grow)
        actual_growth = tf.where(can_grow, desired_growth, 0.0)

        # Update pools (C from atmosphere, N/P/K/O from uptake)
        pool_C = (mass * curr_C) + c_uptake  # C always uptaken
        pool_N = (mass * curr_N) + up_N[:, 0]
        pool_P = (mass * curr_P) + up_P[:, 0]
        pool_K = (mass * curr_K) + up_K[:, 0]
        pool_O = (mass * curr_O) + up_O[:, 0]

        # --- C. GROWTH (Quota + Space + Niche) ---
        # New mass after growth
        # --- C. GROWTH (Quota + Space + Niche) ---
        MIN_QUOTA = 0.05

        # Quotas should be computed relative to CURRENT biomass (no dilution by growth that didn't happen)
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

        # ONLY actual_growth can add biomass
        realized_growth = actual_growth * limit_g * space_f

        # Starvation causes decline because maint is always paid
        maint = mass * self.respiration_rate
        fin_mass = mass + realized_growth - maint

        alive = tf.cast(fin_mass > 0.01, tf.float32)

        # --- D. RECYCLING ---
        # Prevent negative masses from creating negative 'dead' fluxes
        fin_mass_pos = tf.maximum(0.0, fin_mass)

        dead = fin_mass_pos * (1.0 - alive)
        turn = fin_mass_pos * self.turnover_rate * alive

        # --- D. RECYCLING ---
        dead = fin_mass * (1.0 - alive)
        turn = fin_mass * self.turnover_rate * alive
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

        # Clamp ratios to [0.01, 0.99] to avoid numerical issues
        fr_C = tf.clip_by_value(fr_C, 0.01, 0.99)
        fr_N = tf.clip_by_value(fr_N, 0.01, 0.99)
        fr_P = tf.clip_by_value(fr_P, 0.01, 0.99)
        fr_K = tf.clip_by_value(fr_K, 0.01, 0.99)
        fr_O = tf.clip_by_value(fr_O, 0.01, 0.99)

        # Renormalize to sum to 1
        ratio_sum = fr_C + fr_N + fr_P + fr_K + fr_O + 1e-9
        fr_C = fr_C / ratio_sum
        fr_N = fr_N / ratio_sum
        fr_P = fr_P / ratio_sum
        fr_K = fr_K / ratio_sum
        fr_O = fr_O / ratio_sum

        # Scatter Updates for recycled nutrients (to organic pool)
        g_rec_C = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_C)
        g_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_N)
        g_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_P)
        g_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_K)
        g_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_O)

        # Scatter Updates for uptaken nutrients (from inorganic pool)
        g_up_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_N[:, 0])
        g_up_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_P[:, 0])
        g_up_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_K[:, 0])
        g_up_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_O[:, 0])

        # Update Soil
        fresh = tf.stack([g_rec_N, g_rec_P, g_rec_K, g_rec_O], axis=-1)
        org_tot = org + fresh
        flux = org_tot * self.mineralization_rate
        org_fin = org_tot - flux
        up_st = tf.stack([g_up_N, g_up_P, g_up_K, g_up_O], axis=-1)
        inorg_fin = tf.maximum(0.0, inorg_new + flux - up_st)
        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        # --- E. REPRODUCTION ---
        is_fertile = (fin_mass_alive > self.seed_cost)
        do_seed    = tf.random.uniform(tf.shape(fin_mass_alive)) < 0.1
        parents    = is_fertile & do_seed
        fin_mass_alive   = tf.where(parents, fin_mass_alive - self.seed_cost, fin_mass_alive)

        # Update Agents
        up_rows = tf.concat([
            active_data[:, 0:3],
            fin_mass_alive[:, tf.newaxis],
            fr_C[:, tf.newaxis],
            fr_N[:, tf.newaxis], fr_P[:, tf.newaxis], fr_K[:, tf.newaxis], fr_O[:, tf.newaxis],
            alive[:, tf.newaxis]
        ], axis=1)
        self.agents.scatter_nd_update(active_idx, up_rows)

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

        return self.n_agents


    def get_biomass_grid(self):
        active_mask = self.agents[:, 9] > 0.5
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        mass = data[:, 3]
        grid = tf.zeros((self.H, self.W))
        grid = tf.tensor_scatter_nd_add(grid, coords, mass)
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

    def get_space_factor_grid(self):
        """Returns a grid of Space Limitation Factors (0.0 = Full, 1.0 = Empty)."""
        grid_mass = self.get_biomass_grid()
        space_factor = np.maximum(0.0, 1.0 - (grid_mass / self.K_biomass))
        return space_factor

    def _compute_niche_fitness(self, elementome_vals, my_centers, my_left, my_right):
        delta = elementome_vals - my_centers
        tolerance = tf.where(delta < 0, my_left, my_right)

        # DEBUG: Print exact values used for calculation
        #tf.print("\n--- DEEP DEBUG ---")
        #tf.print("Delta[0]:", delta[0])
        #tf.print("Tol[0]:", tolerance[0])

        normalized_deviation = delta / (tolerance + 1e-9)
        #tf.print("NormDev[0]:", normalized_deviation[0])

        sq = tf.square(normalized_deviation)
        #tf.print("Squared[0]:", sq[0])

        ss = tf.reduce_sum(sq, axis=1)
        #tf.print("SumSq[0]:", ss[0])

        dist = tf.sqrt(ss)
        #tf.print("CalcDist[0]:", dist[0])
        # -------------------------------

        niche_fitness = 1.0 - dist
        return tf.clip_by_value(niche_fitness, 0.0, 1.0)