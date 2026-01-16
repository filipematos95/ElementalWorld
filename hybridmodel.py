import tensorflow as tf
import numpy as np

class HybridEcosystem:
    def __init__(self, height, width, max_agents, niche_centers, niche_left, niche_right):
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
        init_inorg = tf.random.uniform((self.H, self.W, 4), 0.2, 0.8)
        init_org   = tf.zeros((self.H, self.W, 4))
        self.soil  = tf.Variable(tf.concat([init_inorg, init_org], axis=-1), name="Soil")

        # Diffusion Kernel
        k = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)
        self.diff_kernel = tf.constant(np.repeat(k[:, :, np.newaxis], 4, axis=2)[:, :, :, np.newaxis])

        # --- 2. AGENTS ---
        # [0:y, 1:x, 2:spp, 3:mass, 4:C, 5:N, 6:P, 7:K, 8:O, 9:alive]
        self.agents = tf.Variable(tf.zeros((self.MAX_AGENTS, 10), dtype=tf.float32), name="Agent_Buffer")
        self.n_agents = tf.Variable(0, dtype=tf.int32)

        # --- 3. PARAMETERS ---
        self.photosynthesis_rate = 0.5
        self.respiration_rate    = 0.05
        self.turnover_rate       = 0.02
        self.mineralization_rate = 0.05
        self.seed_cost           = 0.3
        self.seed_mass           = 0.05
        self.K_biomass           = 1.5

    def add_initial_seeds(self, count=50, species_id=0):
        y = tf.random.uniform((count,), 0, self.H)
        x = tf.random.uniform((count,), 0, self.W)
        spp = tf.ones((count,)) * float(species_id)
        mass = tf.ones((count,)) * 0.1

        # Look up Species Niche Center to initialize stoichiometry
        center = self.niche_centers[species_id] # Shape (5,)
        raw = center[tf.newaxis, :] + tf.random.normal((count, 5), 0, 0.05)
        raw = tf.maximum(0.01, raw)
        stoich = raw / tf.reduce_sum(raw, axis=1, keepdims=True)

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
        inorg_diff = tf.nn.depthwise_conv2d(inorg_in, self.diff_kernel, [1,1,1,1], "SAME")[0]
        inorg_new = inorg_diff + 0.002 # External Input

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

        # Current Internal Status
        curr_N = active_data[:, 5]
        curr_P = active_data[:, 6]
        curr_K = active_data[:, 7]
        curr_O = active_data[:, 8]

        # --- A. BIOGEOCHEMICAL NICHE FITNESS ---
        # 1. Get Environment
        soil_N = tf.gather_nd(inorg_new[:,:,0], coords)
        soil_P = tf.gather_nd(inorg_new[:,:,1], coords)
        soil_K = tf.gather_nd(inorg_new[:,:,2], coords)
        soil_O = tf.gather_nd(inorg_new[:,:,3], coords)
        env_vals = tf.stack([soil_N, soil_P, soil_K, soil_O], axis=1)

        # 2. Get Species Params
        my_centers = tf.gather(self.niche_centers, spp_ids)
        my_left    = tf.gather(self.niche_left,    spp_ids)
        my_right   = tf.gather(self.niche_right,   spp_ids)

        # 3. Compare (Indices 1-4 correspond to N,P,K,O)
        diff = env_vals - my_centers[:, 1:5]

        # 4. Asymmetric Sigma
        sigma = tf.where(diff < 0, my_left[:, 1:5], my_right[:, 1:5])

        # 5. Fitness (0.0 to 1.0)
        fitness_vec = tf.exp(-tf.square(diff) / (2 * tf.square(sigma)))
        niche_fitness = tf.reduce_prod(fitness_vec, axis=1)

        # --- B. DYNAMIC UPTAKE (Luxury) ---
        MAX_RATIO = 0.20
        cap_N = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_N))
        cap_P = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_P))
        cap_K = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_K))
        cap_O = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_O))

        uptake_rate = 0.5
        dem_N = tf.minimum(cap_N, mass * uptake_rate)
        dem_P = tf.minimum(cap_P, mass * uptake_rate)
        dem_K = tf.minimum(cap_K, mass * uptake_rate)
        dem_O = tf.minimum(cap_O, mass * uptake_rate)

        # Scatter Demand
        grid_dem_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_N)
        grid_dem_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_P)
        grid_dem_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_K)
        grid_dem_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_O)

        # Supply Ratios
        rat_N = inorg_new[:,:,0] / (grid_dem_N + 1e-9)
        rat_P = inorg_new[:,:,1] / (grid_dem_P + 1e-9)
        rat_K = inorg_new[:,:,2] / (grid_dem_K + 1e-9)
        rat_O = inorg_new[:,:,3] / (grid_dem_O + 1e-9)

        lim_N = tf.gather_nd(tf.minimum(1.0, rat_N), coords)
        lim_P = tf.gather_nd(tf.minimum(1.0, rat_P), coords)
        lim_K = tf.gather_nd(tf.minimum(1.0, rat_K), coords)
        lim_O = tf.gather_nd(tf.minimum(1.0, rat_O), coords)

        # Uptake
        up_N = dem_N * lim_N
        up_P = dem_P * lim_P
        up_K = dem_K * lim_K
        up_O = dem_O * lim_O

        # Update Pools
        pool_N = (mass * curr_N) + up_N
        pool_P = (mass * curr_P) + up_P
        pool_K = (mass * curr_K) + up_K
        pool_O = (mass * curr_O) + up_O

        # --- C. GROWTH (Quota + Space + Niche) ---
        MIN_QUOTA = 0.05
        Q_N = pool_N / (mass + 1e-9)
        Q_P = pool_P / (mass + 1e-9)
        Q_K = pool_K / (mass + 1e-9)
        Q_O = pool_O / (mass + 1e-9)

        g_N = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_N + 1e-9)))
        g_P = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_P + 1e-9)))
        g_K = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_K + 1e-9)))
        g_O = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_O + 1e-9)))
        limit_g = tf.minimum(tf.minimum(g_N, g_P), tf.minimum(g_K, g_O))

        # Space Limit
        grid_bio = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, mass)
        loc_bio  = tf.gather_nd(grid_bio, coords)
        space_f  = tf.maximum(0.0, 1.0 - (loc_bio / self.K_biomass))

        # Realized Growth: (Photo * Quota * Space * Niche)
        # Niche Fitness reduces photosynthetic efficiency
        gross = mass * self.photosynthesis_rate * limit_g * space_f * niche_fitness

        maint = mass * self.respiration_rate
        net   = gross - maint
        up_mass = mass + net
        alive = tf.cast(up_mass > 0.01, tf.float32)

        # --- D. RECYCLING ---
        dead = up_mass * (1.0 - alive)
        turn = up_mass * self.turnover_rate * alive
        loss_N = (dead + turn) * Q_N
        loss_P = (dead + turn) * Q_P
        loss_K = (dead + turn) * Q_K
        loss_O = (dead + turn) * Q_O

        fp_N = pool_N - loss_N
        fp_P = pool_P - loss_P
        fp_K = pool_K - loss_K
        fp_O = pool_O - loss_O

        fin_mass = (up_mass - turn) * alive
        fr_N = fp_N / (fin_mass + 1e-9)
        fr_P = fp_P / (fin_mass + 1e-9)
        fr_K = fp_K / (fin_mass + 1e-9)
        fr_O = fp_O / (fin_mass + 1e-9)

        # Scatter Updates
        g_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_N)
        g_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_P)
        g_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_K)
        g_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_O)

        g_up_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_N)
        g_up_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_P)
        g_up_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_K)
        g_up_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_O)

        # Update Soil
        fresh = tf.stack([g_rec_N, g_rec_P, g_rec_K, g_rec_O], axis=-1)
        org_tot = org + fresh
        flux = org_tot * self.mineralization_rate
        org_fin = org_tot - flux
        up_st = tf.stack([g_up_N, g_up_P, g_up_K, g_up_O], axis=-1)
        inorg_fin = tf.maximum(0.0, inorg_new + flux - up_st)
        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        # --- E. REPRODUCTION ---
        is_fertile = (fin_mass > self.seed_cost)
        do_seed    = tf.random.uniform(tf.shape(fin_mass)) < 0.1
        parents    = is_fertile & do_seed
        fin_mass   = tf.where(parents, fin_mass - self.seed_cost, fin_mass)

        # Update Agents
        up_rows = tf.concat([
            active_data[:, 0:3],
            fin_mass[:, tf.newaxis],
            active_data[:, 4:5],
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
