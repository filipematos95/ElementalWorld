import tensorflow as tf
import numpy as np

class HybridEcosystem:
    def __init__(self, height=100, width=100, max_agents=100000):
        self.H = height
        self.W = width
        self.MAX_AGENTS = max_agents

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

    def add_initial_seeds(self, count=50, niche_center=None):
        y = tf.random.uniform((count,), 0, self.H)
        x = tf.random.uniform((count,), 0, self.W)
        spp = tf.zeros((count,))
        mass = tf.ones((count,)) * 0.1

        if niche_center is not None:
            raw = tf.convert_to_tensor(niche_center, dtype=tf.float32)[tf.newaxis, :]
            raw = raw + tf.random.normal((count, 5), 0, 0.05)
            raw = tf.maximum(0.01, raw)
        else:
            raw = tf.random.uniform((count, 5), 0.1, 0.2)
            raw = tf.concat([raw[:, 0:1] + 0.5, raw[:, 1:]], axis=1)

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

        # Early exit if empty
        if tf.shape(active_idx)[0] == 0:
            flux = org * self.mineralization_rate
            self.soil.assign(tf.concat([inorg_new + flux, org - flux], axis=-1))
            return self.n_agents

        active_data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(active_data[:, 0:2], tf.int32)
        mass   = active_data[:, 3]

        # Current Internal Status (Concentration per unit mass)
        curr_N = active_data[:, 5]
        curr_P = active_data[:, 6]
        curr_K = active_data[:, 7]
        curr_O = active_data[:, 8]

        # --- A. DYNAMIC UPTAKE (Luxury Consumption) ---
        MAX_RATIO = 0.20
        # Calculate Deficit (Capacity)
        cap_N = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_N))
        cap_P = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_P))
        cap_K = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_K))
        cap_O = tf.maximum(0.0, (mass * MAX_RATIO) - (mass * curr_O))

        # Demand: Try to fill deficit, limited by root uptake rate
        uptake_rate = 0.5
        dem_N = tf.minimum(cap_N, mass * uptake_rate)
        dem_P = tf.minimum(cap_P, mass * uptake_rate)
        dem_K = tf.minimum(cap_K, mass * uptake_rate)
        dem_O = tf.minimum(cap_O, mass * uptake_rate)

        # Scatter Demand to Grid
        grid_dem_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_N)
        grid_dem_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_P)
        grid_dem_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_K)
        grid_dem_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_O)

        # Calculate Supply Ratios
        rat_N = inorg_new[:,:,0] / (grid_dem_N + 1e-9)
        rat_P = inorg_new[:,:,1] / (grid_dem_P + 1e-9)
        rat_K = inorg_new[:,:,2] / (grid_dem_K + 1e-9)
        rat_O = inorg_new[:,:,3] / (grid_dem_O + 1e-9)

        lim_N = tf.gather_nd(tf.minimum(1.0, rat_N), coords)
        lim_P = tf.gather_nd(tf.minimum(1.0, rat_P), coords)
        lim_K = tf.gather_nd(tf.minimum(1.0, rat_K), coords)
        lim_O = tf.gather_nd(tf.minimum(1.0, rat_O), coords)

        # Actual Uptake
        up_N = dem_N * lim_N
        up_P = dem_P * lim_P
        up_K = dem_K * lim_K
        up_O = dem_O * lim_O

        # Update Internal Pools (Old Mass * Old Ratio + Uptake)
        pool_N = (mass * curr_N) + up_N
        pool_P = (mass * curr_P) + up_P
        pool_K = (mass * curr_K) + up_K
        pool_O = (mass * curr_O) + up_O

        # --- B. GROWTH (Quota Model) ---
        MIN_QUOTA = 0.05
        # Current Quota (Pool / Mass)
        Q_N = pool_N / (mass + 1e-9)
        Q_P = pool_P / (mass + 1e-9)
        Q_K = pool_K / (mass + 1e-9)
        Q_O = pool_O / (mass + 1e-9)

        # Droop Factor: Growth slows as Quota approaches Minimum
        g_N = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_N + 1e-9)))
        g_P = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_P + 1e-9)))
        g_K = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_K + 1e-9)))
        g_O = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q_O + 1e-9)))

        limit_g = tf.minimum(tf.minimum(g_N, g_P), tf.minimum(g_K, g_O))

        # Space Limitation
        grid_bio = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, mass)
        loc_bio  = tf.gather_nd(grid_bio, coords)
        space_f  = tf.maximum(0.0, 1.0 - (loc_bio / self.K_biomass))

        # Realized Growth
        gross = mass * self.photosynthesis_rate * limit_g * space_f
        maint = mass * self.respiration_rate
        net   = gross - maint
        up_mass = mass + net

        # Check Survival
        alive = tf.cast(up_mass > 0.01, tf.float32)

        # --- C. TURNOVER & RECYCLING ---
        dead = up_mass * (1.0 - alive)
        turn = up_mass * self.turnover_rate * alive

        # Nutrients lost to Soil (Proportional to current Quota)
        loss_N = (dead + turn) * Q_N
        loss_P = (dead + turn) * Q_P
        loss_K = (dead + turn) * Q_K
        loss_O = (dead + turn) * Q_O

        # Update Pools after loss
        fp_N = pool_N - loss_N
        fp_P = pool_P - loss_P
        fp_K = pool_K - loss_K
        fp_O = pool_O - loss_O

        fin_mass = (up_mass - turn) * alive

        # Calculate New Ratios for next step (The "New Elementome")
        fr_N = fp_N / (fin_mass + 1e-9)
        fr_P = fp_P / (fin_mass + 1e-9)
        fr_K = fp_K / (fin_mass + 1e-9)
        fr_O = fp_O / (fin_mass + 1e-9)

        # Scatter Recycling & Uptake to Grid
        g_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_N)
        g_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_P)
        g_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_K)
        g_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_O)

        g_up_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_N)
        g_up_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_P)
        g_up_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_K)
        g_up_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_O)

        # Update Soil Tensors
        fresh = tf.stack([g_rec_N, g_rec_P, g_rec_K, g_rec_O], axis=-1)
        org_tot = org + fresh
        flux = org_tot * self.mineralization_rate
        org_fin = org_tot - flux

        up_st = tf.stack([g_up_N, g_up_P, g_up_K, g_up_O], axis=-1)
        inorg_fin = tf.maximum(0.0, inorg_new + flux - up_st)

        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        # ------------------------------------------
        # PHASE 4: REPRODUCTION
        # ------------------------------------------
        is_fertile = (fin_mass > self.seed_cost)
        do_seed    = tf.random.uniform(tf.shape(fin_mass)) < 0.1
        parents    = is_fertile & do_seed

        fin_mass = tf.where(parents, fin_mass - self.seed_cost, fin_mass)

        # Update Living Agents
        up_rows = tf.concat([
            active_data[:, 0:3],
            fin_mass[:, tf.newaxis],
            active_data[:, 4:5], # C (unchanged)
            fr_N[:, tf.newaxis], fr_P[:, tf.newaxis], fr_K[:, tf.newaxis], fr_O[:, tf.newaxis],
            alive[:, tf.newaxis]
        ], axis=1)
        self.agents.scatter_nd_update(active_idx, up_rows)

        # Create Seeds
        p_idx = tf.where(parents)[:, 0]
        n_s = tf.shape(p_idx)[0]

        if n_s > 0:
            p_dat = tf.gather(up_rows, p_idx)

            dy = tf.random.uniform((n_s,),
