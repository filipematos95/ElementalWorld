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
        # Initialize: Inorganic random, Organic zero
        init_inorg = tf.random.uniform((self.H, self.W, 4), 0.2, 0.8)
        init_org   = tf.zeros((self.H, self.W, 4))
        self.soil  = tf.Variable(tf.concat([init_inorg, init_org], axis=-1), name="Soil")

        # Diffusion Kernel (Applies to Inorganic only)
        # 3x3 kernel, depth=4 (for N,P,K,O), multiplier=1
        k = np.array([[0.05, 0.1, 0.05],
                      [0.1,  0.4, 0.1],
                      [0.05, 0.1, 0.05]], dtype=np.float32)
        self.diff_kernel = tf.constant(np.repeat(k[:, :, np.newaxis], 4, axis=2)[:, :, :, np.newaxis])

        # --- 2. AGENTS ---
        # Buffer Columns (10 total):
        # [0:y, 1:x, 2:spp, 3:mass]
        # [4:C, 5:N, 6:P, 7:K, 8:O] (Elementome Req)
        # [9:alive]
        self.agents = tf.Variable(tf.zeros((self.MAX_AGENTS, 10), dtype=tf.float32), name="Agent_Buffer")
        self.n_agents = tf.Variable(0, dtype=tf.int32)

        # --- 3. PARAMETERS ---
        self.photosynthesis_rate = 0.5   # Base C-fixation per unit mass
        self.respiration_rate    = 0.05  # Mass loss/step (Maintenance)
        self.turnover_rate       = 0.02  # Litter drop/step
        self.mineralization_rate = 0.05  # Litter -> Soil conversion/step
        self.seed_cost           = 0.3   # Mass needed to reproduce
        self.seed_mass           = 0.05  # Mass of new seedling

    def add_initial_seeds(self, count=50):
        """Seed random agents with random elementomes."""
        y = tf.random.uniform((count,), 0, self.H)
        x = tf.random.uniform((count,), 0, self.W)
        spp = tf.zeros((count,))
        mass = tf.ones((count,)) * 0.1

        # Stoichiometry (Sum ~ 1.0, C dominant)
        # Random N,P,K,O around 0.1-0.2
        raw = tf.random.uniform((count, 5), 0.1, 0.2)
        # Boost Carbon (Idx 0) to be ~0.5 + noise
        raw = tf.concat([raw[:, 0:1] + 0.5, raw[:, 1:]], axis=1)
        # Normalize
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
        print(f"Initialized {count} agents.")

    @tf.function
    def step(self):
        # ==========================================
        # PHASE 1: SOIL PHYSICS (Diffusion)
        # ==========================================
        soil_curr = self.soil
        inorg = soil_curr[:, :, 0:4]
        org   = soil_curr[:, :, 4:8]

        # Diffuse Inorganic N, P, K, O
        inorg_in = inorg[tf.newaxis, ...]
        inorg_diff = tf.nn.depthwise_conv2d(inorg_in, self.diff_kernel, [1,1,1,1], "SAME")[0]

        # Add external input (Atmospheric Deposition)
        inorg_new = inorg_diff + 0.002

        # ==========================================
        # PHASE 2: AGENT BIOLOGY
        # ==========================================
        active_mask = self.agents[:, 9] > 0.5
        active_idx  = tf.where(active_mask)

        # If empty, just run decomposition and return
        if tf.shape(active_idx)[0] == 0:
            flux = org * self.mineralization_rate
            self.soil.assign(tf.concat([inorg_new + flux, org - flux], axis=-1))
            return self.n_agents

        active_data = tf.gather_nd(self.agents, active_idx)

        # Unpack Data
        coords = tf.cast(active_data[:, 0:2], tf.int32)
        mass   = active_data[:, 3]

        # Requirements (Idx 4=C, 5=N, 6=P, 7=K, 8=O)
        req_N = active_data[:, 5]
        req_P = active_data[:, 6]
        req_K = active_data[:, 7]
        req_O = active_data[:, 8]

        # --- A. PHOTOSYNTHESIS & DEMAND ---
        # Potential Growth (Carbon limited by Light/Physiology)
        pot_growth = mass * self.photosynthesis_rate

        # Nutrient Demand needed to support that C-fixation
        dem_N = pot_growth * req_N
        dem_P = pot_growth * req_P
        dem_K = pot_growth * req_K
        dem_O = pot_growth * req_O

        # Scatter Demands to Grid
        grid_dem_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_N)
        grid_dem_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_P)
        grid_dem_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_K)
        grid_dem_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, dem_O)

        # --- B. LIMITATION (Liebig's Law) ---
        # Check supply in Inorganic Pool
        sup_N = inorg_new[:,:,0]
        sup_P = inorg_new[:,:,1]
        sup_K = inorg_new[:,:,2]
        sup_O = inorg_new[:,:,3]

        # Ratios (Supply / Demand)
        rat_N = sup_N / (grid_dem_N + 1e-9)
        rat_P = sup_P / (grid_dem_P + 1e-9)
        rat_K = sup_K / (grid_dem_K + 1e-9)
        rat_O = sup_O / (grid_dem_O + 1e-9)

        # Minimum of N, P, K, O ratios determines the limitation factor L (0.0 to 1.0)
        limit = tf.minimum(1.0, tf.minimum(tf.minimum(rat_N, rat_P), tf.minimum(rat_K, rat_O)))

        # --- C. GROWTH & RESPIRATION ---
        my_limit = tf.gather_nd(limit, coords)

        # Realized Growth
        gross_growth = pot_growth * my_limit

        # Respiration Cost (Maintenance)
        maint_cost   = mass * self.respiration_rate

        # Net Mass Change
        net_change   = gross_growth - maint_cost
        updated_mass = mass + net_change

        # Check Survival (Starvation Threshold)
        still_alive = tf.cast(updated_mass > 0.01, tf.float32)

        # --- D. NECROMASS & TURNOVER ---
        # 1. Mortality Flux (Dead Agents dump 100% mass)
        dead_flux = updated_mass * (1.0 - still_alive)

        # 2. Turnover Flux (Living Agents dump X% leaves)
        turn_flux = updated_mass * self.turnover_rate * still_alive

        total_necro = dead_flux + turn_flux

        # Update Living Mass (Subtract turnover)
        final_mass = (updated_mass - turn_flux) * still_alive

        # Recycle Nutrients -> Organic Pool (Litter)
        # (C is ignored/lost to air, we recycle nutrients)
        rec_N = total_necro * req_N
        rec_P = total_necro * req_P
        rec_K = total_necro * req_K
        rec_O = total_necro * req_O

        grid_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, rec_N)
        grid_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, rec_P)
        grid_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, rec_K)
        grid_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, rec_O)

        # Calculate Uptake (Remove from Inorganic)
        uptake_N = dem_N * my_limit
        uptake_P = dem_P * my_limit
        uptake_K = dem_K * my_limit
        uptake_O = dem_O * my_limit

        grid_up_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, uptake_N)
        grid_up_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, uptake_P)
        grid_up_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, uptake_K)
        grid_up_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, uptake_O)

        # ==========================================
        # PHASE 3: SOIL POOL UPDATES
        # ==========================================
        # 1. Add Necromass to Organic Pool
        fresh_litter = tf.stack([grid_rec_N, grid_rec_P, grid_rec_K, grid_rec_O], axis=-1)
        org_total = org + fresh_litter

        # 2. Mineralization (Organic -> Inorganic)
        flux_to_soil = org_total * self.mineralization_rate
        org_final = org_total - flux_to_soil

        # 3. Update Inorganic (Diffused + Mineralized - Uptake)
        uptake_stack = tf.stack([grid_up_N, grid_up_P, grid_up_K, grid_up_O], axis=-1)
        inorg_final  = tf.maximum(0.0, inorg_new + flux_to_soil - uptake_stack)

        self.soil.assign(tf.concat([inorg_final, org_final], axis=-1))

        # ==========================================
        # PHASE 4: REPRODUCTION (Colonization)
        # ==========================================
        # Identify parents
        is_fertile = (final_mass > self.seed_cost)
        do_seed    = tf.random.uniform(tf.shape(final_mass)) < 0.1
        parents    = is_fertile & do_seed

        # Deduct cost from parents
        final_mass = tf.where(parents, final_mass - self.seed_cost, final_mass)

        # Update Buffer Rows (Living Agents)
        updated_rows = tf.concat([
            active_data[:, 0:3],      # y, x, spp
            final_mass[:, tf.newaxis],# mass
            active_data[:, 4:9],      # stoich
            still_alive[:, tf.newaxis]# alive
        ], axis=1)
        self.agents.scatter_nd_update(active_idx, updated_rows)

        # Create Seeds
        parent_idx = tf.where(parents)[:, 0]
        n_seeds = tf.shape(parent_idx)[0]

        if n_seeds > 0:
            p_data = tf.gather(active_data, parent_idx)

            # Dispersal (Random Offsets)
            dy = tf.random.uniform((n_seeds,), -1, 2, dtype=tf.int32)
            dx = tf.random.uniform((n_seeds,), -1, 2, dtype=tf.int32)
            ny = (tf.cast(p_data[:,0], tf.int32) + dy) % self.H
            nx = (tf.cast(p_data[:,1], tf.int32) + dx) % self.W

            # Inheritance (Identical + Seed Mass)
            child_rows = tf.concat([
                tf.cast(ny, tf.float32)[:, tf.newaxis],
                tf.cast(nx, tf.float32)[:, tf.newaxis],
                p_data[:, 2:3], # spp
                tf.ones((n_seeds, 1)) * self.seed_mass, # mass
                p_data[:, 4:9], # stoich (elementome)
                tf.ones((n_seeds, 1)) # alive
            ], axis=1)

            # Append to Buffer (Simple Append)
            start = self.n_agents.value()
            safe_count = tf.minimum(n_seeds, self.MAX_AGENTS - start)

            if safe_count > 0:
                self.agents.scatter_nd_update(
                    tf.range(start, start+safe_count)[:, tf.newaxis],
                    child_rows[:safe_count]
                )
                self.n_agents.assign_add(safe_count)

        return self.n_agents

    def get_biomass_grid(self):
        """Aggregate agent mass onto a grid for plotting."""
        # 1. Find living agents
        active_mask = self.agents[:, 9] > 0.5
        active_idx = tf.where(active_mask)

        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))

        # 2. Extract coords and mass
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        mass = data[:, 3]

        # 3. Scatter sum onto a blank grid
        grid = tf.zeros((self.H, self.W))
        grid = tf.tensor_scatter_nd_add(grid, coords, mass)
        return grid.numpy()

    def get_element_pools(self):
        """Calculate total C, N, P, K, O in biomass."""
        active_mask = self.agents[:, 9] > 0.5
        idx = tf.where(active_mask)

        if tf.shape(idx)[0] == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        data = tf.gather_nd(self.agents, idx)
        mass = data[:, 3]
        stoich = data[:, 4:9] # Columns for C, N, P, K, O

        # Total Element = Mass * Stoich_Fraction
        element_masses = stoich * mass[:, tf.newaxis]
        totals = tf.reduce_sum(element_masses, axis=0)
        return totals.numpy()