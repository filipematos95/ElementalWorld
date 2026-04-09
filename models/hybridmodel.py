import tensorflow as tf
import numpy as np


class HybridEcosystem:
    def __init__(self, height, width, max_agents, niche_centers, niche_covariances,
                 growth_rate=0.7, respiration_rate=0.02, turnover_rate=0.02,
                 mineralization_rate=0.05, seed_cost=0.3, seed_mass=0.05,
                 K_biomass=1.5, soil_base_ratio=None, soil_pool_mean=1.0,
                 soil_pool_std=0.01, soil_ratio_noise=0.05, soil_input_rate=0.2,
                 input_drift_scale=0.08, soil_availability_rate=[1.0, 1.0, 1.0, 1.0],
                 sigma_threshold=3.0,
                 catastrophe_interval=200, catastrophe_mortality=0.4,
                 p_disturbance=0.01, disturbance_radius=8, disturbance_strength=0.7,
                 demo_noise_std=0.003):

        self.H = height
        self.W = width
        self.MAX_AGENTS = max_agents

        niche_centers_norm = niche_centers / np.sum(niche_centers, axis=1, keepdims=True)
        self.niche_centers = tf.constant(niche_centers_norm, dtype=tf.float32, name="Niche_Centers")
        self.N_spp = self.niche_centers.shape[0]

        self.tolerance_cov = tf.constant(niche_covariances, dtype=tf.float32, name="Niche_Covariances")
        self.tolerance_inv = tf.linalg.inv(self.tolerance_cov)

        self.soil_availability_rate = tf.reshape(
            tf.constant(soil_availability_rate, dtype=tf.float32), [1, 1, 4])
        self.sigma_threshold = sigma_threshold

        self.soil_input_rate = soil_input_rate
        if soil_base_ratio is None:
            self.soil_base_ratio = np.array([0.4, 0.3, 0.2, 0.1])
        else:
            self.soil_base_ratio = np.array(soil_base_ratio, dtype=np.float32)

        noise      = tf.random.normal((self.H, self.W, 4), mean=0.0, stddev=soil_ratio_noise)
        base_tiled = tf.tile(
            tf.constant(self.soil_base_ratio, dtype=tf.float32)[tf.newaxis, tf.newaxis, :],
            [self.H, self.W, 1])
        ratio_raw        = tf.maximum(0.001, base_tiled + noise)
        init_inorg_ratio = ratio_raw / tf.reduce_sum(ratio_raw, axis=2, keepdims=True)
        pool_size        = tf.maximum(0.5, tf.random.normal(
            (self.H, self.W, 1), mean=soil_pool_mean, stddev=soil_pool_std))
        init_inorg = init_inorg_ratio * pool_size
        self.soil  = tf.Variable(
            tf.concat([init_inorg, tf.zeros((self.H, self.W, 4))], axis=-1), name="Soil")

        k = np.array([[0.05, 0.1, 0.05],
                      [0.1,  0.4, 0.1],
                      [0.05, 0.1, 0.05]], dtype=np.float32)
        self.diff_kernel = tf.constant(
            np.repeat(k[:, :, np.newaxis], 4, axis=2)[:, :, :, np.newaxis])

        # ← AGE: 10 → 11 columns  [y, x, spp, mass, C, N, P, K, O, alive, age]
        self.agents   = tf.Variable(tf.zeros((self.MAX_AGENTS, 11), dtype=tf.float32), name="Agent_Buffer")
        self.n_agents = tf.Variable(0, dtype=tf.int32)

        self.growth_rate         = growth_rate
        self.respiration_rate    = respiration_rate
        self.turnover_rate       = turnover_rate
        self.mineralization_rate = mineralization_rate
        self.seed_cost           = seed_cost
        self.seed_mass           = seed_mass
        self.K_biomass           = K_biomass
        self.input_drift_scale   = input_drift_scale
        self.death_fitness_log   = []
        self.last_deficit        = tf.Variable(tf.zeros((self.N_spp, 4)), dtype=tf.float32)

        self.catastrophe_interval  = catastrophe_interval
        self.catastrophe_mortality = catastrophe_mortality
        self.p_disturbance         = p_disturbance
        self.disturbance_radius    = disturbance_radius
        self.disturbance_strength  = disturbance_strength
        self.demo_noise_std        = demo_noise_std
        self.step_count            = tf.Variable(0, dtype=tf.int32)

    # ──────────────────────────────────────────────────────────────────────────
    def add_initial_seeds(self, count=50, species_id=0):
        y    = tf.random.uniform((count,), 0, self.H)
        x    = tf.random.uniform((count,), 0, self.W)
        spp  = tf.ones((count,)) * float(species_id)
        mass = tf.ones((count,)) * 0.1
        center = self.niche_centers[species_id]
        raw    = center[tf.newaxis, :] * tf.random.normal((count, 5), mean=1.0, stddev=0.05)
        stoich = raw / tf.reduce_sum(raw, axis=1, keepdims=True)
        new_data = tf.concat([
            tf.stack([y, x, spp, mass], axis=1),
            stoich,
            tf.ones((count, 1)),   # alive
            tf.zeros((count, 1)),  # ← AGE: age = 0
        ], axis=1)
        curr = self.n_agents.value()
        self.agents.scatter_nd_update(tf.range(curr, curr + count)[:, tf.newaxis], new_data)
        self.n_agents.assign_add(count)
        print(f"Added {count} seeds of Spp {species_id}.")

    # ──────────────────────────────────────────────────────────────────────────
    @tf.function
    def step(self, fitness_metric="mahalanobis"):

        # ── PHASE 1: SOIL PHYSICS ─────────────────────────────────────────────
        soil_curr  = self.soil
        inorg_curr = soil_curr[:, :, 0:4]
        org        = soil_curr[:, :, 4:8]

        inorg_padded = tf.pad(
            inorg_curr[tf.newaxis, ...],
            [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        inorg_diff = tf.nn.depthwise_conv2d(
            inorg_padded, self.diff_kernel, [1, 1, 1, 1], "VALID")[0]

        input_ratio         = tf.constant(self.soil_base_ratio, dtype=tf.float32)
        input_drift         = tf.random.normal(tf.shape(input_ratio), mean=0.0,
                                               stddev=self.input_drift_scale)
        input_ratio_drifted = tf.maximum(0.001, input_ratio + input_drift)
        input_ratio_drifted = input_ratio_drifted / tf.reduce_sum(input_ratio_drifted)

        shock_factor        = tf.random.uniform([], 0.1, 3.0)
        pulse               = tf.cast(tf.random.uniform([]) < 0.005, tf.float32)
        input_ratio_drifted = input_ratio_drifted * (1.0 + pulse * (shock_factor - 1.0))

        inorg_new = inorg_diff + (input_ratio_drifted * self.soil_input_rate)

        do_disturb  = tf.random.uniform([]) < self.p_disturbance
        cy          = tf.random.uniform([], 0, self.H, dtype=tf.int32)
        cx          = tf.random.uniform([], 0, self.W, dtype=tf.int32)
        yy2d, xx2d  = tf.meshgrid(tf.range(self.H), tf.range(self.W), indexing='ij')
        dist2d      = tf.sqrt(tf.cast(
            tf.square(yy2d - cy) + tf.square(xx2d - cx), tf.float32))
        patch       = tf.cast(dist2d < self.disturbance_radius, tf.float32)[:, :, tf.newaxis]
        disturbed   = inorg_new * (1.0 - patch * self.disturbance_strength)
        inorg_new   = tf.where(do_disturb, disturbed, inorg_new)

        inorg_available = inorg_new * self.soil_availability_rate

        # ── PHASE 2: AGENT BIOLOGY ────────────────────────────────────────────
        active_mask = self.agents[:, 9] > 0.5
        active_idx  = tf.where(active_mask)

        if tf.shape(active_idx)[0] == 0:
            flux = org * self.mineralization_rate
            self.soil.assign(tf.concat([inorg_curr + flux, org - flux], axis=-1))
            return self.n_agents

        active_data = tf.gather_nd(self.agents, active_idx)
        spp_ids     = tf.cast(active_data[:, 2], tf.int32)
        coords      = tf.cast(active_data[:, 0:2], tf.int32)
        mass        = active_data[:, 3]
        curr_C      = active_data[:, 4]
        curr_N      = active_data[:, 5]
        curr_P      = active_data[:, 6]
        curr_K      = active_data[:, 7]
        curr_O      = active_data[:, 8]
        age         = active_data[:, 10]  # ← AGE

        # --- A. FITNESS ---
        curr_elementome = active_data[:, 4:9]
        my_centers      = tf.gather(self.niche_centers, spp_ids)
        niche_fitness   = self._compute_niche_fitness_mahalanobis(
            curr_elementome, my_centers, spp_ids)

        # --- B. UPTAKE ---
        desired_growth     = niche_fitness * self.growth_rate * mass
        c_uptake_potential = desired_growth * curr_C
        remaining          = desired_growth - c_uptake_potential

        my_niche_pref  = tf.gather(self.niche_centers[:, 1:], spp_ids)
        niche_norm     = my_niche_pref / (tf.reduce_sum(my_niche_pref, axis=1, keepdims=True) + 1e-9)
        desired_npko   = remaining[:, tf.newaxis] * niche_norm
        available_npko = tf.gather_nd(inorg_available, coords)

        K_m = 0.1
        up_N = desired_npko[:, 0:1] * (available_npko[:, 0:1] / (available_npko[:, 0:1] + K_m))
        up_P = desired_npko[:, 1:2] * (available_npko[:, 1:2] / (available_npko[:, 1:2] + K_m))
        up_K = desired_npko[:, 2:3] * (available_npko[:, 2:3] / (available_npko[:, 2:3] + K_m))
        up_O = desired_npko[:, 3:4] * (available_npko[:, 3:4] / (available_npko[:, 3:4] + K_m))

        deficits = tf.stack([
            tf.maximum(0.0, desired_npko[:, 0] - up_N[:, 0]),
            tf.maximum(0.0, desired_npko[:, 1] - up_P[:, 0]),
            tf.maximum(0.0, desired_npko[:, 2] - up_K[:, 0]),
            tf.maximum(0.0, desired_npko[:, 3] - up_O[:, 0]),
        ], axis=1)

        per_spp_deficit = tf.zeros((self.N_spp, 4))
        for s in range(self.N_spp):
            mask    = tf.cast(spp_ids == s, tf.float32)[:, tf.newaxis]
            spp_def = tf.reduce_sum(deficits * mask, axis=0)
            per_spp_deficit = tf.tensor_scatter_nd_update(per_spp_deficit, [[s]], [spp_def])
        self.last_deficit.assign(per_spp_deficit)

        c_uptake      = c_uptake_potential
        actual_growth = c_uptake + up_N[:, 0] + up_P[:, 0] + up_K[:, 0] + up_O[:, 0]

        pool_C = (mass * curr_C) + c_uptake
        pool_N = (mass * curr_N) + up_N[:, 0]
        pool_P = (mass * curr_P) + up_P[:, 0]
        pool_K = (mass * curr_K) + up_K[:, 0]
        pool_O = (mass * curr_O) + up_O[:, 0]

        # --- C. GROWTH ---
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
        maint    = mass * self.respiration_rate
        fin_mass = mass + realized_growth - maint

        demo_noise = tf.random.normal(tf.shape(fin_mass), mean=0.0, stddev=self.demo_noise_std)
        fin_mass   = fin_mass + demo_noise
        alive      = tf.cast(fin_mass > 0.01, tf.float32)

        self.step_count.assign_add(1)
        is_catastrophe = tf.equal(self.step_count % self.catastrophe_interval, 0)
        survival_roll  = tf.cast(
            tf.random.uniform(tf.shape(alive)) > self.catastrophe_mortality, tf.float32)
        alive = tf.where(is_catastrophe, alive * survival_roll, alive)

        # ← AGE: increment age for survivors, reset to 0 for dead
        new_age = (age + 1.0) * alive

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

        fr_C = tf.clip_by_value(fp_C / (fin_mass_alive + 1e-9), 0.01, 0.99)
        fr_N = tf.clip_by_value(fp_N / (fin_mass_alive + 1e-9), 0.01, 0.99)
        fr_P = tf.clip_by_value(fp_P / (fin_mass_alive + 1e-9), 0.01, 0.99)
        fr_K = tf.clip_by_value(fp_K / (fin_mass_alive + 1e-9), 0.01, 0.99)
        fr_O = tf.clip_by_value(fp_O / (fin_mass_alive + 1e-9), 0.01, 0.99)

        ratio_sum = fr_C + fr_N + fr_P + fr_K + fr_O + 1e-9
        fr_C = fr_C / ratio_sum
        fr_N = fr_N / ratio_sum
        fr_P = fr_P / ratio_sum
        fr_K = fr_K / ratio_sum
        fr_O = fr_O / ratio_sum

        g_rec_N = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_N)
        g_rec_P = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_P)
        g_rec_K = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_K)
        g_rec_O = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, loss_O)

        g_up_N  = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_N[:, 0])
        g_up_P  = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_P[:, 0])
        g_up_K  = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_K[:, 0])
        g_up_O  = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, up_O[:, 0])

        fresh   = tf.stack([g_rec_N, g_rec_P, g_rec_K, g_rec_O], axis=-1)
        org_tot = org + fresh
        flux    = org_tot * self.mineralization_rate
        org_fin = org_tot - flux
        up_st   = tf.stack([g_up_N, g_up_P, g_up_K, g_up_O], axis=-1)

        inorg_fin = tf.maximum(0.0, inorg_curr + flux - up_st)
        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        # --- E. REPRODUCTION ---
        is_fertile = fin_mass_alive > self.seed_cost
        seed_prob  = 0.1 * niche_fitness                                      # ← FIX
        do_seed    = tf.random.uniform(tf.shape(fin_mass_alive)) < seed_prob  # ← FIX
        parents    = is_fertile & tf.cast(do_seed, tf.bool)
        fin_mass_alive = tf.where(parents, fin_mass_alive - self.seed_cost, fin_mass_alive)

        up_rows = tf.concat([
            active_data[:, 0:3],
            fin_mass_alive[:, tf.newaxis],
            fr_C[:, tf.newaxis],
            fr_N[:, tf.newaxis], fr_P[:, tf.newaxis],
            fr_K[:, tf.newaxis], fr_O[:, tf.newaxis],
            alive[:, tf.newaxis],
            new_age[:, tf.newaxis],  # ← AGE
        ], axis=1)
        self.agents.scatter_nd_update(active_idx, up_rows)

        current_agents = self.agents.read_value()
        keep_mask      = tf.logical_and(
            tf.range(self.MAX_AGENTS) < self.n_agents,
            current_agents[:, 9] > 0.5)

        dying_mask = alive < 0.5
        if tf.reduce_any(dying_mask):
            tf.py_function(
                func=lambda f, s: self.death_fitness_log.extend(
                    zip(s.numpy().tolist(), f.numpy().tolist())),
                inp=[tf.boolean_mask(niche_fitness, dying_mask),
                     tf.boolean_mask(spp_ids, dying_mask)],
                Tout=[])

        living_agents    = tf.boolean_mask(current_agents, keep_mask)
        new_count        = tf.shape(living_agents)[0]
        new_tensor_state = tf.concat(
            [living_agents,
             tf.zeros((self.MAX_AGENTS - new_count, 11), dtype=tf.float32)],  # ← AGE: 10→11
            axis=0)
        self.agents.assign(new_tensor_state)
        self.n_agents.assign(new_count)

        # Spawn offspring
        p_idx = tf.where(parents)[:, 0]
        n_s   = tf.shape(p_idx)[0]
        if n_s > 0:
            p_dat  = tf.gather(up_rows, p_idx)
            dy     = tf.random.uniform((n_s,), -1, 2, dtype=tf.int32)
            dx     = tf.random.uniform((n_s,), -1, 2, dtype=tf.int32)
            ny     = (tf.cast(p_dat[:, 0], tf.int32) + dy) % self.H
            nx     = (tf.cast(p_dat[:, 1], tf.int32) + dx) % self.W
            c_rows = tf.concat([
                tf.cast(ny, tf.float32)[:, tf.newaxis],
                tf.cast(nx, tf.float32)[:, tf.newaxis],
                p_dat[:, 2:3],
                tf.ones((n_s, 1)) * self.seed_mass,
                p_dat[:, 4:9],
                tf.ones((n_s, 1)),   # alive
                tf.zeros((n_s, 1)),  # ← AGE: offspring start at age 0
            ], axis=1)
            st   = self.n_agents.value()
            safe = tf.minimum(n_s, self.MAX_AGENTS - st)
            if safe > 0:
                self.agents.scatter_nd_update(
                    tf.range(st, st + safe)[:, tf.newaxis],
                    c_rows[:safe])
                self.n_agents.assign_add(safe)

        return self.n_agents

    # ──────────────────────────────────────────────────────────────────────────
    def get_species_biomass(self, species_id):
        active_mask = (self.agents[:, 9] > 0.5) & (self.agents[:, 2] == float(species_id))
        active_idx  = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data   = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        grid   = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, data[:, 3])
        return grid.numpy()

    def get_biomass_grid(self):
        active_idx = tf.where(self.agents[:, 9] > 0.5)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data   = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, 0:2], tf.int32)
        grid   = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, data[:, 3])
        return grid.numpy()

    def get_element_pools(self):
        idx = tf.where(self.agents[:, 9] > 0.5)
        if tf.shape(idx)[0] == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        data = tf.gather_nd(self.agents, idx)
        return tf.reduce_sum(data[:, 4:9] * data[:, 3:4], axis=0).numpy()

    def get_mean_agent_age(self):  # ← AGE: new getter
        active_idx = tf.where(self.agents[:, 9] > 0.5)
        if tf.shape(active_idx)[0] == 0:
            return 0.0
        data = tf.gather_nd(self.agents, active_idx)
        return float(tf.reduce_mean(data[:, 10]).numpy())

    def _compute_niche_fitness_mahalanobis(self, elementome_vals, my_centers, spp_ids):
        n_agents = tf.shape(elementome_vals)[0]
        delta    = elementome_vals - my_centers
        inv_cov  = (tf.gather(self.tolerance_inv, spp_ids)
                    if spp_ids is not None and tf.shape(spp_ids)[0] == n_agents
                    else self.tolerance_inv[0:1])
        mahal_sq      = tf.reduce_sum(delta * tf.einsum('ni,nij->nj', delta, inv_cov), axis=1)
        mahal_dist    = tf.sqrt(mahal_sq)
        niche_fitness = 1.0 - tf.square(mahal_dist / self.sigma_threshold)
        return tf.clip_by_value(niche_fitness, 0.0, 1.0)

    def get_species_mean_fitness(self, species_id):
        active_mask = (self.agents[:, 9] > 0.5) & (self.agents[:, 2] == float(species_id))
        active_idx  = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return None
        data    = tf.gather_nd(self.agents, active_idx)
        spp_ids = tf.cast(data[:, 2], tf.int32)
        fitness = self._compute_niche_fitness_mahalanobis(
            data[:, 4:9], tf.gather(self.niche_centers, spp_ids), spp_ids)
        return float(tf.reduce_mean(fitness).numpy())

    def get_species_mean_dead_fitness(self, species_id):
        deaths = [f for s, f in self.death_fitness_log if s == species_id]
        return float(np.mean(deaths)) if deaths else None

    def get_nutrient_deficit(self):
        return self.last_deficit.numpy()

    def get_species_mean_age(self, species_id):
        active_mask = (self.agents[:, 9] > 0.5) & (self.agents[:, 2] == float(species_id))
        active_idx  = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return 0.0
        data = tf.gather_nd(self.agents, active_idx)
        return float(tf.reduce_mean(data[:, 10]).numpy())

    def get_agent_elemental_dissimilarity_index_tf(self, eps=1e-6):
        # All living agents
        active_idx = tf.where(self.agents[:, 9] > 0.5)
        n_active   = tf.shape(active_idx)[0]
        if n_active < 2:
            return 0.0

        data = tf.gather_nd(self.agents, active_idx)   # (N, 11)
        X    = data[:, 4:9]                            # (N, 5) elementomes
        Nf   = tf.cast(tf.shape(X)[0], tf.float32)
        Ef   = tf.cast(tf.shape(X)[1], tf.float32)

        # Community-wide covariance Σ
        mean = tf.reduce_mean(X, axis=0, keepdims=True)    # (1, E)
        Xc   = X - mean                                    # (N, E)
        cov  = tf.matmul(Xc, Xc, transpose_a=True) / tf.maximum(Nf - 1.0, 1.0)
        cov  = cov + tf.eye(tf.shape(cov)[0], dtype=cov.dtype) * eps
        inv_cov = tf.linalg.inv(cov)                       # (E, E)

        # Pairwise Mahalanobis distances between agents
        Xi   = tf.expand_dims(X, 1)        # (N, 1, E)
        Xj   = tf.expand_dims(X, 0)        # (1, N, E)
        diff = Xi - Xj                     # (N, N, E)

        left      = tf.einsum('ije,ef->ijf', diff, inv_cov)   # (N, N, E)
        mahal_sq  = tf.einsum('ije,ije->ij', left, diff)      # (N, N)
        Dm        = tf.sqrt(tf.maximum(mahal_sq, 0.0))        # (N, N)

        # Equal weights p_i = 1/N
        p    = tf.fill([tf.shape(X)[0]], 1.0 / Nf)            # (N,)
        p_i  = tf.expand_dims(p, 1)                           # (N, 1)
        p_j  = tf.expand_dims(p, 0)                           # (1, N)
        Pmin = tf.minimum(p_i, p_j)                           # (N, N)

        EDm = tf.reduce_sum(Dm * Pmin) / Ef                   # scalar
        return float(EDm.numpy())



    def get_species_elemental_dissimilarity_index_tf(self, eps=1e-6):
        # aggregate living agents by species
        active_idx = tf.where(self.agents[:, 9] > 0.5)
        if tf.shape(active_idx)[0] < 2:
            return 0.0

        data = tf.gather_nd(self.agents, active_idx)      # (N, 11)
        spp_ids = tf.cast(data[:, 2], tf.int32)
        masses  = data[:, 3]
        elems   = data[:, 4:9]                            # (N, 5)

        S = self.N_spp
        # total biomass per species
        tot_mass = tf.math.unsorted_segment_sum(
            masses, spp_ids, num_segments=S
        )                                                 # (S,)
        alive_mask = tot_mass > 0
        if not bool(tf.reduce_any(alive_mask)):
            return 0.0

        # biomass‑weighted mean elementome per species
        mass_exp = tf.expand_dims(masses, 1)              # (N,1)
        num = tf.math.unsorted_segment_sum(
            mass_exp * elems, spp_ids, num_segments=S
        )                                                 # (S,5)
        mean_elem = num / tf.maximum(tf.expand_dims(tot_mass, 1), 1e-9)  # (S,5)

        # keep only species that are present
        mean_elem = tf.boolean_mask(mean_elem, alive_mask)  # (S_eff,5)
        p = tf.boolean_mask(tot_mass, alive_mask)
        p = p / tf.reduce_sum(p)                            # (S_eff,)

        S_eff = tf.shape(mean_elem)[0]
        E     = tf.cast(tf.shape(mean_elem)[1], tf.float32)
        if S_eff < 2:
            return 0.0

        # community covariance Σ from species means
        m_mean = tf.reduce_mean(mean_elem, axis=0, keepdims=True)  # (1,5)
        Xc     = mean_elem - m_mean                                # (S_eff,5)
        n_eff  = tf.cast(S_eff, tf.float32)
        cov    = tf.matmul(Xc, Xc, transpose_a=True) / tf.maximum(n_eff - 1.0, 1.0)
        cov    = cov + tf.eye(tf.shape(cov)[0], dtype=cov.dtype) * eps
        inv_cov = tf.linalg.inv(cov)                               # (5,5)

        # pairwise Mahalanobis distances between species
        Xi   = tf.expand_dims(mean_elem, 1)     # (S_eff,1,5)
        Xj   = tf.expand_dims(mean_elem, 0)     # (1,S_eff,5)
        diff = Xi - Xj                          # (S_eff,S_eff,5)
        left = tf.einsum('ije,ef->ijf', diff, inv_cov)
        mahal_sq = tf.einsum('ije,ije->ij', left, diff)
        Dm = tf.sqrt(tf.maximum(mahal_sq, 0.0)) # (S_eff,S_eff)

        # ED_M with weights min(p_i, p_j)
        p_i = tf.expand_dims(p, 1)              # (S_eff,1)
        p_j = tf.expand_dims(p, 0)              # (1,S_eff)
        Pmin = tf.minimum(p_i, p_j)             # (S_eff,S_eff)

        EDm = tf.reduce_sum(Dm * Pmin) / E
        return float(EDm.numpy())