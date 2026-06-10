import tensorflow as tf
import numpy as np


class HybridEcosystem:
    def __init__(self, height, width, max_agents, niche_centers, niche_covariances,
                 org_elements=None,
                 soil_elements=None,
                 growth_rate=0.7, respiration_rate=0.02, turnover_rate=0.02,
                 mineralization_rate=0.05, seed_cost=0.3, seed_mass=0.05,
                 seed_mass_by_species=None, seed_range_scale=10.0,
                 seed_range_alpha=1.0, seed_range_eps=1e-6,
                 K_biomass=1.5, soil_base_ratio=None, soil_pool_mean=1.0,
                 soil_pool_std=0.01, soil_ratio_noise=0.05, soil_input_rate=0.2,
                 input_drift_scale=0.08, soil_availability_rate=None,
                 sigma_threshold=3.0,
                 catastrophe_interval=200, catastrophe_mortality=0.4,
                 weak_disturbance_interval=0, weak_disturbance_mortality=0.4,
                 strong_disturbance_interval=0, strong_disturbance_mortality=0.4,
                 p_disturbance=0.01, disturbance_radius=8, disturbance_strength=0.7,
                 demo_noise_std=0.003,
                 env_field_persistence=0.85, env_field_smoothing_passes=2,
                 shock_field_persistence=0.85, shock_field_smoothing_passes=2,
                 temperature_mean=0.5,
                 temperature_amplitude=0.3,
                 temperature_period=100.0,
                 temperature_phase=0.0,
                 temperature_spatial_strength=0.0,
                 temperature_growth_strength=0.5,
                 temperature_respiration_strength=0.3,
                 temperature_mineralization_strength=0.4):

        self.H = height
        self.W = width
        self.MAX_AGENTS = max_agents

        self.org_elements = org_elements or ["C", "N", "P", "K", "O", "Ca", "Mg", "S", "Fe", "Mn", "Zn"]
        self.soil_elements = soil_elements or self.org_elements[1:]

        self.N_org = len(self.org_elements)
        self.N_soil = len(self.soil_elements)

        self.org_to_soil_idx = [self.org_elements.index(e) for e in self.soil_elements]

        self.I_Y = 0
        self.I_X = 1
        self.I_SPP = 2
        self.I_MASS = 3
        self.I_E0 = 4
        self.I_E1 = self.I_E0 + self.N_org
        self.I_ALIVE = self.I_E1
        self.I_AGE = self.I_E1 + 1
        self.AGENT_WIDTH = self.I_E1 + 2

        niche_centers = np.asarray(niche_centers, dtype=np.float32)
        niche_centers_norm = niche_centers / np.sum(niche_centers, axis=1, keepdims=True)
        self.niche_centers = tf.constant(niche_centers_norm, dtype=tf.float32, name="Niche_Centers")
        self.N_spp = self.niche_centers.shape[0]

        self.tolerance_cov = tf.constant(niche_covariances, dtype=tf.float32, name="Niche_Covariances")
        self.tolerance_inv = tf.linalg.inv(self.tolerance_cov)

        if soil_availability_rate is None:
            soil_availability_rate = np.ones(self.N_soil, dtype=np.float32)
        self.soil_availability_rate = tf.reshape(
            tf.constant(soil_availability_rate, dtype=tf.float32), [1, 1, self.N_soil]
        )

        self.sigma_threshold = sigma_threshold
        self.soil_input_rate = soil_input_rate

        if soil_base_ratio is None:
            self.soil_base_ratio = np.ones(self.N_soil, dtype=np.float32) / self.N_soil
        else:
            sbr = np.asarray(soil_base_ratio, dtype=np.float32)
            self.soil_base_ratio = sbr / np.sum(sbr)

        if seed_mass_by_species is None:
            self.seed_mass_by_species = tf.ones((self.N_spp,), dtype=tf.float32) * seed_mass
        else:
            self.seed_mass_by_species = tf.constant(seed_mass_by_species, dtype=tf.float32)

        self.seed_range_scale = tf.constant(seed_range_scale, dtype=tf.float32)
        self.seed_range_alpha = tf.constant(seed_range_alpha, dtype=tf.float32)
        self.seed_range_eps = tf.constant(seed_range_eps, dtype=tf.float32)
        self.seed_range_by_species = self.seed_range_scale * tf.pow(
            self.seed_mass_by_species + self.seed_range_eps,
            self.seed_range_alpha
        )

        raw_noise = tf.random.normal((self.H, self.W, self.N_soil), mean=0.0, stddev=soil_ratio_noise)

        kernel_size = 3
        sigma = 1.0
        ax = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1) / 2.0
        gauss_1d = tf.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / tf.reduce_sum(gauss_1d)
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d[:, :, tf.newaxis, tf.newaxis]
        gauss_2d = tf.tile(gauss_2d, [1, 1, self.N_soil, 1])

        raw_noise_4d = raw_noise[tf.newaxis, ...]
        smoothed_noise = tf.nn.depthwise_conv2d(
            raw_noise_4d, gauss_2d, strides=[1, 1, 1, 1], padding='SAME'
        )[0]

        std_before = tf.math.reduce_std(smoothed_noise)
        smoothed_noise = smoothed_noise * (soil_ratio_noise / (std_before + 1e-9))
        alpha = 0.5
        noise = alpha * smoothed_noise + (1.0 - alpha) * raw_noise

        base_tiled = tf.tile(
            tf.constant(self.soil_base_ratio, dtype=tf.float32)[tf.newaxis, tf.newaxis, :],
            [self.H, self.W, 1]
        )
        ratio_raw = tf.maximum(0.001, base_tiled + noise)
        init_inorg_ratio = ratio_raw / tf.reduce_sum(ratio_raw, axis=2, keepdims=True)
        pool_size = tf.maximum(
            0.5, tf.random.normal((self.H, self.W, 1), mean=soil_pool_mean, stddev=soil_pool_std)
        )
        init_inorg = init_inorg_ratio * pool_size

        self.soil = tf.Variable(
            tf.concat([init_inorg, tf.zeros((self.H, self.W, self.N_soil))], axis=-1),
            name="Soil"
        )

        k = np.array([[0.05, 0.1, 0.05],
                      [0.1,  0.4, 0.1],
                      [0.05, 0.1, 0.05]], dtype=np.float32)
        self.diff_kernel = tf.constant(
            np.repeat(k[:, :, np.newaxis], self.N_soil, axis=2)[:, :, :, np.newaxis]
        )

        self.agents = tf.Variable(
            tf.zeros((self.MAX_AGENTS, self.AGENT_WIDTH), dtype=tf.float32),
            name="Agent_Buffer"
        )
        self.n_agents = tf.Variable(0, dtype=tf.int32)

        self.growth_rate = growth_rate
        self.respiration_rate = respiration_rate
        self.turnover_rate = turnover_rate
        self.mineralization_rate = mineralization_rate
        self.seed_cost = seed_cost
        self.seed_mass = seed_mass
        self.K_biomass = K_biomass
        self.input_drift_scale = input_drift_scale
        self.death_fitness_log = []
        self.last_deficit = tf.Variable(tf.zeros((self.N_spp, self.N_soil)), dtype=tf.float32)

        self.catastrophe_interval = catastrophe_interval
        self.catastrophe_mortality = catastrophe_mortality
        self.p_disturbance = p_disturbance
        self.disturbance_radius = disturbance_radius
        self.disturbance_strength = disturbance_strength
        self.demo_noise_std = demo_noise_std
        self.weak_disturbance_interval = weak_disturbance_interval
        self.weak_disturbance_mortality = weak_disturbance_mortality
        self.strong_disturbance_interval = strong_disturbance_interval
        self.strong_disturbance_mortality = strong_disturbance_mortality

        self.env_field_persistence = env_field_persistence
        self.env_field_smoothing_passes = env_field_smoothing_passes
        self.shock_field_persistence = shock_field_persistence
        self.shock_field_smoothing_passes = shock_field_smoothing_passes
        self.env_field = tf.Variable(tf.zeros((self.H, self.W), dtype=tf.float32), trainable=False)
        self.shock_field = tf.Variable(tf.zeros((self.H, self.W), dtype=tf.float32), trainable=False)
        self.step_count = tf.Variable(0, dtype=tf.int32)

        self.temperature_mean = tf.constant(temperature_mean, dtype=tf.float32)
        self.temperature_amplitude = tf.constant(temperature_amplitude, dtype=tf.float32)
        self.temperature_period = tf.constant(temperature_period, dtype=tf.float32)
        self.temperature_phase = tf.constant(temperature_phase, dtype=tf.float32)
        self.temperature_spatial_strength = tf.constant(temperature_spatial_strength, dtype=tf.float32)
        self.temperature_growth_strength = tf.constant(temperature_growth_strength, dtype=tf.float32)
        self.temperature_respiration_strength = tf.constant(temperature_respiration_strength, dtype=tf.float32)
        self.temperature_mineralization_strength = tf.constant(temperature_mineralization_strength, dtype=tf.float32)
        self.temperature = tf.Variable(temperature_mean, dtype=tf.float32, trainable=False)

    def _smooth_field(self, x, n_passes):
        z = x[tf.newaxis, ..., tf.newaxis]
        for _ in range(int(n_passes)):
            z = tf.nn.avg_pool2d(z, ksize=3, strides=1, padding='SAME')
        return z[0, ..., 0]

    def _update_spatial_field(self, field_var, persistence, smoothing_passes):
        noise = tf.random.normal((self.H, self.W), stddev=1.0)
        smooth = self._smooth_field(noise, smoothing_passes)
        updated = persistence * field_var + (1.0 - persistence) * smooth
        field_var.assign(updated)
        f = field_var.read_value()
        fmin = tf.reduce_min(f)
        fmax = tf.reduce_max(f)
        return (f - fmin) / (fmax - fmin + 1e-8)

    def _update_temperature(self):
        t = tf.cast(self.step_count, tf.float32)
        temp = self.temperature_mean + self.temperature_amplitude * tf.sin(
            2.0 * np.pi * (t + self.temperature_phase) / self.temperature_period
        )
        self.temperature.assign(temp)
        return temp

    def add_initial_seeds(self, count=50, species_id=0):
        y = tf.random.uniform((count,), 0, self.H)
        x = tf.random.uniform((count,), 0, self.W)
        spp = tf.ones((count,)) * float(species_id)
        mass = tf.ones((count,)) * 0.1

        center = self.niche_centers[species_id]
        raw = center[tf.newaxis, :] * tf.random.normal((count, self.N_org), mean=1.0, stddev=0.05)
        stoich = raw / tf.reduce_sum(raw, axis=1, keepdims=True)

        new_data = tf.concat([
            tf.stack([y, x, spp, mass], axis=1),
            stoich,
            tf.ones((count, 1)),
            tf.zeros((count, 1)),
        ], axis=1)

        curr = self.n_agents.value()
        self.agents.scatter_nd_update(tf.range(curr, curr + count)[:, tf.newaxis], new_data)
        self.n_agents.assign_add(count)

    @tf.function
    def step(self, fitness_metric="mahalanobis"):
        soil_curr = self.soil
        inorg_curr = soil_curr[:, :, :self.N_soil]
        org = soil_curr[:, :, self.N_soil:(2 * self.N_soil)]

        inorg_padded = tf.pad(
            inorg_curr[tf.newaxis, ...],
            [[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC'
        )
        inorg_diff = tf.nn.depthwise_conv2d(
            inorg_padded, self.diff_kernel, [1, 1, 1, 1], 'VALID'
        )[0]

        input_ratio = tf.constant(self.soil_base_ratio, dtype=tf.float32)
        input_drift = tf.random.normal(
            tf.shape(input_ratio), mean=0.0, stddev=self.input_drift_scale
        )
        input_ratio_drifted = tf.maximum(0.001, input_ratio + input_drift)
        input_ratio_drifted = input_ratio_drifted / tf.reduce_sum(input_ratio_drifted)

        shock_factor = tf.random.uniform([], 0.1, 3.0)
        pulse = tf.cast(tf.random.uniform([]) < 0.005, tf.float32)
        input_ratio_drifted = input_ratio_drifted * (1.0 + pulse * (shock_factor - 1.0))

        inorg_new = inorg_diff + (input_ratio_drifted * self.soil_input_rate)

        env_field = self._update_spatial_field(
            self.env_field, self.env_field_persistence, self.env_field_smoothing_passes
        )
        shock_field = self._update_spatial_field(
            self.shock_field, self.shock_field_persistence, self.shock_field_smoothing_passes
        )

        do_disturb = tf.random.uniform([]) < self.p_disturbance
        cy = tf.random.uniform([], 0, self.H, dtype=tf.int32)
        cx = tf.random.uniform([], 0, self.W, dtype=tf.int32)

        yy2d, xx2d = tf.meshgrid(tf.range(self.H), tf.range(self.W), indexing='ij')
        dist2d = tf.sqrt(tf.cast(tf.square(yy2d - cy) + tf.square(xx2d - cx), tf.float32))
        patch = tf.cast(dist2d < self.disturbance_radius, tf.float32)

        disturbed = inorg_new * (
                1.0 - patch[:, :, tf.newaxis] * self.disturbance_strength * env_field[:, :, tf.newaxis]
        )
        inorg_new = tf.where(do_disturb, disturbed, inorg_new)

        inorg_available = inorg_new * self.soil_availability_rate

        active_mask = self.agents[:, self.I_ALIVE] > 0.5
        active_idx = tf.where(active_mask)
        active_data = tf.gather_nd(self.agents, active_idx)

        n_active = tf.shape(active_data)[0]
        if tf.equal(n_active, 0):
            self.step_count.assign_add(1)
            self.last_deficit.assign(tf.zeros((self.N_spp, self.N_soil), dtype=tf.float32))
            self.soil.assign(tf.concat([tf.maximum(0.0, inorg_new), org], axis=-1))
            return self.n_agents

        spp_ids = tf.cast(active_data[:, self.I_SPP], tf.int32)
        coords = tf.cast(active_data[:, self.I_Y:self.I_Y + 2], tf.int32)
        mass = active_data[:, self.I_MASS]
        age = active_data[:, self.I_AGE]

        curr_elementome = active_data[:, self.I_E0:self.I_E1]
        curr_C = curr_elementome[:, 0]

        my_centers = tf.gather(self.niche_centers, spp_ids)
        niche_fitness = self._compute_niche_fitness_mahalanobis(
            curr_elementome, my_centers, spp_ids
        )

        temp_scalar = self._update_temperature()
        temp_field = temp_scalar + self.temperature_spatial_strength * (env_field - 0.5)
        temp_field = tf.clip_by_value(temp_field, 0.0, 1.0)
        temp_at_agents = tf.gather_nd(temp_field, coords)

        temp_growth_factor = 1.0 + self.temperature_growth_strength * (temp_at_agents - 0.5)
        desired_growth = niche_fitness * self.growth_rate * mass * temp_growth_factor

        c_uptake_potential = desired_growth * curr_C
        remaining = desired_growth - c_uptake_potential

        my_niche_pref = tf.gather(self.niche_centers, spp_ids)
        my_niche_pref_soil = tf.gather(
            my_niche_pref, tf.constant(self.org_to_soil_idx, dtype=tf.int32), axis=1
        )
        niche_norm = my_niche_pref_soil / (
                tf.reduce_sum(my_niche_pref_soil, axis=1, keepdims=True) + 1e-9
        )

        desired_soil = remaining[:, tf.newaxis] * niche_norm
        available_soil = tf.gather_nd(inorg_available, coords)

        K_m = 0.1
        uptake = desired_soil * (available_soil / (available_soil + K_m))
        deficits = tf.maximum(0.0, desired_soil - uptake)

        per_spp_deficit = tf.math.unsorted_segment_sum(
            deficits, spp_ids, num_segments=self.N_spp
        )
        self.last_deficit.assign(per_spp_deficit)

        c_uptake = c_uptake_potential
        actual_growth = c_uptake + tf.reduce_sum(uptake, axis=1)

        soil_to_org = tf.one_hot(
            tf.constant(self.org_to_soil_idx, dtype=tf.int32),
            depth=self.N_org,
            dtype=tf.float32
        )
        uptake_full = tf.matmul(uptake, soil_to_org)

        elem_pools = mass[:, tf.newaxis] * curr_elementome + uptake_full

        MIN_QUOTA = 0.01
        Q = elem_pools / (mass[:, tf.newaxis] + 1e-9)

        g = tf.maximum(0.0, 1.0 - (MIN_QUOTA / (Q + 1e-9)))
        limit_g = tf.reduce_min(g, axis=1)

        flat_coords = coords[:, 0] * self.W + coords[:, 1]

        mass_flat = tf.math.unsorted_segment_sum(
            mass, flat_coords, num_segments=self.H * self.W
        )
        loc_bio = tf.gather(mass_flat, flat_coords)
        space_f = tf.maximum(0.0, 1.0 - (loc_bio / self.K_biomass))

        realized_growth = actual_growth * limit_g * space_f

        temp_resp_factor = 1.0 + self.temperature_respiration_strength * (temp_at_agents - 0.5)
        maint = mass * self.respiration_rate * temp_resp_factor
        fin_mass = mass + realized_growth - maint

        demo_noise = tf.random.normal(tf.shape(fin_mass), mean=0.0, stddev=self.demo_noise_std)
        fin_mass = fin_mass + demo_noise
        alive = tf.cast(fin_mass > 0.01, tf.float32)

        self.step_count.assign_add(1)

        if self.catastrophe_interval > 0:
            is_catastrophe = tf.equal(self.step_count % self.catastrophe_interval, 0)
            shock_at_agents = tf.gather_nd(shock_field, coords)
            survival_roll_cat = tf.cast(
                tf.random.uniform(tf.shape(alive)) >
                (self.catastrophe_mortality * shock_at_agents),
                tf.float32
            )
            alive = tf.where(is_catastrophe, alive * survival_roll_cat, alive)

        if self.weak_disturbance_interval > 0:
            is_weak_disturbance = tf.equal(self.step_count % self.weak_disturbance_interval, 0)
            alpha_weak = 2.0
            env_at_agents = tf.gather_nd(env_field, coords)
            p_die_weak = (
                    self.weak_disturbance_mortality *
                    env_at_agents *
                    tf.pow(1.0 - niche_fitness, alpha_weak)
            )
            survival_roll_weak = tf.cast(
                tf.random.uniform(tf.shape(alive)) > p_die_weak, tf.float32
            )
            alive = tf.where(is_weak_disturbance, alive * survival_roll_weak, alive)

        if self.strong_disturbance_interval > 0:
            is_strong_disturbance = tf.equal(self.step_count % self.strong_disturbance_interval, 0)
            mass_min = tf.reduce_min(fin_mass)
            mass_max = tf.reduce_max(fin_mass)
            strength_score = (fin_mass - mass_min) / (mass_max - mass_min + 1e-9)
            shock_at_agents = tf.gather_nd(shock_field, coords)
            beta_strong = 2.0
            p_die_strong = (
                    self.strong_disturbance_mortality *
                    shock_at_agents *
                    tf.pow(strength_score, beta_strong)
            )
            survival_roll_strong = tf.cast(
                tf.random.uniform(tf.shape(alive)) > p_die_strong, tf.float32
            )
            alive = tf.where(is_strong_disturbance, alive * survival_roll_strong, alive)

        new_age = (age + 1.0) * alive

        fin_mass_pos = tf.maximum(0.0, fin_mass)
        dead = fin_mass_pos * (1.0 - alive)
        turn = fin_mass_pos * self.turnover_rate * alive

        loss_full = (dead + turn)[:, tf.newaxis] * Q
        fp = elem_pools - loss_full

        fin_mass_alive = (fin_mass - turn) * alive

        fr = tf.clip_by_value(fp / (fin_mass_alive[:, tf.newaxis] + 1e-9), 0.01, 0.99)
        fr = fr / (tf.reduce_sum(fr, axis=1, keepdims=True) + 1e-9)

        rec_vals = tf.gather(
            loss_full, tf.constant(self.org_to_soil_idx, dtype=tf.int32), axis=1
        )
        up_vals = uptake
        bio_vals = mass[:, tf.newaxis]

        n_cells = self.H * self.W

        fresh_flat = tf.math.unsorted_segment_sum(rec_vals, flat_coords, n_cells)
        up_flat = tf.math.unsorted_segment_sum(up_vals, flat_coords, n_cells)
        bio_flat = tf.math.unsorted_segment_sum(bio_vals, flat_coords, n_cells)

        fresh = tf.reshape(fresh_flat, (self.H, self.W, self.N_soil))
        up_st = tf.reshape(up_flat, (self.H, self.W, self.N_soil))
        grid_bio = tf.reshape(bio_flat[:, 0], (self.H, self.W))

        org_tot = org + fresh
        temp_mineral_factor = (
                1.0 + self.temperature_mineralization_strength * (temp_field[:, :, tf.newaxis] - 0.5)
        )
        flux = org_tot * self.mineralization_rate * temp_mineral_factor
        org_fin = org_tot - flux

        inorg_fin = tf.maximum(0.0, inorg_new + flux - up_st)
        self.soil.assign(tf.concat([inorg_fin, org_fin], axis=-1))

        is_fertile = fin_mass_alive > self.seed_cost
        seed_prob = 0.1 * niche_fitness
        do_seed = tf.random.uniform(tf.shape(fin_mass_alive)) < seed_prob
        parents = is_fertile & do_seed

        fin_mass_alive = tf.where(parents, fin_mass_alive - self.seed_cost, fin_mass_alive)

        up_rows = tf.concat([
            active_data[:, self.I_Y:self.I_SPP + 1],
            fin_mass_alive[:, tf.newaxis],
            fr,
            alive[:, tf.newaxis],
            new_age[:, tf.newaxis],
        ], axis=1)

        self.agents.scatter_nd_update(active_idx, up_rows)

        alive_mask_local = up_rows[:, self.I_ALIVE] > 0.5
        living_agents = tf.boolean_mask(up_rows, alive_mask_local)
        new_count = tf.shape(living_agents)[0]

        new_tensor_state = tf.concat(
            [
                living_agents,
                tf.zeros((self.MAX_AGENTS - new_count, self.AGENT_WIDTH), dtype=tf.float32)
            ],
            axis=0
        )
        self.agents.assign(new_tensor_state)
        self.n_agents.assign(new_count)

        dying_mask = alive < 0.5
        tf.py_function(
            func=lambda f, s: self.death_fitness_log.extend(
                zip(s.numpy().tolist(), f.numpy().tolist())
            ),
            inp=[
                tf.boolean_mask(niche_fitness, dying_mask),
                tf.boolean_mask(spp_ids, dying_mask)
            ],
            Tout=[]
        )

        p_idx = tf.where(parents)[:, 0]
        n_s = tf.shape(p_idx)[0]

        if n_s > 0:
            p_dat = tf.gather(up_rows, p_idx)

            spp_parent = tf.cast(p_dat[:, self.I_SPP], tf.int32)
            rng = tf.gather(self.seed_range_by_species, spp_parent)

            spread = tf.cast(tf.maximum(1.0, tf.round(rng)), tf.int32)
            spread_f = tf.cast(spread, tf.float32)
            total_f = 2.0 * spread_f + 1.0

            dy = tf.cast(
                tf.math.floor(tf.random.uniform((n_s,)) * total_f) - spread_f,
                tf.int32
            )
            dx = tf.cast(
                tf.math.floor(tf.random.uniform((n_s,)) * total_f) - spread_f,
                tf.int32
            )

            ny = (tf.cast(p_dat[:, self.I_Y], tf.int32) + dy) % self.H
            nx = (tf.cast(p_dat[:, self.I_X], tf.int32) + dx) % self.W

            child_mass = tf.gather(self.seed_mass_by_species, spp_parent)[:, tf.newaxis]

            c_rows = tf.concat([
                tf.cast(ny, tf.float32)[:, tf.newaxis],
                tf.cast(nx, tf.float32)[:, tf.newaxis],
                p_dat[:, self.I_SPP:self.I_SPP + 1],
                child_mass,
                p_dat[:, self.I_E0:self.I_E1],
                tf.ones((n_s, 1)),
                tf.zeros((n_s, 1)),
            ], axis=1)

            target_density = tf.gather_nd(grid_bio, tf.stack([ny, nx], axis=1))
            establishment_prob = tf.maximum(0.0, 1.0 - target_density / self.K_biomass)
            establish = tf.random.uniform((n_s,)) < establishment_prob
            establish_idx = tf.where(establish)[:, 0]

            c_rows = tf.gather(c_rows, establish_idx)
            n_to_place = tf.shape(c_rows)[0]

            st = self.n_agents.value()
            safe = tf.minimum(n_to_place, self.MAX_AGENTS - st)

            if safe > 0:
                self.agents.scatter_nd_update(
                    tf.range(st, st + safe)[:, tf.newaxis],
                    c_rows[:safe]
                )
                self.n_agents.assign_add(safe)

        return self.n_agents

    def get_species_biomass(self, species_id):
        active_mask = (self.agents[:, self.I_ALIVE] > 0.5) & (self.agents[:, self.I_SPP] == float(species_id))
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, self.I_Y:self.I_Y + 2], tf.int32)
        grid = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, data[:, self.I_MASS])
        return grid.numpy()

    def get_biomass_grid(self):
        active_idx = tf.where(self.agents[:, self.I_ALIVE] > 0.5)
        if tf.shape(active_idx)[0] == 0:
            return np.zeros((self.H, self.W))
        data = tf.gather_nd(self.agents, active_idx)
        coords = tf.cast(data[:, self.I_Y:self.I_Y + 2], tf.int32)
        grid = tf.tensor_scatter_nd_add(tf.zeros((self.H, self.W)), coords, data[:, self.I_MASS])
        return grid.numpy()

    def get_element_pools(self):
        idx = tf.where(self.agents[:, self.I_ALIVE] > 0.5)
        if tf.shape(idx)[0] == 0:
            return [0.0] * self.N_org
        data = tf.gather_nd(self.agents, idx)
        return tf.reduce_sum(data[:, self.I_E0:self.I_E1] * data[:, self.I_MASS:self.I_MASS + 1], axis=0).numpy()

    def get_mean_agent_age(self):
        active_idx = tf.where(self.agents[:, self.I_ALIVE] > 0.5)
        if tf.shape(active_idx)[0] == 0:
            return 0.0
        data = tf.gather_nd(self.agents, active_idx)
        return float(tf.reduce_mean(data[:, self.I_AGE]).numpy())

    def _compute_niche_fitness_mahalanobis(self, elementome_vals, my_centers, spp_ids):
        delta = elementome_vals - my_centers
        inv_cov = tf.gather(self.tolerance_inv, spp_ids)
        mahal_sq = tf.reduce_sum(delta * tf.einsum('ni,nij->nj', delta, inv_cov), axis=1)
        mahal_dist = tf.sqrt(mahal_sq)
        niche_fitness = 1.0 - tf.square(mahal_dist / self.sigma_threshold)
        return tf.clip_by_value(niche_fitness, 0.0, 1.0)

    def get_species_mean_fitness(self, species_id):
        active_mask = (self.agents[:, self.I_ALIVE] > 0.5) & (self.agents[:, self.I_SPP] == float(species_id))
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return None
        data = tf.gather_nd(self.agents, active_idx)
        spp_ids = tf.cast(data[:, self.I_SPP], tf.int32)
        elems = data[:, self.I_E0:self.I_E1]
        fitness = self._compute_niche_fitness_mahalanobis(elems, tf.gather(self.niche_centers, spp_ids), spp_ids)
        return float(tf.reduce_mean(fitness).numpy())

    def get_species_mean_dead_fitness(self, species_id):
        deaths = [f for s, f in self.death_fitness_log if s == species_id]
        return float(np.mean(deaths)) if deaths else None

    def get_nutrient_deficit(self):
        return self.last_deficit.numpy()

    def get_species_mean_age(self, species_id):
        active_mask = (self.agents[:, self.I_ALIVE] > 0.5) & (self.agents[:, self.I_SPP] == float(species_id))
        active_idx = tf.where(active_mask)
        if tf.shape(active_idx)[0] == 0:
            return 0.0
        data = tf.gather_nd(self.agents, active_idx)
        return float(tf.reduce_mean(data[:, self.I_AGE]).numpy())

    def get_agent_elemental_dissimilarity_index_tf(self, eps=1e-6):
        active_idx = tf.where(self.agents[:, self.I_ALIVE] > 0.5)
        n_active = tf.shape(active_idx)[0]
        if n_active < 2:
            return 0.0

        data = tf.gather_nd(self.agents, active_idx)
        X = data[:, self.I_E0:self.I_E1]
        Nf = tf.cast(tf.shape(X)[0], tf.float32)
        Ef = tf.cast(tf.shape(X)[1], tf.float32)

        mean = tf.reduce_mean(X, axis=0, keepdims=True)
        Xc = X - mean
        cov = tf.matmul(Xc, Xc, transpose_a=True) / tf.maximum(Nf - 1.0, 1.0)
        cov = cov + tf.eye(tf.shape(cov)[0], dtype=cov.dtype) * eps
        inv_cov = tf.linalg.inv(cov)

        Xi = tf.expand_dims(X, 1)
        Xj = tf.expand_dims(X, 0)
        diff = Xi - Xj
        left = tf.einsum('ije,ef->ijf', diff, inv_cov)
        mahal_sq = tf.einsum('ije,ije->ij', left, diff)
        Dm = tf.sqrt(tf.maximum(mahal_sq, 0.0))

        p = tf.fill([tf.shape(X)[0]], 1.0 / Nf)
        p_i = tf.expand_dims(p, 1)
        p_j = tf.expand_dims(p, 0)
        Pmin = tf.minimum(p_i, p_j)

        EDm = tf.reduce_sum(Dm * Pmin) / Ef
        return float(EDm.numpy())

    def get_species_elemental_dissimilarity_index_tf(self, eps=1e-6):
        active_idx = tf.where(self.agents[:, self.I_ALIVE] > 0.5)
        if tf.shape(active_idx)[0] < 2:
            return 0.0

        data = tf.gather_nd(self.agents, active_idx)
        spp_ids = tf.cast(data[:, self.I_SPP], tf.int32)
        masses = data[:, self.I_MASS]
        elems = data[:, self.I_E0:self.I_E1]

        S = self.N_spp
        tot_mass = tf.math.unsorted_segment_sum(masses, spp_ids, num_segments=S)
        alive_mask = tot_mass > 0
        if not bool(tf.reduce_any(alive_mask)):
            return 0.0

        mass_exp = tf.expand_dims(masses, 1)
        num = tf.math.unsorted_segment_sum(mass_exp * elems, spp_ids, num_segments=S)
        mean_elem = num / tf.maximum(tf.expand_dims(tot_mass, 1), 1e-9)

        mean_elem = tf.boolean_mask(mean_elem, alive_mask)
        p = tf.boolean_mask(tot_mass, alive_mask)
        p = p / tf.reduce_sum(p)

        S_eff = tf.shape(mean_elem)[0]
        E = tf.cast(tf.shape(mean_elem)[1], tf.float32)
        if S_eff < 2:
            return 0.0

        m_mean = tf.reduce_mean(mean_elem, axis=0, keepdims=True)
        Xc = mean_elem - m_mean
        n_eff = tf.cast(S_eff, tf.float32)
        cov = tf.matmul(Xc, Xc, transpose_a=True) / tf.maximum(n_eff - 1.0, 1.0)
        cov = cov + tf.eye(tf.shape(cov)[0], dtype=cov.dtype) * eps
        inv_cov = tf.linalg.inv(cov)

        Xi = tf.expand_dims(mean_elem, 1)
        Xj = tf.expand_dims(mean_elem, 0)
        diff = Xi - Xj
        left = tf.einsum('ije,ef->ijf', diff, inv_cov)
        mahal_sq = tf.einsum('ije,ije->ij', left, diff)
        Dm = tf.sqrt(tf.maximum(mahal_sq, 0.0))

        p_i = tf.expand_dims(p, 1)
        p_j = tf.expand_dims(p, 0)
        Pmin = tf.minimum(p_i, p_j)

        EDm = tf.reduce_sum(Dm * Pmin) / E
        return float(EDm.numpy())