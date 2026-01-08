import tensorflow as tf
import numpy as np

class EcosystemModel:
    def __init__(self, height=50, width=50):
        self.H = height
        self.W = width
        self.N_CHANNELS = 18

        # === DEFAULT PARAMETERS ===
        self.params = {
            'K': 1.2,                 # Carrying Capacity
            'allelopathy': 0.5,       # Strength of A on B
            'facilitation': 2.0,      # Strength of nurse effect
            'seed_prob_A': 0.02,      # Colonization prob A
            'seed_prob_B': 0.01,      # Colonization prob B
            'inputs_N': 0.002,        # External N input
            'inputs_P': 0.001,        # External P input

            # Species A (Pioneer)
            'A_growth': 0.25,
            'A_mort': 0.02,
            'A_ratios': [0.45, 0.42, 0.03, 0.005, 0.095], # C, O, N, P, Other

            # Species B (Conservative)
            'B_growth': 0.08,
            'B_mort': 0.005,
            'B_ratios': [0.45, 0.42, 0.01, 0.001, 0.119]
        }

        self.kernel = tf.ones([3, 3, self.N_CHANNELS, 1], dtype=tf.float32)
        self.grid = None

    def initialize_grid(self, soil_N_range=(0.8, 1.0), soil_P_range=(0.6, 0.9)):
        """Creates a clean slate grid with random soil conditions."""
        soil_C = tf.random.uniform((self.H, self.W), 0.2, 0.4)
        soil_O = tf.random.uniform((self.H, self.W), 0.4, 0.6)
        soil_N = tf.random.uniform((self.H, self.W), *soil_N_range)
        soil_P = tf.random.uniform((self.H, self.W), *soil_P_range)

        # Zero organisms + metrics
        zeros = tf.zeros((self.H, self.W))
        stack_list = [soil_C, soil_O, soil_N, soil_P] + [zeros]*14

        self.grid = tf.stack(stack_list, axis=-1)
        print(f"Grid Initialized: {self.H}x{self.W} with {self.N_CHANNELS} channels.")

    def set_params(self, **kwargs):
        """Update simulation parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"Parameter '{key}' updated to {value}")
            else:
                print(f"Warning: Parameter '{key}' not found.")

    @tf.function
    def _update_rule(self, grid, neighbors, params_vec):
        """
        TensorFlow Graph Function for Update Logic.
        params_vec allows passing dynamic params into the graph.
        """
        # Unpack Parameters (TF graph needs tensors, not dict lookups)
        K = params_vec[0]
        allelopathy = params_vec[1]
        facil = params_vec[2]
        seed_A = params_vec[3]
        seed_B = params_vec[4]

        # Unpack Grid
        soil_C, soil_O, soil_N, soil_P = tf.unstack(grid[:, :, 0:4], axis=-1)
        A_C, A_O, A_N, A_P, A_Other = tf.unstack(grid[:, :, 4:9], axis=-1)
        B_C, B_O, B_N, B_P, B_Other = tf.unstack(grid[:, :, 9:14], axis=-1)

        # Hardcoded Traits for Graph Efficiency (could also be passed in vec)
        f_A = [0.45, 0.42, 0.03, 0.005, 0.095]
        f_B = [0.45, 0.42, 0.01, 0.001, 0.119]

        # 1. Biomass & Neighbors
        bio_A = A_C / f_A[0]
        bio_B = B_C / f_B[0]
        total_bio = bio_A + bio_B
        neigh_bio_A = neighbors[:, :, 16] / 9.0 # Metric channel 14->16 after conv?
        # Actually: Conv output preserves channel order. Channel 16 is Bio_A from prev step.

        # 2. Interactions
        space_factor = tf.maximum(0.0, 1.0 - (total_bio / K))
        allelo_factor = 1.0 / (1.0 + allelopathy * neigh_bio_A)
        stress_factor = 1.0 / (1.0 + facil * total_bio)

        # 3. Colonization (Seed Rain)
        rand = tf.random.uniform(tf.shape(A_C))
        is_seed_A = tf.cast(rand < seed_A, tf.float32) * tf.cast(A_C < 0.001, tf.float32)
        is_seed_B = tf.cast(rand < seed_B, tf.float32) * tf.cast(B_C < 0.001, tf.float32)

        A_C += is_seed_A * 0.05
        B_C += is_seed_B * 0.05

        # 4. Growth
        grow_A = 0.25 * A_C * space_factor
        grow_B = 0.08 * B_C * space_factor * allelo_factor

        A_C_new = A_C + grow_A
        B_C_new = B_C + grow_B

        # 5. Nutrients (Simplified for Speed)
        target_A = A_C_new / f_A[0]
        target_B = B_C_new / f_B[0]

        dem_N_A = target_A * f_A[2] - A_N
        dem_P_A = target_A * f_A[3] - A_P
        dem_N_B = target_B * f_B[2] - B_N
        dem_P_B = target_B * f_B[3] - B_P

        tot_N_dem = tf.maximum(dem_N_A, 0.) + tf.maximum(dem_N_B, 0.)
        tot_P_dem = tf.maximum(dem_P_A, 0.) + tf.maximum(dem_P_B, 0.)

        scale_N = tf.minimum(1.0, (soil_N * 0.5) / (tot_N_dem + 1e-9))
        scale_P = tf.minimum(1.0, (soil_P * 0.5) / (tot_P_dem + 1e-9))

        A_N_new = A_N + tf.maximum(dem_N_A, 0.) * scale_N
        A_P_new = A_P + tf.maximum(dem_P_A, 0.) * scale_P
        B_N_new = B_N + tf.maximum(dem_N_B, 0.) * scale_N
        B_P_new = B_P + tf.maximum(dem_P_B, 0.) * scale_P

        # 6. Stoichiometry & Mortality
        max_A = tf.reduce_min(tf.stack([A_C_new/f_A[0], A_N_new/f_A[2], A_P_new/f_A[3]], -1), -1)
        max_B = tf.reduce_min(tf.stack([B_C_new/f_B[0], B_N_new/f_B[2], B_P_new/f_B[3]], -1), -1)

        loss_A = 1.0 - (0.02 * stress_factor) - 0.01
        loss_B = 1.0 - (0.005 * stress_factor) - 0.01

        # Finalize (Helper)
        A_final = [max_A * f * loss_A for f in f_A]
        B_final = [max_B * f * loss_B for f in f_B]

        # 7. Soil
        up_N = (A_N_new - A_N) + (B_N_new - B_N)
        up_P = (A_P_new - A_P) + (B_P_new - B_P)

        soil_N_new = tf.clip_by_value(soil_N - up_N + 0.002, 0., 1.)
        soil_P_new = tf.clip_by_value(soil_P - up_P + 0.001, 0., 1.)

        # Metrics
        bio_A_new = A_final[0] / f_A[0]
        bio_B_new = B_final[0] / f_B[0]

        return tf.stack([
            soil_C, soil_O, soil_N_new, soil_P_new,
            *A_final, *B_final,
            bio_A_new, bio_B_new, tf.zeros_like(bio_A), tf.zeros_like(bio_A)
        ], axis=-1)

    @tf.function
    def step(self):
        """Executes one simulation step."""
        # Convert params dict to tensor vector
        p_vec = tf.convert_to_tensor([
            self.params['K'], self.params['allelopathy'], self.params['facilitation'],
            self.params['seed_prob_A'], self.params['seed_prob_B']
        ], dtype=tf.float32)

        padded = tf.pad(self.grid, [[1, 1], [1, 1], [0, 0]])
        padded_batch = padded[tf.newaxis, :, :, :]
        neighbors = tf.nn.depthwise_conv2d(padded_batch, self.kernel, strides=[1, 1, 1, 1], padding='VALID')[0]

        self.grid = self._update_rule(self.grid, neighbors, p_vec)
        return self.grid

    def get_state(self):
        """Returns the grid as a numpy array."""
        return self.grid.numpy()
