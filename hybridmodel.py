import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from ecosystem_model import *
import ecosystem_plotting as plot

tf.random.set_seed(42)
np.random.seed(42)

# 1. Configuration
N = 1
H, W = 100, 100

# 2. Define biogeochemical niches in element space [C, O, N, P, Other]
# Anchors
spp0 = [0.8, 0.2, 0.8, 0.8, 0.5]   # wet/rich

#spp1 = [0.2, 0.8, 0.2, 0.2, 0.5]   # dry/poor

# Random others
#n_random = N - 2
#random_niches = np.random.uniform(0.4, 0.6, (n_random, 5))
#random_niches[:, 4] = 0.5

niche_centers = np.vstack([spp0]).astype(np.float32)

# Niche widths and weights
niche_widths  = np.full((N,), 0.25, dtype=np.float32)  # shape (1,)
niche_weights = np.array([1.0, 2.0, 1.5, 1.5, 0.0], dtype=np.float32)

# 3. Life-history traits (linspace works fine with N=1)
growth_rates = np.linspace(0.1, 0.1, N).astype(np.float32)
mort_rates   = np.linspace(0.007, 0.005, N).astype(np.float32)
seed_probs   = np.linspace(0.1, 0.08, N).astype(np.float32)

# general params
params = {
    "K":        1.5,
    "input_N":  0.02,
    "input_P":  0.01,
    "leach_N":  0.01,
    "leach_P":  0.005,
}

# 4. Create model
model = EcosystemModel(
    n_species=N,
    height=H,
    width=W,
    growth_rates=growth_rates,
    mort_rates=mort_rates,
    seed_probs=seed_probs,
    niche_centers=niche_centers,
    niche_widths=niche_widths,
    niche_weights=niche_weights,
    params=params
)

model.initialize_grid()

# 5. Run simulation and record mean biomass per species
history = []
totC_hist, totN_hist, totP_hist = [], [], []
n_steps = 100

for t in range(n_steps):
    model.step()
    if t % 10 == 0:
        state = model.get_state()
        biomass = state[:, :, 4 + 5*N : 4 + 5*N + N]
        elem_C = state[:, :, 4 : 4 + N]
        elem_N = state[:, :, 4 + N : 4 + 2*N]
        elem_P = state[:, :, 4 + 2*N : 4 + 3*N]

        total_C = elem_C.sum()
        total_N = elem_N.sum()
        total_P = elem_P.sum()
        means = biomass.mean(axis=(0, 1))
        history.append(means)

        totC_hist.append(total_C)
        totN_hist.append(total_N)
        totP_hist.append(total_P)

history = np.array(history)
totC_hist = np.array(totC_hist)
totN_hist = np.array(totN_hist)
totP_hist = np.array(totP_hist)


# 6. Plot biomass trajectories
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(history[:, i], label=f"Spp {i}")
plt.xlabel("Time (x10 steps)")
plt.ylabel("Mean biomass")
plt.title(f"{N}-species biogeochemical‑niche model")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(totC_hist, label="Total C")
plt.plot(totN_hist, label="Total N")
plt.plot(totP_hist, label="Total P")
plt.xlabel("Time (x10 steps)")
plt.ylabel("Total biota element pool")
plt.legend()
plt.tight_layout()
plt.show()