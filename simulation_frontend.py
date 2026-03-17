# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hybridmodel import HybridEcosystem
import ecosystem_plotting as plot
from datetime import datetime
import pickle, os, io

st.set_page_config(page_title="Ecosystem Simulator", page_icon="🌿", layout="wide")
st.title("🌿 Hybrid Ecosystem Simulator")
st.markdown("Configure parameters in the sidebar, then click **▶ Run Simulation**.")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_SPP          = 4
SPP_COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SPP_LABELS     = [f"Species {i+1}" for i in range(N_SPP)]
NUTRIENT_NAMES = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Oxygen (O)"]
ELEMENTS       = ["C", "N", "P", "K", "O"]
DEFAULT_CENTERS = [
    [0.4, 0.35, 0.05, 0.35, 0.10],
    [0.4, 0.25, 0.10, 0.25, 0.10],
    [0.4, 0.35, 0.20, 0.25, 0.10],
    [0.4, 0.25, 0.05, 0.25, 0.05],
]
SPP_COVARIANCES = np.array([
    [[0.01,0,0,0,0],[0,0.01,0,0,0],[0,0,0.01,0,0],[0,0,0,0.01,0],[0,0,0,0,0.01]],
    [[0.01,0,0,0,0],[0,0.02,0,0,0],[0,0,0.01,0,0],[0,0,0,0.02,0],[0,0,0,0,0.01]],
    [[0.01,0,0,0,0],[0,0.01,0,0,0],[0,0,0.008888,0,0],[0,0,0,0.008888,0],[0,0,0,0,0.01]],
    [[0.01,0,0,0,0],[0,0.00625,0,0,0],[0,0,0.00625,0,0],[0,0,0,0.00625,0],[0,0,0,0,0.00625]],
], dtype=np.float32)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [
    ("results", None), ("soil_snapshot", None), ("ran", False),
    ("run_count", 0),  ("payload", None),        ("pkl_default_name", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
def _apply_config(params: dict):
    """Write parameter dict → widget session_state keys, then rerun."""
    scalar_keys = [
        "H", "W", "MAX_AGENTS", "N_STEPS", "SEED",
        "growth_rate", "respiration_rate", "turnover_rate",
        "mineralization_rate", "seed_cost", "seed_mass",
        "K_biomass", "soil_input_rate",
        "soil_pool_mean", "soil_pool_std", "soil_ratio_noise",
    ]
    for k in scalar_keys:
        if k in params:
            st.session_state[k] = params[k]

    for key, val in zip(["sbr_n", "sbr_p", "sbr_k", "sbr_o"],
                        params.get("soil_base_ratio", [0.35, 0.1, 0.35, 0.1])):
        st.session_state[key] = float(val)

    for key, val in zip(["sar_n", "sar_p", "sar_k", "sar_o"],
                        params.get("soil_availability_rate", [0.4, 0.1, 0.1, 0.3])):
        st.session_state[key] = float(val)

    for i, n in enumerate(params.get("initial_seeds", [10, 10, 10, 10])):
        st.session_state[f"seeds_{i}"] = int(n)

    for s, row in enumerate(params.get("spp_centers", DEFAULT_CENTERS)):
        for e, val in enumerate(row):
            st.session_state[f"nc_{s}_{e}"] = float(val)


def _apply_everything(data: dict):
    """Load config + full results into session state."""
    _apply_config(data["parameters"])

    st.session_state["results"] = {
        "history_biomass":      data["history_biomass"],
        "history_agents":       data["history_spp_count"],
        "history_elements":     np.array(data["history_elements"]),
        "history_biomass_grid": data["history_biomass_grid"],
        "history_spp_biomass":  data["history_spp_biomass"],
        "history_spp_fitness":  data.get("history_spp_fitness", [[] for _ in range(N_SPP)]),
        "history_spp_grid":     data["history_spp_biomass_grid"],
        "N_STEPS":              data["parameters"]["N_STEPS"],
    }
    st.session_state["soil_snapshot"] = data.get("soil_snapshot", None)
    st.session_state["payload"]       = data
    st.session_state["ran"]           = True
    st.session_state["run_count"]    += 1
    st.session_state["pkl_default_name"] = (
        f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # ── Load section (must be BEFORE widgets so values are set first) ──
    st.header("📂 Load Simulation")
    uploaded = st.file_uploader("Upload a .pkl file", type=["pkl"], label_visibility="collapsed")

    if uploaded is not None:
        loaded_data = pickle.load(uploaded)
        has_results = "history_biomass" in loaded_data

        col_cfg, col_all = st.columns(2)

        if col_cfg.button("⚙️ Config only", use_container_width=True,
                          help="Populate sidebar parameters; does not restore plots."):
            _apply_config(loaded_data["parameters"])
            st.rerun()

        all_btn = col_all.button(
            "📊 Everything", use_container_width=True,
            disabled=not has_results,
            help="Restore full results + config." if has_results
            else "This file has no results data.",
        )
        if all_btn:
            _apply_everything(loaded_data)
            st.rerun()

    st.divider()

    # ── Simulation config ──────────────────────────────────────────────
    st.header("⚙️ Simulation Config")

    st.subheader("Grid & Steps")
    H          = st.number_input("Grid Height",   10,    500,    100,    key="H")
    W          = st.number_input("Grid Width",    10,    500,    100,    key="W")
    MAX_AGENTS = st.number_input("Max Agents",    10000, 500000, 150000, key="MAX_AGENTS", step=10000)
    N_STEPS    = st.slider("Number of Steps",     100,   5000,   1500,   key="N_STEPS",    step=100)
    SEED       = st.number_input("Random Seed",   0,     9999,   35,     key="SEED")

    st.subheader("Biological Rates")
    growth_rate         = st.slider("Growth Rate",         0.01,  1.0,  0.45,  key="growth_rate",         step=0.01)
    respiration_rate    = st.slider("Respiration Rate",    0.001, 0.1,  0.015, key="respiration_rate",    step=0.001, format="%.3f")
    turnover_rate       = st.slider("Turnover Rate",       0.001, 0.1,  0.03,  key="turnover_rate",       step=0.001, format="%.3f")
    mineralization_rate = st.slider("Mineralization Rate", 0.01,  0.2,  0.05,  key="mineralization_rate", step=0.005)
    seed_cost           = st.slider("Seed Cost",           0.001, 0.1,  0.02,  key="seed_cost",           step=0.001, format="%.3f")
    seed_mass           = st.slider("Seed Mass",           0.001, 0.1,  0.02,  key="seed_mass",           step=0.001, format="%.3f")
    K_biomass           = st.slider("K Biomass",           0.5,   10.0, 2.5,   key="K_biomass",           step=0.1)
    soil_input_rate     = st.slider("Soil Input Rate",     0.1,   2.0,  0.5,   key="soil_input_rate",     step=0.05)

    st.subheader("Soil Parameters")
    soil_pool_mean   = st.slider("Soil Pool Mean",   0.5,  3.0, 1.5,  key="soil_pool_mean",   step=0.1)
    soil_pool_std    = st.slider("Soil Pool Std",    0.01, 0.5, 0.1,  key="soil_pool_std",    step=0.01)
    soil_ratio_noise = st.slider("Soil Ratio Noise", 0.0,  0.2, 0.05, key="soil_ratio_noise", step=0.005)

    st.subheader("Soil Base Ratios [N, P, K, O]")
    c1, c2 = st.columns(2)
    sbr = [
        c1.number_input("N",  0.0, 1.0, 0.35, step=0.05, key="sbr_n"),
        c1.number_input("P",  0.0, 1.0, 0.10, step=0.05, key="sbr_p"),
        c2.number_input("K",  0.0, 1.0, 0.35, step=0.05, key="sbr_k"),
        c2.number_input("O",  0.0, 1.0, 0.10, step=0.05, key="sbr_o"),
    ]

    st.subheader("Soil Availability Rates [N, P, K, O]")
    c1, c2 = st.columns(2)
    sar = [
        c1.number_input("N ", 0.0, 1.0, 0.4, step=0.05, key="sar_n"),
        c1.number_input("P ", 0.0, 1.0, 0.1, step=0.05, key="sar_p"),
        c2.number_input("K ", 0.0, 1.0, 0.1, step=0.05, key="sar_k"),
        c2.number_input("O ", 0.0, 1.0, 0.3, step=0.05, key="sar_o"),
    ]

    st.subheader("Initial Seeds per Species")
    initial_seeds = [
        st.number_input(f"Species {i+1}", 1, 200, 10, key=f"seeds_{i}")
        for i in range(N_SPP)
    ]

    st.divider()
    st.subheader("💾 Save Options")
    save_dir = st.text_input("Save directory", value="results", key="save_dir")

# ─────────────────────────────────────────────
# EXPANDER — Advanced: Niche Centers
# ─────────────────────────────────────────────
with st.expander("🧬 Species Niche Centers (Advanced)", expanded=False):
    st.caption("Each row = one species. Columns = stoichiometric ideal [C, N, P, K, O].")
    spp_centers = []
    for s in range(N_SPP):
        cols = st.columns(5)
        row = [
            cols[e].number_input(
                ELEMENTS[e], 0.0, 1.0, DEFAULT_CENTERS[s][e],
                step=0.01, format="%.3f", key=f"nc_{s}_{e}"
            )
            for e in range(5)
        ]
        spp_centers.append(row)

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────
run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────
if run_btn:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = HybridEcosystem(
        height=H, width=W, max_agents=MAX_AGENTS,
        niche_centers=np.array(spp_centers, dtype=np.float32),
        niche_covariances=SPP_COVARIANCES,
        growth_rate=growth_rate,
        respiration_rate=respiration_rate,
        turnover_rate=turnover_rate,
        mineralization_rate=mineralization_rate,
        seed_cost=seed_cost,
        seed_mass=seed_mass,
        K_biomass=K_biomass,
        soil_base_ratio=np.array(sbr, dtype=np.float32),
        soil_pool_mean=soil_pool_mean,
        soil_pool_std=soil_pool_std,
        soil_ratio_noise=soil_ratio_noise,
        soil_input_rate=soil_input_rate,
        soil_availability_rate=sar,
    )

    st.session_state["soil_snapshot"] = model.soil.numpy().copy()

    for s_id, n in enumerate(initial_seeds):
        model.add_initial_seeds(count=n, species_id=s_id)

    history_biomass      = []
    history_agents       = []
    history_elements     = []
    history_biomass_grid = []
    history_spp_biomass  = [[] for _ in range(N_SPP)]
    history_spp_fitness  = [[] for _ in range(N_SPP)]
    history_spp_grid     = [[] for _ in range(N_SPP)]

    status = st.empty()
    prog   = st.progress(0)

    for t in range(N_STEPS):
        n_agents     = model.step("mahalanobis")
        grid_total   = model.get_biomass_grid()
        mean_biomass = float(np.mean(grid_total))

        history_biomass.append(mean_biomass)
        history_agents.append(int(n_agents.numpy()))
        history_elements.append(model.get_element_pools())

        for s_id in range(N_SPP):
            history_spp_biomass[s_id].append(float(np.mean(model.get_species_biomass(s_id))))
            history_spp_fitness[s_id].append(float(model.get_species_mean_fitness(s_id)))

        if t % 50 == 0:
            history_biomass_grid.append(grid_total)
            for s_id in range(N_SPP):
                history_spp_grid[s_id].append(model.get_species_biomass(s_id))

        prog.progress((t + 1) / N_STEPS)
        if t % 10 == 0:
            status.text(f"Step {t}/{N_STEPS} — Agents: {n_agents.numpy()} | Mean Biomass: {mean_biomass:.4f}")

    status.success(f"✅ Done — {N_STEPS} steps completed.")

    st.session_state["results"] = dict(
        history_biomass=history_biomass,
        history_agents=history_agents,
        history_elements=np.array(history_elements),
        history_biomass_grid=history_biomass_grid,
        history_spp_biomass=history_spp_biomass,
        history_spp_fitness=history_spp_fitness,
        history_spp_grid=history_spp_grid,
        N_STEPS=N_STEPS,
    )

    st.session_state["payload"] = {
        "parameters": {
            "H": H, "W": W, "MAX_AGENTS": MAX_AGENTS, "N_STEPS": N_STEPS, "SEED": SEED,
            "spp_centers": spp_centers,
            "spp_covariances": SPP_COVARIANCES.tolist(),
            "soil_base_ratio": sbr,
            "growth_rate": growth_rate,
            "respiration_rate": respiration_rate,
            "turnover_rate": turnover_rate,
            "mineralization_rate": mineralization_rate,
            "seed_cost": seed_cost,
            "seed_mass": seed_mass,
            "K_biomass": K_biomass,
            "soil_input_rate": soil_input_rate,
            "soil_availability_rate": sar,
            "soil_pool_mean": soil_pool_mean,
            "soil_pool_std": soil_pool_std,
            "soil_ratio_noise": soil_ratio_noise,
            "initial_seeds": initial_seeds,
        },
        "history_biomass":          history_biomass,
        "history_elements":         history_elements,
        "history_biomass_grid":     history_biomass_grid,
        "N_SPP":                    N_SPP,
        "history_spp_biomass":      history_spp_biomass,
        "history_spp_biomass_grid": history_spp_grid,
        "history_spp_fitness":      history_spp_fitness,   # ← new
        "history_spp_count":        history_agents,
        "soil_snapshot":            st.session_state["soil_snapshot"],  # ← new
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state["pkl_default_name"] = f"sim_{timestamp}"
    st.session_state["run_count"] += 1
    st.session_state["ran"] = True

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state["ran"]:
    res   = st.session_state["results"]
    soil  = st.session_state["soil_snapshot"]
    steps = np.arange(res["N_STEPS"])

    tab_soil, tab_bio, tab_spp, tab_maps = st.tabs([
        "🌍 Initial Soil", "📈 Biomass & Pools", "🧬 Species Dynamics", "🗺️ Spatial Maps"
    ])

    with tab_soil:
        if soil is not None:
            st.subheader("Soil Nutrient Distribution at t=0")
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, (name, idx) in zip(axes.flatten(), zip(NUTRIENT_NAMES, range(4))):
                im = ax.imshow(soil[:, :, idx], cmap="viridis")
                ax.set_title(f"Soil {name}", fontsize=12)
                plt.colorbar(im, ax=ax, label="Concentration")
                ax.set_xlabel("X"); ax.set_ylabel("Y")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Soil snapshot not available in this file (saved before v2).")

    with tab_bio:
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Mean Biomass", f"{res['history_biomass'][-1]:.4f}")
        col2.metric("Peak Agent Count",   f"{max(res['history_agents']):,}")
        col3.metric("Final Agent Count",  f"{res['history_agents'][-1]:,}")

        st.subheader("Total Mean Biomass")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(steps, res["history_biomass"], color="green", lw=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Mean Biomass"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

        st.subheader("Agent Count")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(steps, res["history_agents"], color="steelblue", lw=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Agents"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

        elem_arr = res["history_elements"]
        if elem_arr.ndim == 2 and elem_arr.shape[1] > 0:
            st.subheader("Element Pools Over Time")
            n_pools     = elem_arr.shape[1]
            pool_labels = ELEMENTS[:n_pools]
            fig, axes   = plt.subplots(1, n_pools, figsize=(4 * n_pools, 4))
            if n_pools == 1:
                axes = [axes]
            for ax, label, i in zip(axes, pool_labels, range(n_pools)):
                ax.plot(steps, elem_arr[:, i], lw=2)
                ax.set_title(f"{label} Pool"); ax.set_xlabel("Step"); ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    with tab_spp:
        st.subheader("Per-Species Mean Biomass")
        fig, ax = plt.subplots(figsize=(12, 5))
        for s_id in range(N_SPP):
            ax.plot(steps, res["history_spp_biomass"][s_id],
                    label=SPP_LABELS[s_id], color=SPP_COLORS[s_id], lw=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Mean Biomass")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

        if any(len(f) > 0 for f in res["history_spp_fitness"]):
            st.subheader("Per-Species Mean Fitness")
            fig, ax = plt.subplots(figsize=(12, 5))
            for s_id in range(N_SPP):
                ax.plot(steps, res["history_spp_fitness"][s_id],
                        label=SPP_LABELS[s_id], color=SPP_COLORS[s_id], lw=2)
            ax.set_xlabel("Step"); ax.set_ylabel("Mean Fitness")
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    with tab_maps:
        n_snaps = len(res["history_biomass_grid"])
        snap_i  = st.slider("Snapshot (recorded every 50 steps)",
                            0, n_snaps - 1, n_snaps - 1, format="Step %d")
        actual_step = snap_i * 50
        fig, axes
