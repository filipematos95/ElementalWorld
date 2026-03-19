# frontend/app.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import tensorflow as tf
from models.hybridmodel import HybridEcosystem
import backend.plotting as ep
from datetime import datetime
import pickle, io

st.set_page_config(page_title="Ecosystem Simulator", page_icon="🌿", layout="wide")
st.title("🌿 Hybrid Ecosystem Simulator")
st.markdown("Configure parameters in the sidebar, then click **▶ Run Simulation**.")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_SPP          = 4
SPP_LABELS     = [f"Species {i+1}" for i in range(N_SPP)]
DEFAULT_CENTERS = [
    [0.400, 0.247, 0.035, 0.247, 0.071],
    [0.400, 0.214, 0.086, 0.214, 0.086],
    [0.400, 0.233, 0.133, 0.167, 0.067],
    [0.400, 0.250, 0.050, 0.250, 0.050],
]

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
    scalar_keys = [
        "H", "W", "MAX_AGENTS", "N_STEPS", "SEED",
        "growth_rate", "respiration_rate", "turnover_rate",
        "mineralization_rate", "seed_cost", "seed_mass",
        "K_biomass", "soil_input_rate",
        "soil_pool_mean", "soil_pool_std", "soil_ratio_noise",
        "scalar_interval", "snapshot_interval",
        "input_drift_scale",     "sigma_threshold"
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

    if "cov_code" in params:
        st.session_state["cov_code"] = params["cov_code"]


def _apply_everything(data: dict):
    _apply_config(data["parameters"])
    st.session_state["results"] = {
        "history_biomass":               data["history_biomass"],
        "history_agents":                data["history_spp_count"],
        "history_elements":              np.array(data["history_elements"]),
        "history_biomass_grid":          data["history_biomass_grid"],
        "history_spp_biomass":           data["history_spp_biomass"],
        "history_spp_fitness":           data.get("history_spp_fitness",
                                                  [[] for _ in range(N_SPP)]),
        "history_spp_dead_fitness_mean": data.get("history_spp_dead_fitness_mean",
                                                  [[] for _ in range(N_SPP)]),
        "history_spp_grid":              data["history_spp_biomass_grid"],
        "N_STEPS":                       data["parameters"]["N_STEPS"],
        "scalar_interval":               data["parameters"].get("scalar_interval", 20),
        "snapshot_interval":             data["parameters"].get("snapshot_interval", 50),
        "history_deficit": data.get("history_deficit", [[] for _ in range(N_SPP)]),
    }
    st.session_state["soil_snapshot"]    = data.get("soil_snapshot", None)
    st.session_state["payload"]          = data
    st.session_state["ran"]              = True
    st.session_state["run_count"]       += 1
    st.session_state["pkl_default_name"] = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

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
            help="Restore full results + config." if has_results else "No results in this file.",
        )
        if all_btn:
            _apply_everything(loaded_data)
            st.rerun()

    st.divider()
    st.header("⚙️ Simulation Config")

    with st.form("sim_config_form"):

        st.subheader("Grid & Steps")
        H                 = st.number_input("Grid Height",               10,    500,    100,    key="H")
        W                 = st.number_input("Grid Width",                10,    500,    100,    key="W")
        MAX_AGENTS        = st.number_input("Max Agents",                10000, 500000, 150000, key="MAX_AGENTS",        step=10000)
        N_STEPS           = st.slider("Number of Steps",                 100,   5000,   1500,   key="N_STEPS",           step=100)
        SEED              = st.number_input("Random Seed",               0,     9999,   35,     key="SEED")
        scalar_interval   = st.number_input("Scalar record interval",    1,     500,    20,     key="scalar_interval",
                                            help="Record biomass/agents/elements every N steps.")
        snapshot_interval = st.number_input("Spatial snapshot interval", 1,     500,    50,     key="snapshot_interval",
                                            help="Save full grid maps every N steps.")

        st.subheader("Biological Rates")
        growth_rate         = st.slider("Growth Rate",         0.01,  1.0,  0.45,  key="growth_rate",         step=0.01)
        respiration_rate    = st.slider("Respiration Rate",    0.001, 0.1,  0.015, key="respiration_rate",    step=0.001, format="%.3f")
        turnover_rate       = st.slider("Turnover Rate",       0.001, 0.1,  0.03,  key="turnover_rate",       step=0.001, format="%.3f")
        mineralization_rate = st.slider("Mineralization Rate", 0.01,  0.2,  0.05,  key="mineralization_rate", step=0.005)
        seed_cost           = st.slider("Seed Cost",           0.001, 0.1,  0.02,  key="seed_cost",           step=0.001, format="%.3f")
        seed_mass           = st.slider("Seed Mass",           0.001, 0.1,  0.02,  key="seed_mass",           step=0.001, format="%.3f")
        K_biomass           = st.slider("K Biomass",           0.5,   10.0, 2.5,   key="K_biomass",           step=0.1)
        soil_input_rate     = st.slider("Soil Input Rate",     0.1,   2.0,  0.5,   key="soil_input_rate",     step=0.05)
        sigma_threshold = st.slider("Sigma Threshold", 0.1, 5.0, 3.0,
                                    key="sigma_threshold", step=0.1,
                                    help="Niche fitness sensitivity. Lower = steeper fitness drop away from niche center.")
        st.subheader("Soil Parameters")
        soil_pool_mean    = st.slider("Soil Pool Mean",      0.5,  3.0,  1.5,  key="soil_pool_mean",      step=0.1)
        soil_pool_std     = st.slider("Soil Pool Std",       0.01, 0.5,  0.1,  key="soil_pool_std",       step=0.01)
        soil_ratio_noise  = st.slider("Soil Ratio Noise",    0.0,  0.2,  0.05, key="soil_ratio_noise",    step=0.005)
        input_drift_scale = st.slider("Input Drift Scale",   0.0,  0.3,  0.08, key="input_drift_scale",   step=0.01,  # ← new
                                      help="Noise on soil nutrient input ratio each step. Higher = more environmental fluctuation.")

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

        run_btn = st.form_submit_button(
            "▶ Run Simulation", use_container_width=True, type="primary"
        )

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
                ep.ELEMENTS[e], 0.0, 1.0, DEFAULT_CENTERS[s][e],
                step=0.01, format="%.3f", key=f"nc_{s}_{e}"
            )
            for e in range(5)
        ]
        spp_centers.append(row)

# ─────────────────────────────────────────────
# EXPANDER — Advanced: Covariance Matrices
# ─────────────────────────────────────────────
with st.expander("🔬 Species Covariance Matrices (Advanced)", expanded=False):
    st.caption("Paste a valid numpy array expression. Must be shape (N_SPP, 5, 5).")

    default_cov_code = """np.array([
    [[0.03,0,0,0,0],[0,0.015,0,0,0],[0,0,0.04,0,0],[0,0,0,0.015,0],[0,0,0,0,0.04]],
    [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.03,0,0],[0,0,0,0.03,0],[0,0,0,0,0.03]],
    [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.06,0,0],[0,0,0,0.03,0],[0,0,0,0,0.03]],
    [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.04,0,0],[0,0,0,0.04,0],[0,0,0,0,0.015]],
], dtype=np.float32)"""

    cov_code = st.text_area(
        "Covariance matrix code",
        value=st.session_state.get("cov_code", default_cov_code),
        height=220,
        key="cov_code",
        label_visibility="collapsed",
    )

    try:
        SPP_COVARIANCES = eval(cov_code, {"np": np})
        SPP_COVARIANCES = np.array(SPP_COVARIANCES, dtype=np.float32)
        assert SPP_COVARIANCES.shape == (N_SPP, 5, 5), \
            f"Expected shape ({N_SPP}, 5, 5), got {SPP_COVARIANCES.shape}"
        st.success(f"✅ Valid — shape {SPP_COVARIANCES.shape}")
    except Exception as e:
        st.error(f"❌ Invalid: {e}")
        SPP_COVARIANCES = np.array([
            [[0.03,0,0,0,0],[0,0.015,0,0,0],[0,0,0.04,0,0],[0,0,0,0.015,0],[0,0,0,0,0.04]],
            [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.03,0,0],[0,0,0,0.03,0],[0,0,0,0,0.03]],
            [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.06,0,0],[0,0,0,0.03,0],[0,0,0,0,0.03]],
            [[0.03,0,0,0,0],[0,0.03,0,0,0],[0,0,0.04,0,0],[0,0,0,0.04,0],[0,0,0,0,0.015]],
        ], dtype=np.float32)

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
        input_drift_scale=input_drift_scale,
        sigma_threshold=sigma_threshold,# ← new
    )

    st.session_state["soil_snapshot"] = model.soil.numpy().copy()

    for s_id, n in enumerate(initial_seeds):
        model.add_initial_seeds(count=n, species_id=s_id)

    history_biomass               = []
    history_agents                = []
    history_elements              = []
    history_biomass_grid          = []
    history_spp_biomass           = [[] for _ in range(N_SPP)]
    history_spp_fitness           = [[] for _ in range(N_SPP)]
    history_spp_dead_fitness_mean = [[] for _ in range(N_SPP)]
    history_spp_grid              = [[] for _ in range(N_SPP)]
    history_deficit = [[] for _ in range(N_SPP)]    # ← per species now

    mean_biomass = 0.0
    status = st.empty()
    prog   = st.progress(0)

    for t in range(N_STEPS):
        n_agents   = model.step("mahalanobis")
        grid_total = model.get_biomass_grid()
        if t % scalar_interval == 0:
            mean_biomass = float(np.mean(grid_total))
            history_biomass.append(mean_biomass)
            history_agents.append(int(n_agents.numpy()))
            history_elements.append(model.get_element_pools())
            deficit = model.get_nutrient_deficit()           # shape (N_SPP, 4)
            for s_id in range(N_SPP):
                history_spp_biomass[s_id].append(
                    float(np.mean(model.get_species_biomass(s_id))))
                history_spp_fitness[s_id].append(
                    float(model.get_species_mean_fitness(s_id))
                    if model.get_species_mean_fitness(s_id) is not None else None
                )
                dead_fit = model.get_species_mean_dead_fitness(s_id)
                history_spp_dead_fitness_mean[s_id].append(
                    float(dead_fit) if dead_fit is not None else None)
                history_deficit[s_id].append(deficit[s_id].tolist())
        model.death_fitness_log.clear()

        if t % snapshot_interval == 0:
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
        history_spp_dead_fitness_mean=history_spp_dead_fitness_mean,
        history_deficit=history_deficit,
        history_spp_grid=history_spp_grid,
        N_STEPS=N_STEPS,
        scalar_interval=scalar_interval,
        snapshot_interval=snapshot_interval,
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
            "input_drift_scale": input_drift_scale,
            "sigma_threshold": sigma_threshold,# ← new
            "initial_seeds": initial_seeds,
            "scalar_interval": scalar_interval,
            "snapshot_interval": snapshot_interval,
            "cov_code": st.session_state.get("cov_code", ""),
        },
        "history_biomass":               history_biomass,
        "history_elements":              history_elements,
        "history_biomass_grid":          history_biomass_grid,
        "N_SPP":                         N_SPP,
        "history_spp_biomass":           history_spp_biomass,
        "history_spp_biomass_grid":      history_spp_grid,
        "history_spp_fitness":           history_spp_fitness,
        "history_spp_dead_fitness_mean": history_spp_dead_fitness_mean,
        "history_spp_count":             history_agents,
        "soil_snapshot":                 st.session_state["soil_snapshot"],
    }

    st.session_state["pkl_default_name"] = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["run_count"] += 1
    st.session_state["ran"]        = True

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state["ran"]:
    res  = st.session_state["results"]
    soil = st.session_state["soil_snapshot"]

    scalar_interval   = res.get("scalar_interval",   20)
    snapshot_interval = res.get("snapshot_interval", 50)
    steps_scalar = np.arange(len(res["history_biomass"])) * scalar_interval

    tab_soil, tab_bio, tab_spp, tab_maps = st.tabs([
        "🌍 Initial Soil", "📈 Biomass & Pools", "🧬 Species Dynamics", "🗺️ Spatial Maps"
    ])

    with tab_soil:
        if soil is not None:
            st.subheader("Soil Nutrient Distribution at t=0")
            st.plotly_chart(ep.plot_soil(soil), use_container_width=True)
        else:
            st.info("Soil snapshot not available in this file (saved before v2).")

    with tab_bio:
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Mean Biomass", f"{res['history_biomass'][-1]:.4f}")
        col2.metric("Peak Agent Count",   f"{max(res['history_agents']):,}")
        col3.metric("Final Agent Count",  f"{res['history_agents'][-1]:,}")

        st.subheader("Total Mean Biomass")
        st.plotly_chart(ep.plot_biomass(steps_scalar, res["history_biomass"]),
                        use_container_width=True)

        st.subheader("Agent Count")
        st.plotly_chart(ep.plot_agents(steps_scalar, res["history_agents"]),
                        use_container_width=True)

        elem_arr = res["history_elements"]
        if elem_arr.ndim == 2 and elem_arr.shape[1] > 0:
            st.subheader("Element Pools Over Time")
            st.plotly_chart(ep.plot_element_pools(steps_scalar, elem_arr),
                            use_container_width=True)

        deficit_data = res.get("history_deficit", [])
        st.write(f"DEBUG deficit_data length: {len(deficit_data)}, first entry: {deficit_data[0][:2] if deficit_data else 'empty'}")
        if any(len(d) > 0 for d in deficit_data):
            st.subheader("Unmet Nutrient Demand per Species")
            st.plotly_chart(ep.plot_nutrient_deficit(steps_scalar, deficit_data, SPP_LABELS),
                            use_container_width=True)

    with tab_spp:
        st.subheader("Per-Species Mean Biomass")
        st.plotly_chart(ep.plot_species_biomass(steps_scalar, res["history_spp_biomass"],
                                                SPP_LABELS), use_container_width=True)

        if any(len(f) > 0 for f in res["history_spp_fitness"]):
            st.subheader("Per-Species Mean Fitness")
            st.plotly_chart(ep.plot_species_fitness(steps_scalar, res["history_spp_fitness"],
                                                    SPP_LABELS), use_container_width=True)

        dead = res.get("history_spp_dead_fitness_mean", [])
        if any(len(f) > 0 for f in dead):
            st.subheader("Mean Fitness at Time of Death")
            st.plotly_chart(ep.plot_dead_fitness(steps_scalar, dead, SPP_LABELS),
                            use_container_width=True)

    with tab_maps:
        n_snaps = len(res["history_biomass_grid"])
        snap_i  = st.slider(f"Snapshot (recorded every {snapshot_interval} steps)",
                            0, n_snaps - 1, n_snaps - 1, format="Step %d")
        actual_step = snap_i * snapshot_interval
        st.plotly_chart(ep.plot_spatial_maps(res["history_biomass_grid"],
                                             res["history_spp_grid"], snap_i, actual_step,
                                             SPP_LABELS), use_container_width=True)

    # ─────────────────────────────────────────────
    # SAVE — inside ran block, outside tabs
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Save Results")
    col_name, col_save, col_dl = st.columns([3, 1, 1])

    save_name = col_name.text_input(
        "Filename (without .pkl)",
        value=st.session_state["pkl_default_name"],
        key=f"save_name_{st.session_state['run_count']}",
    )
    clean_name = save_name.strip().replace(" ", "_") or st.session_state["pkl_default_name"]

    if col_save.button("💾 Save to disk", use_container_width=True, type="primary"):
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{clean_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(st.session_state["payload"], f, protocol=pickle.HIGHEST_PROTOCOL)
        st.success(f"✅ Saved to `{filepath}`")

    buf = io.BytesIO()
    pickle.dump(st.session_state["payload"], buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    col_dl.download_button(
        "⬇️ Download .pkl", data=buf,
        file_name=f"{clean_name}.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )
