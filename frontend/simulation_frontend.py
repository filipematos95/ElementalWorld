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
import json

st.set_page_config(page_title="Ecosystem Simulator", page_icon="🌿", layout="wide")
st.title("🌿 Hybrid Ecosystem Simulator")
st.markdown("Configure parameters in the sidebar, then click **▶ Run Simulation**.")


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "default_config.json")


def _load_default_config() -> dict | None:
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return json.load(f)
    return None


_saved_defaults = _load_default_config()

DEFAULT_CENTERS = [
    [0.4192, 0.0208, 0.0014, 0.0077, 0.5509],
    [0.4282, 0.0201, 0.0011, 0.0097, 0.5408],
    [0.4020, 0.0223, 0.0011, 0.0072, 0.5674],
    [0.4400, 0.0267, 0.0014, 0.0097, 0.5221],
    [0.3839, 0.0195, 0.0017, 0.0055, 0.5894],
]

DEFAULT_COV_CODE = """np.array([
  [[0.00027, -0.00011, -0.00001, -0.00001, -0.00009], [-0.00011, 0.00014, 0.00001, 0.00001, 0.00001], [-0.00001, 0.00001, 0.00006, 0.00000, -0.00000], [-0.00001, 0.00001, 0.00000, 0.00007, -0.00001], [-0.00009, 0.00001, -0.00000, -0.00001, 0.00015]],
  [[0.00049, -0.00003, -0.00000, -0.00004, -0.00033], [-0.00003, 0.00009, 0.00000, -0.00000, 0.00002], [-0.00000, 0.00000, 0.00008, 0.00000, 0.00000], [-0.00004, -0.00000, 0.00000, 0.00010, 0.00002], [-0.00033, 0.00002, 0.00000, 0.00002, 0.00037]],
  [[0.00098, 0.00002, 0.00000, 0.00000, -0.00087], [0.00002, 0.00015, 0.00000, 0.00001, -0.00005], [0.00000, 0.00000, 0.00013, 0.00000, -0.00000], [0.00000, 0.00001, 0.00000, 0.00014, -0.00002], [-0.00087, -0.00005, -0.00000, -0.00002, 0.00108]],
  [[0.00053, -0.00000, 0.00000, 0.00003, -0.00045], [-0.00000, 0.00016, 0.00000, 0.00001, -0.00007], [0.00000, 0.00000, 0.00010, 0.00000, -0.00000], [0.00003, 0.00001, 0.00000, 0.00010, -0.00004], [-0.00045, -0.00007, -0.00000, -0.00004, 0.00067]],
  [[0.00047, 0.00002, -0.00001, -0.00001, -0.00038], [0.00002, 0.00010, 0.00000, 0.00000, -0.00003], [-0.00001, 0.00000, 0.00009, 0.00000, 0.00000], [-0.00001, 0.00000, 0.00000, 0.00010, 0.00000], [-0.00038, -0.00003, 0.00000, 0.00000, 0.00050]]
], dtype=np.float32)"""

DEFAULT_COV = np.array([
    [[0.00027, -0.00011, -0.00001, -0.00001, -0.00009], [-0.00011, 0.00014, 0.00001, 0.00001, 0.00001], [-0.00001, 0.00001, 0.00006, 0.00000, -0.00000], [-0.00001, 0.00001, 0.00000, 0.00007, -0.00001], [-0.00009, 0.00001, -0.00000, -0.00001, 0.00015]],
    [[0.00049, -0.00003, -0.00000, -0.00004, -0.00033], [-0.00003, 0.00009, 0.00000, -0.00000, 0.00002], [-0.00000, 0.00000, 0.00008, 0.00000, 0.00000], [-0.00004, -0.00000, 0.00000, 0.00010, 0.00002], [-0.00033, 0.00002, 0.00000, 0.00002, 0.00037]],
    [[0.00098, 0.00002, 0.00000, 0.00000, -0.00087], [0.00002, 0.00015, 0.00000, 0.00001, -0.00005], [0.00000, 0.00000, 0.00013, 0.00000, -0.00000], [0.00000, 0.00001, 0.00000, 0.00014, -0.00002], [-0.00087, -0.00005, -0.00000, -0.00002, 0.00108]],
    [[0.00053, -0.00000, 0.00000, 0.00003, -0.00045], [-0.00000, 0.00016, 0.00000, 0.00001, -0.00007], [0.00000, 0.00000, 0.00010, 0.00000, -0.00000], [0.00003, 0.00001, 0.00000, 0.00010, -0.00004], [-0.00045, -0.00007, -0.00000, -0.00004, 0.00067]],
    [[0.00047, 0.00002, -0.00001, -0.00001, -0.00038], [0.00002, 0.00010, 0.00000, 0.00000, -0.00003], [-0.00001, 0.00000, 0.00009, 0.00000, 0.00000], [-0.00001, 0.00000, 0.00000, 0.00010, 0.00000], [-0.00038, -0.00003, 0.00000, 0.00000, 0.00050]]
], dtype=np.float32)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [
    ("N_SPP", 5),
    ("results", None),
    ("soil_snapshot", None),
    ("ran", False),
    ("run_count", 0),
    ("payload", None),
    ("pkl_default_name", ""),
    ("cov_code", DEFAULT_COV_CODE),
    ("loaded_final_states", None),
    ("loaded_completed_steps", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
def _apply_config(params: dict):
    scalar_keys = [
        "H", "W", "MAX_AGENTS", "N_STEPS", "SEED", "NSEEDS",
        "growth_rate", "respiration_rate", "turnover_rate",
        "mineralization_rate", "seed_cost", "seed_mass",
        "K_biomass", "soil_input_rate",
        "soil_pool_mean", "soil_pool_std", "soil_ratio_noise",
        "scalar_interval", "snapshot_interval",
        "input_drift_scale", "sigma_threshold",
        "catastrophe_interval", "catastrophe_mortality",
        "p_disturbance", "disturbance_strength", "demo_noise_std",
    ]
    if "N_SPP" in params:
        st.session_state["N_SPP"] = int(params["N_SPP"])
    n_spp = st.session_state["N_SPP"]

    for k in scalar_keys:
        if k in params:
            st.session_state[k] = params[k]

    for key, val in zip(
            ["sbr_n", "sbr_p", "sbr_k", "sbr_o"],
            params.get("soil_base_ratio", [0.35, 0.1, 0.35, 0.1])
    ):
        st.session_state[key] = float(val)

    for key, val in zip(
            ["sar_n", "sar_p", "sar_k", "sar_o"],
            params.get("soil_availability_rate", [0.4, 0.1, 0.1, 0.3])
    ):
        st.session_state[key] = float(val)

    for i, n in enumerate(params.get("initial_seeds", [10] * n_spp)):
        st.session_state[f"seeds_{i}"] = int(n)

    for s, row in enumerate(params.get("spp_centers", DEFAULT_CENTERS)):
        for e, val in enumerate(row):
            st.session_state[f"nc_{s}_{e}"] = float(val)

    if "cov_code" in params:
        st.session_state["cov_code"] = params["cov_code"]


def _apply_everything(data: dict):
    _apply_config(data["parameters"])
    n_spp = st.session_state["N_SPP"]

    st.session_state["results"] = {
        "history_biomass": data["history_biomass"],
        "history_agents": data["history_spp_count"],
        "history_elements": np.array(data["history_elements"]),
        "history_biomass_grid": data["history_biomass_grid"],
        "history_spp_biomass": data["history_spp_biomass"],
        "history_spp_age": data.get("history_spp_age", [[] for _ in range(n_spp)]),
        "history_spp_biomass_std": data.get("history_spp_biomass_std", [[] for _ in range(n_spp)]),
        "history_spp_fitness": data.get("history_spp_fitness", [[] for _ in range(n_spp)]),
        "history_spp_dead_fitness_mean": data.get("history_spp_dead_fitness_mean", [[] for _ in range(n_spp)]),
        "history_spp_grid": data["history_spp_biomass_grid"],
        "N_STEPS": data.get("completed_steps", data["parameters"]["N_STEPS"]),
        "scalar_interval": data["parameters"].get("scalar_interval", 20),
        "snapshot_interval": data["parameters"].get("snapshot_interval", 50),
        "history_deficit": data.get("history_deficit", [[] for _ in range(n_spp)]),
        "n_seeds_used": data["parameters"].get("NSEEDS", 1),
        "N_SPP": n_spp,
        "history_spp_elemental_dissimilarity": data.get("history_spp_elemental_dissimilarity", []),
    }

    st.session_state["soil_snapshot"] = data.get("soil_snapshot", None)
    st.session_state["payload"] = data
    st.session_state["loaded_final_states"] = data.get("final_states", None)
    st.session_state["loaded_completed_steps"] = data.get("completed_steps", 0)
    st.session_state["ran"] = True
    st.session_state["run_count"] += 1
    st.session_state["pkl_default_name"] = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _make_model(
        H, W, MAX_AGENTS, spp_centers, SPP_COVARIANCES,
        growth_rate, respiration_rate, turnover_rate,
        mineralization_rate, seed_cost, seed_mass, K_biomass, sbr,
        soil_pool_mean, soil_pool_std, soil_ratio_noise, soil_input_rate,
        sar, input_drift_scale, sigma_threshold,
        catastrophe_interval, catastrophe_mortality,
        p_disturbance, disturbance_strength, demo_noise_std
):
    return HybridEcosystem(
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
        sigma_threshold=sigma_threshold,
        catastrophe_interval=catastrophe_interval,
        catastrophe_mortality=catastrophe_mortality,
        p_disturbance=p_disturbance,
        disturbance_strength=disturbance_strength,
        demo_noise_std=demo_noise_std,
    )


# ─────────────────────────────────────────────
# ENSEMBLE HELPER
# ─────────────────────────────────────────────
def _run_one_seed(
        seed, H, W, MAX_AGENTS, N_STEPS, spp_centers, SPP_COVARIANCES,
        initial_seeds, growth_rate, respiration_rate, turnover_rate,
        mineralization_rate, seed_cost, seed_mass, K_biomass, sbr,
        soil_pool_mean, soil_pool_std, soil_ratio_noise, soil_input_rate,
        sar, input_drift_scale, sigma_threshold,
        catastrophe_interval, catastrophe_mortality,
        p_disturbance, disturbance_strength,
        demo_noise_std,
        scalar_interval, snapshot_interval,
        prog_offset, prog_total, prog_bar, status_el,
        resume_state=None, start_step=0
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = _make_model(
        H, W, MAX_AGENTS, spp_centers, SPP_COVARIANCES,
        growth_rate, respiration_rate, turnover_rate,
        mineralization_rate, seed_cost, seed_mass, K_biomass, sbr,
        soil_pool_mean, soil_pool_std, soil_ratio_noise, soil_input_rate,
        sar, input_drift_scale, sigma_threshold,
        catastrophe_interval, catastrophe_mortality,
        p_disturbance, disturbance_strength, demo_noise_std
    )
    n_spp = len(spp_centers)

    if resume_state is None:
        soil_snap = model.soil.numpy().copy()
        for s_id, n in enumerate(initial_seeds):
            model.add_initial_seeds(count=n, species_id=s_id)
    else:
        model.agents.assign(resume_state["agents"])
        model.soil.assign(resume_state["soil"])
        soil_snap = resume_state["soil"].copy()

    history_biomass = []
    history_agents = []
    history_elements = []
    history_biomass_grid = []
    history_spp_biomass = [[] for _ in range(n_spp)]
    history_spp_fitness = [[] for _ in range(n_spp)]
    history_spp_dead_fitness_mean = [[] for _ in range(n_spp)]
    history_spp_grid = [[] for _ in range(n_spp)]
    history_deficit = [[] for _ in range(n_spp)]
    history_spp_age = [[] for _ in range(n_spp)]
    history_spp_elemental_dissimilarity = []

    mean_biomass = 0.0
    total_end_step = start_step + N_STEPS

    for t in range(start_step, total_end_step):
        n_agents = model.step("mahalanobis")
        grid_total = model.get_biomass_grid()

        if t % scalar_interval == 0:
            mean_biomass = float(np.mean(grid_total))
            history_biomass.append(mean_biomass)
            history_agents.append(int(n_agents.numpy()))
            history_elements.append(model.get_element_pools())
            deficit = model.get_nutrient_deficit()
            history_spp_elemental_dissimilarity.append(
                model.get_species_elemental_dissimilarity_index_tf()
            )

            for s_id in range(n_spp):
                history_spp_biomass[s_id].append(float(np.mean(model.get_species_biomass(s_id))))
                fit = model.get_species_mean_fitness(s_id)
                history_spp_fitness[s_id].append(float(fit) if fit is not None else None)
                dead_fit = model.get_species_mean_dead_fitness(s_id)
                history_spp_dead_fitness_mean[s_id].append(float(dead_fit) if dead_fit is not None else None)
                history_deficit[s_id].append(deficit[s_id].tolist())
                history_spp_age[s_id].append(model.get_species_mean_age(s_id))

        model.death_fitness_log.clear()

        if t % snapshot_interval == 0:
            history_biomass_grid.append(grid_total)
            for s_id in range(n_spp):
                history_spp_grid[s_id].append(model.get_species_biomass(s_id))

        overall = (prog_offset + ((t - start_step) + 1) / N_STEPS) / prog_total
        prog_bar.progress(min(overall, 1.0))
        if t % 10 == 0:
            status_el.text(
                f"Seed {seed} — Step {t}/{total_end_step} | "
                f"Agents: {n_agents.numpy()} | Mean Biomass: {mean_biomass:.4f}"
            )

    return dict(
        soil_snap=soil_snap,
        history_biomass=history_biomass,
        history_agents=history_agents,
        history_elements=history_elements,
        history_biomass_grid=history_biomass_grid,
        history_spp_biomass=history_spp_biomass,
        history_spp_fitness=history_spp_fitness,
        history_spp_dead_fitness_mean=history_spp_dead_fitness_mean,
        history_spp_grid=history_spp_grid,
        history_deficit=history_deficit,
        history_spp_age=history_spp_age,
        history_spp_elemental_dissimilarity=history_spp_elemental_dissimilarity,
        final_state=dict(
            agents=model.agents.numpy().copy(),
            soil=model.soil.numpy().copy(),
            step_count=total_end_step,
        ),
    )


def _avg_nullable(runs_list):
    arr = np.array([[v if v is not None else np.nan for v in run] for run in runs_list], dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    return [None if np.isnan(v) else float(v) for v in mean]


def _avg_grids(runs_list):
    n_snaps = len(runs_list[0])
    return [np.mean([run[i] for run in runs_list], axis=0) for i in range(n_snaps)]


def _append_series(old, new):
    if old is None or len(old) == 0:
        return list(new)
    return list(old) + list(new)


def _append_nested(old, new):
    if old is None or len(old) == 0:
        return new
    return [list(old[i]) + list(new[i]) for i in range(len(new))]


if _saved_defaults and st.session_state.get("_defaults_applied") is None:
    _apply_config(_saved_defaults)
    st.session_state["_defaults_applied"] = True


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Simulation")
    uploaded = st.file_uploader("Upload a .pkl file", type=["pkl"], label_visibility="collapsed")
    continue_run = False

    if uploaded is not None:
        loaded_data = pickle.load(uploaded)
        has_results = "history_biomass" in loaded_data
        has_state = "final_states" in loaded_data

        col_cfg, col_all = st.columns(2)

        if col_cfg.button("⚙️ Config only", use_container_width=True,
                          help="Populate sidebar parameters; does not restore plots."):
            _apply_config(loaded_data["parameters"])
            st.session_state["loaded_final_states"] = loaded_data.get("final_states", None)
            st.session_state["loaded_completed_steps"] = loaded_data.get("completed_steps", 0)
            st.rerun()

        all_btn = col_all.button(
            "📊 Everything", use_container_width=True,
            disabled=not has_results,
            help="Restore full results + config." if has_results else "No results in this file.",
        )
        if all_btn:
            _apply_everything(loaded_data)
            st.rerun()

        continue_run = st.checkbox(
            "↩️ Continue from this simulation",
            value=False,
            disabled=not has_state,
            help="Use the saved final state as the starting point for more steps."
        )
        if continue_run and not has_state:
            st.warning("This file does not contain a saved state, so it cannot be resumed.")

    st.divider()
    st.header("⚙️ Simulation Config")

    _prev_n_spp = st.session_state["N_SPP"]
    N_SPP = st.number_input("Number of Species", min_value=1, max_value=10,
                            value=st.session_state["N_SPP"], step=1, key="N_SPP")
    if N_SPP != _prev_n_spp:
        st.rerun()

    SPP_LABELS = [f"Species {i+1}" for i in range(N_SPP)]

    with st.form("sim_config_form"):
        st.subheader("Grid & Steps")
        H = st.number_input("Grid Height", 10, 500, 100, key="H")
        W = st.number_input("Grid Width", 10, 500, 100, key="W")
        MAX_AGENTS = st.number_input("Max Agents", 10000, 500000, 150000, key="MAX_AGENTS", step=10000)
        N_STEPS = st.slider("Number of Steps", 100, 5000, 1500, key="N_STEPS", step=100)
        SEED = st.number_input("Random Seed", 0, 9999, 35, key="SEED")
        NSEEDS = st.slider("Ensemble Seeds", 1, 20, 1, key="NSEEDS",
                           help="Run N simulations from SEED to SEED+N-1 and average results.")
        scalar_interval = st.number_input("Scalar record interval", 1, 500, 20, key="scalar_interval",
                                          help="Record biomass/agents/elements every N steps.")
        snapshot_interval = st.number_input("Spatial snapshot interval", 1, 500, 50, key="snapshot_interval",
                                            help="Save full grid maps every N steps.")

        st.subheader("Biological Rates")
        growth_rate = st.slider("Growth Rate", 0.01, 1.0, 0.45, key="growth_rate", step=0.01)
        respiration_rate = st.slider("Respiration Rate", 0.001, 0.1, 0.015, key="respiration_rate", step=0.001, format="%.3f")
        turnover_rate = st.slider("Turnover Rate", 0.001, 0.1, 0.03, key="turnover_rate", step=0.001, format="%.3f")
        mineralization_rate = st.slider("Mineralization Rate", 0.01, 0.2, 0.05, key="mineralization_rate", step=0.005)
        seed_cost = st.slider("Seed Cost", 0.001, 0.1, 0.02, key="seed_cost", step=0.001, format="%.3f")
        seed_mass = st.slider("Seed Mass", 0.001, 0.1, 0.02, key="seed_mass", step=0.001, format="%.3f")
        K_biomass = st.slider("K Biomass", 0.5, 10.0, 2.5, key="K_biomass", step=0.1)
        soil_input_rate = st.slider("Soil Input Rate", 0.1, 2.0, 0.5, key="soil_input_rate", step=0.05)
        sigma_threshold = st.slider("Sigma Threshold", 0.1, 5.0, 3.0, key="sigma_threshold", step=0.1,
                                    help="Niche fitness sensitivity. Lower = steeper fitness drop away from niche center.")

        st.subheader("Soil Parameters")
        soil_pool_mean = st.slider("Soil Pool Mean", 0.5, 3.0, 1.5, key="soil_pool_mean", step=0.1)
        soil_pool_std = st.slider("Soil Pool Std", 0.01, 0.5, 0.1, key="soil_pool_std", step=0.01)
        soil_ratio_noise = st.slider("Soil Ratio Noise", 0.0, 0.2, 0.05, key="soil_ratio_noise", step=0.005)
        input_drift_scale = st.slider("Input Drift Scale", 0.0, 0.3, 0.08, key="input_drift_scale", step=0.01,
                                      help="Noise on soil nutrient input ratio each step. Higher = more environmental fluctuation.")

        st.subheader("Soil Base Ratios [N, P, K, O]")
        c1, c2 = st.columns(2)
        sbr = [
            c1.number_input("N", 0.0, 1.0, 0.35, step=0.05, key="sbr_n"),
            c1.number_input("P", 0.0, 1.0, 0.10, step=0.05, key="sbr_p"),
            c2.number_input("K", 0.0, 1.0, 0.35, step=0.05, key="sbr_k"),
            c2.number_input("O", 0.0, 1.0, 0.10, step=0.05, key="sbr_o"),
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

        st.subheader("🌩️ Stochastic Destabilization")
        catastrophe_interval = st.slider("Catastrophe Interval", 50, 500, 200, step=50,
                                         key="catastrophe_interval",
                                         help="Steps between mass mortality events.")
        catastrophe_mortality = st.slider("Catastrophe Mortality", 0.0, 0.9, 0.4, step=0.05,
                                          key="catastrophe_mortality",
                                          help="Fraction of agents killed each catastrophe.")
        p_disturbance = st.slider("Disturbance Prob.", 0.0, 0.05, 0.01, step=0.005,
                                  key="p_disturbance",
                                  help="Probability per step of a local soil depletion patch.")
        disturbance_strength = st.slider("Disturbance Strength", 0.0, 1.0, 0.7, step=0.05,
                                         key="disturbance_strength",
                                         help="Fraction of soil nutrients removed in the disturbed patch.")
        demo_noise_std = st.slider("Demographic Noise", 0.0, 0.01, 0.003, step=0.001,
                                   key="demo_noise_std", format="%.3f",
                                   help="Std of random mass perturbation each step.")

        st.divider()
        st.subheader("🧬 Species Niche Centers")
        st.caption("Columns = stoichiometric ideal [C, N, P, K, O].")
        spp_centers = []
        for s in range(N_SPP):
            st.markdown(f"**Species {s+1}**")
            cols = st.columns(5)
            row = [
                cols[e].number_input(
                    ep.ELEMENTS[e], 0.0, 1.0,
                    float(st.session_state.get(
                        f"nc_{s}_{e}",
                        DEFAULT_CENTERS[s][e] if s < len(DEFAULT_CENTERS) else 0.0
                    )),
                    step=0.01, format="%.3f", key=f"nc_{s}_{e}",
                )
                for e in range(5)
            ]
            spp_centers.append(row)

        st.divider()
        st.subheader("🔬 Covariance Matrices")
        st.caption("Numpy array expression, shape (N_SPP, 5, 5).")
        cov_code = st.text_area(
            "Covariance code",
            value=st.session_state.get("cov_code", DEFAULT_COV_CODE),
            height=220,
            key="cov_code",
            label_visibility="collapsed",
        )
        try:
            parsed = eval(cov_code, {"np": np})
            SPP_COVARIANCES = np.array(parsed, dtype=np.float32)
            assert SPP_COVARIANCES.shape == (N_SPP, 5, 5), f"Expected ({N_SPP}, 5, 5), got {SPP_COVARIANCES.shape}"
            st.success(f"✅ Valid — shape {SPP_COVARIANCES.shape}")
        except Exception as e:
            st.error(f"❌ {e}")
            SPP_COVARIANCES = np.array([np.eye(5, dtype=np.float32) * 0.03 for _ in range(N_SPP)])

        st.divider()
        st.subheader("💾 Save Options")
        save_dir = st.text_input("Save directory", value="results", key="save_dir")

        run_btn = st.form_submit_button("▶ Run Simulation", use_container_width=True, type="primary")

    st.divider()
    if st.button("⭐ Set as default config", use_container_width=True,
                 help="Saves current sidebar config as the startup default."):
        _n = st.session_state.get("N_SPP", 5)
        _default = {
            "N_SPP": _n,
            "H": st.session_state.get("H", 100),
            "W": st.session_state.get("W", 100),
            "MAX_AGENTS": st.session_state.get("MAX_AGENTS", 150000),
            "N_STEPS": st.session_state.get("N_STEPS", 1500),
            "SEED": st.session_state.get("SEED", 35),
            "NSEEDS": st.session_state.get("NSEEDS", 1),
            "scalar_interval": st.session_state.get("scalar_interval", 20),
            "snapshot_interval": st.session_state.get("snapshot_interval", 50),
            "growth_rate": st.session_state.get("growth_rate", 0.45),
            "respiration_rate": st.session_state.get("respiration_rate", 0.015),
            "turnover_rate": st.session_state.get("turnover_rate", 0.03),
            "mineralization_rate": st.session_state.get("mineralization_rate", 0.05),
            "seed_cost": st.session_state.get("seed_cost", 0.02),
            "seed_mass": st.session_state.get("seed_mass", 0.02),
            "K_biomass": st.session_state.get("K_biomass", 2.5),
            "soil_input_rate": st.session_state.get("soil_input_rate", 0.5),
            "sigma_threshold": st.session_state.get("sigma_threshold", 3.0),
            "soil_pool_mean": st.session_state.get("soil_pool_mean", 1.5),
            "soil_pool_std": st.session_state.get("soil_pool_std", 0.1),
            "soil_ratio_noise": st.session_state.get("soil_ratio_noise", 0.05),
            "input_drift_scale": st.session_state.get("input_drift_scale", 0.08),
            "catastrophe_interval": st.session_state.get("catastrophe_interval", 200),
            "catastrophe_mortality": st.session_state.get("catastrophe_mortality", 0.4),
            "p_disturbance": st.session_state.get("p_disturbance", 0.01),
            "disturbance_strength": st.session_state.get("disturbance_strength", 0.7),
            "demo_noise_std": st.session_state.get("demo_noise_std", 0.003),
            "soil_base_ratio": [
                st.session_state.get("sbr_n", 0.35),
                st.session_state.get("sbr_p", 0.10),
                st.session_state.get("sbr_k", 0.35),
                st.session_state.get("sbr_o", 0.10),
            ],
            "soil_availability_rate": [
                st.session_state.get("sar_n", 0.4),
                st.session_state.get("sar_p", 0.1),
                st.session_state.get("sar_k", 0.1),
                st.session_state.get("sar_o", 0.3),
            ],
            "initial_seeds": [st.session_state.get(f"seeds_{i}", 10) for i in range(_n)],
            "spp_centers": [
                [st.session_state.get(f"nc_{s}_{e}", 0.0) for e in range(5)]
                for s in range(_n)
            ],
            "cov_code": st.session_state.get("cov_code", DEFAULT_COV_CODE),
        }
        os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(_default, f, indent=2)
        st.success(f"✅ Default config saved to `{DEFAULT_CONFIG_PATH}`")


# ─────────────────────────────────────────────
# SIMULATION — ensemble loop
# ─────────────────────────────────────────────
if run_btn:
    status = st.empty()
    prog = st.progress(0.0)

    resume_states = None
    start_step = 0
    previous_results = None

    if continue_run and st.session_state.get("loaded_final_states") is not None:
        resume_states = st.session_state["loaded_final_states"]
        start_step = st.session_state.get("loaded_completed_steps", 0)
        previous_results = st.session_state.get("results", None)

        if len(resume_states) != NSEEDS:
            st.error("Loaded checkpoint seed count does not match current NSEEDS.")
            st.stop()

    all_runs = []
    for i, seed in enumerate(range(SEED, SEED + NSEEDS)):
        resume_state = None if resume_states is None else resume_states[i]

        result = _run_one_seed(
            seed=seed,
            H=H, W=W, MAX_AGENTS=MAX_AGENTS, N_STEPS=N_STEPS,
            spp_centers=spp_centers, SPP_COVARIANCES=SPP_COVARIANCES,
            initial_seeds=initial_seeds,
            growth_rate=growth_rate, respiration_rate=respiration_rate,
            turnover_rate=turnover_rate, mineralization_rate=mineralization_rate,
            seed_cost=seed_cost, seed_mass=seed_mass, K_biomass=K_biomass,
            sbr=sbr, soil_pool_mean=soil_pool_mean, soil_pool_std=soil_pool_std,
            soil_ratio_noise=soil_ratio_noise, soil_input_rate=soil_input_rate,
            sar=sar, input_drift_scale=input_drift_scale,
            sigma_threshold=sigma_threshold,
            catastrophe_interval=catastrophe_interval,
            catastrophe_mortality=catastrophe_mortality,
            p_disturbance=p_disturbance,
            disturbance_strength=disturbance_strength,
            demo_noise_std=demo_noise_std,
            scalar_interval=scalar_interval, snapshot_interval=snapshot_interval,
            prog_offset=i, prog_total=NSEEDS,
            prog_bar=prog, status_el=status,
            resume_state=resume_state, start_step=start_step,
        )
        all_runs.append(result)

    history_biomass = np.mean([r["history_biomass"] for r in all_runs], axis=0).tolist()
    history_agents = np.mean([r["history_agents"] for r in all_runs], axis=0).astype(int).tolist()
    history_elements = np.mean([r["history_elements"] for r in all_runs], axis=0)
    history_spp_elemental_dissimilarity = np.mean(
        [r["history_spp_elemental_dissimilarity"] for r in all_runs], axis=0
    ).tolist()

    history_spp_biomass = []
    history_spp_biomass_std = []
    for s in range(N_SPP):
        arr = np.array([r["history_spp_biomass"][s] for r in all_runs])
        history_spp_biomass.append(arr.mean(axis=0).tolist())
        history_spp_biomass_std.append(arr.std(axis=0).tolist())

    history_spp_fitness = [
        _avg_nullable([r["history_spp_fitness"][s] for r in all_runs])
        for s in range(N_SPP)
    ]
    history_spp_age = [
        np.mean([r["history_spp_age"][s] for r in all_runs], axis=0).tolist()
        for s in range(N_SPP)
    ]
    history_spp_dead_fitness_mean = [
        _avg_nullable([r["history_spp_dead_fitness_mean"][s] for r in all_runs])
        for s in range(N_SPP)
    ]
    history_deficit = [
        [
            np.mean([r["history_deficit"][s][t] for r in all_runs], axis=0).tolist()
            for t in range(len(all_runs[0]["history_deficit"][s]))
        ]
        for s in range(N_SPP)
    ]

    history_biomass_grid = _avg_grids([r["history_biomass_grid"] for r in all_runs])
    history_spp_grid = [
        _avg_grids([r["history_spp_grid"][s] for r in all_runs])
        for s in range(N_SPP)
    ]

    if previous_results is not None:
        history_biomass = _append_series(previous_results["history_biomass"], history_biomass)
        history_agents = _append_series(previous_results["history_agents"], history_agents)
        history_elements = np.concatenate(
            [np.array(previous_results["history_elements"]), np.array(history_elements)],
            axis=0
        )
        history_spp_elemental_dissimilarity = _append_series(
            previous_results.get("history_spp_elemental_dissimilarity", []),
            history_spp_elemental_dissimilarity
        )
        history_spp_biomass = _append_nested(previous_results["history_spp_biomass"], history_spp_biomass)
        history_spp_biomass_std = _append_nested(previous_results["history_spp_biomass_std"], history_spp_biomass_std)
        history_spp_fitness = _append_nested(previous_results["history_spp_fitness"], history_spp_fitness)
        history_spp_dead_fitness_mean = _append_nested(previous_results["history_spp_dead_fitness_mean"], history_spp_dead_fitness_mean)
        history_spp_age = _append_nested(previous_results["history_spp_age"], history_spp_age)
        history_deficit = _append_nested(previous_results["history_deficit"], history_deficit)
        history_biomass_grid = list(previous_results["history_biomass_grid"]) + list(history_biomass_grid)
        history_spp_grid = [
            list(previous_results["history_spp_grid"][s]) + list(history_spp_grid[s])
            for s in range(N_SPP)
        ]

    soil_snapshot = all_runs[0]["soil_snap"]
    completed_steps = start_step + N_STEPS

    label = f"seeds {SEED}–{SEED + NSEEDS - 1}" if NSEEDS > 1 else f"seed {SEED}"
    if start_step > 0:
        status.success(f"✅ Continued to {completed_steps} total steps × {NSEEDS} seed(s) ({label}).")
    else:
        status.success(f"✅ Done — {N_STEPS} steps × {NSEEDS} seed(s) ({label}) averaged.")
    prog.progress(1.0)

    st.session_state["soil_snapshot"] = soil_snapshot
    st.session_state["results"] = dict(
        history_biomass=history_biomass,
        history_agents=history_agents,
        history_elements=np.array(history_elements),
        history_biomass_grid=history_biomass_grid,
        history_spp_biomass=history_spp_biomass,
        history_spp_biomass_std=history_spp_biomass_std,
        history_spp_fitness=history_spp_fitness,
        history_spp_dead_fitness_mean=history_spp_dead_fitness_mean,
        history_spp_elemental_dissimilarity=history_spp_elemental_dissimilarity,
        history_deficit=history_deficit,
        history_spp_grid=history_spp_grid,
        history_spp_age=history_spp_age,
        N_STEPS=completed_steps,
        scalar_interval=scalar_interval,
        snapshot_interval=snapshot_interval,
        n_seeds_used=NSEEDS,
        N_SPP=N_SPP,
    )

    st.session_state["payload"] = {
        "parameters": {
            "H": H, "W": W, "MAX_AGENTS": MAX_AGENTS, "N_STEPS": completed_steps,
            "SEED": SEED, "NSEEDS": NSEEDS,
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
            "sigma_threshold": sigma_threshold,
            "catastrophe_interval": catastrophe_interval,
            "catastrophe_mortality": catastrophe_mortality,
            "p_disturbance": p_disturbance,
            "disturbance_strength": disturbance_strength,
            "demo_noise_std": demo_noise_std,
            "initial_seeds": initial_seeds,
            "scalar_interval": scalar_interval,
            "snapshot_interval": snapshot_interval,
            "cov_code": cov_code,
            "N_SPP": N_SPP,
        },
        "history_biomass": history_biomass,
        "history_elements": history_elements,
        "history_biomass_grid": history_biomass_grid,
        "N_SPP": N_SPP,
        "history_spp_biomass": history_spp_biomass,
        "history_spp_biomass_std": history_spp_biomass_std,
        "history_spp_biomass_grid": history_spp_grid,
        "history_spp_fitness": history_spp_fitness,
        "history_spp_dead_fitness_mean": history_spp_dead_fitness_mean,
        "history_spp_count": history_agents,
        "history_deficit": history_deficit,
        "soil_snapshot": soil_snapshot,
        "history_spp_age": history_spp_age,
        "history_spp_elemental_dissimilarity": history_spp_elemental_dissimilarity,
        "final_states": [r["final_state"] for r in all_runs],
        "completed_steps": completed_steps,
    }

    st.session_state["loaded_final_states"] = [r["final_state"] for r in all_runs]
    st.session_state["loaded_completed_steps"] = completed_steps
    st.session_state["pkl_default_name"] = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["run_count"] += 1
    st.session_state["ran"] = True


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state["ran"]:
    res = st.session_state["results"]

    RES_N_SPP = res.get("N_SPP", N_SPP)
    RES_SPP_LABELS = [f"Species {i+1}" for i in range(RES_N_SPP)]

    soil = st.session_state["soil_snapshot"]

    scalar_interval = res.get("scalar_interval", 20)
    snapshot_interval = res.get("snapshot_interval", 50)
    n_seeds_used = res.get("n_seeds_used", 1)
    steps_scalar = np.arange(len(res["history_biomass"])) * scalar_interval

    if n_seeds_used > 1:
        st.info(
            f"📊 Showing results averaged over **{n_seeds_used} seeds** "
            f"(seeds {st.session_state.get('SEED', '?')} – "
            f"{st.session_state.get('SEED', 0) + n_seeds_used - 1}). "
            f"Spatial maps show the average grid."
        )

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
        col2.metric("Peak Agent Count", f"{max(res['history_agents']):,}")
        col3.metric("Final Agent Count", f"{res['history_agents'][-1]:,}")

        st.subheader("Total Mean Biomass")
        st.plotly_chart(ep.plot_biomass(steps_scalar, res["history_biomass"]), use_container_width=True)

        st.subheader("Agent Count")
        st.plotly_chart(ep.plot_agents(steps_scalar, res["history_agents"]), use_container_width=True)

        elem_arr = res["history_elements"]
        if elem_arr.ndim == 2 and elem_arr.shape[1] > 0:
            st.subheader("Element Pools Over Time")
            st.plotly_chart(ep.plot_element_pools(steps_scalar, elem_arr), use_container_width=True)

        deficit_data = res.get("history_deficit", [])
        if any(len(d) > 0 for d in deficit_data):
            st.subheader("Unmet Nutrient Demand per Species")
            st.plotly_chart(ep.plot_nutrient_deficit(steps_scalar, deficit_data, RES_SPP_LABELS),
                            use_container_width=True)

        ed_vals = res.get("history_spp_elemental_dissimilarity", [])
        if len(ed_vals) > 0:
            st.subheader("SPP Mean Elemental Dissimilarity Over Time")
            st.plotly_chart(ep.plot_spp_elemental_dissimilarity(steps_scalar, ed_vals),
                            use_container_width=True)

    with tab_spp:
        st.subheader("Per-Species Mean Biomass" + (f" (avg over {n_seeds_used} seeds)" if n_seeds_used > 1 else ""))
        st.plotly_chart(ep.plot_species_biomass(steps_scalar, res["history_spp_biomass"], RES_SPP_LABELS),
                        use_container_width=True)

        if n_seeds_used > 1 and "history_spp_biomass_std" in res:
            st.plotly_chart(ep.plot_species_biomass_std(
                steps_scalar, res["history_spp_biomass"], res["history_spp_biomass_std"], RES_SPP_LABELS
            ), use_container_width=True)

        if any(len(f) > 0 for f in res["history_spp_fitness"]):
            st.subheader("Per-Species Mean Fitness")
            st.plotly_chart(ep.plot_species_fitness(steps_scalar, res["history_spp_fitness"], RES_SPP_LABELS),
                            use_container_width=True)

        dead = res.get("history_spp_dead_fitness_mean", [])
        if any(len(f) > 0 for f in dead):
            st.subheader("Mean Fitness at Time of Death")
            st.plotly_chart(ep.plot_dead_fitness(steps_scalar, dead, RES_SPP_LABELS), use_container_width=True)

        spp_age = res.get("history_spp_age", [])
        if any(len(a) > 0 for a in spp_age):
            st.subheader("Mean Agent Age per Species")
            st.plotly_chart(ep.plot_species_age(steps_scalar, spp_age, RES_SPP_LABELS), use_container_width=True)

    with tab_maps:
        @st.fragment
        def _maps_fragment():
            n_snaps = len(res["history_biomass_grid"])
            snap_i = st.slider(
                f"Snapshot (recorded every {snapshot_interval} steps)",
                0, n_snaps - 1, n_snaps - 1, format="Step %d",
                   )
            actual_step = snap_i * snapshot_interval
            st.plotly_chart(
                ep.plot_spatial_maps(
                    res["history_biomass_grid"],
                    res["history_spp_grid"],
                    snap_i, actual_step, RES_SPP_LABELS,
                ),
                use_container_width=True,
            )

        _maps_fragment()

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
        "⬇️ Download .pkl",
        data=buf,
        file_name=f"{clean_name}.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )