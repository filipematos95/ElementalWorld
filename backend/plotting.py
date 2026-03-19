# backend/plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

NUTRIENT_NAMES = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Oxygen (O)"]
ELEMENTS       = ["C", "N", "P", "K", "O"]
SPP_COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_soil(soil: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=NUTRIENT_NAMES,
                        horizontal_spacing=0.15, vertical_spacing=0.15)
    for idx, (row, col) in enumerate([(1,1), (1,2), (2,1), (2,2)]):
        fig.add_trace(
            go.Heatmap(
                z=soil[:, :, idx].tolist(),
                colorscale="Viridis",
                coloraxis=f"coloraxis{idx+1}",
                name=NUTRIENT_NAMES[idx],
            ),
            row=row, col=col
        )
    fig.update_layout(
        template="plotly_white",
        height=700,
        coloraxis1=dict(colorscale="Viridis", colorbar=dict(x=0.44, y=0.78, len=0.44, thickness=15)),
        coloraxis2=dict(colorscale="Viridis", colorbar=dict(x=1.00, y=0.78, len=0.44, thickness=15)),
        coloraxis3=dict(colorscale="Viridis", colorbar=dict(x=0.44, y=0.22, len=0.44, thickness=15)),
        coloraxis4=dict(colorscale="Viridis", colorbar=dict(x=1.00, y=0.22, len=0.44, thickness=15)),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_biomass(steps: np.ndarray, history_biomass: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps.tolist(), y=list(history_biomass),
        mode="lines", line=dict(color="green", width=2), name="Mean Biomass",
    ))
    fig.update_layout(xaxis_title="Step", yaxis_title="Mean Biomass",
                      template="plotly_white", height=400)
    return fig


def plot_agents(steps: np.ndarray, history_agents: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps.tolist(), y=list(history_agents),
        mode="lines", line=dict(color="steelblue", width=2), name="Agents",
    ))
    fig.update_layout(xaxis_title="Step", yaxis_title="Agents",
                      template="plotly_white", height=400)
    return fig


def plot_element_pools(steps: np.ndarray, elem_arr: np.ndarray) -> go.Figure:
    n_pools     = elem_arr.shape[1]
    pool_labels = ELEMENTS[:n_pools]
    steps_list  = steps.tolist()

    fig = make_subplots(rows=1, cols=n_pools, subplot_titles=pool_labels)
    for i, label in enumerate(pool_labels):
        fig.add_trace(
            go.Scatter(
                x=steps_list,
                y=elem_arr[:, i].tolist(),
                mode="lines",
                line=dict(width=2),
                name=label,
                showlegend=False,
            ),
            row=1, col=i + 1
        )
    fig.update_layout(template="plotly_white", height=400)
    for i in range(1, n_pools + 1):
        fig.update_xaxes(title_text="Step", row=1, col=i)
    return fig


def plot_species_biomass(steps: np.ndarray, history_spp_biomass: list,
                         spp_labels: list) -> go.Figure:
    fig = go.Figure()
    steps_list = steps.tolist()
    for s_id, (data, label) in enumerate(zip(history_spp_biomass, spp_labels)):
        fig.add_trace(go.Scatter(
            x=steps_list, y=list(data),
            mode="lines", name=label, line=dict(color=SPP_COLORS[s_id], width=2),
        ))
    fig.update_layout(xaxis_title="Step", yaxis_title="Mean Biomass",
                      template="plotly_white", height=500)
    return fig


def plot_species_fitness(steps: np.ndarray, history_spp_fitness: list,
                         spp_labels: list) -> go.Figure:
    fig = go.Figure()
    steps_list = steps.tolist()
    for s_id, (data, label) in enumerate(zip(history_spp_fitness, spp_labels)):
        fig.add_trace(go.Scatter(
            x=steps_list, y=list(data),
            mode="lines", name=label, line=dict(color=SPP_COLORS[s_id], width=2),
        ))
    fig.update_layout(xaxis_title="Step", yaxis_title="Mean Fitness",
                      template="plotly_white", height=500)
    return fig


def plot_spatial_maps(history_biomass_grid: list, history_spp_grid: list,
                      snap_i: int, actual_step: int, spp_labels: list) -> go.Figure:
    n_spp  = len(spp_labels)
    titles = [f"Total Biomass — t={actual_step}"] + \
             [f"{l} — t={actual_step}" for l in spp_labels]
    fig = make_subplots(rows=1, cols=n_spp + 1, subplot_titles=titles,
                        horizontal_spacing=0.06)

    fig.add_trace(
        go.Heatmap(
            z=np.array(history_biomass_grid[snap_i]).tolist(),
            colorscale="YlGn",
            coloraxis="coloraxis1",
            name="Total Biomass",
        ),
        row=1, col=1
    )
    for s_id, label in enumerate(spp_labels):
        fig.add_trace(
            go.Heatmap(
                z=np.array(history_spp_grid[s_id][snap_i]).tolist(),
                colorscale="Hot",
                coloraxis=f"coloraxis{s_id + 2}",
                reversescale=True,
                name=label,
            ),
            row=1, col=s_id + 2
        )

    coloraxis_settings = {
        f"coloraxis{i+1}": dict(
            colorscale="YlGn" if i == 0 else "Hot",
            colorbar=dict(
                x=round((i + 1) / (n_spp + 1) - 0.02, 2),
                len=0.9,
                thickness=12,
            )
        )
        for i in range(n_spp + 1)
    }

    fig.update_layout(height=450, template="plotly_white", **coloraxis_settings)
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_dead_fitness(steps: np.ndarray, history_spp_dead_fitness_mean: list,
                      spp_labels: list) -> go.Figure:
    fig = go.Figure()
    steps_list = steps.tolist()
    for s_id, (data, label) in enumerate(zip(history_spp_dead_fitness_mean, spp_labels)):
        fig.add_trace(go.Scatter(
            x=steps_list, y=list(data),
            mode="lines", name=label,
            line=dict(color=SPP_COLORS[s_id], width=2),
            connectgaps=False,
        ))
    fig.update_layout(
        title="Mean Fitness at Time of Death per Species",
        xaxis_title="Simulation Step",
        yaxis_title="Mean Fitness at Death",
        yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white",
        height=500,
    )
    return fig

def plot_nutrient_deficit(steps: np.ndarray, history_deficit: list,
                          spp_labels: list) -> go.Figure:
    """
    history_deficit: list of N_SPP lists, each of shape (T, 4).
    One subplot per species, one line per nutrient [N, P, K, O].
    """
    n_spp      = len(spp_labels)
    steps_list = steps.tolist()
    nutrient_labels = ["N", "P", "K", "O"]
    nutrient_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = make_subplots(rows=1, cols=n_spp, subplot_titles=spp_labels,
                        shared_yaxes=True)

    for s_id, (data, label) in enumerate(zip(history_deficit, spp_labels)):
        arr = np.array(data)          # shape (T, 4)
        for i, (nutrient, color) in enumerate(zip(nutrient_labels, nutrient_colors)):
            fig.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=arr[:, i].tolist(),
                    mode="lines",
                    name=nutrient,
                    line=dict(color=color, width=2),
                    legendgroup=nutrient,              # group same nutrients across subplots
                    showlegend=(s_id == 0),            # only show legend once
                ),
                row=1, col=s_id + 1
            )
        fig.update_xaxes(title_text="Step", row=1, col=s_id + 1)

    fig.update_yaxes(title_text="Unmet Demand", row=1, col=1)
    fig.update_layout(
        title="Unmet Nutrient Demand per Species",
        template="plotly_white",
        height=400,
    )
    return fig
