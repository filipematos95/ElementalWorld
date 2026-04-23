# backend/plotting.py
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

NUTRIENT_NAMES = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Oxygen (O)"]
ELEMENTS       = ["C", "N", "P", "K", "O"]
SPP_COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                  "#bcbd22", "#17becf"]


def plot_soil(soil: np.ndarray) -> go.Figure:
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=NUTRIENT_NAMES,
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

    fig = sp.make_subplots(rows=1, cols=n_pools, subplot_titles=pool_labels)
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


def plot_spatial_maps(history_biomass_grid: list,
                      history_spp_grid: list,
                      snap_i: int,
                      actual_step: int,
                      spp_labels: list,
                      mixed_grid: np.ndarray | None = None) -> go.Figure:
    n_spp = len(spp_labels)
    ncols = max(n_spp, 2)

    row1_titles = [f"Total Biomass — t={actual_step}",
                   f"Species Richness — t={actual_step}" if mixed_grid is not None else ""]
    row1_titles += [""] * (ncols - len(row1_titles))
    row2_titles = [f"{label} — t={actual_step}" for label in spp_labels]
    row2_titles += [""] * (ncols - len(row2_titles))

    fig = sp.make_subplots(
        rows=2,
        cols=ncols,
        subplot_titles=row1_titles + row2_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    # Row 1
    fig.add_trace(
        go.Heatmap(
            z=np.array(history_biomass_grid[snap_i]).tolist(),
            coloraxis="coloraxis1",
            name="Total Biomass",
        ),
        row=1, col=1
    )

    if mixed_grid is not None:
        fig.add_trace(
            go.Heatmap(
                z=np.array(mixed_grid).tolist(),
                coloraxis="coloraxis2",
                name="Species Richness",
            ),
            row=1, col=2
        )

    # Row 2: all species share one color scale
    for s_id, label in enumerate(spp_labels):
        fig.add_trace(
            go.Heatmap(
                z=np.array(history_spp_grid[s_id][snap_i]).tolist(),
                coloraxis="coloraxis3",
                name=label,
            ),
            row=2, col=s_id + 1
        )

    # Compute approximate subplot widths in paper coordinates
    # good enough for consistent placement
    left_margin = 0.0
    right_panel_end = 1.0
    col_w = 1.0 / ncols

    # centers of rows
    y_row1 = 0.79
    y_row2 = 0.21

    # bars just to the right of the relevant subplot blocks
    x_biomass  = col_w * 1 - 0.01
    x_richness = col_w * 2 - 0.01
    x_species  = 1.02

    fig.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=20, r=120, t=70, b=20),

        coloraxis1=dict(
            colorscale="YlGn",
            colorbar=dict(
                title="Biomass",
                x=x_biomass,
                y=y_row1,
                len=0.30,
                thickness=12,
            ),
        ),

        coloraxis2=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="Richness",
                x=x_richness,
                y=y_row1,
                len=0.30,
                thickness=12,
            ),
        ),

        coloraxis3=dict(
            colorscale="Hot",
            colorbar=dict(
                title="Species biomass",
                x=x_species,
                y=y_row2,
                len=0.36,
                thickness=14,
            ),
        ),
    )

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

    fig = sp.make_subplots(rows=1, cols=n_spp, subplot_titles=spp_labels,
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

def plot_species_biomass_std(steps: np.ndarray, history_spp_biomass: list,
                             history_spp_biomass_std: list,
                             spp_labels: list) -> go.Figure:
    fig = go.Figure()
    steps_list = steps.tolist()
    for s, label in enumerate(spp_labels):
        mu  = np.array(history_spp_biomass[s])
        std = np.array(history_spp_biomass_std[s])
        c   = SPP_COLORS[s % len(SPP_COLORS)]
        fig.add_trace(go.Scatter(
            x=steps_list + steps_list[::-1],
            y=np.concatenate([mu + std, (mu - std)[::-1]]).tolist(),
            fill="toself", fillcolor=c, opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=steps_list, y=mu.tolist(), name=label,
            line=dict(color=c, width=2),
        ))
    fig.update_layout(
        title="Per-Species Biomass ± 1 SD across seeds",
        xaxis_title="Step", yaxis_title="Mean Biomass",
        template="plotly_white", height=500,
    )
    return fig

def plot_species_age(steps: np.ndarray, history_spp_age: list,
                     spp_labels: list) -> go.Figure:
    fig = go.Figure()
    steps_list = steps.tolist()
    for s_id, (data, label) in enumerate(zip(history_spp_age, spp_labels)):
        fig.add_trace(go.Scatter(
            x=steps_list, y=list(data),
            mode="lines", name=label,
            line=dict(color=SPP_COLORS[s_id % len(SPP_COLORS)], width=2),
        ))
    fig.update_layout(
        title="Mean Agent Age per Species",
        xaxis_title="Step", yaxis_title="Mean Age (steps)",
        template="plotly_white", height=500,
    )
    return fig

def plot_spp_elemental_dissimilarity(steps, values):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode="lines",
        name="SPP elemental dissimilarity",
        line=dict(width=3, color="#8e44ad")
    ))
    fig.update_layout(
        title="SPP Mean Elemental Dissimilarity Over Time",
        xaxis_title="Step",
        yaxis_title="Mahalanobis-based ED (SPP means)",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_covariance_matrices(covs, labels=None, dim_labels=None):
    n = len(covs)
    labels = labels or [f"Species {i+1}" for i in range(n)]
    dim_labels = dim_labels or ["C", "N", "P", "K", "O"]

    fig = sp.make_subplots(
        rows=1,
        cols=n,
        subplot_titles=labels,
        horizontal_spacing=0.08
    )

    for i, cov in enumerate(covs, start=1):
        fig.add_trace(
            go.Heatmap(
                z=cov,
                x=dim_labels,
                y=dim_labels,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="cov") if i == n else None,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Cov: %{z:.5f}<extra></extra>",
            ),
            row=1,
            col=i
        )

    fig.update_layout(
        title="Covariance matrices",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(type="category", side="top")
    fig.update_yaxes(type="category", autorange="reversed")

    return fig

def plot_mahalanobis_contours(
        cov2d,
        mean=(0, 0),
        xlim=(-3, 3),
        ylim=(-3, 3),
        n=200,
        axis_labels=("X", "Y"),
):
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(x, y)

    pts = np.c_[xx.ravel(), yy.ravel()]
    inv = np.linalg.inv(cov2d)
    d2 = np.array([(p - mean) @ inv @ (p - mean) for p in pts]).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=d2,
        contours=dict(showlabels=True),
        colorscale="Viridis",
        line_width=2,
    ))

    fig.update_layout(
        title="Mahalanobis distance contours",
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        height=500,
    )
    return fig


def plot_mixed_cells(history_spp_grid: list,
                     snap_i: int,
                     actual_step: int,
                     spp_labels: list,
                     min_frac: float = 0.0) -> go.Figure:
    """
    Visualize cells that contain more than one species at snapshot snap_i.

    history_spp_grid: list of length N_spp, each entry is a list/array of
                      grids over snapshots: (n_snaps, H, W).
    min_frac: if > 0, only species contributing at least this fraction of
              total biomass in a cell are counted as 'present'.
    """
    n_spp = len(spp_labels)

    # stack per-species grids for this snapshot → (H, W, N_spp)
    grids = [np.array(history_spp_grid[s_id][snap_i]) for s_id in range(n_spp)]
    grid_spp = np.stack(grids, axis=-1)  # (H, W, N_spp)

    total = grid_spp.sum(axis=-1, keepdims=True)  # (H, W, 1)
    # guard against division by zero
    frac = np.divide(grid_spp, total, out=np.zeros_like(grid_spp), where=total > 0)

    if min_frac > 0.0:
        present = (frac >= min_frac) & (total > 0)
    else:
        present = (grid_spp > 0)

    richness = present.sum(axis=-1)       # (H, W) number of species per cell
    mixed_mask = (richness >= 2).astype(float)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=mixed_mask.tolist(),
        colorscale=[[0.0, "#f7f7f7"], [1.0, "#d62728"]],
        colorbar=dict(title="Mixed cell"),
        showscale=False,
    ))
    fig.update_layout(
        title=f"Cells with multiple species (t={actual_step})",
        template="plotly_white",
        height=450,
    )
    fig.update_yaxes(autorange="reversed")
    return fig