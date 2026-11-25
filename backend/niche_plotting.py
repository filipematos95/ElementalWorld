"""
backend/niche_plotting.py

Plots of the biogeochemical (elemental) niche of each species, one frame per
saved snapshot -- the same idea as the spatial maps, but in elementome space.

Concepts
--------
* Fundamental niche : the niche defined by the model parameters (center +
  tolerance covariance). Drawn as dashed/dotted ellipses at the Mahalanobis
  distance where fitness = 0 (d = sigma_threshold) and fitness = 0.5
  (d = sigma_threshold / sqrt(2)), because fitness = 1 - (d / sigma)^2.
* Realized niche    : where the living individuals actually are. Drawn as the
  cloud of (subsampled) individuals, their mean, and a 95 % covariance ellipse.

Everything lives in an 11-D elementome space, so each frame is shown through a
FIXED linear projection (a pair of elements, or the first two principal
components of the z-scored elementome pooled over all snapshots). Because the
projection and the axis ranges are fixed, frames can be compared and animated.
A projected ellipsoid {d' S^-1 d <= r^2} is the ellipse r^2 * A S A'.
"""
from dataclasses import dataclass
from math import ceil

import numpy as np

SPP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
CHI2_95_2DOF = 5.991464547107979  # 95 % quantile of chi-square with 2 dof


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
def merge_niche_runs(runs, max_per_species=400, seed=0):
    """
    Merge niche snapshots of several ensemble seeds into one list of snapshots.
    `runs` is a list (one per seed) of lists of snapshots (same steps).
    Individuals are pooled and re-subsampled; the covariance is recomputed
    from the pooled sample.
    """
    if len(runs) == 1:
        return runs[0]

    rng = np.random.default_rng(seed)
    merged = []
    for parts in zip(*runs):
        S, E = parts[0]["mean"].shape
        out = {
            "step": parts[0].get("step"),
            "n": np.sum([p["n"] for p in parts], axis=0),
            "mean": np.full((S, E), np.nan, dtype=np.float32),
            "cov": np.full((S, E, E), np.nan, dtype=np.float32),
            "samples": [], "fitness": [], "mass": [],
        }
        for s in range(S):
            X = np.concatenate([p["samples"][s] for p in parts], axis=0)
            f = np.concatenate([p["fitness"][s] for p in parts], axis=0)
            m = np.concatenate([p["mass"][s] for p in parts], axis=0)

            w = np.array([p["n"][s] for p in parts], dtype=np.float64)
            means = np.array([p["mean"][s] for p in parts], dtype=np.float64)
            ok = w > 0
            if ok.any():
                out["mean"][s] = (means[ok] * w[ok, None]).sum(axis=0) / w[ok].sum()

            if len(X) >= 2:
                out["cov"][s] = np.cov(X.astype(np.float64), rowvar=False)
            if len(X) > max_per_species:
                sel = rng.choice(len(X), max_per_species, replace=False)
                X, f, m = X[sel], f[sel], m[sel]

            out["samples"].append(X)
            out["fitness"].append(f)
            out["mass"].append(m)
        merged.append(out)
    return merged


@dataclass
class NicheProjection:
    """Fixed linear map  y = A (x - origin)  from elementome space to 2-D."""
    A: np.ndarray        # (2, E)
    origin: np.ndarray   # (E,)
    xlabel: str
    ylabel: str
    kind: str

    def point(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return (X - self.origin) @ self.A.T

    def cov(self, C):
        return self.A @ np.asarray(C, dtype=np.float64) @ self.A.T


def make_projection(niche, meta, view="pca", pair=None, max_pool=20000, seed=0):
    """
    view="pair": plot raw element fractions of the two elements in `pair`
                 (indices into meta["elements"]).
    view="pca" : first two principal components of the z-scored elementome,
                 pooled over every snapshot + the niche centers.
    """
    elements = list(meta["elements"])
    E = len(elements)

    if view == "pair":
        ix, iy = pair
        A = np.zeros((2, E))
        A[0, ix] = 1.0
        A[1, iy] = 1.0
        return NicheProjection(A, np.zeros(E), elements[ix], elements[iy], "pair")

    pool = [np.asarray(meta["centers"], dtype=np.float64)]
    for snap in niche:
        for X in snap["samples"]:
            if len(X):
                pool.append(np.asarray(X, dtype=np.float64))
    X = np.concatenate(pool, axis=0)
    if len(X) > max_pool:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_pool, replace=False)]

    m = X.mean(axis=0)
    std = X.std(axis=0)
    D = np.maximum(std, std.max() * 1e-3 + 1e-12)   # guard against ~constant elements
    Z = (X - m) / D
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:2]
    var = sv ** 2
    expl = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)

    def _lab(k):
        top = np.argsort(-np.abs(V[k]))[:2]
        return f"PC{k + 1} ({expl[k] * 100:.0f}% var; {', '.join(elements[i] for i in top)})"

    return NicheProjection(V / D[None, :], m, _lab(0), _lab(1), "pca")


# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────
def ellipse_xy(center, cov2, r, n=90):
    """Points of the ellipse {y : (y-c)' cov2^-1 (y-c) = r^2}; None if undefined."""
    center = np.asarray(center, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    if not (np.all(np.isfinite(center)) and np.all(np.isfinite(cov2))):
        return None
    cov2 = 0.5 * (cov2 + cov2.T)
    w, V = np.linalg.eigh(cov2)
    w = np.clip(w, 0.0, None)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    circ = np.stack([np.cos(th), np.sin(th)])
    pts = (V @ (np.sqrt(w)[:, None] * circ)) * r + center[:, None]
    return pts[0], pts[1]


def fundamental_geometry(meta, proj, s):
    """(center2d, outer ellipse [fitness 0], inner ellipse [fitness 0.5])."""
    c = proj.point(meta["centers"][s])[0]
    C2 = proj.cov(meta["covariances"][s])
    sig = float(meta["sigma"])
    return c, ellipse_xy(c, C2, sig), ellipse_xy(c, C2, sig / np.sqrt(2.0))


def realized_geometry(snap, proj, s):
    """(mean2d, 95 % ellipse) of the living individuals, or (None, None)."""
    mu, C = snap["mean"][s], snap["cov"][s]
    if not np.all(np.isfinite(mu)):
        return None, None
    c = proj.point(mu)[0]
    if not np.all(np.isfinite(C)):
        return c, None
    return c, ellipse_xy(c, proj.cov(C), np.sqrt(CHI2_95_2DOF))


def compute_axis_ranges(niche, meta, proj, fit="fundamental", pad=0.08):
    """
    Fixed (xrange, yrange) valid for ALL snapshots, so frames are comparable.
    fit="fundamental": niche centers + the whole fundamental niche (fitness-0 ellipse)
    fit="realized"   : niche centers + individuals + realized ellipses
                       (the fundamental ellipse is usually far outside this view)
    fit="both"       : everything
    """
    use_fund = fit in ("fundamental", "both")
    use_real = fit in ("realized", "both")
    xs, ys = [], []
    S = len(meta["centers"])
    for s in range(S):
        c, e0, _ = fundamental_geometry(meta, proj, s)
        xs.append(np.array([c[0]]))
        ys.append(np.array([c[1]]))
        if use_fund and e0 is not None:
            xs.append(e0[0])
            ys.append(e0[1])
    if use_real:
        for snap in niche:
            for X in snap["samples"]:
                if len(X):
                    P = proj.point(X)
                    xs.append(P[:, 0])
                    ys.append(P[:, 1])
            for s in range(S):
                _, re = realized_geometry(snap, proj, s)
                if re is not None:
                    xs.append(re[0])
                    ys.append(re[1])

    def _rng(v):
        v = np.concatenate(v)
        v = v[np.isfinite(v)]
        lo, hi = float(v.min()), float(v.max())
        w = hi - lo
        if w <= 0:
            w = max(abs(lo), 1e-6) * 0.1
        return [lo - pad * w, hi + pad * w]

    return _rng(xs), _rng(ys)


def niche_distance_series(niche, meta):
    """
    (n_snapshots, S) array: Mahalanobis distance of each species' mean realized
    composition to its niche center, divided by sigma_threshold
    (1 = the fitness-0 boundary of the fundamental niche).
    """
    S = len(meta["centers"])
    sig = float(meta["sigma"])
    inv = [np.linalg.inv(np.asarray(meta["covariances"][s], dtype=np.float64)) for s in range(S)]
    out = np.full((len(niche), S), np.nan)
    for i, snap in enumerate(niche):
        for s in range(S):
            mu = snap["mean"][s]
            if np.all(np.isfinite(mu)):
                d = np.asarray(mu, dtype=np.float64) - np.asarray(meta["centers"][s], dtype=np.float64)
                out[i, s] = np.sqrt(max(d @ inv[s] @ d, 0.0)) / sig
    return out


# ─────────────────────────────────────────────
# PLOTLY (interactive, used in the Streamlit tab)
# ─────────────────────────────────────────────
def plot_niche_snapshot(snap, meta, proj, labels, per_species=False,
                        show_points=True, show_real=True, show_fund=True,
                        axis_range=None, colors=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    S = len(labels)
    colors = colors or SPP_COLORS
    step = snap.get("step", "?")

    if per_species:
        ncols = 2 if S > 1 else 1
        nrows = ceil(S / ncols)
        titles = [f"{labels[s]} · n={int(snap['n'][s])}" for s in range(S)]
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                            horizontal_spacing=0.07, vertical_spacing=0.12 if nrows > 1 else 0.1)
    else:
        nrows = ncols = 1
        fig = go.Figure()

    def _add(trace, s):
        if per_species:
            fig.add_trace(trace, row=s // ncols + 1, col=s % ncols + 1)
        else:
            fig.add_trace(trace)

    for s in range(S):
        col = colors[s % len(colors)]
        name = f"{labels[s]} (n={int(snap['n'][s])})"
        grp = labels[s]

        X = snap["samples"][s]
        if show_points and len(X):
            P = proj.point(X)
            _add(go.Scatter(
                x=P[:, 0], y=P[:, 1], mode="markers", name=name, legendgroup=grp,
                showlegend=not per_species,
                marker=dict(size=5, color=col, opacity=0.35),
                customdata=np.stack([snap["fitness"][s], snap["mass"][s]], axis=1),
                hovertemplate=(f"{labels[s]}<br>fitness=%{{customdata[0]:.2f}}"
                               "<br>mass=%{customdata[1]:.3f}<extra></extra>"),
            ), s)
        elif not per_species:
            # keep an (empty) legend entry so extinct species stay visible
            _add(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                            legendgroup=grp, marker=dict(size=5, color=col)), s)

        c, e0, e5 = fundamental_geometry(meta, proj, s)
        if show_fund:
            if e0 is not None:
                _add(go.Scatter(x=e0[0], y=e0[1], mode="lines", legendgroup=grp, showlegend=False,
                                line=dict(color=col, width=1.5, dash="dash"),
                                hoverinfo="skip"), s)
            if e5 is not None:
                _add(go.Scatter(x=e5[0], y=e5[1], mode="lines", legendgroup=grp, showlegend=False,
                                line=dict(color=col, width=1.2, dash="dot"),
                                hoverinfo="skip"), s)
        _add(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", legendgroup=grp, showlegend=False,
                        marker=dict(symbol="star", size=15, color=col,
                                    line=dict(color="black", width=1)),
                        hovertemplate=f"{labels[s]} niche center<extra></extra>"), s)

        if show_real:
            mu2, re = realized_geometry(snap, proj, s)
            if re is not None:
                _add(go.Scatter(x=re[0], y=re[1], mode="lines", legendgroup=grp, showlegend=False,
                                line=dict(color=col, width=2.5), hoverinfo="skip"), s)
            if mu2 is not None:
                _add(go.Scatter(x=[mu2[0]], y=[mu2[1]], mode="markers", legendgroup=grp,
                                showlegend=False,
                                marker=dict(symbol="diamond", size=10, color=col,
                                            line=dict(color="black", width=1)),
                                hovertemplate=f"{labels[s]} mean composition<extra></extra>"), s)

    fig.update_layout(
        title=f"Biogeochemical niche — step {step}",
        height=560 if not per_species else 380 * nrows + 60,
        legend=dict(orientation="v"),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    if axis_range is not None:
        fig.update_xaxes(range=axis_range[0])
        fig.update_yaxes(range=axis_range[1])
    if per_species:
        for c_ in range(1, ncols + 1):
            fig.update_xaxes(title_text=proj.xlabel, row=nrows, col=c_)
        for r_ in range(1, nrows + 1):
            fig.update_yaxes(title_text=proj.ylabel, row=r_, col=1)
    else:
        fig.update_xaxes(title_text=proj.xlabel)
        fig.update_yaxes(title_text=proj.ylabel)
    return fig


def plot_niche_trajectory(niche, meta, proj, labels, colors=None):
    """Path of each species' mean realized composition across snapshots."""
    import plotly.graph_objects as go

    colors = colors or SPP_COLORS
    fig = go.Figure()
    steps = [snap.get("step", i) for i, snap in enumerate(niche)]
    for s in range(len(labels)):
        col = colors[s % len(colors)]
        pts = np.full((len(niche), 2), np.nan)
        for i, snap in enumerate(niche):
            mu = snap["mean"][s]
            if np.all(np.isfinite(mu)):
                pts[i] = proj.point(mu)[0]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name=labels[s], legendgroup=labels[s],
            line=dict(color=col, width=1.5), marker=dict(size=6, color=col),
            text=[f"step {t}" for t in steps],
            hovertemplate=f"{labels[s]}<br>%{{text}}<extra></extra>",
        ))
        ok = np.where(np.all(np.isfinite(pts), axis=1))[0]
        if len(ok):
            fig.add_trace(go.Scatter(
                x=[pts[ok[0], 0]], y=[pts[ok[0], 1]], mode="markers", showlegend=False,
                legendgroup=labels[s], hovertemplate=f"{labels[s]} start<extra></extra>",
                marker=dict(symbol="circle-open", size=14, color=col, line=dict(width=2))))
            fig.add_trace(go.Scatter(
                x=[pts[ok[-1], 0]], y=[pts[ok[-1], 1]], mode="markers", showlegend=False,
                legendgroup=labels[s], hovertemplate=f"{labels[s]} end<extra></extra>",
                marker=dict(symbol="square", size=11, color=col, line=dict(color="black", width=1))))
        c = proj.point(meta["centers"][s])[0]
        fig.add_trace(go.Scatter(
            x=[c[0]], y=[c[1]], mode="markers", showlegend=False, legendgroup=labels[s],
            hovertemplate=f"{labels[s]} niche center<extra></extra>",
            marker=dict(symbol="star", size=15, color=col, line=dict(color="black", width=1))))
    fig.update_layout(
        title="Mean realized composition through time (○ start, ■ end, ★ niche center)",
        height=460, margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title=proj.xlabel, yaxis_title=proj.ylabel,
    )
    return fig


def plot_niche_distance(niche, meta, labels, colors=None):
    """Distance of mean realized composition to the niche center, in sigma units."""
    import plotly.graph_objects as go

    colors = colors or SPP_COLORS
    steps = [snap.get("step", i) for i, snap in enumerate(niche)]
    dist = niche_distance_series(niche, meta)
    fig = go.Figure()
    for s in range(len(labels)):
        fig.add_trace(go.Scatter(
            x=steps, y=dist[:, s], mode="lines+markers", name=labels[s],
            line=dict(color=colors[s % len(colors)], width=1.8), marker=dict(size=5)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
                  annotation_text="fitness = 0 boundary", annotation_position="top left")
    fig.update_layout(
        title="Offset of mean composition from niche center",
        xaxis_title="Step", yaxis_title="Mahalanobis distance / sigma_threshold",
        height=460, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# MATPLOTLIB -> GIF (one frame per snapshot, same idea as the map GIFs)
# ─────────────────────────────────────────────
def _draw_species_mpl(ax, snap, meta, proj, s, color, show_points, show_real, show_fund):
    X = snap["samples"][s]
    if show_points and len(X):
        P = proj.point(X)
        ax.scatter(P[:, 0], P[:, 1], s=6, color=color, alpha=0.35, linewidths=0)
    c, e0, e5 = fundamental_geometry(meta, proj, s)
    if show_fund:
        if e0 is not None:
            ax.plot(e0[0], e0[1], ls="--", lw=1.2, color=color)
        if e5 is not None:
            ax.plot(e5[0], e5[1], ls=":", lw=1.0, color=color)
    ax.plot(c[0], c[1], ls="none", marker="*", ms=13, color=color, mec="black", mew=0.8)
    if show_real:
        mu2, re = realized_geometry(snap, proj, s)
        if re is not None:
            ax.plot(re[0], re[1], ls="-", lw=2.0, color=color)
        if mu2 is not None:
            ax.plot(mu2[0], mu2[1], ls="none", marker="D", ms=7, color=color, mec="black", mew=0.8)


def make_niche_gif(niche, meta, proj, labels, per_species=False, show_points=True,
                   show_real=True, show_fund=True, fps=2, axis_range=None,
                   colors=None, dpi=80):
    """Render one frame per snapshot and return the animated GIF as bytes."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from PIL import Image

    S = len(labels)
    colors = colors or SPP_COLORS
    if axis_range is None:
        axis_range = compute_axis_ranges(niche, meta, proj, fit="realized")

    frames = []
    for snap in niche:
        step = snap.get("step", "?")
        if per_species:
            ncols = 2 if S > 1 else 1
            nrows = ceil(S / ncols)
            fig = Figure(figsize=(4.4 * ncols, 3.7 * nrows), dpi=dpi)
            axes = [fig.add_subplot(nrows, ncols, s + 1) for s in range(S)]
            for s, ax in enumerate(axes):
                _draw_species_mpl(ax, snap, meta, proj, s, colors[s % len(colors)],
                                  show_points, show_real, show_fund)
                ax.set_title(f"{labels[s]} · n={int(snap['n'][s])}", fontsize=9)
        else:
            fig = Figure(figsize=(6.6, 5.4), dpi=dpi)
            ax0 = fig.add_subplot(1, 1, 1)
            axes = [ax0]
            handles = []
            for s in range(S):
                col = colors[s % len(colors)]
                _draw_species_mpl(ax0, snap, meta, proj, s, col, show_points, show_real, show_fund)
                handles.append(Line2D([0], [0], marker="o", ls="none", color=col,
                                      label=f"{labels[s]} (n={int(snap['n'][s])})"))
            ax0.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)

        for ax in axes:
            ax.set_xlim(axis_range[0])
            ax.set_ylim(axis_range[1])
            ax.set_xlabel(proj.xlabel, fontsize=7)
            ax.set_ylabel(proj.ylabel, fontsize=7)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Biogeochemical niche — step {step}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[..., :3].copy()))

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)
    return buf.getvalue()
