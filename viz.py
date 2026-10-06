"""
3D drawing of the jacket for the Streamlit app (Plotly).

Members are drawn as solid shaded tubes (hollow annuli for tanks with a
through-leg), grouped by class, in the attitude you choose: as built, floating
level, or at the static equilibrium under wind + tow.  Pure presentation - it
never changes the calculation.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

import jacket_stability as js

# class -> (colour, opacity)
STYLE = {
    "Buoyancy tank": ("#2f6fb0", 0.55),
    "Leg / vertical": ("#7a8594", 1.0),
    "Horizontal brace": ("#a3acb8", 1.0),
    "Diagonal brace": ("#c0c7d1", 1.0),
}
FLOODED = ("#d62728", 0.65)
N_SIDES = 20


def _basis(ax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ax, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(ax, e1)


def tube(p1: np.ndarray, p2: np.ndarray, ro: float, ri: float = 0.0, n: int = N_SIDES):
    """Vertices (m,3) and triangles (k,3) for a cylinder / annulus between p1 and p2."""
    ax = p2 - p1
    ax = ax / np.linalg.norm(ax)
    e1, e2 = _basis(ax)
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    v = [p1 + ro * ring, p2 + ro * ring]            # 0:outer bottom, 1:outer top
    idx = np.arange(n)
    nxt = (idx + 1) % n
    tri = []

    def side(b0, b1, flip=False):
        for i, j in zip(idx, nxt):
            q = [(b0 + i, b0 + j, b1 + j), (b0 + i, b1 + j, b1 + i)]
            tri.extend([t[::-1] for t in q] if flip else q)

    side(0, n)
    if ri > 0.0:
        v += [p1 + ri * ring, p2 + ri * ring]       # 2:inner bottom, 3:inner top
        side(2 * n, 3 * n, flip=True)
        for i, j in zip(idx, nxt):                  # annular end caps
            tri += [(i, j, 2 * n + j), (i, 2 * n + j, 2 * n + i)]
            tri += [(n + i, n + j, 3 * n + j), (n + i, 3 * n + j, 3 * n + i)]
    else:
        v += [p1[None], p2[None]]                   # cap centres
        c1, c2 = 2 * n, 2 * n + 1
        for i, j in zip(idx, nxt):
            tri += [(c1, j, i), (c2, n + i, n + j)]
    return np.vstack(v), np.array(tri)


def _merge(parts: Sequence[tuple[np.ndarray, np.ndarray, str]]):
    xs, ii, jj, kk, txt = [], [], [], [], []
    off = 0
    for v, t, name in parts:
        xs.append(v)
        ii.append(t[:, 0] + off)
        jj.append(t[:, 1] + off)
        kk.append(t[:, 2] + off)
        txt += [name] * len(v)
        off += len(v)
    v = np.vstack(xs)
    return v, np.concatenate(ii), np.concatenate(jj), np.concatenate(kk), txt


def _arrow(fig, start, vec, colour, name, size=0.55, show=True):
    start, vec = np.asarray(start, float), np.asarray(vec, float)
    end = start + vec
    fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines",
                               line=dict(color=colour, width=7), name=name, legendgroup=name, hoverinfo="name",
                               showlegend=show))
    ln = float(np.linalg.norm(vec))
    d = vec / max(ln, 1e-9)
    fig.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[d[0]], v=[d[1]], w=[d[2]], anchor="tip",
                          sizemode="absolute", sizeref=ln * size, showscale=False, colorscale=[[0, colour], [1, colour]],
                          legendgroup=name, showlegend=False, hoverinfo="skip"))


def camera_buttons(k: float = 1.0) -> list:
    def cam(eye, up=(0, 0, 1)):
        return dict(eye=dict(x=eye[0] * k, y=eye[1] * k, z=eye[2] * k), up=dict(x=up[0], y=up[1], z=up[2]))
    views = [("Iso", cam((1.5, -1.6, 0.9))), ("Plan", cam((0.0, 0.001, 2.6), up=(0, 1, 0))),
             ("Side (from -y)", cam((0.0, -2.6, 0.25))), ("End (from +x)", cam((2.6, 0.0, 0.25)))]
    # left-aligned so they never sit under the Plotly toolbar (top right); neutral colours read in light and dark
    return [dict(type="buttons", direction="right", x=0.0, y=1.02, xanchor="left", yanchor="bottom",
                 pad=dict(l=0, r=0, t=0, b=0), bgcolor="rgba(127,127,127,0.22)",
                 bordercolor="rgba(127,127,127,0.6)", borderwidth=1, font=dict(size=11), showactive=False,
                 buttons=[dict(label=n, method="relayout", args=[{"scene.camera": c}]) for n, c in views])]


def jacket_figure(elements: Sequence[js.Element], weights: Sequence[js.WeightItem],
                  openings: Sequence[js.Opening], params: js.Params, *,
                  rot: Optional[np.ndarray] = None, zw: Optional[float] = None,
                  g_pt: Optional[Sequence[float]] = None, b_pt: Optional[Sequence[float]] = None,
                  wind_beta_deg: Optional[float] = None, show_triad: bool = False,
                  title: str = "") -> go.Figure:
    """Draw the jacket.  rot = space<-body rotation (None = as built); zw = waterline
    height in the space frame (None = no water drawn); g_pt/b_pt = CoG/CoB in space."""
    r_m = np.eye(3) if rot is None else np.asarray(rot, float)
    fig = go.Figure()

    groups: dict[str, list] = {}
    for e in elements:
        p1, p2 = r_m @ np.asarray(e.p1, float), r_m @ np.asarray(e.p2, float)
        if np.linalg.norm(p2 - p1) < 1e-9:
            continue
        key = "Flooded tank (damage case)" if (e.buoyant and e.flooded) else js.member_kind(e)
        v, t = tube(p1, p2, e.d_out / 2.0, e.d_in / 2.0)
        groups.setdefault(key, []).append((v, t, f"{e.name}  Ø{e.d_out:.2f} m"))
    order = ["Buoyancy tank", "Flooded tank (damage case)", "Leg / vertical", "Horizontal brace", "Diagonal brace"]
    for key in order:
        if key not in groups:
            continue
        col, op = FLOODED if key.startswith("Flooded") else STYLE[key]
        v, i, j, k, txt = _merge(groups[key])
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=i, j=j, k=k, color=col, opacity=op, name=key,
                                showlegend=True, text=txt, hoverinfo="text", flatshading=True,
                                lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.6)))

    allv = np.array([r_m @ np.asarray(p, float) for e in elements for p in (e.p1, e.p2)])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    span = float(max(hi - lo))

    if zw is not None:
        m = max(0.6 * float(max(hi[0] - lo[0], hi[1] - lo[1])), 6.0)   # sea surface sized to the plan footprint
        x0, x1, y0, y1 = lo[0] - m, hi[0] + m, lo[1] - m, hi[1] + m
        fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[zw] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                                color="#1e90ff", opacity=0.22, name="Waterline", showlegend=True, hoverinfo="skip"))

    gp = g_pt if g_pt is not None else (None if not weights else r_m @ _cog(weights))
    if gp is not None:
        fig.add_trace(go.Scatter3d(x=[gp[0]], y=[gp[1]], z=[gp[2]], mode="markers+text", text=["G"], textposition="top center",
                                   name="CoG (G)", marker=dict(size=4, color="#e63946", symbol="cross")))
    if b_pt is not None:
        fig.add_trace(go.Scatter3d(x=[b_pt[0]], y=[b_pt[1]], z=[b_pt[2]], mode="markers+text", text=["B"],
                                   textposition="bottom center", name="CoB (B)",
                                   marker=dict(size=4, color="#4c9be8", symbol="diamond")))
    lines = list(params.lines) if params.lines else [js.Line(
        "Tow", tuple(params.tow_point), params.tow_heading_deg, params.tow_elevation_deg, "tow")]
    shown = set()
    for ln in lines:
        tp = r_m @ np.asarray(ln.point, float)
        kind = "Tow line / leg" if ln.mode == "tow" else "Fixed pull (pull-in etc.)"
        col = "#9467bd" if ln.mode == "tow" else "#e07b00"
        fig.add_trace(go.Scatter3d(x=[tp[0]], y=[tp[1]], z=[tp[2]], mode="markers", name=f"{kind} - connection",
                                   legendgroup=kind, showlegend=kind not in shown, hovertext=[ln.name], hoverinfo="text",
                                   marker=dict(size=5, color=col, symbol="diamond")))
        tau, eps = math.radians(ln.heading_deg), math.radians(ln.elevation_deg)
        d = np.array([math.cos(eps) * math.cos(tau), math.cos(eps) * math.sin(tau), math.sin(eps)])
        _arrow(fig, tp, 0.18 * span * d, col, f"{kind} - force", size=0.55, show=kind not in shown)
        shown.add(kind)

    if openings:
        o = np.array([r_m @ np.array([q.x, q.y, q.z]) for q in openings])
        below = (o[:, 2] <= zw) if zw is not None else np.zeros(len(o), bool)
        fig.add_trace(go.Scatter3d(x=o[:, 0], y=o[:, 1], z=o[:, 2], mode="markers", name="Openings (red = under water)",
                                   hovertext=[q.name for q in openings], hoverinfo="text",
                                   marker=dict(size=4, color=["#d62728" if b else "#ff7f0e" for b in below])))

    if wind_beta_deg is not None:
        bt = math.radians(wind_beta_deg)
        d = np.array([math.cos(bt), math.sin(bt), 0.0])
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        z_a = (zw if zw is not None else lo[2]) + 0.75 * (hi[2] - (zw if zw is not None else lo[2]))
        start = np.array([cx, cy, z_a]) - d * (0.45 * span + 0.2 * span)
        _arrow(fig, start, 0.2 * span * d, "#2a9d8f", "Wind")

    if show_triad:
        o0 = r_m @ np.zeros(3)
        for ax_i, (col, nm) in enumerate((("#d62728", "x (fwd)"), ("#2ca02c", "y"), ("#1f77b4", "z (up)"))):
            v = np.zeros(3)
            v[ax_i] = 0.12 * span
            _arrow(fig, o0, r_m @ v, col, nm, size=0.4)

    # tall, slender structures need the camera pulled back so the whole height is in view
    kz = min(1.5, max(1.0, 0.25 * span / max(float(max(hi[0] - lo[0], hi[1] - lo[1])), 1.0)))
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   camera=dict(eye=dict(x=1.5 * kz, y=-1.6 * kz, z=0.9 * kz))),
        height=640, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.02),
        updatemenus=camera_buttons(kz), title=dict(text=title, x=0.01, font=dict(size=14)))
    return fig


def _cog(weights: Sequence[js.WeightItem]) -> np.ndarray:
    wt = sum(w.mass_t for w in weights)
    return np.array([sum(w.mass_t * getattr(w, a) for w in weights) / wt for a in ("x", "y", "z")])
