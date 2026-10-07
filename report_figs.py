"""report_figs.py - vector 3D-style views of the jacket for the PDF report (reportlab only, no extra packages).

Each view projects the member centre-lines to the page and strokes them with a width that follows the member
diameter, far members first.  Views: iso, side (from -y), end (from +x), plan.  Optional waterline, CoG and CoB.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors

import jacket_stability as js

KIND_COL = {"Buoyancy tank": "#2b7bba", "Leg / vertical": "#2f3a45", "Horizontal brace": "#7b8794", "Diagonal brace": "#9aa5b1"}
FLOODED = "#d62728"


def _basis(view: str):
    """(right, up, toward-camera) unit vectors in the space frame."""
    if view == "iso":
        az, el = math.radians(-50.0), math.radians(22.0)
        c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        r = np.array([-math.sin(az), math.cos(az), 0.0])
        u = np.cross(c, r)
        return r, u, c
    if view == "side":
        return np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
    if view == "end":
        return np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])      # plan


_TITLES = {"iso": "Isometric", "side": "Elevation (from -y)", "end": "End (from +x)", "plan": "Plan"}
_LEG_RE = re.compile(r"Leg x(\d+(?:\.\d+)?) ([ABC])\b")


def leg_labels(elements: Sequence[js.Element]) -> list:
    """[(label, x, y, z_top)] for members named 'Leg x<col> <A|B|C> ...' (grid names A1, A2, B1, B2, A''1, A''2)."""
    out = {}
    for e in elements:
        m = _LEG_RE.match(e.name)
        if not m:
            continue
        row = {"A": "A", "B": "B", "C": "A''"}[m.group(2)]
        key = f"{row}{1 if float(m.group(1)) < 3.5 else 2}"
        top = e.p1 if e.p1[2] >= e.p2[2] else e.p2
        if key not in out or top[2] > out[key][3]:
            out[key] = (key, top[0], top[1], top[2])
    return list(out.values())


def panel(elements: Sequence[js.Element], view: str, w: float, h: float, *, rot=None, zw: Optional[float] = None,
          g_pt=None, b_pt=None, labels: bool = False, title: str = "") -> Drawing:
    r_m = np.eye(3) if rot is None else np.asarray(rot, float)
    right, up, cam = _basis(view)
    pts = np.array([r_m @ np.asarray(p, float) for e in elements for p in (e.p1, e.p2)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if zw is not None:
        m = max(0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1])), 4.0)
        corners = np.array([[lo[0] - m, lo[1] - m, zw], [hi[0] + m, lo[1] - m, zw], [hi[0] + m, hi[1] + m, zw], [lo[0] - m, hi[1] + m, zw]])
        pts = np.vstack([pts, corners])
    sx, sy = pts @ right, pts @ up
    pad_top, pad_bot, pad = 16.0, 6.0, 8.0
    scale = min((w - 2 * pad) / max(float(np.ptp(sx)), 1e-6), (h - pad_top - pad_bot) / max(float(np.ptp(sy)), 1e-6))
    ox = pad + 0.5 * ((w - 2 * pad) - scale * float(np.ptp(sx))) - scale * float(sx.min())
    oy = pad_bot + 0.5 * ((h - pad_top - pad_bot) - scale * float(np.ptp(sy))) - scale * float(sy.min())

    def xy(p):
        p = r_m @ np.asarray(p, float) if p is not None else None
        return ox + scale * float(p @ right), oy + scale * float(p @ up), float(p @ cam)

    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, strokeColor=colors.HexColor("#c8cdd2"), strokeWidth=0.4, fillColor=None))
    d.add(String(5, h - 11, title or _TITLES[view], fontSize=7.5, fontName="Helvetica-Bold"))
    if zw is not None:
        wl = Group()
        if view in ("side", "end"):
            ys = [oy + scale * (zw * 1.0 - float(sy.min())) for _ in (0,)]
            y_w = oy + scale * (zw - float(sy.min()))
            wl.add(Rect(2, 2, w - 4, max(0.0, min(y_w - 2, h - 4)), fillColor=colors.Color(0.12, 0.56, 1.0, alpha=0.2),
                        strokeColor=None))
            wl.add(Line(2, y_w, w - 2, y_w, strokeColor=colors.HexColor("#1e90ff"), strokeWidth=0.8))
        elif view == "iso":
            cs = [xy(c) for c in corners]
            wl.add(Polygon([v for c in cs for v in c[:2]], fillColor=colors.Color(0.12, 0.56, 1.0, alpha=0.22),
                           strokeColor=colors.HexColor("#1e90ff"), strokeWidth=0.5))
        d.add(wl)
    segs = []
    for e in elements:
        a, b = xy(e.p1), xy(e.p2)
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 0.05:
            continue
        kind = "Buoyancy tank" if (e.buoyant and e.tank) else js.member_kind(e)
        col = FLOODED if (e.buoyant and e.flooded) else KIND_COL.get(kind, "#555555")
        width = max(0.3, min(5.0 if kind == "Buoyancy tank" else 3.2, e.d_out * scale))
        segs.append(((a[2] + b[2]) / 2.0, a, b, col, width))
    for _, a, b, col, width in sorted(segs, key=lambda s: s[0]):        # far first
        d.add(Line(a[0], a[1], b[0], b[1], strokeColor=colors.HexColor(col), strokeWidth=width, strokeLineCap=1))
    for pt, col, txt in ((g_pt, "#e63946", "G"), (b_pt, "#1f6fb4", "B")):
        if pt is not None:
            x, y, _ = xy(np.linalg.solve(r_m, np.asarray(pt, float)))        # pt is already in the space frame
            d.add(Line(x - 3, y, x + 3, y, strokeColor=colors.HexColor(col), strokeWidth=1.2))
            d.add(Line(x, y - 3, x, y + 3, strokeColor=colors.HexColor(col), strokeWidth=1.2))
            d.add(String(x + 4, y + 2, txt, fontSize=7, fillColor=colors.HexColor(col), fontName="Helvetica-Bold"))
    if labels and view in ("iso", "plan"):
        for name, x, y, z in leg_labels(elements):
            lx, ly, _ = xy((x, y, z))
            d.add(String(lx + 2, ly + (3 if view == "iso" else 2), name, fontSize=7.5, fillColor=colors.HexColor("#c00000"),
                         fontName="Helvetica-Bold"))
    return d


def figure_row(elements, *, rot=None, zw=None, g_pt=None, b_pt=None, labels=False, title_prefix="",
               width: float = 510.0, height: float = 250.0, views=("iso", "side", "plan")) -> Drawing:
    """Several panels side by side as one Drawing."""
    widths = {"iso": 0.40, "side": 0.30, "end": 0.30, "plan": 0.30}
    tot = sum(widths[v] for v in views)
    row = Drawing(width, height)
    x = 0.0
    for v in views:
        pw = width * widths[v] / tot
        p = panel(elements, v, pw - 4, height, rot=rot, zw=zw, g_pt=g_pt, b_pt=b_pt, labels=labels,
                  title=(f"{title_prefix}{_TITLES[v]}" if title_prefix else None))
        g = Group(p)
        g.translate(x, 0)
        row.add(g)
        x += pw
    return row
