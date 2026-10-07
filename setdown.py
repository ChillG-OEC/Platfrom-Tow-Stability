"""Lowering onto the seabed: ballast, waterline and GM at chosen seabed clearances across the tide.

The structure floats upright.  For each stage (seabed clearance) and tide level the waterline that gives that
clearance is fixed by geometry; the displaced mass at that waterline minus the structure weight is the ballast
that has to be on board (negative = weight would have to be shed).  GM is evaluated with the ballast added as a
point mass.  Free surface of the ballast water is ignored.  Level waterline, screening only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import jacket_stability as js

U0 = np.array([0.0, 0.0, 1.0])


def _z_base(model: js.JacketModel) -> float:
    lo = np.inf
    for e in model.elements:
        p1, p2 = np.asarray(e.p1, float), np.asarray(e.p2, float)
        ax = (p2 - p1) / max(float(np.linalg.norm(p2 - p1)), 1e-12)
        ext = (e.d_out / 2) * np.sqrt(np.maximum(1.0 - ax * ax, 0.0))
        lo = min(lo, float(np.minimum(p1, p2)[2] - ext[2]))
    return lo


def _displaced_t(model: js.JacketModel, zw: float) -> float:
    return model.p.rho_w * js.cut(model.ea, U0, zw)[0]


def stage(els, wts, ops, params, depth_m: float, tide_m: float, clearance_m: float,
          ballast_z: float, *, with_gm: bool = True) -> dict:
    """One stage at one tide level."""
    m0 = js.JacketModel(els, wts, ops, params)
    zb = _z_base(m0)
    zw = zb + depth_m + tide_m - clearance_m
    ballast = _displaced_t(m0, zw) - m0.W
    full = bool(js.cut(m0.ea, U0, zw)[0] >= m0.ea.vol_total * (1.0 - 1e-6))   # every sealed member under water
    out = dict(saturated=full, clearance_m=clearance_m, tide_m=tide_m, depth_total_m=depth_m + tide_m, zw=zw,
               draft_m=zw - zb, ballast_t=ballast, weight_t=m0.W + ballast, gm_min_m=None, trim_deg=None)
    if full:
        return out                      # all buoyancy used: no ballast holds it here, it sinks to the seabed
    if with_gm and abs(ballast) > 1e-9 and m0.W + ballast > 0:
        w2 = list(wts) + [js.WeightItem("Ballast", ballast, float(m0.G[0]), float(m0.G[1]), float(ballast_z))]
        try:
            m1 = js.JacketModel(els, w2, ops, params)
            out["gm_min_m"] = float(min(m1.gm(0.0), m1.gm(90.0)))
            st = m1.solve_trim(0.0, 0.0, loads=False)
            out["trim_deg"] = None if st is None else float(np.degrees(st["theta"]))
        except ValueError:
            pass
    elif with_gm:
        m1 = m0
        out["gm_min_m"] = float(min(m1.gm(0.0), m1.gm(90.0)))
    return out


def default_ballast_z(els) -> float:
    """Body-frame height of the middle of the tank members (where ballast water would sit); else the mid-height."""
    zs = [0.5 * (e.p1[2] + e.p2[2]) for e in els if e.buoyant and e.tank]
    if not zs:
        zs = [0.5 * (e.p1[2] + e.p2[2]) for e in els if e.buoyant]
    return float(np.mean(zs)) if zs else 0.0


def sequence(els, wts, ops, params, depth_m: float, tides: dict, clearances: Sequence[float],
             design_tide: str, ballast_z: float, gm_min: float = 0.0) -> dict:
    """Ballast and GM for every stage and tide, step ballast between stages, and the clearance across the tide
    for the ballast fixed at the design tide."""
    rows = []
    for c in clearances:
        for name, t in tides.items():
            r = stage(els, wts, ops, params, depth_m, t, c, ballast_z)
            r.update(tide=name, stage=c)
            rows.append(r)
    td = tides[design_tide]
    base = {r["stage"]: r for r in rows if r["tide"] == design_tide}
    steps = []
    prev = None
    for c in clearances:
        b = base[c]["ballast_t"]
        steps.append(dict(stage=c, ballast_t=b, added_from_previous_t=None if prev is None else b - prev))
        prev = b
    across = []
    for c in clearances:
        for name, t in tides.items():
            across.append(dict(stage=c, tide=name, clearance_m=c + (t - td),
                               extra_ballast_to_hold_t=next(r["ballast_t"] for r in rows
                                                            if r["stage"] == c and r["tide"] == name) - base[c]["ballast_t"]))
    return dict(rows=rows, steps=steps, across=across, gm_min=gm_min)


def curve(els, wts, ops, params, depth_m: float, tides: dict, c_max: float, c_min: float = 0.0, n: int = 25) -> list:
    """Ballast against clearance for each tide (no GM, cheap)."""
    m0 = js.JacketModel(els, wts, ops, params)
    zb = _z_base(m0)
    out = []
    for name, t in tides.items():
        for c in np.linspace(c_max, c_min, n):
            out.append(dict(tide=name, clearance_m=float(c),
                            ballast_t=_displaced_t(m0, zb + depth_m + t - float(c)) - m0.W))
    return out


def stage_advice(res: dict, names: dict, design_tide: str, gm_min: float = 0.0) -> list:
    """Plain-language verdict per stage from a sequence result: what to do, and when not to go on to the next stage.
    Returns [(stage clearance, ok, text)]."""
    out = []
    stages = [st["stage"] for st in res["steps"]]
    for i, c in enumerate(stages):
        rows = [r for r in res["rows"] if r["stage"] == c]
        problems, fixes = [], []
        for r in rows:
            nm = f"{names.get(c, c)} at {r['tide']}"
            if r["saturated"] and c > 0:
                problems.append(f"{nm}: every sealed member is under water, so buoyancy cannot hold this clearance")
                fixes.append("restore or add buoyancy, or accept that it settles to the seabed")
            elif r["ballast_t"] < 0 and not r["saturated"]:
                problems.append(f"{nm}: the structure is too light to sit this deep; {-r['ballast_t']:.0f} t would have to be shed")
                fixes.append(f"deballast or remove at least {-r['ballast_t']:.0f} t")
            if r["gm_min_m"] is not None and gm_min > 0 and r["gm_min_m"] < gm_min:
                problems.append(f"{nm}: GM {r['gm_min_m']:.2f} m is below {gm_min:.2f} m")
                fixes.append("lower the ballast, press up slack tanks or restore a buoyancy module")
        for a in [a for a in res["across"] if a["stage"] == c]:
            if c > 0 and a["clearance_m"] <= 0:
                problems.append(f"{names.get(c, c)}: with the ballast set at {design_tide}, the structure grounds at {a['tide']}")
                fixes.append("wait for a higher tide" + (f", or deballast {abs(a['extra_ballast_to_hold_t']):.0f} t before it falls"
                                                         if a["extra_ballast_to_hold_t"] < 0 else ""))
        if problems:
            nxt = f"Do not proceed to {names.get(stages[i + 1], stages[i + 1])}" if i + 1 < len(stages) else "Do not proceed"
            out.append((c, False, f"{nxt}. " + "; ".join(dict.fromkeys(problems)) + ". To recover: " + "; ".join(dict.fromkeys(fixes)) + "."))
        else:
            out.append((c, True, f"{names.get(c, c)}: no problem found at any tide."))
    return out
