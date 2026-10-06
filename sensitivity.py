"""
Sensitivity studies for the jacket wet tow: how stability changes with
  (a) the number of buoyancy tanks attached, and
  (b) where the tow line connects to the jacket.

Pure Python on top of jacket_stability.py (no Streamlit).  Each case runs the full
heel sweep over the wind headings; a case can also end as "sinks" (not enough
buoyancy) or "capsizes" (no stable floating attitude with no wind or tow).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Callable, Optional, Sequence

import numpy as np

import jacket_stability as js

OK, SINKS, CAPSIZES = "ok", "sinks", "capsizes"
LIST_TOL_DEG = 0.5   # a no-wind list above this makes the from-upright sweep unreliable


def detach(elements: Sequence[js.Element], names: Sequence[str]) -> list[js.Element]:
    """Copy of the elements with the named members removed from the calculation:
    no buoyancy, no wind area and no drag (the tank is not attached)."""
    gone = set(names)
    return [dataclasses.replace(e, buoyant=False, exposed=False) if e.name in gone else e for e in elements]


def tank_names(elements: Sequence[js.Element]) -> list[str]:
    return [e.name for e in elements if e.buoyant]


def run_case(elements, weights, openings, params: js.Params, crit: js.Criteria,
             betas: Sequence[float], grid: Sequence[float], damaged: bool = False) -> dict:
    """One configuration: outcome status plus the headline stability numbers."""
    out = dict(status=OK, note="", reserve_pct=None, draft=None, tilt=None, gm_min=None, heel_max=None,
               ratio_min=None, gov_beta=None, tow_kn=None, passed=False, listed=False, res=None)
    try:
        model = js.JacketModel(elements, weights, openings, params, damaged=damaged)
        up = model.upright()
    except ValueError as exc:
        out.update(status=SINKS, note=str(exc))
        return out
    out.update(reserve_pct=up["reserve_buoyancy_pct"], draft=up["draft"])
    tilt = model.free_tilt()
    if tilt is None:
        out.update(status=CAPSIZES, note="No stable floating attitude with no wind or tow (within 60° tilt).")
        return out
    res = js.run_study(model, betas, grid, crit)
    s = res["summary"]
    listed = tilt["tilt_deg"] > LIST_TOL_DEG
    out.update(tilt=tilt["tilt_deg"], gm_min=s["gm_min"], heel_max=s["heel_max"], ratio_min=s["ratio_min"],
               gov_beta=s["governing_beta"], tow_kn=s["tow_kn"], listed=listed,
               passed=bool(s["passed"] and not listed), res=res)
    if listed:
        out["note"] = (f"Static list {tilt['tilt_deg']:.1f}° with no wind or tow. The heel sweep is measured from upright, "
                       "so heel and area-ratio results are not shown for this case; it needs a full analysis about "
                       "the listed attitude.")
    return out


def _severity(c: dict) -> tuple:
    """Higher = worse.  Used to pick the worst / best combination."""
    if c["status"] == SINKS:
        return (3, 0.0)
    if c["status"] == CAPSIZES:
        return (2, 0.0)
    r = c["ratio_min"] if c["ratio_min"] is not None else -1.0
    return (1 if not c["passed"] else 0, -r)


def tank_study(elements, weights, openings, params, crit, betas, grid, damaged: bool = False,
               max_combos: int = 12, max_lost: int = 3,
               progress: Optional[Callable[[float, str], None]] = None) -> list[dict]:
    """For k = N, N-1, ... tanks attached (down to max_lost tanks lost): run every
    combination of which tanks are lost (an even sample of max_combos if there are
    more) and keep the outcome counts, the worst and the best combination."""
    names = tank_names(elements)
    n = len(names)
    jobs = []
    for k in range(n, max(n - max_lost, 1) - 1, -1):
        combos = list(itertools.combinations(names, n - k))
        if len(combos) > max_combos:
            idx = np.unique(np.linspace(0, len(combos) - 1, max_combos).round().astype(int))
            combos = [combos[i] for i in idx]
        jobs.append((k, combos))
    total = sum(len(c) for _, c in jobs)
    done = 0
    rows = []
    all_sink = False
    for k, combos in jobs:
        cases = []
        if all_sink:
            rows.append(dict(k=k, n=n, combos=len(combos), counts={OK: 0, CAPSIZES: 0, SINKS: len(combos)},
                             worst=None, best=None, skipped=True))
            continue
        for lost in combos:
            c = run_case(detach(elements, lost), weights, openings, params, crit, betas, grid, damaged)
            c["lost"] = tuple(lost)
            cases.append(c)
            done += 1
            if progress:
                progress(done / total, f"{k} of {n} tanks attached")
        cnt = {s: sum(c["status"] == s for c in cases) for s in (OK, CAPSIZES, SINKS)}
        ok = [c for c in cases if c["status"] == OK]
        worst = max(cases, key=_severity)
        best = min(cases, key=_severity)
        rows.append(dict(k=k, n=n, combos=len(combos), counts=cnt, worst=worst, best=best,
                         worst_ok=(max(ok, key=_severity) if ok else None), skipped=False))
        if cnt[SINKS] == len(cases):
            all_sink = True
    return rows


def primary_line(params: js.Params) -> js.Line:
    """The tow line: the first entry of params.lines, or the legacy tow_* fields."""
    if params.lines:
        return params.lines[0]
    return js.Line("Tow", tuple(params.tow_point), params.tow_heading_deg, params.tow_elevation_deg,
                   "tow" if params.tow_mode == "auto" else "fixed", params.tow_manual_kn, 1.0)


def with_primary(params: js.Params, case: dict) -> js.Params:
    """Params with the tow line moved to the case's point, heading and elevation
    (missing heading / elevation keep the current values).  Other lines are unchanged."""
    ln = primary_line(params)
    ln = dataclasses.replace(
        ln, point=(float(case["x"]), float(case["y"]), float(case["z"])),
        heading_deg=float(case.get("heading_deg", ln.heading_deg)),
        elevation_deg=float(case.get("elevation_deg", ln.elevation_deg)))
    rest = tuple(params.lines[1:]) if params.lines else ()
    return dataclasses.replace(params, lines=(ln,) + rest)


def default_tow_cases(elements: Sequence[js.Element], params: js.Params) -> list[dict]:
    """Tow connection heights to compare: the current point, then the tank bottom,
    tank mid-height, tank top and jacket mid-height (same x, y and direction)."""
    ln0 = primary_line(params)
    x0, y0, z0 = ln0.point
    tz = [z for e in elements if e.buoyant for z in (e.p1[2], e.p2[2])]
    az = [z for e in elements for z in (e.p1[2], e.p2[2])]
    if not tz:
        tz = az
    z_lo, z_hi, z_top = min(tz), max(tz), max(az)
    rows = [("Current tow point", z0), ("Low - tank bottom", z_lo), ("Tank mid-height", 0.5 * (z_lo + z_hi)),
            ("Tank top", z_hi), ("High - jacket mid-height", 0.5 * (z_lo + z_top))]
    seen, out = [], []
    for name, z in rows:
        if any(abs(z - s) < 1e-6 for s in seen) and name != "Current tow point":
            continue
        seen.append(z)
        out.append(dict(name=name, x=float(x0), y=float(y0), z=float(z),
                        heading_deg=float(ln0.heading_deg), elevation_deg=float(ln0.elevation_deg)))
    return out


def tow_study(elements, weights, openings, params, crit, betas, grid, tow_cases: Sequence[dict],
              damaged: bool = False, progress: Optional[Callable[[float, str], None]] = None) -> list[dict]:
    out = []
    for i, t in enumerate(tow_cases):
        c = run_case(elements, weights, openings, with_primary(params, t), crit, betas, grid, damaged)
        c["tow"] = dict(t)
        out.append(c)
        if progress:
            progress((i + 1) / len(tow_cases), f"Tow connection: {t['name']}")
    return out


def matrix_study(configs: Sequence[tuple[str, Sequence[js.Element]]], weights, openings, params, crit, betas, grid,
                 tow_cases: Sequence[dict], damaged: bool = False,
                 progress: Optional[Callable[[float, str], None]] = None) -> list[list[dict]]:
    """Rows = tank configurations, columns = tow connections."""
    total = len(configs) * len(tow_cases)
    done = 0
    grid_out = []
    for label, els in configs:
        row = []
        for t in tow_cases:
            c = run_case(els, weights, openings, with_primary(params, t), crit, betas, grid, damaged)
            row.append(c)
            done += 1
            if progress:
                progress(done / total, f"{label} / {t['name']}")
        grid_out.append(row)
    return grid_out


def line_count_study(elements, weights, openings, params: js.Params, crit, betas, grid, damaged: bool = False,
                     progress: Optional[Callable[[float, str], None]] = None) -> list[dict]:
    """Add the lines one at a time in table order (tow line first): 1 line, 2 lines, ...
    Shows what each extra pull-in line or tow leg does to stability."""
    lines = list(params.lines) if params.lines else [primary_line(params)]
    out = []
    for k in range(1, len(lines) + 1):
        p = dataclasses.replace(params, lines=tuple(lines[:k]))
        c = run_case(elements, weights, openings, p, crit, betas, grid, damaged)
        c["n_lines"] = k
        c["added"] = lines[k - 1].name
        out.append(c)
        if progress:
            progress(k / len(lines), f"{k} line(s)")
    return out
