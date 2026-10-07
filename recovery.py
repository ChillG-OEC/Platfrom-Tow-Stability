"""Recovery measures for a failing case.

When the study does not meet the criteria, each practical lever is searched with the same engine until the case
passes, and the value that restores it is reported: reduce tow speed, reduce a pull-in line, move a connection
lower, press up a slack tank, add or remove ballast at a named tank, change heading, restore a buoyancy module.
Set-down failures add: wait for a higher tide, deballast to restore tank emergence, and do not proceed.

A measure is only listed as verified when a full re-run of the study at the proposed value passes. These are
screening indications from a static model, not an operating procedure.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import jacket_stability as js


@dataclass
class Measure:
    kind: str                    # short key: tow speed, pull-in line, connection, slack tank, ballast, deballast, heading, module, tide, emergence, stop
    action: str                  # what to do, in words
    verified: bool               # a full re-run at the proposed value passes
    detail: str = ""
    effect: dict = field(default_factory=dict)


def _passes(snap: dict, parts_fn: Callable, betas, grid, damaged: bool, extra_weights=None):
    """(passed, summary) of a full study for this snapshot."""
    els, wts, ops, params, crit = parts_fn(snap)
    if extra_weights:
        wts = list(wts) + [js.WeightItem(*w) for w in extra_weights]
    try:
        model = js.JacketModel(els, wts, ops, params, damaged=damaged)
        r = js.run_study(model, betas, grid, crit)
    except ValueError:
        return False, {}
    return bool(r["summary"]["passed"]), r["summary"]


def _boundary(ok: Callable[[float], bool], bad: float, limit: float, tol: float, maxit: int = 9) -> Optional[float]:
    """Passing value nearest to `bad` (which fails) on the way to `limit`; None if `limit` itself fails."""
    if not ok(limit):
        return None
    good = limit
    for _ in range(maxit):
        if abs(good - bad) <= tol:
            break
        mid = 0.5 * (good + bad)
        if ok(mid):
            good = mid
        else:
            bad = mid
    return good


def _with(snap: dict, **widgets) -> dict:
    s = copy.deepcopy(snap)
    s["widgets"].update(widgets)
    return s


def _eff(summary: dict) -> dict:
    return dict(gm_min=summary.get("gm_min"), heel_max=summary.get("heel_max"), ratio_min=summary.get("ratio_min"))


def tables_from_assembled(els, wts, ops) -> dict:
    """Snapshot table rows for an assembled platform."""
    return dict(
        elements=[dict(name=e.name, x1=e.p1[0], y1=e.p1[1], z1=e.p1[2], x2=e.p2[0], y2=e.p2[1], z2=e.p2[2],
                       d_out=e.d_out, d_in=e.d_in, buoyant=e.buoyant, exposed=e.exposed, flooded=e.flooded,
                       tank=e.tank) for e in els],
        weights=[dict(item=w.name, mass_t=w.mass_t, x=w.x, y=w.y, z=w.z) for w in wts],
        openings=[dict(name=o.name, x=o.x, y=o.y, z=o.z) for o in ops])


def propose(snap: dict, parts_fn: Callable, betas, grid, damaged: bool, *, fail_betas=None, tanks=None, module_options=None,
            float_check: Optional[dict] = None, crit_float=None, progress: Optional[Callable] = None) -> list:
    """Search every lever.

    tanks: [{name, x, y, z, cap_t}] named tanks where ballast could be added (cap_t = water the tank holds).
    module_options: [{name, state, snap}] one snapshot per module that is not sealed, with it restored.
    float_check: result of js.float_check for the set-down condition (adds tide and tank-emergence measures)."""
    out: list[Measure] = []
    w = snap["widgets"]
    margin_note = "The value is where the case just passes: leave a margin."
    steps = []

    def tick(msg):
        if progress:
            progress(min(0.99, len(steps) / 12.0), msg)
        steps.append(msg)

    def run(s, extra=None):
        return _passes(s, parts_fn, betas, grid, damaged, extra)

    quick = np.asarray(fail_betas if fail_betas else betas, float)      # headings that failed: cheap search set

    def run_quick(s, extra=None):
        return _passes(s, parts_fn, quick, grid, damaged, extra)

    def solve(make, bad, limit, tol, extra_fn=None):
        """Boundary value on the cheap heading set, then confirmed on every heading (searched again if it fails).
        make(v) -> snapshot; extra_fn(v) -> extra weights. Returns (value, summary, verified) or None."""
        def ok(f):
            return lambda v: f(make(v), extra_fn(v) if extra_fn else None)[0]
        v = _boundary(ok(run_quick), bad, limit, tol)
        if v is None:
            return None
        okk, sm = run(make(v), extra_fn(v) if extra_fn else None)
        if not okk and len(quick) < len(betas):
            v = _boundary(ok(run), bad, limit, tol)
            if v is None:
                return None
            okk, sm = run(make(v), extra_fn(v) if extra_fn else None)
        return v, sm, okk

    # 1. tow speed
    spd = float(w.get("p_tow_speed_kn", 0.0))
    if spd > 0.0 and w.get("p_tow_mode") != "manual":
        tick("tow speed")
        res_ = solve(lambda v: _with(snap, p_tow_speed_kn=v), spd, 0.0, 0.1)
        if res_:
            v, sm, okk = res_
            out.append(Measure("tow speed", f"Reduce tow speed below {v:.1f} kn (now {spd:.1f} kn).", okk,
                               "The tow force is estimated from drag at the tow speed.", _eff(sm)))
        else:
            out.append(Measure("tow speed", "Reducing tow speed alone does not restore the case.", False))
    # 1b. manual pull
    man = float(w.get("p_tow_manual_kn", 0.0))
    if w.get("p_tow_mode") == "manual" and man > 0.0:
        tick("manual tow pull")
        res_ = solve(lambda v: _with(snap, p_tow_manual_kn=v), man, 0.0, 5.0)
        if res_:
            v, sm, okk = res_
            out.append(Measure("tow speed", f"Reduce the tow pull to {v:.0f} kN or less (now {man:.0f} kN).", okk, "", _eff(sm)))
    # 2. fixed (pull-in) lines
    for i, ln in enumerate(snap.get("lines", [])):
        if ln.get("type") != "Tow leg" and float(ln.get("tension_kn", 0.0)) > 0.0:
            tick(f"line {ln.get('name')}")

            def with_t(t, i=i):
                s = copy.deepcopy(snap)
                s["lines"][i]["tension_kn"] = float(t)
                return s
            t0 = float(ln["tension_kn"])
            res_ = solve(with_t, t0, 0.0, max(1.0, 0.01 * t0))
            if res_:
                v, sm, okk = res_
                out.append(Measure("pull-in line", f"Reduce pull-in line '{ln['name']}' to {v:.0f} kN or less (now {t0:.0f} kN).",
                                   okk, "", _eff(sm)))
    # 3. connection lower (main tow point, then each line)
    els = snap.get("elements", [])
    z_lo = min((min(float(e["z1"]), float(e["z2"])) for e in els), default=None)
    if z_lo is not None:
        z0 = float(w.get("p_tow_z", 0.0))
        if z0 > z_lo:
            tick("tow connection height")
            res_ = solve(lambda v: _with(snap, p_tow_z=v), z0, z_lo, 0.1)
            if res_:
                v, sm, okk = res_
                out.append(Measure("connection", f"Move the tow connection lower, to z = {v:.1f} m or below (now {z0:.1f} m).",
                                   okk, "z in the input-table frame (same as tab 2).", _eff(sm)))
        for i, ln in enumerate(snap.get("lines", [])):
            z1 = float(ln.get("z", 0.0))
            if z1 > z_lo and float(ln.get("tension_kn", 0.0)) > 0.0:
                def with_z(z, i=i):
                    s = copy.deepcopy(snap)
                    s["lines"][i]["z"] = float(z)
                    return s
                res_ = solve(with_z, z1, z_lo, 0.1)
                if res_:
                    v, sm, okk = res_
                    out.append(Measure("connection", f"Move the '{ln['name']}' attachment lower, to z = {v:.1f} m or below (now {z1:.1f} m).",
                                       okk, "", _eff(sm)))
    # 4. press up a slack tank
    fsm = float(snap.get("fsc_m", 0.0)) * 1.0
    if float(w.get("p_fsm", 0.0)) > 0.0 or fsm > 0.0:
        tick("slack tank")
        s2 = copy.deepcopy(snap)
        s2["widgets"]["p_fsm"] = 0.0
        s2["fsc_m"] = 0.0
        okk, sm = run(s2)
        if okk:
            out.append(Measure("slack tank", "Press up every slack tank (remove the free-surface moment).", True,
                               "Passes with the free-surface correction set to zero.", _eff(sm)))
        else:
            out.append(Measure("slack tank", "Pressing up slack tanks alone does not restore the case.", False))
    # 5. heading
    h0 = float(w.get("p_tow_heading_deg", 0.0))
    tick("tow heading")
    cands = sorted((hd for hd in range(0, 360, 30) if abs(((hd - h0 + 180) % 360) - 180) > 1e-6),
                   key=lambda x: abs(((x - h0 + 180) % 360) - 180))
    good = []
    for hd in cands:
        if len(good) >= 3:
            break
        if run_quick(_with(snap, p_tow_heading_deg=float(hd)))[0] and run(_with(snap, p_tow_heading_deg=float(hd)))[0]:
            good.append(hd)
    if good:
        sm = run(_with(snap, p_tow_heading_deg=float(good[0])))[1]
        out.append(Measure("heading", f"Change the tow heading to {good[0]}° (also passing: "
                           + (", ".join(f"{x}°" for x in good[1:]) or "none of the others tested") + f"; now {h0:.0f}°).", True,
                           "Tow heading in the body frame, tested every 30°.", _eff(sm)))
    # 6. ballast at a named tank
    for t in tanks or []:
        tick(f"ballast {t['name']}")
        cap = float(t["cap_t"])
        if cap <= 0:
            continue
        res_ = solve(lambda v: snap, 0.0, cap, max(1.0, 0.01 * cap),
                     extra_fn=lambda v, t=t: [(f"Ballast {t['name']}", float(v), t["x"], t["y"], t["z"])])
        if res_:
            v, sm, okk = res_
            out.append(Measure("ballast", f"Add at least {v:.0f} t of ballast in {t['name']} (holds up to {cap:.0f} t).", okk,
                               "Ballast water treated as a point mass at the tank centre; check the reserve buoyancy afterwards.",
                               _eff(sm)))
    # 7. deballast existing ballast items
    for i, wr in enumerate(snap.get("weights", [])):
        if "ballast" in str(wr.get("item", "")).lower() and float(wr.get("mass_t", 0.0)) > 0.0:
            tick(f"deballast {wr['item']}")

            def with_m(m, i=i):
                s = copy.deepcopy(snap)
                s["weights"][i]["mass_t"] = float(m)
                return s
            m0 = float(wr["mass_t"])
            res_ = solve(with_m, m0, 0.0, max(1.0, 0.01 * m0))
            if res_:
                v, sm, okk = res_
                out.append(Measure("deballast", f"Deballast '{wr['item']}' to {v:.0f} t or less (now {m0:.0f} t).", okk, "", _eff(sm)))
    # 8. restore a buoyancy module
    for mo in module_options or []:
        tick(f"restore {mo['name']}")
        okk, sm = run(mo["snap"])
        if okk:
            out.append(Measure("module", f"Restore buoyancy module '{mo['name']}' (now {mo['state']}).", True, "", _eff(sm)))
    # 9. set-down: tide and tank emergence
    if float_check:
        F = float_check
        if F.get("pass_clearance") is False and F.get("depth_total") is not None and crit_float is not None:
            need = crit_float.clear_min_m - (F["water_depth"] - F["draft_total"])
            rng_note = f"The tide in the inputs is {F['tide']:.2f} m above LAT."
            out.append(Measure("tide", f"Wait for a tide of at least {need:.2f} m above LAT to reach {crit_float.clear_min_m:.1f} m clearance. {rng_note}",
                               True, "Clearance rises one for one with the tide; check the tidal range allows it."))
        if F.get("pass_emerged") is False and F.get("ballast_headroom_t") is not None and F["ballast_headroom_t"] < 0:
            out.append(Measure("emergence", f"Deballast or shed at least {-F['ballast_headroom_t']:.0f} t to restore the minimum tank emergence.",
                               True, "Mass removed at the centre of gravity."))
    for m in out:
        if m.verified and m.kind in ("tow speed", "pull-in line", "connection", "ballast", "deballast"):
            m.detail = (m.detail + " " + margin_note).strip()
    # 10. stop
    if not any(m.verified for m in out if m.kind != "stop"):
        out.append(Measure("stop", "No single measure restores the case: do not proceed to the next stage until the "
                           "configuration is changed and the study passes.", True))
    elif float_check and not float_check.get("passed", True):
        out.append(Measure("stop", "Do not proceed to the next stage until the set-down float check passes.", True))
    return out
