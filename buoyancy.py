"""Buoyancy modules: the jacket structure plus switchable buoyancy items.

The jacket structure (steel, no buoyancy of its own) is one layer.  Each source of
buoyancy is a separate BuoyancyModule that can be switched on, off or damaged:

  * sealed members  - the named structure members (or the part of them between two
                      heights, e.g. rip-out diaphragm elevations) are made watertight.
                      The steel is already in the structure weights, so such a module
                      adds displacement only.
  * tanks           - own cylinders plus their own steel weight (and optional ballast).

A case = structure + a state for every module.  Nothing here changes the engine:
`assemble` produces ordinary Element / WeightItem / Opening lists for JacketModel.

States
  "sealed"  - module in place and watertight (buoyant).
  "off"     - module not present: no volume, no steel weight (e.g. tanks removed).
  "damaged" - module present but accidentally flooded: steel weight stays, volume is
              lost.  Any damaged module makes the model a damage case.

Screening tool: not independently checked.  Sealed members use the full outside volume
(wall steel is in the structure weight); diaphragm / rip-out steel is the user's input.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import jacket_stability as js

STATES = ("sealed", "off", "damaged")
_EPS = 1e-9


@dataclass
class Structure:
    """Steel jacket, no buoyancy of its own.  Elements are normally buoyant=False."""
    elements: list
    weights: list
    openings: list = field(default_factory=list)


@dataclass
class BuoyancyModule:
    name: str
    members: tuple = ()            # structure member names to seal (whole member or the clipped part)
    z_lo: Optional[float] = None   # clip sealed members to this height range (body-frame z); None = no limit
    z_hi: Optional[float] = None
    elements: tuple = ()           # own cylinders (tanks), js.Element objects, normally buoyant=True
    weights: tuple = ()            # own steel / diaphragm weights, js.WeightItem objects
    ballast_t: float = 0.0         # ballast water carried when the module is present
    ballast_z: Optional[float] = None   # height of that ballast (None = mean height of own elements)
    note: str = ""


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def _t_interval(e: js.Element, z_lo: Optional[float], z_hi: Optional[float]) -> Optional[tuple]:
    """Parameter interval [t0, t1] (0 = p1, 1 = p2) of the member between two heights, or None."""
    z1, z2 = float(e.p1[2]), float(e.p2[2])
    lo = -math.inf if z_lo is None else float(z_lo)
    hi = math.inf if z_hi is None else float(z_hi)
    dz = z2 - z1
    if abs(dz) < _EPS:
        return (0.0, 1.0) if lo - _EPS <= z1 <= hi + _EPS else None
    ta, tb = (lo - z1) / dz, (hi - z1) / dz
    t0, t1 = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
    return (t0, t1) if t1 - t0 > _EPS else None


def _point(e: js.Element, t: float) -> tuple:
    return tuple(float(a) + t * (float(b) - float(a)) for a, b in zip(e.p1, e.p2))


def _piece(e: js.Element, t0: float, t1: float, name: str, **kw) -> js.Element:
    return replace(e, name=name, p1=_point(e, t0), p2=_point(e, t1), **kw)


def _validate_module(m: BuoyancyModule, known: set) -> None:
    missing = [n for n in m.members if n not in known]
    if missing:
        raise ValueError(f"Module '{m.name}' refers to unknown structure members: {', '.join(missing[:5])}")
    if m.z_lo is not None and m.z_hi is not None and m.z_hi <= m.z_lo:
        raise ValueError(f"Module '{m.name}': z_hi must be above z_lo.")


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def assemble(structure: Structure, modules: Sequence[BuoyancyModule], states: dict,
             ballast: Optional[dict] = None) -> tuple:
    """Return (elements, weights, openings, damaged) for JacketModel.

    states: module name -> 'sealed' | 'off' | 'damaged'; a missing module is 'off'.
    ballast: optional module name -> ballast mass [t] overriding BuoyancyModule.ballast_t."""
    names = [m.name for m in modules]
    if len(set(names)) != len(names):
        raise ValueError("Module names must be unique.")
    for k, s in states.items():
        if k not in names:
            raise ValueError(f"State given for unknown module '{k}'.")
        if s not in STATES:
            raise ValueError(f"State of '{k}' must be one of {STATES}, not '{s}'.")
    known = {e.name for e in structure.elements}
    for m in modules:
        _validate_module(m, known)
    active = [m for m in modules if states.get(m.name, "off") != "off"]

    # claims on structure members, as (t0, t1, module, damaged)
    claims: dict = {}
    for m in active:
        dmg = states[m.name] == "damaged"
        for e in structure.elements:
            if e.name in m.members:
                iv = _t_interval(e, m.z_lo, m.z_hi)
                if iv:
                    claims.setdefault(e.name, []).append((iv[0], iv[1], m.name, dmg))

    elements: list = []
    for e in structure.elements:
        cl = sorted(claims.get(e.name, []))
        if not cl:
            elements.append(e)
            continue
        for a, b in zip(cl, cl[1:]):
            if b[0] < a[1] - _EPS:
                raise ValueError(f"Modules '{a[2]}' and '{b[2]}' both seal member '{e.name}' over the same length.")
        t, k = 0.0, 0
        for t0, t1, mname, dmg in cl:
            if t0 - t > _EPS:
                k += 1
                elements.append(_piece(e, t, t0, f"{e.name}[{k}]"))
            k += 1
            elements.append(_piece(e, t0, t1, f"{e.name}[{k}]", d_in=0.0, buoyant=True, tank=False, flooded=dmg))
            t = t1
        if 1.0 - t > _EPS:
            k += 1
            elements.append(_piece(e, t, 1.0, f"{e.name}[{k}]"))

    weights = list(structure.weights)
    ballast = ballast or {}
    for m in active:
        dmg = states[m.name] == "damaged"
        for e in m.elements:
            elements.append(replace(e, flooded=True) if dmg else e)
        weights.extend(m.weights)
        bt = float(ballast.get(m.name, m.ballast_t))
        if bt:
            bz = m.ballast_z
            if bz is None:
                zs = [z for e in m.elements for z in (e.p1[2], e.p2[2])]
                if not zs:
                    raise ValueError(f"Module '{m.name}' carries ballast but has no height for it (set ballast_z).")
                bz = sum(zs) / len(zs)
            x, y = (_centroid_xy(m) if m.elements else (0.0, 0.0))
            weights.append(js.WeightItem(f"{m.name} ballast", bt, x, y, float(bz)))
    damaged = any(states.get(m.name) == "damaged" for m in modules)
    return elements, weights, list(structure.openings), damaged


def _centroid_xy(m: BuoyancyModule) -> tuple:
    xs = [(e.p1[0] + e.p2[0]) / 2 for e in m.elements]
    ys = [(e.p1[1] + e.p2[1]) / 2 for e in m.elements]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(structure: Structure, modules: Sequence[BuoyancyModule], states: dict,
             params: js.Params, crit: js.Criteria, *, reserve_min_pct: float = 0.0,
             ballast: Optional[dict] = None) -> dict:
    """Float one case (level, no wind, no tow) and return a flat row of results."""
    label = ", ".join(f"{m.name}={states.get(m.name, 'off')}" for m in modules)
    row: dict = dict(case=label, **{f"state:{m.name}": states.get(m.name, "off") for m in modules})
    try:
        els, wts, ops, damaged = assemble(structure, modules, states, ballast)
    except ValueError as exc:
        return {**row, "floats": False, "passed": False, "error": str(exc)}
    row["damaged"] = damaged
    row["weight_t"] = float(sum(w.mass_t for w in wts))
    try:
        mdl = js.JacketModel(els, wts, ops, params, damaged=damaged)
        fc = js.float_check(mdl, crit)
    except ValueError as exc:
        return {**row, "floats": False, "passed": False, "error": str(exc)}
    h = mdl.hydrostatics()
    reserve = h["reserve_buoyancy_pct"]
    pass_reserve = None if reserve_min_pct <= 0 else bool(reserve >= reserve_min_pct)
    flags = [fc["pass_clearance"], fc["pass_emerged"], fc["pass_gm"], pass_reserve]
    ok = bool(fc["passed"] and all(f for f in flags if f is not None))
    row.update(
        floats=True, error="", capacity_t=h["buoyant_volume"] * params.rho_w, reserve_pct=reserve,
        waterline_z=fc["zw"], draft_m=fc["draft_total"], clearance_m=fc["clearance"],
        tank_emerged_min_m=fc["emerged_min"], gm_min_m=fc["gm_min"], kb_m=h["kb"], kg_m=h["kg"],
        waterplane_m2=h["waterplane_area"], free_tilt_deg=fc["free_tilt_deg"], stable=fc["stable"],
        ballast_headroom_t=fc["ballast_headroom_t"], governing_limit=fc["governing_limit"],
        pass_clearance=fc["pass_clearance"], pass_emerged=fc["pass_emerged"], pass_gm=fc["pass_gm"],
        pass_reserve=pass_reserve, passed=ok)
    return row


def capacity(structure: Structure, modules: Sequence[BuoyancyModule], states: dict, params: js.Params,
             *, ballast: Optional[dict] = None) -> dict:
    """Floating capacity of one state combination, with no waterline solved.

    The capacity is the weight of water the buoyant, intact members displace when fully submerged
    (rho x volume).  It is an upper bound on what the jacket can carry: reserve = (capacity - weight) / weight,
    the same definition the float check uses.  Flooded ('damaged') and 'off' members count for nothing."""
    els, wts, _ops, damaged = assemble(structure, modules, states, ballast)
    live = [e for e in els if e.buoyant and not (damaged and e.flooded)]
    vol = float(js.build_elem_arrays(live).vol_total)
    cap = vol * params.rho_w
    weight = float(sum(w.mass_t for w in wts))
    net = cap - weight
    return dict(volume_m3=vol, capacity_t=cap, weight_t=weight, net_t=net,
                reserve_pct=(100.0 * net / weight) if weight > 0 else float("nan"), floats=bool(net > 0.0))


def member_modules(modules: Sequence[BuoyancyModule]) -> list:
    """Modules that seal existing structure members (legs, outriggers), as opposed to adding tanks."""
    return [m for m in modules if m.members and not m.elements]


def combinations(modules: Sequence[BuoyancyModule], options: Sequence[str] = ("sealed", "off"),
                 fixed: Optional[dict] = None) -> list:
    """All state dicts over `options` for the modules not in `fixed`."""
    fixed = dict(fixed or {})
    free = [m.name for m in modules if m.name not in fixed]
    out = []
    for combo in itertools.product(options, repeat=len(free)):
        s = dict(fixed)
        s.update(zip(free, combo))
        out.append(s)
    return out


def sweep(structure: Structure, modules: Sequence[BuoyancyModule], params: js.Params, crit: js.Criteria,
          *, options: Sequence[str] = ("sealed", "off"), fixed: Optional[dict] = None,
          reserve_min_pct: float = 0.0, max_cases: int = 512) -> list:
    """Evaluate every combination of module states.  Rows are plain dicts."""
    cases = combinations(modules, options, fixed)
    if len(cases) > max_cases:
        raise ValueError(f"{len(cases)} cases requested, limit is {max_cases}; fix some modules first.")
    return [evaluate(structure, modules, s, params, crit, reserve_min_pct=reserve_min_pct) for s in cases]


def damage_cases(structure: Structure, modules: Sequence[BuoyancyModule], base_states: dict,
                 params: js.Params, crit: js.Criteria, *, reserve_min_pct: float = 0.0) -> list:
    """Flood each present module in turn (one at a time) starting from base_states."""
    rows = []
    for m in modules:
        if base_states.get(m.name, "off") != "sealed":
            continue
        s = dict(base_states)
        s[m.name] = "damaged"
        r = evaluate(structure, modules, s, params, crit, reserve_min_pct=reserve_min_pct)
        r["damaged_module"] = m.name
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# Tank sizing
# ----------------------------------------------------------------------------
def scale_tanks(modules: Sequence[BuoyancyModule], names: Sequence[str], factor: float) -> list:
    """Copy of `modules` with the tank cylinders of the named modules scaled in diameter by `factor`
    (length and position unchanged).  Their own steel weight is scaled with the volume, factor**2."""
    out = []
    for m in modules:
        if m.name in names and m.elements:
            els = tuple(replace(e, d_out=e.d_out * factor, d_in=e.d_in * factor) for e in m.elements)
            wts = tuple(replace(w, mass_t=w.mass_t * factor ** 2) for w in m.weights)
            m = replace(m, elements=els, weights=wts)
        out.append(m)
    return out


def size_tanks(structure: Structure, modules: Sequence[BuoyancyModule], base_states: dict, params: js.Params,
               crit: js.Criteria, names: Sequence[str], *, reserve_min_pct: float = 0.0,
               damaged_reserve_min_pct: float = 0.0, f_min: float = 0.6, f_max: float = 2.0, steps: int = 29) -> dict:
    """Scale the diameter of the named tank modules over a range and report the intact and one-module-damaged
    results for each size.  Returns {"rows": [...], "f_intact": first factor that passes intact or None,
    "f_all": first factor passing intact and every damage case or None}.  Tank steel weight scales with volume."""
    names = list(names)
    tanks = [m for m in modules if m.name in names and m.elements]
    if not tanks or steps < 2 or f_max <= f_min:
        raise ValueError("size_tanks needs at least one named module with its own tank elements and f_max > f_min")
    d0 = [e.d_out for m in tanks for e in m.elements]
    rows = []
    for i in range(steps):
        f = f_min + (f_max - f_min) * i / (steps - 1)
        mods = scale_tanks(modules, names, f)
        r = evaluate(structure, mods, base_states, params, crit, reserve_min_pct=reserve_min_pct)
        dmg = damage_cases(structure, mods, base_states, params, crit, reserve_min_pct=damaged_reserve_min_pct)
        n_ok = sum(1 for x in dmg if x.get("passed"))
        vol = sum(math.pi / 4 * e.d_out ** 2 * math.dist(e.p1, e.p2) for m in mods if m.name in names for e in m.elements)
        rows.append(dict(factor=f, diameter_m=max(d0) * f, tank_volume_m3=vol, floats=r["floats"],
                         weight_t=r.get("weight_t"), reserve_pct=r.get("reserve_pct"), clearance_m=r.get("clearance_m"),
                         gm_min_m=r.get("gm_min_m"), intact_passed=bool(r.get("passed")),
                         damage_passed=n_ok, damage_total=len(dmg),
                         all_passed=bool(r.get("passed")) and n_ok == len(dmg)))
    f_i = next((x["factor"] for x in rows if x["intact_passed"]), None)
    f_a = next((x["factor"] for x in rows if x["all_passed"]), None)
    return dict(rows=rows, f_intact=f_i, f_all=f_a)
