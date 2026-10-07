"""Buoyancy budget: how much displacement is needed, and where it can come from.

Needs only weights and member sizes (no coordinates), so it works before the member
geometry is available.  Definitions (screening level):

  capacity   = rho_w * enclosed outside volume of the sealed members and tanks, i.e. the
               displacement if everything sealed were fully submerged.  Wall steel is in the
               structure weight, so a sealed tube counts its full outside volume.
  reserve    = (capacity - W) / W, with W the FACTORED weight (the BoD asks for factored
               weight against unfactored buoyancy).
  required   = W * (1 + reserve_min).

The result says how much tank volume is still needed, and, if a tank steel weight per m3
is given, accounts for the extra steel the extra volume brings.  It does not replace the
waterline, GM and clearance checks, which need the geometry (see buoyancy.py).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

RHO_W = 1.025   # t/m3, sea water


def tube_outer_volume_m3(od_mm: float, length_m: float) -> float:
    """Outside volume of a circular tube: pi/4 * OD^2 * L."""
    return math.pi / 4.0 * (od_mm / 1000.0) ** 2 * length_m


def source_volume_m3(src: dict) -> float:
    """Volume of one buoyancy source: explicit volume_m3, else from OD and length."""
    v = src.get("volume_m3")
    if v is not None:
        return float(v)
    if src.get("od_mm") is not None and src.get("length_m") is not None:
        return tube_outer_volume_m3(float(src["od_mm"]), float(src["length_m"]))
    return 0.0


def sources_table(sources: Sequence[dict], rho_w: float = RHO_W) -> list:
    rows = []
    for s in sources:
        v = source_volume_m3(s)
        rows.append(dict(name=s["name"], group=s.get("group", "other"), od_mm=s.get("od_mm"),
                         length_m=s.get("length_m"), volume_m3=v, capacity_t=v * rho_w,
                         note=s.get("note", "")))
    return rows


def budget(weight_t: float, sources: Sequence[dict], share: dict, *, reserve_min_pct: float = 10.0,
           rho_w: float = RHO_W, tank_steel_in_weight_t: float = 0.0,
           tank_steel_t_per_m3: float = 0.0) -> dict:
    """Capacity, reserve and the extra tank volume still needed.

    share: source name -> fraction sealed (0..1); a missing name counts as 0.
    Sources in group 'tank' are listed separately; their volume (if known) counts as capacity.
    tank_steel_in_weight_t: tank steel already inside weight_t (so it can be re-scaled).
    tank_steel_t_per_m3: steel weight per m3 of extra tank volume (0 = extra steel ignored)."""
    if weight_t <= 0:
        raise ValueError("Weight must be positive.")
    if not 0.0 <= reserve_min_pct < 1000.0:
        raise ValueError("Reserve must be between 0 and 1000 %.")
    if tank_steel_t_per_m3 < 0 or tank_steel_in_weight_t < 0 or tank_steel_in_weight_t > weight_t:
        raise ValueError("Tank steel inputs are inconsistent with the weight.")
    r = reserve_min_pct / 100.0
    by_group: dict = {}
    other_t = tank_t = 0.0
    for s in sources:
        f = float(share.get(s["name"], 0.0))
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"Sealed share of '{s['name']}' must be between 0 and 1.")
        cap = source_volume_m3(s) * rho_w * f
        g = s.get("group", "other")
        by_group[g] = by_group.get(g, 0.0) + cap
        if g == "tank":
            tank_t += cap
        else:
            other_t += cap
    capacity_t = other_t + tank_t
    required_t = weight_t * (1.0 + r)
    # extra tank volume on top of what is listed
    if tank_steel_t_per_m3 > 0.0:
        w_base = weight_t - tank_steel_in_weight_t
        den = rho_w - (1.0 + r) * tank_steel_t_per_m3
        v_total = None if den <= 0 else max(0.0, ((1.0 + r) * w_base - other_t) / den)
    else:
        v_total = max(0.0, (required_t - other_t) / rho_w)
    return dict(
        weight_t=weight_t, reserve_min_pct=reserve_min_pct, required_t=required_t, capacity_t=capacity_t,
        other_capacity_t=other_t, tank_capacity_t=tank_t, by_group_t=by_group,
        reserve_pct=100.0 * (capacity_t - weight_t) / weight_t,
        shortfall_t=max(0.0, required_t - capacity_t), meets_reserve=bool(capacity_t >= required_t - 1e-9),
        tank_volume_required_total_m3=v_total,            # all tank volume needed, existing plus new
        tank_volume_existing_m3=tank_t / rho_w,           # sealed tank volume already listed
        tank_volume_additional_m3=(None if v_total is None else max(0.0, v_total - tank_t / rho_w)),
        feasible=v_total is not None,
        tank_steel_existing_t=tank_steel_in_weight_t,     # tank steel already in the weight
        tank_steel_required_total_t=(None if v_total is None else v_total * tank_steel_t_per_m3),
        tank_steel_additional_t=(None if v_total is None else max(0.0, v_total * tank_steel_t_per_m3 - tank_steel_in_weight_t)))


def loss_cases(weight_t: float, sources: Sequence[dict], share: dict, *, reserve_min_pct: float = 5.0,
               rho_w: float = RHO_W) -> list:
    """Reserve left when each source is lost in turn (accidental flooding), against the damaged limit."""
    out = []
    for s in sources:
        f = float(share.get(s["name"], 0.0))
        cap = source_volume_m3(s) * rho_w * f
        if cap <= 0:
            continue
        sh = dict(share)
        sh[s["name"]] = 0.0
        b = budget(weight_t, sources, sh, reserve_min_pct=reserve_min_pct, rho_w=rho_w)
        out.append(dict(lost=s["name"], lost_t=cap, capacity_t=b["capacity_t"], reserve_pct=b["reserve_pct"],
                        meets=b["meets_reserve"]))
    return out
