"""Tow-plan screening for the final approach (tow vessels only): pull needed, time and holding check.

Screening level and independent of the member geometry: the drag area comes from the tube sizes
(OD x length), so it works from the case-file budget sources.  Not a tow-master's plan and not a
replacement for a tug / bollard-pull assessment.

Drag on the upright jacket is taken as isotropic in the horizontal plane (vertical legs; bracing
averaged), F = 1/2 rho Cd A |v_r| v_r, with v_r the water velocity relative to the jacket:
v_r = current - jacket velocity over the ground.
"""
from __future__ import annotations

import math
from typing import Sequence

KN = 0.514444     # m/s per knot
RHO_W = 1.025     # t/m3
G = 9.80665


def drag_area_m2(sources: Sequence[dict], submerged_fraction: float = 1.0, groups=("leg", "brace")) -> float:
    """Projected area of the tubes, sum of OD x length, times the submerged fraction."""
    if not 0.0 <= submerged_fraction <= 1.0:
        raise ValueError("Submerged fraction must be between 0 and 1.")
    a = 0.0
    for s in sources:
        if s.get("group", "other") in groups and s.get("od_mm") is not None and s.get("length_m") is not None:
            a += float(s["od_mm"]) / 1000.0 * float(s["length_m"])
    return a * submerged_fraction


def drag_components(ca_m2: float, ground_speed_kn: float, current_kn: float, current_dir_deg: float,
                    cd: float = 1.0, rho_w: float = RHO_W) -> dict:
    """Drag the tugs must overcome: along the track (>= 0), sideways, resultant [kN], relative speed [m/s].
    current_dir_deg: direction the current flows toward, from the tow direction (0 = with the tow, 180 = head)."""
    if min(ca_m2, ground_speed_kn, current_kn, cd) < 0:
        raise ValueError("Area, speeds and Cd cannot be negative.")
    u, c, a = ground_speed_kn * KN, current_kn * KN, math.radians(current_dir_deg)
    vx, vy = c * math.cos(a) - u, c * math.sin(a)             # water relative to the jacket; x = tow direction
    v = math.hypot(vx, vy)
    k = 0.5 * rho_w * cd * ca_m2 * v           # kN per (m/s) of v_r (rho in t/m3 gives kN directly)
    dx, dy = -k * vx, -k * vy                  # pull the tugs must supply (drag on the jacket is along v_r)
    return dict(v_rel_ms=v, along_kn=max(0.0, dx), side_kn=dy, resultant_kn=math.hypot(max(0.0, dx), dy))


def plan(legs: Sequence[dict], ca_m2: float, current_kn: float, current_dir_deg: float, *, cd: float = 1.0,
         margin: float = 1.0, tug_count: int = 2, tug_bollard_t: float = 0.0, tug_eff: float = 0.8,
         rho_w: float = RHO_W) -> dict:
    """Per-leg pull, time and tug utilisation, plus the hold-position check (stopped over the ground).

    legs: dicts with from_m, to_m (distance to the platform, decreasing) and speed_kn (over ground)."""
    if margin <= 0 or tug_count < 0 or tug_bollard_t < 0 or not 0 < tug_eff <= 1:
        raise ValueError("Check margin, tug count, bollard pull and effective fraction.")
    avail_kn = tug_count * tug_bollard_t * G * tug_eff
    rows, t_total = [], 0.0
    for lg in legs:
        d = float(lg["from_m"]) - float(lg["to_m"])
        v = float(lg["speed_kn"])
        if d <= 0 or v <= 0:
            raise ValueError("Each leg needs from > to and a positive speed.")
        dc = drag_components(ca_m2, v, current_kn, current_dir_deg, cd, rho_w)
        t_h = d / (v * KN) / 3600.0
        t_total += t_h
        need = margin * dc["resultant_kn"]
        rows.append(dict(from_m=lg["from_m"], to_m=lg["to_m"], speed_kn=v, hours=t_h, v_rel_ms=dc["v_rel_ms"],
                         along_kn=margin * dc["along_kn"], side_kn=margin * dc["side_kn"], need_kn=need,
                         need_t=need / G, util_pct=(100.0 * need / avail_kn if avail_kn > 0 else None)))
    hold = drag_components(ca_m2, 0.0, current_kn, current_dir_deg, cd, rho_w)
    hold_kn = margin * hold["resultant_kn"]
    return dict(rows=rows, total_hours=t_total, available_kn=avail_kn, available_t=avail_kn / G,
                hold_kn=hold_kn, hold_t=hold_kn / G,
                hold_util_pct=(100.0 * hold_kn / avail_kn if avail_kn > 0 else None),
                governing_need_kn=max((r["need_kn"] for r in rows), default=0.0))
