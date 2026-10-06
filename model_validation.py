"""Screening-level model-input validation (task C2).

validate_model() checks the geometry, weights, openings and parameters for
mistakes that would otherwise give a wrong or misleading screening result.  It
does NOT solve equilibrium and does NOT change the existing validate_inputs().

Buoyancy convention (matches jacket_stability.ElemArrays.vol_total and
JacketModel.hydrostatics): the displaced volume of a member is the ANNULAR
volume  pi/4 * (d_out^2 - d_in^2) * L.  d_in = 0 for a solid sealed tank.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

from jacket_stability import Criteria, Element, Opening, Params, WeightItem

DEFAULT_EXACT_TOLERANCE_M = 1.0e-9
DEFAULT_COINCIDENT_TOLERANCE_M = 1.0e-3
DEFAULT_DIAMETER_WARNING_M = 20.0
DEFAULT_LENGTH_WARNING_M = 500.0
DEFAULT_COG_MARGIN_FACTOR = 1.0
DEFAULT_CAPACITY_TOLERANCE_T = 1.0e-9

# Codes whose condition the existing js.validate_inputs() already reports as a
# plain string.  merge_with_legacy() skips them so the app does not show the
# same problem twice.
OVERLAPS_VALIDATE_INPUTS = frozenset({
    "NON_POSITIVE_OUTSIDE_DIAMETER",
    "INVALID_ANNULUS_DIAMETERS",
    "ZERO_LENGTH_MEMBER",
    "NON_POSITIVE_TOTAL_MASS",
    "NO_BUOYANT_MEMBERS",
    "FLOTATION_IMPOSSIBLE",
})


def _issue(level: str, code: str, message: str, where: str) -> dict:
    return {"level": level, "code": code, "message": message, "where": where}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _point_is_valid(point: Any) -> bool:
    """Three finite coordinates in any non-string sequence (tuple, list, ndarray)."""
    if point is None or isinstance(point, (str, bytes)):
        return False
    try:
        values = list(point)
    except TypeError:
        return False
    return len(values) == 3 and all(_is_finite_number(v) for v in values)


def _point_tuple(point: Sequence[float]) -> tuple[float, float, float]:
    return float(point[0]), float(point[1]), float(point[2])


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    ax, ay, az = _point_tuple(point_a)
    bx, by, bz = _point_tuple(point_b)
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def _element_length(element: Element) -> float:
    return _distance(element.p1, element.p2)


def _endpoint_pair_distance(element_a: Element, element_b: Element) -> float:
    """Largest endpoint mismatch for the better of the two endpoint orderings."""
    same_order = max(_distance(element_a.p1, element_b.p1),
                     _distance(element_a.p2, element_b.p2))
    reversed_order = max(_distance(element_a.p1, element_b.p2),
                         _distance(element_a.p2, element_b.p1))
    return min(same_order, reversed_order)


def _diameters_match(element_a: Element, element_b: Element, tolerance_m: float) -> bool:
    return (abs(float(element_a.d_out) - float(element_b.d_out)) <= tolerance_m
            and abs(float(element_a.d_in) - float(element_b.d_in)) <= tolerance_m)


def _member_volume_m3(element: Element) -> float:
    """Annular displaced volume, as used by the engine: pi/4 (d_out^2 - d_in^2) L."""
    return (math.pi / 4.0
            * (float(element.d_out) ** 2 - float(element.d_in) ** 2)
            * _element_length(element))


def _append_non_finite_issue(issues: list, value: Any, where: str) -> None:
    if not _is_finite_number(value):
        issues.append(_issue("error", "NON_FINITE_NUMBER",
                             "Value must be a finite real number.", where))


def _validate_named_object_numbers(issues: list, objects: Iterable[Any], object_label: str,
                                   scalar_fields: Sequence[str]) -> None:
    for index, obj in enumerate(objects):
        identifier = str(getattr(obj, "name", "") or "").strip() or f"row {index + 1}"
        for field_name in scalar_fields:
            _append_non_finite_issue(issues, getattr(obj, field_name, None),
                                     f"{object_label} '{identifier}'.{field_name}")


def validate_model(
    elements: Sequence[Element],
    weights: Sequence[WeightItem],
    openings: Sequence[Opening],
    params: Params,
    crit: Criteria,
    *,
    damaged: bool = False,
    exact_tolerance_m: float = DEFAULT_EXACT_TOLERANCE_M,
    coincident_tolerance_m: float = DEFAULT_COINCIDENT_TOLERANCE_M,
    diameter_warning_m: float = DEFAULT_DIAMETER_WARNING_M,
    length_warning_m: float = DEFAULT_LENGTH_WARNING_M,
    cog_margin_factor: float = DEFAULT_COG_MARGIN_FACTOR,
    capacity_tolerance_t: float = DEFAULT_CAPACITY_TOLERANCE_T,
) -> list[dict]:
    """Screening-level model-input validation.

    Returns a list of {"level": "error"|"warning", "code", "message", "where"}.
    Does not mutate its arguments and does not solve hydrostatic equilibrium.

    damaged: when True, flooded members lose buoyancy (as in JacketModel with
    damaged=True).  Then "no buoyant member" and "cannot float" are reported as
    WARNINGS, because they are the expected physical outcome of a damage case
    rather than an input mistake.  When False (intact), flooded members still
    float, exactly as in the engine.
    """
    if exact_tolerance_m < 0.0:
        raise ValueError("exact_tolerance_m must be non-negative.")
    if coincident_tolerance_m < 0.0:
        raise ValueError("coincident_tolerance_m must be non-negative.")
    if diameter_warning_m <= 0.0:
        raise ValueError("diameter_warning_m must be positive.")
    if length_warning_m <= 0.0:
        raise ValueError("length_warning_m must be positive.")
    if cog_margin_factor < 0.0:
        raise ValueError("cog_margin_factor must be non-negative.")
    if capacity_tolerance_t < 0.0:
        raise ValueError("capacity_tolerance_t must be non-negative.")

    issues: list[dict] = []
    element_list = list(elements)
    weight_list = list(weights)
    opening_list = list(openings)

    # 1. Duplicate element names
    names: dict[str, list[int]] = defaultdict(list)
    for index, element in enumerate(element_list):
        name = str(getattr(element, "name", "")).strip()
        if name:
            names[name].append(index)
    for name, indices in names.items():
        if len(indices) > 1:
            rows = ", ".join(str(i + 1) for i in indices)
            issues.append(_issue("error", "DUPLICATE_ELEMENT_NAME",
                                 f"Element name '{name}' occurs more than once at rows {rows}.",
                                 f"elements[{rows}]"))

    # Per-element geometry checks
    valid_geometry: list[bool] = []
    for index, element in enumerate(element_list):
        name = str(getattr(element, "name", "")).strip()
        where = f"element '{name or f'row {index + 1}'}'"
        ok = True

        p1_ok, p2_ok = _point_is_valid(element.p1), _point_is_valid(element.p2)
        if not p1_ok:
            ok = False
            issues.append(_issue("error", "INVALID_POINT",
                                 "p1 must contain three finite coordinates.", f"{where}.p1"))
        if not p2_ok:
            ok = False
            issues.append(_issue("error", "INVALID_POINT",
                                 "p2 must contain three finite coordinates.", f"{where}.p2"))

        d_out_ok = _is_finite_number(element.d_out)
        d_in_ok = _is_finite_number(element.d_in)
        if not d_out_ok:
            ok = False
            issues.append(_issue("error", "NON_FINITE_NUMBER",
                                 "Outside diameter must be a finite number.", f"{where}.d_out"))
        if not d_in_ok:
            ok = False
            issues.append(_issue("error", "NON_FINITE_NUMBER",
                                 "Inside diameter must be a finite number.", f"{where}.d_in"))

        if d_out_ok:
            d_out = float(element.d_out)
            if d_out <= 0.0:
                ok = False
                issues.append(_issue("error", "NON_POSITIVE_OUTSIDE_DIAMETER",
                                     "Outside diameter must be greater than zero.", f"{where}.d_out"))
            if d_out > diameter_warning_m:
                issues.append(_issue(
                    "warning", "DIAMETER_LOOKS_TOO_LARGE",
                    f"Outside diameter is {d_out:g} m, greater than the {diameter_warning_m:g} m "
                    "unit warning threshold. Confirm the units.", f"{where}.d_out"))
        if d_in_ok and float(element.d_in) < 0.0:
            ok = False
            issues.append(_issue("error", "NEGATIVE_INSIDE_DIAMETER",
                                 "Inside diameter cannot be negative.", f"{where}.d_in"))
        if d_out_ok and d_in_ok and float(element.d_in) >= float(element.d_out):
            ok = False
            issues.append(_issue("error", "INVALID_ANNULUS_DIAMETERS",
                                 "Inside diameter must be smaller than the outside diameter.", where))

        if p1_ok and p2_ok:
            length_m = _element_length(element)
            if length_m <= exact_tolerance_m:
                ok = False
                issues.append(_issue(
                    "error", "ZERO_LENGTH_MEMBER",
                    f"Element endpoints are coincident within the {exact_tolerance_m:g} m tolerance.",
                    where))
            if length_m > length_warning_m:
                issues.append(_issue(
                    "warning", "MEMBER_LENGTH_LOOKS_TOO_LARGE",
                    f"Member length is {length_m:g} m, greater than the {length_warning_m:g} m "
                    "unit warning threshold. Confirm the units.", where))
        valid_geometry.append(ok)

    # Weights
    _validate_named_object_numbers(issues, weight_list, "weight", ("mass_t", "x", "y", "z"))
    finite_mass_values = [float(w.mass_t) for w in weight_list if _is_finite_number(w.mass_t)]
    for index, weight in enumerate(weight_list):
        if _is_finite_number(weight.mass_t) and float(weight.mass_t) < 0.0:
            name = str(weight.name).strip() or f"row {index + 1}"
            # Allowed by the existing tool for deductions, so a warning, not an error.
            issues.append(_issue("warning", "NEGATIVE_WEIGHT_ITEM_MASS",
                                 "Weight-item mass is negative (a deduction). Confirm this is intended.",
                                 f"weight '{name}'.mass_t"))
    total_mass_t = sum(finite_mass_values)
    if not finite_mass_values or total_mass_t <= 0.0:
        issues.append(_issue("error", "NON_POSITIVE_TOTAL_MASS",
                             f"Total finite model mass is {total_mass_t:g} t. "
                             "Total mass must be greater than zero.", "weights"))

    # Parameters and criteria
    parameter_fields = (
        "rho_w", "rho_a", "wind_speed_kn", "wind_cs", "wind_alpha", "wind_zref", "cd_water",
        "shielding", "tow_speed_kn", "tow_heading_deg", "tow_manual_kn", "tow_elevation_deg",
        "tow_factor", "fsc_m", "slice_m", "trim_limit_deg", "water_depth_m", "tide_m")
    for field_name in parameter_fields:
        if hasattr(params, field_name):
            _append_non_finite_issue(issues, getattr(params, field_name), f"params.{field_name}")
    if hasattr(params, "tow_point") and not _point_is_valid(params.tow_point):
        issues.append(_issue("error", "INVALID_POINT",
                             "tow_point must contain three finite coordinates.", "params.tow_point"))
    for field_name in ("gm_min", "heel_max_deg", "ratio_min", "cap_deg", "df_min_deg",
                       "clear_min_m", "emerged_min_m"):
        if hasattr(crit, field_name):
            _append_non_finite_issue(issues, getattr(crit, field_name), f"crit.{field_name}")

    # Effective buoyant members and capacity.  Intact: flooded members still float
    # (engine behaviour).  Damaged: flooded members lose buoyancy.
    buoyant_indices = [i for i, e in enumerate(element_list)
                       if bool(e.buoyant) and not (damaged and bool(e.flooded))]
    capacity_level = "warning" if damaged else "error"
    if not buoyant_indices:
        issues.append(_issue(
            capacity_level, "NO_BUOYANT_MEMBERS",
            "The model has no effective buoyant members"
            + (" in the damaged case (all buoyant members are flooded)." if damaged else "."),
            "elements"))

    total_buoyant_volume_m3 = 0.0
    for index in buoyant_indices:
        if not valid_geometry[index]:
            continue
        total_buoyant_volume_m3 += _member_volume_m3(element_list[index])

    rho_w_valid = _is_finite_number(getattr(params, "rho_w", None))
    if rho_w_valid and float(params.rho_w) <= 0.0:
        issues.append(_issue("error", "NON_POSITIVE_WATER_DENSITY",
                             "Water density must be greater than zero.", "params.rho_w"))
    if buoyant_indices and total_mass_t > 0.0 and rho_w_valid and float(params.rho_w) > 0.0:
        max_mass_t = float(params.rho_w) * total_buoyant_volume_m3
        if max_mass_t + capacity_tolerance_t < total_mass_t:
            issues.append(_issue(
                capacity_level, "FLOTATION_IMPOSSIBLE",
                f"Maximum theoretical supported mass is {max_mass_t:.6g} t, below the model mass of "
                f"{total_mass_t:.6g} t (annular volume, full submergence of every effective buoyant "
                "member; no equilibrium solved)."
                + (" Expected for a damage case." if damaged else ""), "model"))

    # Bounding box, CoG, openings, water depth
    valid_points = []
    for element in element_list:
        for point in (element.p1, element.p2):
            if _point_is_valid(point):
                valid_points.append(_point_tuple(point))

    # Lowest member endpoint: the "base" the openings and CoG are compared with.
    base_z_m = min((p[2] for p in valid_points), default=0.0)

    _validate_named_object_numbers(issues, opening_list, "opening", ("x", "y", "z"))
    for index, opening in enumerate(opening_list):
        if _is_finite_number(opening.z) and float(opening.z) < base_z_m - exact_tolerance_m:
            name = str(opening.name).strip() or f"row {index + 1}"
            issues.append(_issue(
                "warning", "OPENING_BELOW_BASE_DATUM",
                f"Opening elevation {float(opening.z):g} m is below the lowest member endpoint "
                f"({base_z_m:g} m).", f"opening '{name}'.z"))

    if valid_points:
        minimums = tuple(min(p[a] for p in valid_points) for a in range(3))
        maximums = tuple(max(p[a] for p in valid_points) for a in range(3))
        spans = tuple(maximums[a] - minimums[a] for a in range(3))

        valid_items = [w for w in weight_list
                       if all(_is_finite_number(v) for v in (w.mass_t, w.x, w.y, w.z))
                       and float(w.mass_t) > 0.0]
        valid_mass = sum(float(w.mass_t) for w in valid_items)
        if valid_mass > 0.0:
            cog = (sum(float(w.mass_t) * float(w.x) for w in valid_items) / valid_mass,
                   sum(float(w.mass_t) * float(w.y) for w in valid_items) / valid_mass,
                   sum(float(w.mass_t) * float(w.z) for w in valid_items) / valid_mass)
            for axis, axis_name in enumerate(("x", "y", "z")):
                margin = cog_margin_factor * max(spans[axis], exact_tolerance_m)
                lo, hi = minimums[axis] - margin, maximums[axis] + margin
                if cog[axis] < lo or cog[axis] > hi:
                    issues.append(_issue(
                        "warning", "COG_OUTSIDE_MEMBER_BOUNDS",
                        f"Combined CoG {axis_name}-coordinate is {cog[axis]:.6g} m, outside the member "
                        f"bounds [{minimums[axis]:.6g}, {maximums[axis]:.6g}] m by more than the "
                        "configured margin.", f"weights.CoG.{axis_name}"))

        if _is_finite_number(getattr(params, "water_depth_m", None)):
            depth = float(params.water_depth_m)
            height = spans[2]
            if depth < 0.0:
                issues.append(_issue("error", "NEGATIVE_WATER_DEPTH",
                                     "Water depth cannot be negative.", "params.water_depth_m"))
            elif 0.0 < depth < height:     # 0 means "seabed clearance not checked"
                issues.append(_issue(
                    "warning", "WATER_DEPTH_LESS_THAN_STRUCTURE_HEIGHT",
                    f"Water depth {depth:.6g} m is less than the structure height {height:.6g} m. "
                    "That can be correct for a floating tow (the draft is smaller than the height); "
                    "the set-down float check reports the seabed clearance. Confirm the datum.",
                    "params.water_depth_m"))

    # 16/17. Duplicate and near-coincident members (independent of endpoint order)
    def _usable(e: Element) -> bool:
        return (_point_is_valid(e.p1) and _point_is_valid(e.p2)
                and _is_finite_number(e.d_out) and _is_finite_number(e.d_in))

    for i in range(len(element_list)):
        first = element_list[i]
        if not _usable(first):
            continue
        for j in range(i + 1, len(element_list)):
            second = element_list[j]
            if not _usable(second):
                continue
            mismatch = _endpoint_pair_distance(first, second)
            exact = (mismatch <= exact_tolerance_m
                     and _diameters_match(first, second, exact_tolerance_m))
            near = (mismatch <= coincident_tolerance_m
                    and _diameters_match(first, second, coincident_tolerance_m))
            pair_where = f"elements '{first.name}' and '{second.name}'"
            if exact:
                issues.append(_issue("error", "DUPLICATE_MEMBER",
                                     "Members have the same endpoints and diameters "
                                     "(endpoint order ignored). Their buoyancy and wind area "
                                     "would be double counted.", pair_where))
            elif near:
                issues.append(_issue(
                    "warning", "NEAR_COINCIDENT_MEMBERS",
                    f"Members are coincident within the {coincident_tolerance_m:g} m tolerance. "
                    f"Maximum endpoint mismatch is {mismatch:.6g} m.", pair_where))
    return issues


def validation_summary(issues: Sequence[dict]) -> dict:
    errors = sum(1 for item in issues if item.get("level") == "error")
    warns = sum(1 for item in issues if item.get("level") == "warning")
    return {"passed_validation": errors == 0, "error_count": errors,
            "warning_count": warns, "issue_count": len(issues)}


def merge_with_legacy(errors: Sequence[str], warnings: Sequence[str],
                      issues: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Append validate_model() items to the legacy (errors, warnings) string lists.

    Items whose condition js.validate_inputs() already reports are skipped, so the
    app shows each problem once.  The legacy lists are not modified."""
    new_errors, new_warnings = list(errors), list(warnings)
    for item in issues:
        if item["code"] in OVERLAPS_VALIDATE_INPUTS:
            continue
        text = f"{item['where']}: {item['message']}"
        (new_errors if item["level"] == "error" else new_warnings).append(text)
    return new_errors, new_warnings
