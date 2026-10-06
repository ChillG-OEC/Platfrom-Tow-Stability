"""Tests for model_validation.validate_model (task C2).

Hand calculations are written out in comments; expected values are NOT taken
from the production code.
"""
import copy
import math

import pytest

import jacket_stability as js
import model_validation as mv

RHO = 1.025  # t/m3, the Params default


def tank(name="T1", d_out=3.0, d_in=0.0, z1=60.0, z2=90.0, x=0.0, **kw):
    return js.Element(name, (x, 0.0, z1), (x, 0.0, z2), d_out, d_in, **kw)


def base_model(mass=100.0):
    """One 3 m x 30 m tank; weight at its mid-height."""
    els = [tank()]
    wts = [js.WeightItem("W", mass, 0.0, 0.0, 75.0)]
    return els, wts, [], js.Params(), js.Criteria()


def synth():
    """The synthetic jacket as engine objects (synthetic_inputs() returns table rows)."""
    d = js.synthetic_inputs()
    els = js.elements_from_rows(d["elements"])
    wts = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in d["weights"]]
    ops = [js.Opening(r["name"], r["x"], r["y"], r["z"]) for r in d["openings"]]
    prm = js.Params(tow_point=d["tow_point"], tow_heading_deg=d["tow_heading_deg"],
                    water_depth_m=d["water_depth_m"], tide_m=d["tide_m"])
    cr = js.Criteria(clear_min_m=d["clear_min_m"], emerged_min_m=d["emerged_min_m"])
    return els, wts, ops, prm, cr


def codes(issues, level=None):
    return [i["code"] for i in issues if level is None or i["level"] == level]


# 1. Valid nominal model --------------------------------------------------
def test_valid_nominal_model_has_no_issues():
    assert mv.validate_model(*base_model()) == []


# 2. Hand-calculated capacity ---------------------------------------------
# V = pi/4 * 3^2 * 30 = 67.5*pi = 212.0575 m3 ; capacity = 1.025 * V = 217.3589 t.
CAP_T = 1.025 * math.pi / 4.0 * 9.0 * 30.0


def test_capacity_hand_calculation_and_message():
    assert CAP_T == pytest.approx(217.3589, abs=1e-3)
    els, wts, ops, prm, cr = base_model(mass=CAP_T + 1.0)
    issues = mv.validate_model(els, wts, ops, prm, cr)
    msg = [i for i in issues if i["code"] == "FLOTATION_IMPOSSIBLE"][0]["message"]
    assert "217.359" in msg            # 6 significant figures of 217.3589


def test_annular_volume_convention():
    # d_out 3, d_in 1: V = pi/4 * (9 - 1) * 30 = 60*pi = 188.4956 m3 -> 193.2080 t
    cap = 1.025 * math.pi / 4.0 * 8.0 * 30.0
    assert cap == pytest.approx(193.2080, abs=1e-3)
    els, wts, ops, prm, cr = base_model(mass=cap + 0.5)
    els = [tank(d_in=1.0)]
    assert "FLOTATION_IMPOSSIBLE" in codes(mv.validate_model(els, wts, ops, prm, cr), "error")
    wts2 = [js.WeightItem("W", cap - 0.5, 0.0, 0.0, 75.0)]
    assert "FLOTATION_IMPOSSIBLE" not in codes(mv.validate_model(els, wts2, ops, prm, cr))


# 3. Capacity boundary -----------------------------------------------------
def test_capacity_boundary_equal_mass_passes_and_just_over_fails():
    els, _, ops, prm, cr = base_model()
    at_limit = [js.WeightItem("W", CAP_T, 0.0, 0.0, 75.0)]
    assert "FLOTATION_IMPOSSIBLE" not in codes(mv.validate_model(els, at_limit, ops, prm, cr))
    over = [js.WeightItem("W", CAP_T + 1e-3, 0.0, 0.0, 75.0)]
    assert "FLOTATION_IMPOSSIBLE" in codes(mv.validate_model(els, over, ops, prm, cr), "error")


# 4. Impossible flotation --------------------------------------------------
def test_impossible_flotation_is_an_error():
    issues = mv.validate_model(*base_model(mass=1000.0))
    assert "FLOTATION_IMPOSSIBLE" in codes(issues, "error")
    assert mv.validation_summary(issues)["passed_validation"] is False


# 5. Flooded buoyant member ------------------------------------------------
def test_flooded_member_floats_when_intact_but_not_when_damaged():
    els = [tank("A", flooded=True)]
    wts = [js.WeightItem("W", 100.0, 0.0, 0.0, 75.0)]
    intact = mv.validate_model(els, wts, [], js.Params(), js.Criteria())
    assert "NO_BUOYANT_MEMBERS" not in codes(intact)          # engine: intact still floats
    damaged = mv.validate_model(els, wts, [], js.Params(), js.Criteria(), damaged=True)
    assert "NO_BUOYANT_MEMBERS" in codes(damaged, "warning")  # expected physical outcome
    assert not codes(damaged, "error")


def test_engine_agrees_with_intact_flooded_treatment():
    els = [tank("A", flooded=True)]
    wts = [js.WeightItem("W", 100.0, 0.0, 0.0, 75.0)]
    js.JacketModel(els, wts, [], js.Params())                       # builds: flooded floats when intact
    with pytest.raises(ValueError):
        js.JacketModel(els, wts, [], js.Params(), damaged=True)     # no buoyant elements


# 6. Non-buoyant member ----------------------------------------------------
def test_non_buoyant_member_gives_no_capacity():
    els = [tank(buoyant=False)]
    wts = [js.WeightItem("W", 10.0, 0.0, 0.0, 75.0)]
    assert "NO_BUOYANT_MEMBERS" in codes(mv.validate_model(els, wts, [], js.Params(), js.Criteria()), "error")
    # a non-buoyant member must not add capacity next to a small buoyant one
    els2 = [tank("B", d_out=1.0, z1=0.0, z2=1.0), tank("A", buoyant=False)]
    # capacity = 1.025 * pi/4 * 1 * 1 = 0.805 t
    wts2 = [js.WeightItem("W", 1.0, 0.0, 0.0, 0.5)]
    assert "FLOTATION_IMPOSSIBLE" in codes(mv.validate_model(els2, wts2, [], js.Params(), js.Criteria()))


# 7. Invalid annulus -------------------------------------------------------
@pytest.mark.parametrize("d_in", [3.0, 3.5])
def test_invalid_annulus(d_in):
    els, wts, ops, prm, cr = base_model()
    els = [tank(d_in=d_in)]
    assert "INVALID_ANNULUS_DIAMETERS" in codes(mv.validate_model(els, wts, ops, prm, cr), "error")


def test_negative_inside_and_non_positive_outside_diameter():
    els, wts, ops, prm, cr = base_model()
    assert "NEGATIVE_INSIDE_DIAMETER" in codes(mv.validate_model([tank(d_in=-0.1)], wts, ops, prm, cr))
    assert "NON_POSITIVE_OUTSIDE_DIAMETER" in codes(mv.validate_model([tank(d_out=0.0)], wts, ops, prm, cr))


# 8. Zero-length member ----------------------------------------------------
def test_zero_length_member_and_tolerance_boundary():
    els, wts, ops, prm, cr = base_model()
    z = [js.Element("Z", (0, 0, 5), (0, 0, 5), 1.0)]
    assert "ZERO_LENGTH_MEMBER" in codes(mv.validate_model(z + els, wts, ops, prm, cr), "error")
    just_over = [js.Element("Z", (0, 0, 5), (0, 0, 5.0 + 1e-6), 1.0)]
    assert "ZERO_LENGTH_MEMBER" not in codes(mv.validate_model(just_over + els, wts, ops, prm, cr))


# 9. Duplicate names -------------------------------------------------------
def test_duplicate_names():
    els, wts, ops, prm, cr = base_model()
    els = [tank("A"), tank("A", x=10.0)]
    issues = mv.validate_model(els, wts, ops, prm, cr)
    assert "DUPLICATE_ELEMENT_NAME" in codes(issues, "error")
    assert "rows 1, 2" in [i for i in issues if i["code"] == "DUPLICATE_ELEMENT_NAME"][0]["message"]


# 10. Reversed-endpoint duplicate members ---------------------------------
def test_reversed_endpoint_duplicate():
    a = js.Element("A", (0, 0, 0), (0, 0, 10), 1.0)
    b = js.Element("B", (0, 0, 10), (0, 0, 0), 1.0)       # same member, endpoints swapped
    wts = [js.WeightItem("W", 1.0, 0, 0, 5)]
    issues = mv.validate_model([a, b], wts, [], js.Params(), js.Criteria())
    assert "DUPLICATE_MEMBER" in codes(issues, "error")


def test_same_endpoints_different_diameter_is_not_a_duplicate():
    a = js.Element("A", (0, 0, 0), (0, 0, 10), 1.0)
    b = js.Element("B", (0, 0, 0), (0, 0, 10), 2.0)
    wts = [js.WeightItem("W", 1.0, 0, 0, 5)]
    issues = mv.validate_model([a, b], wts, [], js.Params(), js.Criteria())
    assert "DUPLICATE_MEMBER" not in codes(issues) and "NEAR_COINCIDENT_MEMBERS" not in codes(issues)


# 11. Near-coincident members ---------------------------------------------
def test_near_coincident_members_boundary():
    a = js.Element("A", (0, 0, 0), (0, 0, 10), 1.0)
    near = js.Element("B", (0.0005, 0, 0), (0.0005, 0, 10), 1.0)    # 0.5 mm < 1 mm
    far = js.Element("C", (0.005, 0, 0), (0.005, 0, 10), 1.0)       # 5 mm > 1 mm
    wts = [js.WeightItem("W", 1.0, 0, 0, 5)]
    assert "NEAR_COINCIDENT_MEMBERS" in codes(mv.validate_model([a, near], wts, [], js.Params(), js.Criteria()), "warning")
    assert "NEAR_COINCIDENT_MEMBERS" not in codes(mv.validate_model([a, far], wts, [], js.Params(), js.Criteria()))


# 12. NaN and infinity -----------------------------------------------------
def test_nan_and_infinity_are_reported_without_crashing():
    els = [js.Element("A", (0, 0, 0), (0, 0, float("nan")), 1.0),
           js.Element("B", (0, 0, 0), (0, 0, 10), float("inf"))]
    wts = [js.WeightItem("W", float("nan"), 0, 0, 5)]
    prm = js.Params(water_depth_m=float("inf"))
    cr = js.Criteria(gm_min=float("nan"))
    issues = mv.validate_model(els, wts, [js.Opening("O", 0, 0, float("nan"))], prm, cr)
    wheres = " ".join(i["where"] for i in issues)
    for needle in ("element 'A'.p2", "element 'B'.d_out", "weight 'W'.mass_t",
                   "params.water_depth_m", "crit.gm_min", "opening 'O'.z"):
        assert needle in wheres
    assert set(codes(issues)) >= {"INVALID_POINT", "NON_FINITE_NUMBER"}


# 13. Zero total mass ------------------------------------------------------
def test_zero_total_mass():
    els, _, ops, prm, cr = base_model()
    assert "NON_POSITIVE_TOTAL_MASS" in codes(mv.validate_model(els, [js.WeightItem("W", 0.0, 0, 0, 75)], ops, prm, cr), "error")
    assert "NON_POSITIVE_TOTAL_MASS" in codes(mv.validate_model(els, [], ops, prm, cr), "error")


# 14. Negative individual mass --------------------------------------------
def test_negative_item_mass_is_a_warning_consistent_with_validate_inputs():
    els, _, ops, prm, cr = base_model()
    wts = [js.WeightItem("W", 100.0, 0, 0, 75), js.WeightItem("Deduct", -5.0, 0, 0, 75)]
    issues = mv.validate_model(els, wts, ops, prm, cr)
    assert "NEGATIVE_WEIGHT_ITEM_MASS" in codes(issues, "warning")
    assert not codes(issues, "error")
    _, legacy_warn = js.validate_inputs(els, wts, ops, prm)
    assert any("negative mass" in w for w in legacy_warn)       # existing tool also only warns


# 15. Opening below base ---------------------------------------------------
def test_opening_below_lowest_member_point():
    els, wts, _, prm, cr = base_model()          # lowest endpoint z = 60
    low = [js.Opening("Vent", 0, 0, 59.0)]
    ok = [js.Opening("Vent", 0, 0, 60.0)]
    assert "OPENING_BELOW_BASE_DATUM" in codes(mv.validate_model(els, wts, low, prm, cr), "warning")
    assert "OPENING_BELOW_BASE_DATUM" not in codes(mv.validate_model(els, wts, ok, prm, cr))


# 16. Water depth below structure height ----------------------------------
def test_water_depth_vs_structure_height():
    els, wts, ops, _, cr = base_model()          # height 30 m
    shallow = js.Params(water_depth_m=20.0)
    assert "WATER_DEPTH_LESS_THAN_STRUCTURE_HEIGHT" in codes(mv.validate_model(els, wts, ops, shallow, cr), "warning")
    for d in (30.0, 50.0, 0.0):                 # equal, deeper, and 0 = "not checked" in the engine
        assert "WATER_DEPTH_LESS_THAN_STRUCTURE_HEIGHT" not in codes(
            mv.validate_model(els, wts, ops, js.Params(water_depth_m=d), cr))
    assert "NEGATIVE_WATER_DEPTH" in codes(mv.validate_model(els, wts, ops, js.Params(water_depth_m=-1.0), cr), "error")


# 17. Large diameter -------------------------------------------------------
def test_large_diameter_warning_boundary():
    _, wts, ops, prm, cr = base_model(mass=1.0)
    assert "DIAMETER_LOOKS_TOO_LARGE" in codes(mv.validate_model([tank(d_out=20.5)], wts, ops, prm, cr), "warning")
    assert "DIAMETER_LOOKS_TOO_LARGE" not in codes(mv.validate_model([tank(d_out=20.0)], wts, ops, prm, cr))


# 18. Large length ---------------------------------------------------------
def test_large_length_warning_boundary():
    wts = [js.WeightItem("W", 1.0, 0, 0, 0)]
    long_ = [js.Element("L", (0, 0, 0), (0, 0, 501.0), 1.0)]
    ok = [js.Element("L", (0, 0, 0), (0, 0, 500.0), 1.0)]
    assert "MEMBER_LENGTH_LOOKS_TOO_LARGE" in codes(mv.validate_model(long_, wts, [], js.Params(), js.Criteria()), "warning")
    assert "MEMBER_LENGTH_LOOKS_TOO_LARGE" not in codes(mv.validate_model(ok, wts, [], js.Params(), js.Criteria()))


# 19. CoG margin -----------------------------------------------------------
def test_cog_margin_warning():
    els, _, ops, prm, cr = base_model()          # z bounds 60..90, span 30, margin factor 1.0
    # Limits: 30 .. 120.  CoG z = 130 is outside; 110 is inside.
    far = [js.WeightItem("W", 100.0, 0, 0, 130.0)]
    near = [js.WeightItem("W", 100.0, 0, 0, 110.0)]
    assert "COG_OUTSIDE_MEMBER_BOUNDS" in codes(mv.validate_model(els, far, ops, prm, cr), "warning")
    assert "COG_OUTSIDE_MEMBER_BOUNDS" not in codes(mv.validate_model(els, near, ops, prm, cr))
    # the margin factor is a named argument
    assert "COG_OUTSIDE_MEMBER_BOUNDS" in codes(mv.validate_model(els, near, ops, prm, cr, cog_margin_factor=0.5))


# 20. Inputs are not mutated -----------------------------------------------
def test_inputs_not_mutated():
    els, wts, ops, prm, cr = synth()
    before = copy.deepcopy((els, wts, ops, prm, cr))
    mv.validate_model(els, wts, ops, prm, cr)
    mv.validate_model(els, wts, ops, prm, cr, damaged=True)
    assert (els, wts, ops, prm, cr) == before


# 21. validation_summary ---------------------------------------------------
def test_validation_summary_counts():
    issues = [{"level": "error"}, {"level": "warning"}, {"level": "warning"}]
    s = mv.validation_summary(issues)
    assert s == {"passed_validation": False, "error_count": 1, "warning_count": 2, "issue_count": 3}
    assert mv.validation_summary([])["passed_validation"] is True


# 22. Existing synthetic baseline still works ------------------------------
def test_synthetic_baseline_has_no_errors_and_still_floats():
    els, wts, ops, prm, cr = synth()
    issues = mv.validate_model(els, wts, ops, prm, cr)
    assert not codes(issues, "error"), issues
    # capacity = 4 tanks * 1.025 * pi/4 * 9 * 30 = 869.4 t (hand calc: 4 * 217.3589)
    assert 4 * CAP_T == pytest.approx(869.436, abs=1e-2)
    assert "FLOTATION_IMPOSSIBLE" not in codes(issues)
    # known expected warning: depth 92.4 m is below the 105 m structure height
    assert "WATER_DEPTH_LESS_THAN_STRUCTURE_HEIGHT" in codes(issues, "warning")
    m = js.JacketModel(els, wts, ops, prm)
    assert js.float_check(m, cr)["zw"] == pytest.approx(84.15, abs=0.05)


def test_damaged_synthetic_with_all_tanks_flooded_is_a_warning_not_an_error():
    els0, wts, ops, prm, cr = synth()
    els = [js.Element(**{**e.__dict__, "flooded": True}) if e.buoyant else e for e in els0]
    issues = mv.validate_model(els, wts, ops, prm, cr, damaged=True)
    assert "NO_BUOYANT_MEMBERS" in codes(issues, "warning")
    assert not codes(issues, "error")


# 23. Legacy merge and argument checks -------------------------------------
def test_merge_with_legacy_skips_overlaps_and_keeps_inputs():
    els, wts, ops, prm, cr = base_model(mass=1000.0)      # impossible flotation + water depth fine
    wts = wts + [js.WeightItem("Neg", -1.0, 0, 0, 75)]
    errs, warns = js.validate_inputs(els, wts, ops, prm)
    issues = mv.validate_model(els, wts, ops, prm, cr)
    e2, w2 = mv.merge_with_legacy(errs, warns, issues)
    assert errs == js.validate_inputs(els, wts, ops, prm)[0]            # inputs unchanged
    assert sum("Insufficient buoyancy" in e or "Maximum theoretical" in e for e in e2) == 1   # shown once
    assert any("Deduct" in w or "negative" in w.lower() for w in w2)


def test_bad_arguments_raise():
    args = base_model()
    for kw in ({"exact_tolerance_m": -1.0}, {"coincident_tolerance_m": -1.0}, {"diameter_warning_m": 0.0},
               {"length_warning_m": 0.0}, {"cog_margin_factor": -1.0}, {"capacity_tolerance_t": -1.0}):
        with pytest.raises(ValueError):
            mv.validate_model(*args, **kw)


def test_numpy_points_are_accepted():
    import numpy as np
    e = js.Element("A", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0]), 1.0)
    wts = [js.WeightItem("W", 1.0, 0, 0, 5)]
    assert "INVALID_POINT" not in codes(mv.validate_model([e], wts, [], js.Params(), js.Criteria()))
