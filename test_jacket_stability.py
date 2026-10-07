"""Validation tests for jacket_stability.py (run: pytest -q)."""
import math

import numpy as np
import pytest

import jacket_stability as js

from jacket_stability import (
    Criteria, Element, JacketModel, Opening, Params, WeightItem, analyse_curve,
    elements_from_rows, example_inputs, run_study, validate_inputs, G_ACC, KN_TO_MS,
)

RHO = 1.025


def spar_model(wind_kn=0.0, tow_kn=0.0, tow_pt=(0, 0, 47.0), cs=0.7, cd=0.7, **kw):
    """Vertical spar D=14, L=60, draft 45, KG=20 (the old spar app's defaults)."""
    d, draft, kg = 14.0, 45.0, 20.0
    w = RHO * math.pi * d**2 / 4.0 * draft
    el = [Element("spar", (0, 0, 0), (0, 0, 60), d, buoyant=True, exposed=True)]
    p = Params(wind_speed_kn=wind_kn, wind_cs=cs, cd_water=cd, tow_speed_kn=tow_kn,
               tow_point=tow_pt, **kw)
    return JacketModel(el, [WeightItem("all", w, 0, 0, kg)], [], p), w


def test_waterline_and_displacement():
    m, w = spar_model()
    up = m.upright()
    assert up["zw"] == pytest.approx(45.0, abs=1e-3)
    assert abs(up["trim_deg"]) < 1e-6


def test_gm_and_gz_match_wall_sided():
    m, _ = spar_model()
    bm = 14.0**2 / (16.0 * 45.0)
    gm = 22.5 + bm - 20.0
    assert m.gm(0.0) == pytest.approx(gm, rel=2e-3)
    for deg in (10, 25, 40):
        r = math.radians(deg)
        st = m.solve_trim(math.radians(90.0), r, loads=False)
        assert st["gz"] == pytest.approx(math.sin(r) * (gm + 0.5 * bm * math.tan(r) ** 2), rel=2e-3)


def test_gz_is_odd_about_upright():
    m, _ = spar_model()
    a = m.solve_trim(0.3, math.radians(15), loads=False)["gz"]
    b = m.solve_trim(0.3, -math.radians(15), loads=False)["gz"]
    assert a == pytest.approx(-b, rel=1e-3)


def test_horizontal_cylinder_gz_is_circle_centre_lever():
    """A floating horizontal circular cylinder: buoyancy always acts through the
    circle centre, so GZ = (z_axis - z_G) sin(phi) at any partial draft."""
    r, zg, zc = 5.0, 3.0, 5.0            # axis at z=5, G at z=3 (below axis)
    el = [Element("hull", (-50, 0, zc), (50, 0, zc), 2 * r)]
    vol = 0.4 * math.pi * r**2 * 100.0   # roughly 40% of full -> partial draft
    m = JacketModel(el, [WeightItem("w", vol * RHO, 0, 0, zg)], [], Params())
    for deg in (5, 20, 45):
        st = m.solve_trim(0.0, math.radians(deg), loads=False)   # heel about x (cylinder axis)
        assert st["gz"] == pytest.approx((zc - zg) * math.sin(math.radians(deg)), rel=2e-3)


def test_drag_and_tow_trim_match_hand_calc():
    """Reproduces the old spar app: Cd 0.70, 2.5 kn, tow point 2 m above SWL.
    Hand calc: drag = 373.9 kN, trim ~ 2.7 deg."""
    m, w = spar_model(tow_kn=2.5, tow_pt=(0, 0, 47.0))
    # psi = 0 -> H = +x, P = +y; tow along +x gives M_P, i.e. trim about P
    st = m.solve_trim(0.0, 0.0, loads=True)
    u = 2.5 * KN_TO_MS
    drag = 0.5 * RHO * 1000 * u**2 * 0.70 * (14.0 * 45.0) / 1000.0
    assert st["Dw"] == pytest.approx(drag, rel=2e-2)
    assert drag == pytest.approx(373.9, rel=2e-3)
    assert math.degrees(st["theta"]) == pytest.approx(2.72, abs=0.1)


def test_wind_force_and_heeling_arm_match_hand_calc():
    m, w = spar_model(wind_kn=50.0, cs=0.7)
    st = m.solve_trim(math.radians(90.0), 0.0, loads=True)    # wind toward +x
    v = 50.0 * KN_TO_MS
    area = 14.0 * 15.0                                         # 15 m above water
    f = 0.5 * 1.225 * v**2 * 0.7 * area / 1000.0
    arm = 52.5 - 22.5                                          # wind centre - drag centre
    assert st["Fw"] == pytest.approx(f, rel=3e-2)
    assert st["M_H"] / (w * G_ACC) == pytest.approx(f * arm / (w * G_ACC), rel=5e-2)


def test_insufficient_buoyancy_is_reported():
    el = [Element("t", (0, 0, 0), (0, 0, 5), 2.0)]
    m_ok = [WeightItem("w", 10000.0, 0, 0, 1.0)]
    errs, _ = validate_inputs(el, m_ok, [], Params())
    assert any("Insufficient buoyancy" in e for e in errs)
    with pytest.raises(ValueError):
        JacketModel(el, m_ok, [], Params()).upright()


def test_downflooding_angle_for_known_opening():
    """Opening 10 m above SWL at the hull edge (r = 7 m): the waterline passes
    through the axis, so it floods at atan(10/7).  Heel axis = x for beta = 90."""
    el = [Element("spar", (0, 0, 0), (0, 0, 60), 14.0)]
    w = RHO * math.pi * 49 * 45
    ops = [Opening("vent +y", 0.0, 7.0, 55.0), Opening("vent -y", 0.0, -7.0, 55.0)]
    m2 = JacketModel(el, [WeightItem("all", w, 0, 0, 20.0)], ops,
                     Params(wind_speed_kn=0.0, tow_speed_kn=0.0))
    sw = m2.sweep(90.0, np.arange(0.0, 60.1, 2.5))
    an = analyse_curve(sw, m2.gm(90.0), Criteria(cap_deg=60.0))
    assert an["theta_df"] == pytest.approx(math.degrees(math.atan(10.0 / 7.0)), abs=1.0)


def test_flooded_tank_loses_buoyancy_and_changes_heel():
    ex = example_inputs()
    els = elements_from_rows(ex["elements"])
    wts = [WeightItem(w["item"], w["mass_t"], w["x"], w["y"], w["z"]) for w in ex["weights"]]
    for e in els:
        if e.name == "Tank NE":
            e.flooded = True
    p = Params(wind_speed_kn=0.0, tow_speed_kn=0.0, tow_point=ex["tow_point"])
    intact = JacketModel(els, wts, [], p, damaged=False).solve_trim(0.0, 0.0, loads=False)
    damaged = JacketModel(els, wts, [], p, damaged=True).solve_trim(0.0, 0.0, loads=False)
    assert abs(intact["gz"]) < 1e-3 and abs(math.degrees(intact["theta"])) < 1e-3
    assert abs(damaged["gz"]) > 0.05 or abs(damaged["theta"]) > math.radians(0.05)


def test_example_study_runs_and_is_symmetric():
    ex = example_inputs()
    els = elements_from_rows(ex["elements"])
    wts = [WeightItem(w["item"], w["mass_t"], w["x"], w["y"], w["z"]) for w in ex["weights"]]
    ops = [Opening(o["name"], o["x"], o["y"], o["z"]) for o in ex["openings"]]
    mdl = JacketModel(els, wts, ops, Params(tow_point=ex["tow_point"]))
    res = run_study(mdl, [0.0, 90.0, 180.0, 270.0], np.arange(-10.0, 50.1, 5.0), Criteria())
    s = res["summary"]
    assert s["unconverged"] == 0
    assert s["gm_min"] > 0
    gm = [h["analysis"]["gm"] for h in res["heads"]]
    assert gm[1] == pytest.approx(gm[3], rel=1e-3)      # +/- beam winds equal for a symmetric jacket
    assert gm[0] == pytest.approx(gm[2], rel=1e-3)


# ---------------------------------------------------------------- technical data / drawing
def _example_model(damaged=False):
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    o = [js.Opening(**r) for r in ex["openings"]]
    return js.JacketModel(els, w, o, js.Params(tow_point=tuple(ex["tow_point"])), damaged=damaged), els, w, o


def test_hydrostatics_buoyancy_balances_weight_and_km_identity():
    m, *_ = _example_model()
    h = m.hydrostatics()
    assert abs(sum(r["buoyancy_t"] for r in h["members"]) - h["W"]) < 1e-3 * h["W"]
    assert abs(h["km_x"] - (h["kg"] + h["gm_x"])) < 1e-9
    assert abs(h["bm_x"] - (h["km_x"] - h["kb"])) < 1e-9
    assert h["waterplane_area"] > 0 and h["tpc"] > 0


def test_hydrostatics_damaged_tank_has_no_buoyancy():
    ex = js.example_inputs()
    rows = ex["elements"]
    rows[4]["flooded"] = True          # Tank NE
    els = js.elements_from_rows(rows)
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    m = js.JacketModel(els, w, [], js.Params(), damaged=True)
    h = m.hydrostatics()
    ne = next(r for r in h["members"] if r["name"] == "Tank NE")
    assert ne["buoyancy_t"] == 0.0 and "flooded" in ne["status"]


def test_attitude_state_matches_trim_solution():
    m, *_ = _example_model()
    h = m.hydrostatics()
    st = m.attitude_state(0.0, 0.0, h["trim_deg"])
    assert abs(st["zw"] - h["zw"]) < 1e-6
    assert abs(st["rot"].T @ st["rot"] - np.eye(3)).max() < 1e-12


def test_tube_mesh_is_closed_and_has_correct_volume_side_count():
    import viz
    v, t = viz.tube(np.zeros(3), np.array([0.0, 0.0, 10.0]), 2.0)
    assert t.max() < len(v) and t.min() >= 0
    v2, t2 = viz.tube(np.zeros(3), np.array([0.0, 0.0, 10.0]), 2.0, 0.5)
    assert len(v2) == 4 * viz.N_SIDES
    # every triangle vertex lies within the cylinder envelope
    r = np.hypot(v[:, 0], v[:, 1])
    assert r.max() <= 2.0 + 1e-9


def test_member_kind():
    e = lambda p2, b=False: js.Element("x", (0, 0, 0), p2, 1.0, buoyant=b)
    assert js.member_kind(e((0, 0, 10))) == "Leg / vertical"
    assert js.member_kind(e((10, 0, 0))) == "Horizontal brace"
    assert js.member_kind(e((10, 0, 10))) == "Diagonal brace"
    assert js.member_kind(e((0, 0, 10), True)) == "Buoyancy tank"


# ---------------------------------------------------------------- sensitivity studies
def test_free_tilt_matches_analytic_list_for_offset_cog():
    """A sideways CoG shift d on a wall-sided body lists it by about atan(d / GM)."""
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    for dy in (0.1, 0.4):
        w = [js.WeightItem(r["item"], r["mass_t"], r["x"], dy, r["z"]) for r in ex["weights"]]
        m = js.JacketModel(els, w, [], js.Params())
        want = math.degrees(math.atan(dy / m.gm(0.0)))
        assert m.free_tilt()["tilt_deg"] == pytest.approx(want, rel=0.02)


def test_free_tilt_symmetric_is_zero():
    m, *_ = _example_model()
    assert m.free_tilt()["tilt_deg"] == pytest.approx(0.0, abs=1e-6)


def test_losing_tanks_capsizes_then_sinks_for_placeholder_jacket():
    import sensitivity as sv
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    kw = dict(params=js.Params(), crit=js.Criteria(), betas=[0.0, 45.0], grid=[0.0, 10.0, 20.0, 30.0])
    base = sv.run_case(els, w, [], kw["params"], kw["crit"], kw["betas"], kw["grid"])
    assert base["status"] == sv.OK and base["tilt"] == pytest.approx(0.0, abs=1e-6)
    one = sv.run_case(sv.detach(els, ["Tank NE"]), w, [], kw["params"], kw["crit"], kw["betas"], kw["grid"])
    assert one["status"] == sv.CAPSIZES
    two = sv.run_case(sv.detach(els, ["Tank NE", "Tank SW"]), w, [], kw["params"], kw["crit"], kw["betas"], kw["grid"])
    assert two["status"] == sv.SINKS


def test_higher_tow_point_increases_static_heel_and_not_gm():
    import sensitivity as sv
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    p = js.Params(tow_point=tuple(ex["tow_point"]))
    cases = [dict(name="mid", x=20.0, y=0.0, z=14.0), dict(name="top", x=20.0, y=0.0, z=28.0),
             dict(name="high", x=20.0, y=0.0, z=37.5)]
    out = sv.tow_study(els, w, [], p, js.Criteria(), [0.0, 45.0, 90.0], [-10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0], cases)
    heels = [c["heel_max"] for c in out]
    assert heels[0] < heels[1] < heels[2]
    assert max(c["gm_min"] for c in out) - min(c["gm_min"] for c in out) < 1e-6


def test_tank_study_counts_outcomes():
    import sensitivity as sv
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    rows = sv.tank_study(els, w, [], js.Params(), js.Criteria(), [0.0, 45.0], [0.0, 10.0, 20.0, 30.0])
    assert [r["k"] for r in rows] == [4, 3, 2, 1]
    assert rows[0]["counts"][sv.OK] == 1
    assert rows[1]["counts"][sv.CAPSIZES] == 4          # lose any one corner tank
    assert rows[2]["counts"][sv.SINKS] == 6             # all pairs sink


def test_negative_gm_pair_loss_is_not_reported_as_stable():
    """Symmetric but with GM < 0: the upright root is unstable, so it must capsize, not 'float at 0 deg'."""
    import copy
    import sensitivity as sv
    ex = js.example_inputs()
    rows = copy.deepcopy(ex["elements"])
    s = 20.0
    for i, (x, y) in enumerate([(s, 0), (0, s), (-s, 0), (0, -s)]):
        rows.append(dict(name=f"Tank M{i+1}", x1=x, y1=y, z1=0.0, x2=x, y2=y, z2=28.0, d_out=7.0, d_in=0.0,
                         buoyant=True, exposed=True, flooded=False))
    els = js.elements_from_rows(rows)
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    c = sv.run_case(sv.detach(els, ["Tank NE", "Tank SW"]), w, [], js.Params(), js.Criteria(), [0.0, 45.0],
                    [0.0, 10.0, 20.0, 30.0])
    assert c["status"] == sv.CAPSIZES


def test_listed_case_is_flagged_not_scored():
    import sensitivity as sv
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], 0.4, r["z"]) for r in ex["weights"]]
    c = sv.run_case(els, w, [], js.Params(), js.Criteria(), [0.0, 45.0], [0.0, 10.0, 20.0, 30.0])
    assert c["status"] == sv.OK and c["listed"] and not c["passed"]
    assert c["tilt"] == pytest.approx(6.5, abs=0.2)


# ---------------------------------------------------------------- line loads (tow legs, pull-in lines)
def _lines_model(lines, **kw):
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    p = js.Params(wind_speed_kn=0.0, tow_speed_kn=kw.pop("tow_speed", 0.0), lines=tuple(lines), **kw)
    return js.JacketModel(els, w, [], p)


def test_vertical_pull_lowers_draft_by_waterplane_balance():
    m0 = _lines_model([js.Line("none", (0, 0, 40), 0.0, 0.0, "fixed", 0.0)])
    m1 = _lines_model([js.Line("lift", (0, 0, 40), 0.0, 90.0, "fixed", 2000.0)])
    h = m0.hydrostatics()
    s0 = m0._eval(math.pi / 2, 0.0, 0.0, True, None)
    s1 = m1._eval(math.pi / 2, 0.0, 0.0, True, None)
    expect = 2000.0 / (1.025 * js.G_ACC) / h["waterplane_area"]        # tanks are wall-sided here
    assert s0["draft"] - s1["draft"] == pytest.approx(expect, rel=1e-4)


def test_vertical_pull_heeling_moment_is_force_times_lever_from_cob():
    # upward 1000 kN at y = +15 m with the body level: moment about the x heel axis = 1000 * 15 kN m
    m = _lines_model([js.Line("lift", (0, 15, 40), 0.0, 90.0, "fixed", 1000.0)])
    st = m._eval(0.0, 0.0, 0.0, True, None)        # psi = 0: heel axis H = +x, P = +y
    assert st["M_H"] == pytest.approx(1000.0 * 15.0, rel=1e-6)
    assert st["M_P"] == pytest.approx(0.0, abs=1e-6)
    assert st["Fz"] == pytest.approx(1000.0, rel=1e-9)


def test_opposing_lines_cancel():
    lines = [js.Line("a", (20, 0, 20), 0.0, 0.0, "fixed", 500.0), js.Line("b", (20, 0, 20), 180.0, 0.0, "fixed", 500.0)]
    m = _lines_model(lines)
    sw = m.sweep(0.0, [-5.0, 0.0, 5.0])
    assert np.nanmax(np.abs(sw["hl"])) < 1e-9 and np.nanmax(sw["T"]) < 1e-9


def test_pull_in_line_reduces_net_pull():
    tow = js.Line("tow", (20, 0, 20), 0.0, 0.0, "fixed", 600.0)
    pull = js.Line("pull-in", (-20, 0, 25), 180.0, 0.0, "fixed", 200.0)
    m = _lines_model([tow, pull])
    st = m._eval(math.pi / 2, 0.0, 0.0, True, None)
    assert st["T"] == pytest.approx(400.0, rel=1e-9)


def test_bridle_legs_carry_the_same_resultant_as_one_tow_line():
    single = _lines_model([js.Line("tow", (20, 0, 20), 0.0, 0.0, "tow", 0.0, 1.0)], tow_speed=2.5)
    bridle = _lines_model([js.Line("P", (20, -5, 20), 20.0, 0.0, "tow", 0.0, 1.0),
                           js.Line("S", (20, 5, 20), -20.0, 0.0, "tow", 0.0, 1.0)], tow_speed=2.5)
    a = single._eval(math.pi / 2, 0.0, 0.0, True, None)
    b = bridle._eval(math.pi / 2, 0.0, 0.0, True, None)
    assert a["T"] > 100.0
    assert b["T"] == pytest.approx(a["T"], rel=1e-9)


def test_legacy_params_and_equivalent_line_agree():
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    legacy = js.Params(tow_point=(20, 0, 20), tow_heading_deg=25.0)
    lined = js.Params(lines=(js.Line("Tow", (20, 0, 20), 25.0, 0.0, "tow", 0.0, 1.0),))
    r1 = js.JacketModel(els, w, [], legacy).sweep(45.0, [0.0, 10.0, 20.0])
    r2 = js.JacketModel(els, w, [], lined).sweep(45.0, [0.0, 10.0, 20.0])
    assert np.allclose(r1["gz"], r2["gz"]) and np.allclose(r1["hl"], r2["hl"])


def test_inclined_off_centre_line_satisfies_3d_moment_balance():
    """Independent check: sum the 3-D moments of weight, buoyancy, the line force and the water reaction about G."""
    line = js.Line("pull", (15.0, 10.0, 45.0), 40.0, 30.0, "fixed", 800.0)
    m = _lines_model([line])
    psi, phi = math.radians(90.0 + 20.0), math.radians(8.0)
    st = m.solve_trim(psi, phi, loads=True)
    assert st["converged"]
    rot, h_ax, p_ax = js.frame(psi, phi, st["theta"])
    ez = np.array([0.0, 0.0, 1.0])
    g = js.G_ACC
    tau, eps = math.radians(line.heading_deg), math.radians(line.elevation_deg)
    f_line = line.tension_kn * np.array([math.cos(eps) * math.cos(tau), math.cos(eps) * math.sin(tau), math.sin(eps)])
    delta = m.W - f_line[2] / g                                   # buoyancy mass = weight less the upward pull
    zw, b, u, h_low = m._hydro(rot, st["zw"], delta / m.p.rho_w)
    gs = rot @ m.G
    f_react = np.array([-f_line[0], -f_line[1], 0.0])
    moment = (delta * g * np.cross(rot @ (b - m.G), ez)
              + np.cross(rot @ (np.array(line.point) - m.G), f_line)
              + np.cross(np.array([0.0, 0.0, st["z_lr"] - gs[2]]), f_react))
    wg = m.W * g
    assert moment @ p_ax == pytest.approx(0.0, abs=2e-3 * wg)               # trim equilibrium
    assert moment @ h_ax == pytest.approx(wg * (st["M_H"] / wg - st["gz"]), rel=2e-3, abs=2e-3 * wg)


# ---------------------------------------------------------------------------
# Synthetic six-over-four-leg jacket and the set-down float check
# ---------------------------------------------------------------------------
def _syn_model(extra_w=0.0, depth=92.4, tide=0.0):
    ex = js.synthetic_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    if extra_w:
        g = sum(x.mass_t * x.z for x in w) / sum(x.mass_t for x in w)
        w.append(js.WeightItem("ballast", extra_w, 0.0, 0.0, g))
    op = [js.Opening(**o) for o in ex["openings"]]
    p = js.Params(tow_point=ex["tow_point"], tow_heading_deg=ex["tow_heading_deg"], water_depth_m=depth, tide_m=tide)
    return js.JacketModel(els, w, op, p)


def test_synthetic_jacket_geometry_and_float():
    ex = js.synthetic_inputs()
    legs = [r for r in ex["elements"] if r["name"].startswith(("Leg", "Outrigger leg"))]
    assert len(legs) == 6
    tall = [r for r in legs if r["z2"] == 105.0]
    short = [r for r in legs if r["z2"] == 30.0]
    assert len(tall) == 4 and len(short) == 2
    # the two outrigger legs stand outside the square formed by the four main legs
    ymax_main = max(r["y1"] for r in tall)
    assert all(r["y1"] > ymax_main + 1e-9 for r in short)
    assert sum(1 for r in ex["elements"] if r["buoyant"]) == 4
    m = _syn_model()
    assert abs(m.W - 700.0) < 1e-9 and 41.0 < m.G[2] < 43.5 and abs(m.G[1]) < 1e-9 and abs(m.G[0]) < 1e-9
    f = js.float_check(m, js.Criteria(clear_min_m=5.0, emerged_min_m=5.0))
    assert f["stable"] and f["gm_min"] > 5.0 and abs(f["trim_deg"]) < 1e-6


def test_float_check_clearance_and_emergence_arithmetic():
    m = _syn_model()
    f = js.float_check(m, js.Criteria())
    assert abs(f["clearance"] - (92.4 - (f["zw"] - f["z_base"]))) < 1e-9
    # emerged tank length = tank top (90 m) - waterline, from the exact volume cut
    assert abs(f["emerged_min"] - (90.0 - f["zw"])) < 1e-6
    # tide shrinks the clearance by exactly the tide
    assert abs(js.float_check(_syn_model(tide=2.0), js.Criteria())["clearance"] - (f["clearance"] + 2.0)) < 1e-9
    # no depth -> no clearance
    assert js.float_check(_syn_model(depth=0.0), js.Criteria())["clearance"] is None


def test_ballast_headroom_reaches_the_limit_exactly():
    crit = js.Criteria(clear_min_m=5.0, emerged_min_m=5.0)
    f0 = js.float_check(_syn_model(), crit)
    assert f0["governing_limit"] == "tank emergence" and f0["ballast_headroom_t"] > 0
    f1 = js.float_check(_syn_model(extra_w=f0["ballast_headroom_t"]), crit)
    assert abs(f1["emerged_min"] - 5.0) < 0.01          # ballast added at the CoG hits the 5 m limit
    f2 = js.float_check(_syn_model(extra_w=f0["ballast_headroom_t"] + 40.0), crit)
    assert f2["pass_emerged"] is False and f2["ballast_headroom_t"] < 0


def test_float_check_off_by_default_for_legacy_inputs():
    ex = js.example_inputs()
    els = js.elements_from_rows(ex["elements"])
    w = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in ex["weights"]]
    f = js.float_check(js.JacketModel(els, w, [], js.Params()), js.Criteria())
    assert f["pass_clearance"] is None and f["pass_emerged"] is None and f["ballast_headroom_t"] is None


def test_kg_kb_base_are_measured_from_lowest_point_of_structure():
    m = _syn_model()
    h = m.hydrostatics()
    assert abs(h["kg_base"] - (m.G[2] - h["bbox_lo"][2])) < 1e-9
    assert abs(h["kb_base"] - (h["B_body"][2] - h["bbox_lo"][2])) < 1e-9
    assert h["kb_base"] > h["kg_base"] - 1e-9 or h["gm_x"] > 0   # tanks high: B above G


def test_criteria_checks_need_their_switch_and_old_files_still_work():
    base = dict(c_gm_min=1.0, c_heel_max_deg=15.0, c_ratio_min=1.3, c_cap_deg=40.0)
    off = js.criteria_from_widgets(dict(base, c_clear_min_m=5.0, c_clear_on=False, c_emerged_min_m=3.0, c_emerged_on=True))
    assert off.clear_min_m == 0.0 and off.emerged_min_m == 3.0 and off.df_min_deg == 0.0
    old = js.criteria_from_widgets(dict(base, c_clear_min_m=5.0, c_emerged_min_m=0.0, c_df_min_deg=0.0))   # no switches saved
    assert old.clear_min_m == 5.0 and old.emerged_min_m == 0.0
