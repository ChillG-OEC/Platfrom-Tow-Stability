"""Tests for the buoyancy-module layer (buoyancy.py) and case files (case_io.py)."""
import json
import math

import pytest

import buoyancy as bu
import case_io
import jacket_stability as js

RHO = 1.025
A1 = math.pi / 4.0          # area of a 1.0 m diameter circle


def leg(name, x=0.0, y=0.0, z1=0.0, z2=100.0, d=1.0):
    return js.Element(name, (x, y, z1), (x, y, z2), d, buoyant=False)


def params():
    return js.Params(wind_speed_kn=0.0, tow_speed_kn=0.0)


def test_clip_splits_member_at_diaphragm():
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    m = bu.BuoyancyModule("lower leg", members=("L1",), z_lo=0.0, z_hi=30.0)
    els, wts, _, damaged = bu.assemble(s, [m], {"lower leg": "sealed"})
    assert not damaged and len(els) == 2
    sealed = [e for e in els if e.buoyant][0]
    rest = [e for e in els if not e.buoyant][0]
    assert sealed.p1[2] == pytest.approx(0.0) and sealed.p2[2] == pytest.approx(30.0)
    assert rest.p1[2] == pytest.approx(30.0) and rest.p2[2] == pytest.approx(100.0)
    assert sealed.tank is False and sealed.d_in == 0.0
    assert len(wts) == 1                       # sealing adds no weight


def test_middle_section_gives_three_pieces():
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    m = bu.BuoyancyModule("mid", members=("L1",), z_lo=20.0, z_hi=60.0)
    els, *_ = bu.assemble(s, [m], {"mid": "sealed"})
    assert [round(e.p1[2], 6) for e in els] == [0.0, 20.0, 60.0]
    assert [e.buoyant for e in els] == [False, True, False]


def test_sealed_leg_floats_at_hand_calculated_draft():
    # one leg d = 1 m, 100 m long, sealed throughout, 60 t: draft = 60 / (1.025 * pi/4)
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    m = bu.BuoyancyModule("leg", members=("L1",))
    r = bu.evaluate(s, [m], {"leg": "sealed"}, params(), js.Criteria())
    assert r["floats"]
    assert r["draft_m"] == pytest.approx(60.0 / (RHO * A1), rel=1e-6)          # 74.53 m
    assert r["capacity_t"] == pytest.approx(RHO * A1 * 100.0, rel=1e-9)         # 80.50 t
    assert r["reserve_pct"] == pytest.approx(100.0 * (RHO * A1 * 100.0 - 60.0) / 60.0, rel=1e-6)


def test_module_off_means_nothing_to_float_on():
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    r = bu.evaluate(s, [bu.BuoyancyModule("leg", members=("L1",))], {"leg": "off"}, params(), js.Criteria())
    assert not r["floats"] and "buoyan" in r["error"].lower()


def test_overlapping_modules_are_rejected():
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    a = bu.BuoyancyModule("a", members=("L1",), z_lo=0.0, z_hi=50.0)
    b = bu.BuoyancyModule("b", members=("L1",), z_lo=40.0, z_hi=90.0)
    with pytest.raises(ValueError, match="same length"):
        bu.assemble(s, [a, b], {"a": "sealed", "b": "sealed"})
    # touching ranges are fine
    c = bu.BuoyancyModule("c", members=("L1",), z_lo=50.0, z_hi=90.0)
    els, *_ = bu.assemble(s, [a, c], {"a": "sealed", "c": "sealed"})
    assert sum(e.buoyant for e in els) == 2


def test_unknown_member_and_bad_state():
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    with pytest.raises(ValueError, match="unknown structure members"):
        bu.assemble(s, [bu.BuoyancyModule("x", members=("nope",))], {"x": "sealed"})
    m = bu.BuoyancyModule("m", members=("L1",))
    with pytest.raises(ValueError, match="must be one of"):
        bu.assemble(s, [m], {"m": "floaty"})
    with pytest.raises(ValueError, match="unknown module"):
        bu.assemble(s, [m], {"zzz": "sealed"})


def test_tank_steel_follows_the_module():
    tank = js.Element("T1", (4, 0, 70), (4, 0, 110), 3.0)
    tk = bu.BuoyancyModule("tank", elements=(tank,), weights=(js.WeightItem("tank steel", 20.0, 4, 0, 90.0),))
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 60.0, 0, 0, 50.0)])
    lg = bu.BuoyancyModule("leg", members=("L1",))
    on = bu.evaluate(s, [lg, tk], {"leg": "sealed", "tank": "sealed"}, params(), js.Criteria())
    off = bu.evaluate(s, [lg, tk], {"leg": "sealed", "tank": "off"}, params(), js.Criteria())
    assert on["weight_t"] == pytest.approx(80.0) and off["weight_t"] == pytest.approx(60.0)


def test_tank_emerged_length_ignores_sealed_legs():
    # leg top 100 m, tank top 110 m: the leg has the shorter emerged length, but only the tank counts
    tank = js.Element("T1", (4, 0, 70), (4, 0, 110), 3.0)
    tk = bu.BuoyancyModule("tank", elements=(tank,))
    # hand calc: leg full to 70 m = 54.98 m3 of 97.56 m3 needed; above 70 m both pierce (0.7854 + 7.0686 m2)
    a_tank = math.pi * 1.5 ** 2
    zw_hand = 70.0 + (100.0 / RHO - A1 * 70.0) / (A1 + a_tank)
    # put G under B so the structure floats level: B_x = tank volume * 4 m / total volume
    xg = 4.0 * a_tank * (zw_hand - 70.0) / (100.0 / RHO)
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 100.0, xg, 0, 10.0)])
    lg = bu.BuoyancyModule("leg", members=("L1",))
    els, wts, ops, dmg = bu.assemble(s, [lg, tk], {"leg": "sealed", "tank": "sealed"})
    mdl = js.JacketModel(els, wts, ops, params())
    fc = js.float_check(mdl, js.Criteria())
    assert fc["zw"] == pytest.approx(zw_hand, rel=1e-6)
    assert len(fc["tanks"]) == 1 and fc["tanks"][0]["name"] == "T1"
    assert fc["emerged_min"] == pytest.approx(110.0 - zw_hand, rel=1e-6)       # 34.58 m, not the leg's 24.58 m


def test_default_element_is_still_a_tank():
    assert js.Element("e", (0, 0, 0), (0, 0, 1), 1.0).tank is True
    assert js.member_kind(js.Element("e", (0, 0, 0), (0, 0, 10), 1.0)) == "Buoyancy tank"
    assert js.member_kind(js.Element("e", (0, 0, 0), (0, 0, 10), 1.0, tank=False)) == "Leg / vertical"
    assert js.elements_from_rows([dict(name="a", x1=0, y1=0, z1=0, x2=0, y2=0, z2=5, d_out=1, tank=False)])[0].tank is False


def test_damaged_module_loses_volume_but_keeps_weight():
    legs = [leg("L1", x=-3), leg("L2", x=3), leg("L3", y=4)]
    wt = [js.WeightItem("w", 100.0, 0, 1.3, 50.0)]
    s = bu.Structure(legs, wt)
    mods = [bu.BuoyancyModule(f"leg{i}", members=(f"L{i}",)) for i in (1, 2, 3)]
    base = {m.name: "sealed" for m in mods}
    ok = bu.evaluate(s, mods, base, params(), js.Criteria())
    assert ok["capacity_t"] == pytest.approx(3 * RHO * A1 * 100.0, rel=1e-9)
    rows = bu.damage_cases(s, mods, base, params(), js.Criteria())
    assert len(rows) == 3 and all(r["damaged"] for r in rows)
    assert all(r["capacity_t"] == pytest.approx(2 * RHO * A1 * 100.0, rel=1e-9) for r in rows)
    assert all(r["weight_t"] == pytest.approx(100.0) for r in rows)


def test_sweep_counts_cases_and_limits():
    s = bu.Structure([leg("L1"), leg("L2", x=5)], [js.WeightItem("w", 50.0, 2.5, 0, 50.0)])
    mods = [bu.BuoyancyModule("a", members=("L1",)), bu.BuoyancyModule("b", members=("L2",)),
            bu.BuoyancyModule("c", members=("L1",), z_lo=0.0, z_hi=0.0 + 1e-12)]
    # module c has an empty clip (zero length) so it adds nothing: still enumerated
    rows = bu.sweep(s, mods[:2], params(), js.Criteria())
    assert len(rows) == 4
    assert sum(r["floats"] for r in rows) == 3                                    # none-on cannot float
    with pytest.raises(ValueError, match="limit"):
        bu.sweep(s, mods[:2], params(), js.Criteria(), max_cases=3)
    assert len(bu.combinations(mods[:2], fixed={"a": "sealed"})) == 2


def test_ballast_adds_weight_at_given_height():
    tank = js.Element("T1", (0, 0, 10), (0, 0, 40), 3.0)
    tk = bu.BuoyancyModule("tank", elements=(tank,), ballast_t=15.0, ballast_z=12.0)
    s = bu.Structure([leg("L1")], [js.WeightItem("w", 50.0, 0, 0, 50.0)])
    _, wts, *_ = bu.assemble(s, [tk], {"tank": "sealed"})
    b = [w for w in wts if "ballast" in w.name][0]
    assert (b.mass_t, b.z) == (15.0, 12.0)
    _, wts2, *_ = bu.assemble(s, [tk], {"tank": "sealed"}, ballast={"tank": 0.0})
    assert not [w for w in wts2 if "ballast" in w.name]


def _case_dict(dz=92.4):
    return {
        "name": "unit", "datum": {"z_offset_m": dz},
        "structure": {
            "elements": [dict(name="L1", x1=0, y1=0, z1=-92.4, x2=0, y2=0, z2=8.5, d_out=1.0)],
            "weights": [dict(item="jacket", mass_t=30.0, x=0, y=0, z=-50.0)],
            "openings": [dict(name="vent", x=0, y=0, z=8.5)]},
        "modules": [dict(name="leg", members=["L1"], z_lo=-92.4, z_hi=-40.0)],
        "states": {"leg": "sealed"},
        "params": {"water_depth_m": 91.3, "tow_point": [0, 0, -50.0]},
        "criteria": {"gm_min": 0.5, "clear_min_m": 5.0},
        "reserve_min_pct": 10.0}


def test_case_loader_applies_datum_offset(tmp_path):
    c = case_io.load_case(_case_dict())
    e = c["structure"].elements[0]
    assert e.p1[2] == pytest.approx(0.0) and e.p2[2] == pytest.approx(100.9)
    assert c["structure"].weights[0].z == pytest.approx(42.4)
    assert c["structure"].openings[0].z == pytest.approx(100.9)
    assert c["modules"][0].z_lo == pytest.approx(0.0) and c["modules"][0].z_hi == pytest.approx(52.4)
    assert c["params"].tow_point[2] == pytest.approx(42.4)
    assert c["criteria"].gm_min == 0.5 and c["reserve_min_pct"] == 10.0
    r = bu.evaluate(c["structure"], c["modules"], c["states"], c["params"], c["criteria"],
                    reserve_min_pct=c["reserve_min_pct"])
    assert r["floats"] and r["pass_reserve"] is not None
    # seal only 0-52.4 m of a 1 m leg: capacity = pi/4 * 52.4 * 1.025
    assert r["capacity_t"] == pytest.approx(RHO * A1 * 52.4, rel=1e-9)


def test_case_roundtrip(tmp_path):
    c = case_io.load_case(_case_dict())
    p = tmp_path / "x.case.json"
    case_io.dump_case(c, p)
    c2 = case_io.load_case(p)
    assert c2["structure"].elements[0].p2 == pytest.approx(c["structure"].elements[0].p2)
    assert c2["modules"][0].z_hi == pytest.approx(c["modules"][0].z_hi)
    assert c2["params"].water_depth_m == 91.3 and c2["states"] == {"leg": "sealed"}


def test_case_loader_reports_problems(tmp_path):
    with pytest.raises(case_io.CaseFileError, match="no 'structure'"):
        case_io.load_case({"name": "x"})
    bad = _case_dict()
    bad["structure"]["elements"][0]["d_out"] = "wide"
    bad["params"]["nonsense"] = 1
    with pytest.raises(case_io.CaseFileError) as ei:
        case_io.load_case(bad)
    msg = str(ei.value)
    assert "d_out" in msg and "unknown fields" in msg
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(case_io.CaseFileError, match="cannot read"):
        case_io.load_case(p)
    with pytest.raises(case_io.CaseFileError, match="cannot read"):
        case_io.load_case(tmp_path / "missing.json")


def test_size_tanks_finds_the_passing_size():
    legs = [js.Element(f"L{i}", (x, y, 0), (x, y, 100), 1.0, 0.0, False) for i, (x, y) in enumerate([(0, 0), (6, 0), (0, 6), (6, 6)])]
    s = bu.Structure(legs, [js.WeightItem("w", 150.0, 3, 3, 10.0)])

    def tank(n, x):
        return bu.BuoyancyModule(n, elements=(js.Element(n + "c", (x, 3, 10), (x, 3, 30), 2.0, 0.0, True),),
                                 weights=(js.WeightItem(n + "s", 5.0, x, 3, 20.0),))
    mods = [bu.BuoyancyModule("legs", members=tuple(f"L{i}" for i in range(4)), z_lo=0, z_hi=40), tank("T1", -3), tank("T2", 9)]
    states = {m.name: "sealed" for m in mods}
    out = bu.size_tanks(s, mods, states, js.Params(wind_speed_kn=0, tow_speed_kn=0), js.Criteria(), ["T1", "T2"],
                        reserve_min_pct=10.0, f_min=0.5, f_max=2.5, steps=9)
    rows = out["rows"]
    assert len(rows) == 9 and rows[0]["diameter_m"] == pytest.approx(1.0) and rows[-1]["diameter_m"] == pytest.approx(5.0)
    assert not rows[0]["intact_passed"]                      # too small: reserve below 10 %
    assert out["f_intact"] == pytest.approx(0.75) and out["f_all"] == pytest.approx(1.25)
    assert rows[2]["intact_passed"] and rows[2]["damage_passed"] == 2 and rows[2]["damage_total"] == 3
    assert not rows[-1]["intact_passed"]                      # too big: GM lost
    big = bu.scale_tanks(mods, ["T1"], 2.0)
    assert big[1].elements[0].d_out == pytest.approx(4.0) and big[1].weights[0].mass_t == pytest.approx(20.0)
    assert big[2].elements[0].d_out == pytest.approx(2.0)     # T2 untouched
