import math

import pytest

import towplan as tp


def test_drag_area_from_tubes():
    src = [dict(group="leg", od_mm=1000, length_m=100), dict(group="brace", od_mm=500, length_m=200),
           dict(group="tank", od_mm=5000, length_m=10), dict(group="leg")]
    assert tp.drag_area_m2(src) == pytest.approx(100 + 100)
    assert tp.drag_area_m2(src, 0.5) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        tp.drag_area_m2(src, 1.5)


def test_calm_water_drag_hand_calc():
    # A = 100 m2, Cd = 1, 1 kn = 0.514444 m/s: 0.5 * 1.025 * 100 * 0.514444^2 = 13.563 kN
    d = tp.drag_components(100.0, 1.0, 0.0, 0.0)
    assert d["along_kn"] == pytest.approx(0.5 * 1.025 * 100 * 0.514444**2, rel=1e-9)
    assert d["side_kn"] == pytest.approx(0.0, abs=1e-12)


def test_head_following_and_cross_current():
    base = lambda v: 0.5 * 1.025 * 100 * (v * tp.KN) ** 2
    assert tp.drag_components(100, 1.0, 1.0, 180.0)["along_kn"] == pytest.approx(base(2.0))
    assert tp.drag_components(100, 2.0, 1.0, 0.0)["along_kn"] == pytest.approx(base(1.0))
    assert tp.drag_components(100, 0.5, 1.0, 0.0)["along_kn"] == 0.0           # current pushes faster than the tow
    x = tp.drag_components(100, 1.0, 1.0, 90.0)
    v = math.sqrt(2.0)
    assert x["along_kn"] == pytest.approx(base(v) / v, rel=1e-9) and abs(x["side_kn"]) == pytest.approx(base(v) / v, rel=1e-9)
    assert tp.drag_components(100, 1.0, 1.0, 270.0)["side_kn"] == pytest.approx(-x["side_kn"])


def test_plan_times_and_utilisation():
    legs = [dict(from_m=1000, to_m=500, speed_kn=1.0), dict(from_m=500, to_m=100, speed_kn=0.5)]
    p = tp.plan(legs, 100.0, 0.0, 180.0, margin=1.5, tug_count=2, tug_bollard_t=20.0, tug_eff=0.8)
    assert p["rows"][0]["hours"] == pytest.approx(500 / (1.0 * tp.KN) / 3600)
    assert p["total_hours"] == pytest.approx(500 / (tp.KN * 3600) + 400 / (0.5 * tp.KN * 3600))
    assert p["available_kn"] == pytest.approx(2 * 20 * tp.G * 0.8)
    r0 = p["rows"][0]
    assert r0["need_kn"] == pytest.approx(1.5 * 0.5 * 1.025 * 100 * (1.0 * tp.KN) ** 2)
    assert r0["util_pct"] == pytest.approx(100 * r0["need_kn"] / p["available_kn"])
    assert p["hold_kn"] == 0.0                       # no current: nothing to hold against


def test_hold_position_against_current():
    p = tp.plan([dict(from_m=200, to_m=100, speed_kn=0.5)], 100.0, 1.0, 180.0, margin=1.0, tug_bollard_t=10.0)
    assert p["hold_kn"] == pytest.approx(0.5 * 1.025 * 100 * tp.KN**2)


def test_plan_rejects_bad_legs():
    with pytest.raises(ValueError):
        tp.plan([dict(from_m=100, to_m=200, speed_kn=1.0)], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        tp.plan([dict(from_m=200, to_m=100, speed_kn=0.0)], 100.0, 0.0, 0.0)
