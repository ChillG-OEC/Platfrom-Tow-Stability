import math

import pytest

import budget as bd

RHO = 1.025


def test_tube_volume_hand_calc():
    # 1.0 m OD, 10 m long: pi/4 * 1 * 10 = 7.854 m3
    assert bd.tube_outer_volume_m3(1000.0, 10.0) == pytest.approx(math.pi / 4 * 10)
    assert bd.source_volume_m3(dict(name="a", volume_m3=12.5, od_mm=1000, length_m=10)) == 12.5
    assert bd.source_volume_m3(dict(name="a")) == 0.0


def test_required_tank_volume_fixed_steel():
    # W = 1000 t, other buoyancy 500 t, reserve 10 %: need 1100 t -> tanks 600 t -> 600 / 1.025 = 585.37 m3
    src = [dict(name="legs", group="leg", volume_m3=500.0 / RHO)]
    b = bd.budget(1000.0, src, {"legs": 1.0}, reserve_min_pct=10.0)
    assert b["required_t"] == pytest.approx(1100.0)
    assert b["capacity_t"] == pytest.approx(500.0)
    assert b["tank_volume_needed_m3"] == pytest.approx(600.0 / RHO)
    assert b["shortfall_t"] == pytest.approx(600.0) and not b["meets_reserve"]


def test_required_tank_volume_with_steel_per_m3():
    # tank steel 100 t already in W = 1000 t; k = 0.2 t/m3: V = (1.1*900 - 500) / (1.025 - 1.1*0.2) = 608.7 m3
    src = [dict(name="legs", group="leg", volume_m3=500.0 / RHO)]
    b = bd.budget(1000.0, src, {"legs": 1.0}, reserve_min_pct=10.0, tank_steel_in_weight_t=100.0,
                  tank_steel_t_per_m3=0.2)
    v = b["tank_volume_needed_m3"]
    assert v == pytest.approx(490.0 / 0.805, rel=1e-9)
    # closing the loop: with that tank volume the reserve is exactly 10 %
    w = 900.0 + 0.2 * v
    assert (500.0 + RHO * v) / w == pytest.approx(1.10, rel=1e-9)


def test_infeasible_when_steel_too_heavy():
    src = [dict(name="legs", group="leg", volume_m3=100.0)]
    b = bd.budget(1000.0, src, {"legs": 1.0}, reserve_min_pct=10.0, tank_steel_in_weight_t=50.0,
                  tank_steel_t_per_m3=1.0)          # 1.1 t per m3 of steel > 1.025 t per m3 of water
    assert b["tank_volume_needed_m3"] is None and not b["feasible"]


def test_shares_and_groups():
    src = [dict(name="legs", group="leg", volume_m3=100.0), dict(name="braces", group="brace", volume_m3=40.0),
           dict(name="tank", group="tank", volume_m3=300.0)]
    b = bd.budget(500.0, src, {"legs": 1.0, "braces": 0.5, "tank": 1.0}, reserve_min_pct=10.0)
    assert b["by_group_t"]["brace"] == pytest.approx(20.0 * RHO)
    assert b["capacity_t"] == pytest.approx(420.0 * RHO)
    assert b["tank_capacity_t"] == pytest.approx(300.0 * RHO)
    assert b["reserve_pct"] == pytest.approx(100 * (420 * RHO - 500) / 500)
    assert b["meets_reserve"] is False        # 430.5 t against 550 t required


def test_inputs_are_checked():
    src = [dict(name="a", group="leg", volume_m3=10.0)]
    with pytest.raises(ValueError):
        bd.budget(0.0, src, {})
    with pytest.raises(ValueError):
        bd.budget(10.0, src, {"a": 1.5})
    with pytest.raises(ValueError):
        bd.budget(10.0, src, {"a": 1.0}, tank_steel_in_weight_t=20.0)


def test_loss_cases_remove_one_source_at_a_time():
    src = [dict(name="a", group="leg", volume_m3=400.0), dict(name="b", group="tank", volume_m3=300.0)]
    rows = bd.loss_cases(600.0, src, {"a": 1.0, "b": 1.0}, reserve_min_pct=5.0)
    assert [r["lost"] for r in rows] == ["a", "b"]
    ra = rows[0]                                       # lose a: capacity = 300 * 1.025 = 307.5 t < 630 t
    assert ra["capacity_t"] == pytest.approx(300 * RHO) and ra["meets"] is False
    rb = rows[1]                                       # lose b: 400 * 1.025 = 410 t < 630 t
    assert rb["capacity_t"] == pytest.approx(400 * RHO)
