import math

import pytest

from jacket_stability import KN_TO_MS
from test_jacket_stability import RHO, spar_model

CD, D, DRAFT = 0.7, 14.0, 45.0


def _drag(v):                       # spar drag at relative water speed v [m/s], kN
    return 0.5 * RHO * 1000 * v**2 * CD * D * DRAFT / 1000.0


def _state(tow_kn=2.5, **kw):
    m, _ = spar_model(tow_kn=tow_kn, tow_pt=(0, 0, 47.0), **kw)
    return m.solve_trim(0.0, 0.0, loads=True)


def test_zero_current_unchanged():
    assert _state()["Dw"] == pytest.approx(_drag(2.5 * KN_TO_MS), rel=2e-2)
    assert _state()["Dc"] == pytest.approx(0.0, abs=1e-9)


def test_head_current_raises_drag_with_speed_squared():
    u, c = 2.5 * KN_TO_MS, 1.0 * KN_TO_MS
    s = _state(current_speed_kn=1.0, current_dir_deg=180.0)       # current flows against the tow
    assert s["Dw"] == pytest.approx(_drag(u + c), rel=2e-2)
    assert abs(s["Dc"]) < 1e-6


def test_following_current_lowers_drag():
    u, c = 2.5 * KN_TO_MS, 1.0 * KN_TO_MS
    assert _state(current_speed_kn=1.0, current_dir_deg=0.0)["Dw"] == pytest.approx(_drag(u - c), rel=2e-2)


def test_strong_following_current_never_gives_negative_pull():
    s = _state(current_speed_kn=4.0, current_dir_deg=0.0)         # pushes faster than the tow speed
    assert s["Dw"] == 0.0


def test_cross_current_splits_into_track_and_side_drag():
    u, c = 2.5 * KN_TO_MS, 1.0 * KN_TO_MS
    v = math.hypot(u, c)
    s = _state(current_speed_kn=1.0, current_dir_deg=90.0)
    assert s["Dw"] == pytest.approx(_drag(v) * u / v, rel=2e-2)
    assert abs(s["Dc"]) == pytest.approx(_drag(v) * c / v, rel=2e-2)
    other = _state(current_speed_kn=1.0, current_dir_deg=270.0)
    assert other["Dc"] == pytest.approx(-s["Dc"], rel=1e-6)


def test_water_reference_ignores_current():
    a = _state(current_speed_kn=2.0, current_dir_deg=180.0, tow_speed_ref="water")
    assert a["Dw"] == pytest.approx(_state()["Dw"], rel=1e-9)


def test_side_drag_adds_a_moment_about_the_tow_point_height():
    s0 = _state()
    s1 = _state(current_speed_kn=1.5, current_dir_deg=90.0)
    assert abs(s1["M_H"]) > abs(s0["M_H"]) + 1.0                   # side force at 47 m acts about the drag centre at 22.5 m
