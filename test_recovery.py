"""Recovery measures: the search finds the passing boundary, verifies it, and reports what cannot be fixed."""
import copy

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import jacket_stability as js
import recovery as rc

SNAP = dict(
    widgets=dict(p_tow_speed_kn=4.0, p_tow_mode="auto", p_tow_manual_kn=0.0, p_tow_z=40.0, p_tow_heading_deg=270.0, p_fsm=0.0),
    fsc_m=0.0,
    elements=[dict(name="L", x1=0, y1=0, z1=0, x2=0, y2=0, z2=60, d_out=2.0)],
    weights=[dict(item="Ballast A", mass_t=100.0, x=0, y=0, z=5.0)],
    lines=[dict(name="Pull-in 1", type="Pull-in", x=0, y=0, z=30.0, heading_deg=0.0, elevation_deg=0.0, tension_kn=800.0, share=1.0)])


def _fake_passes(snap, parts_fn, betas, grid, damaged, extra=None):
    """Passes when speed <= 3, the pull-in line <= 500 kN, the connection <= z 20, the ballast <= 60 t, or heading 240."""
    w = snap["widgets"]
    ln = snap["lines"][0]
    ballast = snap["weights"][0]["mass_t"] + sum(e[1] for e in (extra or []))
    ok = (w["p_tow_speed_kn"] <= 3.0 and ln["tension_kn"] <= 500.0) or w["p_tow_z"] <= 20.0 \
        or w["p_tow_heading_deg"] == 240.0 or ballast <= 60.0 or (extra and sum(e[1] for e in extra) >= 25.0)
    return ok, dict(gm_min=5.0, heel_max=8.0, ratio_min=1.6)


def test_boundary_returns_the_passing_value_nearest_the_failing_one():
    v = rc._boundary(lambda x: x <= 3.0, bad=4.0, limit=0.0, tol=0.01)
    assert 2.98 < v <= 3.0
    assert rc._boundary(lambda x: False, 4.0, 0.0, 0.1) is None


def test_propose_finds_each_lever_and_verifies(monkeypatch):
    monkeypatch.setattr(rc, "_passes", _fake_passes)
    ms = rc.propose(copy.deepcopy(SNAP), None, np.array([0.0, 90.0]), np.array([0.0]), False,
                    tanks=[dict(name="Top tank", x=0, y=0, z=50, cap_t=100.0)])
    kinds = {m.kind for m in ms}
    assert {"connection", "heading", "ballast", "deballast"} <= kinds
    by = {m.kind: m for m in ms}
    assert all(m.verified for m in ms if not m.action.startswith("Reducing"))
    assert any(m.kind == "tow speed" and not m.verified for m in ms)     # speed alone cannot fix it while the line is high
    assert "z = 20.0" in by["connection"].action or "z = 19." in by["connection"].action
    assert "240" in by["heading"].action
    assert 58.0 <= float(by["deballast"].action.split(" to ")[1].split(" t")[0]) <= 60.0
    assert 24.9 < float(by["ballast"].action.split("least ")[1].split(" t")[0]) <= 26.0
    assert not any(m.kind == "stop" for m in ms)             # something works, so no "do not proceed"


def test_nothing_works_gives_do_not_proceed(monkeypatch):
    monkeypatch.setattr(rc, "_passes", lambda *a, **k: (False, {}))
    ms = rc.propose(copy.deepcopy(SNAP), None, np.array([0.0]), np.array([0.0]), False)
    assert ms[-1].kind == "stop" and "do not proceed" in ms[-1].action
    assert not any(m.verified for m in ms if m.kind != "stop")


def test_setdown_tide_and_emergence_measures(monkeypatch):
    monkeypatch.setattr(rc, "_passes", lambda *a, **k: (False, {}))
    F = dict(pass_clearance=False, pass_emerged=False, depth_total=50.0, water_depth=50.0, draft_total=47.0, tide=0.0,
             ballast_headroom_t=-120.0, passed=False)
    ms = rc.propose(copy.deepcopy(SNAP), None, np.array([0.0]), np.array([0.0]), False, float_check=F,
                    crit_float=js.Criteria(clear_min_m=5.0))
    by = {m.kind: m for m in ms}
    assert "at least 2.00 m above LAT" in by["tide"].action          # 5 - (50 - 47) = 2 m
    assert "at least 120 t" in by["emergence"].action
    assert any(m.kind == "stop" for m in ms)


def test_tables_from_assembled_round_trip():
    el = js.Element("a", (0, 0, 0), (0, 0, 5), 1.0, 0.0, True, True, False, False)
    t = rc.tables_from_assembled([el], [js.WeightItem("w", 3.0, 1, 2, 3)], [js.Opening("o", 4, 5, 6)])
    back = js.elements_from_rows(t["elements"])[0]
    assert back == el and t["weights"][0]["mass_t"] == 3.0 and t["openings"][0]["z"] == 6


def test_results_tab_offers_recovery_for_a_failing_case():
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).parent / "app.py"), default_timeout=300)
    at.run()
    at.session_state["p_wind_speed_kn"] = 140.0
    at.run()
    at.sidebar.button[[b.label for b in at.sidebar.button].index("▶ Run analysis")].click().run()
    assert any("do NOT meet" in e.value for e in at.error)
    [b for b in at.button if b.key == "rec_go"][0].click().run()
    assert not at.exception
    ms = at.session_state["recovery"]["measures"]
    assert ms and ms[-1].kind == "stop"
