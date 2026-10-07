"""Set-down sequence: ballast from the sealed volume, saturation, tide, and the tab itself."""
import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

import jacket_stability as js
import setdown as sd

ROOT = Path(__file__).parent


def _column_jacket():
    """One sealed vertical column, 40 m long, D = 4 m, weight 300 t at mid-height: floats upright."""
    els = [js.Element("col", (0, 0, 0), (0, 0, 40), 4.0, 0.0, True, True, False, True)]
    wts = [js.WeightItem("w", 300.0, 0, 0, 10.0)]
    p = js.Params(tow_point=(0, 0, 10), tow_heading_deg=0.0)
    return els, wts, [], p


def test_ballast_matches_displacement_and_tide_shifts_it_linearly():
    els, wts, ops, p = _column_jacket()
    area = np.pi * 4.0 ** 2 / 4.0
    r = sd.stage(els, wts, ops, p, depth_m=40.0, tide_m=0.0, clearance_m=20.0, ballast_z=5.0)
    # waterline 20 m above the lowest point: displaced mass = rho * area * 20
    assert abs(r["draft_m"] - 20.0) < 1e-6
    assert abs(r["ballast_t"] - (1.025 * area * 20.0 - 300.0)) < 1e-6
    r2 = sd.stage(els, wts, ops, p, depth_m=40.0, tide_m=1.0, clearance_m=20.0, ballast_z=5.0, with_gm=False)
    assert abs((r2["ballast_t"] - r["ballast_t"]) - 1.025 * area * 1.0) < 1e-6


def test_all_buoyancy_used_is_reported_as_saturated():
    els, wts, ops, p = _column_jacket()
    r = sd.stage(els, wts, ops, p, depth_m=45.0, tide_m=0.0, clearance_m=0.0, ballast_z=5.0)   # waterline above the top
    assert r["saturated"] and r["gm_min_m"] is None
    assert abs(r["ballast_t"] - (1.025 * np.pi * 4.0 * 40.0 - 300.0)) < 1e-6


def test_sequence_steps_and_tidal_clearance():
    els, wts, ops, p = _column_jacket()
    res = sd.sequence(els, wts, ops, p, 35.0, {"LAT": 0.0, "MSL": 1.0, "HAT": 2.0}, [10.0, 5.0], "MSL", 5.0)
    assert res["steps"][0]["added_from_previous_t"] is None
    area = np.pi * 4.0 ** 2 / 4.0
    assert abs(res["steps"][1]["added_from_previous_t"] - 1.025 * area * 5.0) < 1e-6   # 5 m deeper
    lat = [a for a in res["across"] if a["stage"] == 5.0 and a["tide"] == "LAT"][0]
    assert lat["clearance_m"] == 4.0                       # ballast set at MSL: LAT is 1 m lower
    assert abs(lat["extra_ballast_to_hold_t"] + 1.025 * area * 1.0) < 1e-6


def test_set_down_tab_runs_on_a_sent_case(tmp_path):
    from test_buoyancy_ui import _budget_case, _run
    case = _budget_case()
    case["structure"] = {"elements": [dict(name="L", x1=0, y1=0, z1=0, x2=0, y2=0, z2=40, d_out=0.5)],
                         "weights": [dict(item="w", mass_t=300.0, x=0, y=0, z=10.0)]}
    case["modules"] = [dict(name="Tank", note="", elements=[dict(name="t", x1=0, y1=0, z1=0, x2=0, y2=0, z2=40, d_out=4.0)],
                            weights=[])]
    p = tmp_path / "s.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    at.run()
    assert any("demo jacket" in i.value for i in at.info)       # tab 9 refuses the demo jacket
    [b for b in at.button if b.key and b.key.startswith("bu_send_")][0].click().run()
    at.session_state["p_depth_m"] = 45.0
    at.run()
    assert not at.exception
    assert not any("Set-down tab failed" in e.value for e in at.error)
    assert any("Ballast to add between stages" in s.value for s in at.subheader)
