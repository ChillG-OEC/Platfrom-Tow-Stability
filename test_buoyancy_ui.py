"""The buoyancy tab (7) loads a case file and shows budget, environment and module results."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent


def _run(path):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    at.text_input(key="bu_path").set_value(str(path)).run()
    return at


def _budget_case():
    return {
        "name": "unit budget case", "structure": {"elements": [], "weights": []},
        "reserve_min_pct": 10.0,
        "budget": {
            "weight_cases": [{"name": "dry", "weight_t": 1000.0, "tank_steel_t": 100.0}],
            "sources": [{"name": "legs", "group": "leg", "volume_m3": 500.0 / 1.025, "share_default": 1.0},
                        {"name": "tank", "group": "tank", "volume_m3": 0.0, "share_default": 1.0}],
            "tank_steel_in_weight_t": 100.0},
        "environment": {"operating_limits": {"Hs limit": "1.0 m"},
                        "monthly": [{"Month": "April", "Hs mean [m]": 0.6}], "notes": ["note"]}}


def test_budget_and_environment_render(tmp_path):
    p = tmp_path / "b.case.json"
    p.write_text(json.dumps(_budget_case()))
    at = _run(p)
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Weight (factored)"] == "1000.0 t"
    assert labels["Buoyancy capacity"] == "500.0 t"
    assert labels["Tank volume needed in total"].startswith("585")        # 600 t / 1.025 = 585 m3
    assert any("Hs limit" in m.label for m in at.metric)
    assert any("No member geometry" in i.value for i in at.info)


def test_modules_section_on_template():
    at = _run(ROOT / "docs" / "case_template.json")
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Draft", "GM", "Reserve"} <= labels


def test_bad_file_reports_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    at = _run(p)
    assert not at.exception
    assert any("Could not read" in e.value for e in at.error)


def test_current_preset_fills_the_tow_load_input(tmp_path):
    case = _budget_case()
    case["environment"]["current"] = {"presets": [{"name": "P1", "speed_ms": 0.514444, "source": "test"},
                                                  {"name": "1-yr", "speed_ms": 1.028888, "source": "test"}], "default": "P1"}
    f = tmp_path / "c.case.json"
    f.write_text(json.dumps(case))
    at = _run(f)
    assert not at.exception
    at.selectbox(key="bu_cur_sel").select("1-yr").run()
    at.button(key="bu_cur_use").click().run()
    assert not at.exception
    assert abs(at.session_state["p_current_speed_kn"] - 2.0) < 1e-3


def test_tow_plan_tab_uses_the_case_file_area(tmp_path):
    case = _budget_case()
    case["budget"]["sources"][0].update(od_mm=1000.0, length_m=100.0)
    f = tmp_path / "c.case.json"
    f.write_text(json.dumps(case))
    at = _run(f)
    assert not at.exception
    assert any("Time for the approach" in m.label for m in at.metric)
    t = [m for m in at.metric if m.label == "Tube drag area from the case file"][0]
    assert t.value.replace(",", "").startswith("100")


def test_tow_plan_tab_works_without_a_case_file():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert any("Time for the approach" in m.label for m in at.metric)
