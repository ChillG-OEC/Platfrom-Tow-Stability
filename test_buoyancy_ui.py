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


def test_platform_structure_section_shows_member_totals(tmp_path):
    case = _budget_case()
    case["structure"]["elements"] = [
        dict(name="Leg a", x1=0, y1=0, z1=0, x2=0, y2=0, z2=10, d_out=1.0, d_in=0.9),
        dict(name="Brace a", x1=0, y1=0, z1=0, x2=5, y2=0, z2=0, d_out=0.4, d_in=0.0)]
    p = tmp_path / "s.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Members"] == "2"
    assert labels["Total length"] == "15 m"
    assert labels["Drag area (OD x L)"] == "12 m2"          # 1.0*10 + 0.4*5


def test_tank_sizing_tab_runs_on_the_template():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    at.text_input(key="bu_path").set_value(str(ROOT / "docs" / "case_template.json")).run()
    assert not at.exception
    go = [b for b in at.button if b.key and b.key.startswith("bu_sz_go_")]
    assert go, "sizing button missing"
    go[0].click().run()
    assert not at.exception
    assert any("factor" in str(df.value.columns.tolist()).lower() for df in at.dataframe)


def test_leg_table_uses_grid_names(tmp_path):
    case = _budget_case()
    case["structure"]["elements"] = [
        dict(name="Leg x0 A can00", x1=0, y1=0, z1=0, x2=0, y2=0, z2=10, d_out=1.0, d_in=0.9),
        dict(name="Leg x7 B tube01", x1=7, y1=7, z1=0, x2=7, y2=7, z2=20, d_out=1.0, d_in=0.9),
        dict(name="Leg x7 C can00", x1=7, y1=-7, z1=0, x2=7, y2=-7, z2=5, d_out=1.0, d_in=0.9)]
    p = tmp_path / "l.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    assert not at.exception
    legs = [d.value for d in at.dataframe if "leg" in d.value.columns][0]
    assert legs["leg"].tolist() == ["A1", "A''2", "B2"]


def test_geometry_tab_shows_case_platform_with_leg_names(tmp_path):
    case = _budget_case()
    case["structure"]["elements"] = [
        dict(name="Leg x0 A can00", x1=0, y1=0, z1=0, x2=0, y2=0, z2=10, d_out=1.0, d_in=0.9),
        dict(name="Leg x7 B tube01", x1=7, y1=7, z1=0, x2=7, y2=7, z2=20, d_out=1.0, d_in=0.9)]
    p = tmp_path / "g.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    assert not at.exception
    assert any("Platform from the loaded case file" in s.value for s in at.subheader)
    legs = [d.value for d in at.dataframe if "leg" in d.value.columns]
    assert len(legs) == 2                                    # tab 3 and tab 7 each show the legs table
    assert legs[0]["leg"].tolist() == ["A1", "B2"]


def test_geometry_tab_has_no_case_section_without_a_file():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert not any("Platform from the loaded case file" in s.value for s in at.subheader)


def test_send_platform_to_tabs_1_to_6(tmp_path):
    case = _budget_case()
    case["structure"] = {
        "elements": [dict(name="Leg x0 A can00", x1=0, y1=0, z1=0, x2=0, y2=0, z2=50, d_out=1.0, d_in=0.9),
                     dict(name="Leg x7 A can00", x1=7, y1=0, z1=0, x2=7, y2=0, z2=50, d_out=1.0, d_in=0.9)],
        "weights": [dict(item="Main structure", mass_t=40.0, x=3.5, y=0, z=20.0)]}
    case["modules"] = [dict(name="legs", members=["Leg x0 A can00", "Leg x7 A can00"])]
    p = tmp_path / "send.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    btn = [b for b in at.button if b.key and b.key.startswith("bu_send_")][0]
    btn.click().run()
    assert not at.exception
    names = at.session_state["t_elems"]["name"].tolist()
    assert len(names) == 2 and names[0].startswith("Leg x0 A can00") and names[1].startswith("Leg x7 A can00")
    assert at.session_state["t_elems"]["buoyant"].tolist() == [True, True]      # sealed module => buoyant
    assert at.session_state["t_weights"]["mass_t"].sum() == 40.0
    assert at.session_state["project"] == "unit budget case"


def test_structure_weights_and_cog_are_shown(tmp_path):
    case = _budget_case()
    case["datum"] = {"z_offset_m": 10.0}
    case["structure"] = {
        "elements": [dict(name="Leg x0 A can00", x1=0, y1=0, z1=-10, x2=0, y2=0, z2=40, d_out=1.0)],
        "weights": [dict(item="a", mass_t=30.0, x=0, y=0, z=0.0), dict(item="b", mass_t=10.0, x=4, y=2, z=20.0)]}
    p = tmp_path / "w.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Structure weight"] == "40.00 t"
    assert m["CoG x"] == "1.000 m" and m["CoG y"] == "0.500 m"
    assert m["CoG z (EL)"] == "5.00 m"           # body z 15 above base, datum offset 10 => EL 5


def test_synthetic_modules_are_flagged_after_hand_over(tmp_path):
    case = _budget_case()
    case["structure"] = {
        "elements": [dict(name="Leg x0 A can00", x1=0, y1=0, z1=0, x2=0, y2=0, z2=50, d_out=1.0, d_in=0.9)],
        "weights": [dict(item="Main structure", mass_t=10.0, x=0, y=0, z=20.0)]}
    case["modules"] = [dict(name="Top tank (synthetic)", note="SYNTHETIC placeholder, replace",
                            elements=[dict(name="t", x1=0, y1=0, z1=10, x2=0, y2=0, z2=30, d_out=2.0)],
                            weights=[dict(item="ts", mass_t=1.0, x=0, y=0, z=20)])]
    p = tmp_path / "f.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    [b for b in at.button if b.key and b.key.startswith("bu_send_")][0].click().run()
    assert not at.exception
    assert at.session_state["data_flags"] == ["SYNTHETIC placeholder: Top tank (synthetic)"]
    assert any("SYNTHETIC placeholder" in w.value for w in at.sidebar.warning)


def test_sidebar_says_when_tabs_1_to_6_still_hold_the_demo_jacket(tmp_path):
    case = _budget_case()
    case["structure"] = {"elements": [dict(name="L", x1=0, y1=0, z1=0, x2=0, y2=0, z2=50, d_out=1.0)],
                         "weights": [dict(item="w", mass_t=10.0, x=0, y=0, z=20.0)]}
    p = tmp_path / "d.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    at.run()                                   # second pass so the sidebar sees the case loaded in tab 7
    assert any("still hold the demo jacket" in i.value for i in at.sidebar.info)
    [b for b in at.button if b.key and b.key.startswith("bu_send_")][0].click().run()
    assert not any("still hold the demo jacket" in i.value for i in at.sidebar.info)


def test_status_strip_demo_then_synthetic_after_send(tmp_path):
    case = _budget_case()
    case["structure"] = {"elements": [dict(name="L", x1=0, y1=0, z1=0, x2=0, y2=0, z2=50, d_out=1.0)],
                         "weights": [dict(item="w", mass_t=10.0, x=0, y=0, z=20.0)]}
    case["modules"] = [dict(name="Top tank (synthetic)", note="SYNTHETIC placeholder, replace",
                            elements=[dict(name="t", x1=0, y1=0, z1=10, x2=0, y2=0, z2=30, d_out=2.0)],
                            weights=[dict(item="ts", mass_t=1.0, x=0, y=0, z=20)])]
    p = tmp_path / "s.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    at.run()
    assert any("built-in demo jacket" in w.value for w in at.warning)
    [b for b in at.button if b.key and b.key.startswith("bu_send_")][0].click().run()
    txt = " ".join(w.value for w in at.warning)
    assert "synthetic:" in txt and "Top tank (synthetic)" in txt and "tow point" in txt


def test_window_figure_shades_each_pass_run_with_width():
    import pandas as pd
    import buoyancy_ui as ui
    from types import SimpleNamespace as NS
    df = pd.DataFrame(dict(diameter_m=[2, 3, 4, 5, 6], floats=[True] * 5, reserve_pct=[1, 5, 12, 20, 25],
                           gm_min_m=[1, 2, 3, 4, 5], clearance_m=[1, 2, 3, 4, 5],
                           intact_passed=[False, True, False, True, True], all_passed=[False] * 4 + [True]))
    f = ui._window_figure(df, 10, 5, NS(gm_min=1.0, clear_min_m=0.0))
    rects = {(float(r.x0), float(r.x1)) for r in f.layout.shapes if r.type == "rect"}
    assert rects == {(2.5, 3.5), (4.5, 6.5), (5.5, 6.5)}          # the 4 m gap stays unshaded; single size has width
    assert [a.text for a in f.layout.annotations if "pass" in (a.text or "")] == ["intact passes", "intact + all damage pass"]
    hl = [s for s in f.layout.shapes if s.type == "line" and s.y0 == 0.0]
    assert not hl                                                # clearance limit 0 draws no line


def _sent(tmp_path, pre=None):
    case = _budget_case()
    case["structure"] = {"elements": [dict(name="L", x1=0, y1=0, z1=0, x2=0, y2=0, z2=50, d_out=1.0)],
                         "weights": [dict(item="w", mass_t=10.0, x=0, y=0, z=20.0)]}
    p = tmp_path / "p.case.json"
    p.write_text(json.dumps(case))
    at = _run(p)
    at.run()
    for k, v in (pre or {}).items():
        at.session_state[k] = v
    [b for b in at.button if b.key and b.key.startswith("bu_send_")][0].click().run()
    return at


def _strip(at):
    return " ".join(w.value for w in at.warning)


def test_placeholders_only_for_untouched_defaults(tmp_path):
    at = _sent(tmp_path, pre={"p_wind_speed_kn": 55.0})
    assert "wind speed" not in _strip(at) and "tow heading" in _strip(at) and "tow point" in _strip(at)
    assert set(at.session_state["placeholders"]) == {"p_tow_x", "p_tow_z", "p_tow_heading_deg"}


def test_tow_point_clears_when_x_is_set(tmp_path):
    at = _sent(tmp_path)
    at.session_state["p_tow_x"] = at.session_state["p_tow_x"] + 3.0
    at.run()
    assert "tow point" not in _strip(at)


def test_case_file_in_cases_folder_loads_without_upload(tmp_path, monkeypatch):
    import buoyancy_ui as ui
    case = _budget_case()
    (tmp_path / "x.case.json").write_text(json.dumps(case))
    monkeypatch.setattr(ui, "DEFAULT_CASE_DIR", tmp_path)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert any("Loaded from the deployment: x.case.json" in c.value for c in at.caption)
