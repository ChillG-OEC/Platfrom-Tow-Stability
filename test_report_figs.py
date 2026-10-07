"""Vector figures for the PDF report."""
import report_figs as rf
import jacket_stability as js


def _jacket():
    inp = js.synthetic_inputs()
    return js.elements_from_rows(inp["elements"])


def test_every_view_draws_the_members():
    els = _jacket()
    for view in ("iso", "side", "end", "plan"):
        d = rf.panel(els, view, 200, 260)
        lines = [g for g in d.contents if g.__class__.__name__ == "Line"]
        assert len(lines) >= len(els) * 0.8, view            # nearly every member gives a stroke (some are edge-on in a view)


def test_waterline_and_markers_are_added():
    els = _jacket()
    plain = rf.panel(els, "side", 200, 260)
    wet = rf.panel(els, "side", 200, 260, zw=40.0, g_pt=(0, 0, 40.0), b_pt=(0, 0, 20.0))
    assert len(wet.contents) > len(plain.contents) + 4


def test_leg_labels_use_grid_names():
    mk = lambda n, x, y, z1, z2: js.Element(n, (x, y, z1), (x, y, z2), 1.0, 0.0, False)
    els = [mk("Leg x0 A can00", 0, 0, 0, 10), mk("Leg x0 A tube01", 0, 0, 10, 50), mk("Leg x7 B tube01", 7, 7, 0, 40),
           mk("Leg x7 C can00", 7, -7, 0, 20), mk("Brace x", 0, 0, 0, 5)]
    lab = {n: z for n, x, y, z in rf.leg_labels(els)}
    assert lab == {"A1": 50, "B2": 40, "A''2": 20}


def test_report_builds_with_figures_and_attitudes(monkeypatch):
    import report
    from streamlit.testing.v1 import AppTest
    seen = {}
    orig = report.build_pdf

    def spy(snap, res, meta, *a, **k):
        seen["meta"] = meta
        buf = orig(snap, res, meta, *a, **k)
        seen["size"] = len(buf.getvalue())
        return buf
    monkeypatch.setattr(report, "build_pdf", spy)
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    [b for b in at.sidebar.button if "Run" in b.label][0].click().run(timeout=180)
    assert not at.exception
    assert len(seen["meta"]["attitudes"]) == 2                    # floating level + governing equilibrium
    assert all({"rot", "zw", "G", "B", "title"} <= set(a) for a in seen["meta"]["attitudes"])
    assert seen["size"] > 20_000


def test_demo_jacket_is_still_recognised_with_the_tank_column():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert any("SYNTHETIC illustrative jacket" in w.value for w in at.sidebar.warning)
