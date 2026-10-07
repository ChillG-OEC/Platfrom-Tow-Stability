"""Smoke tests: the 3D figure builds in every view and its camera buttons are valid."""
import jacket_stability as js
import viz


def _synth():
    d = js.synthetic_inputs()
    els = js.elements_from_rows(d["elements"])
    wts = [js.WeightItem(r["item"], r["mass_t"], r["x"], r["y"], r["z"]) for r in d["weights"]]
    ops = [js.Opening(r["name"], r["x"], r["y"], r["z"]) for r in d["openings"]]
    prm = js.Params(tow_point=d["tow_point"], tow_heading_deg=d["tow_heading_deg"],
                    water_depth_m=d["water_depth_m"])
    return els, wts, ops, prm


def test_as_built_figure_builds_with_four_camera_buttons():
    els, wts, ops, prm = _synth()
    fig = viz.jacket_figure(els, wts, ops, prm, show_triad=True, title="As built")
    assert len(fig.data) > 0
    menu = fig.layout.updatemenus[0]
    assert [b.label for b in menu.buttons] == ["Iso", "Plan", "Side (from -y)", "End (from +x)"]
    assert menu.xanchor == "left" and menu.x == 0.0      # clear of the toolbar at the top right


def test_floating_figure_builds():
    els, wts, ops, prm = _synth()
    m = js.JacketModel(els, wts, ops, prm)
    stt = m.attitude_state(0.0, 0.0, m.hydrostatics()["trim_deg"])
    fig = viz.jacket_figure(els, wts, ops, prm, rot=stt["rot"], zw=stt["zw"], g_pt=stt["G"], b_pt=stt["B"])
    assert len(fig.data) > 0


def test_colour_by_diameter_names_traces_by_diameter():
    els, wts, ops, prm = _synth()
    names = {t.name for t in viz.jacket_figure(els, wts, ops, prm, color_by="diameter").data if t.name}
    dias = {f"Ø{round(e.d_out * 1000.0):.0f} mm" for e in els}
    assert dias <= names
    assert not any("Leg" in n for n in names if n.startswith("Leg"))
