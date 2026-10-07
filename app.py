"""
Jacket Wet-Tow Stability (screening)  -  Streamlit front end for jacket_stability.py

Run:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


import jacket_stability as js
import model_validation as mv
import buoyancy_ui
import towplan_ui
import sensitivity as sv
import viz
from report import ASSUMPTIONS, APP_VERSION, build_pdf, fmt

st.set_page_config(page_title="Jacket Wet-Tow Stability", page_icon=str(Path(__file__).with_name("oec_logo.png")) if Path(__file__).with_name("oec_logo.png").exists() else "⚓", layout="wide")

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig: go.Figure) -> None:
    """st.plotly_chart across Streamlit versions."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Defaults and session state
# ----------------------------------------------------------------------------
EX = js.synthetic_inputs()

DEFAULTS = {
    "project": "Conrad Mako jacket wet tow - synthetic screening case",
    "engineer": "",
    "p_rho_w": 1.025, "p_rho_a": 1.225,
    "p_wind_speed_kn": 40.0, "p_wind_cs": 1.0, "p_wind_alpha": 0.0, "p_wind_zref": 10.0,
    "p_shielding": 1.0,
    "p_tow_speed_kn": 2.5, "p_current_speed_kn": 0.0, "p_current_dir_deg": 180.0, "p_tow_speed_ref": "ground",
    "p_tow_x": float(EX["tow_point"][0]), "p_tow_y": float(EX["tow_point"][1]),
    "p_tow_z": float(EX["tow_point"][2]),
    "p_tow_heading_deg": float(EX["tow_heading_deg"]), "p_tow_elev_deg": 0.0, "p_tow_share": 1.0, "p_tow_mode": "auto", "p_tow_manual_kn": 0.0,
    "p_tow_factor": 1.0, "p_cd_water": 1.0,
    "p_depth_m": float(EX["water_depth_m"]), "p_tide_m": float(EX["tide_m"]),
    "p_fsm": 0.0, "p_lr_mode": "area", "p_slice_m": 0.5, "p_trim_limit_deg": 25.0,
    "c_gm_min": 1.0, "c_heel_max_deg": 15.0, "c_ratio_min": 1.3, "c_cap_deg": 40.0,
    "c_df_min_deg": 0.0, "c_clear_min_m": float(EX["clear_min_m"]), "c_emerged_min_m": float(EX["emerged_min_m"]),
    "a_step": 45.0, "a_phi_min": -10.0, "a_phi_max": 50.0, "a_phi_step": 2.5,
    "a_scenario": "Intact",
}

W_COLS = ["item", "mass_t", "x", "y", "z"]
E_COLS = ["name", "x1", "y1", "z1", "x2", "y2", "z2", "d_out", "d_in", "buoyant", "exposed", "flooded"]
O_COLS = ["name", "x", "y", "z"]
E_BOOL = ["buoyant", "exposed", "flooded"]
LINE_COLS = ["name", "x", "y", "z", "heading_deg", "elevation_deg", "type", "tension_kn", "share"]
LINE_TYPES = ["Fixed pull", "Tow leg"]


def empty_lines() -> pd.DataFrame:
    return pd.DataFrame({"name": pd.Series(dtype="object"), "x": pd.Series(dtype="float64"),
                         "y": pd.Series(dtype="float64"), "z": pd.Series(dtype="float64"),
                         "heading_deg": pd.Series(dtype="float64"), "elevation_deg": pd.Series(dtype="float64"),
                         "type": pd.Series(dtype="object"), "tension_kn": pd.Series(dtype="float64"),
                         "share": pd.Series(dtype="float64")})


def init_state() -> None:
    ss = st.session_state
    if "ver" not in ss:
        ss.ver = 0
        ss.t_weights = pd.DataFrame(EX["weights"], columns=W_COLS)
        ss.t_elems = pd.DataFrame(EX["elements"], columns=E_COLS)
        ss.t_open = pd.DataFrame(EX["openings"], columns=O_COLS)
        ss.t_lines = empty_lines()
        ss.res = None
        ss.res_hash = None
        ss.res_snap = None
    for k, v in DEFAULTS.items():
        if k not in ss:
            ss[k] = v


init_state()


def num(label: str, key: str, **kw) -> float:
    return st.number_input(label, key=key, **kw)


def records(df: pd.DataFrame, bools: list[str] | None = None) -> list[dict]:
    d = df.copy().dropna(how="all")
    for c in bools or []:
        if c in d:
            d[c] = d[c].fillna(False).astype(bool)
    d = d.replace({np.nan: None})
    return d.to_dict("records")


def _num_ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ----------------------------------------------------------------------------
# Snapshot (inputs) <-> widgets <-> model
# ----------------------------------------------------------------------------
PARAM_KEYS = [k for k in DEFAULTS if k.startswith(("p_", "c_", "a_"))] + ["project", "engineer"]


def line_rows(lines_df: pd.DataFrame) -> list[dict]:
    rows = []
    for r in records(lines_df):
        if not r.get("name") or not all(_num_ok(r.get(c)) for c in ("x", "y", "z")):
            continue

        def f(key, default):
            return float(r[key]) if _num_ok(r.get(key)) else default
        rows.append(dict(name=str(r["name"]), x=float(r["x"]), y=float(r["y"]), z=float(r["z"]),
                         heading_deg=f("heading_deg", 0.0), elevation_deg=f("elevation_deg", 0.0),
                         type=r.get("type") if r.get("type") in LINE_TYPES else LINE_TYPES[0],
                         tension_kn=f("tension_kn", 0.0), share=f("share", 1.0)))
    return rows


def make_snapshot(weights_df, elems_df, open_df, lines_df=None) -> dict:
    ss = st.session_state
    w_rows = [r for r in records(weights_df) if all(_num_ok(r.get(c)) for c in W_COLS[1:])]
    w_tot = sum(float(r["mass_t"]) for r in w_rows)
    return dict(
        version=APP_VERSION,
        widgets={k: ss[k] for k in PARAM_KEYS},
        weights=w_rows,
        elements=records(elems_df, E_BOOL),
        openings=[r for r in records(open_df) if all(_num_ok(r.get(c)) for c in O_COLS[1:])],
        lines=line_rows(lines_df) if lines_df is not None else [],
        fsc_m=(float(ss["p_fsm"]) / w_tot) if w_tot > 0 else 0.0,
    )


def parts_from_snapshot(snap: dict):
    w = snap["widgets"]
    els = js.elements_from_rows(snap["elements"])
    wts = [js.WeightItem(str(r["item"]), float(r["mass_t"]), float(r["x"]), float(r["y"]), float(r["z"]))
           for r in snap["weights"]]
    ops = [js.Opening(str(r["name"]), float(r["x"]), float(r["y"]), float(r["z"])) for r in snap["openings"]]
    primary = js.Line("Tow line", (w["p_tow_x"], w["p_tow_y"], w["p_tow_z"]), w["p_tow_heading_deg"],
                      w.get("p_tow_elev_deg", 0.0), "tow" if w["p_tow_mode"] == "auto" else "fixed",
                      w["p_tow_manual_kn"], w.get("p_tow_share", 1.0))
    extra = [js.Line(r["name"], (r["x"], r["y"], r["z"]), r["heading_deg"], r["elevation_deg"],
                     "tow" if r["type"] == "Tow leg" else "fixed", r["tension_kn"], r["share"])
             for r in snap.get("lines", [])]
    params = js.Params(
        lines=tuple([primary] + extra),
        rho_w=w["p_rho_w"], rho_a=w["p_rho_a"], wind_speed_kn=w["p_wind_speed_kn"],
        wind_cs=w["p_wind_cs"], wind_alpha=w["p_wind_alpha"], wind_zref=w["p_wind_zref"],
        cd_water=w["p_cd_water"], shielding=w["p_shielding"], tow_speed_kn=w["p_tow_speed_kn"],
        current_speed_kn=w.get("p_current_speed_kn", 0.0), current_dir_deg=w.get("p_current_dir_deg", 180.0),
        tow_speed_ref=w.get("p_tow_speed_ref", "ground"),
        tow_point=(w["p_tow_x"], w["p_tow_y"], w["p_tow_z"]), tow_heading_deg=w["p_tow_heading_deg"],
        tow_mode=w["p_tow_mode"], tow_manual_kn=w["p_tow_manual_kn"], tow_factor=w["p_tow_factor"],
        fsc_m=snap["fsc_m"], lr_mode=w["p_lr_mode"], slice_m=w["p_slice_m"],
        trim_limit_deg=w["p_trim_limit_deg"],
        water_depth_m=w.get("p_depth_m", 0.0), tide_m=w.get("p_tide_m", 0.0))
    crit = js.Criteria(gm_min=w["c_gm_min"], heel_max_deg=w["c_heel_max_deg"], ratio_min=w["c_ratio_min"],
                       cap_deg=w["c_cap_deg"], df_min_deg=w["c_df_min_deg"],
                       clear_min_m=w.get("c_clear_min_m", 0.0), emerged_min_m=w.get("c_emerged_min_m", 0.0))
    return els, wts, ops, params, crit


def apply_snapshot(data: dict) -> None:
    ss = st.session_state
    for k, v in data.get("widgets", {}).items():
        if k in DEFAULTS:
            ss[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], str) else str(v)
    for k in ("p_depth_m", "p_tide_m", "c_clear_min_m", "c_emerged_min_m"):
        if k not in data.get("widgets", {}):
            ss[k] = 0.0          # files saved before the set-down check existed: check off
    ss.t_weights = pd.DataFrame(data.get("weights", []), columns=W_COLS)
    el = pd.DataFrame(data.get("elements", []), columns=E_COLS)
    for c in E_BOOL:
        el[c] = el[c].fillna(False).astype(bool)
    ss.t_elems = el
    ss.t_open = pd.DataFrame(data.get("openings", []), columns=O_COLS)
    ln = pd.DataFrame(data.get("lines", []), columns=LINE_COLS)
    ss.t_lines = ln if len(ln) else empty_lines()
    ss.ver += 1
    ss.res = None


def snap_hash(snap: dict) -> str:
    return json.dumps(snap, sort_keys=True, default=str)


# Load a saved input file before any widget is created
st.session_state.setdefault("up_ver", 0)
_up = st.sidebar.file_uploader("Load saved inputs (.json)", type="json", key=f"up_{st.session_state.up_ver}")
if _up is not None:
    try:
        apply_snapshot(json.load(_up))
        st.session_state.up_ver += 1
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read that file: {exc}")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
_LOGO = Path(__file__).with_name("oec_logo.png")
if _LOGO.exists():
    try:
        st.logo(str(_LOGO), size="large")
    except Exception:  # older Streamlit without st.logo
        pass
    _h1, _h2 = st.columns([1, 7], vertical_alignment="center")
    with _h1:
        st.image(str(_LOGO), width=130)
    with _h2:
        st.title("Jacket Wet-Tow Stability")
else:
    st.title("⚓ Jacket Wet-Tow Stability")
st.caption("Screening tool: free-to-trim heel sweep for a floating jacket on buoyancy tanks, "
           "wind + tow-line heeling, downflooding, damaged case. Not a substitute for a checked calculation.")

tab_in, tab_env, tab_geo, tab_res, tab_sens, tab_rep, tab_buoy, tab_tow = st.tabs(
    ["1 · Inputs", "2 · Environment, tow & criteria", "3 · Geometry check", "4 · Results", "5 · Sensitivity",
     "6 · Report", "7 · Buoyancy (case file)", "8 · Tow plan (approach)"])

# ----------------------------------------------------------------------------
# Tab 1: inputs
# ----------------------------------------------------------------------------
with tab_in:
    c1, c2 = st.columns(2)
    c1.text_input("Project / structure", key="project")
    c2.text_input("Prepared by", key="engineer")
    st.markdown("**Coordinates:** x forward (tow direction), y athwart, z up in the floating attitude. "
                "Any consistent origin; all in metres / tonnes.")

    st.subheader("Weights (floating condition)")
    st.caption("Include structure, equipment, grout/ballast and temporary items. Negative masses allowed for deductions.")
    weights_df = st.data_editor(
        st.session_state.t_weights, num_rows="dynamic", hide_index=True, key=f"ed_w_{st.session_state.ver}",
        column_config={
            "item": st.column_config.TextColumn("Item", required=True),
            "mass_t": st.column_config.NumberColumn("Mass [t]", format="%.1f"),
            "x": st.column_config.NumberColumn("x [m]", format="%.2f"),
            "y": st.column_config.NumberColumn("y [m]", format="%.2f"),
            "z": st.column_config.NumberColumn("z [m]", format="%.2f"),
        })
    _w = [r for r in records(weights_df) if all(_num_ok(r.get(c)) for c in W_COLS[1:])]
    _wt = sum(float(r["mass_t"]) for r in _w)
    if _wt > 0:
        _g = [sum(float(r["mass_t"]) * float(r[a]) for r in _w) / _wt for a in ("x", "y", "z")]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total weight", f"{_wt:,.0f} t")
        m2.metric("CoG x", f"{_g[0]:.2f} m")
        m3.metric("CoG y", f"{_g[1]:.2f} m")
        m4.metric("CoG z", f"{_g[2]:.2f} m")

    st.subheader("Members, tanks and legs (straight cylinders)")
    st.caption("Buoyant = watertight, contributes displacement. Exposed = counts for wind area and tow drag. "
               "Flooded = loses buoyancy in the damaged case. Inner diameter > 0 makes a tank an annulus around a leg. "
               "Do not mark a leg buoyant if a tank annulus already surrounds it.")
    elems_df = st.data_editor(
        st.session_state.t_elems, num_rows="dynamic", hide_index=True, key=f"ed_e_{st.session_state.ver}",
        column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            **{c: st.column_config.NumberColumn(c, format="%.2f") for c in
               ("x1", "y1", "z1", "x2", "y2", "z2")},
            "d_out": st.column_config.NumberColumn("Ø out [m]", format="%.2f"),
            "d_in": st.column_config.NumberColumn("Ø in [m]", format="%.2f"),
            "buoyant": st.column_config.CheckboxColumn("Buoyant"),
            "exposed": st.column_config.CheckboxColumn("Exposed"),
            "flooded": st.column_config.CheckboxColumn("Flooded (damage)"),
        })

    st.subheader("Downflooding points")
    st.caption("Vents, hatches and other openings that would flood if immersed. Leave empty to skip the check.")
    open_df = st.data_editor(
        st.session_state.t_open, num_rows="dynamic", hide_index=True, key=f"ed_o_{st.session_state.ver}",
        column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            "x": st.column_config.NumberColumn("x [m]", format="%.2f"),
            "y": st.column_config.NumberColumn("y [m]", format="%.2f"),
            "z": st.column_config.NumberColumn("z [m]", format="%.2f"),
        })

# ----------------------------------------------------------------------------
# Tab 2: environment / tow / criteria
# ----------------------------------------------------------------------------
with tab_env:
    a, b = st.columns(2)
    with a:
        st.subheader("Environment")
        num("Design wind speed [kn]", "p_wind_speed_kn", min_value=0.0, step=1.0)
        num("Wind shape coefficient Cs", "p_wind_cs", min_value=0.1, max_value=2.5, step=0.05,
            help="Applied to each member's projected area (circular members: confirm against project basis).")
        num("Shielding factor on projected area", "p_shielding", min_value=0.1, max_value=1.0, step=0.05,
            help="1.0 = no shielding (conservative). Applied to wind area and tow drag.")
        num("Wind profile exponent α", "p_wind_alpha", min_value=0.0, max_value=0.4, step=0.025,
            help="0 = uniform speed. Otherwise V(z) = V·(z/zref)^α with z above the waterline.")
        num("Profile reference height zref [m]", "p_wind_zref", min_value=1.0, step=1.0)
        num("Water density [t/m³]", "p_rho_w", min_value=0.9, max_value=1.1, step=0.001, format="%.3f")
        num("Air density [kg/m³]", "p_rho_a", min_value=1.0, max_value=1.4, step=0.005, format="%.3f")
        num("Free-surface moment [t·m]", "p_fsm", min_value=0.0, step=10.0,
            help="Sum of free-surface moments of slack tanks. Applied as FSM/W·sinφ off the righting arm.")
    with b:
        st.subheader("Tow")
        num("Tow speed [kn]", "p_tow_speed_kn", min_value=0.0, step=0.1)
        st.selectbox("Tow speed is measured", ["ground", "water"], key="p_tow_speed_ref",
                     format_func=lambda x: "over the seabed (current adds to the drag)" if x == "ground"
                     else "through the water (current has no effect)",
                     help="Confirm with the tow master. Bollard-pull curves are normally through the water; a schedule is over ground.")
        num("Current speed [kn]", "p_current_speed_kn", min_value=0.0, step=0.1,
            help="Horizontal, uniform with depth (conservative for a deep jacket). Tab 7 can fill this from the case file.")
        num("Current flows toward [° from the tow direction]", "p_current_dir_deg", min_value=0.0, max_value=360.0, step=15.0,
            help="0 = with the tow (following), 180 = against the tow (head), 90 = across, to port of the tow direction.")
        st.selectbox("Tow-line pull", ["auto", "manual"], key="p_tow_mode",
                     help="auto = calm-water drag of submerged members × factor; manual = a pull you specify.")
        num("Margin on calm-water drag (auto)", "p_tow_factor", min_value=0.5, step=0.1)
        num("Manual tow pull [kN]", "p_tow_manual_kn", min_value=0.0, step=10.0)
        st.subheader("Water depth (set-down check)")
        num("Water depth at LAT [m]", "p_depth_m", min_value=0.0, step=1.0,
            help="Depth from seabed to LAT. 0 = seabed clearance not checked. Take the structure base as the lowest point of any member.")
        num("Tide above LAT [m]", "p_tide_m", min_value=0.0, step=0.1,
            help="Use HAT - LAT for the governing high-water case; 0 for LAT.")
        num("Drag coefficient, submerged members", "p_cd_water", min_value=0.1, max_value=3.0, step=0.05)
        t1, t2, t3 = st.columns(3)
        with t1:
            num("Tow connection x [m]", "p_tow_x", step=0.5)
        with t2:
            num("Tow connection y [m]", "p_tow_y", step=0.5)
        with t3:
            num("Tow connection z [m]", "p_tow_z", step=0.5)
        d1, d2, d3 = st.columns(3)
        with d1:
            num("Force heading [°]", "p_tow_heading_deg", step=5.0,
                help="Horizontal direction the line pulls the jacket toward, measured from body +x toward +y (upright).")
        with d2:
            num("Force elevation [°]", "p_tow_elev_deg", min_value=-89.0, max_value=89.0, step=1.0,
                help="Angle of the pull above horizontal (+ up). An upward pull also reduces the buoyancy needed.")
        with d3:
            num("Share of tow pull", "p_tow_share", min_value=0.01, step=0.1,
                help="Only matters when you add 'Tow leg' lines below: legs share the drag-based pull in this ratio.")
        st.selectbox("Hydrodynamic reaction level", ["area", "half_draft"], key="p_lr_mode",
                     help="area = centroid of submerged lateral area; half_draft = half way down the draft.")

    st.subheader("Additional line loads (pull-in lines, extra tow legs)")
    st.caption("Each row is one line acting ON the jacket at its connection point. Heading = horizontal direction of the pull "
               "from body +x toward +y; elevation + = upward. 'Fixed pull' applies the tension you give. 'Tow leg' shares the "
               "drag-based tow pull with the tow line above (a bridle): legs are scaled so their resultant along the mean "
               "heading equals drag × margin. Leave empty for a single tow line.")
    lines_df = st.data_editor(
        st.session_state.t_lines, num_rows="dynamic", hide_index=True, key=f"ed_l_{st.session_state.ver}",
        column_config={
            "name": st.column_config.TextColumn("Line", required=True),
            "x": st.column_config.NumberColumn("x [m]", format="%.2f"),
            "y": st.column_config.NumberColumn("y [m]", format="%.2f"),
            "z": st.column_config.NumberColumn("z [m]", format="%.2f"),
            "heading_deg": st.column_config.NumberColumn("Heading [°]", format="%.1f"),
            "elevation_deg": st.column_config.NumberColumn("Elevation [°]", format="%.1f", min_value=-89.0, max_value=89.0),
            "type": st.column_config.SelectboxColumn("Type", options=LINE_TYPES, default=LINE_TYPES[0]),
            "tension_kn": st.column_config.NumberColumn("Tension [kN] (fixed)", format="%.0f", min_value=0.0),
            "share": st.column_config.NumberColumn("Share (tow leg)", format="%.2f", min_value=0.01, default=1.0),
        })

    st.subheader("Acceptance criteria")
    st.caption("Defaults are placeholders. Set these from the project design basis / marine warranty requirements.")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        num("Min GM [m]", "c_gm_min", min_value=0.0, step=0.1)
    with k2:
        num("Max static heel [°]", "c_heel_max_deg", min_value=0.0, step=1.0)
    with k3:
        num("Min area ratio", "c_ratio_min", min_value=0.0, step=0.05)
    with k4:
        num("Integration cap [°]", "c_cap_deg", min_value=5.0, max_value=80.0, step=1.0)
    with k5:
        num("Min downflood angle [°] (0 = off)", "c_df_min_deg", min_value=0.0, step=1.0)
    q1, q2, _q3 = st.columns(3)
    with q1:
        num("Min seabed clearance [m] (0 = off)", "c_clear_min_m", min_value=0.0, step=0.5,
            help="Lowest point of the jacket to the seabed, level floating condition. Needs the water depth above.")
    with q2:
        num("Min tank length above water [m] (0 = off)", "c_emerged_min_m", min_value=0.0, step=0.5,
            help="Shortest length of any buoyancy tank that stays out of the water.")

    st.subheader("Analysis settings")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.selectbox("Wind heading step [°]", [22.5, 45.0, 90.0], key="a_step")
    with s2:
        num("Heel from [°]", "a_phi_min", min_value=-30.0, max_value=0.0, step=2.5)
    with s3:
        num("Heel to [°]", "a_phi_max", min_value=10.0, max_value=80.0, step=2.5)
    with s4:
        num("Heel step [°]", "a_phi_step", min_value=0.5, max_value=5.0, step=0.5)
    with s5:
        st.selectbox("Scenario", ["Intact", "Damaged (flooded elements lose buoyancy)"], key="a_scenario")
    with st.expander("Numerical settings"):
        num("Slice length for wind / drag integration [m]", "p_slice_m", min_value=0.1, max_value=2.0, step=0.1)
        num("Trim search limit [°]", "p_trim_limit_deg", min_value=5.0, max_value=45.0, step=1.0)

# ----------------------------------------------------------------------------
# Snapshot, validation, run
# ----------------------------------------------------------------------------
snap = make_snapshot(weights_df, elems_df, open_df, lines_df)
cur_hash = snap_hash(snap)
els, wts, ops, params, crit = parts_from_snapshot(snap)
errors, warnings = js.validate_inputs(els, wts, ops, params)
try:   # extra model checks (model_validation.py); a failure here must never stop the app
    _dmg_now = str(st.session_state.get("a_scenario", "Intact")).startswith("Damaged")
    errors, warnings = mv.merge_with_legacy(
        errors, warnings, mv.validate_model(els, wts, ops, params, crit, damaged=_dmg_now))
except Exception as _exc:
    warnings = list(warnings) + [f"Extra model checks could not run: {_exc}"]

st.sidebar.header("Run")
_sj = json.dumps(snap["elements"], sort_keys=True, default=str)
is_example = any(_sj == json.dumps(f()["elements"], sort_keys=True, default=str)
                 for f in (js.example_inputs, js.synthetic_inputs))
if is_example:
    st.sidebar.warning("Tables hold the SYNTHETIC illustrative jacket - not project data.")
for e in errors:
    st.sidebar.error(e)
run = st.sidebar.button("▶ Run analysis", type="primary", disabled=bool(errors))

if run and not errors:
    ss = st.session_state
    damaged = str(ss["a_scenario"]).startswith("Damaged")
    betas = np.arange(0.0, 360.0, float(ss["a_step"]))
    grid = np.unique(np.concatenate([np.arange(ss["a_phi_min"], ss["a_phi_max"] + 1e-9, ss["a_phi_step"]), [0.0]]))
    bar = st.sidebar.progress(0.0, text="Starting...")
    try:
        mdl = js.JacketModel(els, wts, ops, params, damaged=damaged)
        res = js.run_study(mdl, betas, grid, crit, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        res["tech"] = mdl.hydrostatics()
        try:
            res["float"] = js.float_check(mdl, crit)
        except ValueError:
            res["float"] = None
        ss.res, ss.res_hash, ss.res_snap = res, cur_hash, snap
        ss.res_meta = dict(damaged=damaged, betas=betas.tolist(), grid=grid.tolist())
    except ValueError as exc:
        st.sidebar.error(str(exc))
    bar.empty()

st.sidebar.download_button("💾 Save inputs (.json)", data=json.dumps(snap, indent=1, default=str),
                           file_name="jacket_tow_inputs.json", mime="application/json")

# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def curve_figure(h: dict, crit: js.Criteria) -> go.Figure:
    sw, an = h["sweep"], h["analysis"]
    p, gz, hl = sw["phi_deg"], sw["gz"], sw["hl"]
    fig = go.Figure()
    lim = an["theta_lim"]
    if lim:
        m = (p >= 0) & (p <= lim) & ~np.isnan(gz)
        if m.sum() > 1:
            fig.add_trace(go.Scatter(x=np.r_[p[m], p[m][::-1]], y=np.r_[gz[m], np.zeros(m.sum())], fill="toself",
                                     fillcolor="rgba(31,119,180,0.15)", line=dict(width=0), hoverinfo="skip",
                                     name="Area under GZ"))
    fig.add_trace(go.Scatter(x=p, y=gz, mode="lines+markers", name="Righting arm GZ", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=p, y=hl, mode="lines+markers", name="Heeling arm (wind + tow)",
                             line=dict(color="#d62728", width=2, dash="dash")))
    for x, label, col in ((an["theta_s"], "Static heel", "#2ca02c"), (an["theta_2"], "2nd intercept", "#9467bd"),
                          (an["theta_df"], "Downflood", "#ff7f0e"), (lim, "Limit", "#555555")):
        if x is not None:
            fig.add_vline(x=x, line=dict(color=col, dash="dot"), annotation_text=f"{label} {x:.1f}°",
                          annotation_position="top")
    fig.update_layout(title=f"GZ vs heeling arm - wind toward {h['beta']:.0f}° (body +x = 0°)", hovermode="x unified",
                      xaxis_title="Heel angle [deg]", yaxis_title="Lever arm [m]", template="plotly_white", height=430,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ----------------------------------------------------------------------------
# Tab 3: geometry check
# ----------------------------------------------------------------------------
def get_tech(damaged: bool, key: str) -> dict:
    """Model + hydrostatics for the current inputs, cached per input state."""
    ss = st.session_state
    if ss.get("tech_key") != key:
        try:
            mdl = js.JacketModel(els, wts, ops, params, damaged=damaged)
            ss.tech = dict(mdl=mdl, h=mdl.hydrostatics(), err=None)
        except ValueError as exc:
            ss.tech = dict(mdl=None, h=None, err=str(exc))
        ss.tech_key = key
    return ss.tech


def key_data_table(h: dict) -> pd.DataFrame:
    lo, hi = h["bbox_lo"], h["bbox_hi"]
    g, b = h["G_body"], h["B_body"]
    rows = [
        ("Principal dimensions", "", ""),
        ("Overall length x", f"{hi[0] - lo[0]:.1f}", "m"),
        ("Overall width y", f"{hi[1] - lo[1]:.1f}", "m"),
        ("Overall height z", f"{hi[2] - lo[2]:.1f}", "m"),
        ("Members / weight items", f"{h['n_members']} / {h['n_weights']}", ""),
        ("Mass properties", "", ""),
        ("Total weight (displacement)", f"{h['W']:,.0f}", "t"),
        ("CoG  x, y, z (G)", f"{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}", "m"),
        ("Floating condition (level, free trim, no wind/tow)", "", ""),
        ("Displaced volume", f"{h['volume']:,.0f}", "m³"),
        ("Draft (to lowest buoyant point)", f"{h['draft']:.2f}", "m"),
        ("Trim", f"{h['trim_deg']:.2f}", "°"),
        ("CoB  x, y, z (B)", f"{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}", "m"),
        ("Waterplane area", f"{h['waterplane_area']:,.0f}", "m²"),
        ("Immersion TPC", f"{h['tpc']:.2f}", "t/cm"),
        ("Reserve buoyancy", f"{h['reserve_buoyancy_pct']:.0f}", "%"),
        ("Stability (heights above base = lowest point of structure)", "", ""),
        ("KB", f"{h['kb_base']:.2f}", "m"),
        ("KG", f"{h['kg_base']:.2f}", "m"),
        ("BM  (wind +x | wind +y)", f"{h['bm_x']:.2f} | {h['bm_y']:.2f}", "m"),
        ("KM  (wind +x | wind +y)", f"{h['kg_base'] + h['gm_x']:.2f} | {h['kg_base'] + h['gm_y']:.2f}", "m"),
        ("GM  (wind +x | wind +y)", f"{h['gm_x']:.2f} | {h['gm_y']:.2f}", "m"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"])


with tab_geo:
    if is_example:
        st.warning("Showing the SYNTHETIC illustrative jacket (not project data). Replace the tables in tab 1 with project data.")
    if not els:
        st.info("No valid elements yet.")
    else:
        ss = st.session_state
        damaged_now = str(ss["a_scenario"]).startswith("Damaged")
        T = get_tech(damaged_now, cur_hash)
        if T["err"]:
            st.error(T["err"])
            show_view = "As built (body axes)"
        else:
            res_cur = ss.res is not None and ss.res_hash == cur_hash
            views = ["As built (body axes)", "Floating level - no wind or tow"]
            if res_cur:
                views.append("At static equilibrium - wind + tow")
            if ss.get("g_view") not in views:
                ss["g_view"] = views[0]
            show_view = st.radio("View", views, horizontal=True, key="g_view")
            if ss.res is not None and not res_cur:
                st.caption("The equilibrium view returns after you re-run the analysis with the current inputs.")

        left, right = st.columns([3, 2])
        with left:
            if T["err"] or show_view.startswith("As built"):
                fig = viz.jacket_figure(els, wts, ops, params, show_triad=True,
                                        title="As built - body axes (x forward, y athwart, z up)")
            elif show_view.startswith("Floating"):
                h = T["h"]
                stt = T["mdl"].attitude_state(0.0, 0.0, h["trim_deg"])
                fig = viz.jacket_figure(els, wts, ops, params, rot=stt["rot"], zw=stt["zw"], g_pt=stt["G"],
                                        b_pt=stt["B"], title=f"Floating level: draft {h['draft']:.2f} m, "
                                                             f"trim {h['trim_deg']:.2f}°")
            else:
                heads = ss.res["heads"]
                labs = [f"{x['beta']:.0f}°" for x in heads]
                gov_i = labs.index(f"{ss.res['summary']['governing_beta']:.0f}°")
                pick = st.selectbox("Wind heading (toward)", labs, index=gov_i, key="g_head")
                hd = heads[labs.index(pick)]
                an = hd["analysis"]
                stt = None
                if an["theta_s"] is not None:
                    stt = T["mdl"].attitude_state(hd["beta"], an["theta_s"], an["trim_at_s"])
                if stt is None:
                    st.info("No static equilibrium for this heading, so there is no attitude to draw.")
                    fig = viz.jacket_figure(els, wts, ops, params, show_triad=True, title="As built")
                else:
                    fig = viz.jacket_figure(
                        els, wts, ops, params, rot=stt["rot"], zw=stt["zw"], g_pt=stt["G"], b_pt=stt["B"],
                        wind_beta_deg=hd["beta"],
                        title=f"Equilibrium under wind toward {hd['beta']:.0f}°: heel {an['theta_s']:.1f}°, "
                              f"trim {an['trim_at_s']:.2f}°")
            cap_ = fig.layout.title.text or ""
            fig.update_layout(title_text="", margin=dict(l=0, r=0, t=34, b=0))
            if cap_:
                st.markdown(f"**{cap_}**")
            show(fig)
        with right:
            if not T["err"]:
                st.markdown("**Technical data**" + (" - damaged case" if damaged_now else ""))
                lines = ["| | | |", "|:--|--:|:--|"]
                for p_, v_, u_ in key_data_table(T["h"]).itertuples(index=False):
                    lines.append(f"| **{p_}** | | |" if not v_ else f"| {p_} | {v_} | {u_} |")
                st.markdown("\n".join(lines))
        if not T["err"]:
            with st.expander("Member schedule (volumes and buoyancy at the floating condition)"):
                mem = pd.DataFrame(T["h"]["members"]).rename(columns={
                    "name": "Member", "kind": "Type", "length_m": "Length [m]", "d_out": "Ø out [m]", "d_in": "Ø in [m]",
                    "volume_m3": "Volume [m³]", "sub_vol_m3": "Submerged [m³]", "sub_pct": "Submerged [%]",
                    "buoyancy_t": "Buoyancy [t]", "status": "Status"})
                st.dataframe(mem.round(2), hide_index=True)
                st.caption(f"Total buoyancy {mem['Buoyancy [t]'].sum():,.0f} t vs weight {T['h']['W']:,.0f} t.")
            with st.expander("Line loads - connection points and force directions"):
                ld = pd.DataFrame([dict(Line=l_.name, **{"x [m]": l_.point[0], "y [m]": l_.point[1], "z [m]": l_.point[2]},
                                        **{"Heading [°]": l_.heading_deg, "Elevation [°]": l_.elevation_deg,
                                           "Type": "Tow leg" if l_.mode == "tow" else "Fixed pull",
                                           "Tension [kN] / share": (f"share {l_.share:.2f}" if l_.mode == "tow"
                                                                    else f"{l_.tension_kn:,.0f} kN")}) for l_ in params.lines])
                st.dataframe(ld.round(2), hide_index=True)
            if T["h"]["openings"]:
                with st.expander("Openings - height above the waterline at the floating condition"):
                    od = pd.DataFrame(T["h"]["openings"]).rename(columns={"name": "Opening",
                                                                         "above_water_m": "Above water [m]"})
                    od["Status"] = np.where(od["Above water [m]"] > 0, "above water", "UNDER WATER")
                    st.dataframe(od.round(2), hide_index=True)
    for w_ in warnings:
        st.caption(f"⚠ {w_}")
    geo_case_box = st.container()          # filled after tab 7 has read the case file

# ----------------------------------------------------------------------------
# Tab 4: results
# ----------------------------------------------------------------------------
res = st.session_state.res
with tab_res:
    if res is None:
        st.info("Press **Run analysis** in the sidebar.")
    else:
        if st.session_state.res_hash != cur_hash:
            st.warning("Inputs have changed since this analysis was run. Press Run analysis to update.")
        S, U = res["summary"], res["upright"]
        scen = "DAMAGED" if res["damaged"] else "INTACT"
        if S["passed"]:
            st.success(f"{scen}: all headings meet the criteria set in tab 2.")
        else:
            st.error(f"{scen}: one or more headings do NOT meet the criteria. Governing wind heading "
                     f"{S['governing_beta']:.0f}°.")
        if S["no_equilibrium"]:
            st.error("At least one heading has no static equilibrium within the heel range (heeling exceeds righting).")
        if S["unconverged"] or S["trim_unstable"]:
            st.warning(f"{S['unconverged']} unconverged and {S['trim_unstable']} trim-unstable states inside the "
                       "assessed range - results at those angles are unreliable.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min GM", f"{S['gm_min']:.2f} m", help=f"Criterion ≥ {crit.gm_min:.2f} m (small-angle, free to trim)")
        m2.metric("Max static heel", fmt(S["heel_max"], 1, "°"), help=f"Criterion ≤ {crit.heel_max_deg:.1f}°")
        m3.metric("Min area ratio", fmt(S["ratio_min"], 2), help=f"Criterion ≥ {crit.ratio_min:.2f}")
        m4.metric("Min downflood angle", fmt(S["df_min"], 1, "°") if S["df_min"] is not None else "n/a")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Line pull (net horizontal)", fmt(S["tow_kn"], 0, " kN") if S["tow_kn"] is not None else "n/a",
                  help=f"{S['tow_kn'] / js.G_ACC:.1f} t" if S["tow_kn"] else None)
        n2.metric("Draft to lowest buoyant point", f"{U['draft']:.2f} m")
        n3.metric("Reserve buoyancy", f"{U['reserve_buoyancy_pct']:.0f} %")
        n4.metric("Total weight", f"{res['W']:,.0f} t")

        F = res.get("float")
        if F is not None:
            st.markdown("**Set-down float check** (level, no wind or tow)")
            if F["passed"]:
                st.success("Set-down float condition meets the criteria set in tab 2.")
            else:
                st.error("Set-down float condition does NOT meet the criteria set in tab 2 (or has no stable upright attitude).")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Waterline above base", f"{F['draft_total']:.2f} m", help="Waterline height above the lowest point of the structure.")
            f2.metric("Seabed clearance", fmt(F["clearance"], 2, " m") if F["clearance"] is not None else "n/a",
                      help=("Water depth + tide - waterline height. " if F["clearance"] is not None else "Enter a water depth in tab 2.")
                      + (f"Criterion ≥ {crit.clear_min_m:.2f} m" if crit.clear_min_m > 0 else ""))
            f3.metric("Min tank above water", fmt(F["emerged_min"], 2, " m"),
                      help=f"Criterion ≥ {crit.emerged_min_m:.2f} m" if crit.emerged_min_m > 0 else None)
            f4.metric("Freeboard to top of jacket", f"{F['freeboard_top']:.1f} m")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GM (level, no loads)", f"{F['gm_min']:.2f} m")
            g2.metric("Lowest opening above water", fmt(F["opening_min"], 2, " m") if F["opening_min"] is not None else "n/a")
            g3.metric("Ballast headroom", fmt(F["ballast_headroom_t"], 0, " t") if F["ballast_headroom_t"] is not None else "n/a",
                      help=("Ballast (+) that can be added at the CoG before the " + str(F["governing_limit"]) + " limit is reached; "
                            "negative = weight to shed.") if F["governing_limit"] else "Set a clearance or tank-emergence criterion in tab 2.")
            g4.metric("Governing limit", str(F["governing_limit"] or "n/a"))
            trows = [{"Tank": t["name"], "Length [m]": f"{t['length_m']:.1f}", "Submerged [%]": f"{t['sub_pct']:.0f}",
                      "Above water [m]": f"{t['emerged_m']:.2f}", "Buoyancy [t]": f"{t['buoyancy_t']:.0f}"} for t in F["tanks"]]
            with st.expander("Tank table"):
                st.dataframe(pd.DataFrame(trows), hide_index=True)
            if abs(F["trim_deg"]) > 1.0:
                st.caption(f"Level trim is {F['trim_deg']:.1f}°; clearance assumes a level waterline, so check it with a full analysis.")

        rows = []
        for h in res["heads"]:
            a = h["analysis"]
            rows.append({
                "Wind toward [°]": f"{h['beta']:.0f}", "GM [m]": fmt(a["gm"], 2), "Static heel [°]": fmt(a["theta_s"]),
                "Trim at eq. [°]": fmt(a["trim_at_s"], 2), "2nd intercept [°]": fmt(a["theta_2"]),
                "Downflood [°]": fmt(a["theta_df"]), "Limit angle [°]": fmt(a["theta_lim"]),
                "Area ratio": fmt(a["ratio"], 2),
                "GZmax [m] @ [°]": f"{fmt(a['gz_max'], 2)} @ {fmt(a['phi_gz_max'])}",
                "Line pull [kN]": fmt(a["tow_kn"], 0), "Wind [kN]": fmt(a["wind_kn"], 0),
                "Status": "PASS" if a["passed"] else "FAIL"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        labels = [f"{h['beta']:.0f}°" for h in res["heads"]]
        gov = labels.index(f"{S['governing_beta']:.0f}°")
        sel = st.selectbox("Heading to plot", labels, index=gov)
        hsel = res["heads"][labels.index(sel)]
        show(curve_figure(hsel, crit))
        if hsel["analysis"]["note"]:
            st.caption(hsel["analysis"]["note"])
        if hsel["analysis"]["sunk_beyond"] is not None:
            st.caption(f"Buoyancy is insufficient beyond {hsel['analysis']['sunk_beyond']:.1f}° heel.")

        an_s, sw_s = hsel["analysis"], hsel["sweep"]
        st.markdown(f"**Technical data - wind toward {hsel['beta']:.0f}°**")
        c_l, c_r = st.columns(2)
        with c_l:
            crow = [("GM (small angle)", f"≥ {crit.gm_min:.2f} m", fmt(an_s["gm"], 2, " m"), an_s["pass_gm"]),
                    ("Static heel", f"≤ {crit.heel_max_deg:.1f}°", fmt(an_s["theta_s"], 1, "°"), an_s["pass_heel"]),
                    ("Area ratio", f"≥ {crit.ratio_min:.2f}", fmt(an_s["ratio"], 2), an_s["pass_ratio"])]
            if crit.df_min_deg > 0:
                crow.append(("Downflood angle", f"≥ {crit.df_min_deg:.1f}°", fmt(an_s["theta_df"], 1, "°"), an_s["pass_df"]))
            st.dataframe(pd.DataFrame([dict(Criterion=a, Required=b, Actual=c, Status="PASS" if d else "FAIL")
                                       for a, b, c, d in crow]), hide_index=True)
        with c_r:
            if an_s["theta_s"] is not None:
                okk = ~np.isnan(sw_s["gz"])

                def at(k):
                    v = np.interp(an_s["theta_s"], sw_s["phi_deg"][okk], sw_s[k][okk])
                    return float(v)
                wg = res["W"] * js.G_ACC
                ho = at("h_open") if not np.all(np.isnan(sw_s["h_open"])) else None
                eq = [("Heel / trim at equilibrium", f"{an_s['theta_s']:.2f}° / {an_s['trim_at_s']:.2f}°"),
                      ("Draft at equilibrium", f"{at('draft'):.2f} m"),
                      ("Wind force", f"{at('Fw'):,.0f} kN"),
                      ("Line pull (net horizontal)", f"{at('T'):,.0f} kN"),
                      *([("Net vertical line force", f"{at('Fz'):,.0f} kN")] if abs(at("Fz")) > 0.5 else []),
                      ("Heeling moment", f"{at('M_H'):,.0f} kN·m"),
                      ("Righting moment (GZ·W·g)", f"{at('gz') * wg:,.0f} kN·m"),
                      ("Lowest opening above water", "n/a" if ho is None else f"{ho:.2f} m"),
                      ("GZmax @ heel", f"{fmt(an_s['gz_max'], 2)} m @ {fmt(an_s['phi_gz_max'], 1)}°"),
                      ("Second intercept", fmt(an_s["theta_2"], 1, "°")),
                      ("Limit angle used", fmt(an_s["theta_lim"], 1, "°")),
                      ("Residual area (θs→limit)", fmt(an_s["residual"], 3, " m·rad"))]
                st.dataframe(pd.DataFrame(eq, columns=["Quantity", "Value"]), hide_index=True)
            else:
                st.info("No static equilibrium at this heading.")

        ratios = [h["analysis"]["ratio"] or 0.0 for h in res["heads"]]
        fb = go.Figure(go.Bar(x=labels, y=ratios, marker_color=["#2ca02c" if r >= crit.ratio_min else "#d62728" for r in ratios]))
        fb.add_hline(y=crit.ratio_min, line=dict(dash="dash"), annotation_text=f"required {crit.ratio_min:.2f}")
        fb.update_layout(title="Area ratio by wind heading", xaxis_title="Wind toward [°]", yaxis_title="Area ratio",
                         template="plotly_white", height=320)
        show(fb)

        ft = go.Figure(go.Scatter(x=hsel["sweep"]["phi_deg"], y=hsel["sweep"]["theta_deg"], mode="lines+markers"))
        ft.update_layout(title=f"Free-trim angle vs heel - wind toward {hsel['beta']:.0f}°", xaxis_title="Heel [deg]",
                         yaxis_title="Trim [deg]", template="plotly_white", height=300)
        show(ft)

        curves = []
        for h in res["heads"]:
            sw = h["sweep"]
            curves.append(pd.DataFrame({"wind_toward_deg": h["beta"], "heel_deg": sw["phi_deg"], "GZ_m": sw["gz"],
                                        "heeling_arm_m": sw["hl"], "trim_deg": sw["theta_deg"],
                                        "waterline_z_m": sw["zw"], "min_opening_above_water_m": sw["h_open"],
                                        "line_pull_kN": sw["T"], "line_vertical_kN": sw["Fz"],
                                        "wind_force_kN": sw["Fw"]}))
        st.download_button("📥 Download all curves (CSV)", pd.concat(curves).to_csv(index=False).encode(),
                           file_name="jacket_tow_curves.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 5: sensitivity - number of tanks attached, tow connection
# ----------------------------------------------------------------------------
STATUS_COL = {"PASS": "#2ca02c", "FAIL": "#ff7f0e", "CAPSIZES": "#d62728", "SINKS": "#7f1d1d"}


def case_label(c: dict) -> str:
    if c["status"] != sv.OK:
        return c["status"].upper()
    if c.get("listed"):
        return f"LISTED {c['tilt']:.1f}°"
    return "PASS" if c["passed"] else "FAIL"


def label_colour(lab: str) -> str:
    return STATUS_COL.get(lab, "#ff7f0e")


def lost_text(c: dict | None) -> str:
    if not c:
        return "-"
    lost = c.get("lost", ())
    return "none" if not lost else ", ".join(lost)


def tank_table(rows: list[dict], which: str = "worst") -> pd.DataFrame:
    out = []
    for r in rows:
        c = r.get(which)
        cnt = r["counts"]
        base = {"Tanks attached": f"{r['k']} of {r['n']}",
                "Arrangements (stable / capsize / sink)": f"{cnt[sv.OK]} / {cnt[sv.CAPSIZES]} / {cnt[sv.SINKS]}"}
        if r.get("skipped") or c is None:
            out.append({**base, "Tanks lost": "-", "Result": "SINKS", "No-wind list [°]": "-", "Min GM [m]": "-",
                        "Max static heel [°]": "-", "Min area ratio": "-", "Reserve buoyancy [%]": "-"})
            continue
        ok = c["status"] == sv.OK
        out.append({**base, "Tanks lost": lost_text(c), "Result": case_label(c),
                    "No-wind list [°]": fmt(c["tilt"], 1) if ok else "-",
                    "Min GM [m]": fmt(c["gm_min"], 2) if ok else "-",
                    "Max static heel [°]": (fmt(c["heel_max"], 1) if c["heel_max"] is not None else "none") if ok and not c["listed"] else "-",
                    "Min area ratio": fmt(c["ratio_min"], 2) if ok and not c["listed"] else "-",
                    "Reserve buoyancy [%]": fmt(c["reserve_pct"], 0) if c["reserve_pct"] is not None else "-"})
    return pd.DataFrame(out)


def curve_overlay(entries: list[tuple[str, dict]], beta: float, title: str, one_gz: bool) -> go.Figure | None:
    """GZ and heeling-arm curves of several cases at one wind heading."""
    fig = go.Figure()
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    drawn = 0
    for i, (lab, c) in enumerate(entries):
        if c.get("res") is None:
            continue
        hd = next((h for h in c["res"]["heads"] if abs(h["beta"] - beta) < 1e-6), None)
        if hd is None:
            continue
        sw = hd["sweep"]
        col = pal[i % len(pal)]
        if not (one_gz and drawn > 0):
            fig.add_trace(go.Scatter(x=sw["phi_deg"], y=sw["gz"], mode="lines", name="Righting arm GZ" if one_gz else f"GZ - {lab}",
                                     line=dict(color="#555" if one_gz else col, width=3)))
        fig.add_trace(go.Scatter(x=sw["phi_deg"], y=sw["hl"], mode="lines", name=f"Heeling arm - {lab}",
                                 line=dict(color=col, width=2.5, dash="dash")))
        drawn += 1
    if drawn == 0:
        return None
    # zoom on the working range: from 0 to just past the furthest limit angle, y up to the larger of GZ / heeling
    x_hi, y_hi = 10.0, 0.5
    for _, c in entries:
        if c.get("res") is None:
            continue
        hd = next((h for h in c["res"]["heads"] if abs(h["beta"] - beta) < 1e-6), None)
        if hd is None:
            continue
        lim = hd["analysis"]["theta_lim"] or 30.0
        x_hi = max(x_hi, lim + 8.0)
        sw = hd["sweep"]
        m = (sw["phi_deg"] >= 0) & (sw["phi_deg"] <= x_hi) & ~np.isnan(sw["gz"])
        if m.any():
            y_hi = max(y_hi, float(np.nanmax(sw["gz"][m])), float(np.nanmax(sw["hl"][m])))
    fig.update_xaxes(range=[0, x_hi])
    fig.update_yaxes(range=[-0.1 * y_hi, 1.15 * y_hi])
    fig.update_layout(title=title, xaxis_title="Heel angle [deg]", yaxis_title="Lever arm [m]", hovermode="x unified",
                      template="plotly_white", height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


with tab_sens:
    st.subheader("How the tank connections and the tow connection change stability")
    st.caption("Runs the full heel sweep for each variation of the inputs in tabs 1 and 2, so the comparison always matches "
               "your current data. A tank counts as 'lost' when it is not attached: it gives no buoyancy, wind area or drag. "
               "'Capsizes' means there is no stable floating attitude with no wind or tow.")
    if is_example:
        st.warning("These tables hold the SYNTHETIC illustrative jacket. The trends show what the tool does; the numbers "
                   "are not Conrad Mako results.")
    if errors:
        st.error("Fix the input errors shown in the sidebar first.")
    else:
        ss = st.session_state
        s_betas = np.arange(0.0, 360.0, float(ss["a_step"]))
        s_grid = np.unique(np.concatenate([np.arange(ss["a_phi_min"], ss["a_phi_max"] + 1e-9,
                                                     max(float(ss["a_phi_step"]), 5.0)), [0.0]]))
        s_dam = str(ss["a_scenario"]).startswith("Damaged")
        st.caption(f"Settings used: wind headings every {ss['a_step']:.1f}°, heel step {max(float(ss['a_phi_step']), 5.0):.1f}° "
                   f"(coarsened for speed), scenario {'damaged' if s_dam else 'intact'}.")

        # ---- A: number of tanks ------------------------------------------------
        st.markdown("### A · Number of tanks attached")
        n_tanks = len(sv.tank_names(els))
        st.caption(f"{n_tanks} buoyant members found in tab 1. Every combination of lost tanks is tested for each count "
                   "(an even sample of 12 if there are more combinations).")
        max_lost = int(st.number_input("Test up to this many tanks lost", min_value=1, max_value=max(1, n_tanks - 1),
                                       value=max(1, min(3, n_tanks - 1)), step=1, key="sens_max_lost"))
        if st.button("▶ Run tank-count study", key="sens_run_a", disabled=n_tanks < 2):
            bar = st.progress(0.0, text="Starting...")
            rows_a = sv.tank_study(els, wts, ops, params, crit, s_betas, s_grid, s_dam, max_lost=max_lost,
                                   progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            bar.empty()
            ss.sens_a = dict(key=cur_hash, rows=rows_a)
        sa = ss.get("sens_a")
        if sa:
            if sa["key"] != cur_hash:
                st.warning("Inputs have changed since this study was run. Run it again to update.")
            rows_a = sa["rows"]
            df_a = tank_table(rows_a, "worst")
            st.markdown("**Worst arrangement for each tank count**")
            st.dataframe(df_a, hide_index=True)
            with st.expander("Best arrangement for each tank count"):
                st.dataframe(tank_table(rows_a, "best"), hide_index=True)
            labs = [f"{r['k']} of {r['n']}" for r in rows_a]
            vals, txt, cols = [], [], []
            for r in rows_a:
                c = r.get("worst")
                lab = case_label(c) if c is not None else "SINKS"
                vals.append(c["ratio_min"] if c is not None and c["status"] == sv.OK and not c["listed"]
                            and c["ratio_min"] is not None else 0.0)
                txt.append(f"{vals[-1]:.2f}" if lab in ("PASS", "FAIL") else lab)
                cols.append(label_colour(lab))
            fig_a = go.Figure(go.Bar(x=labs, y=vals, text=txt, textposition="outside", marker_color=cols))
            fig_a.add_hline(y=crit.ratio_min, line=dict(dash="dash"), annotation_text=f"required {crit.ratio_min:.2f}")
            fig_a.update_layout(title="Worst-arrangement area ratio by number of tanks attached", template="plotly_white",
                                xaxis_title="Tanks attached", yaxis_title="Min area ratio", height=360)
            show(fig_a)
            ok_cases = [(f"{r['k']} of {r['n']} (lost {lost_text(r['worst'])})", r["worst"]) for r in rows_a
                        if r.get("worst") is not None and r["worst"]["status"] == sv.OK and not r["worst"]["listed"]]
            if len(ok_cases) >= 2:
                bsel = min(ok_cases, key=lambda t: t[1]["ratio_min"] if t[1]["ratio_min"] is not None else -1)[1]["gov_beta"]
                fo = curve_overlay(ok_cases, bsel, f"GZ and heeling arm - wind toward {bsel:.0f}°", one_gz=False)
                if fo is not None:
                    show(fo)
            st.download_button("📥 Tank study (CSV)", df_a.to_csv(index=False).encode(), file_name="tank_count_study.csv",
                               mime="text/csv", key="dl_a")
            for r in rows_a:
                for c in ([r.get("worst")] if r.get("worst") else []):
                    if c["note"] and c["status"] == sv.OK:
                        st.caption(f"{r['k']} of {r['n']} (lost {lost_text(c)}): {c['note']}")

        # ---- B: tow connection --------------------------------------------------
        st.markdown("### B · Tow connection (point and force direction)")
        st.caption("Each case moves the tow line (the first line): its connection point, heading and elevation. Any extra "
                   "lines from tab 2 stay as entered. GM does not change with the tow line; the static heel and the area "
                   "ratio do.")
        pl_ = sv.primary_line(params)
        seed_key = json.dumps([snap["elements"], list(pl_.point), pl_.heading_deg, pl_.elevation_deg], sort_keys=True, default=str)
        if ss.get("tow_seed") != seed_key:
            ss.tow_seed = seed_key
            ss.t_tow = pd.DataFrame(sv.default_tow_cases(els, params),
                                    columns=["name", "x", "y", "z", "heading_deg", "elevation_deg"])
            ss.tow_ver = ss.get("tow_ver", 0) + 1
        tow_df = st.data_editor(
            ss.t_tow, num_rows="dynamic", hide_index=True, key=f"ed_tow_{ss.get('tow_ver', 0)}",
            column_config={"name": st.column_config.TextColumn("Tow connection case", required=True),
                           "x": st.column_config.NumberColumn("x [m]", format="%.2f"),
                           "y": st.column_config.NumberColumn("y [m]", format="%.2f"),
                           "z": st.column_config.NumberColumn("z [m]", format="%.2f"),
                           "heading_deg": st.column_config.NumberColumn("Heading [°]", format="%.1f"),
                           "elevation_deg": st.column_config.NumberColumn("Elevation [°]", format="%.1f",
                                                                          min_value=-89.0, max_value=89.0)})
        tow_cases = [dict(name=str(r["name"]), x=float(r["x"]), y=float(r["y"]), z=float(r["z"]),
                          heading_deg=float(r["heading_deg"]) if _num_ok(r.get("heading_deg")) else float(pl_.heading_deg),
                          elevation_deg=float(r["elevation_deg"]) if _num_ok(r.get("elevation_deg")) else float(pl_.elevation_deg))
                     for r in records(tow_df) if all(_num_ok(r.get(c)) for c in ("x", "y", "z")) and r.get("name")]
        if st.button("▶ Run tow-connection study", key="sens_run_b", disabled=not tow_cases):
            bar = st.progress(0.0, text="Starting...")
            rows_b = sv.tow_study(els, wts, ops, params, crit, s_betas, s_grid, tow_cases, s_dam,
                                  progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            bar.empty()
            ss.sens_b = dict(key=cur_hash, rows=rows_b, cases=tow_cases)
        sb_ = ss.get("sens_b")
        if sb_:
            if sb_["key"] != cur_hash:
                st.warning("Inputs have changed since this study was run. Run it again to update.")
            rows_b = sb_["rows"]
            tb = []
            for c in rows_b:
                ok = c["status"] == sv.OK
                tb.append({"Tow connection": c["tow"]["name"], "x [m]": c["tow"]["x"], "y [m]": c["tow"]["y"],
                           "z [m]": c["tow"]["z"], "Heading [°]": c["tow"].get("heading_deg"),
                           "Elevation [°]": c["tow"].get("elevation_deg"), "Result": case_label(c),
                           "Min GM [m]": fmt(c["gm_min"], 2) if ok else "-",
                           "Max static heel [°]": (fmt(c["heel_max"], 1) if c["heel_max"] is not None else "none") if ok else "-",
                           "Min area ratio": fmt(c["ratio_min"], 2) if ok else "-",
                           "Governing heading [°]": fmt(c["gov_beta"], 0) if ok else "-",
                           "Line pull [kN]": fmt(c["tow_kn"], 0) if ok and c["tow_kn"] is not None else "-"})
            df_b = pd.DataFrame(tb)
            st.dataframe(df_b, hide_index=True)
            okb = [c for c in rows_b if c["status"] == sv.OK]
            if okb:
                names_b = [c["tow"]["name"] for c in okb]
                heel_v = [c["heel_max"] if c["heel_max"] is not None else float("nan") for c in okb]
                ratio_v = [c["ratio_min"] if c["ratio_min"] is not None else 0.0 for c in okb]
                col1, col2 = st.columns(2)
                f1 = go.Figure(go.Bar(x=names_b, y=heel_v, marker_color=["#2ca02c" if (h <= crit.heel_max_deg) else "#d62728" for h in heel_v]))
                f1.add_hline(y=crit.heel_max_deg, line=dict(dash="dash"), annotation_text=f"limit {crit.heel_max_deg:.0f}°")
                f1.update_layout(title="Max static heel by tow connection", template="plotly_white", height=340, yaxis_title="deg")
                f2 = go.Figure(go.Bar(x=names_b, y=ratio_v, marker_color=["#2ca02c" if r >= crit.ratio_min else "#d62728" for r in ratio_v]))
                f2.add_hline(y=crit.ratio_min, line=dict(dash="dash"), annotation_text=f"required {crit.ratio_min:.2f}")
                f2.update_layout(title="Min area ratio by tow connection", template="plotly_white", height=340, yaxis_title="ratio")
                with col1:
                    show(f1)
                with col2:
                    show(f2)
                worst_b = min(okb, key=lambda c: c["ratio_min"] if c["ratio_min"] is not None else -1)
                fo = curve_overlay([(c["tow"]["name"], c) for c in okb], worst_b["gov_beta"],
                                   f"Same GZ, different heeling arm - wind toward {worst_b['gov_beta']:.0f}°", one_gz=True)
                if fo is not None:
                    show(fo)
            st.download_button("📥 Tow study (CSV)", df_b.to_csv(index=False).encode(), file_name="tow_connection_study.csv",
                               mime="text/csv", key="dl_b")

        # ---- C: both together ----------------------------------------------------
        st.markdown("### C · Tank count × tow connection")
        configs = [(f"All {n_tanks} attached", list(els))]
        if sa and sa["key"] == cur_hash:
            for r in sa["rows"]:
                c = r.get("worst_ok")
                if r["k"] < r["n"] and c is not None:
                    configs.append((f"{r['k']} of {r['n']} (lost {lost_text(c)})", sv.detach(els, c["lost"])))
        if len(configs) < 2:
            st.caption("No reduced tank set floats stably, so there is nothing to cross with the tow connections."
                       if sa and sa["key"] == cur_hash else
                       "Run study A first; this grid crosses its tank sets with the tow connections above.")
        elif not tow_cases:
            st.caption("Add at least one tow connection case above.")
        else:
            if st.button("▶ Run tank × tow grid", key="sens_run_c"):
                bar = st.progress(0.0, text="Starting...")
                gm_ = sv.matrix_study(configs, wts, ops, params, crit, s_betas, s_grid, tow_cases, s_dam,
                                      progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                bar.empty()
                ss.sens_c = dict(key=cur_hash, grid=gm_, configs=[c[0] for c in configs], cases=tow_cases)
        sc = ss.get("sens_c")
        if sc and len(configs) >= 2:
            if sc["key"] != cur_hash:
                st.warning("Inputs have changed since this grid was run. Run it again to update.")
            z, t_ = [], []
            for row in sc["grid"]:
                z.append([1.0 if (c["status"] == sv.OK and c["passed"]) else (0.5 if c["status"] == sv.OK else 0.0) for c in row])
                t_.append([(case_label(c) if c["listed"] else (f"{c['ratio_min']:.2f}" if c["ratio_min"] is not None else "no equil."))
                           if c["status"] == sv.OK else c["status"].upper() for c in row])
            fh = go.Figure(go.Heatmap(z=z, x=[c["name"] for c in sc["cases"]], y=sc["configs"], text=t_, texttemplate="%{text}",
                                      colorscale=[[0, "#d62728"], [0.5, "#ffb347"], [1, "#2ca02c"]], zmin=0, zmax=1,
                                      showscale=False, xgap=3, ygap=3))
            fh.update_layout(title="Min area ratio (green = meets all criteria, orange = fails or lists, red = capsizes / sinks)",
                             template="plotly_white", height=140 + 70 * len(sc["configs"]), yaxis=dict(autorange="reversed"))
            show(fh)

        # ---- D: number of lines --------------------------------------------------
        st.markdown("### D · Number of lines (tow legs and pull-in lines)")
        n_lines = len(params.lines)
        st.caption("Adds the lines one at a time in table order (the tow line first) to show what each extra connection "
                   "does: net pull, vertical component and the resulting heel and area ratio.")
        if n_lines < 2:
            st.caption("Add pull-in lines or extra tow legs under 'Additional line loads' in tab 2 to use this study.")
        else:
            if st.button("▶ Run line-count study", key="sens_run_d"):
                bar = st.progress(0.0, text="Starting...")
                rows_d = sv.line_count_study(els, wts, ops, params, crit, s_betas, s_grid, s_dam,
                                             progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                bar.empty()
                ss.sens_d = dict(key=cur_hash, rows=rows_d)
            sd = ss.get("sens_d")
            if sd:
                if sd["key"] != cur_hash:
                    st.warning("Inputs have changed since this study was run. Run it again to update.")
                td = []
                for c in sd["rows"]:
                    ok = c["status"] == sv.OK
                    td.append({"Lines applied": c["n_lines"], "Last line added": c["added"], "Result": case_label(c),
                               "Min GM [m]": fmt(c["gm_min"], 2) if ok else "-",
                               "Max static heel [°]": (fmt(c["heel_max"], 1) if c["heel_max"] is not None else "none") if ok else "-",
                               "Min area ratio": fmt(c["ratio_min"], 2) if ok else "-",
                               "Net line pull [kN]": fmt(c["tow_kn"], 0) if ok and c["tow_kn"] is not None else "-"})
                df_d = pd.DataFrame(td)
                st.dataframe(df_d, hide_index=True)
                okd = [c for c in sd["rows"] if c["status"] == sv.OK and c["ratio_min"] is not None]
                if okd:
                    fd = go.Figure(go.Bar(x=[f"{c['n_lines']} line(s)<br>+ {c['added']}" for c in okd],
                                          y=[c["ratio_min"] for c in okd],
                                          marker_color=["#2ca02c" if c["ratio_min"] >= crit.ratio_min else "#d62728" for c in okd]))
                    fd.add_hline(y=crit.ratio_min, line=dict(dash="dash"), annotation_text=f"required {crit.ratio_min:.2f}")
                    fd.update_layout(title="Min area ratio as lines are added", template="plotly_white", height=340,
                                     yaxis_title="ratio")
                    show(fd)
                st.download_button("📥 Line-count study (CSV)", df_d.to_csv(index=False).encode(),
                                   file_name="line_count_study.csv", mime="text/csv", key="dl_d")

# ----------------------------------------------------------------------------
# Tab 5: report
# ----------------------------------------------------------------------------
with tab_rep:
    if res is None:
        st.info("Run the analysis first.")
    else:
        if st.session_state.res_hash != cur_hash:
            st.warning("The report will use the inputs from the last run, not the current edits.")
        r1, r2, r3 = st.columns(3)
        prepared = r1.text_input("Prepared by", value=st.session_state.engineer, key="rep_prepared")
        checked = r2.text_input("Checked by", value="", key="rep_checked")
        rev = r3.text_input("Revision", value="A", key="rep_rev")
        pdf = build_pdf(st.session_state.res_snap, res, st.session_state.get("res_meta", {}), prepared, checked, rev)
        st.download_button("📄 Download draft report (A4 PDF)", data=pdf, file_name="Jacket_tow_stability_draft.pdf",
                           mime="application/pdf")
        with st.expander("Assumptions and limitations"):
            for t in ASSUMPTIONS:
                st.markdown(f"- {t}")

# ----------------------------------------------------------------------------
# Tab 7: buoyancy budget and modules from a case file (project data stays outside the repository)
# ----------------------------------------------------------------------------
with tab_buoy:
    try:
        buoyancy_ui.render()
    except Exception as _exc:    # this tab must never take the rest of the app down
        st.error(f"The buoyancy tab hit a problem: {_exc}")

with geo_case_box:
    _raw = st.session_state.get("case_raw")
    if _raw and (_raw.get("structure") or {}).get("elements"):
        st.divider()
        st.subheader("Platform from the loaded case file")
        st.caption("The picture above is the jacket from tab 1. This one is the platform in the case file loaded in tab 7.")
        try:
            buoyancy_ui.render_case_platform(_raw)
        except Exception as _exc:
            st.error(f"Could not draw the case-file platform: {_exc}")

with tab_tow:
    try:
        towplan_ui.render()
    except Exception as _exc:   # keep the other tabs usable
        st.error(f"Tow plan tab failed: {_exc}")
