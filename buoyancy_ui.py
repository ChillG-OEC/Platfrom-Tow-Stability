"""Streamlit tab: buoyancy budget, environment and buoyancy-module cases from a case file.

The case file (JSON, see docs/case_template.json) holds project data and stays outside the
repository.  Sections used when present:
  budget       weights and buoyancy sources (sizes only, no coordinates needed)
  environment  operating limits and metocean statistics to show next to the results
  structure + modules  member geometry and switchable buoyancy modules (full float check)
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import budget as bd
import buoyancy as bu
import case_io
import jacket_stability as js
import metocean as mo

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
GROUP_LABEL = {"leg": "Legs", "brace": "Braces", "tank": "Tanks", "other": "Other"}


def _show(fig: go.Figure) -> None:
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def _read_raw():
    up = st.file_uploader("Case file (.json)", type="json", key="bu_up")
    path = st.text_input("...or path to a case file on this computer", key="bu_path")
    try:
        if up is not None:
            return json.loads(up.getvalue().decode("utf-8")), up.name
        if path.strip():
            p = Path(path.strip()).expanduser()
            return json.loads(p.read_text(encoding="utf-8")), p.name
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.error(f"Could not read that case file: {exc}")
    return None, ""


def _budget_section(raw: dict, key: str) -> None:
    b = raw["budget"]
    st.subheader("A · Buoyancy budget (weights and sizes only)")
    st.caption("Capacity = displacement if everything sealed were fully submerged (outside volume; wall steel is in the "
               "weight). Reserve = (capacity − W) / W using the FACTORED weight. Needs no coordinates, so it is only a "
               "first screen: waterline, GM and clearance need the member geometry.")
    wcs = b.get("weight_cases") or []
    if not wcs:
        st.warning("The case file has no budget.weight_cases.")
        return
    c1, c2, c3 = st.columns(3)
    names = [w["name"] for w in wcs]
    wc = wcs[names.index(c1.selectbox("Weight case", names, key=f"bu_wc_{key}"))]
    r_int = c2.number_input("Reserve required, intact [%]", 0.0, 200.0, float(raw.get("reserve_min_pct", 10.0)), 1.0,
                            key=f"bu_ri_{key}")
    r_dam = c3.number_input("Reserve required, damaged [%]", 0.0, 200.0, float(raw.get("reserve_damaged_min_pct", 5.0)), 1.0,
                            key=f"bu_rd_{key}")
    k1, k2 = st.columns(2)
    steel_in = k1.number_input("Tank steel already in the weight [t]", 0.0, float(wc["weight_t"]),
                               min(float(wc.get("tank_steel_t", b.get("tank_steel_in_weight_t", 0.0))), float(wc["weight_t"])), 1.0,
                               key=f"bu_si_{key}_{wc['name']}", help="From the weight report (tank items). Used to rescale the steel "
                               "when the tank volume changes.")
    k_steel = k2.number_input("Steel per m³ of extra tank volume [t/m³] (0 = ignore)", 0.0, 1.0,
                              float(b.get("tank_steel_t_per_m3", 0.0)), 0.01, key=f"bu_k_{key}",
                              help="ASSUMED. Set from the tank design; 0 ignores the extra steel the extra volume brings.")
    srcs = b.get("sources") or []
    rows = bd.sources_table(srcs)
    df = pd.DataFrame(rows)
    df.insert(1, "Sealed share", [float(s.get("share_default", 0.0)) for s in srcs])
    df["Volume [m³]"] = df["volume_m3"]
    show = df[["name", "group", "Sealed share", "od_mm", "length_m", "Volume [m³]", "note"]].rename(
        columns={"name": "Source", "group": "Group", "od_mm": "OD [mm]", "length_m": "Length [m]", "note": "Note"})
    ed = st.data_editor(show, key=f"bu_ed_{key}", hide_index=True, width="stretch" if _PLOTLY_HAS_WIDTH else None,
                        disabled=["Source", "Group", "OD [mm]", "Length [m]", "Note"],
                        column_config={"Sealed share": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05,
                                                                                     help="Fraction of this source that is sealed and buoyant"),
                                       "Volume [m³]": st.column_config.NumberColumn(min_value=0.0, format="%.1f",
                                                                                    help="Edit tank volumes here; pipe volumes come from OD and length")})
    srcs2 = []
    for s, v in zip(srcs, ed["Volume [m³]"]):
        s2 = dict(s)
        s2["volume_m3"] = float(v)
        srcs2.append(s2)
    share = {s["name"]: float(f) for s, f in zip(srcs, ed["Sealed share"])}
    try:
        res = bd.budget(float(wc["weight_t"]), srcs2, share, reserve_min_pct=r_int,
                        tank_steel_in_weight_t=steel_in, tank_steel_t_per_m3=k_steel)
    except ValueError as exc:
        st.error(str(exc))
        return
    m = st.columns(4)
    m[0].metric("Weight (factored)", f"{res['weight_t']:.1f} t")
    m[1].metric("Buoyancy capacity", f"{res['capacity_t']:.1f} t")
    m[2].metric("Reserve", f"{res['reserve_pct']:.1f} %", delta=f"{res['reserve_pct'] - r_int:+.1f} vs {r_int:.0f} % required",
                delta_color="normal")
    tv = res["tank_volume_needed_m3"]
    m[3].metric("Tank volume needed in total", "infeasible" if tv is None else f"{tv:.0f} m³",
                help=f"Listed tank volume: {res['tank_volume_listed_m3']:.0f} m³")
    if res["meets_reserve"]:
        st.success(f"Meets {r_int:.0f} % reserve: capacity {res['capacity_t']:.0f} t ≥ required {res['required_t']:.0f} t.")
    else:
        st.warning(f"Short by {res['shortfall_t']:.0f} t of the {res['required_t']:.0f} t required "
                   f"({r_int:.0f} % reserve on {res['weight_t']:.0f} t).")
    fig = go.Figure()
    for g, val in res["by_group_t"].items():
        fig.add_bar(x=["Capacity"], y=[val], name=GROUP_LABEL.get(g, g))
    fig.add_bar(x=["Weight + reserve"], y=[res["weight_t"]], name="Weight")
    fig.add_bar(x=["Weight + reserve"], y=[res["required_t"] - res["weight_t"]], name="Reserve required")
    fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="t")
    _show(fig)
    loss = bd.loss_cases(float(wc["weight_t"]), srcs2, share, reserve_min_pct=r_dam)
    if loss:
        st.markdown(f"**Loss of one source (damaged, {r_dam:.0f} % reserve required)**")
        ld = pd.DataFrame(loss).rename(columns={"lost": "Source lost", "lost_t": "Lost [t]", "capacity_t": "Capacity left [t]",
                                                "reserve_pct": "Reserve [%]", "meets": "Meets"})
        st.dataframe(ld.style.format({"Lost [t]": "{:.1f}", "Capacity left [t]": "{:.1f}", "Reserve [%]": "{:.1f}"}),
                     hide_index=True)
    for n in b.get("notes", []):
        st.caption("• " + str(n))


def _environment_section(raw: dict) -> None:
    e = raw["environment"]
    st.subheader("B · Environment and operating limits (from the case file)")
    lim = e.get("operating_limits") or {}
    if lim:
        cols = st.columns(len(lim))
        for c, (k, v) in zip(cols, lim.items()):
            c.metric(k, str(v))
    if e.get("monthly"):
        st.dataframe(pd.DataFrame(e["monthly"]), hide_index=True)
    if e.get("tide"):
        st.dataframe(pd.DataFrame([e["tide"]]), hide_index=True)
    rows = e.get("monthly") or []
    if rows and all(k in rows[0] for k in ("Hs P10 [m]", "Wind P1 [m/s]")):
        st.markdown("**Operability against the limits** (share of time below, from the percentile table)")
        c1, c2 = st.columns(2)
        hs = c1.number_input("Hs limit [m]", 0.1, 10.0, 1.0, 0.1, key="bu_hs_lim")
        wk = c2.number_input("Wind limit, 1-min mean [kn]", 1.0, 100.0, 20.0, 1.0, key="bu_wind_lim")
        ops = [mo.operability(r, hs, wk) for r in rows]
        df = pd.DataFrame(ops).rename(columns={"month": "Month", "wind_limit_10min_ms": "Limit as 10-min wind [m/s]",
                                               "below_hs_pct": "Hs below limit [%]", "below_wind_pct": "Wind below limit [%]",
                                               "below_both_pct_low": "Both, independent [%]",
                                               "below_both_pct_high": "Both, fully correlated [%]"})
        st.dataframe(df[["Month", "Limit as 10-min wind [m/s]", "Hs below limit [%]", "Wind below limit [%]",
                         "Both, independent [%]", "Both, fully correlated [%]"]].style.format(
            {c: "{:.1f}" if "%" in c else "{:.2f}" for c in df.columns if c != "Month" and c in
             ["Limit as 10-min wind [m/s]", "Hs below limit [%]", "Wind below limit [%]", "Both, independent [%]",
              "Both, fully correlated [%]"]}), hide_index=True)
        st.caption("Interpolated between mean, P10, P1 and maximum (log in probability), so a screen only. The 1-min limit is "
                   "converted to a 10-min mean with the ISO 19901-1 gust model. The table has no persistence: a tow window "
                   "of several days needs a hindcast time series from Fugro.")
    for n in e.get("notes", []):
        st.caption("• " + str(n))


def _modules_section(case: dict, key: str) -> None:
    s, mods, params, crit = case["structure"], case["modules"], case["params"], case["criteria"]
    st.subheader("C · Buoyancy modules on the member geometry")
    if not s.elements or not mods:
        st.info("No member geometry or modules in this case file yet, so the waterline, GM and clearance cannot be "
                "calculated. Add the members (from the drawings) to the file to enable this section.")
        return
    states = {}
    cols = st.columns(min(4, len(mods)))
    for i, m in enumerate(mods):
        dflt = case["states"].get(m.name, "sealed")
        states[m.name] = cols[i % len(cols)].selectbox(m.name, bu.STATES, index=bu.STATES.index(dflt), key=f"bu_st_{key}_{m.name}")
    rm = float(case.get("reserve_min_pct", 0.0))
    r = bu.evaluate(s, mods, states, params, crit, reserve_min_pct=rm)
    if not r["floats"]:
        st.error(f"Does not float: {r['error']}")
    else:
        k = st.columns(5)
        k[0].metric("Weight", f"{r['weight_t']:.1f} t")
        k[1].metric("Draft", f"{r['draft_m']:.2f} m")
        k[2].metric("Clearance", "-" if r["clearance_m"] is None else f"{r['clearance_m']:.2f} m")
        k[3].metric("GM", f"{r['gm_min_m']:.2f} m")
        k[4].metric("Reserve", f"{r['reserve_pct']:.1f} %")
        (st.success if r["passed"] else st.error)("PASS against the case-file criteria" if r["passed"] else
                                                  "FAIL against the case-file criteria")
    t1, t2 = st.tabs(["Sweep all combinations", "Flood one module at a time"])
    cols = ["case", "weight_t", "capacity_t", "reserve_pct", "draft_m", "clearance_m", "gm_min_m", "free_tilt_deg", "passed", "error"]
    with t1:
        if 2 ** len(mods) > 64:
            st.info("More than 6 modules: fix some in the file before sweeping.")
        elif st.button("Run sweep", key=f"bu_sw_{key}"):
            rows = bu.sweep(s, mods, params, crit, reserve_min_pct=rm, max_cases=64)
            st.dataframe(pd.DataFrame(rows).reindex(columns=cols), hide_index=True)
    with t2:
        if st.button("Run damage cases", key=f"bu_dm_{key}"):
            rows = bu.damage_cases(s, mods, states, params, crit, reserve_min_pct=rm)
            st.dataframe(pd.DataFrame(rows).reindex(columns=["damaged_module"] + cols), hide_index=True)


def render() -> None:
    st.markdown("**Buoyancy: budget, environment and modules.** Project data comes from a case file kept outside "
                "the repository. Screening calculation, not independently checked.")
    st.warning("Case files can hold client-confidential data. An uploaded file is processed on the machine running "
               "this app: use a local run (or a private deployment) for project data.", icon="🔒")
    raw, fname = _read_raw()
    if raw is None:
        st.info("Load a case file to begin. `docs/case_template.json` in the repository shows the layout (dummy numbers).")
        return
    key = f"{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10**8}"
    st.markdown(f"Case: **{raw.get('name', fname)}**")
    if raw.get("budget"):
        _budget_section(raw, key)
    if raw.get("environment"):
        _environment_section(raw)
    try:
        case = case_io.load_case(raw)
    except case_io.CaseFileError as exc:
        st.error("Case file problems:\n\n" + "\n".join(f"- {p}" for p in exc.problems))
        return
    _modules_section(case, key)
