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
    cur = e.get("current") or {}
    if cur.get("presets"):
        st.markdown("**Current for the tow-load analysis**")
        pre = cur["presets"]
        names = [p["name"] for p in pre]
        dflt = names.index(cur["default"]) if cur.get("default") in names else 0
        sel = st.selectbox("Current case", names, index=dflt, key="bu_cur_sel")
        p = pre[names.index(sel)]
        kn = float(p["speed_ms"]) / mo.KN
        cc1, cc2 = st.columns([2, 1])
        cc1.metric("Surface current", f"{p['speed_ms']:.2f} m/s = {kn:.2f} kn", help=p.get("source", ""))

        def _use(kn=kn):
            st.session_state["p_current_speed_kn"] = round(kn, 3)

        cc2.button("Use in tow-load analysis (tab 2)", on_click=_use, key="bu_cur_use",
                   help="Sets the current speed in tab 2. Direction relative to the tow is set there (180° = head current).")
        st.caption(cur.get("basis", "") + " Drag goes with the square of the speed through the water, so a head current "
                   "of 0.55 m/s on a 2.5 kn tow roughly doubles the line pull; a following current of the same size cuts it to a third.")
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


def _structure_section(case: dict) -> None:
    """Platform (steel) members: counts, lengths, tubular steel mass, drag area and a 3D view."""
    els = case["structure"].elements
    if not els:
        return
    st.subheader("Platform structure (members only, no buoyancy items)")
    import math
    import numpy as np
    rows = []
    for e in els:
        L = math.dist(e.p1, e.p2)
        a = math.pi / 4 * (e.d_out ** 2 - e.d_in ** 2)
        rows.append(dict(name=e.name, group=("Legs" if e.name.startswith("Leg") else "Braces" if e.name.startswith("Brace")
                                              else "Plan frames" if e.name.startswith("Plan") else "Other"),
                         od_mm=round(e.d_out * 1000, 1), length_m=L, steel_t=a * L * 7.85, area_m2=e.d_out * L,
                         x1=e.p1[0], y1=e.p1[1], z1=e.p1[2], x2=e.p2[0], y2=e.p2[1], z2=e.p2[2]))
    df = pd.DataFrame(rows)
    pts = np.array([[r["x1"], r["y1"], r["z1"]] for r in rows] + [[r["x2"], r["y2"], r["z2"]] for r in rows])
    k = st.columns(5)
    k[0].metric("Members", f"{len(df)}")
    k[1].metric("Total length", f"{df.length_m.sum():.0f} m")
    k[2].metric("Tubular steel", f"{df.steel_t.sum():.1f} t")
    k[3].metric("Drag area (OD x L)", f"{df.area_m2.sum():.0f} m2")
    k[4].metric("Height", f"{np.ptp(pts[:, 2]):.1f} m")
    st.caption("Tubular steel at 7.85 t/m3 from the member sizes; the bill of materials also includes plate, mudmat and "
               "other items, so compare with the weight control report as a screen only.")
    g = df.groupby(["group", "od_mm"], as_index=False).agg(members=("name", "count"), length_m=("length_m", "sum"),
                                                          steel_t=("steel_t", "sum")).round(1)
    st.dataframe(g, hide_index=True)
    colour = {"Legs": "#1f4e79", "Braces": "#c0504d", "Plan frames": "#7f7f7f", "Other": "#000000"}
    fig = go.Figure()
    for grp, d in df.groupby("group"):
        xs, ys, zs = [], [], []
        for r in d.itertuples():
            xs += [r.x1, r.x2, None]; ys += [r.y1, r.y2, None]; zs += [r.z1, r.z2, None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=grp, line=dict(color=colour[grp], width=4)))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"))
    _weights_cog(case)
    legs = _leg_table(df)
    if legs is not None:
        for r in legs.itertuples():
            fig.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.top_z + 2.0], mode="text", text=[r.leg], textfont=dict(size=14),
                                       showlegend=False, hoverinfo="skip"))
    _show(fig)
    if legs is not None:
        st.markdown("**Legs by grid line** (column 1 and 2 = x, rows A'' / A / B = y, as on the drawings)")
        st.dataframe(legs.rename(columns=dict(x="x [m]", y="y [m]", top_z="top z [m]", bottom_z="bottom z [m]",
                                              length_m="length [m]", pieces="segments", steel_t="steel [t]")).round(2),
                     hide_index=True)


def _weights_cog(case: dict) -> None:
    """Weight items of the structure with the combined centre of gravity (body z and EL on the project datum)."""
    wts = list(case["structure"].weights)
    if not wts:
        return
    dz = float(case.get("datum_offset_m", 0.0))
    tot = sum(w.mass_t for w in wts)
    g = [sum(w.mass_t * getattr(w, a) for w in wts) / tot for a in ("x", "y", "z")]
    st.markdown("**Structure weights and centre of gravity**")
    k = st.columns(4)
    k[0].metric("Structure weight", f"{tot:,.2f} t")
    k[1].metric("CoG x", f"{g[0]:.3f} m")
    k[2].metric("CoG y", f"{g[1]:.3f} m")
    k[3].metric("CoG z (EL)", f"{g[2] - dz:.2f} m", help=f"{g[2]:.2f} m above the base; EL = z - {dz:.1f} m")
    st.dataframe(pd.DataFrame([dict(item=w.name, mass_t=w.mass_t, x=w.x, y=w.y, z_above_base=w.z, EL=w.z - dz)
                               for w in wts]).round(3), hide_index=True)
    st.caption("Tanks and other buoyancy modules carry their own weights, shown in section C with the floating results.")


def _leg_table(df: pd.DataFrame):
    """One row per leg from member names of the form 'Leg x<col> <A|B|C> ...' (C = row A'').  None if no such names."""
    import re
    rows = {}
    for r in df.itertuples():
        m = re.match(r"Leg x(\d+(?:\.\d+)?) ([ABC])\b", r.name)
        if not m:
            continue
        x = float(m.group(1))
        row = {"A": "A", "B": "B", "C": "A''"}[m.group(2)]
        col = 1 if x < 3.5 else 2
        d = rows.setdefault(f"{row}{col}", dict(leg=f"{row}{col}", x=r.x1, y=r.y1, top_z=-1e9, bottom_z=1e9, length_m=0.0,
                                               pieces=0, steel_t=0.0))
        d["top_z"] = max(d["top_z"], r.z1, r.z2)
        d["bottom_z"] = min(d["bottom_z"], r.z1, r.z2)
        d["length_m"] += r.length_m
        d["pieces"] += 1
        d["steel_t"] += r.steel_t
    if not rows:
        return None
    return pd.DataFrame(sorted(rows.values(), key=lambda d: (d["leg"][:-1] != "A", d["leg"])))


def legs_from_elements(elements) -> "pd.DataFrame | None":
    """Leg table (grid names) for a list of js.Element, or None when the members carry no leg names."""
    import math
    rows = []
    for e in elements:
        L = math.dist(e.p1, e.p2)
        rows.append(dict(name=e.name, length_m=L, steel_t=math.pi / 4 * (e.d_out ** 2 - e.d_in ** 2) * L * 7.85,
                         x1=e.p1[0], y1=e.p1[1], z1=e.p1[2], x2=e.p2[0], y2=e.p2[1], z2=e.p2[2]))
    return _leg_table(pd.DataFrame(rows)) if rows else None


def _plan_figure(elements, legs) -> go.Figure:
    """Plan view (x across, y up) of all members in light grey with the legs labelled by grid name."""
    fig = go.Figure()
    xs, ys = [], []
    for e in elements:
        xs += [e.p1[0], e.p2[0], None]
        ys += [e.p1[1], e.p2[1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#9aa3ad", width=1), hoverinfo="skip", showlegend=False))
    if legs is not None:
        fig.add_trace(go.Scatter(x=legs.x, y=legs.y, mode="markers+text", text=[f"<b>{n}</b>" for n in legs.leg],
                                 textposition="top right", textfont=dict(size=16, color="#c00000"),
                                 marker=dict(size=11, color="#1f4e79"), showlegend=False,
                                 hovertext=[f"{r.leg}: x {r.x:.1f}, y {r.y:.1f} m" for r in legs.itertuples()], hoverinfo="text"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), title=dict(text="Plan view (x across, y up)", font=dict(size=14)),
                      xaxis=dict(title="x [m]", scaleanchor="y", constrain="domain"), yaxis=dict(title="y [m]"))
    return fig


def render_case_platform(raw: dict) -> None:
    """Platform drawn from the loaded case file with grid leg labels (used by the Geometry check tab)."""
    import viz
    try:
        case = case_io.load_case(raw)
    except case_io.CaseFileError as exc:
        st.error("Case file problems:\n\n" + "\n".join(f"- {p}" for p in exc.problems))
        return
    s = case["structure"]
    if not s.elements:
        st.info("The loaded case file has no members yet.")
        return
    dz = float(case.get("datum_offset_m", 0.0))
    legs = legs_from_elements(s.elements)
    zs = [z for e in s.elements for z in (e.p1[2], e.p2[2])]
    zlo, zhi = min(zs), max(zs)
    el_lo, el_hi = zlo - dz, zhi - dz
    band = st.slider("Zoom to elevation band (EL on the MSL datum, m)", float(round(el_lo)), float(round(el_hi) + 1),
                     (float(round(el_lo)), float(round(el_hi) + 1)), step=1.0, key="geo_band")
    b0, b1 = band[0] + dz, band[1] + dz
    full = (b0 <= zlo + 0.5) and (b1 >= zhi - 0.5)
    left, right = st.columns([3, 2])
    with left:
        fig = viz.jacket_figure(s.elements, s.weights, s.openings, case["params"], show_triad=False,
                                title="Platform from the case file, as built",
                                color_by="diameter" if st.session_state.get("g_color") == "Diameter" else "type")
        if legs is not None:
            for r in legs.itertuples():
                zl = min(r.top_z, b1) - (0.04 * (b1 - b0) if not full else -4.0 - (6.0 if r.leg.startswith("B") else 0.0))
                if not full and not (r.bottom_z <= b1 and r.top_z >= b0):
                    continue
                fig.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[zl], mode="text", text=[f"<b>{r.leg}</b>"],
                                           textfont=dict(size=16, color="#c00000"), showlegend=False, hoverinfo="skip"))
        scene = dict(aspectmode="data", zaxis=dict(range=[b0 - 1.0, b1 + (12.0 if full else 1.0)]),
                     camera=dict(eye=dict(x=3.0, y=-3.3, z=1.1) if full else dict(x=1.25, y=-1.4, z=0.8)))
        fig.update_layout(height=760, scene=scene, updatemenus=[], title_text="", margin=dict(l=0, r=0, t=0, b=0))
        _show(fig)
    with right:
        _show(_plan_figure(s.elements, legs))
    if legs is not None:
        st.markdown("**Legs by grid line** (columns 1 and 2 = x, rows A'' / A / B = y). z is from the base; add 92.4 m for EL.")
        st.dataframe(legs.rename(columns=dict(x="x [m]", y="y [m]", top_z="top z [m]", bottom_z="bottom z [m]",
                                              length_m="length [m]", pieces="segments", steel_t="steel [t]")).round(2),
                     hide_index=True)


def _queue_send(payload: dict) -> None:
    ss = st.session_state
    ph = {k: payload["widgets"][k] for k in ("p_tow_x", "p_tow_y", "p_tow_z")}
    ph.update({k: ss[k] for k in ("p_tow_heading_deg", "p_wind_speed_kn") if k in ss})
    payload = dict(payload, placeholders=ph)          # values still to be set from the design basis
    ss["send_case_pending"] = payload


def _send_section(case: dict, key: str) -> None:
    """Button that hands the assembled platform (structure + modules in their current states) to tabs 1-6."""
    st.subheader("Use this platform in tabs 1-6")
    states = {m.name: st.session_state.get(f"bu_st_{key}_{m.name}", case["states"].get(m.name, "sealed"))
              for m in case["modules"]}
    try:
        els, wts, ops, damaged = bu.assemble(case["structure"], case["modules"], states)
    except ValueError as exc:
        st.info(f"Cannot hand over this combination: {exc}")
        return
    p, c = case["params"], case["criteria"]
    xs = [v for e in els for v in (e.p1[0], e.p2[0])]
    ys = [v for e in els for v in (e.p1[1], e.p2[1])]
    zs = [v for e in els for v in (e.p1[2], e.p2[2])]
    payload = dict(
        name=case["name"],
        elements=[dict(name=e.name, x1=e.p1[0], y1=e.p1[1], z1=e.p1[2], x2=e.p2[0], y2=e.p2[1], z2=e.p2[2],
                       d_out=e.d_out, d_in=e.d_in, buoyant=e.buoyant, exposed=e.exposed, flooded=e.flooded, tank=e.tank) for e in els],
        weights=[dict(item=w.name, mass_t=w.mass_t, x=w.x, y=w.y, z=w.z) for w in wts],
        openings=[dict(name=o.name, x=o.x, y=o.y, z=o.z) for o in ops],
        widgets={"p_depth_m": p.water_depth_m, "p_tide_m": p.tide_m, "c_gm_min": c.gm_min, "c_clear_min_m": c.clear_min_m,
                 "c_emerged_min_m": c.emerged_min_m, "p_tow_x": min(xs), "p_tow_y": 0.5 * (min(ys) + max(ys)),
                 "p_tow_z": min(zs) + 0.8 * (max(zs) - min(zs))},
        damaged=damaged,
        flags=[f"SYNTHETIC placeholder: {m.name}" for m in case["modules"]
               if "SYNTHETIC" in str(m.note).upper() and states.get(m.name, "off") != "off"])
    st.caption(f"Sends {len(els)} members, {len(wts)} weight items ({sum(w.mass_t for w in wts):,.0f} t) and the "
               "module states chosen above. The tow-line attachment is set to a PLACEHOLDER at the front centre of the "
               "platform, near the waterline: replace it in tab 2 with the real tow point.")
    st.button("Use in tabs 1-6", key=f"bu_send_{key}", on_click=_queue_send, args=(payload,))
    if st.session_state.get("send_case_done") == case["name"]:
        st.success("Loaded into tabs 1-6. Check tab 1 and tab 2, then run the analysis from the sidebar.")


def _window_figure(df, rm, rd, crit):
    """Reserve, GM and clearance against tank diameter, with the passing range shaded."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    f = go.Figure(make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Reserve buoyancy [%]", "GM min [m]", "Deck clearance [m]")))
    ok = df[df.floats]
    x = ok.diameter_m
    for r, (col, lim) in enumerate((("reserve_pct", rm), ("gm_min_m", crit.gm_min), ("clearance_m", crit.clear_min_m)), 1):
        f.add_scatter(x=x, y=ok[col], mode="lines+markers", row=r, col=1, showlegend=False, line=dict(color="#1f77b4"))
        f.add_hline(y=lim, line=dict(color="red", dash="dash"), row=r, col=1)
    for col, colour, name in (("intact_passed", "rgba(255,193,7,0.25)", "intact passes"),
                              ("all_passed", "rgba(46,160,67,0.30)", "intact + all damage pass")):
        d = df[df[col]].diameter_m
        if len(d):
            f.add_vrect(x0=d.min(), x1=d.max(), fillcolor=colour, line_width=0, annotation_text=name,
                        annotation_position="top left" if col == "intact_passed" else "bottom right")
    f.update_xaxes(title_text="Tank diameter [m]", row=3, col=1)
    f.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10))
    return f


def _sizing_tab(s, mods, states, params, crit, rm, rm_dmg, key) -> None:
    """Scale the diameter of chosen tank modules (length and position fixed) and see which sizes pass."""
    tank_names = [m.name for m in mods if m.elements]
    if not tank_names:
        st.info("No module with its own tank cylinders in this case, so there is nothing to size.")
        return
    st.caption("Scales the diameter of the chosen tank modules together (length and position unchanged). Tank steel "
               "weight is scaled with volume. Everything else, including the criteria in the case file, stays as is.")
    chosen = st.multiselect("Tank modules to scale", tank_names, default=tank_names, key=f"bu_sz_n_{key}")
    c = st.columns(4)
    f_min = c[0].number_input("Smallest factor", 0.1, 5.0, 0.6, 0.1, key=f"bu_sz_a_{key}")
    f_max = c[1].number_input("Largest factor", 0.2, 6.0, 2.0, 0.1, key=f"bu_sz_b_{key}")
    steps = int(c[2].number_input("Steps", 5, 61, 29, 1, key=f"bu_sz_s_{key}"))
    rd = c[3].number_input("Damaged reserve min [%]", 0.0, 100.0, rm_dmg, 1.0, key=f"bu_sz_d_{key}")
    if not chosen or f_max <= f_min:
        st.info("Choose at least one module and make the largest factor bigger than the smallest.")
        return
    if st.button("Run sizing", key=f"bu_sz_go_{key}"):
        out = bu.size_tanks(s, mods, states, params, crit, chosen, reserve_min_pct=rm, damaged_reserve_min_pct=rd,
                            f_min=f_min, f_max=f_max, steps=steps)
        df = pd.DataFrame(out["rows"])
        if out["f_intact"] is None:
            st.error("No size in this range passes the intact case. Widen the range or look at the limiting check below.")
        else:
            r1 = df[df.factor == out["f_intact"]].iloc[0]
            st.success(f"Smallest size that passes intact: factor {out['f_intact']:.2f} "
                       f"(diameter {r1.diameter_m:.2f} m, tank volume {r1.tank_volume_m3:.0f} m3).")
        if out["f_all"] is None:
            st.warning("No size in this range passes the intact case and every one-module damage case.")
        else:
            r2 = df[df.factor == out["f_all"]].iloc[0]
            st.success(f"Smallest size that passes intact and all damage cases: factor {out['f_all']:.2f} "
                       f"(diameter {r2.diameter_m:.2f} m, tank volume {r2.tank_volume_m3:.0f} m3).")
        show = df.rename(columns=dict(factor="factor", diameter_m="diameter [m]", tank_volume_m3="tank volume [m3]",
                                      weight_t="weight [t]", reserve_pct="reserve [%]", clearance_m="clearance [m]",
                                      gm_min_m="GM [m]", intact_passed="intact pass", damage_passed="damage passed",
                                      damage_total="damage cases", all_passed="all pass"))
        st.plotly_chart(_window_figure(df, rm, rd, crit), use_container_width=True)
        st.dataframe(show.drop(columns=["floats"]).round(2), hide_index=True)
        st.caption("Bigger is not always better: very large tanks can raise the reserve but lose GM or clearance. "
                   "The table shows each size so you can see the window.")


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
    t1, t2, t3 = st.tabs(["Sweep all combinations", "Flood one module at a time", "Size the tanks"])
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
    with t3:
        _sizing_tab(s, mods, states, params, crit, rm, float(case.get("reserve_damaged_min_pct", 0.0)), key)


def render() -> None:
    st.markdown("**Buoyancy: budget, environment and modules.** Project data comes from a case file kept outside "
                "the repository. Screening calculation, not independently checked.")
    st.warning("Case files can hold client-confidential data. An uploaded file is processed on the machine running "
               "this app: use a local run (or a private deployment) for project data.", icon="🔒")
    raw, fname = _read_raw()
    if raw is None:
        st.session_state.pop("case_raw", None)
        st.info("Load a case file to begin. `docs/case_template.json` in the repository shows the layout (dummy numbers).")
        return
    st.session_state["case_raw"] = raw                      # shared with the tow-plan tab
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
    _structure_section(case)
    _modules_section(case, key)
    _send_section(case, key)
