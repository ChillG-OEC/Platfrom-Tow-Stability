"""Tab: lowering onto the seabed (clearance 10 m -> 5 m -> set-down) across the tidal range."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import setdown as sd

STAGE_NAMES = ("After upending", "Near the target", "Set-down")


def _show(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                                   # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def _figure(curves, rows, clearances, gm_min):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                      subplot_titles=("Ballast needed to hold the waterline [t]", "GM min [m]"))
    df = pd.DataFrame(curves)
    for name, g in df.groupby("tide", sort=False):
        f.add_scatter(x=g.clearance_m, y=g.ballast_t, mode="lines", name=name, row=1, col=1)
    r = pd.DataFrame(rows)
    for name, g in r.groupby("tide", sort=False):
        ok = g[g.gm_min_m.notna()]
        f.add_scatter(x=ok.clearance_m, y=ok.gm_min_m, mode="markers+lines", name=name, showlegend=False, row=2, col=1)
    if gm_min > 0:
        f.add_hline(y=gm_min, line=dict(color="red", dash="dash"), row=2, col=1)
    for c in clearances:
        f.add_vline(x=c, line=dict(color="grey", dash="dot"))
    f.update_xaxes(title_text="Seabed clearance [m]", autorange="reversed", row=2, col=1)
    f.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
    return f


def render(parts, depth0: float, tide0: float, gm_min: float, project: str, is_demo: bool = False) -> None:
    if is_demo:
        st.info("Tabs 1-6 still hold the built-in demo jacket, which is a horizontal tow example, not an upright float. "
                "Load a case file in tab 7 and press 'Use in tabs 1-6' first.")
        return
    els, wts, ops, params, _crit = parts
    st.markdown("**Lowering onto the seabed.** After upending the jacket hangs at about 10 m seabed clearance, is lowered "
                "to about 5 m near the target, then set down. At each stage the waterline is fixed by the water depth, "
                "the tide and the clearance, so the ballast needed follows from the buoyancy of the sealed members.")
    st.caption(f"Data: {project}. Level waterline, ballast water treated as a point mass inside the tanks (no free "
               "surface effect), no current or wave motion. Screening only: it does not check the installation "
               "equipment, lowering loads or on-bottom stability.")
    c = st.columns(4)
    depth = c[0].number_input("Water depth at LAT [m]", 0.0, 3000.0, float(depth0), 0.5, key=f"sd_depth_{depth0:g}")
    rng = c[1].number_input("Tidal range LAT to HAT [m]", 0.0, 20.0, float(tide0), 0.1, key=f"sd_range_{tide0:g}")
    base_h = sd.default_ballast_z(els) - min(min(e.p1[2], e.p2[2]) - e.d_out / 2 for e in els)
    bh = c[2].number_input("Ballast height above base [m]", 0.0, 500.0, float(round(base_h, 1)), 0.5, key="sd_bz",
                           help="Where the ballast water sits. Default: the middle of the tank members.")
    tides = {"LAT": 0.0, "MSL": rng / 2.0, "HAT": rng} if rng > 0 else {"LAT": 0.0}
    dt = c[3].selectbox("Tide the ballast is set at", list(tides), index=min(1, len(tides) - 1), key=f"sd_dt_{len(tides)}")
    cc = st.columns(3)
    clr = [cc[i].number_input(f"Clearance: {STAGE_NAMES[i].lower()} [m]", 0.0, 100.0, v, 0.5, key=f"sd_c{i}")
           for i, v in enumerate((10.0, 5.0, 0.0))]
    if depth <= 0:
        st.info("Enter the water depth (here or in tab 2) to run the sequence.")
        return
    zb = min(min(e.p1[2], e.p2[2]) - e.d_out / 2 for e in els)
    bz = zb + bh
    try:
        res = sd.sequence(els, wts, ops, params, depth, tides, clr, dt, bz, gm_min)
        curves = sd.curve(els, wts, ops, params, depth, tides, max(clr) + 3.0, 0.0)
    except ValueError as exc:
        st.error(f"Cannot run the sequence: {exc}")
        return
    names = dict(zip(clr, STAGE_NAMES))
    rows = res["rows"]
    k = st.columns(3)
    base = {r["stage"]: r for r in rows if r["tide"] == dt}
    for i, col in enumerate(k):
        r = base[clr[i]]
        txt = "sinks (all buoyancy used)" if r["saturated"] else f"{r['ballast_t']:.0f} t ballast"
        col.metric(f"{STAGE_NAMES[i]} ({clr[i]:g} m)", txt,
                   help="Ballast on board at this clearance, at the tide chosen above.")
    st.subheader("Go / no-go by stage")
    for _c, _ok, _txt in sd.stage_advice(res, names, dt, gm_min):
        (st.success if _ok else st.error)(_txt)
    st.subheader("Ballast and stability by stage and tide")
    tbl = pd.DataFrame([dict(stage=names[r["stage"]], clearance_m=r["stage"], tide=r["tide"],
                             water_over_base_m=r["depth_total_m"], waterline_draft_m=r["draft_m"],
                             ballast_t=None if r["saturated"] else r["ballast_t"],
                             gm_min_m=r["gm_min_m"], trim_deg=r["trim_deg"],
                             gm_ok=None if r["gm_min_m"] is None or gm_min <= 0 else bool(r["gm_min_m"] >= gm_min),
                             note="all sealed members submerged: sinks to the seabed" if r["saturated"] else
                             ("weight would have to be shed" if r["ballast_t"] < 0 else ""))
                        for r in rows])
    st.dataframe(tbl.round(2), hide_index=True)
    st.subheader(f"Ballast to add between stages (set at {dt})")
    st.dataframe(pd.DataFrame([dict(stage=names[s["stage"]], clearance_m=s["stage"],
                                    ballast_on_board_t=s["ballast_t"], added_from_previous_t=s["added_from_previous_t"])
                               for s in res["steps"]]).round(1), hide_index=True)
    st.subheader(f"Tidal variation with the ballast fixed at {dt}")
    st.caption("Clearance changes one for one with the tide. The last column is the ballast to add (+) or remove (-) "
               "to keep the stage clearance at that tide.")
    st.dataframe(pd.DataFrame([dict(stage=names[a["stage"]], tide=a["tide"], clearance_m=a["clearance_m"],
                                    ballast_change_to_hold_t=a["extra_ballast_to_hold_t"]) for a in res["across"]]).round(2),
                 hide_index=True)
    low = [a for a in res["across"] if a["clearance_m"] < 0 and a["stage"] > 0]
    if low:
        st.warning("With the ballast set at "
                   f"{dt}, the structure would be on the seabed at: " + ", ".join(f"{names[a['stage']]} at {a['tide']}" for a in low))
    _show(_figure(curves, rows, clr, gm_min))
