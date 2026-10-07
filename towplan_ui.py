"""Streamlit tab: final-approach tow plan, tow vessels only (screening)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import budget as bd
import towplan as tp

DEFAULT_LEGS = [dict(from_m=1000.0, to_m=500.0, speed_kn=1.5, note="Open approach"),
                dict(from_m=500.0, to_m=250.0, speed_kn=1.0, note="Slow down"),
                dict(from_m=250.0, to_m=100.0, speed_kn=0.5, note="Final approach, hold point at 100 m")]


def render() -> None:
    st.markdown("**Final approach, tow vessels only.** Pull needed, time and a hold-position check from the "
                "drag on the tubes. Screening level: the drag area comes from tube sizes, not from the member geometry.")
    raw = st.session_state.get("case_raw") or {}
    srcs = (raw.get("budget") or {}).get("sources") or []
    area_case = tp.drag_area_m2(srcs, 1.0) if srcs else 0.0
    env = (raw.get("environment") or {})

    c1, c2, c3 = st.columns(3)
    sub = c1.slider("Submerged fraction of the tubes", 0.1, 1.0, 1.0, 0.05, key="tp_sub",
                    help="1.0 = whole structure under water (conservative). Lower it when the draft is known.")
    if area_case > 0:
        c2.metric("Tube drag area from the case file", f"{area_case * sub:,.0f} m²",
                  help="Sum of OD x length of the leg and brace sources, times the submerged fraction. Tanks are not included.")
        area = area_case * sub
    else:
        area = c2.number_input("Tube drag area [m²]", 0.0, 1e6, 3000.0, 50.0, key="tp_area",
                               help="No case file loaded. Enter OD x length summed over the legs and braces.")
    tank_a = c3.number_input("Extra area for tanks and appurtenances [m²]", 0.0, 1e5, 0.0, 10.0, key="tp_tank",
                             help="Tank diameter x length, risers, J-tubes. The case file has no tank sizes yet.")
    ca = area + tank_a

    d1, d2, d3, d4 = st.columns(4)
    cd = d1.number_input("Drag coefficient Cd", 0.3, 3.0, float(st.session_state.get("p_cd_water", 1.0)), 0.05, key="tp_cd")
    marg = d2.number_input("Margin on drag", 0.5, 3.0, float(st.session_state.get("p_tow_factor", 1.0)), 0.1, key="tp_margin",
                           help="Applied to the drag; covers waves, wind and interference.")
    cur = d3.number_input("Current [kn]", 0.0, 5.0, float(st.session_state.get("p_current_speed_kn", 0.0)), 0.1, key="tp_cur",
                          help="Starts from the tab 2 value; the presets in tab 7 fill tab 2.")
    cdir = d4.number_input("Current flows toward [° from the tow direction]", 0.0, 360.0, 180.0, 15.0, key="tp_cdir",
                           help="0 = with the tow, 180 = head current, 90 or 270 = across.")
    t1, t2, t3 = st.columns(3)
    n_t = int(t1.number_input("Tow vessels", 0, 8, 2, 1, key="tp_ntug"))
    bp = t2.number_input("Static bollard pull each [t]", 0.0, 500.0, 0.0, 5.0, key="tp_bp",
                         help="Leave at 0 until ICON or the tow contractor gives the vessels.")
    eff = t3.number_input("Effective share of bollard pull", 0.1, 1.0, 0.8, 0.05, key="tp_eff",
                          help="ASSUMED. Pull available while towing and station-keeping (weather, thrust, wire angle).")

    st.markdown("**Legs of the approach** (distance to the platform, speed over the ground)")
    legs = st.data_editor(pd.DataFrame(DEFAULT_LEGS), num_rows="dynamic", hide_index=True, key="tp_legs",
                          column_config={"from_m": st.column_config.NumberColumn("From [m]", min_value=0.0, format="%.0f"),
                                         "to_m": st.column_config.NumberColumn("To [m]", min_value=0.0, format="%.0f"),
                                         "speed_kn": st.column_config.NumberColumn("Speed [kn]", min_value=0.05, format="%.2f"),
                                         "note": st.column_config.TextColumn("Note")})
    rows = [r for r in legs.dropna(subset=["from_m", "to_m", "speed_kn"]).to_dict("records")]
    if not rows or ca <= 0:
        st.info("Enter at least one leg and a drag area above zero.")
        return
    try:
        res = tp.plan(rows, ca, cur, cdir, cd=cd, margin=marg, tug_count=n_t, tug_bollard_t=bp, tug_eff=eff)
    except ValueError as exc:
        st.error(str(exc))
        return

    m = st.columns(4)
    m[0].metric("Time for the approach", f"{res['total_hours']:.1f} h")
    m[1].metric("Highest pull needed", f"{res['governing_need_kn'] / tp.G:.0f} t",
                help="Resultant of the along-track and side drag, with the margin.")
    m[2].metric("Pull to hold position", f"{res['hold_t']:.0f} t",
                help="Stopped over the ground in the current: the abort and hold-point case.")
    m[3].metric("Tug pull available", "-" if res["available_t"] <= 0 else f"{res['available_t']:.0f} t",
                help="Vessels x static bollard pull x effective share.")
    tbl = pd.DataFrame(res["rows"])
    tbl["util_pct"] = tbl["util_pct"].astype(float) if res["available_kn"] > 0 else float("nan")
    tbl.insert(0, "Leg", [f"{int(r['from_m'])} to {int(r['to_m'])} m" for r in res["rows"]])
    show = tbl[["Leg", "speed_kn", "hours", "v_rel_ms", "along_kn", "side_kn", "need_t", "util_pct"]].rename(columns={
        "speed_kn": "Speed [kn]", "hours": "Time [h]", "v_rel_ms": "Water speed rel. [m/s]", "along_kn": "Along-track [kN]",
        "side_kn": "Side [kN]", "need_t": "Pull needed [t]", "util_pct": "Of tug pull [%]"})
    st.dataframe(show.style.format({"Speed [kn]": "{:.2f}", "Time [h]": "{:.1f}", "Water speed rel. [m/s]": "{:.2f}",
                                    "Along-track [kN]": "{:.0f}", "Side [kN]": "{:.0f}", "Pull needed [t]": "{:.0f}",
                                    "Of tug pull [%]": "{:.0f}"}, na_rep="-"), hide_index=True)
    if res["available_kn"] > 0:
        worst = max(r["util_pct"] for r in res["rows"])
        ok = worst <= 100.0 and (res["hold_util_pct"] or 0) <= 100.0
        (st.success if ok else st.error)(
            f"Tug pull is {'enough' if ok else 'NOT enough'}: highest leg {worst:.0f} % of the pull available, "
            f"hold position {res['hold_util_pct']:.0f} %.")
    else:
        st.caption("Enter the tug pull to get the utilisation. The pull needed is shown regardless.")

    lim = (env.get("operating_limits") or {})
    if lim or env.get("monthly"):
        st.markdown("**Weather for this approach**")
        st.caption(f"The approach takes about {res['total_hours']:.0f} h plus hook-up and set-down; the window needs a calm spell "
                   f"at least that long inside the limits ({', '.join(f'{k}: {v}' for k, v in lim.items() if 'limit' in k.lower())}). "
                   "The Fugro tables give how often the limits are exceeded (tab 7), not how long calm spells last, so ask for the "
                   "hindcast time series to size the window.")
    st.caption("Not included: wind on the structure above water, wave drift, tug-to-jacket geometry and wire angles, "
               "tide-dependent draft, and the set-down manoeuvre. Current is assumed uniform with depth (conservative).")
