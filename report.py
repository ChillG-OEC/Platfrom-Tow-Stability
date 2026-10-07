"""
report.py - draft A4 PDF report for the jacket wet-tow screening tool (no Streamlit dependency).
"""
from __future__ import annotations

import datetime
import io
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import jacket_stability as js
import report_figs as rf

APP_VERSION = "1.0"


def fmt(x, nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}{suffix}"


ASSUMPTIONS = [
    "Rigid-body statics. Heave and trim are solved at each imposed heel angle (free to trim); yaw, surge and sway are not modelled.",
    "Displacement comes only from elements flagged buoyant (watertight straight cylinders / annuli). Steel volume of flooded members is ignored. Overlapping buoyant elements are double counted.",
    "Wind force: member-by-member projected area above the waterline × Cs × shielding factor, uniform or power-law profile. No interference or shielding unless a factor is set.",
    "Heeling moments are taken about the hydrodynamic reaction level: the area-weighted centroid of submerged lateral area (or half draft). No wave, current or added resistance.",
    "Line loads (tow legs, pull-in lines) act on the jacket at their connection points in a fixed space direction (heading, elevation). Tow pull = calm-water drag of submerged members × factor shared across 'tow' lines (bridle legs are scaled so their resultant along the mean heading equals the pull), or a fixed tension. The net horizontal line force is reacted by the water at the reaction level; the vertical component reduces the buoyancy needed and acts about the centre of buoyancy. Line tensions are inputs: no catenary, elasticity or dynamics.",
    "Free-surface effect is a uniform FSM/W·sinφ deduction from GZ. GM is the small-angle slope from ±1° free-trim solutions.",
    "Downflooding is checked at user-listed points only; progressive flooding is not modelled. The damaged case removes the buoyancy of flagged elements with no added weight.",
    "Set-down float check: level waterline from the buoyant elements, clearance = water depth + tide - (waterline above the lowest point of the structure). Legs and braces are not credited with buoyancy unless flagged buoyant. Ballast headroom adds or removes weight at the CoG until the clearance or tank-emergence limit is reached; it does not model the ballast tanks or their free surface.",
    "Acceptance criteria are user-set. Defaults are placeholders and must be confirmed against the project design basis.",
    "Static (quasi-steady) analysis: no dynamic response to waves, no gusting or tow-line dynamics.",
]


def build_pdf(snap: dict, res: dict, meta: dict, prepared: str, checked: str, rev: str) -> io.BytesIO:
    w = snap["widgets"]
    S, U = res["summary"], res["upright"]
    crit_ = js.criteria_from_widgets(w)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=92, bottomMargin=40,
                            title="Jacket wet-tow stability - screening")
    ss_ = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss_["Heading1"], fontSize=15, textColor=colors.HexColor("#08306b"))
    h2 = ParagraphStyle("h2", parent=ss_["Heading2"], fontSize=11, textColor=colors.HexColor("#08306b"), spaceBefore=10)
    nrm = ParagraphStyle("n", parent=ss_["Normal"], fontSize=8.5, leading=11)
    sml = ParagraphStyle("s", parent=ss_["Normal"], fontSize=7.2, leading=9)
    bold = ParagraphStyle("b", parent=nrm, fontName="Helvetica-Bold")
    flag = ParagraphStyle("f", parent=nrm, textColor=colors.HexColor("#b30000"), fontName="Helvetica-Bold")

    hdr_style = ParagraphStyle("hdr", parent=sml, textColor=colors.white, fontName="Helvetica-Bold")

    def P(t, st_=nrm):
        return Paragraph(escape(str(t)), st_)

    def HP(t):
        return Paragraph(escape(str(t)), hdr_style)

    def tbl(data, widths, header=True, font=7.2):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("FONTSIZE", (0, 0), (-1, -1), font),
                 ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#08306b")),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")])]
        t.setStyle(TableStyle(style))
        return t

    st_ = []
    st_.append(Paragraph("JACKET WET-TOW STABILITY - SCREENING CALCULATION", h1))
    st_.append(Paragraph("DRAFT - NOT INDEPENDENTLY CHECKED - NOT FOR CONSTRUCTION OR CERTIFICATION USE", flag))
    import json as _json
    _sig = lambda rows: _json.dumps([{k: v for k, v in x.items() if k != "tank"} for x in rows], sort_keys=True, default=str)
    if _sig(snap["elements"]) == _sig(js.synthetic_inputs()["elements"]):
        st_.append(Paragraph("SYNTHETIC ILLUSTRATIVE CASE - geometry, weights and CoG are OEC assumptions, not project data", flag))
    for _fl in snap.get("flags", []):
        st_.append(Paragraph(_fl, flag))
    st_.append(Spacer(1, 6))
    st_.append(tbl([
        [P("Project", bold), P(w["project"]), P("Scenario", bold), P("DAMAGED" if res["damaged"] else "INTACT")],
        [P("Prepared by", bold), P(prepared or "-"), P("Checked by", bold), P(checked or "-")],
        [P("Revision", bold), P(rev or "-"), P("Date", bold), P(datetime.date.today().strftime("%d-%b-%Y"))],
        [P("Tool", bold), P(f"jacket_stability engine v{APP_VERSION}"), P("Total weight", bold),
         P(f"{res['W']:,.1f} t   CoG ({res['G'][0]:.2f}, {res['G'][1]:.2f}, {res['G'][2]:.2f}) m")],
    ], [70, 190, 70, 190], header=False, font=8))

    st_.append(Paragraph("1. Summary against criteria (placeholders unless set by the project)", h2))
    def status(ok):
        return P("PASS" if ok else "FAIL", bold)
    gm_ok = S["gm_min"] >= crit_.gm_min
    heel_ok = (not S["no_equilibrium"]) and S["heel_max"] is not None and S["heel_max"] <= crit_.heel_max_deg
    ratio_ok = S["ratio_min"] is not None and S["ratio_min"] >= crit_.ratio_min
    rows = [[HP("Criterion"), HP("Required"), HP("Worst case"), HP("Status")],
            [P("Small-angle GM"), P(f">= {crit_.gm_min:.2f} m"), P(f"{S['gm_min']:.2f} m"), status(gm_ok)],
            [P("Static heel under wind + tow"), P(f"<= {crit_.heel_max_deg:.1f} deg"),
             P("no equilibrium" if S["no_equilibrium"] else fmt(S["heel_max"], 1, " deg")), status(heel_ok)],
            [P("Area ratio (GZ / heeling) to limit angle"), P(f">= {crit_.ratio_min:.2f}"),
             P(fmt(S["ratio_min"], 2)), status(ratio_ok)]]
    if crit_.df_min_deg > 0:
        rows.append([P("Downflooding angle"), P(f">= {crit_.df_min_deg:.1f} deg"), P(fmt(S["df_min"], 1, " deg")),
                     status(S["df_min"] is None or S["df_min"] >= crit_.df_min_deg)])
    rows.append([P("Overall"), P(""), P(f"governing heading {S['governing_beta']:.0f} deg"), status(S["passed"])])
    rows.append([P("Net line pull (information)"), P("report"),
                 P(f"{fmt(S['tow_kn'], 0, ' kN')}" if S["tow_kn"] is not None else "n/a"), P("INFO")])
    st_.append(tbl(rows, [190, 100, 150, 80], font=8))
    st_.append(Spacer(1, 4))
    st_.append(P(f"Level, no-load condition: draft {U['draft']:.2f} m, reserve buoyancy {U['reserve_buoyancy_pct']:.0f} %. "
                 f"Wind {w['p_wind_speed_kn']:.0f} kn, tow speed {w['p_tow_speed_kn']:.1f} kn.", nrm))

    T = res.get("tech")
    if T:
        st_.append(Spacer(1, 4))
        st_.append(P("Floating condition and hydrostatics (level, free trim, no wind or tow; KB, KG, KM above the lowest point of the structure; draft to the lowest buoyant point)", bold))
        gB, bB = T["G_body"], T["B_body"]
        hd_ = [[HP("Quantity"), HP("Value"), HP("Quantity"), HP("Value")],
               [P("Displacement"), P(f"{T['W']:,.0f} t  ({T['volume']:,.0f} m3)"), P("Draft / trim"),
                P(f"{T['draft']:.2f} m / {T['trim_deg']:.2f} deg")],
               [P("CoG (x, y, z)"), P(f"{gB[0]:.2f}, {gB[1]:.2f}, {gB[2]:.2f} m"), P("CoB (x, y, z)"),
                P(f"{bB[0]:.2f}, {bB[1]:.2f}, {bB[2]:.2f} m")],
               [P("KB / KG"), P(f"{T['kb_base']:.2f} / {T['kg_base']:.2f} m"), P("Waterplane area / TPC"),
                P(f"{T['waterplane_area']:,.0f} m2 / {T['tpc']:.2f} t/cm")],
               [P("GM, heel about y (wind +x)"), P(f"{T['gm_x']:.2f} m   (KM {T['kg_base'] + T['gm_x']:.2f}, BM {T['bm_x']:.2f})"),
                P("GM, heel about x (wind +y)"), P(f"{T['gm_y']:.2f} m   (KM {T['kg_base'] + T['gm_y']:.2f}, BM {T['bm_y']:.2f})")],
               [P("Reserve buoyancy"), P(f"{T['reserve_buoyancy_pct']:.0f} %"), P("Free-surface correction"),
                P(f"{T['fsc_m']:.3f} m")]]
        st_.append(tbl(hd_, [105, 150, 105, 150], font=7.2))

    F = res.get("float")
    if F:
        st_.append(Spacer(1, 4))
        st_.append(P("Set-down float check (level, no wind or tow; heights above the lowest point of the structure)", bold))
        cl, em = crit_.clear_min_m, crit_.emerged_min_m
        fr = [[HP("Quantity"), HP("Value"), HP("Required"), HP("Status")],
              [P("Waterline above base"), P(f"{F['draft_total']:.2f} m"), P(""), P("INFO")],
              [P("Water depth + tide"), P(fmt(F["depth_total"], 2, " m") if F["depth_total"] else "not set"), P(""), P("INFO")],
              [P("Seabed clearance"), P(fmt(F["clearance"], 2, " m")), P(f">= {cl:.2f} m" if cl > 0 else "not checked"),
               P("-" if F["pass_clearance"] is None else ("PASS" if F["pass_clearance"] else "FAIL"))],
              [P("Shortest tank length above water"), P(fmt(F["emerged_min"], 2, " m")), P(f">= {em:.2f} m" if em > 0 else "not checked"),
               P("-" if F["pass_emerged"] is None else ("PASS" if F["pass_emerged"] else "FAIL"))],
              [P("GM, level, no loads"), P(f"{F['gm_min']:.2f} m"), P(f">= {crit_.gm_min:.2f} m"),
               P("PASS" if F["pass_gm"] else "FAIL")],
              [P("Stable upright attitude"), P("yes" if F["stable"] else "NO - capsizes"), P(""), P("PASS" if F["stable"] else "FAIL")],
              [P("Freeboard to top of jacket"), P(f"{F['freeboard_top']:.1f} m"), P(""), P("INFO")]]
        if F["ballast_headroom_t"] is not None:
            fr.append([P(f"Ballast headroom ({F['governing_limit']} limit)"), P(f"{F['ballast_headroom_t']:,.0f} t"),
                       P("ballast added at CoG; negative = weight to shed"), P("INFO")])
        st_.append(tbl(fr, [170, 90, 160, 60], font=7.6))
        st_.append(P("Tank lengths above water: " + "; ".join(f"{t['name']} {t['emerged_m']:.1f} m" for t in F["tanks"]), sml))

    st_.append(Paragraph("2. Results by wind heading (0 deg = toward body +x)", h2))
    hdr = ["Wind", "GM m", "Heel deg", "Trim deg", "2nd int.", "Downfl.", "Limit", "Ratio", "GZmax m", "Pull kN", "Status"]
    data = [[HP(x) for x in hdr]]
    for h in res["heads"]:
        a = h["analysis"]
        data.append([P(f"{h['beta']:.0f}", sml), P(fmt(a["gm"], 2), sml), P(fmt(a["theta_s"]), sml),
                     P(fmt(a["trim_at_s"], 2), sml), P(fmt(a["theta_2"]), sml), P(fmt(a["theta_df"]), sml),
                     P(fmt(a["theta_lim"]), sml), P(fmt(a["ratio"], 2), sml),
                     P(fmt(a["gz_max"], 2), sml), P(fmt(a["tow_kn"], 0), sml), P("PASS" if a["passed"] else "FAIL", sml)])
    st_.append(tbl(data, [34, 40, 44, 44, 44, 44, 40, 40, 46, 44, 44]))

    gov = next(h for h in res["heads"] if h["beta"] == S["governing_beta"])
    sw = gov["sweep"]
    ok = ~(np.isnan(sw["gz"]) | np.isnan(sw["hl"]))
    if ok.sum() > 2:
        st_.append(Paragraph(f"3. Governing curve - wind toward {gov['beta']:.0f} deg", h2))
        d = Drawing(500, 230)
        lp = LinePlot()
        lp.x, lp.y, lp.width, lp.height = 45, 35, 430, 165
        lp.data = [list(zip(sw["phi_deg"][ok], sw["gz"][ok])), list(zip(sw["phi_deg"][ok], sw["hl"][ok]))]
        lp.lines[0].strokeColor = colors.HexColor("#1f77b4")
        lp.lines[0].strokeWidth = 1.8
        lp.lines[1].strokeColor = colors.HexColor("#d62728")
        lp.lines[1].strokeWidth = 1.4
        lp.lines[1].strokeDashArray = [4, 2]
        lp.xValueAxis.valueMin = float(sw["phi_deg"][ok].min())
        lp.xValueAxis.valueMax = float(sw["phi_deg"][ok].max())
        ymax = float(max(np.nanmax(sw["gz"][ok]), np.nanmax(sw["hl"][ok]), 0.1))
        ymin = float(min(np.nanmin(sw["gz"][ok]), np.nanmin(sw["hl"][ok]), 0.0))
        lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = ymin * 1.1, ymax * 1.1
        lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7
        lp.xValueAxis.labelTextFormat = "%.0f"
        lp.yValueAxis.labelTextFormat = "%.1f"
        d.add(lp)
        d.add(String(250, 12, "Heel angle [deg]", fontSize=8, textAnchor="middle"))
        d.add(String(10, 205, "Lever arm [m]", fontSize=8))
        d.add(String(300, 212, "blue = GZ    red dashed = heeling arm", fontSize=8))
        st_.append(d)

    try:
        els_f = js.elements_from_rows(snap["elements"])
        if els_f:
            st_.append(PageBreak())
            st_.append(Paragraph("4. Geometry and floating attitude", h2))
            st_.append(P("Member centre-lines drawn with stroke width following diameter (blue = tanks, dark = legs, "
                         "grey = braces, red = flooded). Leg names are grid names where the members carry them.", sml))
            st_.append(rf.figure_row(els_f, labels=True, height=300, title_prefix="As built - "))
            for att in meta.get("attitudes", []):
                st_.append(Spacer(1, 4))
                st_.append(KeepTogether([P(att["title"], bold),
                                         rf.figure_row(els_f, rot=att["rot"], zw=att["zw"], g_pt=att.get("G"),
                                                       b_pt=att.get("B"), height=300, views=("iso", "side", "end"))]))
    except Exception as _exc:          # a drawing problem must never stop the report
        st_.append(P(f"Figures could not be drawn: {_exc}", sml))

    st_.append(Paragraph("5. Input data", h2))
    st_.append(P("Weights", bold))
    wd = [[HP(c) for c in ("Item", "Mass t", "x m", "y m", "z m")]]
    for r in snap["weights"]:
        wd.append([P(r["item"], sml)] + [P(f"{float(r[c]):.2f}", sml) for c in ("mass_t", "x", "y", "z")])
    st_.append(tbl(wd, [210, 70, 70, 70, 70]))
    st_.append(Spacer(1, 5))
    st_.append(P("Elements (B = buoyant, E = exposed, F = flooded in damaged case)", bold))
    ed = [[HP(c) for c in ("Name", "Start (x,y,z)", "End (x,y,z)", "Ø out", "Ø in", "B", "E", "F")]]
    for r in snap["elements"]:
        try:
            ed.append([P(r["name"], sml),
                       P(f"{float(r['x1']):.1f}, {float(r['y1']):.1f}, {float(r['z1']):.1f}", sml),
                       P(f"{float(r['x2']):.1f}, {float(r['y2']):.1f}, {float(r['z2']):.1f}", sml),
                       P(f"{float(r['d_out']):.2f}", sml), P(f"{float(r.get('d_in') or 0):.2f}", sml),
                       P("Y" if r.get("buoyant") else "-", sml), P("Y" if r.get("exposed") else "-", sml),
                       P("Y" if r.get("flooded") else "-", sml)])
        except (TypeError, ValueError):
            continue
    st_.append(tbl(ed, [105, 105, 105, 40, 40, 22, 22, 22]))
    if snap["openings"]:
        st_.append(Spacer(1, 5))
        st_.append(P("Downflooding points", bold))
        od = [[HP(c) for c in ("Name", "x m", "y m", "z m")]]
        for r in snap["openings"]:
            od.append([P(r["name"], sml)] + [P(f"{float(r[c]):.2f}", sml) for c in ("x", "y", "z")])
        st_.append(tbl(od, [200, 70, 70, 70]))
    if snap.get("lines"):
        st_.append(Spacer(1, 5))
        st_.append(P("Additional line loads (force acts on the jacket; heading from body +x toward +y, elevation + up)", bold))
        ld = [[HP(c) for c in ("Line", "Point (x,y,z) m", "Heading deg", "Elev. deg", "Type", "Tension kN / share")]]
        for l_ in snap["lines"]:
            ld.append([P(l_["name"], sml), P(f"({l_['x']:.2f}, {l_['y']:.2f}, {l_['z']:.2f})", sml),
                       P(f"{l_['heading_deg']:.1f}", sml), P(f"{l_['elevation_deg']:.1f}", sml), P(l_["type"], sml),
                       P(f"share {l_['share']:.2f}" if l_["type"] == "Tow leg" else f"{l_['tension_kn']:.0f}", sml)])
        st_.append(tbl(ld, [110, 140, 60, 50, 70, 100]))
    st_.append(Spacer(1, 5))
    st_.append(P("Parameters", bold))
    pd_ = [
        ("Wind speed", f"{w['p_wind_speed_kn']:.1f} kn"), ("Cs / shielding / α / zref",
         f"{w['p_wind_cs']:.2f} / {w['p_shielding']:.2f} / {w['p_wind_alpha']:.3f} / {w['p_wind_zref']:.0f} m"),
        ("Densities (water, air)", f"{w['p_rho_w']:.3f} t/m3, {w['p_rho_a']:.3f} kg/m3"),
        ("Tow speed / Cd water", f"{w['p_tow_speed_kn']:.2f} kn / {w['p_cd_water']:.2f}"),
        ("Tow pull mode", f"{w['p_tow_mode']}" + (f" (x{w['p_tow_factor']:.2f})" if w['p_tow_mode'] == 'auto'
                                                    else f" ({w['p_tow_manual_kn']:.0f} kN)")),
        ("Tow line", f"({w['p_tow_x']:.2f}, {w['p_tow_y']:.2f}, {w['p_tow_z']:.2f}) m, heading {w['p_tow_heading_deg']:.0f} deg, "
                     f"elevation {w.get('p_tow_elev_deg', 0.0):.0f} deg"),
        ("Free-surface correction", f"FSM {w['p_fsm']:.0f} t.m -> {snap['fsc_m']:.3f} m"),
        ("Reaction level", w["p_lr_mode"]),
        ("Heel sweep", f"{w['a_phi_min']:.1f} to {w['a_phi_max']:.1f} deg, step {w['a_phi_step']:.1f}; headings every {w['a_step']:.1f} deg"),
        ("Integration cap", f"{w['c_cap_deg']:.0f} deg"),
    ]
    st_.append(tbl([[P(a_, sml), P(b_, sml)] for a_, b_ in pd_], [150, 380], header=False))

    st_.append(Paragraph("6. Assumptions and limitations", h2))
    for t in ASSUMPTIONS:
        st_.append(Paragraph("• " + escape(t), sml))
    st_.append(Spacer(1, 6))
    st_.append(P("This is a screening calculation. Results must be independently checked and the criteria "
                 "confirmed against the project design basis before use.", flag))

    logo_path = Path(__file__).with_name("oec_logo.png")

    def footer(canvas, doc_):
        canvas.saveState()
        if logo_path.exists():
            lh = 40.0
            lw = lh * 831.0 / 306.0
            canvas.drawImage(str(logo_path), A4[0] - 36 - lw, A4[1] - 36 - lh, width=lw, height=lh,
                             mask="auto", preserveAspectRatio=True)
            canvas.setStrokeColor(colors.HexColor("#08306b"))
            canvas.setLineWidth(0.6)
            canvas.line(36, A4[1] - 36 - lh - 5, A4[0] - 36, A4[1] - 36 - lh - 5)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(36, 24, f"{w['project']} - DRAFT screening calculation")
        canvas.drawRightString(A4[0] - 36, 24, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(st_, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf


