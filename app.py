import io
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# NumPy 2.0+ compatibility for trapezoidal integration
integrate_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Platform Tow Stability & VIM",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Offshore Platform Towing Stability & VIM")
st.caption("DNV-ST-N001 & DNV-RP-C205 Verification Engine | Intact Stability, Wind Heeling, and Vortex-Induced Motion")

# ==========================================
# 2. SIDEBAR INPUTS
# ==========================================
with st.sidebar.form("input_form"):
    st.subheader("Asset Parameters")
    
    with st.expander("📐 Platform Geometry", expanded=True):
        diameter = st.number_input("Outer Diameter (D) [m]", min_value=1.0, max_value=50.0, value=14.0, step=0.5,
                                   help="External hull cylinder diameter.")
        total_length = st.number_input("Total Length (L_tot) [m]", min_value=2.0, max_value=150.0, value=60.0, step=1.0,
                                       help="Overall length from bottom keel to upper deck.")
        draft = st.number_input("Towing Draft (T) [m]", min_value=1.0, max_value=120.0, value=45.0, step=0.5,
                                help="Submerged draft at transit waterline.")
        z_df = st.number_input("Downflooding Elevation [m]", min_value=0.0, max_value=150.0, value=55.0, step=0.5,
                               help="Height of lowest downflooding opening above keel.")

    with st.expander("⚖️ Mass & Ballast", expanded=True):
        displacement = st.number_input("Displacement [tonnes]", min_value=10.0, value=7100.0, step=100.0,
                                       help="Total mass including structural lightship and ballast.")
        kg = st.number_input("Vertical CoG (KG) [m]", min_value=0.1, value=20.0, step=0.5,
                             help="Center of gravity height above keel.")
        fsm = st.number_input("Free Surface Moment [t·m]", min_value=0.0, value=0.0, step=50.0,
                              help="Virtual KG rise due to slack internal tanks.")

    with st.expander("🌪️ Metocean / Environment", expanded=False):
        wind_speed = st.number_input("Design Wind Speed [kts]", min_value=0.0, value=50.0, step=1.0,
                                     help="1-minute sustained wind speed.")
        wind_area = st.number_input("Projected Wind Area [m²]", min_value=1.0, value=280.0, step=10.0,
                                    help="Exposed area above waterline.")
        wind_z = st.number_input("Wind Center above SWL [m]", min_value=0.0, value=22.0, step=0.5,
                                 help="Elevation of wind centroid above waterline.")
        cd_wind = st.number_input("Wind Shape Factor (Cs)", min_value=0.4, max_value=1.5, value=0.7, step=0.05,
                                  help="0.65-0.70 for cylinders.")

    with st.expander("⚓ Towing & VIM", expanded=False):
        tow_speed = st.number_input("Tow Speed [kts]", min_value=0.1, max_value=10.0, value=2.5, step=0.1,
                                    help="Transit speed through water.")
        z_tow = st.number_input("Tow Bracket Elev. from SWL [m]", min_value=-60.0, max_value=30.0, value=2.0, step=0.5,
                                help="+ is above SWL, - is submerged.")
        has_strakes = st.checkbox("Helical Strakes Installed?", value=False,
                                  help="VIM suppression strakes cap cross-flow amplitude Ay/D <= 0.10.")

    submitted = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

# ==========================================
# 3. DEFENSIVE VALIDATION
# ==========================================
errors = []
warnings = []

if draft >= total_length:
    errors.append(f"Towing Draft ({draft:.1f} m) must be less than Total Length ({total_length:.1f} m).")
if kg >= total_length:
    errors.append(f"Vertical CoG ({kg:.1f} m) cannot exceed Total Length ({total_length:.1f} m).")
if z_df <= draft:
    errors.append(f"Downflooding opening ({z_df:.1f} m) is submerged below Draft ({draft:.1f} m).")

vol_cyl = (np.pi * (diameter ** 2) / 4.0) * draft
expected_disp = vol_cyl * 1.025
if displacement > 0 and abs(displacement - expected_disp) / expected_disp > 0.08:
    warnings.append(
        f"Input displacement ({displacement:,.0f} t) differs by >8% from ideal cylinder volume ({expected_disp:,.0f} t). "
        "Verify internal void spaces, floodings, or taper geometry."
    )

for err in errors:
    st.error(f"❌ **Validation Error:** {err}")
for wrn in warnings:
    st.warning(f"⚠️ **Engineering Notice:** {wrn}")

if errors:
    st.stop()

# ==========================================
# 4. CALCULATION ENGINE
# ==========================================
# Hydrostatics
kb = draft / 2.0
bm = (diameter ** 2) / (16.0 * draft)
km = kb + bm
delta_kg_fsm = fsm / displacement if displacement > 0 else 0.0
kg_eff = kg + delta_kg_fsm
gm_calc = km - kg_eff

# Intact stability curve (clamped to 40° to avoid tan() singularity)
angles_deg = np.linspace(0.0, 40.0, 100)
angles_rad = np.radians(angles_deg)

# Wall-sided formula for cylinder
gz_curve = np.sin(angles_rad) * (gm_calc + 0.5 * bm * (np.tan(angles_rad) ** 2))

# Wind heeling arm (Defensive check for 0 wind speed)
v_wind_ms = wind_speed * 0.514444
f_wind_kn = 0.5 * 1.225 * (v_wind_ms ** 2) * cd_wind * wind_area / 1000.0
lever_arm_wind = wind_z + (draft / 2.0)
m_wind_knm = f_wind_kn * lever_arm_wind
wind_arm_0 = m_wind_knm / (displacement * 9.80665) if displacement > 0 else 0.0
wind_curve = wind_arm_0 * (np.cos(angles_rad) ** 2)

# Downflood angle limit
freeboard_df = max(0.001, z_df - draft)
theta_df_deg = np.degrees(np.arctan(freeboard_df / (diameter / 2.0)))
theta_limit_deg = float(np.clip(theta_df_deg, 2.0, 40.0))

mask_limit = angles_deg <= theta_limit_deg
area_gz = integrate_trapz(gz_curve[mask_limit], angles_rad[mask_limit])
area_wind = integrate_trapz(wind_curve[mask_limit], angles_rad[mask_limit])

if area_wind > 1e-6:
    energy_ratio = area_gz / area_wind
else:
    energy_ratio = 999.0  # Infinite margin under zero/negligible wind

# Natural dynamics & VIM
r_yy = np.sqrt((draft ** 2) / 12.0 + (kg_eff - draft / 2.0) ** 2)
i_pitch_total = (displacement * 1000.0) * (r_yy ** 2) * 2.0  # (1 + Ca=1.0)
stiffness_pitch = (displacement * 1000.0) * 9.80665 * max(gm_calc, 0.001)

if gm_calc > 0.0:
    t_n_pitch = 2.0 * np.pi * np.sqrt(i_pitch_total / stiffness_pitch)
    f_n_pitch = 1.0 / t_n_pitch
else:
    t_n_pitch = 999.0
    f_n_pitch = 0.001

u_tow_ms = tow_speed * 0.514444
vr_val = u_tow_ms / (f_n_pitch * diameter) if (f_n_pitch * diameter) > 0 else 0.0
vim_lockin = (4.0 <= vr_val <= 8.5) and not has_strakes

if has_strakes:
    ay_over_d = 0.10
elif 4.0 <= vr_val <= 8.5:
    ay_over_d = float(np.sin(np.pi * (vr_val - 4.0) / 4.5))
    ay_over_d = max(0.0, ay_over_d)
else:
    ay_over_d = 0.0

cd_eff = 0.70 * (1.0 + 2.0 * ay_over_d)
proj_area = diameter * draft
drag_kn = (0.5 * 1025.0 * (u_tow_ms ** 2) * cd_eff * proj_area) / 1000.0
drag_tonnes = drag_kn / 9.80665

# Tow pitch-trim moment
lever_arm_tow = z_tow + (draft / 2.0)
tow_moment_knm = drag_kn * lever_arm_tow
tow_trim_deg = np.degrees((tow_moment_knm * 1000.0) / stiffness_pitch) if stiffness_pitch > 0 else 90.0

pass_gm = gm_calc >= 1.0
pass_energy = energy_ratio >= 1.3
pass_trim = abs(tow_trim_deg) <= 2.0
pass_all = bool(pass_gm and pass_energy and pass_trim)

# ==========================================
# 5. ELEVATION SILHOUETTE
# ==========================================
def render_elevation():
    radius = diameter / 2.0
    z_keel = -draft
    z_deck = total_length - draft
    z_kb_swl = -draft / 2.0
    z_kg_swl = kg_eff - draft
    z_df_swl = z_df - draft

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color="#1f77b4", width=2, dash="dash"),
                  annotation_text="SWL (0.0 m)", annotation_position="top left")
    
    # Submerged Hull
    fig.add_trace(go.Scatter(
        x=[-radius, radius, radius, -radius, -radius],
        y=[z_keel, z_keel, 0.0, 0.0, z_keel],
        fill="toself", fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="#08306b", width=2), name="Submerged Hull", hoverinfo="skip"
    ))
    # Emerged Hull
    fig.add_trace(go.Scatter(
        x=[-radius, radius, radius, -radius, -radius],
        y=[0.0, 0.0, z_deck, z_deck, 0.0],
        fill="toself", fillcolor="rgba(189, 189, 189, 0.3)",
        line=dict(color="#525252", width=2), name="Freeboard", hoverinfo="skip"
    ))

    pts = [
        (0.0, z_kb_swl, "#2ca02c", "circle", f"KB: {z_kb_swl:+.2f} m"),
        (0.0, z_kg_swl, "#d62728", "x", f"KG: {z_kg_swl:+.2f} m"),
        (radius, z_tow, "#9467bd", "triangle-right", f"Tow Point: {z_tow:+.2f} m"),
        (radius, z_df_swl, "#ff7f0e", "diamond", f"Downflood: {z_df_swl:+.2f} m")
    ]
    for x, y, c, s, name in pts:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=10, color=c, symbol=s),
            text=[name], textposition="top right" if x >= 0 else "top left",
            name=name
        ))

    fig.update_layout(
        title="<b>Asset Elevation Profile (SWL Reference)</b>",
        xaxis=dict(title="Radial Offset [m]", range=[-radius * 2.2, radius * 3.0], zeroline=True),
        yaxis=dict(title="Elevation from SWL [m]", range=[z_keel * 1.15, max(z_deck, z_df_swl) * 1.25]),
        template="plotly_white", height=320, showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    return fig

st.plotly_chart(render_elevation(), use_container_width=True)

# ==========================================
# 6. RESULTS DASHBOARD & CHARTS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Hydrostatics & Stability", "🌀 VIM & Resistance", "📄 Report Export"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Intact GM", f"{gm_calc:.2f} m", "≥ 1.00 m (Pass)" if pass_gm else "Fail (< 1.0 m)",
                delta_color="normal" if pass_gm else "inverse")
    col2.metric("Energy Ratio", f"{energy_ratio:.2f}" if energy_ratio < 100 else "N/A (Calm)",
                "≥ 1.30 (Pass)" if pass_energy else "Fail (< 1.30)",
                delta_color="normal" if pass_energy else "inverse")
    col3.metric("Downflood Angle", f"{theta_limit_deg:.1f}°", "Acceptable")
    col4.metric("Tow-Induced Trim", f"{tow_trim_deg:+.2f}°", "≤ 2.0° (Pass)" if pass_trim else "Fail (> 2.0°)",
                delta_color="normal" if pass_trim else "inverse")

    fig_gz = go.Figure()
    fig_gz.add_trace(go.Scatter(x=angles_deg, y=gz_curve, mode='lines', name='Righting Arm (GZ)', line=dict(color='#1f77b4', width=3)))
    fig_gz.add_trace(go.Scatter(x=angles_deg, y=wind_curve, mode='lines', name='Wind Heeling Arm', line=dict(color='#d62728', width=2, dash='dash')))

    if np.sum(mask_limit) > 1:
        fig_gz.add_trace(go.Scatter(
            x=np.concatenate([angles_deg[mask_limit], angles_deg[mask_limit][::-1]]),
            y=np.concatenate([gz_curve[mask_limit], np.zeros(np.sum(mask_limit))]),
            fill='toself', fillcolor='rgba(31, 119, 180, 0.15)', line=dict(color='rgba(255,255,255,0)'),
            name='Area under GZ', hoverinfo='skip'
        ))

    fig_gz.add_vline(x=theta_limit_deg, line_dash='dot', line_color='#ff7f0e',
                     annotation_text=f"Limit θ = {theta_limit_deg:.1f}°")
    fig_gz.update_layout(
        title="<b>GZ Righting Arm vs. Wind Heeling Arm</b> (DNV-ST-N001 Sec 11)",
        xaxis=dict(title="Heel Angle [deg]", range=[0, 40]),
        yaxis=dict(title="Lever Arm [m]"),
        template="plotly_white", hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_gz, use_container_width=True)

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Reduced Velocity (Vr)", f"{vr_val:.2f}", "Lock-In (4.0 - 8.5)" if vim_lockin else "Clear")
    c2.metric("Amplified Tow Drag", f"{drag_tonnes:.1f} t", f"{drag_kn:.0f} kN")
    c3.metric("Drag Coefficient (Cd)", f"{cd_eff:.2f}", f"Baseline: 0.70 (+{((cd_eff/0.7)-1)*100:.0f}%)")

    # Speed sweep for VIM curve
    sweep_kts = np.linspace(0.5, 5.5, 80)
    sweep_u = sweep_kts * 0.514444
    sweep_vr = sweep_u / (f_n_pitch * diameter)

    sweep_ay = np.zeros_like(sweep_vr)
    if not has_strakes:
        l_mask = (sweep_vr >= 4.0) & (sweep_vr <= 8.5)
        sweep_ay[l_mask] = np.sin(np.pi * (sweep_vr[l_mask] - 4.0) / 4.5)
        sweep_ay = np.clip(sweep_ay, 0.0, None)
    else:
        sweep_ay.fill(0.10)

    sweep_cd = 0.70 * (1.0 + 2.0 * sweep_ay)
    sweep_drag = (0.5 * 1025.0 * (sweep_u ** 2) * sweep_cd * proj_area) / 9806.65
    bare_drag = (0.5 * 1025.0 * (sweep_u ** 2) * 0.70 * proj_area) / 9806.65

    fig_vim = make_subplots(specs=[[{"secondary_y": True}]])
    u_lock_start = (4.0 * f_n_pitch * diameter) / 0.514444
    u_lock_end = (8.5 * f_n_pitch * diameter) / 0.514444

    fig_vim.add_vrect(x0=u_lock_start, x1=u_lock_end, fillcolor="rgba(255, 0, 0, 0.12)", line_width=0,
                      annotation_text="Lock-In (4.0 ≤ Vr ≤ 8.5)", annotation_position="top left")
    fig_vim.add_trace(go.Scatter(x=sweep_kts, y=bare_drag, name="Bare Drag", line=dict(color="#2ca02c", dash="dash")), secondary_y=False)
    fig_vim.add_trace(go.Scatter(x=sweep_kts, y=sweep_drag, name="Amplified Drag", line=dict(color="#d62728" if not has_strakes else "#1f77b4", width=2.5)), secondary_y=False)
    fig_vim.add_trace(go.Scatter(x=sweep_kts, y=sweep_vr, name="Vr", line=dict(color="#9467bd", dash="dot")), secondary_y=True)
    fig_vim.add_trace(go.Scatter(x=[tow_speed], y=[drag_tonnes], mode="markers+text", marker=dict(size=10, color="black"),
                                 name="Design Speed", text=[f"{tow_speed} kts"], textposition="top left"), secondary_y=False)

    fig_vim.update_layout(
        title="<b>VIM Screening & Tow Resistance Curve</b> (DNV-RP-C205)",
        xaxis=dict(title="Tow Speed [kts]", range=[0.5, 5.5]),
        template="plotly_white", hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_vim.update_yaxes(title_text="Tow Drag [tonnes]", secondary_y=False)
    fig_vim.update_yaxes(title_text="Reduced Velocity (Vr)", secondary_y=True)
    st.plotly_chart(fig_vim, use_container_width=True)

with tab3:
    st.subheader("Verification Report & Data Dispatch")
    col_rep1, col_rep2 = st.columns([1, 1])

    with col_rep1:
        st.markdown("#### Download Stamped PDF")
        proj_name = st.text_input("Asset / Project Name", value="Offshore Vertical Spar - Wet Tow")
        checker_name = st.text_input("Lead Engineer", value="Principal Consultant")

        def build_clean_pdf():
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            story = []
            styles = getSampleStyleSheet()

            t_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#08306b"))
            c_bold = ParagraphStyle('CB', parent=styles['Normal'], fontSize=8.5, fontName="Helvetica-Bold")
            c_norm = ParagraphStyle('CN', parent=styles['Normal'], fontSize=8.5)

            story.append(Paragraph("<b>WET TOW STABILITY & VIM VERIFICATION REPORT</b>", t_style))
            story.append(Paragraph(f"Asset: {proj_name} | Date: {datetime.date.today().strftime('%d-%b-%Y')} | Checker: {checker_name}", c_norm))
            story.append(Spacer(1, 12))

            summary_data = [
                [Paragraph("Criterion", c_bold), Paragraph("Required", c_bold), Paragraph("Calculated", c_bold), Paragraph("Status", c_bold)],
                [Paragraph("Initial GM", c_norm), Paragraph("≥ 1.00 m", c_norm), Paragraph(f"{gm_calc:.2f} m", c_norm),
                 Paragraph("<font color='green'><b>PASS</b></font>" if pass_gm else "<font color='red'><b>FAIL</b></font>", c_norm)],
                [Paragraph("Area Ratio (GZ/Wind)", c_norm), Paragraph("≥ 1.30", c_norm), Paragraph(f"{energy_ratio:.2f}", c_norm),
                 Paragraph("<font color='green'><b>PASS</b></font>" if pass_energy else "<font color='red'><b>FAIL</b></font>", c_norm)],
                [Paragraph("Tow Trim Angle", c_norm), Paragraph("≤ 2.0°", c_norm), Paragraph(f"{tow_trim_deg:.2f}°", c_norm),
                 Paragraph("<font color='green'><b>PASS</b></font>" if pass_trim else "<font color='red'><b>FAIL</b></font>", c_norm)],
                [Paragraph("VIM Lock-in", c_norm), Paragraph("Clear", c_norm), Paragraph(f"Vr={vr_val:.2f}", c_norm),
                 Paragraph("<font color='orange'><b>LOCK-IN</b></font>" if vim_lockin else "<font color='green'><b>CLEAR</b></font>", c_norm)],
                [Paragraph("Total Bollard Pull Req.", c_norm), Paragraph("Report", c_norm), Paragraph(f"{drag_tonnes:.1f} t ({drag_kn:.0f} kN)", c_norm),
                 Paragraph("INFO", c_norm)]
            ]
            t = Table(summary_data, colWidths=[160, 110, 130, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#08306b")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t)
            doc.build(story)
            buf.seek(0)
            return buf

        pdf_bytes = build_clean_pdf()
        st.download_button("📄 Download Official Verification PDF", data=pdf_bytes,
                           file_name=f"Tow_Stability_{proj_name.replace(' ', '_')}.pdf",
                           mime="application/pdf", use_container_width=True)

    with col_rep2:
        st.markdown("#### Export Numerical Data")
        st.write("Extract the $GZ$ arms and drag simulation points directly into CSV format.")
        csv_df = pd.DataFrame({
            "Heel_Angle_deg": angles_deg,
            "GZ_Arm_m": gz_curve,
            "Wind_Arm_m": wind_curve
        })
        st.download_button("📥 Download Hydrostatics CSV", data=csv_df.to_csv(index=False).encode('utf-8'),
                           file_name="hydrostatics_gz_data.csv", mime="text/csv", use_container_width=True)