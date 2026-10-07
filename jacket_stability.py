"""
jacket_stability.py
===================
Screening-level stability engine for a *floating jacket on buoyancy tanks*
during a wet tow.  Pure NumPy (no Streamlit), so it can be unit-tested and
reused outside the app.

What it does
------------
* Hydrostatics of any set of straight circular cylinders / annular tanks
  (arbitrary axis, closed-form cross-section cuts, GL quadrature along the
  axis) - no lumped "D**2/16T" shortcuts.
* Free-to-trim heel sweep about any horizontal axis: for each imposed heel
  angle the draft and trim are solved so that displacement = weight and the
  moment about the trim axis is zero.
* Wind (member-by-member projected area) and tow-line loads, reacted by a
  hydrodynamic force at the lateral-resistance level.
* Downflooding angle, static equilibrium angle, intercepts, area ratio,
  small-angle GM and a damaged (lost-buoyancy) case.

Conventions (SI + tonnes)
-------------------------
Body frame: x forward (tow direction), y athwart, z up when floating level.
Mass in tonnes, lengths in m, forces in kN, moments in kN*m.

Governing equations
-------------------
Weight / buoyancy balance      :  rho_w * V_sub(z_w, orientation) = W
Free-to-trim equilibrium       :  (B - G)_H = M_P,ext / (W g)       [m]
Righting arm (heel about H)    :  GZ = -(B - G)_P - FSC * sin(phi)  [m]
Wind force on member slice     :  F = 1/2 rho_a V(z)^2 Cs * D * dl * |a x w|
Heeling arm                    :  HL = M_H,ext / (W g)              [m]
where H is the horizontal heel axis, P = z x H the horizontal trim axis,
B/G are the centres of buoyancy/gravity in the space frame, FSC is the
free-surface correction (m) and a is the member axis unit vector.

Limits of validity are listed in README.md.  This is a screening tool: all
results must be checked against the project design basis before use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

G_ACC: float = 9.80665      # m/s^2
KN_TO_MS: float = 0.514444  # knots -> m/s
EZ: np.ndarray = np.array([0.0, 0.0, 1.0])

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
_XG, _WG = np.polynomial.legendre.leggauss(40)


# ----------------------------------------------------------------------------
# Input data model
# ----------------------------------------------------------------------------
@dataclass
class Element:
    """Straight circular cylinder (or annulus) between two points, body frame [m]."""
    name: str
    p1: tuple
    p2: tuple
    d_out: float
    d_in: float = 0.0
    buoyant: bool = True     # contributes displacement (watertight)
    exposed: bool = True     # contributes wind area / tow drag
    flooded: bool = False    # damage case: loses buoyancy
    tank: bool = True        # buoyant member counted as a buoyancy tank in the emerged-length check
                             # (False for sealed legs / braces that only add displacement)


@dataclass
class WeightItem:
    name: str
    mass_t: float
    x: float
    y: float
    z: float


@dataclass
class Opening:
    """Downflooding point (vent, hatch...) in the body frame [m]."""
    name: str
    x: float
    y: float
    z: float


@dataclass
class Line:
    """External line load on the jacket: a tow leg, a pull-in line, etc.

    The load acts ON the jacket at `point` (body frame) in the direction given by
    heading and elevation, both fixed in space relative to the upright body axes.
    mode 'fixed': tension_kn is applied.  mode 'tow': the line carries `share`
    (relative) of the tow pull set by the calm-water drag, scaled so the legs'
    resultant along their mean heading equals drag x margin (a bridle)."""
    name: str
    point: tuple
    heading_deg: float = 0.0     # horizontal direction of pull, deg from body +x toward +y
    elevation_deg: float = 0.0   # + = pulls upward
    mode: str = "fixed"          # 'fixed' | 'tow'
    tension_kn: float = 0.0
    share: float = 1.0


@dataclass
class Params:
    rho_w: float = 1.025          # t/m3
    rho_a: float = 1.225          # kg/m3
    wind_speed_kn: float = 40.0
    wind_cs: float = 1.0          # shape coefficient on member projected area
    wind_alpha: float = 0.0       # height profile exponent (0 = uniform)
    wind_zref: float = 10.0       # m above waterline for the profile
    cd_water: float = 1.0         # drag coefficient on submerged members
    shielding: float = 1.0        # multiplier on projected areas (1 = none)
    tow_speed_kn: float = 2.5
    tow_point: tuple = (0.0, 0.0, 0.0)
    tow_heading_deg: float = 0.0  # tow direction relative to body +x (upright)
    tow_mode: str = "auto"        # 'auto' (calm-water drag x factor) | 'manual'
    tow_manual_kn: float = 0.0
    tow_elevation_deg: float = 0.0  # legacy single tow line: + = pulls upward
    lines: tuple = ()             # tuple of Line; empty = one tow line from the tow_* fields above
    tow_factor: float = 1.0       # margin on calm-water drag (auto mode)
    fsc_m: float = 0.0            # free-surface correction FSM/W [m]
    lr_mode: str = "area"         # 'area' | 'half_draft'
    slice_m: float = 0.5          # slice length for wind/drag integration
    trim_limit_deg: float = 25.0  # search limit for free trim
    water_depth_m: float = 0.0    # depth to seabed at LAT [m]; 0 = seabed clearance not checked
    tide_m: float = 0.0           # tide height above LAT [m] (HAT-LAT for the worst case)


@dataclass
class Criteria:
    """User-set acceptance limits.  Defaults are PLACEHOLDERS - confirm against
    the project design basis / marine warranty requirements."""
    gm_min: float = 1.0       # m
    heel_max_deg: float = 15.0  # static heel angle under wind + tow
    ratio_min: float = 1.3    # area under GZ / area under heeling arm
    cap_deg: float = 40.0     # upper limit of integration
    df_min_deg: float = 0.0   # minimum downflooding angle (0 = not checked)
    clear_min_m: float = 0.0  # minimum clearance, lowest point to seabed (0 = not checked)
    emerged_min_m: float = 0.0  # minimum length of each buoyancy tank above the waterline (0 = not checked)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
@dataclass
class ElemArrays:
    p1: np.ndarray
    a: np.ndarray
    length: np.ndarray
    ro: np.ndarray
    ri: np.ndarray

    @property
    def n(self) -> int:
        return len(self.length)

    @property
    def vol_total(self) -> float:
        return float(np.sum(np.pi * (self.ro**2 - self.ri**2) * self.length))


def build_elem_arrays(elements: Sequence[Element]) -> ElemArrays:
    p1, a, L, ro, ri = [], [], [], [], []
    for e in elements:
        q1 = np.asarray(e.p1, float)
        q2 = np.asarray(e.p2, float)
        ln = float(np.linalg.norm(q2 - q1))
        if ln < 1e-9 or e.d_out <= 0.0:
            continue
        p1.append(q1)
        a.append((q2 - q1) / ln)
        L.append(ln)
        ro.append(0.5 * e.d_out)
        ri.append(0.5 * max(e.d_in, 0.0))
    if not p1:
        z3 = np.zeros((0, 3))
        z1 = np.zeros(0)
        return ElemArrays(z3, z3.copy(), z1, z1.copy(), z1.copy())
    return ElemArrays(np.array(p1), np.array(a), np.array(L), np.array(ro), np.array(ri))


@dataclass
class Slices:
    """Thin discs along exposed members, used for wind / drag integration."""
    c: np.ndarray
    a: np.ndarray
    r_o: np.ndarray
    t: np.ndarray

    @property
    def n(self) -> int:
        return len(self.t)


def build_slices(elements: Sequence[Element], slice_m: float, min_slices: int = 4) -> Slices:
    cs, as_, ro, ts = [], [], [], []
    for e in elements:
        q1 = np.asarray(e.p1, float)
        q2 = np.asarray(e.p2, float)
        ln = float(np.linalg.norm(q2 - q1))
        if ln < 1e-9 or e.d_out <= 0.0:
            continue
        a = (q2 - q1) / ln
        n = max(min_slices, int(math.ceil(ln / slice_m)))
        t = ln / n
        s = (np.arange(n) + 0.5) * t
        cs.append(q1[None, :] + s[:, None] * a[None, :])
        as_.append(np.tile(a, (n, 1)))
        ro.append(np.full(n, 0.5 * e.d_out))
        ts.append(np.full(n, t))
    if not cs:
        return Slices(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    return Slices(np.vstack(cs), np.vstack(as_), np.concatenate(ro), np.concatenate(ts))


def _segment(r: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Part of a circle (radius r) on the side x' <= d of a chord.

        A(d) = r^2 [pi/2 + asin(d/r)] + d sqrt(r^2 - d^2)
        M(d) = integral of x' dA = -(2/3) (r^2 - d^2)^(3/2)      (about centre)

    d is clipped to [-r, r] so |d| >= r returns the empty / full disc.
    """
    rs = np.where(r > 0.0, r, 1.0)
    dc = np.clip(d, -r, r)
    rt = np.sqrt(np.maximum(r * r - dc * dc, 0.0))
    area = r * r * (0.5 * np.pi + np.arcsin(np.clip(dc / rs, -1.0, 1.0))) + dc * rt
    mom = -(2.0 / 3.0) * rt**3
    return area, mom


def cut(ea: ElemArrays, u: np.ndarray, zw: float) -> tuple[float, np.ndarray]:
    """Submerged volume and first moment of volume (body frame) below the plane
    u . p = zw, for all cylinders in ``ea``.

    For a cylinder with axis a, start point P1 and length L the plane meets the
    disc at axial position s along a chord at signed distance

        d(s) = (zw - u.P1 - s (u.a)) / m ,   m = sqrt(1 - (u.a)^2)

    from the disc centre.  Fully submerged / dry stretches are exact; the short
    stretch where |d| <= r is integrated with 40-point Gauss-Legendre:

        V = int A(d(s)) ds ,  int s A ds ,  int M(d(s)) ds
        first moment = P1 V + a int(sA) + e int(M),  e = (u - (u.a) a)/m
    """
    if ea.n == 0:
        return 0.0, np.zeros(3)
    h0 = ea.p1 @ u
    sa = ea.a @ u
    m = np.sqrt(np.maximum(1.0 - sa * sa, 0.0))
    vert = m < 1e-9
    m_s = np.where(vert, 1.0, m)
    sa_s = np.where(np.abs(sa) < 1e-12, 1e-12, sa)
    L, ro, ri = ea.length, ea.ro, ea.ri
    afull = np.pi * (ro**2 - ri**2)

    # --- non-vertical axes -------------------------------------------------
    s_a = (zw - h0 - ro * m) / sa_s          # d = +ro
    s_b = (zw - h0 + ro * m) / sa_s          # d = -ro
    s1 = np.clip(np.minimum(s_a, s_b), 0.0, L)
    s2 = np.clip(np.maximum(s_a, s_b), 0.0, L)
    pos = sa_s > 0.0
    ell_f = np.where(pos, s1, L - s2)
    ia_full = afull * ell_f
    isa_full = np.where(pos, afull * s1**2 / 2.0, afull * (L**2 - s2**2) / 2.0)

    half = 0.5 * (s2 - s1)
    x = half[:, None] * _XG[None, :] + 0.5 * (s1 + s2)[:, None]
    w = half[:, None] * _WG[None, :]
    d = (zw - h0[:, None] - x * sa[:, None]) / m_s[:, None]
    ao, mo = _segment(ro[:, None], d)
    ai, mi = _segment(ri[:, None], d)
    a_tr = ao - ai
    m_tr = mo - mi
    ia = ia_full + (w * a_tr).sum(axis=1)
    isa = isa_full + (w * x * a_tr).sum(axis=1)
    im = (w * m_tr).sum(axis=1)

    # --- vertical axes (plane normal to axis) -------------------------------
    s_star = np.clip((zw - h0) / sa_s, 0.0, L)
    v_v = np.where(sa > 0.0, afull * s_star, afull * (L - s_star))
    isa_v = np.where(sa > 0.0, afull * s_star**2 / 2.0, afull * (L**2 - s_star**2) / 2.0)

    vol_e = np.where(vert, v_v, ia)
    isa_e = np.where(vert, isa_v, isa)
    im_e = np.where(vert, 0.0, im)
    e_dir = (u[None, :] - sa[:, None] * ea.a) / m_s[:, None]
    mom = ea.p1 * vol_e[:, None] + ea.a * isa_e[:, None] + e_dir * im_e[:, None]
    return float(vol_e.sum()), mom.sum(axis=0)


def _illinois(f: Callable[[float], float], a: float, b: float, fa: float, fb: float,
              tol_f: float, tol_x: float = 1e-10, maxit: int = 100) -> float:
    """Bracketed root finder (Illinois variant of regula falsi)."""
    if abs(fa) <= tol_f:
        return a
    if abs(fb) <= tol_f:
        return b
    side = 0
    x = a
    for _ in range(maxit):
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        if abs(fx) <= tol_f or abs(b - a) <= tol_x:
            return x
        if fx * fb > 0:
            b, fb = x, fx
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = x, fx
            if side == 1:
                fb *= 0.5
            side = 1
    return x


def solve_waterline(ea: ElemArrays, u: np.ndarray, v_target: float,
                    guess: Optional[float] = None) -> Optional[float]:
    """Plane height z_w with submerged volume = v_target, or None if the
    buoyant volume is insufficient.   V(z_w) is monotone, so a bracketed
    solve is robust."""
    h0 = ea.p1 @ u
    sa = ea.a @ u
    m = np.sqrt(np.maximum(1.0 - sa * sa, 0.0))
    h1 = h0 + ea.length * sa
    zmin = float((np.minimum(h0, h1) - ea.ro * m).min())
    zmax = float((np.maximum(h0, h1) + ea.ro * m).max())
    if ea.vol_total < v_target * (1.0 + 1e-9):
        return None

    def f(z: float) -> float:
        return cut(ea, u, z)[0] - v_target

    tol = 1e-8 * max(v_target, 1.0)
    lo, hi = zmin, zmax
    flo, fhi = -v_target, ea.vol_total - v_target
    if guess is not None and zmin <= guess <= zmax:
        dz = 0.5
        l2, h2 = max(zmin, guess - dz), min(zmax, guess + dz)
        fl2, fh2 = f(l2), f(h2)
        k = 0
        while fl2 > 0 and l2 > zmin and k < 20:
            h2, fh2 = l2, fl2
            dz *= 2.0
            l2 = max(zmin, l2 - dz)
            fl2 = f(l2)
            k += 1
        while fh2 < 0 and h2 < zmax and k < 40:
            l2, fl2 = h2, fh2
            dz *= 2.0
            h2 = min(zmax, h2 + dz)
            fh2 = f(h2)
            k += 1
        if fl2 <= 0 <= fh2:
            lo, hi, flo, fhi = l2, h2, fl2, fh2
    return _illinois(f, lo, hi, flo, fhi, tol)


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    ax = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + math.sin(ang) * k + (1.0 - math.cos(ang)) * (k @ k)


def frame(psi: float, phi: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation R (space <- body) plus the space-fixed horizontal axes.

    H = (cos psi, sin psi, 0) heel axis,  P = z x H trim axis,
    R = Rot_H(phi) @ Rot_P(theta)  (trim first, then heel about space H).
    Positive phi lowers the -P side (wind blowing toward -P heels positive).
    """
    h = np.array([math.cos(psi), math.sin(psi), 0.0])
    p = np.array([-math.sin(psi), math.cos(psi), 0.0])
    return _rot(h, phi) @ _rot(p, theta), h, p


def member_kind(e: Element) -> str:
    """Display class of a member, from its flags and inclination."""
    if e.buoyant and e.tank:
        return "Buoyancy tank"
    a = np.asarray(e.p2, float) - np.asarray(e.p1, float)
    s = abs(a[2]) / max(float(np.linalg.norm(a)), 1e-12)
    if s > 0.8:
        return "Leg / vertical"
    if s < 0.2:
        return "Horizontal brace"
    return "Diagonal brace"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class JacketModel:
    """Floating-jacket model: weights, buoyant cylinders, exposed members."""

    def __init__(self, elements: Sequence[Element], weights: Sequence[WeightItem],
                 openings: Sequence[Opening], params: Params, damaged: bool = False) -> None:
        self.p = params
        self.damaged = damaged
        self.W = float(sum(w.mass_t for w in weights))
        if self.W <= 0.0:
            raise ValueError("Total weight must be positive.")
        self.G = np.array([
            sum(w.mass_t * w.x for w in weights) / self.W,
            sum(w.mass_t * w.y for w in weights) / self.W,
            sum(w.mass_t * w.z for w in weights) / self.W,
        ])
        self.v_target = self.W / params.rho_w
        self.elements = list(elements)
        self.weights = list(weights)
        self.openings = list(openings)
        buoy = [e for e in elements if e.buoyant and not (damaged and e.flooded)]
        expo = [e for e in elements if e.exposed]
        self.ea = build_elem_arrays(buoy)
        if self.ea.n == 0:
            raise ValueError("No buoyant elements - nothing to float on.")
        self.sl = build_slices(expo, params.slice_m)
        self.open_pts = (np.array([[o.x, o.y, o.z] for o in openings], float)
                         if len(openings) else None)
        self.open_names = [o.name for o in openings]
        self.tow_pt = np.asarray(params.tow_point, float)
        self._set_lines(params)
        self.reserve_buoyancy = (self.ea.vol_total * params.rho_w - self.W) / self.W

    def _set_lines(self, p: Params) -> None:
        lines = list(p.lines) if p.lines else [Line(
            "Tow", tuple(p.tow_point), p.tow_heading_deg, p.tow_elevation_deg,
            "tow" if p.tow_mode == "auto" else "fixed", p.tow_manual_kn, 1.0)]
        self.lines = lines
        n = len(lines)
        self.line_pts = np.array([l.point for l in lines], float).reshape(n, 3)
        tau = np.radians([l.heading_deg for l in lines])
        eps = np.radians([l.elevation_deg for l in lines])
        self.line_hdir = np.stack([np.cos(tau), np.sin(tau), np.zeros(n)], axis=1)
        self.line_cos, self.line_sin = np.cos(eps), np.sin(eps)
        self.line_tension = np.array([l.tension_kn for l in lines], float)
        self.line_share = np.array([l.share for l in lines], float)
        self.tow_idx = np.array([i for i, l in enumerate(lines) if l.mode == "tow"], int)
        self.has_vertical = bool(np.any(np.abs(self.line_sin) > 1e-12))

    # -- hydrostatics at a fixed orientation ---------------------------------
    def _hydro(self, rot: np.ndarray, guess: Optional[float], v_target: Optional[float] = None):
        u = rot.T @ EZ
        zw = solve_waterline(self.ea, u, self.v_target if v_target is None else v_target, guess)
        if zw is None:
            return None
        vol, mom = cut(self.ea, u, zw)
        b = mom / vol
        h0 = self.ea.p1 @ u
        sa = self.ea.a @ u
        m = np.sqrt(np.maximum(1.0 - sa * sa, 0.0))
        h_low = float((np.minimum(h0, h0 + self.ea.length * sa) - self.ea.ro * m).min())
        return zw, b, u, h_low

    # -- external loads ------------------------------------------------------
    def _loads(self, rot, h_ax, p_ax, u, zw, h_low, b=None) -> dict:
        p = self.p
        w_s = -p_ax
        sl = self.sl
        hd = self.line_hdir
        ti = self.tow_idx
        # direction of travel for the calm-water drag: mean heading of the tow legs
        if len(ti):
            sh = self.line_share[ti]
            mean = (sh[:, None] * hd[ti]).sum(axis=0)
            nm = float(np.linalg.norm(mean))
            t_s = mean / nm if nm > 1e-9 else hd[ti[0]]
        else:
            t_s = hd[0]
        out = dict(Fw=0.0, T=0.0, Dw=0.0, Fz=0.0, M_H=0.0, M_P=0.0, z_lr=float("nan"))
        if sl.n:
            h = sl.c @ u
            sa = sl.a @ u
            m = np.sqrt(np.maximum(1.0 - sa * sa, 0.0))
            ev = sl.r_o * m + 0.5 * sl.t * np.abs(sa)
            f_sub = np.clip((zw - h + ev) / (2.0 * np.maximum(ev, 1e-9)), 0.0, 1.0)
            w_b = rot.T @ w_s
            t_b = rot.T @ t_s
            dia = 2.0 * sl.r_o
            sin_w = np.linalg.norm(np.cross(sl.a, w_b), axis=1)
            sin_t = np.linalg.norm(np.cross(sl.a, t_b), axis=1)
            v_ref = p.wind_speed_kn * KN_TO_MS
            z_abv = np.maximum(h - zw, 1.0)
            q = 0.5 * p.rho_a * (v_ref * (z_abv / p.wind_zref) ** p.wind_alpha) ** 2
            f_k = q * p.wind_cs * p.shielding * dia * sl.t * sin_w * (1.0 - f_sub) / 1000.0
            fw = float(f_k.sum())
            u_t = p.tow_speed_kn * KN_TO_MS
            dw = float((0.5 * p.rho_w * 1000.0 * u_t**2 * p.cd_water * p.shielding
                        * dia * sl.t * sin_t * f_sub).sum()) / 1000.0
        else:
            h = np.zeros(0)
            f_k = np.zeros(0)
            f_sub = np.zeros(0)
            fw = dw = 0.0

        # line tensions: fixed ones as given; tow legs share the drag-based pull
        tens = self.line_tension.copy()
        if len(ti):
            t_total = p.tow_factor * dw
            den = float((sh * self.line_cos[ti] * (hd[ti] @ t_s)).sum())
            tens[ti] = (t_total / den) * sh if den > 1e-9 else 0.0
        fh_vec = (tens * self.line_cos)[:, None] * hd           # horizontal part of each line force
        fz = tens * self.line_sin                               # vertical part [kN]
        f_vec = fw * w_s + fh_vec.sum(axis=0)

        # reaction level: area-weighted centroid of submerged lateral area
        fmag = float(np.linalg.norm(f_vec))
        n_s = f_vec / fmag if fmag > 1e-9 else w_s
        z_lr = zw - 0.5 * (zw - h_low)
        if p.lr_mode == "area" and sl.n:
            n_b = rot.T @ n_s
            sin_n = np.linalg.norm(np.cross(sl.a, n_b), axis=1)
            om = 2.0 * sl.r_o * sl.t * sin_n * f_sub
            if om.sum() > 1e-9:
                z_lr = float((om * h).sum() / om.sum())

        h_i = self.line_pts @ u
        t_p, t_h = fh_vec @ p_ax, fh_vec @ h_ax
        m_h = float((f_k * (h - z_lr)).sum()) - float((t_p * (h_i - z_lr)).sum())
        m_p = float((t_h * (h_i - z_lr)).sum())
        if self.has_vertical and b is not None:
            arm = (self.line_pts - b) @ rot.T                    # R (p_i - B) for each line
            m_h += float((fz * (arm @ p_ax)).sum())
            m_p -= float((fz * (arm @ h_ax)).sum())
        out.update(Fw=fw, T=float(np.linalg.norm(fh_vec.sum(axis=0))), Dw=dw, Fz=float(fz.sum()),
                   M_H=m_h, M_P=m_p, z_lr=z_lr)
        return out

    # -- state at (psi, phi, theta) -----------------------------------------
    def _eval(self, psi: float, phi: float, theta: float, loads: bool,
              guess: Optional[float]) -> Optional[dict]:
        rot, h_ax, p_ax = frame(psi, phi, theta)
        v_t, fz_tot, ld = None, 0.0, None
        for _ in range(5 if (loads and self.has_vertical) else 1):
            hy = self._hydro(rot, guess, v_t)
            if hy is None:
                return None
            zw, b, u, h_low = hy
            if not loads:
                break
            ld = self._loads(rot, h_ax, p_ax, u, zw, h_low, b)
            if not self.has_vertical or abs(ld["Fz"] - fz_tot) < 1e-7 * self.W * G_ACC:
                break
            fz_tot = ld["Fz"]                       # an upward line pull reduces the buoyancy needed
            v_t = (self.W - fz_tot / G_ACC) / self.p.rho_w
            if v_t <= 0.0:
                return None
        d = rot @ (b - self.G)
        d_h, d_p = float(d @ h_ax), float(d @ p_ax)
        st = dict(zw=zw, d_H=d_h, d_P=d_p, draft=zw - h_low,
                  gz=-d_p - self.p.fsc_m * math.sin(phi),
                  Fw=0.0, T=0.0, Dw=0.0, Fz=0.0, M_H=0.0, M_P=0.0, z_lr=float("nan"),
                  h_open=float("nan"))
        if loads:
            st.update(ld)
        if self.open_pts is not None:
            st["h_open"] = float((self.open_pts @ u - zw).min())
        return st

    def solve_trim(self, psi: float, phi: float, loads: bool = True,
                   theta0: float = 0.0, guess: Optional[float] = None) -> Optional[dict]:
        """Free-to-trim equilibrium at heel phi [rad].  Returns the state dict
        with extra keys theta [rad], converged, trim_stable; None if the body sinks."""
        wg = self.W * G_ACC
        lim = math.radians(self.p.trim_limit_deg)
        hold = [guess]
        tol = 1e-4

        def resid(th: float):
            st = self._eval(psi, phi, th, loads, hold[0])
            if st is None:
                return None, None
            hold[0] = st["zw"]
            return st["d_H"] - st["M_P"] / wg, st

        t0 = float(np.clip(theta0, -lim, lim))
        r0, s0 = resid(t0)
        if r0 is None:
            return None
        best = (abs(r0), t0, s0)
        slope = float("nan")
        converged = abs(r0) < tol
        if not converged:
            t1 = float(np.clip(t0 + math.radians(0.5), -lim, lim))
            r1, s1 = resid(t1)
            if r1 is None:
                return None
            if abs(r1) < best[0]:
                best = (abs(r1), t1, s1)
            for _ in range(25):
                if abs(r1) < tol:
                    converged = True
                    break
                if r1 == r0 or t1 == t0:
                    break
                t2 = float(np.clip(t1 - r1 * (t1 - t0) / (r1 - r0), -lim, lim))
                if t2 == t1:
                    break
                slope = (r1 - r0) / (t1 - t0)
                t0, r0 = t1, r1
                t1 = t2
                r1, s1 = resid(t1)
                if r1 is None:
                    return None
                if abs(r1) < best[0]:
                    best = (abs(r1), t1, s1)
            if abs(r1) < tol:
                converged = True
                best = (abs(r1), t1, s1)
        if not converged:
            # bracketed fallback over the full trim range
            grid = np.linspace(-lim, lim, 41)
            vals = []
            for th in grid:
                r, s = resid(float(th))
                if r is None:
                    return None
                vals.append((r, s))
                if abs(r) < best[0]:
                    best = (abs(r), float(th), s)
            for i in range(len(grid) - 1):
                if vals[i][0] * vals[i + 1][0] < 0:
                    lo, hi = float(grid[i]), float(grid[i + 1])
                    flo, fhi = vals[i][0], vals[i + 1][0]
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        fm, sm = resid(mid)
                        if fm is None:
                            return None
                        if abs(fm) < tol:
                            best = (abs(fm), mid, sm)
                            converged = True
                            break
                        if fm * flo > 0:
                            lo, flo = mid, fm
                        else:
                            hi, fhi = mid, fm
                    slope = (fhi - flo) / (hi - lo)
                    if converged:
                        break
        _, th_best, st = best
        st = dict(st)
        st.update(theta=th_best, converged=bool(converged),
                  trim_stable=bool(slope > 0) if not math.isnan(slope) else True)
        return st

    # -- reporting helpers ---------------------------------------------------
    def upright(self) -> dict:
        """Level, no-load equilibrium: draft, trim, reserve buoyancy."""
        st = self.solve_trim(0.0, 0.0, loads=False)
        if st is None:
            raise ValueError("Insufficient buoyancy to float the structure.")
        return dict(zw=st["zw"], draft=st["draft"], trim_deg=math.degrees(st["theta"]),
                    gz0=st["gz"], h_open=st["h_open"],
                    reserve_buoyancy_pct=100.0 * self.reserve_buoyancy)

    def gm(self, beta_deg: float, dphi_deg: float = 1.0) -> float:
        """Small-angle GM about the heel axis for wind heading beta:
        GM = [GZ(+d) - GZ(-d)] / (2 sin d), free to trim, no loads."""
        psi = math.radians(beta_deg + 90.0)
        sp = self.solve_trim(psi, math.radians(dphi_deg), loads=False)
        sm = self.solve_trim(psi, -math.radians(dphi_deg), loads=False)
        if sp is None or sm is None:
            return float("nan")
        return (sp["gz"] - sm["gz"]) / (2.0 * math.sin(math.radians(dphi_deg)))

    def sweep(self, beta_deg: float, phi_grid_deg: Sequence[float]) -> dict:
        """Heel sweep for wind travelling toward body heading beta (deg from +x).
        Heel axis is perpendicular to the wind; positive heel = wind direction."""
        psi = math.radians(beta_deg + 90.0)
        grid = np.asarray(phi_grid_deg, float)
        n = len(grid)
        keys = ["gz", "hl", "theta_deg", "zw", "h_open", "T", "Fz", "Fw", "M_H", "M_P", "z_lr", "draft"]
        out = {k: np.full(n, np.nan) for k in keys}
        conv = np.zeros(n, bool)
        stable = np.ones(n, bool)
        wg = self.W * G_ACC
        idx0 = int(np.argmin(np.abs(grid)))
        start = {}
        for indices in (range(idx0, n), range(idx0 - 1, -1, -1)):
            th_g = start.get("th", 0.0)
            zw_g = start.get("zw")
            for i in indices:
                st = self.solve_trim(psi, math.radians(grid[i]), True, th_g, zw_g)
                if st is None:
                    break
                th_g, zw_g = st["theta"], st["zw"]
                if i == idx0:
                    start = dict(th=th_g, zw=zw_g)
                out["gz"][i] = st["gz"]
                out["hl"][i] = st["M_H"] / wg
                out["theta_deg"][i] = math.degrees(st["theta"])
                for k in ("zw", "h_open", "T", "Fz", "Fw", "M_H", "M_P", "z_lr", "draft"):
                    out[k][i] = st[k]
                conv[i] = st["converged"]
                stable[i] = st["trim_stable"]
        out.update(phi_deg=grid, converged=conv, trim_stable=stable)
        return out


    def free_tilt(self, span_deg: float = 60.0, step_deg: float = 3.0) -> Optional[dict]:
        """Free-floating attitude with no wind or tow (static list + trim).

        The heel axis is chosen perpendicular to the horizontal B-G offset of the
        level state, so the tilt is carried almost entirely by the heel angle and
        GZ(phi) = 0 (free trim) is solved for the first stable up-crossing.
        Returns the tilt of the body z-axis from vertical [deg] with the heel and
        trim angles; None if there is no stable floating attitude within +/-span_deg
        (the structure lists over or capsizes) or it cannot float."""
        hy = self._hydro(np.eye(3), None)
        if hy is None:
            return None
        _, b, _, _ = hy
        off = (b - self.G)[:2]
        psi = (math.atan2(-off[0], off[1]) if float(np.hypot(*off)) > 1e-6 else math.radians(90.0))

        def g(phi_deg: float):
            return self.solve_trim(psi, math.radians(phi_deg), loads=False)

        grid = np.arange(-span_deg, span_deg + 1e-9, step_deg)
        prev = None
        root = None
        for ph in grid:
            st = g(float(ph))
            if st is None or not st["converged"]:
                prev = None
                continue
            if abs(st["gz"]) < 1e-9:                      # a root on the grid: keep it only if stable
                nxt = g(float(ph) + step_deg)
                if nxt is not None and nxt["converged"] and nxt["gz"] > 0:
                    root = (float(ph), st)
                    break
                prev = None
                continue
            if prev is not None and prev[1]["gz"] < 0 < st["gz"]:          # stable up-crossing
                lo, hi = prev[0], float(ph)
                best = st
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    sm = g(mid)
                    if sm is None or not sm["converged"]:
                        break
                    best = sm
                    if abs(sm["gz"]) < 1e-7:
                        lo = hi = mid
                        break
                    if sm["gz"] < 0:
                        lo = mid
                    else:
                        hi = mid
                root = (0.5 * (lo + hi), best)
                break
            prev = (float(ph), st)
        if root is None:
            return None
        phi0, st = root
        rot = frame(psi, math.radians(phi0), st["theta"])[0]
        tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))
        return dict(tilt_deg=tilt, phi_deg=phi0, trim_deg=math.degrees(st["theta"]))

    def attitude_state(self, beta_deg: float, phi_deg: float, theta_deg: float) -> Optional[dict]:
        """Floating state at a given attitude (for drawing): rotation, waterline
        height, and B and G in the space frame.  None if it cannot float there."""
        rot = frame(math.radians(beta_deg + 90.0), math.radians(phi_deg), math.radians(theta_deg))[0]
        hy = self._hydro(rot, None)
        if hy is None:
            return None
        zw, b, u, h_low = hy
        return dict(rot=rot, zw=zw, B=rot @ b, G=rot @ self.G, draft=zw - h_low)

    def hydrostatics(self) -> dict:
        """Technical data for the level, free-to-trim, no-load floating condition."""
        st = self.solve_trim(0.0, 0.0, loads=False)
        if st is None:
            raise ValueError("Insufficient buoyancy to float the structure.")
        th = st["theta"]
        rot = frame(0.0, 0.0, th)[0]
        zw, b, u, h_low = self._hydro(rot, st["zw"])
        vol, _ = cut(self.ea, u, zw)
        dz = 0.05
        aw = (cut(self.ea, u, zw + dz)[0] - cut(self.ea, u, zw - dz)[0]) / (2.0 * dz)
        g_s, b_s = rot @ self.G, rot @ b
        kg, kb = float(g_s[2] - h_low), float(b_s[2] - h_low)
        rho = self.p.rho_w
        gm_x, gm_y = self.gm(0.0), self.gm(90.0)   # wind toward +x (heel about y), +y (heel about x)
        rows = []
        for e in self.elements:
            a = np.asarray(e.p2, float) - np.asarray(e.p1, float)
            length = float(np.linalg.norm(a))
            v_tot = math.pi * ((e.d_out / 2) ** 2 - (e.d_in / 2) ** 2) * length
            if e.buoyant and not (self.damaged and e.flooded):
                vs = float(cut(build_elem_arrays([e]), u, zw)[0])
                status = "buoyant"
            elif e.buoyant:
                vs, status = 0.0, "flooded - no buoyancy"
            else:
                vs, status = 0.0, "not buoyant"
            rows.append(dict(name=e.name, kind=member_kind(e), length_m=length, d_out=e.d_out, d_in=e.d_in,
                             volume_m3=v_tot, sub_vol_m3=vs, sub_pct=(100.0 * vs / v_tot if v_tot > 0 else 0.0),
                             buoyancy_t=vs * rho, status=status))
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for e in self.elements:                       # exact extent of each cylinder
            p1, p2 = np.asarray(e.p1, float), np.asarray(e.p2, float)
            ax = (p2 - p1) / max(float(np.linalg.norm(p2 - p1)), 1e-12)
            ext = (e.d_out / 2) * np.sqrt(np.maximum(1.0 - ax * ax, 0.0))
            lo = np.minimum(lo, np.minimum(p1, p2) - ext)
            hi = np.maximum(hi, np.maximum(p1, p2) + ext)
        opens = []
        if self.open_pts is not None:
            for n_, hgt in zip(self.open_names, self.open_pts @ u - zw):
                opens.append(dict(name=n_, above_water_m=float(hgt)))
        return dict(
            W=self.W, volume=float(vol), rho_w=rho, G_body=self.G.tolist(), B_body=b.tolist(),
            zw=float(zw), draft=float(zw - h_low), trim_deg=math.degrees(th),
            kg_base=float(g_s[2] - lo[2]), kb_base=float(b_s[2] - lo[2]),
            kg=kg, kb=kb, gm_x=gm_x, gm_y=gm_y, km_x=kg + gm_x, km_y=kg + gm_y,
            bm_x=kg + gm_x - kb, bm_y=kg + gm_y - kb,
            waterplane_area=float(aw), tpc=float(aw * rho / 100.0),
            reserve_buoyancy_pct=100.0 * self.reserve_buoyancy,
            buoyant_volume=float(self.ea.vol_total), fsc_m=self.p.fsc_m, damaged=self.damaged,
            bbox_lo=lo.tolist(), bbox_hi=hi.tolist(), members=rows, openings=opens,
            n_members=len(self.elements), n_weights=len(self.weights))


# ----------------------------------------------------------------------------
# Installation / set-down float check: seabed clearance, tank emergence, ballast headroom
# ----------------------------------------------------------------------------
def float_check(model: "JacketModel", crit: Criteria) -> dict:
    """Level, no-wind, no-tow floating condition checked against the water depth.

    Reports the waterline, the clearance from the lowest point of the structure to
    the seabed, how much of each buoyancy tank is out of the water, the freeboard,
    and the ballast (+) or weight to shed (-) before a limit is reached.  The
    structure is taken as level (trim is reported; the clearance uses the level
    waterline, so a trim above a degree or so needs a full check)."""
    h = model.hydrostatics()
    p = model.p
    rho = p.rho_w
    zw = h["zw"]
    z_base, z_top = h["bbox_lo"][2], h["bbox_hi"][2]
    draft_total = zw - z_base
    tank_names = {e.name for e in model.elements if e.buoyant and e.tank}
    tanks = [r for r in h["members"] if r["status"] == "buoyant" and r["name"] in tank_names]
    for r in tanks:
        r["emerged_m"] = r["length_m"] * (1.0 - r["sub_pct"] / 100.0)
    emerged_min = min((r["emerged_m"] for r in tanks), default=float("nan"))
    tank_top = {e.name: max(e.p1[2], e.p2[2]) for e in model.elements if e.buoyant and e.tank}
    depth = p.water_depth_m + p.tide_m if p.water_depth_m > 0 else None
    clearance = None if depth is None else depth - draft_total
    opens = [o["above_water_m"] for o in h["openings"]]
    u0 = np.array([0.0, 0.0, 1.0])

    def mass_at(zw_x: float) -> float:
        return rho * cut(model.ea, u0, zw_x)[0]

    limits = {}
    if depth is not None and crit.clear_min_m > 0.0:
        limits["clearance"] = z_base + depth - crit.clear_min_m
    if crit.emerged_min_m > 0.0 and tank_top:
        limits["tank emergence"] = min(tank_top.values()) - crit.emerged_min_m
    gov, headroom = None, None
    if limits:
        gov = min(limits, key=limits.get)
        headroom = mass_at(limits[gov]) - model.W
    gm = min(h["gm_x"], h["gm_y"])
    tilt = model.free_tilt()
    out = dict(
        zw=zw, z_base=z_base, z_top=z_top, draft_total=draft_total, trim_deg=h["trim_deg"],
        water_depth=p.water_depth_m, tide=p.tide_m, depth_total=depth, clearance=clearance,
        emerged_min=emerged_min, tanks=tanks, freeboard_top=z_top - zw,
        opening_min=(min(opens) if opens else None), gm_min=gm, gm_x=h["gm_x"], gm_y=h["gm_y"],
        free_tilt_deg=(None if tilt is None else tilt["tilt_deg"]), stable=tilt is not None,
        displacement_t=model.W, reserve_buoyancy_pct=h["reserve_buoyancy_pct"],
        governing_limit=gov, ballast_headroom_t=headroom,
        pass_clearance=(None if (clearance is None or crit.clear_min_m <= 0) else bool(clearance >= crit.clear_min_m)),
        pass_emerged=(None if crit.emerged_min_m <= 0 else bool(emerged_min >= crit.emerged_min_m)),
        pass_gm=bool(gm >= crit.gm_min))
    flags = [out[k] for k in ("pass_clearance", "pass_emerged", "pass_gm") if out[k] is not None]
    out["passed"] = bool(all(flags) and out["stable"])
    return out


# ----------------------------------------------------------------------------
# Criteria extraction
# ----------------------------------------------------------------------------
def _interp_cross(x0: float, x1: float, y0: float, y1: float) -> float:
    return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)


def analyse_curve(sw: dict, gm: float, crit: Criteria) -> dict:
    """Equilibrium angle, intercepts, downflooding, area ratio and pass flags.

        f(phi)      = GZ - HL
        theta_s     = first up-crossing of f  (stable static heel)
        theta_2     = next down-crossing of f (second intercept)
        theta_lim   = min(theta_2, theta_df, cap)
        ratio       = int_0^theta_lim GZ dphi / int_0^theta_lim HL dphi
        residual    = int_theta_s^theta_lim (GZ - HL) dphi            [m rad]
    """
    phi, gz, hl = sw["phi_deg"], sw["gz"], sw["hl"]
    ok = ~(np.isnan(gz) | np.isnan(hl))
    res = dict(gm=gm, theta_s=None, theta_2=None, theta_df=None, theta_lim=None,
               ratio=None, residual=None, gz_max=None, phi_gz_max=None,
               trim_at_s=None, tow_kn=None, wind_kn=None, note="", pass_gm=False,
               pass_heel=False, pass_ratio=False, pass_df=True, passed=False,
               sunk_beyond=None, trim_unstable_in_range=0, unconverged_in_range=0)
    if ok.sum() < 3:
        res["note"] = "Sweep failed (insufficient buoyancy at small heel)."
        return res
    if not ok.all():
        first_bad = phi[~ok][phi[~ok] > 0]
        if len(first_bad):
            res["sunk_beyond"] = float(first_bad.min())
    p, g, h = phi[ok], gz[ok], hl[ok]
    th = sw["theta_deg"][ok]
    f = g - h
    ups, downs = [], []
    for i in range(len(f) - 1):
        if f[i] <= 0 < f[i + 1]:
            ups.append(_interp_cross(p[i], p[i + 1], f[i], f[i + 1]))
        elif f[i] > 0 >= f[i + 1]:
            downs.append(_interp_cross(p[i], p[i + 1], f[i], f[i + 1]))
    if f[0] > 0:
        theta_s = float(p[0])
        res["note"] = "Equilibrium at or below the start of the sweep."
    elif ups:
        theta_s = float(ups[0])
    else:
        theta_s = None
        res["note"] = "No static equilibrium found within the sweep (heeling exceeds righting)."
    theta_2 = None
    if theta_s is not None:
        later = [d for d in downs if d > theta_s]
        theta_2 = float(later[0]) if later else None

    theta_df = None
    ho = sw["h_open"][ok]
    if not np.all(np.isnan(ho)):
        for i in range(len(p)):
            if p[i] >= 0 and ho[i] <= 0:
                if i > 0 and p[i - 1] >= 0 and ho[i - 1] > 0:
                    theta_df = float(_interp_cross(p[i - 1], p[i], ho[i - 1], ho[i]))
                else:
                    theta_df = float(p[i])
                break

    cands = [crit.cap_deg] + [x for x in (theta_2, theta_df) if x is not None]
    theta_lim = float(min(cands))
    if res["sunk_beyond"] is not None:
        theta_lim = min(theta_lim, res["sunk_beyond"])
    in_rng = (p >= 0) & (p <= theta_lim)
    res["trim_unstable_in_range"] = int((~sw["trim_stable"][ok] & in_rng).sum())
    res["unconverged_in_range"] = int((~sw["converged"][ok] & in_rng).sum())
    ratio = None
    if theta_lim > 0:
        pf = np.linspace(0.0, theta_lim, 600)
        gzf = np.interp(pf, p, g)
        hlf = np.interp(pf, p, h)
        a_gz = float(_trapz(gzf, np.radians(pf)))
        a_hl = float(_trapz(hlf, np.radians(pf)))
        ratio = 999.0 if a_hl <= 1e-9 else a_gz / a_hl
    residual = None
    if theta_s is not None and theta_lim > theta_s:
        pr = np.linspace(theta_s, theta_lim, 400)
        residual = float(_trapz(np.interp(pr, p, g) - np.interp(pr, p, h), np.radians(pr)))
    pos = p >= 0
    if pos.any():
        j = int(np.argmax(g[pos]))
        res["gz_max"], res["phi_gz_max"] = float(g[pos][j]), float(p[pos][j])
    if theta_s is not None:
        res["trim_at_s"] = float(np.interp(theta_s, p, th))
        res["tow_kn"] = float(np.interp(theta_s, p, sw["T"][ok]))
        res["wind_kn"] = float(np.interp(theta_s, p, sw["Fw"][ok]))
    res.update(theta_s=theta_s, theta_2=theta_2, theta_df=theta_df, theta_lim=theta_lim,
               ratio=ratio, residual=residual)
    res["pass_gm"] = bool(not math.isnan(gm) and gm >= crit.gm_min)
    res["pass_heel"] = bool(theta_s is not None and theta_s <= crit.heel_max_deg)
    res["pass_ratio"] = bool(theta_s is not None and ratio is not None and ratio >= crit.ratio_min)
    res["pass_df"] = bool(crit.df_min_deg <= 0 or theta_df is None or theta_df >= crit.df_min_deg)
    res["passed"] = bool(res["pass_gm"] and res["pass_heel"] and res["pass_ratio"] and res["pass_df"])
    return res


def run_study(model: JacketModel, betas_deg: Sequence[float], phi_grid_deg: Sequence[float],
              crit: Criteria, progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Run all wind headings; returns per-heading sweeps/analyses plus a summary."""
    up = model.upright()
    heads = []
    for k, b in enumerate(betas_deg):
        sw = model.sweep(b, phi_grid_deg)
        gm = model.gm(b)
        an = analyse_curve(sw, gm, crit)
        heads.append(dict(beta=float(b), sweep=sw, analysis=an))
        if progress:
            progress((k + 1) / len(betas_deg), f"Heading {b:.0f}° done")
    an = [h["analysis"] for h in heads]
    ratios = [a["ratio"] for a in an if a["ratio"] is not None]
    heels = [a["theta_s"] for a in an]
    gov = min(range(len(an)), key=lambda i: (an[i]["ratio"] if an[i]["ratio"] is not None else -1.0))
    dfs = [a["theta_df"] for a in an if a["theta_df"] is not None]
    summary = dict(
        gm_min=float(np.nanmin([a["gm"] for a in an])) if an else float("nan"),
        heel_max=None if any(x is None for x in heels) else float(max(heels)),
        no_equilibrium=any(x is None for x in heels),
        ratio_min=float(min(ratios)) if ratios and len(ratios) == len(an) else None,
        df_min=float(min(dfs)) if dfs else None,
        governing_beta=heads[gov]["beta"] if heads else None,
        tow_kn=max((a["tow_kn"] for a in an if a["tow_kn"] is not None), default=None),
        passed=all(a["passed"] for a in an),
        unconverged=int(sum(a["unconverged_in_range"] for a in an)),
        trim_unstable=int(sum(a["trim_unstable_in_range"] for a in an)),
    )
    return dict(upright=up, heads=heads, summary=summary, W=model.W, G=model.G.tolist(),
                damaged=model.damaged)


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
def validate_inputs(elements: Sequence[Element], weights: Sequence[WeightItem],
                    openings: Sequence[Opening], params: Params) -> tuple[list, list]:
    errors: list[str] = []
    warnings: list[str] = []
    if not weights or sum(w.mass_t for w in weights) <= 0:
        errors.append("Weights table is empty or sums to zero.")
    if any(w.mass_t < 0 for w in weights):
        warnings.append("A weight item has negative mass (allowed for deductions).")
    for e in elements:
        if e.d_out <= 0:
            errors.append(f"{e.name}: outer diameter must be > 0.")
        if e.d_in >= e.d_out > 0:
            errors.append(f"{e.name}: inner diameter must be < outer diameter.")
        if float(np.linalg.norm(np.asarray(e.p2, float) - np.asarray(e.p1, float))) < 1e-9:
            errors.append(f"{e.name}: start and end points coincide.")
    buoy = build_elem_arrays([e for e in elements if e.buoyant])
    if buoy.n == 0:
        errors.append("No buoyant elements are defined.")
    elif weights and sum(w.mass_t for w in weights) > 0:
        w_tot = sum(w.mass_t for w in weights)
        if buoy.vol_total * params.rho_w < w_tot:
            errors.append(f"Insufficient buoyancy: {buoy.vol_total * params.rho_w:,.0f} t available "
                          f"vs {w_tot:,.0f} t weight.")
        elif buoy.vol_total * params.rho_w < 1.15 * w_tot:
            warnings.append("Reserve buoyancy is under 15%.")
    if not any(e.exposed for e in elements):
        warnings.append("No exposed members: wind force and drag will be zero.")
    if params.lines:
        for ln in params.lines:
            if ln.mode not in ("tow", "fixed"):
                errors.append(f"Line {ln.name}: type must be 'tow' or 'fixed'.")
            if not (-89.0 <= ln.elevation_deg <= 89.0):
                errors.append(f"Line {ln.name}: elevation must be between -89 and 89 degrees.")
            if ln.mode == "fixed" and ln.tension_kn < 0:
                errors.append(f"Line {ln.name}: tension cannot be negative.")
            if ln.mode == "tow" and ln.share <= 0:
                errors.append(f"Line {ln.name}: share must be > 0.")
        if all(l.mode == "fixed" and l.tension_kn <= 0 for l in params.lines):
            warnings.append("No line carries a load - there is no tow or pull-in moment.")
    elif params.tow_mode == "manual" and params.tow_manual_kn <= 0:
        warnings.append("Manual tow pull is zero - tow-line moment ignored.")
    warnings.append("Overlapping buoyant elements are double counted - check that tanks and "
                    "legs do not overlap unless the tank is an annulus (inner diameter set).")
    return errors, warnings


# ----------------------------------------------------------------------------
# Illustrative example (NOT project data)
# ----------------------------------------------------------------------------
def example_inputs() -> dict:
    """Generic four-leg jacket on four corner tanks.  Placeholder numbers only."""
    s = 20.0
    leg_h, tank_h = 75.0, 28.0
    corners = [(+s, +s), (+s, -s), (-s, -s), (-s, +s)]
    names = ["NE", "SE", "SW", "NW"]
    rows = []
    for (x, y), n in zip(corners, names):
        rows.append(dict(name=f"Leg {n}", x1=x, y1=y, z1=0.0, x2=x, y2=y, z2=leg_h,
                         d_out=2.4, d_in=0.0, buoyant=False, exposed=True, flooded=False))
    for (x, y), n in zip(corners, names):
        rows.append(dict(name=f"Tank {n}", x1=x, y1=y, z1=0.0, x2=x, y2=y, z2=tank_h,
                         d_out=9.0, d_in=2.4, buoyant=True, exposed=True, flooded=False))
    levels = [20.0, 45.0, 70.0]
    for z in levels:
        for i in range(4):
            (xa, ya), (xb, yb) = corners[i], corners[(i + 1) % 4]
            rows.append(dict(name=f"Ring z{z:.0f} {names[i]}", x1=xa, y1=ya, z1=z, x2=xb, y2=yb, z2=z,
                             d_out=1.4, d_in=0.0, buoyant=False, exposed=True, flooded=False))
    for z0, z1 in zip(levels[:-1], levels[1:]):
        for i in range(4):
            (xa, ya), (xb, yb) = corners[i], corners[(i + 1) % 4]
            rows.append(dict(name=f"Brace {names[i]} a z{z0:.0f}", x1=xa, y1=ya, z1=z0, x2=xb, y2=yb, z2=z1,
                             d_out=1.1, d_in=0.0, buoyant=False, exposed=True, flooded=False))
            rows.append(dict(name=f"Brace {names[i]} b z{z0:.0f}", x1=xb, y1=yb, z1=z0, x2=xa, y2=ya, z2=z1,
                             d_out=1.1, d_in=0.0, buoyant=False, exposed=True, flooded=False))
    weights = [
        dict(item="Jacket steel", mass_t=3200.0, x=0.0, y=0.0, z=30.0),
        dict(item="Buoyancy tank steel", mass_t=450.0, x=0.0, y=0.0, z=20.0),
        dict(item="Appurtenances / topside", mass_t=200.0, x=0.0, y=0.0, z=55.0),
    ]
    openings = [dict(name=f"Tank vent {n}", x=x, y=y, z=30.0) for (x, y), n in zip(corners, names)]
    openings.append(dict(name="Access hatch", x=0.0, y=0.0, z=60.0))
    return dict(elements=rows, weights=weights, openings=openings,
                tow_point=(s, 0.0, 20.0))


def synthetic_inputs() -> dict:
    """OEC illustrative jacket (SYNTHETIC - not project data).

    Four main legs on a 7 m x 7 m square (x = +-3.5, y = +-3.5) run the full 105 m.
    Two outrigger legs, 7 m outside the main legs at y = +10.5, stop at 30 m, so the
    base is 7 m x 14 m with six legs.  Four 3 m x 30 m buoyancy tanks sit high on the
    main legs (z 60-90 m).  Legs are free-flooding and carry no buoyancy.  700 t,
    CoG about 42 m above the base, balanced about the centreline (a trim ballast item
    offsets the outrigger steel)."""
    xs, y_main, y_out = (-3.5, 3.5), (-3.5, 3.5), 10.5
    h_top, h_out, tank_z0, tank_len = 105.0, 30.0, 60.0, 30.0
    leg_d, tank_d, ring_d, brace_d = 1.5, 3.0, 0.9, 0.8
    rows = []

    def add(name, p1, p2, d, buoyant=False):
        rows.append(dict(name=name, x1=p1[0], y1=p1[1], z1=p1[2], x2=p2[0], y2=p2[1], z2=p2[2],
                         d_out=d, d_in=0.0, buoyant=buoyant, exposed=True, flooded=False))

    for x in xs:
        for y in y_main:
            add(f"Leg x{x:+.1f} y{y:+.1f}", (x, y, 0.0), (x, y, h_top), leg_d)
        add(f"Outrigger leg x{x:+.1f} y{y_out:+.1f}", (x, y_out, 0.0), (x, y_out, h_out), leg_d)
    for x in xs:
        for y in y_main:
            add(f"Tank x{x:+.1f} y{y:+.1f}", (x, y, tank_z0), (x, y, tank_z0 + tank_len), tank_d, True)
    y3 = (y_main[0], y_main[1], y_out)
    low = [0.0, 15.0, 30.0]
    up = [45.0, 65.0, 85.0, 105.0]
    for z in low:                                       # horizontal rings, six-leg section
        for x in xs:
            for ya, yb in zip(y3[:-1], y3[1:]):
                add(f"Ring z{z:.0f} x{x:+.1f} y{ya:+.1f}", (x, ya, z), (x, yb, z), ring_d)
        for y in y3:
            add(f"Ring z{z:.0f} y{y:+.1f}", (xs[0], y, z), (xs[1], y, z), ring_d)
    for z in [45.0] + up[1:]:                           # horizontal rings, four-leg section
        for x in xs:
            add(f"Ring z{z:.0f} x{x:+.1f}", (x, y_main[0], z), (x, y_main[1], z), ring_d)
        for y in y_main:
            add(f"Ring z{z:.0f} y{y:+.1f}", (xs[0], y, z), (xs[1], y, z), ring_d)
    for z0, z1 in zip(low[:-1], low[1:]):               # X bracing, six-leg section
        for x in xs:
            for ya, yb in zip(y3[:-1], y3[1:]):
                add(f"Brace x{x:+.1f} y{ya:+.1f} a z{z0:.0f}", (x, ya, z0), (x, yb, z1), brace_d)
                add(f"Brace x{x:+.1f} y{ya:+.1f} b z{z0:.0f}", (x, yb, z0), (x, ya, z1), brace_d)
        for y in (y_main[0], y_out):
            add(f"Brace y{y:+.1f} a z{z0:.0f}", (xs[0], y, z0), (xs[1], y, z1), brace_d)
            add(f"Brace y{y:+.1f} b z{z0:.0f}", (xs[1], y, z0), (xs[0], y, z1), brace_d)
    for x in xs:                                        # X bracing between the main legs, 30-45 m
        add(f"Brace x{x:+.1f} a z30", (x, y_main[0], 30.0), (x, y_main[1], 45.0), brace_d)
        add(f"Brace x{x:+.1f} b z30", (x, y_main[1], 30.0), (x, y_main[0], 45.0), brace_d)
        add(f"Outrigger strut x{x:+.1f}", (x, y_main[1], 45.0), (x, y_out, h_out), brace_d)   # sloping member
    for y in y_main:                                    # end-face X bracing, 30-45 m
        add(f"Brace y{y:+.1f} a z30", (xs[0], y, 30.0), (xs[1], y, 45.0), brace_d)
        add(f"Brace y{y:+.1f} b z30", (xs[1], y, 30.0), (xs[0], y, 45.0), brace_d)
    for z0, z1 in zip([45.0] + up[1:-1], up[1:]):       # X bracing, four-leg section
        for x in xs:
            add(f"Brace x{x:+.1f} a z{z0:.0f}", (x, y_main[0], z0), (x, y_main[1], z1), brace_d)
            add(f"Brace x{x:+.1f} b z{z0:.0f}", (x, y_main[1], z0), (x, y_main[0], z1), brace_d)
        for y in y_main:
            add(f"Brace y{y:+.1f} a z{z0:.0f}", (xs[0], y, z0), (xs[1], y, z1), brace_d)
            add(f"Brace y{y:+.1f} b z{z0:.0f}", (xs[1], y, z0), (xs[0], y, z1), brace_d)
    weights = [
        dict(item="Lower section steel (main legs, bracing)", mass_t=260.0, x=0.0, y=0.0, z=15.0),
        dict(item="Outrigger legs and bracing", mass_t=50.0, x=0.0, y=y_out, z=15.0),
        dict(item="Trim ballast (assumed, offsets outriggers)", mass_t=50.0, x=0.0, y=-y_out, z=15.0),
        dict(item="Upper section steel (4 legs, bracing)", mass_t=180.0, x=0.0, y=0.0, z=60.0),
        dict(item="Buoyancy tank steel", mass_t=70.0, x=0.0, y=0.0, z=75.0),
        dict(item="Appurtenances / topside", mass_t=90.0, x=0.0, y=0.0, z=90.0),
    ]
    openings = [dict(name=f"Tank vent x{x:+.1f} y{y:+.1f}", x=x, y=y, z=tank_z0 + tank_len)
                for x in xs for y in y_main]
    openings.append(dict(name="Access hatch", x=0.0, y=0.0, z=100.0))
    return dict(elements=rows, weights=weights, openings=openings, tow_point=(0.0, -3.5, 80.0),
                tow_heading_deg=270.0, water_depth_m=92.4, tide_m=0.0, clear_min_m=5.0, emerged_min_m=5.0)


def elements_from_rows(rows: Sequence[dict]) -> list[Element]:
    out = []
    for r in rows:
        try:
            out.append(Element(
                name=str(r["name"]),
                p1=(float(r["x1"]), float(r["y1"]), float(r["z1"])),
                p2=(float(r["x2"]), float(r["y2"]), float(r["z2"])),
                d_out=float(r["d_out"]), d_in=float(r.get("d_in") or 0.0),
                buoyant=bool(r.get("buoyant", True)), exposed=bool(r.get("exposed", True)),
                flooded=bool(r.get("flooded", False)), tank=bool(r.get("tank", True))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ----------------------------------------------------------------------------
# Self-check demonstration
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Vertical spar D=14 m, draft 45 m, KG=20 m: wall-sided analytic comparison.
    d_cyl, draft, kg = 14.0, 45.0, 20.0
    rho = 1.025
    w_t = rho * math.pi * d_cyl**2 / 4.0 * draft
    spar = [Element("spar", (0, 0, 0), (0, 0, 60), d_cyl, buoyant=True, exposed=True)]
    mdl = JacketModel(spar, [WeightItem("all", w_t, 0, 0, kg)], [], Params(wind_speed_kn=0.0, tow_speed_kn=0.0))
    bm = d_cyl**2 / (16.0 * draft)
    gm_an = draft / 2.0 + bm - kg
    gm_num = mdl.gm(0.0)
    print("| Quantity | Analytic | Numeric |\n|---|---|---|")
    print(f"| GM [m] | {gm_an:.4f} | {gm_num:.4f} |")
    for deg in (10.0, 20.0, 30.0):
        r = math.radians(deg)
        gz_an = math.sin(r) * (gm_an + 0.5 * bm * math.tan(r) ** 2)
        st = mdl.solve_trim(math.radians(90.0), r, loads=False)
        print(f"| GZ({deg:.0f} deg) [m] | {gz_an:.4f} | {st['gz']:.4f} |")
    assert abs(gm_num - gm_an) / gm_an < 0.005, "GM check failed"
    print("\nSelf-check passed.")
