"""Operability screen from percentile tables (screening level, not a weather-window study).

The Fugro report gives only mean, P10, P1 and maximum per month (P10 = value exceeded 10 % of
the time).  The exceedance probability of a limit is interpolated between those points, linear
in value and logarithmic in probability.  It says how often the limit is exceeded, not how long
a window lasts: there is no persistence data, so a tow window needs an hindcast time series.
"""
from __future__ import annotations

import math
from typing import Optional

KN = 0.514444   # m/s per knot


def wind_averaging_ratio(u10_ms: float, t_from_s: float = 60.0, t_to_s: float = 600.0) -> float:
    """U(t_from)/U(t_to) at 10 m, ISO 19901-1 gust model: U(T) = U1h (1 - 0.41 Iu ln(T/3600)),
    Iu = 0.06 (1 + 0.043 U1h).  u10_ms is the mean wind used to set the turbulence intensity."""
    if min(u10_ms, t_from_s, t_to_s) <= 0:
        raise ValueError("Wind speed and averaging times must be positive.")
    iu = 0.06 * (1.0 + 0.043 * u10_ms)
    f = lambda t: 1.0 - 0.41 * iu * math.log(t / 3600.0)
    return f(t_from_s) / f(t_to_s)


def exceedance_prob(limit: float, points: list) -> float:
    """P(X > limit) from (value, probability) points in ascending value order, e.g.
    [(mean, 0.35), (P10, 0.10), (P1, 0.01), (max, 0.001)]; linear in value, log in probability."""
    xs = [p[0] for p in points]
    if any(b < a for a, b in zip(xs, xs[1:])):
        raise ValueError("Points must be in ascending value order.")
    if limit < points[0][0]:
        return 0.5
    if limit >= points[-1][0]:
        return float(points[-1][1])
    for (x0, q0), (x1, q1) in zip(points, points[1:]):
        if x0 <= limit <= x1:
            if x1 == x0:
                return float(q1)
            w = (limit - x0) / (x1 - x0)
            return float(math.exp(math.log(q0) + w * (math.log(q1) - math.log(q0))))
    return float(points[-1][1])


def operability(month: dict, hs_limit_m: float, wind_limit_kn_1min: float,
                wind_key: str = "Wind 10-min mean [m/s]") -> dict:
    """Share of time below the Hs and wind limits for one monthly row of the case file.
    The 1-minute wind limit is converted to a 10-minute mean before comparing with the table."""
    u10 = wind_limit_kn_1min * KN / wind_averaging_ratio(wind_limit_kn_1min * KN)
    ph = exceedance_prob(hs_limit_m, [(month["Hs mean [m]"], 0.35), (month["Hs P10 [m]"], 0.10),
                                      (month["Hs P1 [m]"], 0.01), (month["Hs max [m]"], 0.001)])
    pw = exceedance_prob(u10, [(month[wind_key], 0.35), (month["Wind P1 [m/s]"], 0.01), (month["Wind max [m/s]"], 0.001)])
    return dict(month=month["Month"], wind_limit_10min_ms=u10, p_exceed_hs=ph, p_exceed_wind=pw,
                below_hs_pct=100 * (1 - ph), below_wind_pct=100 * (1 - pw),
                below_both_pct_low=100 * (1 - ph) * (1 - pw), below_both_pct_high=100 * (1 - max(ph, pw)))
