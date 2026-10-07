import pytest

import metocean as mo

PTS = [(0.5, 0.35), (1.0, 0.10), (1.5, 0.01), (2.5, 0.001)]


def test_exceedance_hits_the_table_points():
    assert mo.exceedance_prob(1.0, PTS) == pytest.approx(0.10)
    assert mo.exceedance_prob(1.5, PTS) == pytest.approx(0.01)
    assert mo.exceedance_prob(3.0, PTS) == pytest.approx(0.001)


def test_exceedance_log_interpolation_midpoint():
    assert mo.exceedance_prob(1.25, PTS) == pytest.approx((0.10 * 0.01) ** 0.5)   # geometric mean
    assert mo.exceedance_prob(0.1, PTS) == 0.5


def test_exceedance_rejects_unsorted():
    with pytest.raises(ValueError):
        mo.exceedance_prob(1.0, [(2.0, 0.1), (1.0, 0.01)])


def test_wind_ratio_hand_calc():
    # U = 8 m/s: Iu = 0.06 * (1 + 0.344) = 0.08064; ratio = (1 + 0.41*Iu*ln 60) / (1 + 0.41*Iu*ln 6)
    import math
    iu = 0.06 * (1 + 0.043 * 8)
    exp = (1 + 0.41 * iu * math.log(60)) / (1 + 0.41 * iu * math.log(6))
    assert mo.wind_averaging_ratio(8.0) == pytest.approx(exp)
    assert 1.05 < exp < 1.10


def test_operability_april_hs_is_p10():
    row = {"Month": "April", "Hs mean [m]": 0.6, "Hs P10 [m]": 1.0, "Hs P1 [m]": 1.5, "Hs max [m]": 2.5,
           "Wind 10-min mean [m/s]": 3.2, "Wind P1 [m/s]": 7.2, "Wind max [m/s]": 9.3}
    r = mo.operability(row, 1.0, 20.0)
    assert r["below_hs_pct"] == pytest.approx(90.0)
    assert r["wind_limit_10min_ms"] == pytest.approx(20 * 0.514444 / mo.wind_averaging_ratio(20 * 0.514444))
    assert r["below_wind_pct"] > 99.0 and r["below_both_pct_low"] <= r["below_both_pct_high"]
