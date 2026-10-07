import numpy as np
import pandas as pd
import pytest

import jacket_stability as js
from parsing import num_ok, parse_bool


def test_num_ok_rejects_everything_that_is_not_a_finite_number():
    for bad in (None, float("nan"), np.nan, pd.NA, float("inf"), -float("inf"), "abc", "", True, False, [1]):
        assert not num_ok(bad), bad
    for good in (0, 1.5, "2.5", np.float64(3.0), -4, np.int64(7)):
        assert num_ok(good), good


def test_parse_bool_is_strict():
    assert parse_bool(True) and not parse_bool(False) and parse_bool(np.bool_(True))
    assert parse_bool("True") and parse_bool("yes") and parse_bool(1) and parse_bool("1")
    assert not parse_bool("False") and not parse_bool("no") and not parse_bool(0) and not parse_bool("")
    assert parse_bool(None, True) and parse_bool(float("nan"), True) and not parse_bool(pd.NA, False)
    with pytest.raises(ValueError, match="not true or false"):
        parse_bool("maybe", what="Member 'L'")


def test_text_false_no_longer_turns_into_true_in_element_rows():
    row = dict(name="a", x1=0, y1=0, z1=0, x2=0, y2=0, z2=10, d_out=1.0, buoyant="False", exposed="no", flooded="true", tank="0")
    e = js.elements_from_rows([row])[0]
    assert (e.buoyant, e.exposed, e.flooded, e.tank) == (False, False, True, False)


def test_non_finite_and_bad_rows_are_dropped_not_kept():
    ok = dict(name="ok", x1=0, y1=0, z1=0, x2=0, y2=0, z2=10, d_out=1.0)
    rows = [ok, dict(ok, name="inf", z2=float("inf")), dict(ok, name="nan", d_out=float("nan")),
            dict(ok, name="bad", buoyant="maybe")]
    assert [e.name for e in js.elements_from_rows(rows)] == ["ok"]
