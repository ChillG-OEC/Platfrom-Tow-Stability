"""Strict parsing of values read from files and tables."""
from __future__ import annotations

import math

import numpy as np

_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off", ""}


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v)) if isinstance(v, (float, np.floating)) else (type(v).__name__ == "NAType")
    except (TypeError, ValueError):
        return False


def num_ok(v) -> bool:
    """True for a finite number (or numeric text). Rejects None, NaN, pd.NA, infinities, booleans and other text."""
    if _is_missing(v) or isinstance(v, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def parse_bool(v, default: bool = False, what: str = "value") -> bool:
    """Strict Boolean: real booleans, 0/1 and true/false/yes/no text. Blank or missing gives the default; anything
    else is an error rather than silently becoming True."""
    if _is_missing(v):
        return default
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"{what}: '{v}' is not true or false.")
