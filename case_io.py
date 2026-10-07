"""Case files: load / save a structure + buoyancy-module case as JSON.

Project data (weights, geometry, criteria) lives in a case file kept OUTSIDE the
repository (see .gitignore).  Heights in the file may be on the project's own datum
(e.g. elevation relative to MSL); `datum.z_offset_m` is added to every height on load,
so z_body = z_file + z_offset_m (e.g. +92.4 puts the jacket base at z = 0).

File layout (all sections optional except structure):
{
  "name": "...", "note": "...",
  "datum": {"z_offset_m": 92.4, "note": "EL relative to MSL; base at EL -92.4"},
  "structure": {"elements": [{name,x1,y1,z1,x2,y2,z2,d_out,d_in,exposed}...],
                "weights":  [{item,mass_t,x,y,z}...], "openings": [{name,x,y,z}...]},
  "modules":   [{name, members:[...], z_lo, z_hi,
                 elements:[{name,x1..z2,d_out,d_in}...], weights:[{item,mass_t,x,y,z}...],
                 ballast_t, ballast_z, note}...],
  "states":    {"module name": "sealed"|"off"|"damaged"},
  "params":    {Params fields},  "criteria": {Criteria fields},
  "reserve_min_pct": 0.0
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Union

import buoyancy as bu
import jacket_stability as js


class CaseFileError(ValueError):
    """Raised with a list of readable problems when a case file cannot be used."""

    def __init__(self, problems: list) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _num(v, where: str, problems: list, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        problems.append(f"{where}: '{v}' is not a number")
        return default


def _elements(rows, dz: float, where: str, problems: list, *, buoyant: bool) -> list:
    out = []
    for i, r in enumerate(rows or []):
        w = f"{where}[{i}]"
        try:
            vals = [_num(r[k], f"{w}.{k}", problems) for k in ("x1", "y1", "z1", "x2", "y2", "z2", "d_out")]
        except KeyError as exc:
            problems.append(f"{w}: missing {exc}")
            continue
        if any(v is None for v in vals):
            continue
        x1, y1, z1, x2, y2, z2, d_out = vals
        out.append(js.Element(
            name=str(r.get("name", f"{where}{i}")), p1=(x1, y1, z1 + dz), p2=(x2, y2, z2 + dz),
            d_out=d_out, d_in=float(r.get("d_in") or 0.0), buoyant=bool(r.get("buoyant", buoyant)),
            exposed=bool(r.get("exposed", True)), flooded=bool(r.get("flooded", False)),
            tank=bool(r.get("tank", True))))
    return out


def _weights(rows, dz: float, where: str, problems: list) -> list:
    out = []
    for i, r in enumerate(rows or []):
        w = f"{where}[{i}]"
        try:
            vals = [_num(r[k], f"{w}.{k}", problems) for k in ("mass_t", "x", "y", "z")]
        except KeyError as exc:
            problems.append(f"{w}: missing {exc}")
            continue
        if any(v is None for v in vals):
            continue
        out.append(js.WeightItem(str(r.get("item", r.get("name", f"{where}{i}"))),
                                 vals[0], vals[1], vals[2], vals[3] + dz))
    return out


def _dataclass_from(cls, d: dict, where: str, problems: list, dz: float = 0.0):
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = [k for k in d if k not in names]
    if unknown:
        problems.append(f"{where}: unknown fields {unknown}")
    kw = {k: v for k, v in d.items() if k in names}
    if cls is js.Params:
        if "tow_point" in kw:
            p = list(kw["tow_point"])
            if len(p) == 3:
                kw["tow_point"] = (p[0], p[1], p[2] + dz)
        kw.pop("lines", None)
    try:
        return cls(**kw)
    except TypeError as exc:
        problems.append(f"{where}: {exc}")
        return cls()


def load_case(source: Union[str, Path, dict]) -> dict:
    """Read a case file (path or already-parsed dict).  Returns a dict with keys
    name, structure, modules, states, params, criteria, reserve_min_pct, datum_offset_m."""
    problems: list = []
    if isinstance(source, dict):
        raw = source
    else:
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError([f"cannot read case file: {exc}"])
    if "structure" not in raw:
        raise CaseFileError(["case file has no 'structure' section"])
    dz = _num((raw.get("datum") or {}).get("z_offset_m", 0.0), "datum.z_offset_m", problems, 0.0)
    st = raw["structure"]
    structure = bu.Structure(
        elements=_elements(st.get("elements"), dz, "structure.elements", problems, buoyant=False),
        weights=_weights(st.get("weights"), dz, "structure.weights", problems),
        openings=[js.Opening(str(o.get("name", f"opening{i}")), float(o["x"]), float(o["y"]), float(o["z"]) + dz)
                  for i, o in enumerate(st.get("openings") or []) if all(k in o for k in ("x", "y", "z"))])
    modules = []
    for i, m in enumerate(raw.get("modules") or []):
        w = f"modules[{i}]"
        if "name" not in m:
            problems.append(f"{w}: missing name")
            continue
        zl, zh, bz = m.get("z_lo"), m.get("z_hi"), m.get("ballast_z")
        modules.append(bu.BuoyancyModule(
            name=str(m["name"]), members=tuple(m.get("members") or ()),
            z_lo=None if zl is None else float(zl) + dz, z_hi=None if zh is None else float(zh) + dz,
            elements=tuple(_elements(m.get("elements"), dz, f"{w}.elements", problems, buoyant=True)),
            weights=tuple(_weights(m.get("weights"), dz, f"{w}.weights", problems)),
            ballast_t=float(m.get("ballast_t") or 0.0), ballast_z=None if bz is None else float(bz) + dz,
            note=str(m.get("note", ""))))
    params = _dataclass_from(js.Params, raw.get("params"), "params", problems, dz)
    crit = _dataclass_from(js.Criteria, raw.get("criteria"), "criteria", problems)
    states = {str(k): str(v) for k, v in (raw.get("states") or {}).items()}
    if problems:
        raise CaseFileError(problems)
    return dict(name=str(raw.get("name", "case")), note=str(raw.get("note", "")), structure=structure,
                modules=modules, states=states, params=params, criteria=crit,
                reserve_min_pct=float(raw.get("reserve_min_pct", 0.0)),
                reserve_damaged_min_pct=float(raw.get("reserve_damaged_min_pct", 0.0)), datum_offset_m=dz)


def _el_row(e: js.Element, dz: float) -> dict:
    return dict(name=e.name, x1=e.p1[0], y1=e.p1[1], z1=e.p1[2] - dz, x2=e.p2[0], y2=e.p2[1], z2=e.p2[2] - dz,
                d_out=e.d_out, d_in=e.d_in, buoyant=e.buoyant, exposed=e.exposed, flooded=e.flooded, tank=e.tank)


def _w_row(w: js.WeightItem, dz: float) -> dict:
    return dict(item=w.name, mass_t=w.mass_t, x=w.x, y=w.y, z=w.z - dz)


def dump_case(case: dict, path: Union[str, Path]) -> None:
    """Write a case back to JSON on its original datum (inverse of load_case)."""
    dz = float(case.get("datum_offset_m", 0.0))
    s = case["structure"]
    p = asdict(case["params"])
    p["lines"] = []
    tp = list(p["tow_point"])
    p["tow_point"] = [tp[0], tp[1], tp[2] - dz]
    out = dict(
        name=case["name"], note=case.get("note", ""), datum=dict(z_offset_m=dz),
        structure=dict(elements=[_el_row(e, dz) for e in s.elements], weights=[_w_row(w, dz) for w in s.weights],
                       openings=[dict(name=o.name, x=o.x, y=o.y, z=o.z - dz) for o in s.openings]),
        modules=[dict(name=m.name, members=list(m.members),
                      z_lo=None if m.z_lo is None else m.z_lo - dz, z_hi=None if m.z_hi is None else m.z_hi - dz,
                      elements=[_el_row(e, dz) for e in m.elements], weights=[_w_row(w, dz) for w in m.weights],
                      ballast_t=m.ballast_t, ballast_z=None if m.ballast_z is None else m.ballast_z - dz,
                      note=m.note) for m in case["modules"]],
        states=case["states"], params=p, criteria=asdict(case["criteria"]),
        reserve_min_pct=case.get("reserve_min_pct", 0.0),
        reserve_damaged_min_pct=case.get("reserve_damaged_min_pct", 0.0))
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
