"""Command-line runner for a case file.

    python run_case.py my.case.json              # the case as saved (states in the file)
    python run_case.py my.case.json --sweep      # every sealed / off combination of the modules
    python run_case.py my.case.json --damage     # flood each present module in turn

Prints a plain table.  Case files hold project data: keep them out of the repository.
"""
from __future__ import annotations

import argparse
import sys

import buoyancy as bu
import case_io

COLS = [("case", "Case", "{}"), ("weight_t", "W [t]", "{:.1f}"), ("capacity_t", "Cap [t]", "{:.1f}"),
        ("reserve_pct", "Res [%]", "{:.1f}"), ("draft_m", "Draft [m]", "{:.2f}"),
        ("clearance_m", "Clear [m]", "{:.2f}"), ("gm_min_m", "GM [m]", "{:.2f}"),
        ("free_tilt_deg", "Tilt [deg]", "{:.1f}"), ("passed", "Pass", "{}")]


def _fmt(row: dict, key: str, f: str) -> str:
    v = row.get(key)
    if v is None or v == "":
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def table(rows: list) -> str:
    cells = [[h for _, h, _ in COLS]]
    for r in rows:
        if r.get("floats"):
            cells.append([_fmt(r, k, f) for k, _, f in COLS])
        else:
            cells.append([r.get("case", "")] + ["-"] * (len(COLS) - 2) + [f"NO FLOAT: {r.get('error', '')}"])
    w = [max(len(str(c[i])) for c in cells) for i in range(len(COLS))]
    lines = ["  ".join(str(c[i]).ljust(w[i]) for i in range(len(COLS))) for c in cells]
    lines.insert(1, "  ".join("-" * x for x in w))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--sweep", action="store_true", help="all sealed/off combinations of the modules")
    ap.add_argument("--damage", action="store_true", help="flood each present module in turn")
    a = ap.parse_args(argv)
    try:
        c = case_io.load_case(a.case)
    except case_io.CaseFileError as exc:
        print("Case file problems:\n  - " + "\n  - ".join(exc.problems), file=sys.stderr)
        return 2
    s, mods, p, cr, rm = c["structure"], c["modules"], c["params"], c["criteria"], c["reserve_min_pct"]
    print(f"{c['name']}  (datum offset {c['datum_offset_m']:+.2f} m; screening calculation, not independently checked)")
    if a.sweep:
        rows = bu.sweep(s, mods, p, cr, reserve_min_pct=rm)
    elif a.damage:
        rows = bu.damage_cases(s, mods, c["states"], p, cr, reserve_min_pct=rm)
    else:
        rows = [bu.evaluate(s, mods, c["states"], p, cr, reserve_min_pct=rm)]
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
