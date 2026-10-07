# Jacket Wet-Tow Stability (screening tool)

Streamlit app plus a pure-NumPy engine for the intact and damaged stability of a
**floating jacket on buoyancy tanks** during a wet tow. It replaces the earlier
single-cylinder spar model: the jacket is described by its real tanks, legs and
braces, so nothing is idealised to one diameter.

**Status: screening tool.** It has been validated against closed-form cases (below) but
has not been independently checked. The acceptance criteria shipped as defaults are
placeholders. Confirm both before anything goes into a client brief.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: tables, inputs, geometry check, results, exports |
| `jacket_stability.py` | Engine (no Streamlit): hydrostatics, free-trim sweep, loads, criteria |
| `report.py` | Draft A4 PDF report |
| `sensitivity.py` | Tank-count and tow-connection studies |
| `viz.py` | 3D jacket drawing (solid tubes, waterline, CoG/CoB, tow and wind arrows) |
| `test_jacket_stability.py` | Validation tests (`pytest -q`) |
| `requirements.txt` | streamlit, numpy, pandas, plotly, reportlab |

Run locally: `pip install -r requirements.txt && streamlit run app.py`.
On Streamlit Community Cloud, point the app at `app.py` in the repo; nothing else changes.

## Geometry tab and technical data

Tab 3 draws the jacket as solid tubes (legs, braces, translucent buoyancy tanks, flooded tanks in red) with three views:
as built, floating level, and the attitude at static equilibrium under wind + tow for a chosen heading (after a run).
Alongside it: principal dimensions, weight and CoG, displaced volume, draft, trim, CoB, waterplane area, TPC, reserve
buoyancy, KB / KG / BM / KM / GM, a member schedule with buoyancy per member, and the height of each opening above water.
KB, KG and KM are measured from the lowest buoyant point at the floating attitude. The Results tab adds a criteria check
and the equilibrium data (draft, wind, tow, heeling and righting moments) for the selected heading.

## Sensitivity tab (tank connections and tow connection)

Tab 5 shows how stability depends on how the jacket is connected.

- **A - Number of tanks attached.** Detaches tanks (no buoyancy, wind area or drag) in every combination, down to the chosen
  number lost (an even sample of 12 combinations if there are more), and reports for each tank count the worst and best
  arrangement: no-wind list, minimum GM, maximum static heel, minimum area ratio, reserve buoyancy. Outcomes are PASS / FAIL,
  LISTED (static list above 0.5 deg with no wind: the from-upright sweep is not trusted, so heel and ratio are not scored),
  CAPSIZES (no stable floating attitude with no wind or tow within 60 deg) or SINKS (not enough buoyancy).
- **B - Tow connection.** Editable table of tow points (a bridle is entered as its resultant point). The tow line is horizontal
  at that point. GM does not depend on the tow point; static heel and area ratio do.
- **C - Tank count x tow connection.** Grid of the reduced tank sets that still float stably against the tow points.

The no-wind list is checked against the wall-sided result (a sideways CoG shift d lists the body by atan(d / GM), within 0.3 %).
Studies run at the wind-heading step of the Analysis settings and a heel step of at least 5 deg to stay quick.
Limit: a case that lists with no wind is flagged, not scored; it needs a full analysis about the listed attitude.

## Synthetic jacket and set-down float check

The tables are pre-loaded with an OEC **synthetic** illustrative jacket (not project data): 105 m tall; four main legs
on a 7 m x 7 m square run the full height; two outrigger legs, 7 m outside the main legs, stop at 30 m (so the base is
7 m x 14 m with six legs) and are tied back to the main legs by a sloping strut. Four 3 m x 30 m buoyancy tanks sit high
on the four main legs (z 60-90 m). Legs and braces are free-flooding with no buoyancy credit; 700 t, CoG about 42 m,
balanced about the centreline by an assumed trim-ballast item that offsets the outrigger steel. The tank dimension is
taken as the full 3 m cylinder (the leg inside is not subtracted). Replace the tables in tab 1 with project data.

Tab 2 has **water depth (at LAT)** and **tide**, and two criteria: **minimum seabed clearance** and **minimum tank
length above water** (0 = off). The Results tab then shows the set-down float check: waterline above the base,
clearance = depth + tide - waterline height, tank length out of the water, freeboard, GM, lowest opening and the
**ballast headroom** - the weight that can be added at the CoG (negative = weight to shed) before the clearance or
tank-emergence limit is reached. It assumes a level waterline and does not model the ballast tanks or their free surface.
KB, KG and KM are measured from the lowest point of the structure.

## Tow and pull-in lines (connection points and force direction)

Tab 2 has an "Additional line loads" table. Each row is one line acting ON the jacket at a body-frame
connection point (x, y, z), with a force direction given as heading (horizontal, from body +x toward +y)
and elevation (+ = upward), plus a type:

- **Fixed pull**: the tension you enter (kN) is applied as given (pull-in line, tugger, restraint).
- **Tow leg**: shares the drag-based tow pull with the main tow line (a bridle). Leg tensions are scaled by
  "share" so the resultant along the mean heading equals calm-water drag x margin.

The 3D view (tab 3) draws every connection point and force arrow. Sensitivity section D adds the lines one at a
time to show what each extra line does to stability.

Assumptions: tensions are inputs (no catenary, no tug or line dynamics). A vertical force component reduces the
buoyancy needed (W - Fz/g) and its moment acts about the centre of buoyancy. The net horizontal line and wind
force is reacted by water at the submerged lateral-area centroid. Old saved inputs without lines load and give
identical results.

## Coordinates and conventions

* Body frame: **x forward (tow direction), y athwart, z up** in the floating attitude. Any
  origin, as long as every table uses the same one. Metres, tonnes, kN.
* **Wind heading** = direction the wind is blowing *toward*, in degrees from body +x
  (90° = toward +y). Heel is applied about the horizontal axis perpendicular to the wind;
  positive heel is in the wind direction. Headings are swept around 360° so a non-symmetric
  jacket is covered.
* The curves are **free to trim**: at every heel angle, draft and trim are solved so
  displacement = weight and the moment about the trim axis is zero.

## What you enter

1. **Weights**: every item with mass and CoG. Total weight and CoG are computed.
2. **Elements**: straight circular cylinders between two points (legs, braces, tanks).
   * *Buoyant*: watertight, contributes displacement. Flooded members are not buoyant.
   * *Exposed*: counts toward wind area and tow drag.
   * *Flooded (damage)*: loses buoyancy when the damaged scenario is run.
   * *Inner diameter > 0* makes an annulus (tank around a leg). Do not also mark that leg
     buoyant.
3. **Downflooding points**: vents and hatches (x, y, z).
4. **Environment, tow, criteria** (tab 2): wind speed, Cs, shielding, tow speed and point,
   tow-pull mode, free-surface moment, and the acceptance limits.

Save and reload everything with **Save inputs / Load saved inputs** in the sidebar so a study
is reproducible. The first screen opens with an **illustrative placeholder jacket**; it is
not Conrad Mako data and the sidebar warns while it is unchanged.

## What it calculates

* Exact hydrostatics of every buoyant cylinder against a tilted waterplane (closed-form
  cross-section cuts, Gauss-Legendre along the axis). No `D²/16T` shortcuts.
* GZ and heeling arm (wind + tow) vs heel for each heading; static equilibrium angle, second
  intercept, downflooding angle, area ratio, residual area, small-angle GM, trim at
  equilibrium, tow pull, reserve buoyancy.
* Wind: member-by-member projected area above the waterline, optional height profile.
* Tow: calm-water drag of submerged members (or a manual pull), tow-point height, and
  heeling/trim from the tow line. Both reacted by a hydrodynamic force at the lateral-resistance
  level (centroid of submerged lateral area, or half draft).
* Damaged scenario: flagged elements lose buoyancy.

### Criteria (user-set)

| Check | Default |
|---|---|
| Small-angle GM ≥ | 1.0 m |
| Static heel ≤ | 15° |
| Area under GZ / area under heeling arm ≥ | 1.3, both from 0° to the limit angle |
| Limit angle | min(second intercept, downflooding angle, cap 40°) |
| Downflooding angle ≥ | off (set above 0 to enforce) |

These are placeholders. Replace them with the project design basis / warranty requirements.

## Validation performed

Run `pytest -q` (10 tests). Highlights:

| Check | Reference | Result |
|---|---|---|
| Vertical cylinder, GM | wall-sided formula, 2.7722 m | 2.7723 m |
| Vertical cylinder, GZ at 10/20/30° | wall-sided formula | match to 4 decimals |
| Horizontal cylinder, GZ at 5/20/45° | (z_axis − z_G)·sinφ | match (< 0.001%) |
| Tow drag, 2.5 kn, Cd 0.70, D 14 m, draft 45 m | hand calc 373.9 kN | 373.4 kN (−0.13%) |
| Tow-induced trim, same case | hand calc 2.72° | 2.71° |
| Wind force and heeling arm, vertical cylinder | hand calc | match (59.57 kN, 0.0257 m) |
| Downflooding angle, known opening | atan(10/7) = 55.0° | within 1° |
| Square jacket | GM equal for all headings | yes |

Not independently checked against a commercial package. Before relying on a result for
a real jacket, run one representative case in your normal software (e.g. MOSES / GHS /
SACS) and compare GM, GZ and equilibrium draft.

## Limits and simplifications

* Static analysis: no wave-induced motion, gusting, or tow-line dynamics.
* Straight circular cylinders only (cones, boxes and tapered tanks are not modelled; model them
  as an equivalent cylinder or a short series of cylinders).
* Steel volume of flooded members is ignored; overlapping buoyant elements are double counted.
* Tow line is horizontal; no bridle geometry; no wave, current or added resistance in the tow pull.
* Free-surface effect is a uniform FSM/W·sinφ deduction.
* Downflooding is checked at the listed points only; progressive flooding is not modelled.
* Damage = lost buoyancy of flagged elements, with no added weight or sinkage progression.
* If a free-trim solution is unstable or unconverged inside the assessed range, the app says so
  and those results should not be used.
* Elements pierce the waterline at their true diameter. The wind area of braces uses
  full diameters with an optional shielding factor, which is conservative for dense lattices.

## Data needed for the Conrad Mako study

* Tow attitude (which way up, which way the jacket is facing), and the coordinate origin.
* Tank shape, size and position at each corner; leg and brace geometry for the wind area.
* Weight report with CoG (jacket, tanks, grout, ballast, rigging, temporary items).
* Vent, hatch and other downflooding coordinates.
* Tow point and bridle arrangement, tow speed, design wind, and the criteria to be met.
* Any ballast with free surface (free-surface moments).

## Buoyancy modules and case files

The jacket structure and each source of buoyancy are separate items:

* `buoyancy.py` - `Structure` (steel, no buoyancy) plus `BuoyancyModule` items: sealed members
  (whole, or the part between two heights such as rip-out diaphragms) and tanks (own volume,
  own steel weight, optional ballast). Each module is `sealed`, `off` or `damaged`.
  `evaluate`, `sweep` (every combination) and `damage_cases` (flood one module at a time)
  return plain rows.
* `case_io.py` - load / save a case as JSON. Heights can be on the project datum;
  `datum.z_offset_m` is added on load. See `docs/case_template.json` (dummy numbers).
* `run_case.py` - `python run_case.py my.case.json [--sweep | --damage]`.

Project data (weights, geometry, criteria) belongs in a case file kept outside the repository;
`cases/`, `private/` and `*.case.json` are git-ignored.
Elements have a `tank` flag (default True); sealed legs and braces are `tank=False`, so the
"tank emerged length" check only looks at real tanks.
