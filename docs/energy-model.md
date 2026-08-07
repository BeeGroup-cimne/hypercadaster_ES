# ⚡ How the energy model is built

How `hypercadaster_ES` turns one cadastral building into an EnergyPlus model, and
how the occupant-behaviour scenarios are run on top of it.

This is the *why* document. For the API signatures see
[Library Structure](library-structure.md); for the knobs see
[Configuration & Examples](configuration-examples.md); for the KPI columns see
[Output Data Schema](output-schema.md).

Everything below is configured from one file — `hypercadaster_ES/energy_config.yaml`.
Point `HC_ENERGY_CONFIG` at your own copy, or call `energy.configure(path=...)`;
values are merged over the built-in defaults, so you only write what you change.

---

## The two entry points

```python
idf  = hc.energy.build_energy_model(gdf, ref, wd=wd, weather_year=2021)
kpis = hc.energy.run_behavior_scenarios(gdf, wd=wd, weather_year=2021, idf=idf)
```

`build_energy_model` returns a **complete, simulable** model: geometry, envelope,
loads, HVAC and a full occupant model. `run_behavior_scenarios` copies it once per
scenario and re-applies only the archetype assignment.

That matters more than it sounds. There is exactly **one occupant engine**
(`_wire_occupant_model`), called by both, so the `.idf` written to disk, the one
drawn by `plot_idf_geometry`, and the one simulated all carry the same wiring. A
scenario is an archetype assignment and nothing else.

---

## 1. Geometry

| step | what happens |
|---|---|
| Footprint | Cadastral parcel geometry, optionally eroded to the CAT built area (`geometry.cat_area_shrink`); the leftover ring becomes self-shading |
| Storeys | `floors.floor_height`, with a taller ground floor for buildings of ≥ `ground_floor_min_floors` |
| Thermal zones | One per **dwelling/premise per floor**, from the double-loaded-corridor floorplan partition — façade-fronted dwellings, an interior communal corridor, an entrance corridor on floor 0 |
| Cell complex | Watertight space snapping with shared-face walls, slabs and roofs (`topology.py`), so adjacent dwellings share one surface rather than two coincident ones |
| Windows | Equally-spaced panes sized to the WWR (`windows.*`), skipping parking façades, party walls and doors; each gets a reveal shade |
| Shading | Balconies and awnings by construction period and orientation; neighbour panels filtered to those that can actually shade this building (`shading.neighbours`) |

Only **residential, office and commercial** uses are thermally modelled. A building
whose spaces are all parking/agricultural/communal has no thermal model at all.

## 2. Weather and seasons

The ERA5-Land EPW for the building's cell is attached **before** the systems are
built. This is load-bearing: seasons are derived from the weather, not the
calendar, and the schedule writers no-op on a name that already exists — wiring the
model before the EPW is attached would freeze a fixed month split into every
scenario.

A **day** is classified by its mean outdoor temperature, smoothed over a centred
week so a single warm day cannot flip the season:

- `heating` — below `seasons` lower bound (16 °C)
- `cooling` — above the upper bound (24 °C)
- `in_between` — between them: **no HVAC for any user**, whoever they are

Every seasonal schedule in the model is built from that per-day map.

## 3. Envelope and loads

Constructions are assembled to hit the target U-values of the building's
**construction period** (`archetype_envelope`: `pre1980`, `1980_2006`, `2007_2013`,
`post2013`) by solving for insulation thickness. Glazing takes the period's U, SHGC
and visible transmittance.

Infiltration is the period's window technology (`windows.infiltration`), given as
the **mean seasonal rate**, not the n50 blower-door figure — derived by the n50/20
rule for a sheltered urban site. It is then scaled by the dwelling **conservation
status** from the social layer (`ec__main_dwellings_conservation_status~*`), once,
at build time: the factor scales the infiltration objects in place, so a second
pass would square it.

Occupant density comes from the building's census-tract population
(`ctagg__people_per_m2`) rather than a national default, and the same head count
drives the DHW demand.

## 4. The occupant model

This is where most of the modelling lives.

### One occupancy, for everything

Each dwelling has exactly **one** occupancy source, used for its setpoints, its
HVAC availability *and* every window opening:

1. the census tract's **INE Time Use Survey** week (via `social_ES`, matched by
   autonomous community and the tract's household income band), where it reaches
   the building;
2. otherwise the stylised profile `_AWARE_OCCUPIED_SPANS` — weekdays home
   00–09, 14–16 and 19–24; weekends all day.

`_apply_occupant_control` records which applies to each zone on `idf._hc_occ_tag`,
so the passes that run afterwards can ask rather than guess, and the stylised
"someone is home" availability schedule is *generated* from the same spans. Two
hand-maintained copies of "when is somebody home" is precisely how a model ends up
conditioning for one occupancy and ventilating for another.

### Archetypes

An archetype is a set of branches in the engine; the numbers live in `hvac:`,
`residential_cte:` and the airing blocks. `residential_behavior.behaviors` only
documents which is which.

| archetype | thermostat | availability | windows | shading |
|---|---|---|---|---|
| **aware** | EN 16798-1 adaptive, confined to the comfort field, setback when empty and asleep | around the clock, **off while a window is open** | hygiene + midday + summer night purge | operates awnings and persianas |
| **unaware** | fixed band (`hvac.unaware`) | every **waking** hour, home or not — plus the nights it forgets to switch off | 1–3 short random airings a day; sleeps with the windows open on some nights | never touches them |
| **passive** | as aware, widest tolerance | as aware | as aware | as aware |
| **always_on** | fixed band (`hvac.always_on`) | 24 h | none | none |
| **cte** | CTE normative 21/26, no setback | all year | fixed 4 ren/h on summer nights 01–08 | radiation-triggered persiana only |

`passive` and `always_on` are in no default scenario; reach them with
`build_energy_model(residential_behavior=…)` or a custom `scenarios=` dict.

The archetype is drawn per dwelling, weighted by the section's mean age
(`assignment.behavior_weights`), deterministically seeded by the dwelling
reference, unless a scenario fixes it.

### Setpoints

The aware/passive thermostat **tracks** the running-mean outdoor temperature —
`Tc = 0.33·θrm + 18.8`, ± the Category-II band pulled inward — and is then
**confined** to the extended comfort field (`comfort_zone.setpoints`), so the
occupant adapts to the climate without adapting out of comfort. Three modifiers,
applied in this order:

1. **Night setback** (`aware_night_setback`, 23:00–07:00, heating only). The
   occupied/away split never sets back at night — the dwelling is at its *fullest*
   at 03:00 — so it held the full comfort setpoint through every sleeping hour.
   Cooling is left alone: those nights are already purged with the windows open.
2. **Preconditioning** (`aware_preconditioning`, 60 min). Every comfort period
   starts an hour early, so the system is already working when the household walks
   in — and when it wakes — instead of spending their first hour recovering.
3. **Income offset** (`income_setpoints`). An energy-poor household underheats;
   applied *after* the comfort-field confinement, so confining the target cannot
   quietly erase the behaviour being modelled.

The unaware and always_on bands are constant. The normative `cte` band is never
shifted by anything.

### Ventilation

Four independent routes, and they answer to different things:

| route | who | when |
|---|---|---|
| **Infiltration** | the fabric | always; the building's, not the occupant's |
| **CTE DB-HS3** | the building, if built ≥ 2007 | continuously — a ventilation system is part of the building, so *every* occupant of a post-CTE dwelling has it, the normative one included |
| **RITE mechanical** | office/commercial | whenever the premises are open, including the shoulder season when the HVAC is gated off |
| **Window openings** | the occupant | see below |

Every window opening is **gated on occupancy** — intersected, per day type, with
the hours somebody is home. Nobody opens a window in a dwelling they are not in,
and ungated the airings ran on the clock alone, ventilating an empty flat.

One rate for all of them (`opening_ach`, 4 ren/h): what separates the behaviours is
*when* and *for how long* they open, not how wide.

- **Hygiene** (aware/passive) — an air-quality habit, not a thermal strategy, so
  it is ungated on temperature. Per season: morning + evening in the shoulder
  season; **one 15-minute airing in a 13:00–14:00 band** in heating (08:30 and
  20:30 are the two coldest hours a winter household is awake for); morning only in
  cooling, since the night purge already opens the dwelling from midnight.
- **Midday free cooling** (aware/passive) — gated on outdoor ≥ 12 °C.
- **Night purge** (aware/passive, cooling season) — 00:00–08:00, **unattended**.
  No indoor-temperature gate: the household is asleep and cannot reopen or reclose
  a window at 04:00. This is also what keeps `aware` and `aware_ff` differing only
  by the HVAC being off — an indoor gate opened the free-floating twin's windows
  for more hours, because it drifts warmer.
- **Unaware daytime** — 1–3 openings a day of 15–30 min, drawn per dwelling,
  halved through the heating season.
- **Unaware night** — sleeps with the windows open on a share of nights
  (cooling season only).

Durations snap to the simulation timestep, because EnergyPlus rounds a sub-timestep
`Until:` anyway — better that the config states what will actually run.

**While a window is open, the aware HVAC does not run.** Its availability schedule
is the exact complement of the same gated openings; heating into a dwelling you
have just decided to air is what an occupant who ventilates deliberately does not
do.

---

## 5. The scenarios

```python
kpis = hc.energy.run_behavior_scenarios(gdf, wd=wd, weather_year=2021, idf=idf)
```

| scenario | archetype | HVAC |
|---|---|---|
| `random_mix` | each dwelling independently aware or unaware, 50/50 | yes |
| `unaware` | all unaware | yes |
| `unaware_ff` | all unaware | **stripped** |
| `aware` | all aware | yes |
| `aware_ff` | all aware | **stripped** |
| `cte` | the Spanish standard's normative user | yes |

Two invariants hold, and both have tests:

- **`_ff` twins differ from their parent ONLY by the HVAC being off.** Nothing else
  may vary — which is why no ventilation carries an indoor-temperature gate.
- **A property belongs either to the BUILDING or to the OCCUPANT.** Fabric,
  ventilation systems, head count and metabolism go on every scenario; only
  behaviour varies between them. This is why the DB-HS3 ventilation is added ahead
  of the per-behaviour branch, and why a `cte` run houses the same census
  population as the others.

> **`cte` is normative, not comparable.** The standard prescribes a fixed occupancy
> gain and says nothing about sleeping metabolism, so its People sit at a waking
> 110 W all night while the behaviour scenarios drop to 72 W. EnergyPlus derives
> CO₂ from metabolic watts, so the normative case exhales ~1.5× more at 03:00 for
> the same head count. That is a property of the law, kept deliberately. Read its
> CO₂ apart from the others.

## 6. KPIs and the report

One row per **use type × season**, area-weighted across zones. Seasons are the
temperature-derived ones, so a row is a real stretch of weather rather than a
calendar quarter.

**Occupied hours** are one fixed set of hours across the whole comparison — the
Time Use Survey week — rather than each run's own People output. A percentage of
discomfort over one occupancy is not comparable with the same percentage over
another. The `cte` case is the exception: it carries no occupancy profile of its
own, so its KPIs are taken over *all* hours.

Reported per row: thermal demand, internal and solar gains, delivered setpoint,
out-of-comfort percentages (adaptive band *and* the Mollier comfort field), indoor
temperature percentiles, indoor CO₂, outdoor-air change rate, and DHW demand.

> **No medians on the outdoor-air rates.** An air change rate is a *flow*: what a
> dwelling renews over a season is the sum of its hourly flows, the mean. A series
> that is 0 for twenty hours and 4 ACH for one has a median of 0 — a dwelling that
> never opens a window, which is the one thing the number is asked. Percentiles
> stay, to separate a dwelling flung open for an hour from one left ajar.

`hc.reporting.scenario_report(kpis, …)` writes one self-contained interactive page
per use, with the summarised comparisons and the raw hourly series each scenario
produced. Occupancy is shown as an **average week** at hourly resolution —
occupancy is a weekly rhythm, and a daily mean flattened the one thing worth
seeing.

---

## Where to change what

| you want to change | edit |
|---|---|
| storey heights, footprint erosion, glazing | `floors`, `geometry`, `windows` |
| envelope U-values by period | `archetype_envelope`, `insulation` |
| balconies, awnings, persianas, neighbour shading | `shading` |
| setpoints and HVAC limits | `hvac`, `comfort_zone.setpoints`, `income_setpoints` |
| who the occupants are and what they do | `residential_behavior` |
| the normative user | `residential_cte` |
| season boundaries, timestep, outputs | `seasons`, `simulation` |

Changing anything under `residential_behavior` or `hvac` changes simulation output:
**re-run the scenarios**, don't reuse cached KPIs.
