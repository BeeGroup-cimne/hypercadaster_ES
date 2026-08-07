# Changelog & Version History

All notable changes to hypercadaster_ES will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-08-07

### 🧹 One occupant engine (breaking: `residential_behavior` config keys removed)

There were **two** behaviour models. `build_energy_model` wrote thermostats, HVAC
availability and `ZoneVentilation` objects from the `behaviors:` block; then
`run_behavior_scenarios` copied that model, **deleted every `ZoneVentilation`
object** and re-authored setpoints, availability, occupancy and ventilation from
scratch. Every key of the first model was overwritten before a single KPI was
computed — editing them changed nothing.

`_apply_residential_behavior` is now `_assign_residential_behavior` and authors
**nothing**: it assigns the archetype and returns the map. Both entry points then
call one `_wire_occupant_model`, so the .idf that is written, plotted and simulated
all carry the same wiring, and `build_energy_model` returns a **complete, simulable
model** rather than a shell the scenarios replaced.

**Removed from `residential_behavior`** (they were read only by the discarded
engine): `behaviors.*.heating` / `.cooling` / `.hvac_availability` /
`.ventilation_ach` / `.ventilation_schedule`, the whole `ventilation_schedules`
block, `occupancy_periods`, and `assignment.household_buckets` /
`.period_weights`. `behaviors:` is now four `description:` entries saying where
each archetype is wired — an archetype is a set of branches in the engine, and the
numbers live in `hvac:`, `residential_cte:` and the airing blocks.

Consequences worth stating:

- **Occupancy is per BUILDING, not per dwelling.** The per-dwelling period draw
  (`nights_workdays` / `days` / `all_day` / `weekends_away` by household size) is
  gone. It has moved no KPI since occupancy moved to the census tract's Time Use
  Survey agenda, with one stylised profile as the fallback.
  `residential_period=` is still accepted and now documented as ignored.
- **`hvac.always_on: {heating: 21, cooling: 24}`** is new. That archetype's
  setpoints used to live in `behaviors`; it now has its own thermostat, so it
  works in the unified engine instead of inheriting whatever ran last.
- **`always_on` and `passive` stay** as extension points — no default scenario
  uses them, but `build_energy_model(residential_behavior=…)` and a custom
  `scenarios=` dict both reach them. The yaml now says so.
- **The EPW is attached BEFORE the systems are built.** The occupant engine and the
  persiana control take their seasons from the weather and fall back to a fixed
  month split without it — and since the schedule writers no-op on an existing
  name, a model wired before the weather was attached kept that fallback through
  every scenario. The conservation-status infiltration factor moved to build time
  for the opposite reason: it scales in place, so a second pass would square it.
- Dead code removed: `_apply_summer_night_purge`, `_apply_random_dwelling_behavior`
  (both superseded and already unreachable).

### 📖 New: `docs/energy-model.md`

A single page explaining how the EnergyPlus model is assembled and **why** — the
geometry and envelope pipeline, weather-derived seasons, the one occupant engine,
the occupancy source, the archetypes, the four ventilation routes, the scenarios
and their invariants, and the KPI definitions. Linked from the README and from the
energy sections of the quick-start, configuration and library-structure pages.

While writing it, three dead keys came out of `_DEFAULT_SCENARIOS`:
`random_ventilation`, `random_hvac_use` and `seasonal_setpoints`. Nothing had read
them since the behaviour model was unified — the stochastic ventilation they named
is now part of the unaware archetype itself — and `period` went with the
per-dwelling occupancy period. A scenario spec is now four keys: `behavior`,
`free_floating`, `mix_aware_unaware`, `residential_only`, pinned by a test so none
can creep back.

### 📐 The occupied hours are one set of hours, not six

Every comfort, CO₂ and air-change KPI (report figures 6.1–7.4 and the raw
psychrometric chart) is taken over the OCCUPIED hours — and each scenario was
taking them over **its own** People output. A percentage of discomfort over one
occupancy is not comparable with the same percentage over another, so every column
of the comparison was answering a slightly different question.

The mask is now imposed from outside the run: the census tract's Time Use Survey
week, the same agenda that drives the setpoints and the airings, for every
scenario. The normative `cte` case is the exception — it carries no occupancy
profile of its own, so its KPIs are taken over **all** hours rather than over an
occupancy the standard never defines. Without a survey agenda for the tract the
KPIs fall back to the run's People output as before, and say so in the log.

New `_time_use_occupied_mask` (agenda → `{day_type: spans}`) and `_hourly_occupied`
(mask + calendar → hourly boolean); `_zone_scenario_kpis` takes `occupied_mask=`.

### 📅 Figure 5 is now an average WEEK

*"Occupancy of people, daily mean"* → **"Occupancy of people, average week"**: 168
points, Monday 00h to Sunday 23h, each hour-of-week averaged across the run.
Occupancy is a weekly rhythm, and a daily mean flattened the one thing worth
seeing — who is home at 03:00, who leaves at nine, how Saturday differs from
Tuesday — into a single number a day, on a year-long axis that hid what was left.
New `reporting._fig_weekly_profile`, with dotted day boundaries and weekday ticks.

### 👥 One occupancy definition, everywhere

The thermostat and the airings were reading **different** occupancies: window
openings and the windows-open HVAC gate followed the census tract's INE Time Use
Survey week where it was available, while the adaptive setpoints followed a
hard-coded stylised profile. In a building with a survey agenda the model was
conditioning for a household that comes home at 14:00 while ventilating for one
that comes home at 18:00.

There is now ONE occupancy source per dwelling, used for everything the occupant
does — setpoints, HVAC availability, and every window opening:

1. the census tract's Time Use Survey week, where it reaches the building;
2. `_AWARE_OCCUPIED_SPANS` otherwise (weekdays home 00–09, 14–16, 19–24; weekends
   all day).

`_apply_occupant_control` records which applies to each zone (`idf._hc_occ_tag`)
along with the masks themselves, and the adaptive thermostat groups its zones by
it — so a building whose survey covers only some behaviours gets one thermostat per
mask (`SP_ADAPT_HEAT_TUS` / `_STY`) rather than one mask imposed on all of them.

`_AVAIL_PRESENT`, the stylised "someone is home" availability, is now **generated**
from `_AWARE_OCCUPIED_SPANS` instead of written out by hand — that duplication is
exactly how a model ends up conditioning for one occupancy and ventilating for
another.

Note the UNAWARE availability is deliberately *not* occupancy: that archetype
leaves the HVAC on through every WAKING hour, at home or not, which is what makes
it the unaware one. Its window openings are gated on occupancy like everyone
else's — you cannot open a window you are not standing next to.

### 🔥 The aware system starts an hour early

`residential_behavior.aware_preconditioning` (`lead_minutes: 60`) pulls the start
of every comfort period of the AWARE/PASSIVE thermostat forward, so the system is
already working when the household walks in — and when it wakes — rather than
starting from a setback dwelling at the moment somebody arrives. That is what a
thermostat with a schedule on it actually does; without it the occupant's first
hour home was always spent recovering.

Heating and cooling alike. With the defaults the weekday heating setpoint arrives
at 06:00 (home 07:00), 13:00 (home 14:00) and 18:00 (home 19:00):

```
For: Weekdays
  Until: 06:00,16    Until: 13:00,16    Until: 18:00,16    Until: 24:00,16
  Until: 09:00,21    Until: 16:00,21    Until: 23:00,21
```

### 🚪 Nobody opens a window in a dwelling they are not in

**Every window opening is now gated on OCCUPANCY** — the aware hygiene, midday and
night-purge airings, and the unaware daytime and night ones alike. Each opening is
intersected, per day type, with the hours somebody is home. Ungated they ran on the
clock alone, so an 08:30 airing (or a random burst at 11:00) happened just the same
on a weekday when the flat was empty from nine to two, ventilating — and in winter
cooling — nobody.

The mask is the census tract's Time Use Survey week where it drives the dwelling
and the stylised presence profile otherwise, so the airing schedules are built once
per source (at most two per building) and picked per zone. The "HVAC off while a
window is open" availability is derived from the *same* gated spans, or it would
idle the system for airings that no longer happen.

**The heating-season hygiene airing moved to the middle of the day**, 15 minutes in
a `heating_window: ["13:00", "14:00"]`. Airing at 08:30 and again at 20:30 in
January opens the dwelling at the two coldest hours it is awake for. It is a BAND
plus a duration, not a fixed time: gated on occupancy, an airing pinned to 13:00
sharp would never fire on a day the household is out until two, so it is placed at
the first quarter-hour of the band the dwelling is actually occupied.

**The unaware occupant airs half as often in the heating season**
(`unaware_day_ventilation.heating_factor: 0.5`) — a cold day is a day you open the
window less; this occupant is careless, not immune to the weather. Randomised
rounding again, so the half is exact across the building.

⚠️ **Interaction to be aware of:** the default 13:00–14:00 band and the *stylised*
presence profile do not overlap on weekdays (that profile has the household out
09:00–14:00), so where the Time Use Survey does not reach the tract the winter
hygiene airing lands at weekends only. Widen the band to `["13:00", "16:00"]` to
have it fall in the comida hours on weekdays too.

### 🐛 The report's daily HVAC series was a flat zero

Figure 3 (*HVAC energy, daily total*) drew nothing for any scenario, while the KPI
table above it showed a real `thermal_demand_kwh_m2`.

ExpandObjects reports the ideal-loads variables under the **system** it built —
`FLAT_F1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy` —
not under the zone. `_hourly_building_series` matched the output key against the
kept zone names **exactly**, so every HVAC column was dropped the moment `zones`
was passed, which is always: the report keeps each use apart. With no columns
matched, `_sum_kwh` returns zeros, and the plot was honest about a series that had
been filtered away.

The zone filter now accepts a key that starts with a kept zone name at a space
boundary — the same rule `energy._zone_column` already used to find those columns
for the KPIs, which is exactly why the table and the plot disagreed.

### 🛌 Nobody holds 21 °C in their sleep

**The AWARE/PASSIVE heating setpoint now sets back through the night**
(`residential_behavior.aware_night_setback`, 23:00–07:00 → `hvac.aware_setback.
heating`). The adaptive thermostat sets back when the dwelling is *empty*, which
never happens at night — the household is at its fullest at 03:00, so it read as
occupied and held the full comfort setpoint until morning. Eight sleeping hours at
21 °C is a large share of the winter demand, spent on hours nobody is awake to
feel.

Cooling is deliberately untouched: in the cooling season this occupant is already
purging the night with the windows open and the HVAC off, so a cooling setback
would gate hours that are not conditioned anyway.

**The UNAWARE occupant now FORGETS to switch the HVAC off on some nights**
(`unaware_night_hvac.nights_fraction: 0.25`). It leaves the system on through every
waking hour and it went off overnight, every night, without exception — so the
careless dwelling came out looking better overnight than the careful one, which is
backwards. Forgetting is the archetype's defining failure.

Drawn per dwelling as whole nights of the week, on a random stream independent of
the nights the same occupant sleeps with the windows open: it may well leave both
the heating on and a window open on the same night. They do not coordinate. New
`_split_day_blocks` / `_join_day_blocks` rewrite a single day type of a
`Schedule:Compact` in place, so this works on the survey-driven per-weekday
availability as well as the stylised one, with every day type still named exactly
once (naming one twice is an EnergyPlus error).

### 🪟 The aware occupant does not heat an open dwelling

`aware` and `passive` now have their HVAC switched **off while the windows are
open** — a new `AVAIL_AWARE_WINDOWS` availability schedule built as the complement
of that season's airings (hygiene + midday while heating and in-between, hygiene +
night purge while cooling). Leaving the thermostat live through an airing had the
ideal-loads system paying for the air the occupant had just decided to throw away.

Because it is a schedule rather than an EMS program, the outdoor-gated midday
airing idles the HVAC for its slot even on a day too cold for the window to open.
That is the smaller error; heating with the window open on every mild day was the
larger one. Relatedly, `midday_ventilation.periods` is now `14:00–14:15`: on the
4-per-hour timestep EnergyPlus rounded the old 20 minutes anyway, so the config was
claiming a duration that was never simulated.

**`unaware_night_ventilation.seasons` is now `[cooling]` only.** Sleeping with the
windows open is a hot-night habit; the in-between season is mild precisely because
the night is already cool, so an unattended 8-hour opening there dumps heat the
dwelling wanted.

### 🌙 The night, at the window: three corrections

**The UNAWARE occupant's daytime airings are drawn, not hard-coded.** They were
fixed at exactly three openings of exactly 15 minutes, so every unaware dwelling
in Spain aired for the same 45 minutes a day and only the hours differed. New
`residential_behavior.unaware_day_ventilation` — `openings_per_day: [1, 3]`,
`duration_minutes: [15, 30]`, inside an `hours: ["08:00", "22:00"]` band you can
narrow. Both are drawn per dwelling, they start on the hour (at most one an hour),
and durations are **snapped to the simulation timestep** since EnergyPlus rounds a
sub-timestep `Until:` anyway — the config now says what will actually run.

**The UNAWARE occupant now sleeps with the windows open some of the time.** It was
modelled as sealed from dusk to dawn — three 15-minute daytime bursts and nothing
else — which put its night-time CO₂ above every other archetype's for a reason
that was an artefact of the model, not of the behaviour. New
`residential_behavior.unaware_night_ventilation`:

```yaml
unaware_night_ventilation:
  nights_fraction: 0.30            # share of nights with the windows open
  periods: [["00:00", "08:00"]]
  seasons: [cooling, in_between]   # not January
```

The share is drawn **per dwelling**, seeded by the zone name, and realised as
whole nights of the week — a dwelling gets 2 or 3 nights, randomised so that the
0.30 is exact across the *building* rather than rounded in every dwelling. It
reaches `unaware`, `unaware_ff` and the unaware share of `random_mix`.

**The AWARE night purge is no longer gated on the indoor temperature.** It carried
`min_indoor: 22 °C` + a 1 K indoor–outdoor delta, which made the opening a
*thermostat* rather than a habit — and the household is **asleep**: nothing
reopens or recloses a window when the room crosses a threshold at 04:00. The
decision is made at bedtime and holds until morning (`night_purge.periods`).

That gate was also breaking a scenario invariant. A free-floating dwelling drifts
warmer than a conditioned one, so an indoor-temperature gate opened `aware_ff`'s
windows for **more hours** than `aware`'s — the two must differ *only* by the HVAC
being off. They now do, on the ventilation side.

**No medians on the outdoor-air rates.** An air change rate is a FLOW, and the
median of a series that is 0 for twenty hours and 4 ACH for one is 0 — a reading
that describes a dwelling which never opens a window, which is the one thing the
number is asked. `ach_total_p50_*` is **gone**, replaced by `ach_total_mean_*`;
`_ACH_PERCENTILES` is now `(10, 20, 80, 90)` and the report's spread chart (8.1)
marks the **mean**. The two route columns were already means. Temperatures keep
their median, where it means what it says.

### 🎛️ …and the setpoints follow it (`comfort_zone.setpoints`)

Measuring against the comfort field while steering by the adaptive band left the
model marking hours comfortable on the criterion that produced them and far
outside the field on the same hours. The AWARE/PASSIVE thermostat is now
**confined** to the field:

- **Temperature.** The adaptive tracking is kept and simply bounded — the setpoint
  still follows the running-mean outdoor temperature, but stops at the field's
  edge (20.5 / 25.5 °C with `margin_c: 0.5`, taken off both ends because the
  thermostat holds AIR temperature while the criterion is on OPERATIVE). The
  occupant still adapts to the climate; it just no longer adapts out of comfort.
- **Both ends are confined, with a `min_deadband_c`.** Flooring the heating
  setpoint while only capping the cooling one is not enough: in a hot week the
  adaptive heating setpoint climbs past the ceiling and EnergyPlus dies on the
  inversion (*"Effective heating set-point higher than effective cooling
  set-point"*). Found by simulating, not by reading.
- **Humidity**, via the ideal-loads humidistat — on the TEMPLATE's setpoint
  fields, because ExpandObjects writes the `ZoneControl:Humidistat` itself and a
  second one is a fatal duplicate. Default `dehumidification`: a Spanish dwelling
  has a cooling coil, not a humidifier, so the 30 % floor is reported but not
  conditioned for.
- **`humidity_setpoint: field`.** A humidistat holds a relative humidity, but the
  field's ceiling is the tightest of the RH, absolute-humidity and enthalpy
  bounds — above ~22.3 °C that is the enthalpy line. Held at the bare `rh_max` of
  65 % the dwelling still sits outside the zone the setpoints exist to deliver;
  the field's effective ceiling at the 25.5 °C cooling setpoint is **47.1 % RH**.
- The income-driven offset is applied **after** the confinement, so an energy-poor
  household still underheats — confining the comfort *target* must not quietly
  erase a modelled behaviour.

Simulated on the reference building (`aware`, 2021, occupied hours):

| | in the comfort field | too warm | too humid | heating | cooling |
|---|---|---|---|---|---|
| adaptive band only | 29.7 % | 28.7 % | 25.1 % | 33.1 MWh | 9.6 MWh |
| confined + humidistat | **37.6 %** | 20.1 % | 18.5 % | 34.6 MWh | **26.9 MWh** |

Holding the field nearly triples the cooling demand. That is the honest price of
the criterion, and it is now visible instead of avoided by a setpoint that drifted
with the weather.

### 🎯 Extended thermal comfort — the Mollier h-x comfort field, not a temperature band

Comfort was assessed on temperature alone (the EN 16798-1 adaptive band), which is
a 1-D criterion drawn on a 2-D chart. It now also uses the **extended comfort
zone** — the classic Mollier h-x comfort field of EN ISO 7730 / DIN 1946-2 /
ASHRAE 55 ([source](https://www.andivi.com/comfort-zone-indoor-en-iso773-din-1960-ashrae-standard-55/)) —
configurable under `comfort_zone`:

| bound | default |
|---|---|
| operative temperature | 20–26 °C |
| relative humidity | 30–65 % |
| absolute humidity | ≤ 11.5 g/kg |
| enthalpy | 30–50 kJ/kg |

The field is a **polygon, not a rectangle**: every bound is evaluated at every
temperature and the tightest wins, so the zone re-derives itself from the config
instead of carrying hard-coded vertices. With the defaults the 65 % RH ceiling
governs up to ≈22.3 °C, the 50 kJ/kg enthalpy line takes over above it, and the
11.5 g/kg cap never binds — which is exactly the bent top edge the standard's
diagram shows.

- **`energy.comfort_zone_polygon()`**, `_outside_comfort_zone()`,
  `_comfort_zone_humidity_limits()` and the psychrometrics (`_p_sat`,
  `_humidity_ratio`, `_relative_humidity`, `_moist_air_enthalpy`) now live in
  `energy.py` with the rest of the physics; `reporting.py` draws the model rather
  than defining a second copy of it, so the chart and the table cannot disagree.
- **New KPI columns** `pct_hours_out_of_comfort_zone_day` / `_night`, beside the
  existing adaptive `pct_hours_out_of_comfort_*`, and report figures **6.3/6.4**.
- The psychrometric chart's filled region is now the comfort zone (the criterion
  behind its `% in zone`); the adaptive band is drawn as a **dashed outline**,
  because a temperature-only criterion has nothing to say about the vertical axis.

**Both criteria are kept, because they disagree and the disagreement is the
finding.** On the reference building the aware occupant spends **82 %** of its
occupied hours inside the climate-relative adaptive band but only **28 %** inside
the static comfort field — the field caps at 26 °C, which a Lleida summer simply
exceeds. That gap is the argument for the adaptive model, and it is now visible
rather than assumed.

`_aggregate_scenario_kpis` also stopped indexing records with `[]`, so a KPI added
later can no longer make the aggregator raise on an older record.

### 🌡️ Thermal comfort on a psychrometric chart, by season and by occupant

A temperature cannot say whether a dwelling was comfortable: 26 °C at 8 g/kg and
26 °C at 16 g/kg are different rooms to be in. The report's raw section gains
**2. Thermal comfort, psychrometric** — every OCCUPIED hour as a point in
(operative temperature, humidity ratio), one colour per occupant, over the
EN 16798-1 adaptive band, with buttons for *whole year / heating / in_between /
cooling*.

- **x is OPERATIVE temperature**, not dry-bulb: it is what the adaptive criterion
  is defined on (and what ASHRAE 55's graphical method plots), and it carries the
  radiant half of comfort a dry-bulb chart drops. **y is humidity ratio**, so the
  constant-RH curves are the background nomogram and equal vertical distances mean
  equal water.
- The shaded band is the Category II band at the **season's mean** running-mean
  outdoor temperature; the **"% in band"** beside each occupant in the legend is
  counted **hour by hour** against that hour's own band — the same arithmetic
  behind the discomfort KPIs, so the two never disagree. The EN 16798-1 12 g/kg
  humidity guidance is a dashed reference line, deliberately *not* a side of the
  band, for the same reason.
- Hours are binned onto a 0.2 °C × 0.2 g/kg grid with opacity by hours per cell,
  so the dense core of a year shows through its outliers and the page does not
  carry 50 000 raw points.
- All seasons share one frame, so the clouds can be compared between selections
  and not only within one.

**This needs a re-run.** `Zone Mean Air Humidity Ratio` is now requested from
EnergyPlus; runs made before it simply have no chart (the figure returns `None`
rather than failing).

### 🏢 One report per USE, and a ventilation system for post-CTE dwellings

**The scenario report is now written per use — `scenario_report` returns
`{use: path}`.** It used to collapse every zone into one "building" series, so the
raw hourly section answered nobody's question: in the reference building a 144 m²
parking box (no occupants, 1 ren/h) lifted the dwellings' air-change curve from
0.58 to 0.64 ACH and pulled their CO₂ from 573–627 ppm down to 519–553, towards
the outdoor background. Each use now gets `{stem}_{use}.html` over its own zones,
split by `zone_uses` (`energy.zone_use_categories(idf)`, else the builder's
`.hcmeta.json` sidecar). The non-residential pages show the `normative` run they
were summarised from, since those zones carry the same RITE user in every
scenario. `_hourly_building_series` gained a `zones=` filter (the site weather
survives it); `scenario_report` returns a dict rather than a path, and `{}` when
only a CSV was asked for.

**Post-CTE dwellings carry a DB-HS3 continuous-ventilation system, in every
scenario** (`hs3_ventilation` in the config). CTE DB-HS3 has required continuous
dwelling ventilation since the code came into force (obligatory 29-09-2006), yet
the model gave dwellings nothing but leakage — which left the *normative* `cte`
case the **worst**-ventilated of the set, its only renewal outside the CTE
summer-night airing being infiltration. A ventilation system is a property of the
building, not of the person in it, so it is added on the residential path ahead of
the per-behaviour branch and no occupant can skip it. Sized per DB-HS3 Tabla 2.1
as `max(admission, extraction)` — `occupants × 8 l/s` (5 dormitorios + 3 salas)
against `area × 0.55 l/s·m²` (baños + cocina) — which on the reference dwelling is
62 l/s ≈ **0.66 ACH**, in line with the ~0.6 ren/h rule of thumb for an HS3
system. Gated on `from_year: 2007`; an unknown construction year is treated as
non-compliant, since inventing a system a building may not have would flatter its
air quality on no evidence.

### 🪟 Awnings follow the façade bay, not the individual window

Awning adoption was hashed on the WINDOW NAME (`{wall}_win{i}`), and every floor
has its own walls — so the stack of windows sitting at one façade XY drew
independently on each storey and the awnings looked scattered up the façade. The
draw is now made once per **façade column** (`energy._window_columns`, plan
position clustered with a 0.4 m snap so a float ulp cannot split a stack), and
shared by every floor of that bay — which is how a real façade reads. The
adoption *rate* (income × conservation state) is unchanged, so the building-level
fraction of awninged windows is the same; only their arrangement is.
`shading.awning.adoption_scope` selects `column` (default), `window` (the old
per-window draw) or `building` (all-or-nothing).

### 🌙 Night shutters, a lighter summer hygiene airing, and a numbered report

**The aware occupant closes the persianas at night** in the heating and in-between
seasons — a Spanish household does, and a closed persiana plus its air gap cuts
the loss through the glazing. Heating demand **43.9 → 41.7 kWh/m²** (−5 %) with the
winter solar gain untouched (28.05 → 27.98).

Getting there took two EnergyPlus corrections, both of which failed *silently* in
the sense that the results simply carried on without the feature:

- `OnNightIfLowOuterTempAndOffDay` is not a valid enum (it is `…LowOutdoorTemp…`).
  E+ raised a Severe error, ran anyway, and the shutters never came down — the
  demand was byte-identical to the run before, which is what gave it away.
- Two **dissimilar** `WindowShadingControl` objects on one window are refused
  (`CheckWindowShadingControlSimilarForWindow`, 45 Severe errors). The night rule
  therefore rides on the control TYPE of the single control instead:
  `OnNightAndOnDayIfCoolingAndHighSolarOnWindow` shades at night always and by day
  only when cooling is called for — so the heating season keeps its sun, which is
  exactly the rule set earlier. The normative `cte` case keeps plain
  `OnIfHighSolarOnWindow`.

**The evening hygiene airing is dropped in the cooling season**
(`hygiene_ventilation.cooling_periods`): the night purge already opens the
dwelling from midnight, and a 20:00 opening on a hot evening brings in air warmer
than the room. Aware's cooling-season window ventilation 0.3 → 0.1 ACH.

**The report is numbered and reordered** into a table of contents — 1 thermal
demand … 6.1/6.2 discomfort, 7.1–7.4 outdoor air split by route *and* by day/night,
8.1/8.2 the two spreads, 9.1/9.2 CO₂ mean while occupied. New columns
`co2_mean_day` / `co2_mean_night` back the last pair (the mean, not the median,
which on a lopsided CO₂ distribution is a different number).

The raw section gains **occupancy of people, daily mean** and an **air-hygiene**
figure — CO₂ breathed, the outdoor-air change renewing it, and the excess over the
~420 ppm background that ventilation has to remove — rendered daily, over the 7
worst days, and over each of the extreme cooling and heating weeks.

### 📉 Occupancy from the survey's household SHARES, not its argmax label

`WeeklyScheduleHouseholdOccupancy` reports the *dominant* state each hour — an
argmax. Driving the occupancy from it made every hour a hard 0 or 1, and on the
reference tract that turned a Sunday where **58–84 % of households are home** into
**100 % occupied for all 24 hours**: a far harder profile than the survey
measured, and the source of an implausible weekend CO₂ peak.

The companion `WeeklyScheduleHouseholdOccupancyValue` gives the actual shares, and
`social.time_use_household_occupancy` was already fetching them. They now drive:

- **occupancy** — the fraction of households at home (`Sleeping + WithActivity`);
- **gains** — the share-weighted blend of the three per-state gain fractions;
- **metabolic rate** — that of those *at home*, split between asleep and awake.

The two binary decisions (HVAC availability for `aware` and `unaware`) still come
from a threshold on the share, where a yes/no is what is wanted. Without shares it
falls back to the argmax unchanged.

Weekend CO₂ excess over the weekday baseline, daily mean:

| | before | after |
|---|---|---|
| `aware`, Sunday | +72 ppm | **+26** |
| `unaware`, Sunday | +71 ppm | **+22** |

Still there — Sunday genuinely has 73 % of households home against 33–45 % on a
weekday — but no longer a cliff.

### 🌤️ The aware occupant now airs at midday in winter too

The midday airing was gated at **outdoor > 20 °C**, which in a Lleida January never
fires, so the aware occupant had no midday airing at all in the heating season.
The gate is now 12 °C and the airing runs in the **heating and in-between** seasons
as well as cooling (`residential_behavior.midday_ventilation`). The mild season is
when opening a window is cheapest, and the occupant who cares about air quality is
exactly the one who does it.

### 📊 Hourly CO₂ for the cooling and heating weeks; ACH everywhere

The hourly CO₂-and-air-change figure now also renders over the **worst cooling
week** and the **worst heating week**, beside the temperature/demand panels for
the same days — so the windows opened (or not) for temperature can be read
against what they did for the air. The unit label is **ACH** throughout, replacing
`1/h`.

> The Sunday dip in the building-average air change is **real and correct**: the
> ground-floor shop is shut on Sundays, so its RITE mechanical ventilation stops
> (0.89 → 0.00 ACH), pulling the whole-building mean from 0.55 to 0.39. The
> dwellings' own air change is unchanged; the raw-output charts average every
> zone, commercial included.

### 📅 The weekly schedules were running two days off the calendar

`RunPeriod`'s **Day of Week for Start Day** was left blank, so EnergyPlus assumed
1 January is a **Sunday**. With 2021 weather — 1 January was a **Friday** — every
`For: Saturday` / `For: Sunday` block ran two days early. The Time Use Survey
occupancy agenda is the main casualty: its Saturday (22 occupied hours) and Sunday
(24) landed on the model's Thursday and Friday, so the household's weekend CO₂
peak appeared midweek.

`_set_run_period` now sets the start weekday from the weather year (taken from the
`weather_year` argument, or parsed from the EPW filename). Daily-mean indoor CO₂ by
weekday, before and after:

| | Mon | Tue | Wed | Thu | Fri | **Sat** | **Sun** |
|---|---|---|---|---|---|---|---|
| before | 509 | 489 | 499 | **562** | **572** | 494 | 500 |
| after | 494 | 498 | 509 | 489 | 501 | **562** | **572** |

The peak is unchanged in size — it simply sits on the weekend now, where the
agenda puts the household at home all day. `cte` is flat at 532–534 ppm on every
day of the week, before and after, because its profile is `For: AllDays`; it has
no weekday structure to misalign.

> This affected every weekday-dependent schedule, not only occupancy — so **results
> from before this fix are not comparable** with after.

### 🕐 Hourly indoor CO₂ with the air change that produced it

New report figure: a week of **hourly** CO₂ over the worst 7 days, with the total
outdoor-air change on a shared time axis below it, and the **weekday** on the
ticks (`scenario_report(year=…)`).

A daily mean cannot answer the question CO₂ raises — whether the air recovers
between occupied periods — because an overnight build-up followed by a morning
airing averages to the same number as a flat mediocre day. Reading a spike
straight down onto the ventilation at that hour is the point of the pairing, and
the weekday ticks make the weekly rhythm of the occupancy agenda legible, which is
what surfaced the calendar offset above.

### 🏢 RITE mechanical ventilation for offices, shops and the other non-residential uses

Non-residential zones had **no outdoor air at all** — only 0.5 ren/h of leakage —
because the ideal-loads systems carry `Outdoor Air Method: None`. A full shop in
the mild shoulder season, with its HVAC gated off, therefore reached **1 962 ppm
of CO₂ with every occupied hour above 1 000 ppm**. That was a modelling gap, not a
finding: real premises are mechanically ventilated, and RITE (RD 1027/2007,
IT 1.1.4.2.3) requires it.

Each occupied non-residential zone now gets a `ZoneVentilation:DesignFlowRate` of
type `Intake`, sized per occupant by RITE air-quality category
(`nonresidential_ventilation.litres_per_second_person`): **12.5 l/s·person** for
offices (IDA 2), **8.0** for commercial and other uses (IDA 3).

**It runs on the zone's OCCUPANCY schedule, not on HVAC availability.** That is
the point of the fix — premises are ventilated whenever they are open, including
the shoulder season when heating and cooling are off, which is exactly when the
CO₂ was worst.

On the reference building's ground-floor premises:

| | before | after |
|---|---|---|
| window/mechanical ventilation | 0.0 ren/h | **1.4–1.5** |
| total air change (median) | 0.5 | **1.9–2.0** |
| CO₂ median, in-between | 1 962 ppm | **859** |
| occupied hours above 1 000 ppm | 100 % | **0 %** |

Conditioning that outdoor air costs what it should: commercial heating demand
21.0 → 37.5 kWh/m². Dwellings are deliberately excluded — their renewal is
infiltration plus the windows the occupant opens, which the behaviour model
already carries, and the residential results are unchanged.

### 🧮 Occupancy blends the per-dwelling and per-area census estimates

How the census population is spread over a tract's dwellings had one estimator,
and it was the more fragile of the two. Now it is the average of both, weighted by
``loads.occupancy_area_weight`` (0.5):

```
people = max(1, round((1-w)·(population / tract dwellings)
                    +    w ·(population / tract residential area)·dwelling_area))
```

Each half is wrong alone. **Per dwelling** gives every home the tract's average
household whatever its size — a studio and a 200 m² flat house the same. **Per
area** scales with size but takes it too literally: household size does grow with
dwelling size, far more slowly than floor area does. On a tract of 2 565 people,
1 000 dwellings and 112 000 m²:

| dwelling | per dwelling | per area | blend |
|---|---|---|---|
| 40 m² | 2.57 | 0.92 | **2** |
| 90 m² | 2.57 | 2.06 | **2** |
| 132 m² | 2.57 | 3.02 | **3** |
| 200 m² | 2.57 | 4.58 | **4** |

Either half alone is used when the other is unavailable. `area_weight` 0 is purely
per-dwelling, 1 purely per-m².

> The 0.0054 people/m² that prompted this was **a fault in the test fixture, not
> in the code**: it forced all 2 662 buildings of a municipality (4 245 dwellings,
> 476 976 m²) into one census tract while giving it a single tract's population of
> 2 565. A correctly scoped tract yields ~0.023 people/m². A regression test now
> pins the blended density into 0.015–0.04 people/m² so an order-of-magnitude slip
> fails loudly.

### 👥 Every scenario houses the same people, the normative case included

Occupancy is a property of the building's neighbourhood, not of the occupant
archetype simulated in it. The `cte` case was given the CTE-default density
(`loads.residential.ppl_per_m2`, 0.028) while the behaviour scenarios used the
census — **3.5 people per dwelling against the census's 1** — so its results
differed from theirs for two reasons at once and the comparison no longer isolated
behaviour.

All scenarios now take the same head count, from the census-tract density applied
to each dwelling's own area:

```
people_in_dwelling = max(1, round(tract_population / tract_residential_area × dwelling_area))
```

`loads.residential.ppl_per_m2` becomes the **fallback** for a building whose tract
has no census population, not a per-scenario override. It drives the People
internal gains, the DHW demand and the CO₂ generation alike.

On the reference building every scenario now reports `people_per_dwelling` 1.0 and
the same `dhw_demand_kwh_m2` (0.50 heating / 0.42 cooling). What still separates
`cte` is behaviour and its own normative gain profile — internal gains 10.1
against 11.5 kWh/m², and night CO₂ 483 ppm against 467 because it never opens a
window, where before it read 638 ppm mostly for having 3.5× the occupants.

### 🔧 Infiltration recalibrated, one open-window rate, and a gains bug the CO₂ work exposed

**Infiltration halved to realistic seasonal means.** The per-period values were an
n50/10 reading; they are now n50/20, which is the mean rate a sheltered urban
building actually leaks:

| period | before | now | n50 |
|---|---|---|---|
| pre1980 | 1.4 | **0.80** | 16 |
| 1980–2006 | 1.1 | **0.55** | 11 |
| 2007–2013 | 0.55 | **0.35** | 7 |
| post2013 | 0.35 | **0.25** | 5 |

This was suppressing the whole air-quality question: at 1.1 ren/h the dwelling is
flushed so hard that no occupant behaviour can move the CO₂, so the KPIs were flat
by construction rather than by result. They now discriminate — `cte` reaches
**638 ppm** at night (its CTE-default 3.5 people per dwelling, windows shut) while
`aware` drops to **426 ppm** on a summer night purge. Heating demand falls with the
leakage (`aware` 29.0 → 23.1 kWh/m²), so **results from before this change are not
comparable**.

**One open-window rate.** Hygiene, the midday free-cooling airing, the summer
night purge and the unaware occupant's random bursts all mean "the windows are
open", and now all use `residential_behavior.opening_ach` (4 ren/h). What
distinguishes the behaviours is *when* and *for how long* they open, not how wide
— the night purge was at 10 ren/h and the midday airing at 3, for no reason but
their own history.

**A gains bug this exposed.** Setting `Sleeping: 1.0` (correct: a sleeping
household is fully home) also put the LIGHTS and APPLIANCES at full power all
night, because People, Lights and Equipment shared one schedule. That is what made
the aware/unaware internal gains overtake the normative CTE ones despite having a
third of its occupants — a result that should have looked wrong immediately.

Gains now have their own `state_gain_fractions` (Sleeping 0.10, WithActivity 1.0,
NonOccupied 0.15): they follow what the household is *doing*, while presence
follows how many are *home*. An empty dwelling keeps a small fraction because the
fridge runs. Residential internal gains 18.9 → **11.5 kWh/m²**, and the ordering is
right again: `cte` 13.9 with 3.5 people per dwelling, above the census-driven 11.5
with 1.0.

**`n_dwellings`** is now a KPI column and a tile in the report, beside
`people_per_dwelling`.

### 🪟 An open window now delivers the air change it is configured with

EnergyPlus computes a `ZoneVentilation` rate as
`design_ach × schedule × (constant + velocity × wind_speed)`. The coefficients
were `(0.15 + 0.20·V)`, so a window nominally at 4 ren/h delivered **0.6 ren/h in
calm air** and reached its nominal only at 4.25 m/s of wind. The air change
reported therefore bore no resemblance to the number configured for it — the
ventilation column read 0.1–0.2 where an open window belongs.

The default is now `constant: 1.0, velocity: 0.0`
(`residential_behavior.ventilation_wind`): these `ach` values are open-window
figures already, and a rate you cannot read off the config is worse than one that
ignores the breeze. Set `velocity` above zero, with `constant` reduced to match
your site's mean wind, to bring the modulation back.

Peak air change during airing, occupied day hours:

| | before | after |
|---|---|---|
| `aware` | 1.8 | **3.3** |
| `unaware` | 2.4 | 2.2 |
| `cte` | 1.2 | 1.2 |

(3.3 rather than 5.1 because the output is an hourly mean and the window is open
for half of the reported hour — 1.1 infiltration + 4 × 0.5.) Airing for real costs
heat: `aware` heating demand 26.1 → 29.0 kWh/m².

**The report gains an air-change distribution chart**, drawn like the indoor
temperature one — min–max hairline, p10–p90, p20–p80, median marker. It is the
right form here: the median sits at the infiltration rate because most hours have
the windows shut, and the whisker is the airing. A mean collapses those into one
number that describes neither.

> On infiltration: it reads 1.1 ren/h for every occupant, identically, and that is
> correct — infiltration is fabric, not behaviour, and only the ventilation column
> is the occupant's. Whether **1.1** is the right value for a 1980–2006 envelope is
> a separate calibration question: it is a flat design rate, not a wind-driven one,
> and typical mean infiltration for that period is nearer 0.5–0.7 ren/h. It is in
> `windows.infiltration.<period>.air_changes_per_hour` and dominates the CO₂
> results, so it is worth a decision before publishing.

### 💨 Air-renewal KPIs — and the aware occupant was ventilating LESS than the unaware one

**New `ach_*` columns**, reported like the temperature ones: `ach_total_p{10,50,90}_*`
and `ach_total_{min,max}_*` (percentiles, because an hourly mean hides a dwelling
sealed for twenty hours and flung open for one), plus `ach_infiltration_mean_*`
and `ach_ventilation_mean_*` — the split between the fabric, which is the same for
every occupant, and the windows, which are the only part behaviour controls. Day
and night, over occupied hours. Two charts in the report.

Both `ZoneInfiltration` and `ZoneVentilation` draw **outdoor** air in EnergyPlus
and so enter the contaminant balance at the outdoor concentration — the dilution
was already working, and the new columns show it directly: where ventilation is
highest (aware, cooling night, 3.2 ren/h on top of 1.1 infiltration) CO₂ sits at
426 ppm, barely above the 420 ppm background; where it is zero and the dwelling is
full (`cte`, heating night) it reaches 538 ppm.

**What the columns exposed.** In the HEATING season the aware occupant had
`ach_ventilation_mean_day` = 0.0 against the unaware occupant's 0.1 — the careful
user was getting *less* outdoor air than the careless one, and the CO₂ said so.
Two causes, both in the aware profile:

- the hygiene airing was 15 minutes at 3 ren/h = 0.75 ren·h/day, against the
  unaware occupant's three random bursts at 3 ren·h/day;
- the midday airing that was meant to make up the difference is gated on outdoor
  > 20 °C, which in a Lleida January essentially never fires.

The hygiene airing is an **air-quality** habit, not a thermal strategy, so it is
now sized to do something and is ungated — no season, no outdoor-temperature
condition, because a Spanish household airs the flat daily whatever the weather.
New `residential_behavior.hygiene_ventilation` (4 ren/h, 08:00–08:30 and
20:00–20:30 = 4 ren·h/day). The free-cooling airings keep their gates; those
genuinely are thermal strategies.

After the change, aware overtakes unaware on outdoor air
(`ach_total_p90_day` 2.0 vs 1.6) and pays for it in winter heat: heating demand
24.6 → 26.1 kWh/m². That trade is the point — it is what airing a flat in January
costs.

### 📆 The season split now uses a 7-day moving average

Each day is still classified from its mean outdoor temperature, but the smoothing
window goes from 3 days to a **centred week** (`seasons.smoothing_days`, with the
band as `seasons.in_between_low_c` / `_high_c`).

Three days is short enough that a warm spell still moves the classification: on a
synthetic March with three 26 °C days inside a 12 °C month, the 3-day window
produces a **cooling** day, and the results for that day are then reported under
the cooling season of a heating month. A week absorbs it — no cooling day at all —
while still turning over within a few days of a real change of season.

On the reference building's 2021 weather the year moves from
192/96/77 heating/in-between/cooling days to **194/89/82**, and each season lands
in a handful of contiguous runs rather than being speckled.

This changes which season a given hour's results are reported under, so **KPI rows
from before this change are not directly comparable** with after. The occupant
scenarios' HVAC in-between gate follows the same classification, so their demand
shifts slightly too; the normative `cte` case is unaffected, since it follows its
own calendar.

### ⚖️ The normative CTE case follows the standard's calendar, with no in-between season

**The in-between season does not belong in the normative case.** CTE DB-HE
Apéndice D divides the year by **calendar date** and leaves no hour unassigned:
every hour is either heating-enabled or cooling-enabled. The "in-between season"
this model gates the HVAC off in — mild weather, nobody touches the controls — is
a description of *behaviour*, correct for the occupant scenarios and wrong for a
compliance case. The `cte` zones were inheriting it, so the standard's user simply
stopped conditioning for 44 days of the simulated period.

The CTE case now has its HVAC **available all year**, with the out-of-season
service made unreachable by its setpoint (EnergyPlus offers no way to disable one
side of a `HVACTemplate:Thermostat`). And within each season the standard
prescribes a shape rather than a constant, which the flat 21/26 °C did not carry:

| | months | profile |
|---|---|---|
| heating | Jan–May, Oct–Dec | 17 °C 00–07 h, 20 °C 07–24 h |
| cooling | Jun–Sep | 27 °C 00–16 h, 25 °C 16–24 h |

Effect on the reference building (Jan + Apr + Jul), `cte` only — the occupant
scenarios are untouched:

| scenario | in-between demand | in-between HVAC hours |
|---|---|---|
| `aware` | 0.00 kWh/m² | 0 |
| `cte` | **1.55 kWh/m²** | **154.9** |

All of this lives in `residential_cte.seasons` / `.setpoints`.

> ⚠️ The **structure** above is what the standard prescribes — calendar seasons,
> a heating night setback, a tighter cooling consigna for the occupied evening.
> The **numbers** are the widely-cited 2019 residential profile and are worth
> checking against the current Apéndice D before publishing results; they sit in
> the config so a correction is a one-line edit.

### 🌬️ Indoor CO₂ — how renewed the air is

`ZoneAirContaminantBalance` is switched on, so every zone reports
`Zone Air CO2 Concentration` and the KPI frame carries, per season and over the
**occupied** hours, `co2_p50_*`, `co2_p90_*`, `co2_max_*` and
`co2_pct_above_{1000,1400}_*` — day and night. The report gains a median-CO₂
chart, a "share of occupied night hours above threshold" chart, and the daily CO₂
series in the raw section. Configure under `co2` (outdoor background, generation
rate, thresholds).

CO₂ is the air-**renewal** indicator: occupants raise it, infiltration and open
windows flush it, so one number reads the census occupancy and the occupant's
ventilation behaviour together.

**Day and night generation differ, through the metabolic rate.** EnergyPlus
derives a person's CO₂ output from the ACTIVITY LEVEL —
`People.Carbon_Dioxide_Generation_Rate` is m³/s per *watt* — so a sleeping body
emits less heat and less CO₂ from one schedule. Which hours are asleep comes from
the Time Use Survey agenda where the building has one (its `Sleeping` state
becomes `loads.residential_activity_sleeping`, 72 W ≈ 0.8 met, against 110 W
awake) and from the 23:00–08:00 night band otherwise.

**A double-discount this exposed.** `time_use_occupancy.state_fractions` had
`Sleeping: 0.35`, but `Number_of_People_Schedule_Name` scales the people
**count** — that modelled a third of the household as absent overnight. With the
sleeping metabolism now carried by the activity level, discounting there as well
applied the reduction twice: night CO₂ came out ~3× too low and the night gains
with it. `Sleeping` is now **1.0** — a sleeping household is fully home. On the
reference building, night CO₂ moved 429 → 444 ppm and residential internal gains
12.4 → 18.9 kWh/m².

> Worth reading in context: on this building CO₂ stays near 450 ppm against a
> 420 ppm background, because the 1980–2006 envelope leaks at 1.1 ACH and the
> census puts about one person in each dwelling. That is a result, not a bug —
> at that infiltration rate ventilation is never the binding constraint. The
> normative `cte` case reaches 538 ppm at night, on its CTE-default occupancy of
> 21 people rather than the census's 6.

### 🛠️ `run_behavior_scenarios(idf_path=…)` — hand-edit the model between build and run

The scenarios can now be driven from an ``.idf`` **on disk** rather than an
in-memory object, so a model can be generated, adjusted by hand or by another
tool, and then simulated with those adjustments intact:

```python
idf = hc.energy.build_energy_model(gdf, ref, wd=wd, weather_year=2021)
# …edit the .idf however you like…
kpis = hc.energy.run_behavior_scenarios(one_building, wd=wd, idf_path=idf.idfname,
                                        weather_year=2021)
```

`idf_path` takes precedence over `idf`, so the file is unambiguously what runs.

**This needed a sidecar.** The zone → use-type map lives on the IDF object as
`_hc_zone_meta`; EnergyPlus has no object to hold it and eppy drops stray comments
on save, so a plain reload came back with no idea which zone is a dwelling and
would have simulated nothing. `build_energy_model` now writes
`{model}.idf.hcmeta.json` beside every model it saves (zone metadata, the building
reference, the attached EPW, and whether the shading was already filtered), and
`load_energy_model(path)` reads both back. Keep the pair together; if you rename a
zone in the IDF, rename it in the sidecar too — an unmapped zone is **skipped, not
guessed at**, so a rename can never quietly turn a shop into a dwelling.

Verified end to end: generating a model, tripling every zone's infiltration in the
file, and running `idf_path=` moved January heating demand from ~31 to 38.8
kWh/m² — the edit is genuinely in the simulation.

### 🌡️ Temperature extremes, richer spread chart, worst-week detail, CSV output

- **`t_min_*` / `t_max_*`** join the percentiles on every KPI row. A band cannot
  say whether a dwelling ever touched 12 °C, and a compliance argument usually
  turns on exactly that — the one freezing night the p10 never reaches.
- The report's **temperature spread** chart now draws the whole distribution:
  min–max as a hairline, p10–p90, p20–p80, and the median as a marker, so a tight
  `cte` box and a sprawling free-floating one are distinguishable at a glance.
- **Worst cooling week AND worst heating week**, each at hourly resolution with
  indoor temperature above and HVAC demand below on a shared time axis. Two
  stacked panels rather than a second y-scale: with two scalings on one plot the
  eye reads crossings the data does not contain, and the apparent relationship
  changes with whichever scaling happened to be picked.
- **`output_path_html` and `output_path_csv`** on `scenario_report`, either or
  both. `output_path` still works as an alias.

### ☀️ The CTE case no longer deploys its awnings unconditionally

CTE DB-HE specifies the solar protection by **incident radiation**, all year — not
"deployed whenever it is daylight". The radiation trigger is what the persiana's
`WindowShadingControl` does (`OnIfHighSolarOnWindow`, ungated for the normative
case), but the awnings were *also* pinned at τ=0.4 through every daylight hour of
the year. That applied the normative protection twice and, being unconditional,
shaded the glazing on midwinter mornings when no control would have closed
anything.

The awnings are now retracted for `cte`; its solar protection is the
radiation-triggered persiana alone. `SHADE_CTE` is still authored, so the old
blanket-daytime behaviour is one assignment away for anyone who wants it back.

### 📊 `reporting.scenario_report` — an interactive HTML report of a simulation

New module `hypercadaster_ES.reporting`. One call turns a scenario run into a
self-contained page:

```python
kpis = hc.energy.run_behavior_scenarios(one_building, wd=wd, idf=idf, weather_year=2021)
hc.reporting.scenario_report(
    kpis, output_path=f"{wd}/{ref}_scenarios.html",
    output_dir=f"{wd}/energyplus/scenarios",     # adds the raw hourly section
    building_reference=ref, zone_areas=hc.energy._zone_floor_areas(idf))
```

**Summary half** — headline tiles (the demand spread between occupants), then a
chart per KPI: thermal demand, solar and internal gains, delivered setpoint,
day/night discomfort, DHW, and the indoor-temperature p10–p90 bands with the
median marked, all grouped by season with the scenarios as the comparison.

**Raw half** — the hourly EnergyPlus output each run actually produced: daily
indoor temperature against the outdoor dry bulb, daily HVAC energy, daily
transmitted solar, and an hourly window over the hottest days, where the
behaviours diverge most. Read from the per-scenario `eplusout.csv` the runner
leaves behind, so a summary that looks wrong can be traced to the year.

Plotly is inlined — the page opens from disk with no network and survives being
emailed. Notes on how it is drawn, since they are decisions rather than defaults:

- A **validated categorical palette**, assigned to scenarios in fixed order so a
  scenario keeps its colour across every chart and between runs. Both light and
  dark steps are checked (lightness band, chroma floor, CVD separation,
  normal-vision floor) on the adjacent pairlist these bar and line forms use.
- **Free-floating twins are dotted**, not just differently coloured: an `_ff` run
  often traces its conditioned partner exactly (identical shading, identical
  gains), and one line would otherwise hide the other completely.
- The building temperature is **floor-area weighted** across zones, not a plain
  mean of unequal rooms.
- Dates are plotted as **categories**. Plotly reads `"01-02"` as a date and labels
  the axis in years (2001, 2002 …); a category axis also draws a non-contiguous
  run period — simulating January and July — honestly.
- Every bar is directly labelled and the full KPI table is included, which is the
  relief the palette's contrast check requires and the accessible view of the
  charts.

### 🌡️ Indoor-temperature percentiles, one setpoint per row, CTE back to the standard

**Temperature percentiles over the occupied hours.** Each row now carries
`t_p10/p20/p50/p80/p90_day` and `…_night` — the indoor operative temperature the
occupants actually saw, split at 22:00–07:59 like the discomfort percentages. The
spread is what an average hides: on the reference building a free-floating
January night sits at `t_p10_night` = 8.9 °C while the CTE user never leaves
19.5–20.8 °C, and the two look far more alike through a mean. Percentiles cannot
be pooled exactly across zones without the raw hours, so each zone's percentile is
area-weighted, the same treatment the discomfort percentages get.

**The heating/cooling column pairs are gone.** A row is one season, so
`heating_demand_kwh_m2` next to `cooling_demand_kwh_m2` left one of them
meaningless in every row. What remains is season-agnostic:

| removed | kept |
|---|---|
| `heating_demand_kwh_m2`, `cooling_demand_kwh_m2` | `thermal_demand_kwh_m2` |
| `avg_heating_setpoint_c`, `avg_cooling_setpoint_c` | `avg_setpoint_c` |
| `heating_hours`, `cooling_hours` | `hvac_hours` |

`avg_setpoint_c` uses the HEATING setpoint in the hours the zone heated and the
COOLING one in the hours it cooled, weighted by those hours — so an in-between
season that saw both reports the blend rather than two half-empty columns.

**The CTE shadow control follows the standard again.** CTE DB-HE specifies it by
incident radiation all year, so the normative case is no longer season-gated —
unlike the occupant behaviours, which never shade in the heating season. The
difference is visible and intended: January solar gains are 17.93 kWh/m² for every
occupant behaviour and 6.38 for `cte`.

### 🪟 The persiana is a real shade material on a WindowShadingControl

The exterior roller blind was a `Shading:Building:Detailed` panel pasted 3 cm in
front of the glazing. That only removes the beam its own shadow happens to catch
and lets the sky diffuse straight through, so even a **fully closed** persiana
topped out at ~52 % of the transmitted solar — no schedule could make it behave
like a closed blind.

It is now a `WindowMaterial:Shade` layer driven by one `WindowShadingControl` per
residential zone (`ExteriorShade`, all the zone's windows listed). As a layer it
attenuates beam and diffuse alike and enters the window heat balance through its
air gap and long-wave properties. Held closed, it now blocks **92.3 %** of the
transmitted solar; under its normal control it blocks 46.5 % in July. Configure it
under `shading.persiana` (optical properties, air gap, openness, solar setpoint).

**The control never runs in the heating season.** It is `OnIfHighSolarOnWindow`
*and* scheduled, which in EnergyPlus means the persiana closes only when the
schedule allows **and** the sun on the window exceeds `solar_setpoint_w_m2`
(140 W/m² default). The schedule is the season gate: 0 through every heating day,
1 through the cooling and in-between seasons — in a Spanish winter that solar gain
is wanted, so nothing that shades the glazing operates. The **awnings** follow the
same rule (they used to retract in the in-between season too, and now deploy
there), as does the normative `cte` shadow control.

Measured on the reference building, January solar gains are now **identical
(17.93 kWh/m²) for every occupant behaviour**, while July separates them:

| scenario | July solar kWh/m² | July cooling kWh/m² |
|---|---|---|
| `unaware` (never touches them) | 6.85 | 15.85 |
| `random_mix` | 5.69 | 9.54 |
| `aware` | **3.79** | **1.75** |
| `cte` | 3.70 | 11.03 |

Two details worth knowing. The season gate the scenarios use is a **separate
schedule** (`PERSIANA_ALLOW_SEAS`) from the one authored at build time: the build
happens before the EPW is attached, so it can only use a month split, while the
seasons the KPIs report come from the weather. Sharing one name silently kept the
month version — which left the persianas operating on days the results called
heating, and was visible as a 12 % solar difference between `aware` and `unaware`
in the heating season. And the coincident `Blind_*` panels are gone: the geometry
they occupied is now a material property, so nothing has to be trimmed, oriented
or kept out of the shading cap.

### 🧭 Walls are now wound outward at build time

The root cause behind the façade shading: the surface builder emitted wall
vertices in a fixed order, `(x0,y0,z0) → (x1,y1,z0) → (x1,y1,z1) → (x0,y0,z1)`,
whose normal is outward only when the segment happens to be walked in the
direction of the room's counterclockwise ring. `_build_surfaces_topological`
takes its segments from the cell complex, whose face lines run in arbitrary
directions, so the winding came out inward on most walls — **0 of 195 walls faced
out of their own room** on the reference building.

Every wall is now wound against **its own room's plan polygon**
(`_wall_quad_outward`): step off the segment midpoint along the candidate normal,
and if that lands inside the room, emit the other order. Both builders use it, and
paired interior faces stay exact mirrors. After the fix, **195 of 195 walls face
outward**, and the balconies, awnings, exterior blinds and window reveals that
hang off those normals all sit outside the room — verified per element, not in
aggregate:

| element | before | after |
|---|---|---|
| exterior blinds outside the room | 30 of 45 | **45 of 45** |
| awnings outside the room | some | **10 of 10** |
| window reveals outside the room | 0 of 204 | **204 of 204** |

This is the upstream fix for the shading bugs below, and it removes the need for
the downstream guesswork: `_oriented_outward_normal` now just reads the winding.
Two heuristics that were tried and **measured to be worse** are recorded there so
they are not reintroduced — orienting against the zone centroid inverted 46 of 49
walls, and against the zone's floor surfaces inverted all of them (the floor
spans the un-shrunk footprint while the walls are inset by the CAT shrink, so
"inside the floor polygon" is not "inside the room").

> ⚠️ Wall winding changes the geometry EnergyPlus sees, so **existing `.idf` files
> and cached results are not comparable with new ones**. Rebuild them.

### 🪟 Awnings and exterior blinds were doing nothing — two bugs, both fixed

Every occupant type returned identical results because the operable façade shading
never reached the simulation. Two independent faults:

**1. The shading cap was deleting them.** `_trim_shading` sorted *every*
`Shading:Building:Detailed` panel by size and dropped the smallest past the cap —
and the smallest panels in any model are the window-sized blinds and awnings. On
the reference building that left exactly 200 panels: 197 neighbour + 3 balconies,
with all 45 blinds and 8 awnings deleted. `_apply_occupant_control` then had
nothing to deploy, so `aware`, `unaware` and `cte` produced the same numbers.

The cap now applies **only to neighbour panels** (`Shade_F*`), which is what
`max_shading` always meant and what the docs already said. The building's own
shading — balconies, awnings, blinds, window reveals, footprint remainders — is
part of the model, never context detail to be budgeted away. This bit hardest
after the filter moved into `build_energy_model` with the config default of 200;
the old example passed `max_shading=100000`, which is why it had not shown up.

**2. The shades were being placed on the wrong side of the wall.**
`_wall_outward_normal` flipped the surface winding to point away from the ZONE
CENTROID — but the centroid of an L-shaped or wrapped-around zone can sit on the
far side of its own wall, so the rule inverted 46 of 49 walls. Balconies, awnings
and blinds ended up **inside** the dwellings, where EnergyPlus dutifully simulated
them shading nothing.

Verified against EnergyPlus rather than by inspection: large opaque panels placed
along the surface winding block **61 %** of transmitted solar, the same panels on
the opposite side block **1.3 %**. The winding is therefore the outward direction,
and it is now trusted, with the building's convex hull — not a zone centroid — as
the tie-breaker for the walls whose winding is genuinely reversed (a step outward
must leave the envelope; a patio wall, where both steps stay inside, keeps its
winding).

Fully opaque blinds went from blocking **1.7 % → 59.5 %** of transmitted solar.
The scenarios now separate as intended (July, residential):

| scenario | solar gains kWh/m² | cooling kWh/m² |
|---|---|---|
| `unaware` (never operates them) | 6.52 | 12.25 |
| `random_mix` (half and half) | 5.55 | 8.68 |
| `cte` (normative daytime τ=0.4) | 4.56 | 10.39 |
| `aware` (blinds by orientation + awnings) | **4.26** | **3.85** |

### 📐 Awning geometry: angle measured from the vertical

`shading.awning.inclination_deg` (below the horizontal) is replaced by
**`inclination_from_vertical_deg`, default 75°** — 90° is a horizontal canopy, 0°
a fabric hanging flat down the glazing, so a Spanish retractable awning sits well
open, reaching far out and dropping little. The old key is still honoured, so an
existing config keeps meaning what it did.

### 🗓️ `season_ranges` replaces `season_start` / `season_end`

Seasons are classified from daily temperature and genuinely break into several
stretches, so a first/last pair described a Spanish heating season as spanning
January to December. Each row now carries **`season_ranges`**: the contiguous runs
as `DDMM/DDMM`, e.g. `"0101/0401, 0111/3112"` for a heating season wrapping the
turn of the year, or `"0806/1408"` for a summer. Runs join across month
boundaries and across 31/12 → 01/01.

### 📋 Richer scenario KPIs: delivered setpoints, season dates, solar gains, household size

`run_behavior_scenarios` now returns, per use type × season:

| Column | |
|---|---|
| `season_start` / `season_end` | the `MM-DD` calendar span of the season, alongside its day count. Seasons are temperature-derived and need not be contiguous, so this is the outer envelope of the days that fell in them — and it follows the simulated run period |
| `avg_heating_setpoint_c` / `avg_cooling_setpoint_c` | mean thermostat setpoint over the hours that service was **actually running** |
| `heating_demand_kwh_m2` / `cooling_demand_kwh_m2` | the split behind `thermal_demand_kwh_m2` |
| `solar_gains_kwh_m2` | solar transmitted through the glazing, separate from the people/lights/equipment `internal_gains_kwh_m2` |
| `people_per_dwelling` | the occupancy the run assumed, next to the `dhw_demand_kwh_m2` it produces |
| `heating_hours` / `cooling_hours` | per zone, the weights behind the setpoint means |

**Setpoints are averaged over running hours, not over the schedule.** Averaging the
schedule would mix in the setback and the free-floating hours, which is not a
temperature the building was ever held at. The distinction is the point of the
column: the scenarios move the setpoint around (EN 16798-1 adaptive band, income
offset, unoccupied setback, CTE constant), so what a run was actually driven by is
not readable from the config. On the Lleida building in the heating season:
`cte` 21.00 °C exactly, `unaware` 20.11 (21 less the income offset), `aware` 18.99
(adaptive plus setback), the shop 16.60 (RITE ≤ 19 with setback). Across zones the
mean is weighted by running hours, so a zone that heated for an hour does not
count as much as one that heated all season.

**Solar gains needed an EnergyPlus-version fix to work at all.** The variable was
renamed `Zone Windows Total Transmitted Solar Radiation Energy` →
`Enclosure Windows …` in 23.1, when radiant enclosures were split from thermal
zones. Requesting the old name on a modern EnergyPlus produces a warning in
`eplusout.err`, no column in the CSV, and a silent **0.0** in the KPIs — which is
exactly what the first run of this feature reported. The name is now picked from
the IDD version, and either spelling is read back, so old result CSVs still parse.

### 🛋️ Out-of-comfort hours: the occupied denominator is now visible and enforced

`pct_hours_out_of_comfort_day` / `_night` have always been taken over the
**occupied** hours only (`occ & season & day`, occupancy from the hourly
`Zone People Occupant Count`) — on the Lleida test building that is 49.6 % day
discomfort over occupied hours versus 37.1 % if every hour were counted. Two
changes make that trustworthy rather than merely true:

**The denominator is reported.** New `pct_occupied_hours` on every KPI row: the
share of the season's hours the comfort percentages rest on. It makes the basis
auditable and exposes a real caveat — the normative `cte` user reads **100 %**,
because `_CTE_OCC_PROFILE` never drops to zero (CTE DB-HE assumes the dwelling is
never empty), while the behaviour scenarios read ~77 % and a shop ~51 %. The `cte`
comfort figures are therefore over *all* hours by construction, which is worth
knowing before comparing them with the others.

**A zone with no occupancy output no longer counts as always occupied.** It used
to fall back to "every hour is occupied", which quietly answered a different
question under the same column name. Such a zone now yields NaN comfort
percentages and a warning, and is excluded from `pct_occupied_hours` on both
sides — it reports no comfort, so it must not deflate its use type's occupied
share either (a two-zone commercial use read 25.7 % instead of the correct
51.3 % when one zone had no `People` object).

### 🏪 Non-residential uses are reported once, under the normative case

The behaviour scenarios describe **residents**. Office and commercial premises run
the same normative user in every one of them — the RITE / RD 486-2022 tertiary
restriction (`_apply_nonresidential_setpoints`) — so emitting a `Commercial` row
under `random_mix`, `unaware`, `aware`, `cte`… read as if the shop had six
occupant behaviours of its own. It never did.

Those rows were not identical, which made it worse: the premises are thermally
coupled to the dwellings around them (68 interzone floor/ceiling surfaces on the
Lleida test building), so the shop's demand drifts a little with what the
residents do while its own setpoints and schedules never change. The spread was
easy to misread as an occupant effect on the tertiary premises.

Non-residential use types now appear **once**, under the scenario label
`normative`, taken from the `cte` run — the one where the dwellings around them
are normative too (`NONRESIDENTIAL_SOURCE_SCENARIO`; the first conditioned
scenario when `cte` is not being run, never a free-floating one, whose zero demand
would be a nonsense normative figure). Residential rows are unchanged.

```
  scenario    use_type  ...
random_mix Residential  ...
   unaware Residential  ...
     aware Residential  ...
       cte Residential  ...
 normative  Commercial  ...     <- once, not per scenario
```

Pass `nonresidential_reporting="per_scenario"` to restore the previous output.

### 👥 Census occupancy drives the internal gains, not just the DHW

The census-tract occupancy now sets the **`People` internal gains** of residential
zones, where it previously only fed the DHW demand — so metabolic heat, the
cooling load that follows from it, comfort, and hot water all describe the same
people. A building carrying `ctagg__people_per_m2` takes its `People/Area` from
it; one without keeps the `loads.residential.ppl_per_m2` default (0.028 people/m²,
CTE DB-HE). New `loads.residential_occupancy_from_census` (default `true`) turns
it off.

The density is derived from the per-dwelling head count, not passed straight
through: each dwelling gets `max(1, round(density × its area))` occupants — a
dwelling is never modelled as empty — and the zone density is that count over the
residential floor area. So the gains and the DHW are the same people, and the
min-1 floor lifts the effective density above the tract average for small
dwellings (0.0054 → 0.0081 people/m² on the Lleida test building).

The normative **`cte` scenario keeps the standards density**: it restores
`loads.residential.ppl_per_m2` over any census-derived value, exactly as its DHW
already used the CTE-default head count. It is a compliance case, not a
description of who lives there.

Measured on the test building (July, residential zones), census occupancy versus
the CTE default:

| | CTE default | census |
|---|---|---|
| internal gains | 3.62 kWh/m² | **2.93** |
| cooling demand (unaware) | 12.64 kWh/m² | **12.15** |
| DHW | 1.34 kWh/m² | **0.38** |
| `cte` scenario | 3.15 / 12.25 / 1.34 | **unchanged** |

**Removed: `residential_behavior.m2_per_person`.** It was documented as the
"fallback people/home" but the value was read into a local and never used — an
inert setting that looked like it controlled occupancy. The census occupancy above
is what actually determines it now.

### 🧮 `ctagg__*` — census-tract aggregations computed in `hc.merge`

`hc.merge` now aggregates the frame **by census tract** and writes the result onto
every building (`census_tract_aggregations=True`, on by default, a no-op without
the inputs): tract building and dwelling counts, summed residential floor area,
population, and the occupancy densities `ctagg__people_per_m2` /
`ctagg__people_per_dwelling`. See the output schema for the full list.

These are properties of the *tract*, not of the building, and the energy model
needs them. Computing them at merge time — while every building of the tract is
in hand — is what makes each row self-sufficient afterwards:

```python
one = gdf[gdf["building_reference"] == ref]      # carries its tract's density
hc.energy.run_behavior_scenarios(one, wd=wd, weather_year=2022)   # DHW is right
```

`energy._dwelling_occupants` prefers `ctagg__people_per_m2` over anything it could
re-derive, so `context_gdf=` is no longer needed for a frame from a current merge.
It stays as the fallback for frames merged before this existed.

**Partial frames publish no density.** `ctagg__dwelling_coverage` measures the
frame's dwellings against the tract's 2021-census main dwellings; below 50 %
(`mergers.CTAGG_MIN_COVERAGE`) the density columns are left NaN and
`ctagg__complete_tract` is False, rather than emitting a figure inflated by
however much of the tract is missing. The counts are still emitted. Without the
census yardstick the frame is assumed to hold the tract — what a
whole-municipality merge does.

### 🏠 `run_behavior_scenarios` takes the ONE building's frame

**Breaking signature change.** The building is now identified by the frame you
pass, not by a separate argument:

```python
# before
hc.energy.run_behavior_scenarios(gdf, ref, wd, …)
# now
hc.energy.run_behavior_scenarios(gdf[gdf["building_reference"] == ref], wd, …)
```

`building_reference` survives as a keyword for the case where the frame really
does hold several buildings. Without it, an ambiguous frame raises instead of
silently taking the first row. The old positional order is detected (its third
positional was a path where the new one is a scenarios dict, so there is no
ambiguity) and still works, with a `DeprecationWarning`.

**`context_gdf=` — and a guard that stops a one-row frame corrupting DHW.** The
census-tract DHW occupancy is the tract population divided by the residential
floor area of *every* building in the tract, so it reads the whole frame, not
just the simulated building. Handing it one row made the denominator that
building alone: on a Lleida test building the rate came out at **3.48 people/m²,
putting all 2 565 people of the tract inside one 630 m² block** — a ~120× DHW
overstatement that succeeded silently.

`_census_tract_person_rate` now rejects rates outside a plausible band
(`_DHW_PLAUSIBLE_RATE`, 0.005–0.10 people per m² of dwelling; the CTE default is
0.028) as evidence of a truncated frame, and those tracts fall back to the
CTE-default density. This protects every caller, not just the one-row one — a
frame filtered to a district or a sample was always vulnerable to it.

Pass `context_gdf=` (the municipality/tract frame) to keep the population-driven
figure. Measured on the same building: `dhw_demand_kwh_m2` 0.38 with the context
(6 people, from the tract population) versus 1.34 falling back to CTE (21
people). Thermal demand is unaffected either way, and the normative `cte`
scenario always used the CTE density, so it does not move.

### 🏗️ Shading filtered in the IDF generator; one IDF drives every scenario

**The neighbour-shading filter moved into `build_energy_model`.** It used to run
inside `run_behavior_scenarios`, just before each simulation — so the `.idf` on
disk and anything `plot_idf_geometry` drew still carried the full, unfiltered set
of context panels, while EnergyPlus saw a much smaller one. What you inspected was
not what you simulated. Now the filter and the cap are applied as the model is
built, before it is written or returned:

```python
idf = hc.energy.build_energy_model(gdf, ref, wd=wd, max_shading=200)
hc.energy.plot_idf_geometry(idf)     # the panels here ARE the simulated ones
```

New `filter_shading` / `max_shading` arguments on `build_energy_model`, defaulting
to the new `shading.neighbours` config block (`filter: true`, `max_panels: 200`,
`sun_exclude_halfwidth`, `occlusion`). The returned IDF carries
`_hc_shading_filtered`, so `run_behavior_scenarios` does not redo the work; its own
`filter_shading`/`max_shading` still apply when it builds the model itself, or when
handed an IDF from an older release.

**`run_behavior_scenarios(..., idf=…)`** takes an already-built model and runs each
scenario on a *copy* of it, instead of rebuilding from `gdf` six times. The
scenarios then share one geometry, envelope and shading **by construction** rather
than by trusting six independent rebuilds to agree, and the build cost is paid
once — measured at 57.2 s → 29.5 s for a six-scenario July run of one Lleida
building, with **byte-identical KPIs** to the rebuild path.

Only the behaviour-dependent wiring is re-applied per copy (occupant archetype,
setpoints, HVAC availability, ventilation), so scenarios cannot contaminate one
another. `_apply_residential_behavior` now seeds its "already made" set from the
objects the model carries, so re-applying it reuses the named schedules and
thermostats instead of authoring duplicates EnergyPlus would reject.

Copies go through the IDF text (`_copy_idf`) rather than `copy.deepcopy`, which
would duplicate or choke on the shared IDD tables; the `_hc_*` bookkeeping and the
attached EPW are carried over explicitly.

**Per-scenario `.idf` files now match what ran.** They were written by
`build_energy_model` *before* the scenario rewiring, so the file next to a
scenario's results described the pre-scenario model. They are now saved after the
rewiring, so `{ref}_{scenario}.idf` is the model EnergyPlus executed.

### 🕒 Time Use Survey occupancy agenda for the residential scenarios

The `aware` and `unaware` residential dwellings of
`energy.run_behavior_scenarios` now take their occupancy from the **INE Time Use
Survey** (EET 2009-2010) instead of a stylised profile. New
`social.time_use_household_occupancy(wd, census_tract_code)` wraps
`social_ES.INE.TimeUseSurvey` → `WeeklyScheduleHouseholdOccupancy` and returns the
census tract's **weekly household occupancy agenda**: the dominant state at each
of the 168 hours of the week — `Sleeping`, `WithActivity` or `NonOccupied` — for
the survey cell of the tract's autonomous community and household income band,
plus the per-state household shares.

The agenda drives three schedules per survey cell, all still season-gated:

- **occupancy** (People / Lights / Equipment) — a gains fraction per state
  (`Sleeping` 0.35, `WithActivity` 1.0, `NonOccupied` 0.0 by default): an hour
  asleep is at home but at rest, so it sits between the two extremes;
- **`aware` HVAC availability** — on while someone is in the dwelling, which is
  what "condition only when home" means;
- **`unaware` HVAC availability** — on through every waking hour, at home or not,
  which is what "leaves the HVAC on" means.

That last split is deliberate: both behaviours read the same measured week, but
the *availability* difference is what the archetype is, so it survives. Because
the survey resolves all seven weekdays, the weekday/weekend contrast is kept
rather than averaged away. Configure it under
`residential_behavior.time_use_occupancy` in `energy_config.yaml`
(`enabled`, `apply_to`, `state_fractions`, `present_states`, `awake_states`).

Tracts the income atlas does not cover fall back to the stylised residential
profile, as do runs without a `wd`.

> ⚠️ `social_ES` 1.0.3 ships a bug that makes `TimeUseSurvey` raise
> `TypeError: Expected numeric dtype, got object instead` on **every** call — its
> `*_share` columns are seeded with `None` and keep `object` dtype, so the
> `.round(2)` it then calls fails. `social._social_es_numeric_round()` works
> around it for the duration of the call by coercing object columns to numeric
> before rounding, and becomes a no-op once upstream fixes the dtype.

### 📊 Education KPIs are now cumulative; `so__pct_woman` replaces `so__pct_male`

**Breaking for these columns.** The four educational-attainment shares are
recomputed as a cumulative ladder — the share of the population aged 16+ that
reached **at most** that level:

| Before | After |
|--------|-------|
| `so__pct_edu_primary_or_below_{year}` | `so__pct_edu_lte_primary_or_below_{year}` |
| `so__pct_edu_lower_secondary_{year}` | `so__pct_edu_lte_lower_secondary_{year}` |
| `so__pct_edu_upper_secondary_{year}` | `so__pct_edu_lte_upper_secondary_{year}` |
| `so__pct_edu_tertiary_{year}` | `so__pct_edu_lte_tertiary_{year}` (= 100) |

INE publishes the four levels as *disjoint* shares, which answer "how many
stopped exactly here" — awkward across tracts, because a tract can read better or
worse depending on where the mass sits. The running sum is monotonic and directly
comparable. A level missing from the source leaves every KPI from it upwards NaN
rather than silently under-counting the ones above it.

`so__pct_male_{year}` becomes **`so__pct_woman_{year}`** (from the census's
`Population ~ Sex:Females`). The two are complementary, so only one is emitted.

### 💶 `so__household_income_group_{year}` — the Time Use Survey income band

The income atlas map now also emits the tract's monthly net household income,
deflated by `social_ES` to 2010 euros, bucketed into the four bands the Time Use
Survey respondents answered in (1 = ≤1,200 €, 2 = 1,201–2,000 €,
3 = 2,001–3,000 €, 4 = >3,000 €). It is the join key into the survey's schedules,
and is what `time_use_household_occupancy` matches a tract on.

### 🎯 `ec__*`: exact `social_ES` names, and a curated selection

**Breaking for every `ec__*` column.** The names `social_ES` returns are now kept
**verbatim**, `~` separator included — only the `ec__` prefix is added:

```
ec__main_dwellings_heating_type_individual   →  ec__main_dwellings_heating_type~individual
ec__main_dwellings_conservation_status_good  →  ec__main_dwellings_conservation_status~good
```

Flattening the separator to an underscore made the variable/class boundary
unrecoverable from the name (`..._heating_type_individual` could be read as
variable `heating` and class `type_individual`). With the `~` kept, everything
left of it is the variable and everything right of it the class it measures.
Re-run `hc.merge` to get the new spelling; the energy model reads both, so a
frame merged by an earlier release keeps driving the envelope.

**Only a curated 27 of the ~70 variables are emitted by default**
(`social.EC_SELECTED_VARIABLES`), chosen because they say something the `so__`
census layer and the cadaster do not already say — the dwelling's energy systems
(heating type and fuel, cooling, insulation, renewables, lighting, appliances),
its condition and accessibility, its interior, presence/absence, mobility,
housing cost burden and neighbourhood quality. Tenure, building type, floor area
and the single-person-household and parents'-origin blocks are dropped: the
census publishes them as head counts, and the cadaster measures them.

Pass your own sequence — or `None` for all ~256 columns — through
`social.join_social_ES(ec_variables=…)` /
`social.join_essential_characteristics(ec_variables=…)`.

### 🧹 social_ES source tables are freed as soon as they are joined

`social_ES` hands back every geographic level of a dataset (census tract /
district / municipality / province), of which only one is ever joined, and each of
the eight otherwise stayed alive until `join_social_ES` returned. New
`social.release(*frames)` empties them in place — the block manager's buffers, not
just the reference — as soon as each wide indicator table exists. The disk cache
under `{wd}/INE` is untouched, so a later dataset needing the same file re-reads
rather than re-downloads it.

Sizing this honestly: the tables come back already filtered to the municipalities
being enriched, so what is freed is **~30 MB for Barcelona** across the eight
datasets — measured, and it moves peak RSS by well under 1 %. The layer's real
peak (~2.2 GB) is inside `social_ES` while it reads the national source files, and
nothing in `hypercadaster_ES` can reach that. This is housekeeping, not a fix for
a memory ceiling.

### 🧮 social_ES 1.0.3: new dwelling datasets, de-duplicated indicators (breaking for some `so__`/`ec__` columns)

The socioeconomic layer is rebuilt on **`social_ES >= 1.0.3`** (now the minimum
in `setup.py`), which adds two dwelling censuses and already publishes the
Essential-Characteristics variables in English.

**Two new datasets** in `merge(social_ES_layer=True)`, joining `ALL_DATASETS` as
`"dwellings"` and `"empty_dwellings"`:

- **Empty and secondary dwellings**, per **municipality**, for the 2001, 2011 and
  2021 censuses, nationwide — `so__pct_dwellings_use_main_{year}`,
  `so__pct_dwellings_use_secondary_{year}`, `so__pct_dwellings_use_empty_{year}`
  and `so__municipality_dwellings_{year}`. 2021 replaced the census agent's
  judgement with a classification derived from each dwelling's yearly electricity
  consumption; `social_ES` harmonises the two into one series readable across all
  three censuses. Where INE withholds the split (the 2011 secondary/empty split
  below 2,000 inhabitants, the 2021 classification below 1,000), `social_ES`
  **models** it with gradient-boosted trees calibrated so each province
  reproduces the published figure; those rows are flagged in
  `so__dwelling_use_predicted_{year}` (1.0 = modelled) and should be excluded
  from anything turning on a single small municipality.
- **2021 dwellings census**, per **census tract** — `so__dwellings_2021`,
  `so__dwellings_main_2021`, `so__households_2021`, `so__pct_dwellings_main_2021`
  and the tenure of the main dwellings
  (`so__pct_main_dwellings_owned_2021` / `_rented_2021` / `_other_tenure_2021`).

**De-duplicated indicators.** Every indicator now comes from exactly one dataset
— the one publishing it at the finest geography over the longest run of years —
so the same figure no longer arrives twice under two names. Removed from the
income atlas, where the annual population census already publishes them (per
census tract, and further forward in time):

| Removed | Use instead |
|---|---|
| `so__population_{year}` (atlas) | `so__population_{year}` — same name, now the population census head count |
| `so__pct_over_65_{year}` | `so__pct_65_plus_{year}` |
| `so__pct_under_18_{year}` | `so__pct_under_15_{year}` |
| `so__pct_spanish_{year}` | `so__pct_foreign_nationality_{year}` (its complement) |

Added: **`so__net_income_per_household_2010eur_{year}`** (household net income
deflated to 2010 euros, so income is comparable across years — every other income
column is nominal) and **`so__pct_born_same_municipality_{year}`**.

**`ec__*` translation removed.** `social_ES` ≥ 1.0.3 emits the
Essential-Characteristics variables in short English snake_case itself and drops
the ones INE publishes twice, so `social.translate_ec_column()`,
`social.rename_ec_columns()` and `social.ec_column_rename_map()` — about 450
lines of Spanish→English dictionaries — are **deleted**. `hypercadaster_ES` now
only prefixes the columns and turns the `~` separator into `_`, giving names like
`ec__main_dwellings_heating_type_individual` and
`ec__main_dwellings_conservation_status_good` (~250 columns over 70 topics).

**Fixes:**

- **INE codes are stripped before zero-padding.** The rental index publishes its
  district codes as `"01 "`; padding alone carried the trailing space into a key
  that matched nothing, so **every district-level rental join came back empty**.
- **Rental index now joins by district** where INE publishes it (5,698 districts),
  falling back to the municipal series elsewhere — previously the whole city got
  its municipal average.
- **`ec__conservation_*` was never the real column name.** The energy model looked
  up `ec__conservation_{status}`, but the social layer emits
  `ec__main_dwellings_conservation_status_{status}`, so
  `_conservation_infiltration_factor()` and `_building_conservation()` silently
  returned the default envelope for every real frame. Both now read the emitted
  name (the short form still works).

### 🇪🇸 CTE residential user profile (Spanish standards)

A **third residential occupant profile — `cte`** (alongside *aware* / *unaware*) and a
new **`cte`** default scenario, following the CTE DB-HE *perfiles de uso* +
UNE-EN 16798, per the RETABIT reference (Mont Lecocq et al., *Energy & Buildings* 328
(2025) 115215, Table 25). The CTE user:

- **Setpoints**: constant **21 °C heating / 26 °C cooling**, no setback.
- **Internal gains**: the CTE 24-h fixed profiles — lighting and appliances peak at
  **4.40 W/m²** with the standard hourly steps (0.44 / 1.32 / 2.20 / 4.40 W/m² =
  4.40 × 0.1/0.3/0.5/1.0), occupancy sensible peaking at 2.15 W/m² (× 1.0/0.25/0.5).
- **Natural ventilation**: a **fixed 4 ren/h** on summer (cooling-season) nights
  **01:00–08:00** (not wind-driven).
- **Shadow control**: movable shades at **opaque fraction 0.6** (transmittance 0.4),
  the CTE value for keeping monthly incident radiation < 2 kWh/m²/month — the paper's
  140 W/m² radiation trigger is approximated here as **daytime deployment**.

All values live in `energy_config.yaml` → `residential_cte`.

### 💶 Income-driven setpoint variability

Following the RETABIT sensitivity result (Mont Lecocq et al. 2025, §5 / Fig. 14 — the
heating setpoint drives ±15–24 % of demand and **correlates with household income**),
the residential setpoints now shift with the building's **household net income** (from
the social_ES `so__*income*` layer). A linear ramp lowers the heating setpoint and
raises the cooling setpoint as income falls (energy-poverty under-heating / AC
under-use), reaching `heating_offset_max` / `cooling_offset_max` at/below `income_floor`
and zero at/above `reference_income`. It applies only to the real-behaviour profiles in
`income_setpoints.apply_to` (default `[unaware, aware]`) — the normative **`cte`** user
keeps its standard band. When the income layer is absent, `residential_behavior.
default_income` is used; disable with `income_setpoints.enabled: false`. Configured
under `energy_config.yaml` → `income_setpoints`.

### 🚿 Domestic hot water (DHW) end-use

The scenario KPIs now carry a **DHW** column (**`dhw_demand_kwh_m2`**) on each
residential per-season row (heating / cooling / in_between) — DHW is used year-round,
so its annual demand is **split across the simulated seasons** by the same EPW day→season
map the other KPIs use (and it honours `run_period_months`). Grounded in **CTE DB-HE
HE4** (28 L/person·day at 60 °C, monthly cold-water inlet temperatures).
Occupancy per household is derived from the **census-tract population** (social_ES
`population`) distributed over the tract's residential non-communal floor area, then
allocated to each Housing space by its own non-communal area and **floored at 1
person**. The normative **`cte`** scenario — and the fallback when population/tract data
is unavailable — uses the CTE-default occupancy density
(`loads.residential.ppl_per_m2`). Configured under `energy_config.yaml` →
`residential_dhw`; disable with `residential_dhw.enabled: false`.

### 🐛 Reproducibility fix

`_adopted()` (the `random_mix` aware/unaware split and awning/blind adoption) and the
unaware random-ventilation hour draw used Python's builtin `hash()`, which is **salted
per process** — so results changed every run. Both now use a **stable `zlib.crc32`**.

### 📊 Occupant-behaviour scenario analysis → per-use-type KPIs

New **`energy.run_behavior_scenarios(gdf, building_reference, wd, ...)`** builds and
runs a building through EnergyPlus under six occupant-behaviour scenarios and
returns global KPIs per thermal-zone **use type** (occupied zones only):

- **Scenarios**: (1) a demographic random mix of user types across dwellings;
  (2) all *unaware* users (HVAC on, random per-dwelling ventilation + erratic
  daytime HVAC use); (3) *unaware* **free-floating** with random ventilation;
  (4) *aware* **free-floating** with optimal-moment ventilation; (5) all *aware*
  users maintaining **19–22 °C in winter / 24–26 °C in summer** with optimal
  ventilation. The `*_ff` scenarios *are* the free-floating cases, so each scenario
  is simulated once (no separate twin).
- **Occupancy** follows fixed per-use-type schedules — residential (sleeping +
  morning/evening activity, partial midday, half-occupied weekends), office
  (08:00–20:00 weekdays), commercial (07:00–22:00 Mon–Sat).
- **Three temperature-based seasons**, from each day's **mean outdoor temperature**
  (from the EPW) **smoothed over a 3-day window** so seasons stay continuous (a
  single warm/cold day no longer flips the season): **in-between = 16–24 °C** (no
  HVAC for any user, for both the availability gate and the KPI split), below →
  heating, above → cooling. The KPI
  output reports the **number of days per season** (`season_days`).
- **EN 16798-1 adaptive setpoints** for the aware dwellings: instead of a fixed band,
  the heating/cooling setpoints **track the daily comfort temperature**
  Tc = 0.33·θrm + 18.8 (θrm = running-mean outdoor temperature, clamped 10–30 °C),
  conditioning to the Category-II adaptive band pulled 1 °C inward
  (`_apply_adaptive_setpoints`). So the HVAC keeps each dwelling inside the same
  comfort band the KPI evaluates against. **When the dwelling is unoccupied the aware
  setpoints set BACK to `hvac.aware_setback` (default 16 °C heating / 30 °C cooling)**,
  and the HVAC is left **available around the clock** in the heating/cooling seasons
  (season-gated always-on) so the setback is actually held rather than the zone
  free-floating.
- **Unaware users hold a fixed band** — `hvac.unaware` = **21 °C heating / 25 °C
  cooling** — whenever the HVAC is available.
- **Non-residential buildings have a single normative user** (no aware/unaware split —
  that behaviour model is residential-only): office/commercial/tertiary zones follow
  the **RITE / RD 486-2022 restriction** for tertiary buildings — **heating ≤ 19 °C,
  cooling ≥ 27 °C** while occupied (`hvac.nonresidential`), setting back to
  `hvac.nonresidential_setback` (16/30) when empty, with 24 h availability in the
  heating/cooling seasons (off in-between). The default (non-scenario) build's
  `schedules.setpoints` for office/commercial were aligned to the same 19/27 band.
  Their **fresh-air ventilation follows RITE IT 1.1.4.2** (IDA air-quality
  categories): offices/tertiary **IDA 2 = 12.5 l/s·person**, commercial/public-
  concurrence **IDA 3 = 8 l/s·person**, supplied mechanically by the HVAC while
  occupied (`hvac.nonresidential_ventilation_ls_per_person`). Their occupancy,
  lighting, equipment and infiltration follow the **CTE DB-HE "perfiles de uso"**
  for terciario (operating-hour occupancy schedules driving the internal gains).
- **Finite HVAC**: the ideal-loads plant is no longer an infinite system; heating and
  cooling capacity are **capped (`LimitCapacity`)** at `hvac.max_heating_w_per_m2` /
  `hvac.max_cooling_w_per_m2` (default **100 / 80 W·m⁻²**), sized from each
  conditioned zone's floor area.
- **Ventilation by behaviour**: *aware* users do a daily 07:45–08:00 hygiene airing,
  a heating-season 14:00–14:15 airing when it's mild (>20 °C outdoors), and a summer
  **night purge** (00:00–08:00) that runs only when the zone is warm (≥22 °C) — else
  the HVAC handles it; *unaware* users do short random daytime bursts. All natural
  ventilation is **wind-driven** (scales with wind speed, not a fixed rate).
- **KPIs per use type AND season** (heating / cooling / in-between; area-weighted):
  seasonal thermal demand per m², internal gains per m², and — using **EN 16798-1
  adaptive comfort** (Category II) — the % of occupied hours out of comfort split
  into **DAY (08:00–21:59)** and **NIGHT (22:00–07:59)**. The aware comfort band is
  **21.5–23 °C in winter** / 24–26 °C in summer.
- **Only residential/office/commercial buildings are thermally modelled**; any other
  use gets no HVAC even as a zone inside a modelled building, and a building with no
  conditioned use returns no model.
- **Use classification always prefers the CAT-file usage** (via the CAT-derived
  thermal zones), falling back to the cadastral `building_use` only when the CAT
  attributes are unavailable (`_resolve_use_category`).

### 🏠 Façade shading (balconies, awnings, blinds) and per-use glazing

RESIDENTIAL-only façade shading (`_add_facade_shading`):

- **Balconies** on residential **main-façade** walls from the first floor up, sized
  by construction period (old → short/shallow, modern → deep/full-width); each
  balcony wall's windows become 2 m balcony doors. Config: `shading.balcony`.
- **Awnings** over E/SE/S/SW/W windows (never north): an **inclined fabric** (1.5 m
  long, sloping 30° below horizontal) projecting **outward**, *installed* with a
  probability from **area income and dwelling conservation state** (`ec__*` /
  social_ES when present, else defaults). They are **operable**: aware users deploy
  them by day in the cooling season; unaware users never do (kept retracted). On the
  **top floor** (no balcony above) a horizontal awning of the balcony's projection
  depth is created over the window instead. Config: `shading.awning`.
  (All façade shading — balconies, awnings, blinds, and window reveals — is oriented
  by the wall's own-zone centroid so it always projects to the OUTSIDE.)
- **Window blinds** on every residential window, also operable: aware users close
  the sun-facing window to 75 % at its peak-sun time (east in the morning, south at
  midday, west in the afternoon/evening), otherwise open; unaware users never use
  the blinds.
- **`random_mix`** is now a 50/50 split of aware/unaware users across the dwellings.
  It is **residential-only** — skipped for buildings with no residential use.
- **Commercial premises are far less glazed** — a per-use WWR multiplier
  (`windows.wwr_by_use`, commercial 0.35) and, by default, glazing only on the
  commercial main façade.
- **Inputs from social_ES**: occupancy (people) from the sociodemographic layer
  (`br__average_household_size`) via the residential-behaviour engine, and dwelling
  **conservation status** from the Essential-Characteristics `ec__conservation_*`
  columns → an envelope-infiltration multiplier; both fall back to model defaults
  when the layers are absent.
- **Aware summer night purge**: in the cooling period the *aware* dwellings (the
  `aware`/`aware_ff` scenarios and the aware share of the random mix) open all
  windows through the night into the early morning (22:00–08:00) at a high air-
  change rate, with the HVAC off during that window — the standard passive night-
  cooling strategy.
- **Relevant-shading filter** (`filter_shading`, on by default): neighbour-shading
  panels that cannot affect the simulated building are dropped — those below its
  horizon from every vantage, those to the N/NE/NW of its centroid (no direct sun
  in the northern hemisphere), and those **occluded by a nearer panel** as seen
  from any part of the building — keeping the building's own self-shading. This
  both speeds the runs and removes irrelevant obstructions. `max_shading` is a
  further hard cap and `run_period_months` can restrict the simulated period for
  quick checks (drop it for the true annual KPIs). Wired into
  `examples/buildings_complete_inference.py` (writes `<ref>_scenario_kpis.csv`).

### 🏗️ `plot_idf_geometry` — party vs adiabatic walls, dimmed shading

- **Party walls and adiabatic partitions are now distinct** legend groups/colours
  in the 3-D IDF view. Both share the Adiabatic boundary condition, so they are
  kept apart by construction (party → `HC_PartyWall`, internal adiabatic →
  `HC_IntWall`; same thermal layers, so no change to simulation results): a party
  wall abuts the neighbour building (its outer face sees outdoor air) whereas an
  adiabatic wall is a purely internal partition.
- **Neighbour shading is rendered much more transparent** than the building
  (`shading_opacity=0.2` vs the `opacity=0.8` building surfaces) so the context
  obstructions no longer compete with the model; a surface seen through one or more
  others reads as clearly blurred/dimmed.

### 🧩 Thermal-zone floorplan synthesis — connected, proportional, façade-aware

`energy.synthesize_space_geometry` / `energy._partition_floor` (the on-demand
per-space geometry behind the EnergyPlus model, the 3-D view and
`plot_building_thermal_zones`) were reworked so every floor tiles **exactly** and
every zone is a shape you can actually simulate:

- **Single-connected zones.** Each building space *and* the communal core is now a
  single connected polygon instead of a scattered `MultiPolygon` (which was awkward
  to model as one thermal zone). The partition is a recursive area-balanced
  bisection (`_bisect_connected`) whose every cut is complementary and whose cut
  **angle is chosen to keep both halves connected**.
- **Areas stay proportional.** Spaces share each floor **in proportion to their CAT
  area**, scaled to fill exactly what is left after the communal core — so nothing
  is dropped and nothing over-fills, even when the cadastral footprint is smaller
  than the sum of constructed areas. This fixes both the *lost volume* on C-/U-shaped
  footprints (a far lobe was previously discarded) and the *disproportionate areas*
  the earlier connectivity-repair produced.
- **Communal core forced interior.** The communal stair/lift core is carved as a
  compact, connected block kept **off the main façade**, leaving the whole façade to
  the occupiable/parking spaces.
- **Every space fronts the main façade.** On floors without a communal core
  (ground-floor shops + parking, all-garage basements) all spaces are cut **normal
  to the façade** so shops and garages each reach the street — a ground-floor
  commercial premise is no longer landlocked behind the parking block.
- **Parking/warehouses need façade too.** Parking, warehouse, storage and garage
  spaces are no longer exempt from the ≥ 3 m main-façade rule — they need it for the
  vehicle door — and are flagged (`facade_deficit`) when they cannot reach it.
- **Zone labels.** `plot_building_thermal_zones` now labels each zone with its
  **distinguishing building-space reference** (e.g. `0003IL`) instead of the shared
  parcel prefix; the communal core is labelled `communal`.

### 🏭 Taller ground floor for industrial & agricultural buildings

Industrial and agricultural buildings now always use the taller
`ground_floor_height` (5 m) for the ground storey — regardless of how many floors
they have — for machinery/vehicle clearance, in addition to the existing
"≥ 5 floors above ground" rule. Centralised in
`utils.effective_ground_floor_height(n_floors_above_ground, ground_floor_height,
floor_height, building_use)` and `utils.is_tall_ground_floor_use(building_use)`
(keyed on the cadastral `building_use`, e.g. `3_industrial`, `2_agriculture`), and
applied consistently across the EnergyPlus geometry, the shadow eye-levels /
wall-typing (`building_inference`) and the 3-D / shadow plots.

## [1.2.3] - 2026-07-15

### 🐛 Whole cadastral zones lost when a building had no CAT record

A building absent from the CAT file (rural parcels, undeveloped "solares")
reached `aggregate_CAT_file_building_spaces` with an empty slice. pandas'
`groupby(...).apply(...)` never invokes the aggregation on an empty frame and
passes the *input* columns through instead, so the result carried no `br__*`
column at all and the downstream aggregation raised
`KeyError: ['br__building_spaces_by_floor']`. That exception reached the
zone-level handler, so **a single CAT-less building discarded every building in
its cadastral zone**. On Lleida (`25900`) this silently dropped **39 of 1086
zones — 4540 buildings, ~29% of the municipality**.

- `aggregate_CAT_file_building_spaces` now returns empty frames carrying the
  full column schema (declared with dtypes, so an empty frame stays
  concat-compatible instead of degrading numeric columns to `object`).
- The consumer in `process_zone` now reads the per-building aggregation cache
  through `_get_agg_CAT`, honouring the `None` it stores for buildings with no
  CAT rows rather than treating it as a cache miss and recomputing on the empty
  slice.

### 🏚️ Theoretical building space for buildings absent from the CAT file

Such buildings (1426 of 13619 in Lleida — 1392 `2_agriculture`, 31 with no
usage, 2 residential, 1 industrial) had no building spaces at all: every
CAT-derived `br__*` column came out NaN, `br__thermal_zones` was empty and no
energy model could be built. `building_inference.add_theoretical_CAT_spaces`
now gives each one a single theoretical space:

- `building_space_reference` == `building_reference` (the building *is* its only
  space);
- use type mapped from the cadaster `building_use` (`2_agriculture` →
  `Agricultural`, …), defaulting to `Unknown`;
- area from the cadaster's official built area (`building_area`, which equals
  the CAT total built area across floors for buildings that do have CAT data),
  falling back to the footprint area; all private, no communals;
- effective year from `year_of_construction`.

Injected in `mergers` *after* the CAT cache read/write, so `cat_building_spaces`
stays a pure function of the CAT files. Undeveloped-parcel supplements
("solares", no geometry) are left alone.

### 🚜 Agricultural buildings model as uninsulated warehouses

- New `agricultural` use category: same assumptions as a warehouse (no people,
  no equipment, no HVAC, vehicle door) but **uninsulated whatever the year**
  (new `archetype_envelope.uninsulated`, which omits the wall/roof/floor U
  targets so the solver drops the `INS` layer) and **very leaky**
  (`infil_ach: 5.0`). Kept out of the `parking` bucket so it does not inherit
  parking's vehicle-door *sizing* semantics elsewhere.
- `kind='non_occupied'` for agricultural zones across all five classifiers
  (`building_inference`, `mergers`, `utils`, and two in `energy`).
- `_dominant_use_type` now falls back to non-occupied zones when a building has
  no occupied ones — a barn's only zone is non-occupied, and it would otherwise
  resolve to the generic `other` archetype.

### 🏗️ BuildingParts declaring no floors at all

`n_floors_above_ground == 0` yields `range(0)` — no floor footprints — so the
building dropped out of the per-floor analysis and, downstream, out of the CAT
integration (gated on a non-empty per-floor frame). **55 buildings in Lleida
lost every CAT attribute this way, 48 of them with real CAT data**, including
residential buildings with dwellings. Such a part is now read as a single
storey, and `_augment_building_floors_from_cat` then rebuilds the floors the CAT
file reports, buffered to their CAT area.

Deliberately narrow — only when a building has no floors above **and** below:
a 0-floor part *inside* a building that has floors is a canopy/porch (763 in
Lleida) and stays excluded from the storeys, and the 1 underground-only
structure gains no spurious ground floor.

## [1.2.2] - 2026-07-15

### ⚡ CAT attribute caches in the central DuckDB (big re-run speedup)

A no-inference CAT merge over a whole province took ~1 h of "CAT" time. Profiling
showed the cost is **not** the CAT file read (~1 s even for a 160k-space
municipality) but `br_attributes_from_CAT` — a per-building aggregation that runs
~99 s for a 21k-space municipality (minutes for the largest). Both are now cached
in `results/building_results.duckdb`, keyed by `cadaster_code`:

- **`cat_building_spaces`** — the parsed per-space CAT table
  (`parse_horizontal_division_buildings_CAT_files`).
- **`cat_building_attributes`** — the per-building `br__*` aggregates
  (`br_attributes_from_CAT` output); complex dict/list columns are pickled.

Measured on the full Lleida province (231 municipalities, 831k spaces / 145k
buildings): the cold aggregation costs **71 min** in total — 938 s for Lleida
city (`25900`) alone — while the cached re-read of the same municipality takes
**2.6 s (~366×)**; smaller ones land at ~0.6–0.9 s (~99 s → 0.6 s, ~165× for
`25049`). Cached output is identical, nested `br__*` dict/list values included.
Like the raw `.CAT`→parquet cache, both caches are pure
functions of the static CAT files, so they are **reused regardless of
`force_update`** (the first run populates them). Clear them after replacing CAT
files with `hc.utils.clear_cat_cache(wd[, cadaster_code])`.

- Writes are idempotent per municipality and degrade gracefully — any cache
  read/write error falls back to a normal parse/aggregate and never breaks the
  merge.
- New helpers: `_db.cat_cache_read/write`, `_db.cat_attrs_read/write`,
  `_db.cat_cache_clear`, `utils.clear_cat_cache`.

### 🔤 Shorter `ec__*` names + income-source fixes (breaking for these columns)

- **`ec__*` Essential Characteristics** columns are now **shorter snake_case**,
  same `group_variable_subclass` hierarchy. The curated Spanish→English maps hold
  the final short values directly, so digit brackets are cleanly separated
  (fixes e.g. `ec__households_useful_floor_area_up_to45m2` →
  `ec__households_floor_area_le_45m2`, `..._heating_type_individual` →
  `..._heating_individual`).
- **Income Atlas "source of income"** indicators are **percentages**, not €, and
  renamed accordingly: `so__salary_income_per_person` →
  `so__pct_income_source_salary`, `…pension…` → `so__pct_income_source_pension`,
  `…unemployment…` → `so__pct_income_source_unemployment`,
  `so__other_benefits_income_per_person` → `so__pct_income_source_other_benefits`,
  `so__other_income_per_person` → `so__pct_income_source_capital_investments_secondary`.

### 🔊 Logging & warnings

- `merge()` now announces each contextual join on stderr regardless of `verbose`:
  `Joining the socioeconomic and sociodemographic data of last N years (datasets: …)`
  and `Joining the Weather data of last N years (YYYY-YYYY)`.
- Silenced the flood of `RuntimeWarning: invalid value encountered in divide`
  emitted by the third-party `social_ES` package during the `ec__*` downscaling
  (harmless 0/0 → NaN for empty section groups).

## [1.2.1] - 2026-07-14

### 🏷️ Namespaced, snake_case contextual columns (breaking)

The contextual `merge()` layers now emit **namespace-prefixed, snake_case**
column names, so their origin is unambiguous and the whole output schema is
consistent (no more CamelCase keys):

- **Weather** (`weather_layer`): `we__` prefix, e.g. `HDD_2023` → `we__hdd_2023`,
  `airTemperatureMean_2023` → `we__air_temperature_mean_2023`. Stems in
  `weather.WEATHER_INDICATORS` are now snake_case.
- **Socioeconomic** (`social_ES_layer`): `so__` prefix, e.g.
  `NetIncomePerHousehold_2020` → `so__net_income_per_household_2020`. All
  `social.*_INDICATORS` output names are now snake_case.
- **POIs** (`pois_layer`): `poi_` prefix → `poi__`, e.g. `poi_amenity_food_count`
  → `poi__amenity_food_count` (group/subgroup snake-cased).
- **Essential Characteristics** (`ec__*`): the Spanish→English translation now
  emits snake_case, e.g. `ec__MainDwellings_HeatingType_Individual` →
  `ec__main_dwellings_heating_type_individual`. `translate_ec_column()` /
  `rename_ec_columns()` output snake_case keys.
- New helper `utils.to_snake_case()` for normalising identifiers to snake_case.

Consumers referencing the old column names must update to the new ones.

## [1.2.0] - 2026-07-13

### 🌦️ Annual weather indicators from ERA5-Land

A new **`weather_layer`** in `merge()` attaches annually-aggregated ERA5-Land
climate indicators to every building. Each building is snapped to the nearest
0.1° ERA5-Land cell **with data** (coastal buildings fall back to the closest
inland cell), and the per-cell yearly statistics are added as `{Indicator}_{year}`
columns (year as a suffix, mirroring `social_ES`). It runs **before** the POI join
and uses `hypercadaster_ES.weather`.

#### Added
- `merge()` parameters: `weather_layer` (bool), `weather_years` (int = N most-recent
  **complete** calendar years, or an explicit list, default 5), `weather_indicators`
  (subset selector, default all), `weather_hdd_base` / `weather_cdd_base` (°C degree-day
  bases, default 18 / 21), `weather_bbox` (a predefined WGS84
  `[min_lon, min_lat, max_lon, max_lat]` download extent).
- Nine indicators per year: `HDD`, `CDD` (heating/cooling degree days),
  `precipitation` (mm), `GHI` (kWh/m²), `airTemperatureMean` / `_Min` / `_Max` (°C),
  `relativeHumidityMean` (%), `windSpeedMean` (m/s). See
  `hypercadaster_ES.weather.WEATHER_INDICATORS`.
- `weather.join_weather()` orchestrator. Cell resolution and per-cell statistics
  are computed over the **unique** snapped cells, not per building, so the cost is
  independent of the number of buildings (scales to city/province frames).
- ERA5-Land cache directory `utils.weather_era5_dir_(wd)` = `{wd}/weather/era5land`,
  created by `utils.create_dirs`.

#### ⚡ ARCO analysis-ready Zarr gatherer (priority, no queue)
- `weather.load_ERA5Land` now gathers with a two-tier strategy: **first** the CDS
  [Analysis-Ready Cloud-Optimised (ARCO)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data)
  Zarr stores over HTTPS (`_load_ARD`, Bearer-authenticated with the `~/.cdsapirc`
  key — **no `retrieve` queue**), and **only if that fails** the classic cdsapi
  retrieve (GRIB grid / CSV timeseries). Toggle with `HC_ERA5_ARD=0/1`.
  A full year over a small area is fetched in ~seconds vs the queue's minutes-to-hours.
- **Province-adaptive download**: the grid download/cache extent is derived from the
  requested area (snapped outward to whole degrees), replacing the previous hard-coded
  Catalan region — any Spanish province now works, and same-tile municipalities share
  one cache.
- **Single-municipality** merges fetch only the ERA5-Land point nearest the municipality
  centroid (timeseries endpoint) instead of the whole grid, falling back to the grid if
  that point has no data.
- New dependencies for the ARD path: `zarr`, `aiohttp` (missing → graceful fallback to
  the cdsapi retrieve).

### 🧮 Socioeconomic indicators from `social_ES` (INE)

A new **`social_ES_layer`** in `merge()` attaches INE socioeconomic and
demographic indicators to every building, joining each source dataset at its own
finest geographic level (census **section** → **district** → **municipality** →
**province**) using the codes added by `census_layer`. It runs **before** the POI
join. Output columns use the `{Indicator}_{year}` convention (year as a suffix),
e.g. `NetIncomePerHousehold_2020`, keeping the most recent `social_ES_years`
(default 5) per dataset.

#### Added
- `hypercadaster_ES.social` module with `join_social_ES()` and the curated
  indicator maps. Datasets (`social.ALL_DATASETS`): `atlas` (Household Income
  Distribution Atlas — ~20 headline income/inequality/demographic indicators,
  section level), `population` (census population shares: foreign nationality/
  birth, sex, broad age bands), `education` (education level + labour-force status
  shares), `rental` (municipal rental price index), `price` (provincial housing
  price index, quarterly→annual), `electricity` (district electricity-consumption
  percentiles).
- `merge()` parameters: `social_ES_layer` (bool), `social_ES_years` (int, default
  5), `social_ES_datasets` (subset selector, default all). Source files are
  fetched once for the whole of Spain and cached under `{wd}/INE/…` by
  `social_ES` (EUPL-1.2), then reused on subsequent runs.
- Degrades gracefully: missing `social_ES`, absent `section_code`, or a dataset
  with no coverage for the selected area is skipped with a warning rather than
  failing the merge.
- **Essential Characteristics downscaling** — the INE *Essential Characteristics
  of Population and Households* survey is downscaled to building level via
  `social.join_essential_characteristics()`, adding `ec__*` proportion columns
  (age, income band, household nationality, dwelling size, building type/age,
  heating, vehicles, internet, …). It is exposed as a `social_ES` dataset
  (`social.ESSENTIAL_CHARACTERISTICS = "essential_characteristics"`) folded into
  the `social_ES_layer`: included when `social_ES_datasets` is `None` or lists
  `"essential_characteristics"`, **and additionally gated on `use_CAT_files=True`**
  (it consumes the CAT-based `br__area_without_communals` / `br__building_spaces`
  per-use-type dicts and `year_of_construction`). Requesting it without
  `use_CAT_files` skips it with a warning. Uses the updated `social_ES` signature
  that accepts the GeoDataFrame directly (`hypercadaster_ES_input_gdf=`) instead of
  a pickle path. Note the underlying INE survey is only published for larger
  municipalities (~147 nationwide), so `ec__*` values populate only there;
  buildings elsewhere are left unset.
- `ec__*` columns are now emitted with **English PascalCase keys directly** from
  `merge()`, e.g.
  `ec__Viviendas principales - Tipo de calefacción ~ Sí, individual` →
  `ec__MainDwellings_HeatingType_Individual`. The translation is compositional
  (curated group/variable/subclass dictionaries + accent-stripped fallback for
  unseen categories) and idempotent. Exposed as `social.rename_ec_columns(df)`
  (for pre-existing Spanish-named frames), `social.translate_ec_column()` and
  `social.ec_column_rename_map()`.

### 🏠 Addresses now sourced from CartoCiudad

Addresses are now taken from **CartoCiudad** (CNIG) instead of the Cadastre's
INSPIRE Address feed. CartoCiudad is distributed as one GeoPackage per province
and is auto-downloaded, by province name, from the CNIG *Centro de Descargas*
(the most recent monthly edition is always fetched). The Cadastre is still used
for parcels, buildings and CAT files.

#### Added
- `downloaders.cartociudad_downloader()` — resolves each province's current file
  and downloads/extracts its address GeoPackage to `{wd}/cartociudad/{province}.gpkg`.
  (Matches provinces by **name**: the CNIG `codProvincia` filter does not follow
  INE province codes.)
- `mergers.get_cartociudad_address()` — reads the CartoCiudad `portalpk_publi`
  address points, maps them to the canonical address schema, and recovers each
  point's `building_reference` via a spatial join to the cadastral parcels
  (`mergers._assign_points_to_parcels`, extracted from the old Barcelona path).
- New address columns: `population_entity`, `address_source`,
  `address_last_modification`, `postal_address_specification` (= "Portal").
- `downloaders.normalize_cartociudad_street_type()` — normalises the inconsistent
  `tipo_vial` to the canonical Catastro street-type name (see *Changed*). A full
  scan of all 52 provinces produced a mapping-review CSV (with per-province
  provenance) under `{wd}/cartociudad/street_type_mapping_review.csv`.
- `utils.match_building_space_postal_addresses()` — matches each building space's
  CAT postal address to the closest parcel address, producing three per-space
  columns: `building_space_postal_address` (the **matched CartoCiudad address**),
  `building_space_postal_address_from_CAT_file` (the **original CAT value**), and
  `building_space_postal_address_location` (the matched point). All three are
  building-level `{use_type: […]}` dicts that explode at the building-space
  resolution.
- `hc.merge(verbose=…)` — new parameter (default `"simple"`) that suppresses the
  per-stage `[stage +… | RSS …]` markers and per-municipality progress logs
  (`assign_building_zones` / `join_other_constructions`); pass `verbose="full"`
  to restore them (`utils.set_verbose` / `utils.verbose_log`).

#### Changed
- **Addresses are always compacted** to one row per parcel (`building_reference`),
  with per-address columns as aligned lists. The `compact_addresses` parameter of
  `hc.merge()` has been **removed** (behaviour is now always-on).
- **Renamed the compacted address columns** for clarity:
  `street_full_labels` → `postal_addresses_with_extensions`,
  `street_full_clean` → `postal_addresses_clean`,
  `address_location` → `postal_addresses_location`,
  `specification` → `postal_address_specification`.
- **Street-type normalisation.** `tipo_vial` regional-language/bilingual/
  descriptor/abbreviation/typo variants map to the Spanish canonical name
  (`AVINGUDA`/`KALEA`→`AVENIDA`/`CALLE`, `PLAÇA`→`PLAZA`, `PARTICULAR`→`PARTIDA`);
  genuine Spanish names with no Catastro type are kept; unrecognised short codes
  become `None`.
- **`building_space_postal_address` now holds the matched CartoCiudad address**
  (was the raw CAT value, which moved to `…_from_CAT_file`).
- A wrongly-elided Catalan `D'` before a consonant is normalised to `DE` in the
  full-address labels (`PASEO D'GUITART` → `PASEO DE GUITART`); valid elisions
  before a vowel/h (`D'URGELL`) are kept (`utils._expand_catalan_elision`).
- `postal_code` now comes from CartoCiudad (`cod_postal`); the postal-code boundary
  spatial join (`join_by_postal_codes`) is kept only as a fallback for parcels with
  no CartoCiudad address, and no longer overwrites an existing code.

#### Fixed
- POI join for large municipalities (e.g. Lleida city `25900`) failed with
  *"Must have equal len keys and value when setting with an iterable"*:
  `gpd.sjoin_nearest` emits one row per equidistant tie, inflating the fallback
  assignment. The nearest-parcel match (and the within-parcel join) are now
  de-duplicated to one row per POI (`utils.get_pois_open_street_maps`).
- `cadaster_downloader` works around a Catastro INSPIRE feed bug where
  special-character municipality names (e.g. the `Ç` in *Sant Llorenç de Morunys*,
  `25237`) produce a broken download URL: it validates the download is a real zip
  and retries with collapsed spaces (`downloaders._download_inspire_zip`).

#### Deprecated / removed
- The INSPIRE **Address** layer (`ES.SDGC.AD`) is no longer downloaded
  (`cadaster_downloader` still fetches parcels and buildings).
- The Barcelona Open Data **carrerer** street/address register download and its
  reader (`_process_barcelona_open_data` / `process_open_data_addresses`) and the
  old `get_cadaster_address` are deprecated (commented out).

## [1.1.0] - 2026-06-04

### 🚀 DuckDB Migration — Eliminating User-Facing RAM Peaks

The internal persistence layer has been migrated from pickle-based artifacts to
**DuckDB** with native spatial support. The central `building_results.duckdb`
file is now the canonical output of ``hc.merge()`` — pickle files are no longer
written by the library (user scripts may still materialise and save pickles as
an escape hatch).

#### Added

**DuckDB Persistence Layer** (`_db.py`)
- Central DuckDB file per ``results_dir`` with spatial extension (native GEOMETRY)
- Per-zone shard DBs for parallel worker writes (no single-writer contention)
- Schema reconciliation across shards (lazy ``ALTER TABLE ADD COLUMN``)
- Lossless round-trip for opaque Python objects (dict/list → pickled BLOB)
- CRS sidecar storage in ``_meta`` table
- ``ResultsHandle`` — lightweight wrapper around DB + cadaster_code
- Streaming readers (``read_df_safe_chunks``) for bounded-RAM materialisation
- Automatic WAL checkpointing and memory cap

**API Additions**
- ``hc.ResultsHandle`` — public class for lazy DB access
- ``hc.utils.get_heavy_columns_for_building()`` — fetch shadow/terrain BLOBs per building
- ``plot_zones_ratio`` parameter on ``hc.merge()`` to control plot scope
- ``force_update`` and ``cleanup_shards`` parameters for cache management

#### Changed

**Building Inference Pipeline**
- ``process_building_parts()`` now returns ``ResultsHandle`` (was ``(sbr_df, br_df)``)
- Per-zone writes: pickle → per-zone ``{zone}.duckdb`` shard files
- Phase A: shard ingestion via ``ATTACH``/``INSERT`` into central DB
- Phase B materialisation removed from ``process_building_parts`` (DB is the artifact)
- Aux frames (``gdf_global``, ``gdf_footprints_global``) stored in DuckDB tables
- Workers query aux slices on demand instead of COW-inheriting full frames

**Merger**
- SBR aggregation rewritten in SQL (``GROUP BY`` + ``json_group_object``) —
  replaces pandas ``groupby``/``apply`` dict construction (dominant RAM peak)
- ``buildings`` table written to central DB for every municipality

**Data Reading**
- ``read_br_inferred_indicators()`` / ``read_sbr_inferred_indicators()`` now query DuckDB
- ``read_addresses_indicators()`` queries DuckDB with pickle fallback

#### Removed

- Municipality-level pickle writes (``{code}_{sbr,br}_results.pkl``)
- ``_read_compressed_pickle()`` and gzip/zstd dual-decoder
- Per-zone pickle writes (replaced by shard DBs)
- ``_downcast_floats()`` (DuckDB column types provide same memory shape)

#### Performance Impact

- **RAM peak**: eliminated — the full sbr/br DataFrames are no longer held
  in-process during merging; SQL runs against the DB on disk
- **Resume semantics**: re-running with same ``(code, results_dir)`` skips
  completed zones (``zone_status`` table); delete the DB file for clean run
- **Dependency**: ``duckdb >= 1.5.3`` (spatial extension)

#### Migration Notes for User Scripts

- ``hc.merge()`` still returns a GeoDataFrame (backward compatible)
- The central ``building_results.duckdb`` is the persistent artifact
- ``read_*_indicators()`` still return pandas DataFrames (signature unchanged)
- Heavy BLOB columns (shadows, terrain grids) are excluded from bulk reads;
  use ``hc.utils.get_heavy_columns_for_building()`` for per-building access
- Pass ``wd=`` to ``plot_building_shadows_analysis()`` to auto-hydrate heavy columns

### ⚡ EnergyPlus Interoperability

Building inference outputs can now be exported to runnable **EnergyPlus** models
with real local weather.

#### Added

**Energy model generation** (`energy.py`)
- ``hc.energy.build_energy_model(gdf, building_reference, ...)`` — per-space 3-D thermal
  model exported as an EnergyPlus **IDF** via ``eppy`` + ``geomeppy`` (both MIT)
- Spanish archetype envelopes and constructions (``hc.energy.building_archetype``),
  window-to-wall-ratio prediction (``hc.energy.building_wwr``)
- YAML-driven defaults (``energy_config.yaml``) with
  ``hc.energy.configure()`` / ``hc.energy.get_config()`` overrides
- ``hc.energy.plot_idf_geometry()`` — 3-D viewer for a generated IDF

**Weather → EPW** (`weather.py`)
- ERA5-Land nearest-cell extraction with coastal fallback
- ``hc.weather.get_municipality_weather()`` — hourly timeseries for the closest cell
- ``hc.weather.weather_statistics()`` — heating/cooling degree days + climate summary
- ``hc.weather.municipality_epw()`` — cached per-year EnergyPlus EPW files, fed into
  ``build_energy_model(weather_year=...)``

### 📍 Points of Interest (OpenStreetMap)

#### Added

- ``pois_layer=True`` on ``hc.merge()`` — attaches per-building POI indicators
  from OpenStreetMap (groups: ``amenity``, ``emergency``, ``healthcare``,
  ``leisure``, ``office``, ``shop``, ``tourism``)
- Per group/subgroup columns: on-building counts and IDs
  (``poi_{prefix}_count``, ``poi_{prefix}_ids``) plus proximity counts within
  100 m / 250 m / 500 m / 1000 m / 1500 m of the building centroid
- Raw OSM query results cached under ``open_street/`` so re-runs skip the network

---

## [1.0.0] - 2025-08-28

### 🚀 Major Release - Initial Public Release

This marks the first major release of hypercadaster_ES, providing comprehensive Spanish cadastral data analysis capabilities.

#### Added

**Core Architecture**
- **Complete library structure** with 6 main modules for specialized functionality
- **Modular design** enabling flexible usage patterns from basic to advanced analysis
- **Comprehensive error handling** and robust data processing pipeline
- **Memory-efficient processing** optimized for municipal to provincial-level analysis

**Data Integration Capabilities**
- **Spanish cadastral data integration** with buildings, parcels, addresses, and zones
- **Multi-source data merging** with census tracts, postal codes, and elevation data
- **Barcelona open data integration** with neighborhoods, establishments, and enhanced addressing
- **CAT file processing** for detailed building space and horizontal property division data

**Advanced Analysis Features**
- **Building inference pipeline** with geometric analysis and space classification
- **Orientation and environmental analysis** including shadow calculations and street relationships
- **Floor footprint calculation** and building envelope analysis
- **Thermal analysis properties** for energy modeling applications

**Geographic Coverage**
- **National coverage** of peninsular Spain, Balearic Islands, and Canary Islands
- **Provincial and municipal-level** processing capabilities
- **Enhanced Barcelona coverage** with city-specific open data layers
- **Scalable processing** from individual buildings to entire autonomous communities

**External Tool Integration**
- **IREC energy simulation** dataset preparation and format conversion
- **Interoperability functions** for building energy modeling workflows
- **Visualization utilities** for spatial analysis and result interpretation
- **Export capabilities** for GIS and mapping applications

**Documentation and Examples**
- **Comprehensive documentation** with installation, configuration, and API reference
- **Practical examples** demonstrating various usage scenarios
- **Complete data schema reference** with 200+ possible output columns
- **Use case documentation** covering multiple application domains

#### Core Functions

**Primary Entry Points**
- `download()`: Automated data acquisition from multiple Spanish government sources
- `merge()`: Comprehensive data integration and processing pipeline

**Data Processing Modules**
- `mergers.py`: Advanced data joining and integration engine
- `downloaders.py`: Multi-source data acquisition and management
- `building_inference.py`: Geometric analysis and building space inference
- `utils.py`: 50+ utility functions for spatial analysis and data processing
- `interoperability.py`: External tool integration and format conversion

#### Geographic Data Sources

**Spanish Government Sources**
- **Spanish Cadastral Service**: Buildings, parcels, addresses, CAT files
- **National Statistics Institute (INE)**: Census tracts, demographics, administrative boundaries
- **National Geographic Institute (IGN)**: Digital elevation models, topographic data
- **Spanish Postal Service**: Postal code boundaries and delivery areas

**Regional and Municipal Sources**
- **Barcelona Open Data Portal**: Commercial establishments, neighborhood boundaries
- **OpenStreetMap Integration**: Street networks, points of interest, validation data

#### Output Data Schema

**Basic Building Information**
- Core identifiers, geometry, and physical characteristics
- Complete address information with administrative context
- Usage classifications and construction details

**Advanced Analysis Results** (with building_inference=True)
- Floor-by-floor geometric calculations and area distributions
- Building orientation and environmental context analysis
- Thermal envelope properties for energy modeling
- Shadow analysis and solar exposure calculations

**Detailed Space Analysis** (with CAT files)
- Individual space-level information and usage classifications
- Economic valuations and communal area calculations
- Horizontal property division analysis
- Statistical aggregations by space type and floor level

#### Technical Specifications

**Performance Characteristics**
- **Memory Efficiency**: Optimized for datasets up to 2M+ buildings
- **Processing Speed**: Municipal analysis in minutes to hours
- **Scalability**: Provincial analysis capabilities with appropriate hardware
- **Data Quality**: Automated validation and consistency checking

**Software Requirements**
- **Python 3.10+** compatibility with modern dependency management
- **Core Dependencies**: pandas, geopandas, shapely, requests
- **Optional Dependencies**: Advanced visualization and analysis libraries

### 🔄 Enhanced Core Functionality (Latest Updates)

This section documents significant enhancements implemented to strengthen the v1.0.0 release with improved usability, performance, and reliability.

#### Geographic Parameter Handling Enhancements

**Flexible Parameter Combination**
- **Multiple parameter support**: `download()` and `merge()` functions now accept any combination of `province_codes`, `ine_codes`, and `cadaster_codes`
- **Automatic list conversion**: All geographic parameters automatically converted to lists for consistent processing
- **Smart deduplication**: Automatic removal of duplicate municipalities across different parameter types
- **Backward compatibility**: All existing usage patterns remain fully functional

```python
# ✅ NEW: Combine multiple geographic parameters
hc.download(
    "./data",
    province_codes="25",                    # Entire Lleida province
    ine_codes=["08019", "46250"],          # Barcelona and Valencia cities
    cadaster_codes=["03014", "11012"]      # Additional municipalities
)
```

#### Building Analysis Safety & Performance

**Single Municipality Constraint**
- **Resource protection**: Building inference (`building_parts_inference=True` and `building_parts_plots=True`) restricted to single municipalities only
- **Clear error handling**: Descriptive `ValueError` raised when multiple municipalities attempted with building analysis
- **Performance optimization**: Prevents accidental resource-intensive operations on large datasets
- **Guided workflows**: Example files demonstrate proper usage patterns for multi-municipality analysis

**Random Zone Visualization**
- **New parameter**: `plot_zones_ratio=0.01` controls percentage of zones visualized (default: 1%)
- **Smart sampling**: Random zone selection reduces memory usage while maintaining representative coverage
- **Minimum guarantee**: At least 1 zone always plotted regardless of ratio setting
- **Configurable visualization**: Adjustable ratio allows fine-tuning of plot generation vs. performance

```python
# ✅ NEW: Control visualization scope
gdf = hc.merge(
    "./data",
    cadaster_codes=["25023"],           # Single municipality required
    building_parts_inference=True,      # Advanced analysis enabled
    building_parts_plots=True,          # Visualization enabled
    plot_zones_ratio=0.02              # Plot 2% of zones
)
```

#### Code Quality & Reliability Improvements

**Enhanced Error Handling**
- **Standardized logging**: Consistent `sys.stderr.write()` usage throughout data processing pipeline
- **Robust data processing**: Improved handling of edge cases and data inconsistencies
- **Critical bug fixes**: Added missing '0000' entry in `building_space_typologies` preventing processing failures
- **Comprehensive validation**: Better parameter validation and user feedback

**Example Files Overhaul**
- **Constraint compliance**: Updated `buildings_complete_inference.py` to handle single municipality constraint properly
- **Multi-municipality patterns**: Individual processing with result combination for building inference workflows
- **New example file**: `multi_municipality_basic.py` demonstrates efficient multi-municipal processing without inference
- **Enhanced documentation**: Improved comments, usage guidance, and parameter explanations

#### API Parameter Updates

**New Function Signatures**
```python
# Enhanced download function
hc.download(
    wd: str,
    province_codes: Optional[Union[str, List[str]]] = None,    # ✅ Enhanced
    ine_codes: Optional[Union[str, List[str]]] = None,         # ✅ Enhanced  
    cadaster_codes: Optional[Union[str, List[str]]] = None,    # ✅ Enhanced
    # ... existing parameters
)

# Enhanced merge function
hc.merge(
    wd: str,
    province_codes: Optional[Union[str, List[str]]] = None,    # ✅ Enhanced
    ine_codes: Optional[Union[str, List[str]]] = None,         # ✅ Enhanced
    cadaster_codes: Optional[Union[str, List[str]]] = None,    # ✅ Enhanced
    # ... existing parameters
    plot_zones_ratio: float = 0.01,                           # ✅ NEW
)
```

#### Workflow Improvements

**Efficient Multi-Municipality Processing**
- **Individual analysis**: Process municipalities separately when building inference is required
- **Result combination**: Automatic merging of individual municipality results
- **Performance optimization**: Memory-efficient processing for large-scale analysis
- **Flexible deployment**: Support for both single and multi-municipality workflows

**Enhanced User Experience**
- **Clear constraints**: Explicit validation prevents common usage errors
- **Better examples**: Comprehensive example files for different usage scenarios
- **Improved documentation**: Enhanced parameter descriptions and usage guidance
- **Performance predictability**: Clearer expectations for resource requirements

### 🔧 Technical Implementation

#### Data Processing Pipeline
1. **Download Phase**: Automated acquisition from multiple government sources with enhanced parameter handling
2. **Integration Phase**: Spatial and attribute joining with topology validation and smart deduplication
3. **Analysis Phase**: Constraint-aware geometric inference with configurable visualization
4. **Export Phase**: Format conversion and result preparation with improved error handling

#### Coordinate Reference Systems
- **Primary CRS**: EPSG:25831 (ETRS89 / UTM zone 31N) for Spain
- **Regional Support**: EPSG:25830 for western Spain, EPSG:25828/25829 for islands
- **Automatic Projection**: Coordinate system handling and transformation

#### Data Quality Assurance
- **Geometric Validation**: Automatic topology checking and correction
- **Attribute Consistency**: Standardized classification systems
- **Temporal Synchronization**: Quarterly data update alignment
- **Error Handling**: Comprehensive exception management

### 📊 Usage Statistics and Capabilities

#### Scale Capabilities
- **Individual Building**: Single building detailed analysis
- **Neighborhood**: 100-10,000 buildings with full inference
- **Municipal**: 1,000-500,000 buildings with configurable analysis depth
- **Provincial**: 100,000-2M+ buildings with performance optimizations

#### Analysis Depth Options
- **Basic**: Core building and address information
- **Enhanced**: Additional administrative and environmental context
- **Advanced**: Full geometric inference and thermal properties
- **Complete**: All features including detailed space analysis

### 🏆 Key Achievements

#### Research and Development
- **4+ years** of development and testing with real-world datasets
- **Collaboration** with CIMNE research institute and UPC university
- **Integration** of multiple Spanish government data sources
- **Validation** through academic research and practical applications

#### Community and Applications
- **Open Source**: EUPL v1.2 license supporting academic and commercial use
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Practical use cases across multiple domains
- **Extensibility**: Modular architecture supporting custom analysis

### 🔮 Future Roadmap

#### Recently shipped
- ✅ INE socioeconomic integration via `social_ES` (income atlas, population,
  education, rental & housing price, electricity, Essential Characteristics) — v1.2.0
- ✅ EnergyPlus IDF generation and automatic simulations — v1.1.0+
- ✅ OpenStreetMap POI proximity indicators — v1.1.0
- ✅ Nationwide addresses from CartoCiudad — v1.2.0

#### Short-term Improvements
- Performance optimizations for large-scale processing
- Enhanced error handling and user feedback
- Additional data validation and quality checks

#### Medium-term Features
- Basque Country and Navarre cadastral integration
- Enhanced visualization and reporting capabilities
- Predicting WWR / façade material / energy systems (`epc_CAT`)
- Interoperability with VoxCity and city2graph

---

## Development Notes

### Version Numbering

hypercadaster_ES follows [Semantic Versioning](https://semver.org/):
- **Major version** (1.x.x): Breaking changes or major new capabilities
- **Minor version** (x.1.x): New features, backward compatible
- **Patch version** (x.x.1): Bug fixes, improvements, backward compatible

### Release Schedule

- **Major releases**: Annually or for significant architectural changes
- **Minor releases**: Quarterly for new features and data source additions
- **Patch releases**: Monthly for bug fixes and improvements

### Supported Versions

- **Current version** (1.0.x): Full support with updates and bug fixes
- **Previous major** (0.x.x): No previous versions exist
- **Legacy versions**: Will be supported for 1 year after major version release

---

*For more information about specific changes, see individual pull requests and commit history on [GitHub](https://github.com/BeeGroup-cimne/hypercadaster_ES).*

---

[← Contributing](contributing.md) | [Back to README](../README.md)