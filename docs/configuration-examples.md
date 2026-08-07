# Configuration & Examples

This document provides comprehensive configuration options and practical examples for using hypercadaster_ES effectively across different scenarios.

## ⚙️ Advanced Configuration & Customization

### Geographic Area Selection Methods

#### By Administrative Levels

```python
import hypercadaster_ES as hc

# Spanish provinces (2-digit codes) - Complete province analysis
hc.download("./data", province_codes=["08", "28", "46"])  # Barcelona, Madrid, Valencia

# Municipality INE codes (5-digit codes) - Specific cities
hc.download("./data", ine_codes=["08019", "28079", "46250"])  # Barcelona, Madrid, Valencia cities

# Cadastral codes - Direct cadastral reference
hc.download("./data", cadaster_codes=["08900", "28900", "46250"])  # Barcelona, Madrid, Valencia cities
```

#### **New: Flexible Parameter Combination**

You can now combine different geographic parameters in the same call:

```python
# ✅ Combine province, INE codes, and cadaster codes
hc.download(
    "./data", 
    province_codes=["25"],              # Entire Lleida province
    ine_codes=["08019"],               # Barcelona city
    cadaster_codes=["46250"]          # Valencia city
)

# ✅ Mix single values and lists (auto-converted)
hc.download(
    "./data",
    province_codes="08",               # Single province (auto-converted to list)
    cadaster_codes=["25023", "25285"] # Multiple specific municipalities
)

# ✅ All parameters are automatically deduplicated
hc.download(
    "./data",
    province_codes=["25"],             # Includes municipality 25023
    cadaster_codes=["25023"]          # Duplicate - automatically removed
)
```

### Comprehensive Data Layer Configuration

#### Basic Data Layers

```python
hc.download(
    "./data",
    province_codes=["08"],
    neighborhood_layer=True,        # Barcelona neighborhood boundaries
    postal_code_layer=True,         # Spanish postal code boundaries
    census_layer=True,             # INE census tract data
    elevation_layer=True,          # Digital elevation model
    open_data_layers=True,         # Barcelona open data integration
    force=False                    # Skip download if files exist
)
```

#### Advanced Analysis Configuration

```python
gdf = hc.merge(
    "./data",
    cadaster_codes=["25023"],   # Alpicat municipality
    
    # Basic data integration
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    
    # Advanced building analysis
    building_parts_inference=True,     # Enable geometric analysis
    building_parts_plots=False,        # Generate visualization plots (only for small areas)
    
    # CAT file integration (requires separate download)
    use_CAT_files=True,               # Use detailed building space data
    CAT_files_rel_dir="CAT_files",    # CAT files directory location

    # Socioeconomic enrichment (INE via social_ES; runs before POIs)
    social_ES_layer=True,             # income, population, education, dwellings,
                                      # empty/secondary dwellings, prices, electricity
    social_ES_years=5,                # keep the 5 most recent years -> so__{indicator}_{year}
    social_ES_datasets=None,          # all datasets; Essential Characteristics (ec__*) added
                                      # automatically because use_CAT_files=True

    # Annual weather indicators (ERA5-Land; runs before POIs)
    weather_layer=True,               # HDD/CDD, precipitation, GHI, temperature, humidity, wind
    weather_years=5,                  # 5 most recent complete years -> we__{indicator}_{year}
)
```

### Socioeconomic layer (`social_ES`)

```python
# Adds so__{indicator}_{year} columns (income, population, education, dwelling
# stock & tenure, empty/secondary dwellings, rental & housing prices,
# electricity), each joined at its own finest geographic level. With
# use_CAT_files=True, the per-building Essential Characteristics survey is also
# downscaled to ec__* proportion columns.
gdf = hc.merge(
    wd, cadaster_codes=["25900"],     # Lleida city
    census_layer=True,                # required — provides census_tract_code
    use_CAT_files=True,               # enables the ec__* Essential Characteristics
    social_ES_layer=True,
    # Optional: pick a subset of datasets
    # social_ES_datasets=["atlas", "empty_dwellings", "essential_characteristics"],
)

# Share of the municipality's dwellings standing empty at the 2021 census, and
# whether that figure was modelled rather than published by INE.
gdf[["so__pct_dwellings_use_empty_2021", "so__dwelling_use_predicted_2021"]]
```

> The first `social_ES_layer=True` run downloads and caches the whole-Spain INE
> source files under `{wd}/INE/…` (a slow one-off); later runs reuse the cache.
> `social_ES` is EUPL-1.2 licensed and must be ≥ 1.0.3.

### Weather layer (ERA5-Land)

```python
# Adds annual we__{indicator}_{year} climate columns per building: we__hdd, we__cdd,
# we__precipitation, we__ghi, we__air_temperature_mean/min/max, we__relative_humidity_mean,
# we__wind_speed_mean. Each building is snapped to the nearest ERA5-Land cell with data.
gdf = hc.merge(
    wd, cadaster_codes=["25900"],     # Lleida city (single municipality -> nearest-point download)
    weather_layer=True,
    weather_years=5,                  # or an explicit list, e.g. [2021, 2022, 2023]
    # weather_indicators=["HDD", "CDD", "precipitation"],  # subset (default: all 9)
    # weather_hdd_base=18.0, weather_cdd_base=21.0,        # degree-day bases (°C)
)

# Predefined bounding box (e.g. when processing many municipalities at once):
ine_codes = hc.utils.get_ine_codes_from_bounding_box(wd, [1.9, 41.2, 2.6, 41.6])
gdf = hc.merge(
    wd, ine_codes=ine_codes,
    weather_layer=True,
    weather_bbox=[1.9, 41.2, 2.6, 41.6],   # [min_lon, min_lat, max_lon, max_lat]
)
```

> Weather data is gathered from the CDS **ERA5-Land** dataset (requires a
> configured `~/.cdsapirc` key) and cached under `{wd}/weather/era5land`. The
> fast **ARCO analysis-ready Zarr** path is tried first (no download queue),
> falling back to the classic `cdsapi` retrieve; disable it with the
> `HC_ERA5_ARD=0` environment variable. Install `zarr` and `aiohttp` for the
> ARCO path.

### CAT Files Integration

CAT files provide the most detailed building information available from Spanish cadastral sources.

#### Downloading CAT Files

Visit the [Spanish Cadastral Service](https://www.sedecatastro.gob.es/Accesos/SECAccDescargaDatos.aspx) portal:

1. Click "Descarga de información alfanumérica por provincia (formato CAT)"
2. Select your province of interest
3. Download the provincial CAT files
4. Extract to `CAT_files/` directory in your working directory

#### CAT File Directory Structure

```
data_directory/
├── cadaster/
├── CAT_files/
│   ├── 08279U_17012025.zip
│   ├── 08900U_17012025.zip
│   └── ...
└── ...
```

#### Using CAT Files

```python
# Ensure CAT files are available in the specified directory
gdf = hc.merge(
    "./data",
    cadaster_codes=["08900"],
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"  # Relative to working directory
)
```

## 🚀 Practical Examples

Runnable, self-contained versions of these live in [`examples/`](../examples).

### Lightweight municipal analysis (no inference)

```python
import hypercadaster_ES as hc

wd, muni = "./municipal_analysis", ["46250"]  # Valencia
hc.download(wd, cadaster_codes=muni, postal_code_layer=True, census_layer=True)
gdf = hc.merge(wd, cadaster_codes=muni, postal_code_layer=True,
               census_layer=True, building_parts_inference=False)
gdf.to_parquet(f"{wd}/valencia_basic.parquet")
print(f"Processed {len(gdf)} buildings")
```

### Full building inference (single municipality)

```python
import hypercadaster_ES as hc

wd, muni = "./advanced_analysis", ["25023"]  # Alpicat (Lleida)
hc.download(wd, cadaster_codes=muni, elevation_layer=True)
gdf = hc.merge(
    wd, cadaster_codes=muni,
    building_parts_inference=True,   # geometric inference (single municipality only)
    use_CAT_files=True,              # detailed CAT space classification
    CAT_files_rel_dir="CAT_files",
    # neighborhood/open-data layers are Barcelona-only — leave them off here
    neighborhood_layer=False, open_data_layers=False,
)

# heavy shadow/terrain BLOBs are excluded from the frame; fetch per building
ref = gdf.iloc[0]["building_reference"]
heavy = hc.utils.get_heavy_columns_for_building(wd, "25023", ref)
print(sorted(heavy.keys()))
```

### Provincial scale (memory-conscious)

Disable inference/CAT/plots for province-wide runs and query the DuckDB instead
of materialising the full frame:

```python
import hypercadaster_ES as hc
from hypercadaster_ES import _db

wd, prov = "./provincial_analysis", ["25"]  # Lleida
hc.download(wd, province_codes=prov, census_layer=True, elevation_layer=True)
gdf = hc.merge(wd, province_codes=prov, census_layer=True, elevations_layer=True,
               building_parts_inference=False)

con = _db.open_db(wd, read_only=True)
counts = con.execute("""
    SELECT municipality_name, COUNT(*) AS buildings
    FROM buildings GROUP BY municipality_name ORDER BY buildings DESC LIMIT 10
""").fetchdf()
con.close()
print(counts)
```

## ⚡ Energy Model (EnergyPlus IDF)

> 📖 See **[Energy Model & Scenarios](energy-model.md)** for what these settings
> actually control: how the model is assembled, the occupant-behaviour model and
> the scenarios.

`build_energy_model()` turns one inferred building into a 3-D per-space thermal
model and returns a `geomeppy.IDF`. It needs a frame produced with
`building_parts_inference=True`.

```python
import hypercadaster_ES as hc

gdf = hc.merge("./data", cadaster_codes=["08900"], building_parts_inference=True)
ref = gdf.iloc[0]["building_reference"]

# geometry only (no weather) — fastest way to inspect the model
idf = hc.energy.build_energy_model(gdf, ref, wd="./data", output_path="./data/energyplus")

# with real local weather attached (see the Weather section for era5_dir)
idf = hc.energy.build_energy_model(
    gdf, ref, wd="./data",
    weather_year=2022, era5_dir="./data/era5", epw_dir="./data/epw",
)

hc.energy.plot_idf_geometry(idf)        # 3-D viewer — pass the OBJECT, so the
                                       # plot shows the filtered shading that
                                       # will actually be simulated
# idf.run()   # requires a local EnergyPlus install
```

### Mechanical ventilation of non-residential uses (`nonresidential_ventilation`)

```yaml
nonresidential_ventilation:
  enabled: true
  litres_per_second_person:      # RITE RD 1027/2007 IT 1.1.4.2.3
    office: 12.5                 # IDA 2
    commercial: 8.0              # IDA 3
    other: 8.0
```

Offices and shops are mechanically ventilated on this rate; dwellings get the
DB-HS3 system below instead (and, before the CTE, nothing but infiltration plus
the windows the occupant opens). The intake runs on the zone's **occupancy**
schedule rather than on HVAC availability, so premises get fresh air whenever they
are open — including the mild shoulder season when heating and cooling are gated
off, which is when indoor CO₂ would otherwise be worst.

### Extended thermal comfort zone (`comfort_zone`)

```yaml
comfort_zone:
  enabled: true
  t_min_c: 20.0
  t_max_c: 26.0
  rh_min: 30
  rh_max: 65
  humidity_ratio_max_g_kg: 11.5
  humidity_ratio_min_g_kg: null
  enthalpy_min_kj_kg: 30.0
  enthalpy_max_kj_kg: 50.0
  setpoints:
    enabled: true
    margin_c: 0.5                    # pull the band inside the field's edges
    min_deadband_c: 1.0              # smallest heating↔cooling gap kept
    humidity_control: dehumidification   # none | dehumidification | full
    humidity_setpoint: field             # field | rh_max
```

The Mollier h-x comfort field of EN ISO 7730 / DIN 1946-2 / ASHRAE 55. Comfort is a
**region** in (temperature, humidity), not an interval on a thermometer: 26 °C at
8 g/kg and 26 °C at 16 g/kg are different rooms to be in.

The field is a **polygon**, because each bound binds over a different stretch of the
temperature range — with the defaults the 65 % RH ceiling governs the cool end, the
50 kJ/kg enthalpy line takes over at ≈22.3 °C, and the absolute-humidity cap never
binds at all. Every bound is evaluated at every temperature and the tightest wins,
so editing one number reshapes the field rather than breaking a vertex list; set any
bound to `null` to drop it.

It is reported **alongside** the EN 16798-1 adaptive band
(`pct_hours_out_of_comfort_zone_*` beside `pct_hours_out_of_comfort_*`), not instead
of it: one asks whether the indoor state sat inside the comfort field, the other
whether the temperature was acceptable *for this climate*, and a dwelling can pass
one and fail the other. Both are drawn on the report's psychrometric chart — the
zone filled, the adaptive band as a dashed outline.

`setpoints` closes the loop: the **AWARE/PASSIVE thermostat is confined to the
field** rather than only measured against it. The adaptive tracking is kept and
bounded — the setpoint still follows the running-mean outdoor temperature and
stops at the field's edge (20.5 / 25.5 °C by default). Both ends are confined and
`min_deadband_c` is held between them, because capping only the cooling end lets
the heating setpoint climb above it in a hot week, which EnergyPlus treats as a
fatal error. The income-driven offset is applied *after*, so an energy-poor
household still underheats.

`humidity_control` drives the field's humidity bounds through the ideal-loads
humidistat; the default `dehumidification` matches what a dwelling actually has (a
cooling coil, not a humidifier). `humidity_setpoint: field` holds the field's
**effective** ceiling at the cooling setpoint (47.1 % RH at 25.5 °C) rather than
the bare `rh_max` of 65 % — above ~22.3 °C the enthalpy line is the binding bound,
and a humidistat set at 65 % leaves the dwelling outside the zone.

Expect a **large cooling increase**: on the reference building this took the aware
occupant from 9.6 to 26.9 MWh/yr while raising its time inside the field from
29.7 % to 37.6 % of occupied hours. Set `setpoints.enabled: false` to keep the
plain adaptive band.

### Continuous ventilation of post-CTE dwellings (`hs3_ventilation`)

```yaml
hs3_ventilation:
  enabled: true
  from_year: 2007            # CTE obligatory from 29-09-2006
  l_s_per_occupant: 8.0      # DB-HS3 Tabla 2.1 admission (5 dormitorios + 3 salas)
  l_s_per_m2: 0.55           # extraction (baños + cocina) over the dwelling area
  min_ach: 0.0
  max_ach: 1.5
```

CTE DB-HS3 requires a dwelling built under the code to be ventilated
**continuously**, so every occupant scenario of a post-`from_year` building gets
the same always-on intake — the normative `cte` user included. It is a property of
the building, not of the behaviour: without it the one case the law actually
describes came out the *worst* ventilated of the set. The flow is
`max(occupants × l_s_per_occupant, area × l_s_per_m2)` — the two ends of one
airflow, sized on the larger, as HS3 itself does — which lands near 0.66 ACH on a
typical dwelling. An unknown construction year is treated as non-compliant.

### Season classification (`seasons`)

```yaml
seasons:
  smoothing_days: 7          # centred moving average of the daily mean outdoor T
  in_between_low_c: 16.0     # below -> heating
  in_between_high_c: 24.0    # above -> cooling
```

Seasons come from the weather, not the calendar: each day is classified by its
mean outdoor temperature, smoothed over a centred week so the season follows the
trend rather than the weather. A shorter window lets a warm spell in March read as
a cooling day, which then carries a heating-month result under the cooling label.
The occupant scenarios run no HVAC on in-between days; the normative `cte` case
ignores this and follows its own calendar (`residential_cte.seasons`).

### Residential occupancy from the census (`loads`)

```yaml
loads:
  residential: {ppl_per_m2: 0.028, ...}      # CTE DB-HE default (~2.8 per 100 m²)
  residential_occupancy_from_census: true    # prefer the building's real occupancy
```

With this on (the default), a building carrying `ctagg__people_per_m2` from
`hc.merge` takes its **`People` internal gains** from that census-tract occupancy
instead of the national default — the same head count that drives its DHW demand,
so metabolic gains, cooling load and hot water all describe one population. Each
dwelling gets `max(1, round(density × area))` occupants. Buildings without the
column, and the normative `cte` scenario, keep `loads.residential.ppl_per_m2`.

### Exterior persianas (`shading.persiana`)

```yaml
shading:
  persiana:
    enabled: true
    solar_transmittance: 0.05    # closed slats: almost opaque
    solar_reflectance: 0.55
    distance_to_glass: 0.05      # air gap, enters the window heat balance
    solar_setpoint_w_m2: 140     # close above this incident solar
```

A `WindowMaterial:Shade` layer on one `WindowShadingControl` per residential zone
— not a panel in front of the glazing, which cannot attenuate sky diffuse. The
control is solar-triggered **and** scheduled, and the schedule is shut through the
whole heating season: in winter the gain is wanted, so persianas and awnings alike
stay fully open whatever the occupant does. Which behaviours operate them at all
in the cooling and in-between seasons follows the occupant model (`aware`,
`passive` and `cte` do; `unaware` and `always_on` never touch them).

### Neighbour shading (`shading.neighbours`)

The context obstructions around the building are filtered **as the model is
built**, not at simulation time — so one set of panels is written to the `.idf`,
drawn by `plot_idf_geometry` and seen by EnergyPlus:

```yaml
shading:
  neighbours:
    filter: true          # drop panels that cannot shade this building: below its
                          # horizon everywhere, in the N/NE/NW sector (no direct
                          # sun in the northern hemisphere), or hidden behind a
                          # nearer panel
    max_panels: 200       # cap on what survives, largest first; null = no cap
```

Override per call — `build_energy_model(..., filter_shading=False, max_shading=None)`
keeps every obstruction. The building's own shading — footprint remainders,
balconies, awnings, exterior blinds, window reveals — is never filtered **and
never counted against the cap**: `max_shading` budgets neighbour context only.

### Reusing one IDF across the behaviour scenarios

```python
idf = hc.energy.build_energy_model(gdf, ref, wd="./data", weather_year=2022)
hc.energy.plot_idf_geometry(idf, output_path="./model.html")

kpis = hc.energy.run_behavior_scenarios(
    gdf[gdf["building_reference"] == ref],   # the ONE building to simulate
    wd="./data", weather_year=2022,
    idf=idf,               # every scenario runs on a COPY of this model
)
```

Each scenario copies the model and re-applies only the behaviour-dependent wiring
(archetype, setpoints, HVAC availability, ventilation), so the six runs share one
geometry/envelope/shading by construction rather than by trusting six rebuilds to
agree — and the build cost is paid once. Omit `idf=` to rebuild per scenario; the
KPIs come out identical either way.

### What the returned KPI frame contains

One row per **scenario × use type × season** for the residential zones, and a
single set of rows for any non-residential use, labelled `normative`:

| scenario | use_type |
|---|---|
| `random_mix`, `unaware`, `unaware_ff`, `aware_ff`, `aware`, `cte` | Residential |
| `normative` | Commercial / Office |

The behaviour scenarios describe **residents**; office and commercial premises run
the same normative user (RITE / RD 486-2022 tertiary restriction) in all of them,
so they are reported once — taken from the `cte` run, where the dwellings around
them are normative too. They are thermally coupled to those dwellings through
shared floors and ceilings, so their demand does drift slightly between scenarios
even though their own user never changes; reporting once avoids reading that
coupling as an occupant effect on the premises.

Pass `nonresidential_reporting="per_scenario"` to get them under every scenario
name instead.

Each row also carries the season's contiguous date runs (`season_ranges`, as
`DDMM/DDMM` strings — `"0101/0401, 0111/3112"` for a heating season that wraps the
year — alongside `season_days`), `solar_gains_kwh_m2` (glazing-transmitted,
separate from the people/lights/equipment `internal_gains_kwh_m2`),
`people_per_dwelling` next to the `dhw_demand_kwh_m2` it produces, and the
**delivered** setpoint `avg_setpoint_c` — the heating setpoint in the hours the
zone heated and the cooling one in the hours it cooled, over the `hvac_hours` it
actually ran, so setback and free-floating hours do not dilute a temperature the
building was never held at. Columns are season-agnostic: the row's `season` already
says which service it describes.

**Indoor temperatures**: `t_p10_day` … `t_p90_day` and `t_p10_night` …
`t_p90_night` give the operative-temperature percentiles over the occupied hours,
so the spread is visible and not just its mean.

**Comfort is measured over occupied hours.** `pct_hours_out_of_comfort_day` and
`_night` count only the hours the zone actually has occupants (hourly
`Zone People Occupant Count` > 0), split at 22:00–07:59. `pct_occupied_hours`
reports the share of the season those percentages rest on, so the basis is
visible: the behaviour scenarios sit around 77 %, a shop around 51 %, and the
normative `cte` user reads 100 % — its profile never empties the dwelling, so its
comfort figures span every hour of the year by construction. A zone that reports
no occupancy at all yields NaN rather than being treated as permanently occupied.

Defaults (storey heights, constructions, WWR, loads, schedules) come from
`energy_config.yaml`. Override them with your own YAML:

```python
hc.energy.configure("./my_energy_config.yaml")   # merged on top of defaults
cfg = hc.energy.get_config()
```

### HVAC power limits & occupant-scenario setpoints (`hvac`)

The ideal-loads systems are **finite** — heating/cooling capacity is capped
(`LimitCapacity`) at a specific power sized from each conditioned zone's floor area —
and the occupant-behaviour scenarios read their setpoints from the `hvac` block:

```yaml
hvac:
  max_heating_w_per_m2: 100    # heating capacity cap (W per m² of zone floor)
  max_cooling_w_per_m2: 80     # cooling capacity cap
  unaware:       {heating: 21, cooling: 25}   # unaware users: fixed band
  aware_setback: {heating: 16, cooling: 30}   # aware users, dwelling empty
  # Non-residential (office/commercial/tertiary) — one normative user, RITE / RD
  # 486-2022 restriction: heating ≤ 19, cooling ≥ 27 occupied; setback when empty.
  nonresidential:         {heating: 19, cooling: 27}
  nonresidential_setback: {heating: 16, cooling: 30}
  # RITE IT 1.1.4.2 fresh-air ventilation, per person (l/s): office IDA 2 / commercial IDA 3.
  nonresidential_ventilation_ls_per_person: {office: 12.5, commercial: 8.0}
```

A third residential profile, **`cte`** (with its own `cte` scenario), follows the
CTE DB-HE *perfiles de uso* + UNE-EN 16798 (RETABIT reference): constant 21/26 °C
setpoints, CTE 24-h internal-gain profiles (lighting/appliances 0.44–4.40 W/m²), a
fixed 4 ren/h summer-night ventilation (01:00–08:00), and shadow control at opaque
fraction 0.6. Configure it under `residential_cte`:

```yaml
residential_cte:
  setpoints:               {heating: 21, cooling: 26}   # constant, no setback
  lighting_w_m2:           4.4
  appliances_w_m2:         4.4
  night_ventilation_ach:   4.0
  night_ventilation_hours: [1, 8]
  shade_transmittance:     0.4                           # opaque fraction 0.6
```

*Unaware* dwellings hold the fixed `unaware` band whenever their HVAC is available.
*Aware* dwellings track the EN 16798-1 adaptive band while occupied and **set back**
to `aware_setback` when empty; their HVAC stays available around the clock in the
heating/cooling seasons (off in-between) so the setback is held rather than the zone
free-floating. **Non-residential** buildings are not split into aware/unaware — they
run a **single normative user** on the RITE / RD 486-2022 tertiary restriction
(heating ≤ 19 °C, cooling ≥ 27 °C occupied, `nonresidential_setback` when empty), with
**RITE IT 1.1.4.2 fresh-air ventilation** (IDA 2 offices 12.5 l/s·person, IDA 3
commercial 8 l/s·person) supplied by the HVAC while occupied, and occupancy / lighting
/ equipment / infiltration following the **CTE DB-HE perfiles de uso** for terciario.

### Income-driven setpoints and DHW

The residential setpoints can shift with the building's **household net income** (from
the social_ES `so__*income*` layer) — the RETABIT sensitivity finding that lower income
correlates with a lower heating setpoint. The offset is a linear ramp, applied only to
the profiles in `apply_to` (the normative `cte` user is never shifted):

```yaml
income_setpoints:
  enabled: true
  reference_income: 30000     # €/yr at/above which no offset is applied
  income_floor: 12000         # €/yr at/below which the max offset applies
  heating_offset_max: -2.0    # °C added to the heating setpoint at/below income_floor
  cooling_offset_max: 2.0     # °C added to the cooling setpoint at/below income_floor
  apply_to: [unaware, aware]  # 'cte' stays normative
```

**Domestic hot water** is reported as a `dhw_demand_kwh_m2` column on each residential
per-season KPI row (its year-round demand split across heating / cooling / in_between by
the EPW day→season map), computed per **CTE DB-HE HE4**. The people per household come from
the **census-tract population** (social_ES `population`) spread over the tract's
residential non-communal area, floored at 1 person; the `cte` scenario and the
no-population fallback use the CTE-default density `loads.residential.ppl_per_m2`:

```yaml
residential_dhw:
  enabled: true
  litres_per_person_day: 28   # CTE DB-HE HE4 reference demand (vivienda)
  reference_temp: 60          # °C service (accumulation) temperature
  inlet_temp_monthly: [8, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 8]  # CTE cold-water inlet, °C
```

Both features fall back gracefully (and are a no-op under `enabled: false`) when the
social_ES layer is absent, so runs without it are unaffected.

## 🌤️ Weather (ERA5-Land → EPW)

Weather helpers extract the nearest ERA5-Land cell (with coastal fallback) and
produce degree-day statistics and cached EnergyPlus EPW files. Downloading
ERA5-Land requires a configured [CDS API key](https://cds.climate.copernicus.eu/).

```python
from hypercadaster_ES import weather

lat, lon, year = 41.39, 2.16, 2022   # Barcelona
df = weather.get_municipality_weather("./data/era5", lat, lon, year)
print(weather.weather_statistics(df))            # HDD/CDD + climate summary

epw_path = weather.municipality_epw("./data/era5", "./data/epw", lat, lon, year)
print("EPW cached at", epw_path)                 # feed into build_energy_model
```

## 🔧 Performance & Memory Notes

- **Provinces / large cities:** keep `building_parts_inference=False` (it is
  single-municipality only) and read results from the DuckDB rather than the
  returned frame.
- **Per-municipality loop:** process big cities one at a time into separate
  working directories to bound memory.
- **Heavy columns:** shadow/terrain BLOBs are never materialised into the merge
  output; pull them on demand with
  `hc.utils.get_heavy_columns_for_building(wd, code, ref)`.
- **CAT files** add detail but cost parse time — enable only when you need
  space-level classification.