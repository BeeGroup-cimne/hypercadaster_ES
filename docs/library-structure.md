# Library Structure

A conceptual map of the package. For the **exhaustive, always-current function
and class reference**, build the HTML API docs from the docstrings:

```bash
pip install pdoc
./scripts/build_api_docs.sh      # writes docs/api/index.html
```

## 📁 Module map

```
hypercadaster_ES/
├── functions.py         # Primary entry points: download(), merge()
├── downloaders.py       # Data acquisition (Catastro, CartoCiudad addresses, INE, DEM, Barcelona open data)
├── mergers.py           # Joins all sources into one GeoDataFrame; grid/H3 aggregation
├── building_inference.py# Geometric/space inference (br__* columns, shadows, footprints)
├── energy.py            # Per-space 3-D thermal model → EnergyPlus IDF
├── social.py            # INE socioeconomic indicators via social_ES (income, population, ...)
├── weather.py           # ERA5-Land → weather_layer indicators, climate stats + EPW files
├── utils.py             # Geometry, codes, plotting and DB read helpers
├── interoperability.py  # Dataset builders for external tools (e.g. IREC)
└── _db.py               # DuckDB persistence layer (internal)
```

The public API is what `import hypercadaster_ES as hc` exposes (see
`__init__.py`); everything else is internal and may change.

## 🎮 functions.py — entry points

- **`download(wd, ...)`** — fetches every requested layer into `wd`. Select the
  area with `cadaster_codes`, `ine_codes`, or `province_codes` (combinable).
  Toggle layers with `neighborhood_layer`, `postal_code_layer`, `census_layer`,
  `elevation_layer`, `open_data_layers`.
- **`merge(wd, ...)`** — joins the downloaded layers into a single GeoDataFrame
  and persists it to the central DuckDB. Key switches:
  - `building_parts_inference` — run the geometric inference engine
    (single municipality only; memory-intensive).
  - `use_CAT_files` — pull detailed space classification from official CAT files.
  - `pois_layer` — attach OpenStreetMap POI proximity indicators.
  - `social_ES_layer` — attach INE socioeconomic indicators (income atlas,
    population, education, rental & housing prices, electricity, and — when
    `use_CAT_files=True` — the per-building Essential Characteristics
    downscaling). Tune with `social_ES_years` and `social_ES_datasets`.
  - `sociodemographic_layer`, `floor_height`, `ground_floor_height`, `num_workers`.
  - `verbose` — `"simple"` (default) hides the `[stage | RSS]` markers and
    per-municipality progress logs; `"full"` shows them.

  Addresses (from CartoCiudad) are always compacted to one row per parcel, with
  the per-address columns collapsed into aligned lists; street types are
  normalised to canonical Catastro names.

## 🔗 mergers.py — integration & aggregation

Joins buildings, addresses, parcels, zones, census tracts, postal codes,
neighborhoods (Barcelona), DEM elevation and POIs. Also provides the public
spatial-aggregation helpers:

- **`aggregate_to_grid(gdf, cell_size_m=100, ...)`** — buildings → regular
  square grid cells.
- **`aggregate_to_h3(gdf, resolution=10, ...)`** — buildings → H3 hexagons.

Both take a metric-CRS building GeoDataFrame and a column→aggregation mapping
(defaults in `DEFAULT_BUILDING_AGGS`).

## 🏗️ building_inference.py — geometric inference

The heavy analysis behind `building_parts_inference=True`: floor footprints,
adiabatic/party walls, patios, orientation, street relationships, and
per-floor shadow patterns. Outputs the `br__*` (building-reference) and
`sbr__*` (space-reference) indicator families consumed by `energy.py`.

## ⚡ energy.py — EnergyPlus model generation

> 📖 **[Energy Model & Scenarios](energy-model.md)** explains how the model is
> assembled and why — the occupant model, the ventilation routes, the scenarios
> and the KPI definitions. This section is the module/API map.

Turns the `br__*` inference columns into a 3-D, per-space thermal model and
exports an EnergyPlus **IDF** via `eppy` + `geomeppy` (both MIT — no AGPL
coupling). Public entry points:

- **`build_energy_model(gdf, building_reference, ...)`** → `geomeppy.IDF`
  (optionally attaches real weather via `weather_year`). The **neighbour-shading
  filter and cap** (`filter_shading`, `max_shading`, defaults from
  `shading.neighbours`) are applied here, so the returned IDF, the `.idf` written
  to disk, the 3-D plot and the simulated context are the same panels.
- **`plot_idf_geometry(idf, ...)`** — 3-D viewer; takes the IDF **object** or a
  path. Pass the object to see exactly what will be simulated.
- **`load_energy_model(idf_path)`** — an `.idf` written by `build_energy_model`
  read back with its `{model}.idf.hcmeta.json` sidecar (zone → use type, building
  reference, EPW, shading-filtered flag), which the IDF format cannot carry.
- **`run_behavior_scenarios(gdf, wd, ..., idf=None, idf_path=None, context_gdf=None)`** —
  occupant-behaviour scenarios → per-use-type KPIs. `gdf` is the **one building**
  to simulate; its reference is read off the `building_reference` field (a
  multi-building frame is an error unless `building_reference=` names one). Pass
  `idf=` to run every scenario on a *copy* of a model you already built and
  inspected — geometry, envelope and shading shared by construction, built once,
  with only the behaviour-dependent wiring re-applied (KPIs identical to
  rebuilding). Pass `idf_path=` instead to run the model **as it stands on disk**,
  so hand edits to the `.idf` are what get simulated. Pass `context_gdf=` (the surrounding buildings) for the
  census-tract DHW occupancy; without it that rate cannot be derived from one row
  and the CTE-default density is used. Non-residential use types are reported
  once under the `normative` scenario label (they run the same RITE user in every
  scenario); `nonresidential_reporting="per_scenario"` restores the old output.
- **`zone_use_categories(idf)`** — `{ZONE: use category}` off the builder's zone
  metadata. Feed it to `reporting.scenario_report(zone_uses=…)` so each use is
  reported on its own page instead of averaged with the others.
- **`comfort_zone_polygon(cfg=None)`** — `(t, w_lo, w_hi)` tracing the extended
  (Mollier h-x) comfort field of EN ISO 7730 / DIN 1946-2 / ASHRAE 55, derived by
  intersecting the temperature, relative-humidity, absolute-humidity and enthalpy
  bounds of the `comfort_zone` config at every temperature. The psychrometrics it
  rests on (`_p_sat`, `_humidity_ratio`, `_relative_humidity`,
  `_moist_air_enthalpy`) live here too — `reporting.py` draws the model, it does
  not define one.
- **`configure(path)` / `get_config()`** — load/read the YAML defaults
  (`energy_config.yaml`).
- **`building_archetype(...)`**, **`building_wwr(...)`** — Spanish archetype
  envelopes and window-to-wall-ratio prediction.

### Per-space floorplan synthesis

`synthesize_space_geometry(row, ...)` turns the lightweight `br__thermal_zones`
list into an on-demand per-space, per-floor polygon partition (the shared
geometry behind the IDF, the 3-D view and `utils.plot_building_thermal_zones`).
Guarantees for each floor:

- **Exact tiling** — the building spaces plus the communal core fill the cadastral
  footprint exactly; the footprint (not the CAT constructed area, which can over- or
  under-subscribe it) is the physical constraint.
- **Single-connected, area-proportional zones** — each space and the communal core
  is one connected polygon, and spaces share the floor in proportion to their CAT
  area (`_bisect_connected`, a connectivity-preserving area-balanced bisection).
- **Communal interior; every space fronts the façade** — the communal stair/lift
  core is a compact block kept off the main façade; dwellings, shops and
  parking/warehouse each get main-façade frontage (≥ `_MIN_FACADE_FRONTAGE`, 3 m,
  for access / windows / the vehicle door), and a space that cannot reach it is
  flagged `facade_deficit`.
- Attached/row houses use a **column** layout (a `building_space_reference` stacks
  its garage under its dwelling), apartments a **central-core** layout, and
  spread-addressed houses an **anchored** layout near each real address.

The taller ground storey (`ground_floor_height`) is applied to buildings with ≥ 5
floors above ground **or** industrial/agricultural buildings, via
`utils.effective_ground_floor_height(...)`.

- **`utils.plot_building_thermal_zones(gdf, building_reference, ...)`** — per-floor
  2-D plan of the estimated thermal-zone distribution (occupied zones by use type,
  the communal core, non-occupied parking/pools), each zone labelled with its
  building-space reference and area.

## 🧮 social.py — INE socioeconomic indicators

Attaches INE socioeconomic/demographic indicators to every building via the
`social_ES` package (EUPL-1.2, ≥ 1.0.3), joining each source dataset at its
finest geographic level (census section → district → municipality → province)
from the codes added by `census_layer`. Wired into `merge()` through
`social_ES_layer` and executed **before** the POI join. Whole-Spain source files
are cached once under `{wd}/INE/…`.

- **`join_social_ES(gdf, wd, n_years=5, datasets=None, use_CAT_files=False)`** —
  adds `so__{indicator}_{year}` columns (`so__` prefix, snake_case indicator,
  year as a suffix) for the most recent
  `n_years` of each dataset in `ALL_DATASETS` (`atlas`, `population`,
  `education`, `dwellings`, `empty_dwellings`, `rental`, `price`,
  `electricity`), plus — when `use_CAT_files=True` and selected — the Essential
  Characteristics downscaling. The curated `*_INDICATORS` maps are deliberately
  non-overlapping: every indicator comes from the one dataset that publishes it
  at the finest geography over the longest run of years.
- **`join_essential_characteristics(gdf, wd, ec_variables=EC_SELECTED_VARIABLES)`**
  — downscales the INE *Essential Characteristics of Population and Households*
  survey to building level (needs the CAT-based `br__*` columns), adding
  `ec__{variable}~{class}` proportion columns. `social_ES` names them in English
  snake_case and drops its own duplicate variables, so **the names are kept
  verbatim** and only the `ec__` prefix is added. By default only
  `EC_SELECTED_VARIABLES` (27 of the ~70) is kept — what the `so__` layer and the
  cadaster do not already carry; `ec_variables=None` keeps all of them. The survey
  is published only for larger municipalities.
- **`time_use_household_occupancy(wd, census_tract_code)`** — a census tract's
  weekly household occupancy agenda from the INE Time Use Survey
  (`social_ES.INE.TimeUseSurvey` → `WeeklyScheduleHouseholdOccupancy`): the 168
  hourly `Sleeping`/`WithActivity`/`NonOccupied` states of the survey cell the
  tract's autonomous community and income band fall into, plus the per-state
  household shares. `energy.py` turns it into the residential occupancy schedule
  of the `aware`/`unaware` scenarios; `None` for tracts the income atlas misses.
- **`mergers.join_census_tract_aggregations(gdf)`** — aggregates the frame by
  census tract and writes the result onto every building as `ctagg__*` (tract
  building/dwelling counts, residential floor area, and the
  `ctagg__people_per_m2` / `ctagg__people_per_dwelling` occupancy densities the
  energy model reads). Called from `hc.merge` after the social layer. Densities
  are left NaN for tracts the frame only partly covers, so a partial frame never
  publishes an inflated one.
- **`release(*frames)`** — empties a dataset's source tables as soon as its wide
  indicator table exists, so the eight are not all held at once (the disk cache
  under `{wd}/INE` is untouched). Tens of MB, not GB: the tables arrive already
  filtered to the municipalities being enriched, and the layer's real peak is
  inside `social_ES` while it reads the national files.

## 📊 reporting.py — HTML reports of a simulation

- **`scenario_report(kpis, output_path_html, output_dir=None, zone_uses=None, ...)`**
  — self-contained interactive pages from an `energy.run_behavior_scenarios`
  result, **one per USE**: each carries that use's summarised comparisons (demand,
  gains, delivered setpoint, discomfort, the indoor-temperature percentile bands,
  DHW) and, when `output_dir` points at the runner's scenario directories, the
  **raw hourly** EnergyPlus series behind them (daily temperature/energy/solar,
  a **psychrometric comfort chart** of every occupied hour against the EN 16798-1
  adaptive band with a season selector, plus an hourly window over the hottest
  days) restricted to that use's zones.
  Files are named `{stem}_{use}.html` and the call returns `{use: path}`.
  `zone_uses` (from `energy.zone_use_categories(idf)`, else the builder's
  `.hcmeta.json` sidecar) is what separates them — without the split, the parking
  box lifts the dwellings' air-change curve and drags their CO₂ towards the
  outdoor background. Plotly is inlined, so it needs no network.

## 🌤️ weather.py — climate data → EPW

Nearest-cell ERA5-Land extraction (with coastal fallback) feeding degree-day
statistics and cached per-year EPW files:

- **`get_municipality_weather(era5_dir, lat, lon, year, ...)`** — station-style
  hourly timeseries for the closest cell with data.
- **`weather_statistics(df_station, hdd_base=18, cdd_base=21, ...)`** — HDD/CDD
  and temperature/humidity/wind/solar summaries.
- **`municipality_epw(era5_dir, epw_dir, lat, lon, year, ...)`** — path to a
  cached EnergyPlus EPW file (built on demand).
- **`join_weather(gdf, era5_dir, years=5, ...)`** — the `merge(weather_layer=True)`
  orchestrator: snaps every building to its nearest ERA5-Land cell and adds annual
  `we__{indicator}_{year}` columns (`WEATHER_INDICATORS`).
- **`load_ERA5Land(...)`** — gathers ERA5-Land with the **ARCO analysis-ready Zarr**
  stores first (`_load_ARD`, no CDS queue; `HC_ERA5_ARD=0` to disable) and the classic
  `cdsapi` retrieve (GRIB/CSV) as fallback; the grid extent adapts to the requested
  province.

## 🗄️ _db.py — DuckDB persistence (internal)

The DuckDB backend that replaced pickle artifacts in v1.1.0: connection
lifecycle, schema DDL, CRS sidecar, and GeoDataFrame ↔ DuckDB round-trips.

**Tables** written under `results/building_results.duckdb`:

| Table | Contents |
|-------|----------|
| `buildings` | Final merged building GeoDataFrame per municipality |
| `br_results` | Building-reference inference output (shadows, wall segments) |
| `sbr_results` | Space-reference inference output |
| `cat_building_spaces` | Parsed CAT per-space table, cached per `cadaster_code` (feeds the area pivot / inference) |
| `cat_building_attributes` | Per-building `br__*` CAT aggregates (`br_attributes_from_CAT` output), cached per `cadaster_code` — reused on every re-run (the ~99 s→0.4 s/municipality speedup); clear with `hc.utils.clear_cat_cache` |
| `aux_building_parts` | Building parts + proximity flags |
| `aux_footprints` | Floor footprints |
| `_meta`, `zone_status` | CRS store and per-zone progress ledger |

**Reading results** (see also `README.md`):

- `hc.utils.read_br_inferred_indicators(cadaster_code, wd)` → pandas DataFrame.
- `hc.utils.get_heavy_columns_for_building(wd, code, ref, table=...)` — fetch the
  huge BLOB columns one building at a time (they are excluded from `merge()`
  output to keep memory bounded).
- `hc.ResultsHandle(db_path, cadaster_code)` — lazy relations / batch iterators
  / `.sbr_to_pandas()` / `.br_to_pandas()` escape hatches.
- `hc._db.open_db(results_dir, read_only=True)` — raw connection for custom SQL.

## 🔄 Data flow

```
download()  →  merge()  →  [building_parts_inference]  →  energy.py + weather.py
 (downloaders)  (mergers)      (building_inference)          (IDF + EPW)
       └──────────────── DuckDB persistence (_db.py) ───────────────┘
```

---

[← Installation & Quick Start](installation-quickstart.md) | [Data Sources & Coverage →](data-sources-coverage.md) | [Back to README](../README.md)
