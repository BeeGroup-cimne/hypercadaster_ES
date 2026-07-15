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

Turns the `br__*` inference columns into a 3-D, per-space thermal model and
exports an EnergyPlus **IDF** via `eppy` + `geomeppy` (both MIT — no AGPL
coupling). Public entry points:

- **`build_energy_model(gdf, building_reference, ...)`** → `geomeppy.IDF`
  (optionally attaches real weather via `weather_year`).
- **`plot_idf_geometry(idf_path, ...)`** — 3-D viewer for a generated IDF.
- **`configure(path)` / `get_config()`** — load/read the YAML defaults
  (`energy_config.yaml`).
- **`building_archetype(...)`**, **`building_wwr(...)`** — Spanish archetype
  envelopes and window-to-wall-ratio prediction.

## 🧮 social.py — INE socioeconomic indicators

Attaches INE socioeconomic/demographic indicators to every building via the
`social_ES` package (EUPL-1.2), joining each source dataset at its finest
geographic level (census section → district → municipality → province) from the
codes added by `census_layer`. Wired into `merge()` through `social_ES_layer`
and executed **before** the POI join. Whole-Spain source files are cached once
under `{wd}/INE/…`.

- **`join_social_ES(gdf, wd, n_years=5, datasets=None, use_CAT_files=False)`** —
  adds `so__{indicator}_{year}` columns (`so__` prefix, snake_case indicator,
  year as a suffix) for the most recent
  `n_years` of each dataset in `ALL_DATASETS` (`atlas`, `population`,
  `education`, `rental`, `price`, `electricity`), plus — when
  `use_CAT_files=True` and selected — the Essential Characteristics downscaling.
- **`join_essential_characteristics(gdf, wd)`** — downscales the INE *Essential
  Characteristics of Population and Households* survey to building level (needs
  the CAT-based `br__*` columns), adding English-keyed `ec__*` proportion
  columns. The survey is published only for larger municipalities.
- **`rename_ec_columns(df)` / `translate_ec_column()` / `ec_column_rename_map()`**
  — Spanish→English snake_case translation for the `ec__*` columns (applied
  automatically inside the merge; the helper is for pre-existing Spanish frames).

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
