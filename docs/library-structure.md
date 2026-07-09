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
├── downloaders.py       # Data acquisition (Catastro, INE, DEM, Barcelona open data)
├── mergers.py           # Joins all sources into one GeoDataFrame; grid/H3 aggregation
├── building_inference.py# Geometric/space inference (br__* columns, shadows, footprints)
├── energy.py            # Per-space 3-D thermal model → EnergyPlus IDF
├── weather.py           # ERA5-Land → climate stats + EPW files
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
  - `compact_addresses` — collapse multi-address buildings into one row of lists.
  - `sociodemographic_layer`, `floor_height`, `ground_floor_height`, `num_workers`.

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

## 🌤️ weather.py — climate data → EPW

Nearest-cell ERA5-Land extraction (with coastal fallback) feeding degree-day
statistics and cached per-year EPW files:

- **`get_municipality_weather(era5_dir, lat, lon, year, ...)`** — station-style
  hourly timeseries for the closest cell with data.
- **`weather_statistics(df_station, hdd_base=18, cdd_base=21, ...)`** — HDD/CDD
  and temperature/humidity/wind/solar summaries.
- **`municipality_epw(era5_dir, epw_dir, lat, lon, year, ...)`** — path to a
  cached EnergyPlus EPW file (built on demand).

## 🗄️ _db.py — DuckDB persistence (internal)

The DuckDB backend that replaced pickle artifacts in v1.1.0: connection
lifecycle, schema DDL, CRS sidecar, and GeoDataFrame ↔ DuckDB round-trips.

**Tables** written under `results/building_results.duckdb`:

| Table | Contents |
|-------|----------|
| `buildings` | Final merged building GeoDataFrame per municipality |
| `br_results` | Building-reference inference output (shadows, wall segments) |
| `sbr_results` | Space-reference inference output |
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
