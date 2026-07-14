# Changelog & Version History

All notable changes to hypercadaster_ES will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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