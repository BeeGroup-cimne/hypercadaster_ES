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
    social_ES_layer=True,             # income atlas, population, education, prices, electricity
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
# Adds so__{indicator}_{year} columns (income, population, education, rental &
# housing prices, electricity) joined at each dataset's finest geographic level.
# With use_CAT_files=True, the per-building Essential Characteristics survey is
# also downscaled to English-keyed ec__* proportion columns.
gdf = hc.merge(
    wd, cadaster_codes=["25900"],     # Lleida city
    census_layer=True,                # required — provides section_code
    use_CAT_files=True,               # enables the ec__* Essential Characteristics
    social_ES_layer=True,
    # Optional: pick a subset of datasets
    # social_ES_datasets=["atlas", "education", "essential_characteristics"],
)

# For a frame produced by an older run with Spanish ec__ names:
gdf = hc.social.rename_ec_columns(gdf)
```

> The first `social_ES_layer=True` run downloads and caches the whole-Spain INE
> source files under `{wd}/INE/…` (a slow one-off); later runs reuse the cache.
> `social_ES` is EUPL-1.2 licensed.

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

hc.energy.plot_idf_geometry(f"./data/energyplus/{ref}.idf")   # 3-D viewer
# idf.run()   # requires a local EnergyPlus install
```

Defaults (storey heights, constructions, WWR, loads, schedules) come from
`energy_config.yaml`. Override them with your own YAML:

```python
hc.energy.configure("./my_energy_config.yaml")   # merged on top of defaults
cfg = hc.energy.get_config()
```

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