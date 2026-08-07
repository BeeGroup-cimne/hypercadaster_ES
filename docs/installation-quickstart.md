# Installation & Quick Start Guide

This guide provides step-by-step instructions for installing hypercadaster_ES and getting started with basic usage.

## 🚀 Installation

### Method 1: Direct from pip (Recommended)

The simplest way to install hypercadaster_ES:

```bash
pip install hypercadaster-ES
```

### Method 2: From GitHub source (Development version)

For the latest development version or if you want to contribute:

```bash
# Clone the repository
git clone https://github.com/BeeGroup-cimne/hypercadaster_ES.git
cd hypercadaster_ES

# Create virtual environment
python -m venv hypercadaster_env
source hypercadaster_env/bin/activate  # Linux/Mac
# or: hypercadaster_env\Scripts\activate  # Windows

# Install setuptools and build
pip install setuptools wheel
python setup.py sdist
pip install dist/hypercadaster_es-1.1.0.tar.gz
```

## 🏃‍♂️ Quick Start

### Basic Usage

The simplest way to get started with hypercadaster_ES:

```python
import hypercadaster_ES as hc

# Download data for Barcelona municipality
hc.download("./data", cadaster_codes=["08900"])

# Merge all data into a unified GeoDataFrame (persisted to building_results.duckdb internally)
gdf = hc.merge("./data", cadaster_codes=["08900"])

# Optional: save the GeoDataFrame for later use (e.g., as Parquet or pickle)
gdf.to_parquet("./data/barcelona_data.parquet")  # or to_pickle with compression
```

### What this does:
1. **Downloads** all available cadastral data for Barcelona (code 08900)
2. **Merges** building, address, and contextual data into a single dataset
3. **Persists** the result to a DuckDB database (`./data/results/building_results.duckdb`)
4. **Returns** a GeoDataFrame for immediate use (or save it for later)

### Accessing Results via DuckDB

Instead of (or in addition to) saving the GeoDataFrame, you can query the persistent DuckDB directly:

```python
# Read inferred indicators from DB (returns pandas DataFrame)
br = hc.utils.read_br_inferred_indicators("08900", "./data")

# Fetch heavy shadow/terrain columns for one building (on demand)
heavy = hc.utils.get_heavy_columns_for_building(
    "./data", "08900", "8341105DF2884A"
)

# Access other tables via ResultsHandle (advanced)
from hypercadaster_ES import _db
handle = hc.ResultsHandle("./data/results/building_results.duckdb", "08900")
spaces_df = handle.sbr_to_pandas()  # space-level results
```

### Advanced Usage Example

For more comprehensive analysis with all available features:

```python
import hypercadaster_ES as hc

# Download with all data layers
hc.download(
    wd="./barcelona_complete",
    cadaster_codes=["08900"],  # Barcelona
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevation_layer=True,
    open_data_layers=True
)

# Complete analysis with building inference
gdf = hc.merge(
    wd="./barcelona_complete",
    cadaster_codes=["08900"],
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    building_parts_inference=True,  # Enable advanced analysis
    building_parts_plots=True,      # Generate plots
    use_CAT_files=True,             # Use detailed CAT data
    CAT_files_rel_dir="CAT_files"   # CAT files directory
)

# Save results in columnar format for efficient reuse
gdf.to_parquet("./barcelona_complete/barcelona_data.parquet")

# Example: compute average building area per district using DuckDB
from hypercadaster_ES import _db
con = _db.open_db("./barcelona_complete/results", read_only=True)
result = con.execute("""
    SELECT district_name, AVG(building_area) as avg_area_m2
    FROM buildings
    WHERE cadaster_code = '08900'
    GROUP BY district_name
    ORDER BY avg_area_m2 DESC
""").fetchdf()
print(result.head())
con.close()
```

## 🗂️ Selecting a Geographic Area

Both `download()` and `merge()` accept three interchangeable (and combinable)
selectors — pass the same arguments to both:

| Argument | Granularity | Example |
| --- | --- | --- |
| `cadaster_codes` | Cadastral municipality | `["08900"]` (Barcelona) |
| `ine_codes` | INE municipality | `["08019", "28079"]` (Barcelona, Madrid) |
| `province_codes` | Whole province | `["08"]` (Barcelona province) |

```python
hc.download("./data", ine_codes=["08019", "28079", "46250"])
gdf = hc.merge("./data", ine_codes=["08019", "28079", "46250"])
gdf.to_parquet("./data/multi_city.parquet")
```

> **Scale note:** `building_parts_inference=True` is single-municipality only
> and memory-intensive. For provinces, leave it `False` (the default).
> Barcelona-only layers (`neighborhood_layer`, `open_data_layers`) must be
> disabled for other municipalities.

## 💾 Data Persistence & Storage

hypercadaster_ES uses DuckDB as its central persistence layer (replacing the previous pickle-based storage). Key points:

- **Central Database**: `building_results.duckdb` in your results directory
- **Memory Efficient**: Heavy BLOB columns (shadows, terrain grids) are loaded on demand
- **ACID Transactions**: Safe concurrent reads and writes
- **Spatial Extension**: Native GEOMETRY support for efficient spatial queries
- **CRS Preservation**: Coordinate reference systems stored and restored automatically

### Working with the DuckDB Directly

```python
import duckdb
from hypercadaster_ES import _db

# Open connection (read-only recommended for querying)
con = _db.open_db("./data/results", read_only=True)

# Query building counts per municipality
result = con.execute("""
    SELECT municipality_name, COUNT(*) as buildings
    FROM buildings
    GROUP BY municipality_name
    ORDER BY buildings DESC
""").fetchdf()

con.close()
```

### Accessing Specific Tables

- `buildings`: Final merged building data per municipality
- `br_results`: Building-reference inference output (shadows, wall segments, etc.)
- `sbr_results`: Space-reference inference output (detailed space-level data)
- `aux_building_parts`: Building parts + proximity flags
- `aux_footprints`: Floor footprints

### Fetching Heavy Columns On Demand

To avoid loading massive BLOB columns unnecessarily, use the helper functions:

```python
# Get heavy data for one building (returns dict of unpickled values)
heavy = hc.utils.get_heavy_columns_for_building(
    wd="./data",
    cadaster_code="08900",
    building_reference="8341105DF2884A",
    table="br_results"  # or "sbr_results"
)

# Common heavy columns: br__shadows, br__wall_segments, br__terrain_grid
```

## ☀️ EnergyPlus Integration (Optional)

> 📖 **[Energy Model & Scenarios](energy-model.md)** explains how the model is built
> and how the occupant-behaviour scenarios work.

For building energy simulation workflows:

```python
import hypercadaster_ES as hc
from hypercadaster_ES import weather

# After merging with building_parts_inference=True
gdf = hc.merge("./data", cadaster_codes=["08900"], building_parts_inference=True)

# Select a building for EnergyPlus modeling
building_ref = gdf.iloc[0]['building_reference']

# Generate EnergyPlus IDF (requires eppy+geomeppy installed)
idf = hc.energy.build_energy_model(
    gdf,
    building_reference=building_ref,
    wd="./data",  # for caching weather data
    weather_year=2022,  # attach real local weather
    epw_dir="./data/weather/epw"  # optional: specify EPW cache directory
)

# Save the IDF
idf.saveas("./data/energyplus/building_08900_8341105DF2884A.idf")

# To run the simulation (requires EnergyPlus installed):
# idf.run()
```

## 📊 Output Formats & Recommendations

- **GeoDataFrame**: Returned by `hc.merge()` for immediate processing
- **Parquet**: Recommended for saving intermediate results (columnar, efficient, compressible)
- **Pickle**: Still supported but not recommended for large datasets due to security and performance concerns
- **DuckDB**: Persistent storage for massive datasets and on-demand column loading
- **EnergyPlus IDF**: Generated via `hc.energy.build_energy_model()` for simulation workflows

---

*Last updated: July 2026 | Version 1.1.0*