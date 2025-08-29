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
pip install dist/hypercadaster_es-1.0.0.tar.gz
```

## 🏃‍♂️ Quick Start

### Basic Usage

The simplest way to get started with hypercadaster_ES:

```python
import hypercadaster_ES as hc

# Download data for Barcelona municipality
hc.download("./data", cadaster_codes=["08900"])

# Merge all data into a unified GeoDataFrame
gdf = hc.merge("./data", cadaster_codes=["08900"])

# Save results
gdf.to_pickle("barcelona_data.pkl", compression="gzip")
```

### What this does:
1. **Downloads** all available cadastral data for Barcelona (code 08900)
2. **Merges** building, address, and contextual data into a single dataset
3. **Saves** the result as a compressed pickle file

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
```

## 🗂️ Working with Different Geographic Areas

### By Municipality Codes (Cadastral)

```python
# Single municipality
hc.download("./data", cadaster_codes=["08900"])  # Barcelona

# Multiple municipalities
hc.download("./data", cadaster_codes=["08900", "28079", "46250"])  # Barcelona, Madrid, Valencia
```

### By Province Codes

```python
# Entire provinces (2-digit codes)
hc.download("./data", province_codes=["08", "28"])  # Barcelona province, Madrid province
```

### By INE Municipality Codes

```python
# INE codes (5-digit codes)
hc.download("./data", ine_codes=["08019", "28079"])  # Barcelona city, Madrid city
```

## 📁 Understanding the Output

After running the basic example, you'll have:

### Directory Structure
```
./data/
├── cadaster/           # Raw cadastral data
├── census_tracts/      # INE census data
├── postal_codes/       # Postal code boundaries
├── elevation/          # Digital elevation models
└── results/           # Processed results
```

### Output Data
The main result is a GeoDataFrame with comprehensive building information:
- **Building geometry and attributes**
- **Complete address information**
- **Administrative context** (municipality, census tract, postal code)
- **Environmental data** (elevation, orientation)
- **Advanced analysis results** (when inference is enabled)

## 🔧 Configuration Options

### Data Layers (Optional)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `neighborhood_layer` | Barcelona neighborhood boundaries | `False` |
| `postal_code_layer` | Spanish postal code boundaries | `False` |
| `census_layer` | INE census tract data | `False` |
| `elevation_layer` | Digital elevation model | `False` |
| `open_data_layers` | Barcelona open data layers | `False` |

### Analysis Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `building_parts_inference` | Advanced building analysis | `False` |
| `building_parts_plots` | Generate visualization plots | `False` |
| `use_CAT_files` | Use detailed CAT format files | `False` |

## 📋 Requirements

- **Python**: 3.10 or higher
- **Main dependencies**: pandas, geopandas, shapely, requests
- **Storage**: Variable (100MB-10GB depending on area size)
- **Memory**: 4GB+ recommended for large municipalities

## 🚨 Common Issues

### Large Dataset Warning
For large areas (entire provinces), disable inference for better performance:

```python
gdf = hc.merge(
    "./data",
    province_codes=["08"],
    building_parts_inference=False,  # Disable for speed
    building_parts_plots=False       # Disable for memory
)
```

### CAT Files Setup
CAT files must be downloaded separately from the [Spanish Cadastral Service](https://www.sedecatastro.gob.es/Accesos/SECAccDescargaDatos.aspx):

1. Visit the cadastral download portal
2. Download "Descarga de información alfanumérica por provincia (formato CAT)"
3. Place files in `CAT_files/` directory within your working directory

## 🎯 Next Steps

- See [Configuration & Examples](configuration-examples.md) for advanced workflows
- Check [Library Structure](library-structure.md) for detailed function reference
- Explore [Use Cases & Applications](use-cases-applications.md) for real-world examples
- Review [Output Data Schema](output-schema.md) to understand the data structure

---

[← Back to README](../README.md) | [Configuration & Examples →](configuration-examples.md)