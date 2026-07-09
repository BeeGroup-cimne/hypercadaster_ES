# hypercadaster_ES

A comprehensive Python library for downloading, processing, and analyzing Spanish cadastral data with integration of external geographic datasets and EnergyPlus model generation.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-EUPL%20v1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/BeeGroup-cimne/hypercadaster_ES)

## 🎯 Overview

**hypercadaster_ES** is a powerful Python library designed for comprehensive analysis of Spanish cadastral data. It provides tools for downloading official cadastral information, integrating it with external geographic datasets, performing advanced building analysis including geometric inference, orientation analysis, and exporting building models for energy simulation (EnergyPlus IDF).

### Key Features

- 🏘️ **Comprehensive Cadastral Data Access**: Download building, parcel, and address data from Spanish cadastral services
- 🗺️ **Multi-source Data Integration**: Merge with census tracts, postal codes, elevation models, and OpenStreetMap data
- 🏗️ **Advanced Building Analysis**: Geometric inference, floor footprint calculation, building space classification, shadow analysis
- 📊 **Orientation & Environmental Analysis**: Building orientation analysis, street relationships, shadow calculations, terrain analysis
- 🔌 **External Tool Integration**: Export capabilities for building energy simulation tools (EnergyPlus via eppy+geomeppy)
- 📈 **Scalable Processing**: Optimized for large-scale municipal and provincial analysis with DuckDB persistence
- ⚡ **Memory Efficient**: Central DuckDB database reduces memory footprint; heavy BLOB columns loaded on demand

## 📚 Documentation

### Getting Started
- [**Installation & Quick Start**](docs/installation-quickstart.md) - Installation methods, basic usage, and first steps
- [**Configuration & Examples**](docs/configuration-examples.md) - Advanced configuration and example workflows

### Library Reference
- [**Library Structure**](docs/library-structure.md) - Module overview and key entry points
- [**Data Sources & Coverage**](docs/data-sources-coverage.md) - Available datasets and geographic coverage
- [**Output Data Schema**](docs/output-schema.md) - Complete data structure and column reference
- **API Reference** - Generated from docstrings with [pdoc](https://pdoc.dev): run `./scripts/build_api_docs.sh` and open `docs/api/index.html`

### Development
- [**Contributing**](docs/contributing.md) - How to contribute to the project
- [**Changelog**](docs/changelog.md) - Version history and changes

## 🚀 Quick Start

### Library installation
```bash
pip install hypercadaster-ES
```

### Basic Usage
```python
import hypercadaster_ES as hc

# Download data for Barcelona municipality
hc.download("./data", cadaster_codes=["08900"])

# Merge all data into a unified GeoDataFrame (persisted to building_results.duckdb internally)
gdf = hc.merge("./data", cadaster_codes=["08900"])

# Optional: save the GeoDataFrame for later use (e.g., as Parquet or pickle)
gdf.to_parquet("./data/barcelona_data.parquet")  # or to_pickle(..., compression="gzip")
```

### Accessing Results via DuckDB
The central `building_results.duckdb` is the canonical artifact. Query
directly instead of materialising full GeoDataFrames:

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

For detailed installation instructions and advanced examples, see [Installation & Quick Start](docs/installation-quickstart.md).

## 🎯 Key Applications

- **Urban Planning & Municipal Management**: Building stock analysis, zoning compliance, infrastructure planning
- **Energy & Environmental Analysis**: Building energy modeling, carbon footprint analysis
- **Real Estate & Economic Analysis**: Property valuation, market analysis, location intelligence
- **Academic Research**: Urban geography, transportation research, social sciences applications

## 📊 Geographic Coverage

- **National Coverage**: Complete coverage of peninsular Spain, Balearic Islands, Canary Islands (except Basque Country and Navarre)
- **Enhanced Coverage**: Additional Barcelona open data layers
- **Scale Range**: Individual buildings to entire autonomous communities

## 🏗️ Advanced Features

- **Building Inference Engine**: Advanced geometric analysis, floor footprint calculation, orientation analysis, shadow and terrain analysis
- **Multi-source Integration**: Cadastral, census, elevation, postal, and OpenStreetMap data
- **CAT Files Support**: Detailed building space classification from official cadastral CAT format
- **Energy Simulation Ready**: Export formats compatible with building energy simulation tools (EnergyPlus IDF via hc.energy.build_energy_model)
- **POI Proximity Analysis**: Points of interest from OpenStreetMap with distance-based indicators
- **Commercial Premises Analysis**: Barcelona-specific ground floor commercial data

## 👥 Authors & Contributors

**Primary Authors:**
- **Jose Manuel Broto Vispe** - jmbrotovispe@gmail.com
- **Gerard Mor** - gmor@cimne.upc.edu

**Institutional Affiliations:**
- **CIMNE** - Centre Internacional de Mètodes Numèrics en Enginyeria, Building Energy and Environment (BEE) group
- **Universitat Politècnica de Catalunya (UPC)** - Technical University of Catalonia

## 📄 License

This project is licensed under the **EUPL v1.2**. See the [license](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) for details.

---

*hypercadaster_ES - Built with ❤️ for the Spanish urban analysis and building research community*

*Last updated: July 2026 | Version 1.1.0*