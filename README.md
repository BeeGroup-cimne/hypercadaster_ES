# hypercadaster_ES

A comprehensive Python library for downloading, processing, and analyzing Spanish cadastral data with integration of external geographic datasets.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-EUPL%20v1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/BeeGroup-cimne/hypercadaster_ES)

## 🎯 Overview

**hypercadaster_ES** is a powerful Python library designed for comprehensive analysis of Spanish cadastral data. It provides tools for downloading official cadastral information, integrating it with external geographic datasets, and performing advanced building analysis including geometric inference, orientation analysis, and energy simulation data preparation.

### Key Features

- 🏘️ **Comprehensive Cadastral Data Access**: Download building, parcel, and address data from Spanish cadastral services
- 🗺️ **Multi-source Data Integration**: Merge with census tracts, postal codes, elevation models, and OpenStreetMap data
- 🏗️ **Advanced Building Analysis**: Geometric inference, floor footprint calculation, and building space classification
- 📊 **Orientation & Environmental Analysis**: Building orientation analysis, street relationships, and shadow calculations
- 🔌 **External Tool Integration**: Export capabilities for building energy simulation tools
- 📈 **Scalable Processing**: Optimized for large-scale municipal and provincial analysis

## 📚 Documentation

### Getting Started
- [**Installation & Quick Start**](docs/installation-quickstart.md) - Installation methods, basic usage, and first steps
- [**Configuration & Examples**](docs/configuration-examples.md) - Advanced configuration and example workflows

### Library Reference
- [**Library Structure**](docs/library-structure.md) - Detailed module documentation and function reference
- [**Data Sources & Coverage**](docs/data-sources-coverage.md) - Available datasets and geographic coverage
- [**Output Data Schema**](docs/output-schema.md) - Complete data structure and column reference

### Applications & Use Cases
- [**Use Cases & Applications**](docs/use-cases-applications.md) - Real-world applications and case studies

### Development
- [**Contributing**](docs/contributing.md) - How to contribute to the project
- [**Changelog**](docs/changelog.md) - Version history and changes

## 🚀 Quick Start


#### Copernicus Data Space
1. If you don't have a user in Copernicus Data Space Ecosystem, register in:
https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration?client_id=account-console&tab_id=I9M6-7Mx3Cw&client_data=eyJydSI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRS9hY2NvdW50LyMvcGVyc29uYWwtaW5mbyIsInJ0IjoiY29kZSIsInJtIjoicXVlcnkiLCJzdCI6ImNlMzljODM2LTRmYTgtNGRhNy05ZTZlLTZkNDkwZGEyMjE3YSJ9
2. Once you are logged in the portal: Go to Dashboard / SentinelHub, and inside the https://shapps.dataspace.copernicus.eu/dashboard/#/ 
3. Edit your User settings and add a new OAuth client with client name as: hypercadaster_ES.
4. We recommend to click on the Never expire button.
5. Once you create it, you'll be able to see the client ID and the secret. Store them in your system in ~/.dataspaces
```bash
nano ~/.dataspaces
```
6. The format of the file is the following (The hashes are examples, not real ones):
```bash
client:hypercadaster_ES
clientId:sh-djhf1a5a-9193-4262-b786-b912744ca675
clientSecret:P85fgGylvSV5y87HU8Z76HiJgZraoegQ
```

### Library installation
```bash
pip install hypercadaster-ES
```

### Basic Usage
```python
import hypercadaster_ES as hc

# Download data for Barcelona municipality
hc.download("./data", cadaster_codes=["08900"])

# Merge all data into a unified GeoDataFrame
gdf = hc.merge("./data", cadaster_codes=["08900"])

# Save results
gdf.to_pickle("barcelona_data.pkl", compression="gzip")
```

For detailed installation instructions and advanced examples, see [Installation & Quick Start](docs/installation-quickstart.md).

## 🎯 Key Applications

- **Urban Planning & Municipal Management**: Building stock analysis, zoning compliance, infrastructure planning
- **Energy & Environmental Analysis**: Building energy modeling, solar potential assessment, carbon footprint analysis
- **Real Estate & Economic Analysis**: Property valuation, market analysis, location intelligence
- **Academic Research**: Urban geography, transportation research, social sciences applications

See [Use Cases & Applications](docs/use-cases-applications.md) for detailed descriptions and examples.

## 📊 Geographic Coverage

- **National Coverage**: Complete coverage of peninsular Spain, Balearic Islands, Canary Islands (except Basque Country and Navarre)
- **Enhanced Coverage**: Additional Barcelona open data layers
- **Scale Range**: Individual buildings to entire autonomous communities

## 🏗️ Advanced Features

- **Building Inference Engine**: Advanced geometric analysis, floor footprint calculation, orientation analysis
- **Multi-source Integration**: Cadastral, census, elevation, postal, and OpenStreetMap data
- **CAT Files Support**: Detailed building space classification from official cadastral CAT format
- **Energy Simulation Ready**: Export formats compatible with building energy simulation tools

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

*Last updated: August 2025 | Version 1.0.0*