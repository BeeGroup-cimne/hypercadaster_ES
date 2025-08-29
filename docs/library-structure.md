# Library Structure Documentation

This document provides comprehensive documentation of the hypercadaster_ES library structure, including detailed descriptions of all modules, functions, and their capabilities.

## 📁 Core Modules Overview

```
hypercadaster_ES/
├── 📋 __init__.py              # Main package interface
├── 🎮 functions.py             # Primary entry points (download, merge)
├── 🔗 mergers.py              # Data joining and merging operations
├── 📥 downloaders.py          # Data download functionality
├── 🛠️ utils.py                # Utility functions and helpers
├── 🏗️ building_inference.py   # Advanced building analysis
└── 🔌 interoperability.py     # External system integrations
```

## 🎮 functions.py - Main Entry Points

The primary interface providing the two main functions users interact with:

### Main Functions

#### `download(wd, province_codes=None, ine_codes=None, cadaster_codes=None, ...)`

Downloads all required datasets for specified geographic areas.

**Parameters:**
- `wd` (str): Working directory path
- `province_codes` (list): 2-digit province codes
- `ine_codes` (list): 5-digit INE municipality codes  
- `cadaster_codes` (list): Municipal cadastral codes
- `neighborhood_layer` (bool): Include Barcelona neighborhoods
- `postal_code_layer` (bool): Include postal code boundaries
- `census_layer` (bool): Include census tract data
- `elevation_layer` (bool): Include elevation data
- `open_data_layers` (bool): Include Barcelona open data
- `force` (bool): Force re-download existing files

**Features:**
- Supports province-level, municipality-level, or cadastral code-based selection
- Configurable data layers (neighborhoods, postal codes, census, elevation, etc.)
- Automatic directory structure creation
- Progress tracking and error handling

#### `merge(wd, province_codes=None, ine_codes=None, cadaster_codes=None, ...)`

Processes and merges all downloaded data into unified GeoDataFrame.

**Parameters:**
- All download parameters plus:
- `building_parts_inference` (bool): Enable advanced building analysis
- `building_parts_plots` (bool): Generate visualization plots
- `use_CAT_files` (bool): Use detailed CAT format files
- `CAT_files_rel_dir` (str): CAT files directory path

**Features:**
- Optional advanced building inference and analysis
- Configurable output complexity and detail level
- Duplicate removal and data quality assurance
- Memory-efficient processing options

## 🔗 mergers.py - Data Integration Engine

Comprehensive data joining and integration functionality.

### Cadastral Data Integration

#### `join_cadaster_data()`
Main orchestrator for all cadastral data joins. Coordinates the integration of buildings, addresses, zones, and parcels.

#### `get_cadaster_address()` 
Extract and process address information with multiple sources:
- Street names and types
- Building numbers and specifications
- Coordinate information
- Administrative context

#### `join_cadaster_building()`
Building geometry, attributes, and space classification:
- Building footprint polygons
- Construction years and status
- Usage types and dwelling counts
- Floor information and areas

#### `join_cadaster_zone()`
Cadastral zoning and administrative boundary integration:
- Zone references and classifications
- Urban vs. disseminated areas
- Administrative boundaries

#### `join_cadaster_parcel()`
Parcel geometry and property boundary data:
- Parcel polygons and centroids
- Property boundaries
- Ownership context

#### `join_adm_div_naming()`
Administrative division names and metadata:
- Municipality, province, and autonomous community names
- Official naming and codes

### External Data Integration

#### `join_DEM_raster()`
Digital Elevation Model height extraction:
- Ground elevation extraction
- Terrain analysis preparation
- Elevation-based calculations

#### `join_by_census_tracts()`
INE census tract demographic and administrative data:
- Census boundaries
- Demographic indicators
- Statistical context

#### `join_by_neighbourhoods()`
Barcelona-specific neighborhood boundary integration:
- Neighborhood boundaries
- District relationships
- Local administrative context

#### `join_by_postal_codes()`
Spanish postal code boundary and routing information:
- Postal delivery areas
- Address validation
- Geographic routing context

### Utility Functions

#### `make_valid()`
Fix invalid geometries in GeoDataFrames for robust spatial operations.

## 📥 downloaders.py - Data Acquisition

Manages downloading from various Spanish government and open data sources.

### Core Download Infrastructure

#### `download_file()`
Generic HTTP file downloader with progress tracking:
- Resume incomplete downloads
- Progress bars and status reporting
- Error handling and retry logic

#### `cadaster_downloader()`
Spanish Cadastral Service (Catastro) data acquisition:
- Building, parcel, and address data
- Multiple data formats (SHP, GML)
- Batch processing for multiple areas

#### `download_postal_codes()`
Spanish postal code boundary downloads:
- National postal code coverage
- Boundary polygons
- Delivery area information

#### `download_census_tracts()`
INE census tract boundary and demographic data:
- Census section boundaries
- Statistical data integration
- Administrative hierarchy

#### `download_DEM_raster()`
National Geographic Institute elevation raster data:
- Digital elevation models
- Terrain analysis data
- Multi-resolution options

### Barcelona Open Data Integration

#### `load_and_transform_barcelona_ground_premises()`
Commercial establishment data processing:
- Business location data
- Activity type classification
- Ground floor commercial analysis

**Additional Barcelona Features:**
- Automatic neighborhood and district boundary processing
- Street addressing and location reference integration
- Enhanced building context data

### Advanced CAT File Processing

#### `parse_CAT_file()`
Parse Spanish cadastral CAT format files:
- Horizontal property divisions
- Detailed space information
- Building usage classification
- Area calculations by space type

**CAT File Capabilities:**
- Process horizontal property divisions (condominiums)
- Extract individual unit information and characteristics
- Building space usage type classification
- Floor-by-floor area calculations and distributions

## 🛠️ utils.py - Utility Functions & Helpers

Comprehensive collection of utility functions organized by functional category.

### Parallel Processing Utilities

#### `tqdm_joblib()`
Context manager for progress tracking in parallel operations, enabling efficient processing of large datasets.

### Directory Management

#### Path Generation Functions
- `create_dirs()`: Create complete directory structure
- `cadaster_dir_()`, `results_dir_()`: Specialized path generators
- `open_street_dir_()`, `DEM_raster_dir_()`: Component-specific paths
- `postal_codes_dir_()`, `neighborhoods_dir_()`: Data layer paths

### Data Conversion & Code Mapping

#### Municipality and Code Management
- `list_municipalities()`: Available municipalities from cadastral service
- `ine_to_cadaster_codes()`: Convert INE codes to cadastral codes
- `cadaster_to_ine_codes()`: Convert cadastral codes to INE codes
- `municipality_name()`: Official municipality names
- `get_administrative_divisions_naming()`: Administrative metadata

### File Operations

#### Archive and Format Processing
- `unzip_directory()`: Batch ZIP file extraction
- `untar_directory()`: TAR archive extraction with filtering
- `kml_to_geojson()`: KML to GeoJSON conversion

### Geometric Utilities

#### Basic Geometric Operations
- `get_bbox()`: Calculate bounding boxes from GeoDataFrames
- `make_valid()`: Fix invalid geometric objects
- `concatenate_tiffs()`: Merge multiple GeoTIFF files
- `create_graph()`: Build spatial relationship graphs

### Spatial Analysis Functions

#### Building Proximity Analysis
- `detect_close_buildings()`: Identify spatially proximate buildings
- `detect_close_buildings_parallel()`: Parallel processing version
- `detect_close_parcels_parallel()`: Parcel proximity analysis

#### Advanced Geometric Processing
- `union_geoseries_with_tolerance()`: Geometric union with gap filling
- `calculate_floor_footprints()`: Floor-by-floor geometric calculations
- `get_all_patios()`: Extract interior courtyards and patios
- `patios_in_the_building()`: Determine patio-building relationships

#### Geometric Cleaning and Standardization
- `remove_duplicate_points()`: Clean coordinate sequences
- `normalize_polygon()`: Standardize polygon vertex ordering
- `unique_polygons()`: Remove duplicate geometries

### Building Analysis Functions

#### Orientation and Facade Analysis
- `calculate_wall_outdoor_normal_orientation()`: Building facade orientation
- `create_ray_from_centroid()`: Generate directional rays
- `segment_intersects_with_tolerance()`: Geometric intersection analysis
- `distance_from_points_to_polygons_by_orientation()`: Directional distances
- `distance_from_centroid_to_polygons_by_orientation()`: Building relationships

### Street & Infrastructure Analysis

#### OpenStreetMap Integration
- `get_municipality_open_street_maps()`: OSM data integration
- `detect_number_of_orientations()`: Analyze building orientation complexity
- `is_segment_in_contact_with_street()`: Building-street relationships
- `split_linestring_to_segments()`: Linear geometry processing

### Data Processing & Classification

#### Statistical and Classification Functions
- `weighted_circular_mean()`: Circular statistics for orientation
- `discrete_orientation()`: Discretize continuous orientations
- `classify_above_ground_floor_names()`: Spanish floor naming
- `classify_below_ground_floor_names()`: Basement classification
- `classify_cadaster_floor_names()`: Complete floor standardization
- `agg_op()`: Advanced aggregation operations

### Result Management

#### Data Loading Functions
- `read_br_inferred_indicators()`: Building-level analysis results
- `read_sbr_inferred_indicators()`: Space-level analysis results
- `read_addresses_indicators()`: Address-level analysis results

### Visualization Utilities

#### Comprehensive Plotting Functions
- `plot_shapely_geometries()`: Geometric plotting with clustering
- `plot_points_with_indices()`: Point visualization with labels
- `plot_polygons_group()`: Multi-polygon comparisons
- `plot_linestrings_and_polygons()`: Mixed geometry visualization

#### Advanced Geometric Operations
- `cut()`: Slice LineString geometries between distances
- `shorten_linestring()`: Trim LineStrings by percentage

## 🏗️ building_inference.py - Advanced Building Analysis

Computationally intensive building analysis and inference pipeline.

### Main Building Analysis Pipeline

#### `process_building_parts()`
Primary building analysis orchestrator providing:
- Comprehensive geometric analysis and space inference
- Building proximity detection and clustering analysis
- Multi-scale spatial relationship detection
- Floor footprint calculation and validation
- Shadow pattern analysis and environmental assessment

### Zone-Level Processing Functions

#### `process_zone()`
Detailed analysis for individual cadastral zones:
- Building orientation analysis with street relationship detection
- Parcel geometry analysis and space optimization
- Shadow pattern calculation for all floor levels
- Environmental context analysis (sunlight, wind patterns)
- Building type classification and usage inference

### CAT File Processing & Analysis

#### `parse_horizontal_division_buildings_CAT_files()`
Extract detailed building data from CAT format:
- Process horizontal property divisions (condominiums)
- Extract individual unit information and characteristics
- Building space usage type classification
- Floor-by-floor area calculations and distributions

#### `aggregate_CAT_file_building_spaces()`
Building space aggregation and summary:
- Aggregate individual spaces by building reference
- Calculate usage type percentages and distributions
- Generate floor-level summaries and statistics
- Infer missing building characteristics from space data

## 🔌 interoperability.py - External System Integration

Bridges to external analysis and simulation tools.

### Energy Simulation Integration

#### `input_files_for_IREC_simulations()`
Creates the needed input dataset for IREC's building energy simulations:
- Building type classification (single-family, multi-family, etc.)
- Usage percentage calculations (residential, commercial, office, parking)
- Construction year analysis and renovation detection
- Floor distribution analysis for energy modeling
- Location and climate zone assignment

### Visualization & Analysis Support

#### Specialized Plotting Functions
- `plot_weather_stations()`: Weather cluster visualization for climate analysis
- `converter_()`: Data structure mapping utilities for external tool compatibility

## 🔄 Data Flow Architecture

### Typical Processing Pipeline

1. **Download Phase** (`functions.download()`)
   - `downloaders.py` handles all data acquisition
   - `utils.py` manages directory structure and file operations

2. **Merge Phase** (`functions.merge()`)
   - `mergers.py` integrates all data sources
   - `utils.py` provides geometric and processing utilities

3. **Analysis Phase** (optional, `building_parts_inference=True`)
   - `building_inference.py` performs advanced analysis
   - `utils.py` provides specialized analysis functions

4. **Export Phase** (optional)
   - `interoperability.py` prepares data for external tools
   - `utils.py` provides visualization and export utilities

## 🎯 Usage Patterns

### Basic Usage (Address-focused)
Uses primarily `functions.py`, `downloaders.py`, and `mergers.py` for straightforward data integration.

### Advanced Analysis
Adds `building_inference.py` for comprehensive building analysis with geometric inference.

### Energy Simulation Workflow
Incorporates `interoperability.py` for preparing datasets compatible with building energy simulation tools.

### Custom Analysis
Leverages `utils.py` functions directly for specialized spatial analysis and visualization workflows.

---

[← Installation & Quick Start](installation-quickstart.md) | [Data Sources & Coverage →](data-sources-coverage.md) | [Back to README](../README.md)