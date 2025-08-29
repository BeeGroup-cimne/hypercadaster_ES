# Changelog & Version History

All notable changes to hypercadaster_ES will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

#### Short-term Improvements (v1.1.x)
- Performance optimizations for large-scale processing
- Enhanced error handling and user feedback
- Additional data validation and quality checks
- Expanded documentation and examples

#### Medium-term Features (v1.2.x)
- Basque Country and Navarre cadastral integration
- Enhanced visualization and reporting capabilities
- Additional external tool integrations

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