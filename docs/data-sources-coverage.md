# Data Sources & Coverage

This document provides comprehensive information about all data sources integrated by hypercadaster_ES, their coverage areas, update frequencies, and data characteristics.

## 📊 Spanish Official Government Data Sources

### Spanish Cadastral Service (Dirección General del Catastro)

The primary data source providing comprehensive property and building information across Spain.

#### **Buildings Data**
- **Content**: Footprint geometries, construction years, usage types, unit counts
- **Format**: Shapefile, GML
- **Coverage**: National (except Basque Country and Navarre)
- **Update Frequency**: Quarterly
- **Geometric Accuracy**: Sub-meter precision
- **Attributes Include**:
  - Building footprint polygons
  - Construction years and renovation dates
  - Usage classifications (residential, commercial, industrial, etc.)
  - Number of dwelling units per building
  - Floor counts and building height information

#### **Parcels Data**
- **Content**: Property boundaries, ownership information, land use classifications
- **Format**: Shapefile, GML
- **Coverage**: National (except Basque Country and Navarre)
- **Update Frequency**: Quarterly
- **Attributes Include**:
  - Cadastral parcel boundaries
  - Land use classifications
  - Property reference codes
  - Area calculations
  - Ownership context information

#### **Addresses Data**
- **Content**: Complete addressing with coordinates, street naming, postal integration
- **Format**: Shapefile, GML
- **Coverage**: National (except Basque Country and Navarre)
- **Update Frequency**: Quarterly
- **Attributes Include**:
  - Complete postal addresses
  - Coordinate locations (EPSG:25830/25831)
  - Street types and names
  - Building numbers and entrance specifications
  - Administrative context

#### **CAT Files Data**
- **Content**: Detailed property division data, horizontal ownership structures
- **Format**: CAT (custom cadastral format)
- **Coverage**: Provincial level downloads
- **Update Frequency**: Quarterly
- **Special Features**:
  - Individual space-level information
  - Horizontal property divisions (condominiums)
  - Detailed usage type classifications
  - Economic valuation data
  - Floor-by-floor space distributions

### National Statistics Institute (Instituto Nacional de Estadística - INE)

Spain's official statistical organization providing demographic and administrative data.

#### **Census Tracts Data**
- **Content**: Demographic boundaries, population statistics, socioeconomic data
- **Format**: Shapefile
- **Coverage**: National
- **Update Frequency**: Annual (boundaries), Census every 10 years (detailed demographics)
- **Geometric Accuracy**: High precision administrative boundaries
- **Attributes Include**:
  - Census section boundaries
  - Population counts and demographics
  - Household characteristics
  - Socioeconomic indicators
  - Age and gender distributions

#### **Administrative Divisions**
- **Content**: Official municipality, province, and autonomous community boundaries
- **Format**: Shapefile
- **Coverage**: National
- **Update Frequency**: Annual
- **Hierarchical Structure**:
  - Autonomous Communities (17)
  - Provinces (50)
  - Municipalities (8,100+)
  - Census Districts and Sections

#### **Population Data**
- **Content**: Age distributions, household characteristics, economic indicators
- **Format**: Statistical tables linked to geographic boundaries
- **Coverage**: National down to census section level
- **Update Frequency**: Annual estimates, decennial census

### National Geographic Institute (Instituto Geográfico Nacional - IGN)

Spain's national mapping agency providing topographic and elevation data.

#### **Digital Elevation Models (DEM)**
- **Content**: High-resolution terrain data, slope calculations, aspect analysis
- **Format**: GeoTIFF raster
- **Coverage**: National
- **Resolution**: 5-25 meter grid spacing
- **Update Frequency**: Periodic (5-10 years)
- **Vertical Accuracy**: ±1-2 meters
- **Coordinate System**: EPSG:25830 (Peninsula), EPSG:25828/25829 (Islands)

#### **Topographic Data**
- **Content**: Hydrography, transportation networks, land cover classifications
- **Format**: Various (Shapefile, GeoTIFF)
- **Coverage**: National
- **Scale**: 1:25,000 to 1:200,000

### Spanish Postal Service (Correos)

Official postal service providing addressing and delivery area information.

#### **Postal Code Boundaries**
- **Content**: Delivery area polygons, routing zones, address validation data
- **Format**: Shapefile
- **Coverage**: National
- **Update Frequency**: Annual
- **Postal Codes**: ~52,000 active codes
- **Attributes Include**:
  - Postal delivery area boundaries
  - Postal code numbers (5-digit)
  - Delivery route information
  - Urban vs. rural classifications

## 🌐 Regional & Municipal Data Sources

### Barcelona Open Data Portal

Enhanced data availability for Barcelona municipality providing detailed urban information.

#### **Commercial Establishments**
- **Content**: Business locations, activity types, licensing information
- **Format**: CSV, GeoJSON
- **Coverage**: Barcelona municipality
- **Update Frequency**: Monthly
- **Attributes Include**:
  - Business location coordinates
  - Activity classifications (CNAE codes)
  - Licensing status and dates
  - Establishment characteristics

#### **Ground Floor Premises**
- **Content**: Commercial space characteristics, accessibility features
- **Format**: CSV, GeoJSON
- **Coverage**: Barcelona municipality
- **Update Frequency**: Quarterly
- **Special Features**:
  - Ground floor commercial mapping
  - Accessibility information
  - Building entrance analysis
  - Street-level activity classification

#### **Neighborhood Boundaries**
- **Content**: Official district and neighborhood administrative boundaries
- **Format**: Shapefile, GeoJSON
- **Coverage**: Barcelona municipality
- **Update Frequency**: As needed (administrative changes)
- **Administrative Levels**:
  - 10 Districts
  - 73 Neighborhoods
  - Historical boundary evolution

#### **Enhanced Street Addressing**
- **Content**: Enhanced addressing system with building-level precision
- **Format**: Various
- **Coverage**: Barcelona municipality
- **Update Frequency**: Continuous
- **Features**:
  - Building-level address validation
  - Entrance and access point mapping
  - Multi-language street naming
  - Historical address evolution

### OpenStreetMap Integration

Community-contributed geographic data providing additional context and validation.

#### **Street Networks**
- **Content**: Road geometries, intersection analysis, accessibility metrics
- **Format**: OSM XML, Shapefile conversion
- **Coverage**: Global (used for Spanish municipalities)
- **Update Frequency**: Continuous community updates
- **Attributes Include**:
  - Street centerlines and intersections
  - Road classifications and hierarchy
  - Accessibility and transportation modes
  - Street names and alternative naming

#### **Points of Interest**
- **Content**: Public facilities, transportation nodes, commercial areas
- **Format**: OSM XML, Point data
- **Coverage**: Variable by municipality
- **Quality**: Community-contributed, varies by location

#### **Building Footprints**
- **Content**: Community-contributed building outline validation
- **Format**: OSM XML, Polygon data
- **Coverage**: Variable by municipality
- **Use**: Cross-validation with cadastral data

## 🗺️ Geographic Coverage & Scale

### National Coverage Areas

#### **Complete Coverage Regions**
- **Peninsular Spain**: Full data availability
- **Balearic Islands**: Complete coverage
- **Canary Islands**: Complete coverage
- **Total Municipalities**: ~6,500 covered municipalities

#### **Limited Coverage Regions**
- **Basque Country**: Limited cadastral data (separate cadastral system)
- **Navarre**: Limited cadastral data (separate cadastral system)
- **Alternative Sources**: Regional cadastral systems may be integrated in future versions

### Scale Range Capabilities

#### **Individual Building Level**
- **Minimum Unit**: Single building analysis
- **Detail Level**: Space-by-space breakdown (with CAT files)
- **Geometric Precision**: Sub-meter accuracy
- **Attributes**: 100+ building characteristics possible

#### **Neighborhood/District Level**
- **Aggregation**: Building clusters and neighborhoods
- **Analysis Scale**: 100-10,000 buildings
- **Processing Time**: Minutes to hours
- **Memory Requirements**: 1-8 GB

#### **Municipal Level**
- **Coverage**: Complete municipalities
- **Analysis Scale**: 1,000-500,000 buildings
- **Processing Time**: Hours to days
- **Memory Requirements**: 4-32 GB

#### **Provincial Level**
- **Coverage**: Entire provinces
- **Analysis Scale**: 100,000-2,000,000 buildings
- **Processing Time**: Days to weeks
- **Memory Requirements**: 16-128 GB

#### **Autonomous Community Level**
- **Coverage**: Multi-province regions
- **Analysis Scale**: Up to 8,000,000 buildings
- **Processing Time**: Weeks
- **Memory Requirements**: 64+ GB

### Enhanced Coverage Areas

#### **Barcelona Metropolitan Area**
- **Special Features**:
  - Enhanced open data integration
  - Neighborhood-level administrative boundaries
  - Commercial establishment data
  - Ground floor premises mapping
  - Enhanced addressing systems

#### **Future Enhanced Areas**
- **Madrid**: Potential for enhanced open data integration
- **Valencia**: Regional data integration possibilities
- **Seville**: Andalusian regional data opportunities

## 🔄 Data Update Frequency & Quality

### Update Schedules

#### **Real-time Updates**
- **OpenStreetMap**: Continuous community updates
- **Barcelona Open Data**: Monthly/quarterly for most layers

#### **Regular Updates**
- **Cadastral Data**: Quarterly official updates
- **Census Boundaries**: Annual updates
- **Postal Codes**: Annual updates

#### **Periodic Updates**
- **Elevation Data**: 5-10 year cycles
- **Administrative Boundaries**: As needed for administrative changes
- **Topographic Data**: 5-10 year update cycles

#### **Census Updates**
- **Population Estimates**: Annual
- **Detailed Census**: Every 10 years (2021, 2031, etc.)
- **Continuous Household Survey**: Annual sample-based updates

### Data Quality Characteristics

#### **Geometric Accuracy**
- **Cadastral Data**: Sub-meter precision (±0.5-2m)
- **Administrative Boundaries**: High precision (±1-5m)
- **Elevation Data**: ±1-2m vertical accuracy
- **OpenStreetMap**: Variable (±1-20m depending on source)

#### **Attribute Completeness**
- **Cadastral Data**: >95% completeness for core attributes
- **Census Data**: 100% coverage, statistical reliability varies
- **Commercial Data**: Variable by municipality (60-95%)

#### **Data Consistency**
- **Temporal Consistency**: Quarterly synchronization cycles
- **Spatial Consistency**: Automated topology validation
- **Attribute Consistency**: Standardized classification systems

## 🔍 Data Access and Limitations

### Access Methods

#### **Automated Download**
- **Cadastral Service**: REST API and batch download
- **INE Data**: FTP and web services
- **IGN Data**: Web services and download portals
- **Open Data**: REST APIs and data portals

#### **Manual Requirements**
- **CAT Files**: Manual download from cadastral portal
- **Some Provincial Data**: May require manual intervention
- **Historical Data**: Limited automated access

### Known Limitations

#### **Geographic Limitations**
- **Basque Country**: Separate cadastral system
- **Navarre**: Separate cadastral system
- **Small Islands**: Some data gaps in remote areas

#### **Temporal Limitations**
- **Historical Data**: Limited availability before 2010
- **Construction Dates**: Accuracy varies for older buildings
- **Address Evolution**: Limited historical address tracking

#### **Attribute Limitations**
- **Building Heights**: Estimated from floor counts
- **Usage Classifications**: May not reflect current use
- **Energy Performance**: Not directly available (requires modeling)

---

[← Library Structure](library-structure.md) | [Use Cases & Applications →](use-cases-applications.md) | [Back to README](../README.md)