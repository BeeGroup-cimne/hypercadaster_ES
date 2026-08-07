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

> **Note — addresses now come from CartoCiudad.** The INSPIRE Address feed
> (`ES.SDGC.AD`) and the Barcelona Open Data *carrerer* are deprecated; addresses
> are sourced from CartoCiudad (see below). The Cadastre is still used for parcels,
> buildings and CAT files.

#### **Addresses Data (CartoCiudad — CNIG)**
- **Content**: Nationwide postal address points (the CartoCiudad *portales* layer)
- **Source**: CNIG *Centro de Descargas* (`centrodedescargas.cnig.es`), auto-downloaded
  per province by name (one GeoPackage / province, `portalpk_publi` layer)
- **Format**: GeoPackage (EPSG:4258), reprojected to the cadastral CRS
- **Coverage**: National (all 52 provinces, incl. Basque Country and Navarre)
- **Update Frequency**: Monthly (the most recent published file is always fetched)
- **Attributes Include**:
  - Address point coordinates
  - Street type (`tipo_vial`) and name (`nombre_via`)
  - Building number + extension (`numero`, `extension`)
  - Postal code (`cod_postal`) and population entity (`poblacion`)
  - Municipality / province / autonomous-community and source metadata
- **Parcel link**: CartoCiudad points carry no cadastral reference, so each is
  assigned a `building_reference` by a spatial join to the cadastral parcels.
- **Output**: always compacted to one row per parcel, with per-address columns
  as aligned lists.

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

#### **Socioeconomic & Demographic Indicators** *(via the `social_ES` package, `social_ES_layer=True`)*
- **Source**: INE statistical operations, downloaded and cached whole-Spain under `{wd}/INE/…`
- **Requires**: `social_ES >= 1.0.3`
- **Datasets & finest level joined onto each building** — each indicator is taken
  from exactly one of them, so nothing is published twice under two names:
  - **Household Income Distribution Atlas** — census **section** (net/gross income
    per household·person·consumption-unit, a 2010-€ deflated series, income by
    source, Gini, P80/P20, mean age and household structure), 2015–2023
  - **Population Census** — census **section** (resident head count; foreign
    nationality/birth, born-in-municipality, sex and broad age-band shares), 2021–2025
  - **Education & Employment Census** — census **section** (education level and
    labour-force status shares), 2021–2024
  - **Dwellings and Population Census** — census **section** (dwelling stock,
    main-dwelling share, tenure of the main dwellings, households), 2021
  - **Empty and Secondary Dwellings Census** — **municipality**, the
    main/secondary/**empty** split harmonised across the 2001, 2011 and 2021
    censuses. Nationwide: INE withholds the split for small municipalities and
    `social_ES` models it there (province-calibrated), flagging those rows in
    `so__dwelling_use_predicted_{year}`
  - **Household Rental Price Index** — **district** where INE publishes it, the
    **municipal** series elsewhere, 2011–2024
  - **Housing Price Index** — **province** (quarterly → annual mean), 2007–2025
  - **Aggregated Electricity Consumption** — **district** (consumption percentiles), 2021
  - **Essential Characteristics of Population and Households** — per-building
    downscaling to `ec__{variable}~{class}` proportion columns (requires
    `use_CAT_files=True`); a curated 27 of the ~70 variables by default, the ones
    the census layer and the cadaster do not already cover — heating type and
    fuel, cooling, insulation, renewables, conservation status, accessibility,
    vehicles, housing cost burden (`ec_variables=None` keeps all ~256 columns).
    The INE survey is published only for ~147 larger municipalities nationwide
  - **Time Use Survey (EET 2009-2010)** — not joined as columns: a weekly
    **household occupancy agenda** per census tract (168 hourly
    `Sleeping`/`WithActivity`/`NonOccupied` states), matched by autonomous
    community and by the tract's `so__household_income_group_{year}` band and read
    by the energy model — `social.time_use_household_occupancy(wd, tract)`
- **Output columns**: `so__{indicator}_{year}` (`so__` namespace prefix,
  snake_case indicator, year as suffix); most recent `social_ES_years`
  (default 5) kept per dataset
- **License note**: `social_ES` is distributed under **EUPL-1.2**

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

#### **Enhanced Street Addressing** *(deprecated)*
- **Status**: **Deprecated** — the Barcelona *carrerer* street/address register is
  no longer downloaded or used. Addresses for Barcelona (and all of Spain) now come
  from CartoCiudad. The other Barcelona open-data layers (districts, neighborhoods,
  ground-floor commercial premises) are unaffected.

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

### Copernicus Climate Data Store (ECMWF ERA5-Land)

Hourly reanalysis climate data used for the `weather_layer` (annual per-building
indicators) and the EnergyPlus weather files (EPW) exported by the energy module.

#### **ERA5-Land Reanalysis**
- **Content**: 2 m air/dew-point temperature, wind components, surface solar
  radiation, total precipitation, surface pressure, soil temperature
- **Resolution**: 0.1° (~9 km) grid, hourly, 1950–present (trails real time by ~5 days)
- **Coverage**: Global land surface (no data over sea — coastal cells resolve to the
  nearest inland cell)
- **Access**: CDS API — a configured `~/.cdsapirc` key is required. Two access paths:
  the **ARCO analysis-ready Zarr** stores over HTTPS (tried first, no download queue)
  and the classic `cdsapi` retrieve (GRIB grid / CSV timeseries) as fallback
- **Derived indicators**: heating/cooling degree days, annual precipitation, global
  horizontal irradiation, temperature/humidity/wind summaries
- **License**: Copernicus / ECMWF (free, attribution required)

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

[← Library Structure](library-structure.md) | [Output Data Schema →](output-schema.md) | [Back to README](../README.md)