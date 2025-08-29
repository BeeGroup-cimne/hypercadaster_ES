# Output Data Schema

This document provides a complete reference for the data structure and columns available in hypercadaster_ES output GeoDataFrames.

## 📊 Overview

The resulting GeoDataFrame provides comprehensive building-level information organized in logical column groups. The exact columns available depend on the configuration parameters used during processing.

## 🏗️ Core Building Identification & Geometry

### Primary Identifiers
| Column | Type | Description |
|--------|------|-------------|
| `building_reference` | String | Unique cadastral building identifier |
| `gml_id` | String | GML feature identifier |

### Geometric Information
| Column | Type | Description |
|--------|------|-------------|
| `building_geometry` | Polygon | Building footprint polygon |
| `building_area` | Float | Total gross floor area (m²) |
| `building_centroid` | Point | Building geometric center point |
| `address_location` | Point | Primary address coordinates |

## 🏠 Building Characteristics & Attributes

### Physical Characteristics
| Column | Type | Description |
|--------|------|-------------|
| `building_use` | String | Primary building usage classification |
| `building_status` | String | Construction status (completed, under construction, etc.) |
| `year_of_construction` | Integer | Construction year |
| `n_building_units` | Integer | Number of individual units/dwellings |
| `n_dwellings` | Integer | Number of residential units |
| `n_floors_above_ground` | Integer | Floors above ground level |
| `n_floors_below_ground` | Integer | Basement/parking levels (building_inference=True only) |

## 📍 Complete Address Information

### Structured Addressing
| Column | Type | Description |
|--------|------|-------------|
| `street_name` | String | Street name (e.g., "SANT AGUSTÍ") |
| `street_type` | String | Street type (e.g., "PLAÇA DE", "CALLE") |
| `street_number` | String | Building number |
| `street_number_clean` | Integer | Numeric building number |
| `specification` | String | Address specification (entrance, etc.) |

## 🗺️ Administrative Geography & Context

### Municipal Administration
| Column | Type | Description |
|--------|------|-------------|
| `cadaster_code` | String | Municipal cadastral code |
| `ine_code` | String | INE municipality code |
| `municipality_name` | String | Official municipality name |
| `province_name` | String | Province name |
| `autonomous_community_name` | String | Autonomous community name |

### Postal and Census Geography
| Column | Type | Description |
|--------|------|-------------|
| `postal_code` | String | Spanish postal code |
| `section_code` | String | Census tract identifier |
| `district_code` | String | Census district identifier |

### Barcelona-Specific (when applicable)
| Column | Type | Description |
|--------|------|-------------|
| `neighborhood_code` | String | Barcelona neighborhood identifier |
| `neighborhood_name` | String | Barcelona neighborhood name |
| `district_name` | String | Barcelona district name |

## 🏘️ Cadastral Zoning & Parcels

### Zoning Information
| Column | Type | Description |
|--------|------|-------------|
| `zone_reference` | String | Cadastral zone reference |
| `zone_type` | String | Zone classification (urban/disseminated) |

### Parcel Information (building_inference=True only)
| Column | Type | Description |
|--------|------|-------------|
| `parcel_geometry` | Polygon | Cadastral parcel boundary polygon |
| `parcel_centroid` | Point | Parcel geometric center |

## 🌍 Environmental & Topographic Data

### Elevation Information (when elevation_layer=True)
| Column | Type | Description |
|--------|------|-------------|
| `elevation` | Float | Ground elevation above sea level (meters) |

## 🏢 Building Usage Classification by Area (CAT files available)

### Detailed Usage Breakdown by Area (m²)
| Column | Type | Description |
|--------|------|-------------|
| `building_area_residential` | Float | Residential space area |
| `building_area_commercial` | Float | Commercial space area |
| `building_area_offices` | Float | Office space area |
| `building_area_warehouse_parking` | Float | Storage/parking area |
| `building_area_industrial` | Float | Industrial space area |
| `building_area_religious` | Float | Religious buildings area |
| `building_area_cultural` | Float | Cultural facilities area |
| `building_area_healthcare_and_charity` | Float | Healthcare facilities area |
| `building_area_sports_facilities` | Float | Sports facilities area |
| `building_area_entertainment_venues` | Float | Entertainment venues area |
| `building_area_leisure_and_hospitality` | Float | Hotels, restaurants area |
| `building_area_singular_building` | Float | Special/unique buildings area |
| `building_area_urbanization_and_landscaping_works_undeveloped_land` | Float | Landscaping area |

## 🔍 Advanced Building Space Analysis (building_inference=True and CAT files)

### Space Type Breakdown from Cadastral Records
| Column | Type | Description |
|--------|------|-------------|
| `br__building_spaces` | Dict | {space_type: count} |
| `br__area_without_communals` | Dict | {space_type: private_area_m2} |
| `br__area_with_communals` | Dict | {space_type: total_area_m2} |
| `br__area_with_communals_by_floor` | Dict | {space_type: {floor: area}} |
| `br__area_without_communals_by_floor` | Dict | {space_type: {floor: area}} |

### Space Metadata by Type
| Column | Type | Description |
|--------|------|-------------|
| `br__building_spaces_reference` | Dict | {space_type: [reference_codes]} |
| `br__building_spaces_floor_number` | Dict | {space_type: [floor_numbers]} |
| `br__building_spaces_category` | Dict | {space_type: [categories]} |
| `br__building_spaces_detailed_use_type` | Dict | {space_type: [detailed_uses]} |
| `br__building_spaces_economic_value` | Dict | {space_type: [values]} |
| `br__building_spaces_effective_year` | Dict | {space_type: [construction_years]} |
| `br__building_spaces_postal_address` | Dict | {space_type: [addresses]} |
| `br__building_spaces_inner_address` | Dict | {space_type: [internal_refs]} |

## 🏗️ Building Geometry & Structure Analysis (building_inference=True)

### Floor and Area Calculations
| Column | Type | Description |
|--------|------|-------------|
| `br__floors_above_ground` | Integer | Number of floors above ground |
| `br__floors_below_ground` | Integer | Number of basement floors |
| `br__building_footprint_area` | Float | Ground floor area (m²) |
| `br__above_ground_built_area` | Float | Total built area above ground |
| `br__below_ground_built_area` | Float | Total built area below ground |
| `br__above_ground_roof_area` | Float | Roof area above ground |

### Floor-by-Floor Breakdowns
| Column | Type | Description |
|--------|------|-------------|
| `br__above_ground_built_area_by_floor` | List | Built area per floor |
| `br__above_ground_roof_area_by_floor` | List | Roof area per floor |
| `br__below_ground_built_area_by_floor` | List | Basement area per floor |
| `br__building_perimeter_by_floor` | List | Perimeter per floor |
| `br__building_footprint_by_floor` | List | Footprint geometry per floor |
| `br__building_footprint_geometry` | LineString | Building outline |

## 🧭 Building Orientation & Environmental Analysis (building_inference=True)

### Orientation Analysis
| Column | Type | Description |
|--------|------|-------------|
| `br__parcel_orientations` | Dict | {orientation_degrees: facade_length} |
| `br__parcel_main_orientation` | Float | Primary facade orientation (degrees) |
| `br__street_width_by_orientation` | Dict | {orientation: street_width_m} |
| `br__street_width_main_orientation` | Float | Street width at main facade |

### Building Context Analysis
| Column | Type | Description |
|--------|------|-------------|
| `br__detached` | Boolean | Is building detached |
| `br__next_building_by_orientation` | Dict | {orientation: distance_to_next_m} |
| `br__next_building_main_orientation` | Float | Distance to next building at main orientation |

### Building Contour Analysis (Shadow Casting)
| Column | Type | Description |
|--------|------|-------------|
| `br__building_contour_at_distance` | Dict | {orientation: shadow_distance_m} |
| `br__shadows_at_distance` | Dict | {orientation: [shadow_polygons]} |

## 🌡️ Thermal Analysis Properties (building_inference=True)

### Wall Analysis by Orientation
| Column | Type | Description |
|--------|------|-------------|
| `br__air_contact_wall` | Dict | {orientation: exterior_wall_area_m2} |
| `br__air_contact_wall_by_floor` | Dict | {orientation: [area_per_floor]} |
| `br__air_contact_wall_significant_orientations` | List | Significant facade orientations |
| `br__air_contact_wall_significant_orientations_by_floor` | List | Per-floor significant orientations |

### Internal Walls and Structures
| Column | Type | Description |
|--------|------|-------------|
| `br__adiabatic_wall` | Float | Total internal wall area (m²) |
| `br__adiabatic_wall_by_floor` | List | Internal wall area per floor |
| `br__walls_between_slabs` | Float | Total wall area between floor slabs |

### Courtyard/Patio Analysis
| Column | Type | Description |
|--------|------|-------------|
| `br__patios_area_by_floor` | List | Patio area per floor |
| `br__patios_number_by_floor` | List | Number of patios per floor |
| `br__patios_wall_by_floor` | List | Patio wall area per floor |
| `br__patios_wall_total` | Float | Total patio wall area |

## 💰 Economic & Statistical Aggregations (CAT files + building_inference=True)

### Economic Valuations
| Column | Type | Description |
|--------|------|-------------|
| `br__economic_value` | Dict | {space_type: cadastral_value} |
| `br__economic_value_by_floor` | Dict | {space_type: {floor: value}} |
| `br__communal_area` | Dict | {space_type: shared_area_m2} |
| `br__communal_area_by_floor` | Dict | {space_type: {floor: shared_area}} |

### Statistical Averages per Space Type
| Column | Type | Description |
|--------|------|-------------|
| `br__mean_building_space_area_with_communals` | Dict | {space_type: avg_area} |
| `br__mean_building_space_area_without_communals` | Dict | {space_type: avg_private_area} |
| `br__mean_building_space_economic_value` | Dict | {space_type: avg_value} |
| `br__mean_building_space_effective_year` | Dict | {space_type: avg_year} |
| `br__mean_building_space_category` | Dict | {space_type: avg_category} |
| `br__mean_building_space_communal_area` | Dict | {space_type: avg_shared_area} |

### Floor-Specific Averages
| Column | Type | Description |
|--------|------|-------------|
| `br__mean_building_space_area_with_communals_by_floor` | Dict | Per-floor averages |
| `br__mean_building_space_area_without_communals_by_floor` | Dict | Per-floor private averages |
| `br__mean_building_space_economic_value_by_floor` | Dict | Per-floor value averages |
| `br__mean_building_space_effective_year_by_floor` | Dict | Per-floor construction years |
| `br__mean_building_space_category_by_floor` | Dict | Per-floor category averages |
| `br__mean_building_space_communal_area_by_floor` | Dict | Per-floor shared area averages |

## 🏪 Commercial Premises Analysis (Barcelona open_data_layers=True)

### Ground Floor Commercial Analysis
| Column | Type | Description |
|--------|------|-------------|
| `br__exists_ground_commercial_premises` | Boolean | Has ground floor commercial |
| `br__ground_commercial_premises_names` | List | Commercial establishment names |
| `br__ground_commercial_premises_typology` | List | Types of commercial premises |
| `br__ground_commercial_premises_last_revision` | List | Last update dates |

## 🗂️ Data Types and Formats

### Geometric Data Types
- **Point**: Shapely Point geometry (coordinates)
- **Polygon**: Shapely Polygon geometry (building footprints, parcels)
- **LineString**: Shapely LineString geometry (building outlines)

### Dictionary Data Types
- **Dict**: Python dictionary with structured keys
- **Lists of Dictionaries**: Complex nested structures for floor-by-floor data

### Coordinate Reference Systems
- **Default CRS**: EPSG:25831 (ETRS89 / UTM zone 31N) for Spain
- **Alternative**: EPSG:25830 for western Spain, EPSG:25828/25829 for islands

## 📋 Column Availability Matrix

| Feature Category | Basic | +CAT Files | +Inference | +Inference+CAT |
|-----------------|--------|------------|------------|----------------|
| Core Building Info | ✓ | ✓ | ✓ | ✓ |
| Address Info | ✓ | ✓ | ✓ | ✓ |
| Administrative | ✓ | ✓ | ✓ | ✓ |
| Usage by Area | - | ✓ | - | ✓ |
| Geometric Analysis | - | - | ✓ | ✓ |
| Orientation Analysis | - | - | ✓ | ✓ |
| Thermal Properties | - | - | ✓ | ✓ |
| Space Breakdown | - | - | - | ✓ |
| Economic Data | - | ✓ | - | ✓ |
| Commercial Premises | Barcelona Only | Barcelona Only | Barcelona Only | Barcelona Only |

---

[← Configuration & Examples](configuration-examples.md) | [Contributing →](contributing.md) | [Back to README](../README.md)