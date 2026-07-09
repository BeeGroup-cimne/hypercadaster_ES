# Output Data Schema

This document provides a complete reference for the data structure and columns available in hypercadaster_ES output GeoDataFrames.

## 📊 Overview

The resulting GeoDataFrame provides comprehensive building-level information organized in logical column groups. The exact columns available depend on the configuration parameters used during processing.

Prefixes tell you where a column comes from:

- **no prefix** — cadastral/address/administrative attributes and join results.
- **`br__`** — building-reference inference outputs (one row per building;
  requires `building_parts_inference=True`).
- **`sbr__`** — single-building-reference inference outputs (one row per
  physically distinct building within a shared parcel; stored in the
  `sbr_results` table, see [Space-level results](#-space-level-results-sbr_results)).
- **`poi_…`** — OpenStreetMap POI indicators (requires `pois_layer=True`).

> **Heavy columns are not in the returned frame.** Eight per-building BLOB
> columns (shadows, terrain grids, wall segments) are deliberately **excluded**
> from `merge()`'s output and from the `buildings` table to keep memory bounded
> — they are only retrievable one building at a time via
> `get_heavy_columns_for_building()`. See
> [Heavy BLOB columns](#-heavy-blob-columns-on-demand-only).

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
| `street_number_clean` | String | Building number, normalised |
| `street_number_label` | String | Building number as labelled in the source |
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
| `br__building_perimeter_by_floor` | List | Perimeter per floor (above ground) |
| `br__building_perimeter_below_ground_by_floor` | List | Perimeter per basement floor |
| `br__building_footprint_by_floor` | Dict | Footprint geometry per floor |
| `br__building_footprint_geometry` | Geometry | Building outline |

### Structural Floors
| Column | Type | Description |
|--------|------|-------------|
| `br__structural_floors` | Float | Total number of structural floor slabs |
| `br__structural_floors_by_floor` | List | Structural slabs per above-ground floor |
| `br__structural_floors_below_ground_by_floor` | List | Structural slabs per basement floor |

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

### Building Contour (light column)
| Column | Type | Description |
|--------|------|-------------|
| `br__building_contour_at_distance` | Dict | `{orientation_deg: distance_from_centroid_to_edge_m}` |

> The full per-floor shadow casting and terrain results
> (`br__shadows`, `br__shadows_at_distance`, `br__terrain_grid`) are **heavy
> BLOB columns** excluded from the frame — see
> [Heavy BLOB columns](#-heavy-blob-columns-on-demand-only).

## 🌡️ Thermal Analysis Properties (building_inference=True)

### Exterior Walls by Orientation
Exterior walls are split into **main-façade** walls (facing air/street) and
**party** walls (shared with an adjacent building on the same orientation).

| Column | Type | Description |
|--------|------|-------------|
| `br__main_facade_wall` | Dict | `{orientation_deg: facade_wall_area_m2}` |
| `br__main_facade_wall_by_floor` | Dict | `{orientation_deg: [area_per_floor]}` |
| `br__main_facade_wall_significant_orientations` | List | Dominant façade orientations |
| `br__main_facade_wall_significant_orientations_by_floor` | List | Per-floor dominant orientations |
| `br__party_wall` | Dict | `{orientation_deg: party_wall_area_m2}` (shared with neighbour) |
| `br__party_wall_by_floor` | Dict | `{orientation_deg: [area_per_floor]}` |
| `br__party_wall_significant_orientations` | List | Dominant party-wall orientations |
| `br__party_wall_significant_orientations_by_floor` | List | Per-floor dominant orientations |

### Internal & Ground-Contact Walls
| Column | Type | Description |
|--------|------|-------------|
| `br__adiabatic_wall` | Float | Total adiabatic (fully shared) wall area (m²) |
| `br__adiabatic_wall_by_floor` | List | Adiabatic wall area per floor |
| `br__terrain_contact_wall` | Float | Total wall area in contact with terrain (m²) |
| `br__terrain_contact_wall_by_floor` | List | Terrain-contact wall area per above-ground floor |
| `br__terrain_contact_wall_below_ground_by_floor` | List | Terrain-contact wall area per basement floor |
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

## 🧱 Heavy BLOB columns (on-demand only)

To keep memory bounded on large municipalities (each of these is on the order of
gigabytes for Barcelona), `merge()` **never materialises** the following eight
columns into the returned frame or the `buildings` table. They live in the
`br_results` / `sbr_results` tables and must be fetched one building at a time:

```python
heavy = hc.utils.get_heavy_columns_for_building(
    wd, cadaster_code, building_reference, table="br_results"  # or "sbr_results"
)
```

`table="br_results"` returns the four `br__` columns; `table="sbr_results"`
returns the four `sbr__` equivalents. Missing rows yield `None` per column.

| Column (`br__` / `sbr__`) | Type | Structure |
|---------------------------|------|-----------|
| `…__shadows` | Dict | `{floor_int: (Multi)Polygon}` — per-floor visible-sky / cast-shadow footprint |
| `…__shadows_at_distance` | Dict | `{orientation_deg_str: [distance_m per floor]}` — distance to the first taller obstruction |
| `…__terrain_grid` | Dict | Local DEM raster patch: `{x_min, y_min, cell_size, nx, ny, elev}`, where `elev` is an `ny × nx` grid of elevations (m) |
| `…__wall_segments` | List | Records `{geometry: LineString, type: "main_facade"｜"party"｜…, floor: int, orientation: …}` |

Plot helpers auto-hydrate these: passing `wd=` to
`hc.utils.plot_building_shadows_analysis(gdf, ref, wd=wd)` fetches the missing
shadow/terrain columns for the requested building automatically.

## 🏢 Space-level results (`sbr_results`)

When a cadastral parcel holds several physically distinct buildings, inference
also produces a **single-building-reference** row per building in the
`sbr_results` table (read with `hc.utils.read_sbr_inferred_indicators(code, wd)`).
It mirrors the `br__*` geometry/wall/shadow schema with an `sbr__` prefix, plus:

| Column | Type | Description |
|--------|------|-------------|
| `single_building_reference` | String | Identifier of the individual building within the parcel |
| `sbr__n_buildings` | Integer | Number of buildings sharing the parent building reference |
| `sbr__detached` | Boolean | Whether this building is detached |

All the per-building geometry, floor, wall (`sbr__main_facade_wall`,
`sbr__party_wall`, `sbr__terrain_contact_wall`, …), patio, structural-floor and
shadow/terrain columns exist with the `sbr__` prefix (the four
`sbr__shadows`/`sbr__shadows_at_distance`/`sbr__terrain_grid`/`sbr__wall_segments`
are heavy — fetch with `table="sbr_results"`).

## 📍 Points of Interest (pois_layer=True)

POI indicators are added when `pois_layer=True` is passed to `merge()`. Data is sourced from OpenStreetMap and joined at building level for the following groups: `amenity`, `emergency`, `healthcare`, `leisure`, `office`, `shop`, `tourism`.

For groups with subgroups (e.g. `amenity` → `food`, `education`, …) the column prefix is `poi_{group}_{subgroup}`. For groups without subgroups (e.g. `emergency`, `healthcare`) the prefix is `poi_{group}`.

### Per-group/subgroup Columns

| Column | Type | Description |
|--------|------|-------------|
| `poi_{prefix}_count` | Integer | Number of POIs located on the building footprint |
| `poi_{prefix}_ids` | List | OSM identifiers of POIs on the building |
| `poi_{prefix}_100m` | Integer | POI count within 100 m of building centroid |
| `poi_{prefix}_250m` | Integer | POI count within 250 m of building centroid |
| `poi_{prefix}_500m` | Integer | POI count within 500 m of building centroid |
| `poi_{prefix}_1000m` | Integer | POI count within 1000 m of building centroid |
| `poi_{prefix}_1500m` | Integer | POI count within 1500 m of building centroid |

### POI Groups and Subgroups

| Group | Subgroups | Example values |
|-------|-----------|----------------|
| `amenity` | `food`, `education`, `transport`, `financial`, `health`, `culture`, `public`, `facilities`, `waste`, `other` | restaurants, schools, banks, hospitals… |
| `emergency` | *(none)* | ambulance stations, defibrillators, fire hydrants… |
| `healthcare` | *(none)* | hospitals, clinics, doctors, dentists… |
| `leisure` | *(none)* | parks, playgrounds, sports centres, swimming pools… |
| `office` | *(none)* | company offices, government, lawyers… |
| `shop` | `food`, `general`, `clothing`, `health`, `diy`, `electronics`, `sports`, `culture`, `other` | supermarkets, pharmacies, electronics stores… |
| `tourism` | `accommodation`, `camping`, `culture`, `attraction`, `leisure`, `information`, `rental` | hotels, museums, viewpoints… |

**Examples of generated column names:**
- `poi_amenity_food_count` — restaurants/cafes/bars on the building
- `poi_amenity_food_250m` — food amenities within 250 m
- `poi_emergency_100m` — emergency facilities within 100 m
- `poi_shop_clothing_500m` — clothing shops within 500 m
- `poi_leisure_1500m` — leisure POIs within 1500 m
- `poi_tourism_accommodation_ids` — OSM IDs of accommodation on the building

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
| Heavy shadow/terrain columns | - | - | on-demand¹ | on-demand¹ |
| Space-level `sbr_results` | - | - | ✓ | ✓ |
| POI Proximity | pois_layer=True | pois_layer=True | pois_layer=True | pois_layer=True |
| Commercial Premises | Barcelona Only | Barcelona Only | Barcelona Only | Barcelona Only |

¹ Never in the returned frame — fetch with `get_heavy_columns_for_building()`
(see [Heavy BLOB columns](#-heavy-blob-columns-on-demand-only)).

---

[← Configuration & Examples](configuration-examples.md) | [Contributing →](contributing.md) | [Back to README](../README.md)