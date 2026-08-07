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
| `postal_addresses_location` | List[Point] | Coordinates of each of the parcel's addresses (from CartoCiudad) |

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

Addresses come from **CartoCiudad** (CNIG) and are always compacted to **one row
per parcel** (`building_reference`): the per-address columns are **lists**, all
aligned to the same order (the *i*-th entry of each list describes the parcel's
*i*-th address). The raw `street_type` / `street_name` / `street_number_*` columns
are collapsed into the combined `postal_addresses_*` label lists below.

**Street-type normalisation.** CartoCiudad's `tipo_vial` is inconsistent (regional
languages, bilingual `SPANISH / REGIONAL` forms, `X, Y` descriptors, abbreviations,
typos). It is normalised to the canonical Catastro full name via
`downloaders.normalize_cartociudad_street_type` — e.g. `AVINGUDA`/`KALEA`→`AVENIDA`/`CALLE`,
`PLAÇA`→`PLAZA`, `PARTICULAR`→`PARTIDA`. Genuine Spanish names with no Catastro
equivalent (`BULEVAR`, `AUTOPISTA`, `DISEMINADO`, `PARAJE`, …) are kept; unrecognised
short codes become `None`. The full mapping (with per-province provenance) is in
`{wd}/cartociudad/street_type_mapping_review.csv`.

### Structured Addressing
| Column | Type | Description |
|--------|------|-------------|
| `postal_addresses_with_extensions` | List[String] | Full address per entry, using the source number label with extension (e.g. "CALLE GRAN, 4", "CARRER MAJOR, 20(B)") |
| `postal_addresses_clean` | List[String] | Full address per entry, using the normalised number |
| `postal_code` | List[String] | Postal code of each address (from CartoCiudad `cod_postal`; boundary-polygon fallback) |
| `population_entity` | List[String] | Population entity / hamlet / estate name (`poblacion`) per address |
| `postal_addresses_location` | List[Point] | Address point coordinates (see Geometric Information) |
| `postal_address_specification` | String | Address type (always "Portal" for CartoCiudad) |
| `address_source` | String | Origin of the address data (`fuente_datos`) |
| `address_last_modification` | Datetime | Source last-modification timestamp (`fecha_modificacion`) |

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
| `building_area_agricultural` | Float | Agricultural building area (rural sheds/barns; from the synthesized space for buildings absent from the CAT file) |
| `building_area_religious` | Float | Religious buildings area |
| `building_area_cultural` | Float | Cultural facilities area |
| `building_area_healthcare_and_charity` | Float | Healthcare facilities area |
| `building_area_sports_facilities` | Float | Sports facilities area |
| `building_area_entertainment_venues` | Float | Entertainment venues area |
| `building_area_leisure_and_hospitality` | Float | Hotels, restaurants area |
| `building_area_singular_building` | Float | Special/unique buildings area |
| `building_area_urbanization_and_landscaping_works_undeveloped_land` | Float | Landscaping area |
| `building_area_unknown` | Float | Area of spaces whose use type could not be resolved |

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
| `br__building_spaces_postal_address` | Dict | {space_type: [addresses]} — each space matched to the closest `postal_addresses_with_extensions` entry (CartoCiudad); None where the parcel has no address |
| `br__building_spaces_postal_address_from_CAT_file` | Dict | {space_type: [addresses]} — the original CAT-file per-space postal address (before the CartoCiudad match) |
| `br__building_spaces_postal_address_location` | Dict | {space_type: [Points]} — the matched address's point (`postal_addresses_location`) |
| `br__building_spaces_inner_address` | Dict | {space_type: [internal_refs]} |

> At the building-space resolution (`utils.flat_gdf_to_building_spaces_resolution`),
> these `br__building_spaces_*` dicts explode to one value per space, including
> `building_space_postal_address` (the matched CartoCiudad address),
> `building_space_postal_address_from_CAT_file` (the original CAT value) and
> `building_space_postal_address_location` (the matched point).

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

## 🧮 Socioeconomic Indicators (social_ES_layer=True)

INE socioeconomic/demographic indicators from the `social_ES` package, joined
(before the POI join) at each dataset's finest geographic level using the census
codes added by `census_layer`. Column names follow `so__{indicator}_{year}`
(`so__` namespace prefix, snake_case indicator, year as a **suffix**); the most
recent `social_ES_years` (default 5) are kept per dataset.

Every indicator comes from exactly **one** dataset — the one publishing it at the
finest geography over the longest run of years — so no figure appears twice under
two names.

### Section/district/municipality/province indicators

| Column pattern | Level | Description |
|----------------|-------|-------------|
| `so__net_income_per_household_{year}`, `so__net_income_per_person_{year}`, `so__net_income_per_consumption_unit_{year}`, `so__gross_income_per_household_{year}`, `so__gross_income_per_person_{year}`, `so__median_income_per_consumption_unit_{year}` | Section | Income Atlas — mean/median income (€, nominal) |
| `so__net_income_per_household_2010eur_{year}` | Section | Income Atlas — household net income deflated to **2010 €** (comparable across years) |
| `so__pct_income_source_salary_{year}`, `so__pct_income_source_pension_{year}`, `so__pct_income_source_unemployment_{year}`, `so__pct_income_source_other_benefits_{year}`, `so__pct_income_source_capital_investments_secondary_{year}` | Section | Income Atlas — **% share** of household income by source (not €) |
| `so__gini_index_{year}`, `so__income_p80_p20_{year}`, `so__average_age_{year}`, `so__average_household_size_{year}`, `so__pct_single_person_households_{year}` | Section | Income Atlas — inequality & household structure |
| `so__household_income_group_{year}` | Section | Income Atlas — Time Use Survey household income band (1 = ≤1,200 €/month, 2 = 1,201–2,000 €, 3 = 2,001–3,000 €, 4 = >3,000 €, on income deflated to 2010 €). Join key into the survey's weekly occupancy agenda — see [Time Use occupancy](#-time-use-occupancy-agenda) |
| `so__population_{year}` | Section | Population Census — resident head count |
| `so__pct_foreign_nationality_{year}`, `so__pct_foreign_born_{year}`, `so__pct_born_same_municipality_{year}`, `so__pct_woman_{year}`, `so__pct_under_15_{year}`, `so__pct_working_age_{year}`, `so__pct_65_plus_{year}` | Section | Population Census — population shares (%) |
| `so__pct_edu_lte_primary_or_below_{year}`, `so__pct_edu_lte_lower_secondary_{year}`, `so__pct_edu_lte_upper_secondary_{year}`, `so__pct_edu_lte_tertiary_{year}` | Section | Education & Employment Census — **cumulative** attainment: % of the population aged 16+ that reached **at most** that level. Monotonically increasing; `lte_tertiary` = 100 |
| `so__pct_employed_{year}`, `so__pct_unemployed_{year}`, `so__pct_student_{year}`, `so__pct_pensioner_{year}`, `so__pct_other_inactive_{year}` | Section | Education & Employment Census — labour force status (%) |
| `so__dwellings_2021`, `so__dwellings_main_2021`, `so__households_2021` | Section | 2021 Dwellings Census — dwelling stock and households (counts) |
| `so__pct_dwellings_main_2021` | Section | 2021 Dwellings Census — main dwellings as % of all dwellings |
| `so__pct_main_dwellings_owned_2021`, `so__pct_main_dwellings_rented_2021`, `so__pct_main_dwellings_other_tenure_2021` | Section | 2021 Dwellings Census — tenure of the main dwellings (%, sums to 100) |
| `so__municipality_dwellings_{2001,2011,2021}` | Municipality | Empty & Secondary Dwellings — municipal dwelling stock at each census |
| `so__pct_dwellings_use_main_{year}`, `so__pct_dwellings_use_secondary_{year}`, `so__pct_dwellings_use_empty_{year}` | Municipality | Empty & Secondary Dwellings — main/secondary/**empty** split harmonised across the 2001, 2011 and 2021 censuses (%, sums to 100) |
| `so__dwelling_use_predicted_{year}` | Municipality | 1.0 where that split was **modelled** by `social_ES` (INE withholds it for small municipalities), 0.0 where published |
| `so__rental_price_index_{year}` | District, else Municipality | Household rental price index — by district where INE publishes it, municipal series elsewhere |
| `so__housing_price_index_total_{year}`, `so__housing_price_index_new_{year}`, `so__housing_price_index_second_hand_{year}` | Province | Housing price index (annual mean) |
| `so__electricity_consumption_p10_{year}`…`p90_{year}` | District | Electricity-consumption percentiles (kWh) |

Coverage reflects INE publication scope (e.g. rental index and electricity exist
only for some municipalities/districts); unmatched buildings get `NaN`.

> **Empty and secondary dwellings.** 2021 replaced the census agent's judgement
> at the door with a classification derived from each dwelling's yearly
> electricity consumption, so a change between 2011 and 2021 mixes a real change
> in use with the change of instrument — most visibly in tourist municipalities.
> Where INE withholds the split (the 2011 secondary/empty split below 2,000
> inhabitants, the 2021 classification below 1,000), `social_ES` models it from
> what is published everywhere and calibrates each province to the figure INE
> gives for it; those rows are flagged in `so__dwelling_use_predicted_{year}` and
> should be left out of anything turning on a single small municipality.

### Essential Characteristics of Households and People downscaling — `ec__*` (use_CAT_files=True only)

When `use_CAT_files=True` and the `essential_characteristics` dataset is selected
(default when `social_ES_datasets=None`), the INE *Essential Characteristics of
Population and Households* survey is downscaled to each building as `ec__*`
proportion columns (percentages; each variable group sums to 100).

Column names are exactly the ones `social_ES` (≥ 1.0.3) returns — short English
snake_case, with the class a column measures joined to its variable by a `~` —
with only the namespace prefix added:

`ec__{variable}~{class}`, e.g. `ec__main_dwellings_heating_type~individual`,
`ec__main_dwellings_conservation_status~good`, `ec__households_vehicles_number~3plus`.

Reading a column name therefore tells you unambiguously which variable it belongs
to (everything left of the `~`) and which class of it the proportion measures.

> ⚠️ **Breaking change (1.2.4).** These used to be flattened to
> `ec__{variable}_{class}` (a plain underscore). Frames merged by an earlier
> release keep the old spelling; re-run `hc.merge` to get the new one. The energy
> model reads both.

**Only a curated selection is emitted by default** — 27 of the ~70 variables
`social_ES` publishes, chosen because they say something the `so__` census layer
and the cadaster do not already say. In particular the survey is the *only*
source for the dwelling's energy systems:

| Group | Variables |
|-------|-----------|
| Dwelling energy systems | `main_dwellings_heating_type`, `main_dwellings_heating_fuel`, `main_dwellings_has_cooling`, `main_dwellings_insulation_issues`, `main_dwellings_renewable_device`, `main_dwellings_installations`, `main_dwellings_lighting_type`, `main_dwellings_domestic_appliances_type`, `main_dwellings_water_supply` |
| Condition & accessibility | `main_dwellings_conservation_status`, `main_dwellings_accessibility_status`, `main_dwellings_adapted_to_elderly` |
| Dwelling interior | `main_dwellings_n_bathrooms`, `main_dwellings_kitchen_sized_4m2plus`, `main_dwellings_n_rooms`, `main_dwellings_internet_connection` |
| Presence & absence | `households_second_home_ownership`, `households_secondary_second_home_occupancy`, `people16plus_work_or_study_place`, `people16plus_transport_daily_time` |
| Mobility & vehicles | `households_vehicles_number`, `households_vehicles_eco`, `people16plus_transport_mode` |
| Housing cost burden | `households_rented_monthly_rent`, `households_owned_monthly_mortgage` |
| Household composition | `households_n_children` |
| Neighbourhood quality | `main_dwellings_surroundings_issues` |

Dropped on purpose because a better source already covers them: tenure, building
type and floor area (`so__pct_main_dwellings_*` / the cadaster), the
single-person-household block (`so__pct_single_person_households` plus the `so__`
income / education / labour columns), the parents' origin and nationality blocks
(`so__pct_foreign_*` / `so__pct_born_*`), and the individual digital-access
questions (`main_dwellings_internet_connection` covers it at dwelling level).

Pass `social_ES.EC_SELECTED_VARIABLES` your own sequence — or `None` for all ~70
variables (~256 columns) — through `hc.merge(social_ES_ec_variables=…)` /
`social.join_social_ES(ec_variables=…)`.

The survey is published only for larger municipalities (~147 nationwide), so these
columns populate only there.

### 🕒 Time Use occupancy agenda

Not a column: `hypercadaster_ES.social.time_use_household_occupancy(wd,
census_tract_code)` returns a census tract's **weekly household occupancy agenda**
from the INE Time Use Survey (EET 2009-2010) via
`social_ES.INE.TimeUseSurvey` → `WeeklyScheduleHouseholdOccupancy`. The tract is
matched to a survey cell by its autonomous community and by its
`so__household_income_group_{year}` band, and the cell's dominant occupancy state
at each of the 168 hours of the week is returned (hour 0 = Monday 00:00):

| State | Meaning |
|-------|---------|
| `Sleeping` | at home, asleep |
| `WithActivity` | at home, awake and doing something |
| `NonOccupied` | out (paid work, study, social life, sport, travel …) |

```python
agenda = hc.social.time_use_household_occupancy(wd, "0801901001")
agenda["states"][:24]                      # Monday, hour by hour
agenda["shares"]["NonOccupied"][8]         # % of households out at Monday 08:00
agenda["household_income_group_label"]     # e.g. "2,001 to 3,000 €"
```

The energy model reads it to give its `aware` and `unaware` residential dwellings
a measured occupancy agenda — see `residential_behavior.time_use_occupancy` in
`energy_config.yaml`. Returns `None` for tracts the income atlas does not cover,
in which case the stylised residential profile is used instead.

## 🧮 Census-Tract Aggregations — `ctagg__*` (census_tract_aggregations=True, default)

The frame aggregated **by census tract** and written back onto every building, so
each row carries its tract's context. These are the tract-level quantities the
energy simulations need but cannot derive from one building — which is why they
are computed here, while the whole frame is in hand, rather than at simulation
time from whatever subset the caller still holds.

| Column | Description |
|--------|-------------|
| `ctagg__buildings` | Buildings of the tract present in this frame |
| `ctagg__residential_buildings` | How many of them carry residential space |
| `ctagg__residential_area_m2` | Σ non-communal residential floor area of the tract |
| `ctagg__dwellings` | Σ residential building spaces (dwellings) of the tract |
| `ctagg__population` | Tract head count, from `so__population_{year}` |
| `ctagg__people_per_m2` | `population / residential_area_m2` — the occupancy density driving the **DHW demand** in `energy.run_behavior_scenarios` |
| `ctagg__people_per_dwelling` | `population / dwellings` — mean household size as the cadaster counts dwellings (cf. `so__average_household_size_{year}`, counted the census's own way) |
| `ctagg__dwelling_coverage` | Share of the tract's 2021-census main dwellings (`so__dwellings_main_2021`) this frame's spaces account for |
| `ctagg__complete_tract` | Whether coverage clears 50 % (`mergers.CTAGG_MIN_COVERAGE`) |

**What reads it.** `ctagg__people_per_m2` is the single occupancy figure behind
every people-driven quantity of the energy model:

- **`People` internal gains** — metabolic heat, and the cooling load and comfort
  that follow from it (`loads.residential_occupancy_from_census`, on by default);
- **DHW demand** (`dhw_demand_kwh_m2`) — litres per person per CTE DB-HE HE4;
- **household size** — the occupants each dwelling is modelled with.

Each dwelling gets `max(1, round(density × its area))` occupants, so a dwelling is
never modelled as empty, and the zone density is that head count over the
residential floor area — gains and DHW therefore describe exactly the same people.
The normative `cte` scenario keeps the `loads.residential.ppl_per_m2` default
instead: it is a standards-compliance case, not a description of who lives there.

**Why it matters.** `ctagg__people_per_m2` is what lets a one-building frame be
simulated on its own:

```python
one = gdf[gdf["building_reference"] == ref]      # carries its tract's density
hc.energy.run_behavior_scenarios(one, wd=wd, weather_year=2022)
```

Deriving that density from `one` instead would divide the whole tract's
population by a single building's floor area — a rate as many times too high as
the tract has buildings, and one that looks perfectly plausible in the output.

**Partial frames publish no density.** When `ctagg__dwelling_coverage` falls below
50 %, the frame demonstrably holds only part of the tract, so
`ctagg__people_per_m2` and `ctagg__people_per_dwelling` are left **NaN** and
`ctagg__complete_tract` is False — the counts are still emitted. Without
`so__dwellings_main_2021` there is no yardstick to check against, so the frame is
assumed to hold the tract (what a whole-municipality merge does).

Requires `census_layer=True` (for `census_tract_code`) and `use_CAT_files=True`
(for `br__building_spaces_*`); `ctagg__population`, the densities and the coverage
check additionally need `social_ES_layer=True`. Anything missing degrades to a
no-op or to the columns that can be computed.

## 🌦️ Weather Indicators (weather_layer=True)

Annually-aggregated ERA5-Land climate indicators, joined (before the POI join) by
snapping each building to the nearest 0.1° ERA5-Land cell that carries data
(coastal buildings fall back to the closest inland cell). Column names follow
`we__{indicator}_{year}` (`we__` namespace prefix, snake_case indicator, year as
a **suffix**); one column per requested year (`weather_years`, default the 5 most
recent complete calendar years).

| Column pattern | Unit | Description |
|----------------|------|-------------|
| `we__hdd_{year}` | °C·day | Heating degree days (base `weather_hdd_base`, default 18 °C) |
| `we__cdd_{year}` | °C·day | Cooling degree days (base `weather_cdd_base`, default 21 °C) |
| `we__precipitation_{year}` | mm | Total annual precipitation |
| `we__ghi_{year}` | kWh/m² | Global horizontal irradiation (annual sum) |
| `we__air_temperature_mean_{year}` / `we__air_temperature_min_{year}` / `we__air_temperature_max_{year}` | °C | Mean / min / max 2 m air temperature |
| `we__relative_humidity_mean_{year}` | % | Mean relative humidity |
| `we__wind_speed_mean_{year}` | m/s | Mean 10 m wind speed |

`weather_indicators` selects a subset of these stems (default: all nine). Buildings
with no resolvable cell (e.g. invalid geometry) get `NaN`. Source: Copernicus
Climate Data Store ERA5-Land (see [Data Sources](data-sources-coverage.md)).

## 📍 Points of Interest (pois_layer=True)

POI indicators are added when `pois_layer=True` is passed to `merge()`. Data is sourced from OpenStreetMap and joined at building level for the following groups: `amenity`, `emergency`, `healthcare`, `leisure`, `office`, `shop`, `tourism`.

For groups with subgroups (e.g. `amenity` → `food`, `education`, …) the column prefix is `poi__{group}_{subgroup}`. For groups without subgroups (e.g. `emergency`, `healthcare`) the prefix is `poi__{group}`. Group/subgroup names are snake_case.

### Per-group/subgroup Columns

| Column | Type | Description |
|--------|------|-------------|
| `poi__{prefix}_count` | Integer | Number of POIs located on the building footprint |
| `poi__{prefix}_ids` | List | OSM identifiers of POIs on the building |
| `poi__{prefix}_100m` | Integer | POI count within 100 m of building centroid |
| `poi__{prefix}_250m` | Integer | POI count within 250 m of building centroid |
| `poi__{prefix}_500m` | Integer | POI count within 500 m of building centroid |
| `poi__{prefix}_1000m` | Integer | POI count within 1000 m of building centroid |
| `poi__{prefix}_1500m` | Integer | POI count within 1500 m of building centroid |

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
- `poi__amenity_food_count` — restaurants/cafes/bars on the building
- `poi__amenity_food_250m` — food amenities within 250 m
- `poi__emergency_100m` — emergency facilities within 100 m
- `poi__shop_clothing_500m` — clothing shops within 500 m
- `poi__leisure_1500m` — leisure POIs within 1500 m
- `poi__tourism_accommodation_ids` — OSM IDs of accommodation on the building

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

| Feature Category                                           | Basic                | +CAT Files           | +Inference           | +Inference+CAT       |
|------------------------------------------------------------|----------------------|----------------------|----------------------|----------------------|
| Core Building Info                                         | ✓                    | ✓                    | ✓                    | ✓                    |
| Address Info                                               | ✓                    | ✓                    | ✓                    | ✓                    |
| Administrative                                             | ✓                    | ✓                    | ✓                    | ✓                    |
| Usage by Area                                              | -                    | ✓                    | -                    | ✓                    |
| Geometric Analysis                                         | -                    | -                    | ✓                    | ✓                    |
| Orientation Analysis                                       | -                    | -                    | ✓                    | ✓                    |
| Thermal Properties                                         | -                    | -                    | ✓                    | ✓                    |
| Space Breakdown                                            | -                    | -                    | -                    | ✓                    |
| Economic Data                                              | -                    | ✓                    | -                    | ✓                    |
| Heavy shadow/terrain columns                               | -                    | -                    | on-demand¹           | on-demand¹           |
| Space-level `sbr_results`                                  | -                    | -                    | ✓                    | ✓                    |
| POI Proximity                                              | pois_layer=True      | pois_layer=True      | pois_layer=True      | pois_layer=True      |
| Socioeconomic indicators                                   | social_ES_layer=True | social_ES_layer=True | social_ES_layer=True | social_ES_layer=True |
| Weather indicators                                         | weather_layer=True   | weather_layer=True   | weather_layer=True   | weather_layer=True   |
| Essential Characteristics of Households and People `ec__*` | -                    | social_ES_layer=True | -                    | social_ES_layer=True |
| Commercial Premises                                        | Barcelona Only       | Barcelona Only       | Barcelona Only       | Barcelona Only       |

¹ Never in the returned frame — fetch with `get_heavy_columns_for_building()`
(see [Heavy BLOB columns](#-heavy-blob-columns-on-demand-only)).

---

[← Configuration & Examples](configuration-examples.md) | [Contributing →](contributing.md) | [Back to README](../README.md)