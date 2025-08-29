#!/usr/bin/env python3
"""
GeoJSON Export for Geographic Analysis and Visualization

This script exports processed building and administrative boundary data to GeoJSON
format for use in web mapping applications, GIS analysis, and visualization tools.
It creates multiple geographic layers including buildings, census tracts, districts,
and Barcelona-specific neighborhood boundaries.

Key features demonstrated:
- Building data export with residential filtering
- Census tract boundary processing and export
- Administrative district boundary aggregation
- Barcelona neighborhood boundary processing
- Coordinate reference system management
- Data cleaning and standardization

Output files:
- bcn_buildings_v2.geojson: Residential buildings
- census_tracts.geojson: INE census tract boundaries
- districts.geojson: Administrative district boundaries
- neighborhoods.geojson: Barcelona neighborhood boundaries (when applicable)

Use cases:
- Web mapping applications
- GIS analysis and visualization
- Urban planning applications
- Climate vulnerability mapping
- Spatial analysis tools

Author: hypercadaster_ES examples
"""

import pandas as pd
import geopandas as gpd
import hypercadaster_ES as hc
from shapely import wkt

# Configuration
# Working directory where processed data is stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Municipality to process (08900: Barcelona)
cadaster_codes = ["08900"]

# Output directory for GeoJSON files
# Note: Update this path to match your desired output location
output_dir = "/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA"

print("Processing buildings for GeoJSON export...")

# Load processed building data
print(f"Loading building data from: {wd}")
gdf = pd.read_pickle(f"{wd}/{'~'.join(cadaster_codes)}_only_addresses.pkl", compression="gzip")

# Clean and filter building data for export
print("Filtering and cleaning building data...")

# Remove duplicate building references
gdf = gdf.drop_duplicates("building_reference")

# Filter for residential buildings only and exclude problematic building
# Note: "6873901DF2767D" is excluded due to known data quality issues
gdf = gdf[(gdf.building_area_residential > 0) & 
          ~gdf.building_reference.isin(["6873901DF2767D"])]

print(f"Filtered to {len(gdf)} residential buildings")

# Prepare building data for export
gdf = gdf[["building_reference", "building_geometry"]]
gdf = gdf.rename(columns={
    "building_reference": "reference", 
    "building_geometry": "geometry"
})
gdf = gdf.set_geometry("geometry")

# Ensure proper coordinate reference system (EPSG:25831 - ETRS89 / UTM zone 31N)
gdf = gdf.to_crs(epsg=25831)

# Export buildings to GeoJSON
buildings_output = f"{output_dir}/bcn_buildings_v2.geojson"
print(f"Exporting buildings to: {buildings_output}")
gdf.to_file(buildings_output)

print("Processing census tracts...")

# Load Spanish census tract data from INE (Instituto Nacional de Estadística)
census_shapefile = "/home/gmor/Downloads/seccionado_2025/España_Seccionado2025_ETRS89H30/SECC_CE_20250101.shp"
gdf_ct = gpd.read_file(census_shapefile)

# Filter census tracts for the specified municipalities
# Convert cadastral codes to INE codes for filtering
ine_codes = hc.functions.utils.cadaster_to_ine_codes(
    cadaster_dir=hc.functions.utils.cadaster_dir_(wd),
    cadaster_codes=cadaster_codes
)

gdf_ct = gdf_ct[gdf_ct.CLAU2.isin(ine_codes)]
print(f"Filtered to {len(gdf_ct)} census tracts")

# Prepare census tract data for export
gdf_ct = gdf_ct[["CUSEC", "geometry"]]  # CUSEC is the unique census section code
gdf_ct.rename(columns={"CUSEC": "reference"}, inplace=True)
gdf_ct = gdf_ct.to_crs(epsg=25831)

# Export census tracts to GeoJSON
census_output = f"{output_dir}/census_tracts.geojson"
print(f"Exporting census tracts to: {census_output}")
gdf_ct.to_file(census_output)

print("Processing administrative districts...")

# Load and process district boundaries (aggregated from census data)
gdf_d = gpd.read_file(census_shapefile)
gdf_d = gdf_d[gdf_d.CLAU2.isin(ine_codes)]

# Select district information and aggregate by district code
gdf_d = gdf_d[["CUDIS", "geometry"]]  # CUDIS is the unique district code
gdf_d.rename(columns={"CUDIS": "reference"}, inplace=True)

# Dissolve census tracts by district to create district boundaries
gdf_d = gdf_d.dissolve(by="reference", as_index=False)
print(f"Created {len(gdf_d)} district boundaries")

gdf_d = gdf_d.to_crs(epsg=25831)

# Export districts to GeoJSON
districts_output = f"{output_dir}/districts.geojson"
print(f"Exporting districts to: {districts_output}")
gdf_d.to_file(districts_output)

# Process Barcelona-specific neighborhood data (if Barcelona is included)
print("Checking for Barcelona-specific neighborhood data...")
if any([c == "08900" for c in cadaster_codes]):
    print("Processing Barcelona neighborhoods...")
    
    # Load Barcelona neighborhood data from hypercadaster_ES data directory
    neighborhoods_file = f"{wd}/neighbourhoods/neighbourhoods.csv"
    gdf_n = gpd.read_file(neighborhoods_file)
    
    # Define column mapping for Barcelona neighborhood data
    columns_mapping = {
        "codi_barri": "neighborhood_code",
        "nom_barri": "neighborhood_name", 
        "nom_districte": "district_name",
        "geometria_etrs89": "geometry"
    }
    
    # Rename columns and select relevant data
    gdf_n.rename(columns=columns_mapping, inplace=True)
    gdf_n = gdf_n[list(columns_mapping.values())]
    
    # Convert WKT geometry strings to shapely geometries
    gdf_n["geometry"] = gdf_n["geometry"].apply(wkt.loads)
    gdf_n = gpd.GeoDataFrame(gdf_n, geometry="geometry", crs='EPSG:25831')
    
    # Prepare for export
    gdf_n.rename(columns={"neighborhood_code": "reference"}, inplace=True)
    
    # Create descriptive reference names combining district and neighborhood
    gdf_n["reference_name"] = gdf_n["district_name"] + ', ' + gdf_n["neighborhood_name"]
    
    # Clean up columns
    gdf_n = gdf_n.drop(columns=["district_name", "neighborhood_name"])
    
    # Dissolve by neighborhood reference (in case of multiple polygons per neighborhood)
    gdf_n = gdf_n.dissolve(by="reference", as_index=False)
    print(f"Processed {len(gdf_n)} Barcelona neighborhoods")
    
    gdf_n = gdf_n.to_crs(epsg=25831)
    
    # Export Barcelona neighborhoods to GeoJSON
    neighborhoods_output = f"{output_dir}/neighborhoods.geojson"
    print(f"Exporting Barcelona neighborhoods to: {neighborhoods_output}")
    gdf_n.to_file(neighborhoods_output)
else:
    print("No Barcelona municipalities found - skipping neighborhood processing")

print("\nGeoJSON export completed successfully!")
print("Files created:")
print(f"  - Buildings: {buildings_output}")
print(f"  - Census tracts: {census_output}")
print(f"  - Districts: {districts_output}")
if any([c == "08900" for c in cadaster_codes]):
    print(f"  - Neighborhoods: {neighborhoods_output}")

print("\nAll files are in EPSG:25831 coordinate reference system (ETRS89 / UTM zone 31N)")
print("Files are ready for use in web mapping applications and GIS analysis.")