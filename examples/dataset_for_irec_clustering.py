#!/usr/bin/env python3
"""
IREC Building Energy Simulation Dataset Preparation

This script prepares building data for energy simulation analysis by combining
cadastral building information with weather clustering data, social demographics,
and energy performance indicators. It transforms hypercadaster_ES output into
the format required for IREC (Institute for Energy Research of Catalonia)
building energy simulations.

Key features demonstrated:
- Building data transformation for energy simulation
- Weather station clustering and spatial assignment
- Social demographic data integration from INE
- Energy performance certificate (EPC) data integration
- Spatial analysis and visualization

Dependencies:
- hypercadaster_ES
- social_ES (for INE demographic data)
- geopandas, pandas, shapely

Author: hypercadaster_ES examples
"""

import pandas as pd
from hypercadaster_ES import interoperability
import social_ES.utils_INE as sc
import geopandas as gpd
from shapely.geometry import Point

# Configuration
# Working directory where all data is stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Municipality to analyze (08900: Barcelona)
cadaster_codes = ["08900"]

print("Loading processed building data...")
# Read the previously processed pickle file with building data
gdf = pd.read_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")

# Filter buildings that have valid location information
gdf = gdf[gdf["location"].notnull()]
print(f"Loaded {len(gdf)} buildings with valid locations")

print("Converting to IREC simulation format...")
# Transform general building data to IREC's specific format
# This function restructures the data with proper column names and units
gdf_irec = interoperability.input_files_for_IREC_simulations(gdf)

print("Loading and processing weather station data...")
# Load weather station clustering data
weather = pd.read_parquet(f"{wd}/results/third_party_datasets/weatherStation_by_cluster.parquet")

# Create proper geometry column for weather stations (longitude, latitude order)
weather["geometry"] = weather.apply(lambda x: Point(float(x["longitude"]), float(x["latitude"])), axis=1)
weather_gdf = gpd.GeoDataFrame(weather, geometry="geometry", crs="EPSG:4326")

# Ensure building data has proper geometry column for spatial operations
if "geometry" not in gdf_irec.columns:
    # Use 'Location' column as geometry if no geometry column exists
    gdf_irec = gpd.GeoDataFrame(gdf_irec, geometry="Location", crs="EPSG:25831")
else:
    gdf_irec = gdf_irec.set_geometry("geometry")

# Project weather data to same CRS as building data for accurate distance calculations
weather_gdf = weather_gdf.to_crs("EPSG:25831")

print("Assigning weather clusters to buildings...")
# Perform spatial join to assign nearest weather station cluster to each building
gdf_irec = gpd.sjoin_nearest(
    gdf_irec,
    weather_gdf[["Cluster", "geometry"]],
    how="left",
    distance_col="distance"  # Adds distance to nearest weather station
)

# Rename cluster column to be more descriptive
gdf_irec.rename(columns={"Cluster": "WeatherCluster"}, inplace=True)

# Clean up unnecessary join columns
gdf_irec = gdf_irec.drop(columns=["index_right"])

print("Integrating social demographic data...")
# Load INE (Spanish Statistical Institute) social demographic data
atlas = sc.INERentalDistributionAtlas(wd)
atlas = atlas["Sections"]  # Use census section level data

# Create unique census tract identifier
atlas["census_tract"] = atlas["Municipality code"] + atlas["District code"] + atlas["Section code"]

# Use most recent available data (2022)
atlas = atlas[atlas["Year"] == 2022]

# Merge household size data with building data
gdf_irec = gdf_irec.merge(
    atlas[["census_tract", "Tamaño medio del hogar"]],
    left_on="CensusTract", 
    right_on="census_tract", 
    how="left"
)

# Clean up and rename columns
gdf_irec.drop(columns=["census_tract"], inplace=True)
gdf_irec.rename(columns={"Tamaño medio del hogar": "NumberOfPeoplePerHousehold"}, inplace=True)

print("Adding energy performance data...")
# Load energy performance certificate (EPC) prediction results
epc = pd.read_parquet(f"{wd}/results/third_party_datasets/epc_predictor_results.parquet")

# Merge EPC data with building data
gdf_irec = gdf_irec.merge(
    epc[["building_reference", "WindowToWallRatio", "EPCs_ratio"]],
    left_on="BuildingReference", 
    right_on="building_reference", 
    how="left"
)
gdf_irec.drop(columns=["building_reference"], inplace=True)

print("Finalizing dataset...")
# Calculate additional derived metrics
gdf_irec["AverageDwellingArea"] = gdf_irec["UsefulResidentialArea"] / gdf_irec["NumberOfDwelling"]

# Convert to geographic coordinate system for final output
gdf_irec = gdf_irec.to_crs("EPSG:4326")

# Extract latitude and longitude for compatibility with simulation tools
gdf_irec["Latitude"] = gdf_irec["Location"].y
gdf_irec["Longitude"] = gdf_irec["Location"].x
gdf_irec["Projection"] = "EPSG:4326"

print("Generating visualization plots...")
# Create visualization of building weather cluster assignments
interoperability.plot_weather_stations(
    gdf_irec, 
    "WeatherCluster",
    f"{wd}/results/building_weather_clusters.png"
)

# Create visualization of weather station clusters
interoperability.plot_weather_stations(
    weather_gdf, 
    "Cluster",
    f"{wd}/results/weather_stations_clusters.png"
)

print("Saving final dataset...")
# Remove geometry column before saving (keep lat/lon instead)
gdf_irec.drop(columns=["Location"], inplace=True)

# Save final dataset for IREC simulations
output_path = f"{wd}/results/IREC_bcn_input.pkl"
gdf_irec.to_pickle(output_path)

print(f"Dataset preparation completed!")
print(f"Final dataset contains {len(gdf_irec)} buildings")
print(f"Output saved to: {output_path}")
print("Visualization plots saved to results/ directory")