#!/usr/bin/env python3
"""
Building Facade Geometry Analysis Example

This script demonstrates how to extract and analyze detailed building facade
geometry information from hypercadaster_ES building inference results.
It shows how to work with building orientation data, wall areas, and 
geometric properties for building energy modeling applications.

Key features demonstrated:
- Extraction of air contact wall data by orientation
- Calculation of facade areas for different cardinal directions
- Analysis of adiabatic walls and interior patios
- Building orientation analysis and main facade identification
- Facade ratio calculations for thermal analysis

Use cases:
- Building energy modeling input preparation
- Facade solar gain analysis
- Thermal envelope characterization
- Building orientation optimization studies

Author: hypercadaster_ES examples
"""

import pandas as pd
# import pprint  # Uncomment for detailed data inspection

# Configuration
# Working directory where processed data is stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Municipality to analyze (08900: Barcelona)
cadaster_codes = ["08900"]

print("Loading building data with facade inference results...")
# Load previously processed building data with building inference enabled
gdf = pd.read_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")

# Filter for buildings with valid location data
gdf = gdf[gdf["location"].notnull()]
print(f"Loaded {len(gdf)} buildings with valid locations")

# Select a specific building for detailed facade analysis
# This is a real building reference from Barcelona cadastral data
building_reference = "8523515DF2882D"
gdf_ = gdf[gdf.building_reference == building_reference]

if len(gdf_) == 0:
    print(f"Building {building_reference} not found in dataset")
    exit(1)

print(f"Analyzing facade geometry for building: {building_reference}")

# Uncomment the following line to see all available building data
# pprint.pp(gdf_.iloc[0].to_dict())

# Extract air contact wall data (exterior facade walls by orientation)
# This data comes from building inference analysis
air_contact_raw = gdf_.iloc[0]["br__air_contact_wall"]
air_contact = pd.DataFrame(
    [(int(k), float(v)) for k, v in air_contact_raw.items()],
    columns=["angle", "value"]  # angle in degrees, value is wall length in meters
)

print("Air contact wall data extracted:")
print(f"  - Total facade orientations: {len(air_contact)}")
print(f"  - Total facade length: {air_contact['value'].sum():.2f} meters")

# Building parameters for area calculations
# Average height of walls (assumption for area calculations)
avg_height_of_walls = 3  # meters

# Extract adiabatic walls (internal walls not in contact with exterior)
adiabatic = gdf_.iloc[0]["br__adiabatic_wall"]  # total length in meters

# Extract patio walls (interior courtyards)
patios = gdf_.iloc[0]['br__patios_wall_total']  # total length in meters

print(f"Building wall summary:")
print(f"  - Adiabatic walls: {adiabatic:.2f} meters")
print(f"  - Patio walls: {patios:.2f} meters")

# Calculate facade areas by cardinal directions (8 orientations + totals)
# Each orientation covers 45-degree sectors centered on cardinal/intercardinal directions
print("Calculating facade areas by orientation...")

walls_abs = [
    # North: 337.5° - 22.5° (wrapping around 0°)
    (air_contact[(air_contact["angle"] <= 22.5) | (air_contact["angle"] > 337.5)].value.sum() * avg_height_of_walls),
    # Northeast: 22.5° - 67.5°
    (air_contact[(air_contact["angle"] <= 67.5) & (air_contact["angle"] > 22.5)].value.sum() * avg_height_of_walls),
    # East: 67.5° - 112.5°
    (air_contact[(air_contact["angle"] <= 112.5) & (air_contact["angle"] > 67.5)].value.sum() * avg_height_of_walls),
    # Southeast: 112.5° - 157.5°
    (air_contact[(air_contact["angle"] <= 157.5) & (air_contact["angle"] > 112.5)].value.sum() * avg_height_of_walls),
    # South: 157.5° - 202.5°
    (air_contact[(air_contact["angle"] <= 202.5) & (air_contact["angle"] > 157.5)].value.sum() * avg_height_of_walls),
    # Southwest: 202.5° - 247.5°
    (air_contact[(air_contact["angle"] <= 247.5) & (air_contact["angle"] > 202.5)].value.sum() * avg_height_of_walls),
    # West: 247.5° - 292.5°
    (air_contact[(air_contact["angle"] <= 292.5) & (air_contact["angle"] > 247.5)].value.sum() * avg_height_of_walls),
    # Northwest: 292.5° - 337.5°
    (air_contact[(air_contact["angle"] <= 337.5) & (air_contact["angle"] > 292.5)].value.sum() * avg_height_of_walls),
    # Total air contact (exterior facade)
    air_contact.value.sum() * avg_height_of_walls,
    # Adiabatic walls (interior)
    adiabatic * avg_height_of_walls,
    # Patio walls (interior courtyards)
    patios * avg_height_of_walls
]

# Calculate total wall area for percentage calculations
total_wall_area = (adiabatic * avg_height_of_walls + 
                   air_contact.value.sum() * avg_height_of_walls + 
                   patios * avg_height_of_walls)

# Calculate percentage distribution of wall areas
walls_ratio = [round(wall_abs * 100 / total_wall_area, 2) for wall_abs in walls_abs]

# Round absolute areas for readability
walls_abs = [round(area, 2) for area in walls_abs]

# Determine main building orientation
main_orientation_degrees = int(gdf_.iloc[0]['br__parcel_main_orientation'])
# Convert to orientation index (0-7 for 8 cardinal directions)
orient_ind = int(((main_orientation_degrees + 22.5) % 360) // 45)

# Define orientation labels
air_contact_labels = [
    "N", "NE", "E", "SE", "S", "SW", "W", "NW", 
    "Total air contact", "Adiabatic", "Patio"
]

# Get main orientation label
main_orientation = air_contact_labels[orient_ind]

print(f"Main building orientation: {main_orientation} ({main_orientation_degrees}°)")

# Create comprehensive facade analysis results
facade_analysis = pd.DataFrame({
    "orientation": air_contact_labels,
    "facade_ratio": walls_ratio,    # Percentage of total wall area
    "facade_area": walls_abs        # Absolute area in m²
})

print("\nFacade Analysis Results:")
print(facade_analysis)

print(f"\nBuilding Summary:")
print(f"  - Main orientation: {main_orientation}")
print(f"  - Total wall area: {total_wall_area:.2f} m²")
print(f"  - Exterior facade area: {walls_abs[8]:.2f} m² ({walls_ratio[8]:.1f}%)")
print(f"  - Interior wall area: {walls_abs[9]:.2f} m² ({walls_ratio[9]:.1f}%)")
print(f"  - Patio wall area: {walls_abs[10]:.2f} m² ({walls_ratio[10]:.1f}%)")

# The facade_analysis DataFrame and main_orientation variable are now ready
# for further analysis or export to building energy simulation tools