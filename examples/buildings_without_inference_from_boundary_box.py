#!/usr/bin/env python3
"""
Building Analysis from Bounding Box Example

This script demonstrates how to analyze building data for all municipalities
within a specified geographical bounding box using hypercadaster_ES.
The script automatically discovers municipalities intersecting the bounding box,
converts INE codes to cadastral codes, and processes building data without
building inference for faster execution.

Key features demonstrated:
- Automatic municipality discovery from bounding box coordinates
- INE to cadastral code conversion
- Multi-municipality data download and processing
- All auxiliary data layers included (neighborhoods, postal codes, census, elevations, open data)
- Building processing without geometric inference for performance
- Comprehensive output with address-focused results

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc
import pprint as pp

# Configuration
# Working directory where all data will be stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"  # "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Define bounding box in longitude/latitude format [min_lon, min_lat, max_lon, max_lat]
# Example: Barcelona metropolitan area
bbox = [1.9, 41.2, 2.6, 41.6]

print("Discovering municipalities within bounding box...")
print(f"Bounding box: {bbox} (lon_min, lat_min, lon_max, lat_max)")

# Get INE municipality codes for municipalities intersecting the bounding box
ine_codes = hc.get_ine_codes_from_bounding_box(wd, bbox)
print(f"Found {len(ine_codes)} municipalities in bounding box:")
names = hc.municipality_name(ine_codes)
pp.pprint(names, compact=True)

# Convert INE codes to cadastral codes
cadaster_codes = hc.ine_to_cadaster_codes(ine_codes)
print(f"\nCorresponding cadastral codes: {cadaster_codes}")

# Download all required data for discovered municipalities
print("Downloading cadastral and auxiliary data for all municipalities...")
hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

# Configuration parameters for merge operation
# These are explicitly defined for clarity and potential modification
province_codes = None              # Not using province-level codes
neighborhood_layer = True          # Include neighborhood data where available
postal_code_layer = True           # Include postal code boundaries
census_layer = True                # Include INE census tract data
elevations_layer = True            # Include digital elevation model
open_data_layers = True            # Include open data layers where available
building_parts_inference = False   # Disable building geometric inference for faster processing
building_parts_plots = False       # Disable plot generation
plot_zones_ratio = None            # No plot generation needed
use_CAT_files = True               # Use detailed CAT format files
CAT_files_rel_dir = "CAT_files"    # Directory containing CAT files

# Perform analysis for all municipalities in bounding box
print("Processing data without building inference for faster execution...")
gdf = hc.merge(
    wd=wd, 
    cadaster_codes=cadaster_codes, 
    province_codes=province_codes, 
    ine_codes=ine_codes,
    neighborhood_layer=neighborhood_layer, 
    postal_code_layer=postal_code_layer, 
    census_layer=census_layer,
    elevations_layer=elevations_layer, 
    open_data_layers=open_data_layers,
    building_parts_inference=building_parts_inference, 
    building_parts_plots=building_parts_plots,
    plot_zones_ratio=plot_zones_ratio,
    use_CAT_files=use_CAT_files, 
    CAT_files_rel_dir=CAT_files_rel_dir
)

# Save results with descriptive naming
bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}".replace(".", "p")
output_filename = f"{wd}/bbox_{bbox_str}_municipalities_{'~'.join(cadaster_codes)}.pkl"
print(f"Saving results to: {output_filename}")
gdf.to_pickle(output_filename, compression="gzip")

print(f"Analysis completed. Processed {len(gdf)} building records from {len(cadaster_codes)} municipalities.")
print(f"Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
print("Note: Building inference was disabled for faster multi-municipality processing.")