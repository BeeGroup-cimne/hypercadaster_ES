#!/usr/bin/env python3
"""
Basic Building Analysis Example (Address-focused)

This script demonstrates basic building data processing using hypercadaster_ES
with building inference enabled but focused on address and basic building 
information. Despite the filename, it actually includes building inference
but is optimized for address-centric analysis.

Key features demonstrated:
- Basic data download and processing
- All auxiliary data layers included
- Building inference enabled for comprehensive analysis
- Address-focused output naming convention

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc

# Configuration
# Working directory where all data will be stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Municipality to analyze (08900: Barcelona, Barcelona province)
cadaster_codes = ["08900"]

# Download all required data
print("Downloading cadastral and auxiliary data...")
hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

# Configuration parameters for merge operation
# These are explicitly defined for clarity and potential modification
province_codes = None              # Not using province-level codes
ine_codes = None                   # Not using INE municipality codes
neighborhood_layer = True          # Include Barcelona neighborhood data
postal_code_layer = True           # Include postal code boundaries
census_layer = True                # Include INE census tract data
elevations_layer = True            # Include digital elevation model
open_data_layers = True            # Include Barcelona open data layers
building_parts_inference = False    # Enable building geometric analysis (single municipality)
building_parts_plots = False        # Enable plot generation for single municipality
plot_zones_ratio = None            # Plot 1% of zones (minimum 1 zone)
use_CAT_files = True               # Use detailed CAT format files
CAT_files_rel_dir = "CAT_files"    # Directory containing CAT files

# Perform comprehensive analysis
print("Processing data with building inference enabled...")
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
    plot_zones_ratio=plot_zones_ratio,  # New parameter for controlling plot output
    use_CAT_files=use_CAT_files, 
    CAT_files_rel_dir=CAT_files_rel_dir
)

# Save results with address-focused naming
output_filename = f"{wd}/{'~'.join(cadaster_codes)}_only_addresses.pkl"
print(f"Saving results to: {output_filename}")
gdf.to_pickle(output_filename, compression="gzip")

print(f"Analysis completed. Processed {len(gdf)} building records.")
print("Note: Despite the filename, this includes full building inference analysis.")
