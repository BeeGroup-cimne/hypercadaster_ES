#!/usr/bin/env python3
"""
Basic Building Analysis Example (Address-focused)

This script demonstrates basic, province-scale building data processing with
hypercadaster_ES **without** the (memory-intensive) building-parts inference —
it focuses on addresses, CAT space classification and POIs for the whole Lleida
province.

Key features demonstrated:
- Province-level download and merge
- National auxiliary layers (postal codes, census tracts, elevation)
- CAT files + POI enrichment + address compaction, inference disabled

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc
import os
import pandas as pd

# Configuration
# Working directory where all data will be stored
wd = "/mnt/data2/hypercadaster_ES"

# Configuration parameters for merge operation
# These are explicitly defined for clarity and potential modification
output_filename = f"{wd}/lleida.pkl"
province_codes = ["25"]            # Whole Lleida province
cadaster_codes = None#["25257"]#None
neighborhood_layer = False         # Barcelona-only; must be False elsewhere
postal_code_layer = True           # Include postal code boundaries
census_layer = True                # Include INE census tract data
elevations_layer = True            # Include digital elevation model
open_data_layers = False           # Barcelona-only; must be False elsewhere
building_parts_inference = False    # Geometric inference (single municipality only)
building_parts_plots = False        # Plot generation (single municipality only)
plot_zones_ratio = 0.01            # Plot 1% of zones (only used when plots enabled)
use_CAT_files = True               # Use detailed CAT format files
CAT_files_rel_dir = "cadaster/CAT_files"    # Directory containing CAT files
compact_addresses = True
pois_layer = True
force_update = True
cleanup_shards = True

if not os.path.exists(output_filename) or force_update:
    # Download all required data
    print("Downloading cadastral and auxiliary data...")
    hc.download(
        wd=wd,
        province_codes=province_codes,
        cadaster_codes=cadaster_codes,
        neighborhood_layer=neighborhood_layer,      # Barcelona-only layer
        open_data_layers=open_data_layers,        # Barcelona-only layer
    )

    # Perform the merge (inference disabled)
    print("Merging cadastral + auxiliary layers (inference disabled)...")
    gdf = hc.merge(
        wd=wd,
        province_codes=province_codes,
        cadaster_codes=cadaster_codes,
        neighborhood_layer=neighborhood_layer,
        postal_code_layer=postal_code_layer,
        census_layer=census_layer,
        elevations_layer=elevations_layer,
        open_data_layers=open_data_layers,
        pois_layer=pois_layer,
        building_parts_inference=building_parts_inference,
        building_parts_plots=building_parts_plots,
        plot_zones_ratio=plot_zones_ratio,  # New parameter for controlling plot output
        use_CAT_files=use_CAT_files,
        CAT_files_rel_dir=CAT_files_rel_dir,
        compact_addresses=compact_addresses,
        force_update=force_update,
        cleanup_shards=cleanup_shards
    )

    # Save results with address-focused naming
    print(f"Saving results to: {output_filename}")
    gdf = hc.utils.flat_gdf_to_building_spaces_resolution(gdf)
    gdf.to_pickle(output_filename, compression="gzip")
    print(f"Analysis completed. Processed {len(gdf)} building spaces records.")
else:
    gdf = pd.read_pickle(output_filename, compression="gzip")
    print(f"Analysis completed. {len(gdf)} building spaces records were read.")
