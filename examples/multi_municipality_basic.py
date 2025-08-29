#!/usr/bin/env python3
"""
Multi-Municipality Basic Analysis Example

This script demonstrates efficient processing of multiple municipalities without
building inference. This approach is optimized for performance when you need
basic cadastral data across many municipalities.

Key features demonstrated:
- Multiple municipality processing in one operation
- All auxiliary data layers included
- No building inference (for faster processing)
- Efficient memory usage for large datasets
- Basic building and address information

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc

# Configuration
# Working directory where all data will be stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Multiple municipality codes to analyze
cadaster_codes = ["25023", "25285", "25120"]  # Alpicat, Torrefarrera, Lleida

print(f"Processing {len(cadaster_codes)} municipalities: {', '.join(cadaster_codes)}")

# Download all required data for all municipalities at once
print("Downloading cadastral and auxiliary data for all municipalities...")
hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

# Process all municipalities together (no inference constraints)
print("Processing data without building inference for optimal performance...")
gdf = hc.merge(
    wd=wd,
    cadaster_codes=cadaster_codes,  # ✅ Multiple municipalities allowed without inference
    neighborhood_layer=True,        # Include Barcelona neighborhood data (if applicable)
    postal_code_layer=True,         # Include postal code boundaries
    census_layer=True,              # Include INE census tract data
    elevations_layer=True,          # Include digital elevation model
    open_data_layers=True,          # Include Barcelona open data layers (if applicable)
    building_parts_inference=False, # ✅ Disabled for multi-municipality processing
    building_parts_plots=False,     # ✅ Disabled for performance
    use_CAT_files=True,             # Use detailed CAT format files for space analysis
    CAT_files_rel_dir="CAT_files"   # Directory containing CAT files
)

# Save results
output_filename = f"{wd}/{'~'.join(cadaster_codes)}_basic_multi.pkl"
print(f"Saving results to: {output_filename}")
gdf.to_pickle(output_filename, compression="gzip")

# Summary statistics
print("\n" + "="*60)
print("MULTI-MUNICIPALITY ANALYSIS COMPLETED")
print("="*60)
print(f"Total buildings processed: {len(gdf):,}")
print(f"Municipalities analyzed: {len(cadaster_codes)}")

# Show municipality breakdown if cadaster_code column exists
if 'cadaster_code' in gdf.columns:
    print(f"Municipality breakdown:")
    for municipality in cadaster_codes:
        count = len(gdf[gdf['cadaster_code'] == municipality])
        print(f"  - {municipality}: {count:,} buildings")
else:
    print("Municipality breakdown not available (cadaster_code column missing)")

print(f"\nResults saved with compression to reduce file size.")
print(f"Output file: {output_filename}")
print("\nNote: For building inference analysis, use buildings_complete_inference.py")
print("which processes municipalities individually due to computational constraints.")