#!/usr/bin/env python3
"""
Complete Building Inference Analysis Example

This script demonstrates a comprehensive building analysis using hypercadaster_ES
with all available inference features enabled. Due to computational constraints,
building inference can only be enabled for single municipalities, so this example
processes multiple municipalities individually and then combines the results.

Key features demonstrated:
- Complete data download with all available layers
- Advanced building inference with geometric analysis (single municipality processing)
- CAT files integration for detailed building space information
- Multi-municipality processing with result combination
- Random zone plotting for visualization

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc
import pandas as pd

# Configuration
# Working directory where all data will be stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Municipality codes to analyze (25023: Alpicat, 25285: Torrefarrera - both in Lleida province)
cadaster_codes = ["25023", "25285"]

# Download all required data for all municipalities at once
print("Downloading cadastral and auxiliary data for all municipalities...")
hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

# Process each municipality individually due to building inference constraints
print("\n" + "="*60)
print("PROCESSING MUNICIPALITIES INDIVIDUALLY FOR BUILDING INFERENCE")
print("="*60)

all_results = []

for i, municipality in enumerate(cadaster_codes, 1):
    print(f"\n[{i}/{len(cadaster_codes)}] Processing municipality: {municipality}")
    print("-" * 50)
    
    # Process single municipality with full inference
    gdf_municipality = hc.merge(
        wd=wd,
        cadaster_codes=municipality,    # Single municipality
        neighborhood_layer=True,        # Include Barcelona neighborhood data (if applicable)
        postal_code_layer=True,         # Include postal code boundaries
        census_layer=True,              # Include INE census tract data
        elevations_layer=True,          # Include digital elevation model
        open_data_layers=True,          # Include Barcelona open data layers (if applicable)
        building_parts_inference=True,  # ✅ Now allowed: single municipality
        building_parts_plots=True,      # ✅ Enable plots for single municipality
        plot_zones_ratio=0.02,          # Plot 2% of zones for visualization
        use_CAT_files=True,             # Use detailed CAT format files for space analysis
        CAT_files_rel_dir="CAT_files"   # Directory containing CAT files
    )
    
    # Add municipality identifier for tracking
    gdf_municipality['processing_municipality'] = municipality
    
    # Save individual result
    individual_filename = f"{wd}/{municipality}_complete_inference.pkl"
    gdf_municipality.to_pickle(individual_filename, compression="gzip")
    print(f"Individual result saved: {individual_filename}")
    print(f"Buildings processed: {len(gdf_municipality):,}")
    
    all_results.append(gdf_municipality)

# Combine all results
print("\n" + "="*60)
print("COMBINING RESULTS FROM ALL MUNICIPALITIES")
print("="*60)

if len(all_results) > 1:
    print("Concatenating results from all municipalities...")
    combined_gdf = pd.concat(all_results, ignore_index=True)
else:
    combined_gdf = all_results[0]

# Save combined results
output_filename = f"{wd}/{'~'.join(cadaster_codes)}_complete_inference.pkl"
print(f"Saving combined results to: {output_filename}")
combined_gdf.to_pickle(output_filename, compression="gzip")

# Summary statistics
print("\n" + "="*60)
print("ANALYSIS COMPLETED")
print("="*60)
print(f"Total buildings processed: {len(combined_gdf):,}")
print(f"Municipalities analyzed: {len(cadaster_codes)}")
print(f"Municipality breakdown:")
for municipality in cadaster_codes:
    count = len(combined_gdf[combined_gdf['processing_municipality'] == municipality])
    print(f"  - {municipality}: {count:,} buildings")

print(f"\nResults saved with compression to reduce file size.")
print(f"Individual files: [municipality]_complete_inference.pkl")
print(f"Combined file: {'~'.join(cadaster_codes)}_complete_inference.pkl")

# Note: The following commented list contains problematic zone references
# from Barcelona analysis that may require special handling in future versions
# failing_zones_bcn = [
#     '03735DF3807C', '05192DF3801H', '08169DF3801F', '11859DF3718E', '12078DF3910E', '13182DF3811G', '14063DF3810E',
#     '15923DF3819C', '18654DF3816F', '22043DF3820C', '22157DF3821E', '22343DF3823C', '23157DF3821E', '23166DF3821E',
#     '25281DF3822H', '25423DF3824D', '26474DF3824H', '27737DF2837A', '28394DF3823H', '28833DF3828D', '29646DF3826D',
#     '31326DF3833C', '31458DF3834E', '44546DF2845C', '47518DF2845B', '49494DF3844H', '54525DF2855A', '59243DF2852D',
#     '65192DF2861H', '75688DF2876H', '81041DF2880C', '83526DF2885A', '84654DF2786E', '86789DF2887H', '88235DF2882D',
#     '92827DF2898C', '93655DF2796E', 'unknown'
# ]