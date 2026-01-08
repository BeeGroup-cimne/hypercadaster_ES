#!/usr/bin/env python3
"""
Optimized Building Analysis from Bounding Box - Batched Processing

This script demonstrates how to efficiently analyze building data for all municipalities
within a specified geographical bounding box by processing them in smaller batches
to avoid memory issues and provide better monitoring.

Key optimizations:
- Processes municipalities in configurable batches 
- Shows detailed progress and timing information
- Uses optimized zone joining with vectorized operations
- Handles memory more efficiently
- Provides checkpoints and can resume from failures

Author: hypercadaster_ES examples  
"""

import hypercadaster_ES as hc
import pandas as pd
import time
import pprint as pp
import os

# Configuration
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Define bounding box in longitude/latitude format [min_lon, min_lat, max_lon, max_lat]
# Barcelona metropolitan area (full)
bbox = [1.9, 41.2, 2.6, 41.6]

# Processing configuration
BATCH_SIZE = 10  # Process 10 municipalities at a time
MAX_TOTAL_TIME = 60 * 30  # Maximum 30 minutes total processing

print("=== OPTIMIZED BATCHED BUILDING ANALYSIS ===")
print(f"Bounding box: {bbox} (lon_min, lat_min, lon_max, lat_max)")
print(f"Batch size: {BATCH_SIZE} municipalities")

overall_start = time.time()

try:
    print("Discovering municipalities within bounding box...")
    ine_codes = hc.get_ine_codes_from_bounding_box(wd, bbox)
    print(f"Found {len(ine_codes)} municipalities in bounding box")
    
    names = hc.municipality_name(ine_codes)
    print("Municipalities:")
    pp.pprint(names[:20], compact=True)  # Show first 20 
    if len(names) > 20:
        print(f"... and {len(names)-20} more")

    # Convert INE codes to cadastral codes
    all_cadaster_codes = hc.ine_to_cadaster_codes(ine_codes)
    print(f"\nTotal cadastral codes to process: {len(all_cadaster_codes)}")
    
    # Process in batches
    all_gdfs = []
    total_buildings = 0
    
    for batch_start in range(0, len(all_cadaster_codes), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_cadaster_codes))
        batch_codes = all_cadaster_codes[batch_start:batch_end]
        batch_ine_codes = ine_codes[batch_start:batch_end]
        
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_cadaster_codes) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n=== BATCH {batch_num}/{total_batches} ===")
        print(f"Processing municipalities {batch_start+1}-{batch_end}/{len(all_cadaster_codes)}")
        print(f"Cadaster codes: {batch_codes}")
        
        batch_start_time = time.time()
        
        # Check if we're approaching time limit
        elapsed_time = time.time() - overall_start
        if elapsed_time > MAX_TOTAL_TIME:
            print(f"⚠️  Approaching time limit ({elapsed_time/60:.1f} min), stopping processing")
            break
        
        try:
            # Download data for this batch if needed
            print("Downloading/checking data for batch...")
            hc.download(
                wd=wd,
                cadaster_codes=batch_codes
            )
            
            # Process with optimized functions 
            print("Processing batch with optimized merge function...")
            gdf_batch = hc.merge(
                wd=wd, 
                cadaster_codes=batch_codes,
                ine_codes=batch_ine_codes,
                neighborhood_layer=False,   # Disable for faster processing
                postal_code_layer=False,    # Disable for faster processing  
                census_layer=False,         # Disable for faster processing
                elevations_layer=False,     # Disable for faster processing
                open_data_layers=False,     # Disable for faster processing
                building_parts_inference=False,
                building_parts_plots=False,
                use_CAT_files=False
            )
            
            batch_time = time.time() - batch_start_time
            batch_buildings = len(gdf_batch)
            total_buildings += batch_buildings
            
            print(f"✓ Batch {batch_num} completed in {batch_time:.1f}s")
            print(f"✓ Processed {batch_buildings:,} buildings")
            print(f"✓ Running total: {total_buildings:,} buildings")
            
            # Check zone assignment quality
            if 'zone_reference' in gdf_batch.columns:
                missing_zones = gdf_batch['zone_reference'].isna().sum()
                zone_coverage = (1 - missing_zones/len(gdf_batch)) * 100
                print(f"✓ Zone assignment coverage: {zone_coverage:.1f}%")
            
            all_gdfs.append(gdf_batch)
            
        except Exception as e:
            print(f"✗ Batch {batch_num} failed: {e}")
            print("Continuing with next batch...")
            continue
        
        # Memory management hint
        if batch_num % 3 == 0:  # Every 3 batches
            print(f"💾 Processed {batch_num} batches, {total_buildings:,} total buildings")
    
    # Combine results if we have any successful batches
    if all_gdfs:
        print(f"\n=== COMBINING RESULTS ===")
        print(f"Combining {len(all_gdfs)} successful batches...")
        
        combine_start = time.time()
        final_gdf = pd.concat(all_gdfs, ignore_index=True)
        combine_time = time.time() - combine_start
        
        print(f"✓ Combined results in {combine_time:.1f}s")
        print(f"✓ Final dataset: {len(final_gdf):,} building records")
        
        # Save results
        output_filename = f"{wd}/optimized_barcelona_metro_{len(final_gdf)}_buildings.pkl" 
        print(f"Saving results to: {output_filename}")
        final_gdf.to_pickle(output_filename, compression="gzip")
        
        # Summary statistics
        if 'zone_type' in final_gdf.columns:
            zone_stats = final_gdf['zone_type'].value_counts()
            print(f"✓ Zone type distribution: {zone_stats.to_dict()}")
        
    else:
        print("✗ No successful batches processed")

except Exception as e:
    print(f"✗ Processing failed: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - overall_start
print(f"\n=== PROCESSING COMPLETED ===")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Final building count: {total_buildings:,}")
print(f"Average rate: {total_buildings/(total_time/60):.0f} buildings/minute")