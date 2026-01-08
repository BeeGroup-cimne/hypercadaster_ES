#!/usr/bin/env python3
"""
Test script for optimized zone joining performance

This script tests the performance improvements made to join_cadaster_zone
using a small bounding box to ensure the optimizations work correctly
before running on the full Barcelona area.

Author: hypercadaster_ES examples
"""

import hypercadaster_ES as hc
import time
import pprint as pp

# Configuration - using a smaller area for testing
# Working directory where all data will be stored
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"

# Define a smaller bounding box for initial testing - just a small part of Barcelona
# This covers roughly the Eixample district
bbox = [2.15, 41.38, 2.17, 41.40]

print("=== TESTING OPTIMIZED ZONE JOINING ===")
print(f"Using small test bounding box: {bbox} (lon_min, lat_min, lon_max, lat_max)")

start_time = time.time()

# Get INE municipality codes for municipalities intersecting the bounding box
try:
    print("Discovering municipalities within test bounding box...")
    ine_codes = hc.get_ine_codes_from_bounding_box(wd, bbox)
    print(f"Found {len(ine_codes)} municipalities in bounding box:")
    names = hc.municipality_name(ine_codes)
    pp.pprint(names, compact=True)

    # Convert INE codes to cadastral codes
    cadaster_codes = hc.ine_to_cadaster_codes(ine_codes)
    print(f"\nCorresponding cadastral codes: {cadaster_codes}")

    # Test if required data files exist
    print("Checking if required data files exist...")
    for code in cadaster_codes:
        try:
            import os
            zone_file = f"{wd}/cadaster/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralzoning.gml"
            if os.path.exists(zone_file):
                print(f"✓ Zone file exists for {code}")
            else:
                print(f"✗ Zone file missing for {code}: {zone_file}")
        except Exception as e:
            print(f"✗ Error checking {code}: {e}")

    # Process with optimized join_cadaster_zone function
    print("\n=== Testing optimized merge function ===")
    merge_start = time.time()
    
    try:
        gdf = hc.merge(
            wd=wd, 
            cadaster_codes=cadaster_codes,
            ine_codes=ine_codes,
            neighborhood_layer=False,   # Disable for faster testing
            postal_code_layer=False,    # Disable for faster testing
            census_layer=False,         # Disable for faster testing
            elevations_layer=False,     # Disable for faster testing
            open_data_layers=False,     # Disable for faster testing
            building_parts_inference=False,
            building_parts_plots=False,
            use_CAT_files=False
        )
        
        merge_time = time.time() - merge_start
        print(f"✓ Merge completed successfully in {merge_time:.2f} seconds")
        print(f"✓ Processed {len(gdf)} building records")
        
        # Check zone assignment results
        zone_stats = gdf['zone_type'].value_counts()
        print(f"✓ Zone type distribution: {zone_stats.to_dict()}")
        
        missing_zones = gdf['zone_reference'].isna().sum()
        print(f"✓ Records without zone assignment: {missing_zones}/{len(gdf)} ({missing_zones/len(gdf)*100:.1f}%)")
        
    except Exception as e:
        print(f"✗ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time
print(f"\n=== TEST COMPLETED in {total_time:.2f} seconds ===")