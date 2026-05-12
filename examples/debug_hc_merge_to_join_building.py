#!/usr/bin/env python3
"""
Script to trace from hc.merge() to join_cadaster_building() and debug line by line.
This follows the exact execution path that leads to join_cadaster_building().
"""

import sys

# Add the hypercadaster_ES module to path
sys.path.insert(0, '/')

print("=== STARTING HC.MERGE() EXECUTION PATH ===")

# Configuration - same as your example
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]  # Barcelona
code = cadaster_codes[0]

print(f"Working directory: {wd}")
print(f"Cadaster codes: {cadaster_codes}")

# Import hypercadaster_ES
import hypercadaster_ES as hc

print("Loaded hypercadaster_ES successfully")

# === STEP 1: Download the data ===
print("\n=== STEP 1: Downloading data ===")

# Import the functions that hc.merge() uses
import hypercadaster_ES as hc

# Download data
hc.download(wd, cadaster_codes=cadaster_codes)

# Set up directories like hc.merge() does
cadaster_dir = hc.utils.cadaster_dir_(wd)
census_tracts_dir = hc.utils.census_tracts_dir_(wd)
DEM_raster_dir = hc.utils.DEM_raster_dir_(wd)
results_dir = hc.utils.results_dir_(wd)
open_street_dir = hc.utils.open_street_dir_(wd)
open_data_layers_dir = hc.utils.open_data_dir_(wd)
CAT_files_dir = f"{wd}/CAT_files"

print("\n=== STEP 2: Preparing join_cadaster_data arguments ===")

# Prepare all arguments separately (including defaults)
cadaster_dir = cadaster_dir
census_tracts_dir = census_tracts_dir
DEM_raster_dir = DEM_raster_dir
cadaster_codes = cadaster_codes
results_dir = results_dir
open_street_dir = open_street_dir
building_parts_plots = False  # Default: False
building_parts_inference = True  # We want inference
plot_zones_ratio = 0.02  # Default: 0.01, we use 0.02
use_CAT_files = True  # We want CAT files (this becomes building_parts_inference_using_CAT_files internally)
open_data_layers = True  # We want open data layers
open_data_layers_dir = open_data_layers_dir  # Default: None, we specify it
CAT_files_dir = CAT_files_dir  # Default: None, we specify it
building_parts_inference_using_CAT_files=use_CAT_files

print("Arguments prepared:")
print(f"  cadaster_dir = {cadaster_dir}")
print(f"  census_tracts_dir = {census_tracts_dir}")
print(f"  cadaster_codes = {cadaster_codes}")
print(f"  results_dir = {results_dir}")
print(f"  open_street_dir = {open_street_dir}")
print(f"  building_parts_plots = {building_parts_plots}")
print(f"  building_parts_inference = {building_parts_inference}")
print(f"  plot_zones_ratio = {plot_zones_ratio}")
print(f"  use_CAT_files = {use_CAT_files}")
print(f"  open_data_layers = {open_data_layers}")
print(f"  open_data_layers_dir = {open_data_layers_dir}")
print(f"  CAT_files_dir = {CAT_files_dir}")

print("\n=== STEP 3: Importing and calling join_cadaster_data ===")

# Import the function directly
from hypercadaster_ES.mergers import join_cadaster_data
print("join_cadaster_data imported successfully")

gdf = join_cadaster_data(
    cadaster_dir=cadaster_dir,
    census_tracts_dir=census_tracts_dir,
    DEM_raster_dir=DEM_raster_dir,
    cadaster_codes=cadaster_codes,
    results_dir=results_dir,
    open_street_dir=open_street_dir,
    building_parts_plots=building_parts_plots,
    building_parts_inference=building_parts_inference,
    plot_zones_ratio=plot_zones_ratio,
    use_CAT_files=use_CAT_files,
    open_data_layers=open_data_layers,
    open_data_layers_dir=open_data_layers_dir,
    CAT_files_dir=CAT_files_dir
)

print(f"join_cadaster_data completed!")
print(f"Result type: {type(gdf)}")

if gdf is not None:
    if hasattr(gdf, 'shape'):
        print(f"Result shape: {gdf.shape}")
        print(f"Columns: {list(gdf.columns[:10])}...")  # First 10 columns
        
        # Look for building inference results
        inference_cols = [col for col in gdf.columns if 'br__' in col or 'sbr__' in col]
        print(f"Found {len(inference_cols)} inference columns")
        
        # Look specifically for elevation_at_shadow
        elevation_cols = [col for col in gdf.columns if 'elevation_at_shadow' in col]
        if elevation_cols:
            print(f"Found elevation_at_shadow columns: {elevation_cols}")
        else:
            print("No elevation_at_shadow columns found")
            
        # Sample data
        print("\nSample building references:")
        if 'building_reference' in gdf.columns:
            sample_refs = gdf['building_reference'].dropna().unique()[:5]
            print(f"Sample refs: {sample_refs}")
else:
    print("Result is None")

print("\n=== EXECUTION COMPLETE ===")
print("Now you can debug join_cadaster_building() by setting breakpoints in mergers.py")
print("The execution path is: hc.merge() -> join_cadaster_data() -> join_cadaster_building()")

# === STEP 4: Show where to set breakpoints ===
print("\n=== DEBUG INFORMATION ===")
print("To debug join_cadaster_building():")
print("1. Set breakpoint in /home/gmor/GitHub/hypercadaster_ES/hypercadaster_ES/mergers.py")
print("2. Look for the function definition around line 534:")
print("   def join_cadaster_building(gdf, cadaster_dir, cadaster_codes, results_dir, ...")
print("3. The function is called from join_cadaster_data() around line 135")
print("4. Key parameters passed:")
print(f"   - cadaster_codes: {cadaster_codes}")
print(f"   - building_parts_inference: True")
print(f"   - building_parts_plots: False")
print(f"   - plot_zones_ratio: 0.02")