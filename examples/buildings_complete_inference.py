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
import pprint

# Configuration
wd = "/mnt/data2/hypercadaster_ES"
# Resume mode: reuse the per-zone shards already in results/cache instead of
# wiping and recomputing the (multi-hour) KPI inference. Set back to True for
# a clean production run.
force_update = True
cadaster_codes = ["25900"] #08900: Barcelona, 25023: Alpicat, 25285: Torrefarrera

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
    #break
    print(f"\n[{i}/{len(cadaster_codes)}] Processing municipality: {municipality}")
    print("-" * 50)
    individual_filename = f"{wd}/{municipality}_complete_inference.pkl"
    # Process single municipality with full inference
    gdf_municipality = hc.merge(
        wd=wd,
        cadaster_codes=[municipality],  # Single municipality (as list)
        neighborhood_layer=True,        # Include Barcelona neighborhood data (if applicable)
        postal_code_layer=True,         # Include postal code boundaries
        census_layer=True,              # Include INE census tract data
        elevations_layer=True,          # Include digital elevation model
        open_data_layers=True,          # Include Barcelona open data layers (if applicable)
        building_parts_inference=True,  # ✅ Now allowed: single municipality
        building_parts_plots=False,     # ✅ Enable plots for single municipality
        plot_zones_ratio=0.02,          # Plot 2% of zones for visualization
        use_CAT_files=True,             # Use detailed CAT format files for space analysis
        CAT_files_rel_dir="cadaster/CAT_files",   # Directory containing CAT files
        pois_layer=True,
        force_update=force_update,              # wipe central DB rows + cache shards for this code
    )

    hc.utils.stage_log(
        f"hc.merge() returned {len(gdf_municipality)} rows for {municipality}; "
        f"starting plots + pickle write"
    )

    # Get a building
    _test_br = gdf_municipality["building_reference"].iloc[0]
    pprint.pp(gdf_municipality[gdf_municipality["building_reference"]==_test_br].iloc[0].to_dict())

    # Generate plots for that building
    hc.utils.plot_building_reference(gdf_municipality,_test_br,output_path=f"{wd}/test.png")
    # `wd=wd` auto-fetches the heavy shadow / terrain / walls_segments BLOB columns from
    # the central DuckDB for this one building (excluded from the bulk
    # `hc.merge` frame for memory reasons — see the [hc.merge] notice at
    # the end of the run, and `hc.utils.get_heavy_columns_for_building`).
    hc.utils.plot_building_shadows_analysis(gdf_municipality, _test_br, output_path=f"{wd}/test2.png", wd=wd)
    hc.utils.plot_building_dem(gdf_municipality, _test_br, output_path=f"{wd}/test3.png", wd=wd)
    hc.utils.plot_building_geometry(gdf_municipality, _test_br, output_path=f"{wd}/test4.png", wd=wd)
    hc.utils.plot_building_3d_geometry(gdf_municipality, _test_br, output_path=f"{wd}/test5.png", wd=wd)

    # Create the EnergyPlus model of the building
    idf = hc.energy.build_energy_model(gdf_municipality, _test_br, wd=wd)
    hc.energy.plot_idf_geometry(f"{wd}/energyplus/{_test_br}.idf",
                         output_path=f"{wd}/test6.png",
                         include_shading=True,
                         title=_test_br)
    # Save individual result
    with hc.utils.stage_timer(
        f"example: to_pickle gzip ({len(gdf_municipality)} rows -> {individual_filename})"
    ):
        gdf_municipality.to_pickle(individual_filename, compression="gzip")
    print(f"Individual result saved: {individual_filename}")
    print(f"Buildings processed: {len(gdf_municipality):,}")

    all_results.append(gdf_municipality)

if len(cadaster_codes)>1:

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

