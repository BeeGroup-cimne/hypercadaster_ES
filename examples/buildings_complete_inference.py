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
force_update = False
cadaster_codes = ["25011"] #08900: Barcelona, 25023: Alpicat, 25285: Torrefarrera

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
        # The census occupancy the energy model runs on: so__population_{year} is
        # what ctagg__people_per_m2 / _per_dwelling are built from, and without it
        # every dwelling falls back to the CTE default density.
        social_ES_layer=True,
        force_update=force_update,              # wipe central DB rows + cache shards for this code
    )

    hc.utils.stage_log(
        f"hc.merge() returned {len(gdf_municipality)} rows for {municipality}; "
    )

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
    count = len(combined_gdf[combined_gdf['cadaster_code'] == municipality])
    print(f"  - {municipality}: {count:,} buildings")

print(f"\nResults saved with compression to reduce file size.")
print(f"Individual files: [municipality]_complete_inference.pkl")
print(f"Combined file: {'~'.join(cadaster_codes)}_complete_inference.pkl")

### One building analysis
# Get a building
_test_br = "3444707BG9034S"#combined_gdf["building_reference"].iloc[10]#"3352402BG9035S"#
pprint.pp(combined_gdf[combined_gdf["building_reference"]==_test_br].iloc[0].to_dict())

# Generate plots for that building
hc.utils.plot_building_reference(combined_gdf,_test_br,output_path=f"{wd}/test.png")
# `wd=wd` auto-fetches the heavy shadow / terrain / walls_segments BLOB columns from
# the central DuckDB for this one building (excluded from the bulk
# `hc.merge` frame for memory reasons — see the [hc.merge] notice at
# the end of the run, and `hc.utils.get_heavy_columns_for_building`).
hc.utils.plot_building_shadows_analysis(combined_gdf, _test_br, output_path=f"{wd}/test2.png", wd=wd)
hc.utils.plot_building_dem(combined_gdf, _test_br, output_path=f"{wd}/test3.png", wd=wd)
hc.utils.plot_building_geometry(combined_gdf, _test_br, output_path=f"{wd}/test4.png", wd=wd)
# 2-D per-floor thermal-zone floor plan (façade-fronted dwellings, communal core
# with the ground-floor entrance corridor) — same partition the 3-D view / IDF use.
hc.utils.plot_building_thermal_zones(combined_gdf, _test_br, output_path=f"{wd}/test4b.png", wd=wd)
hc.utils.plot_building_3d_geometry(combined_gdf, _test_br, output_path=f"{wd}/test5.png", wd=wd)

# Create the EnergyPlus model of the building. The neighbour-shading filter runs
# HERE, as the model is built (config `shading.neighbours`), so the returned IDF,
# the .idf on disk, the plot below and the simulation all carry the same panels:
# those that can actually shade this building, capped at `max_shading`.
idf = hc.energy.build_energy_model(
    combined_gdf, _test_br, wd=wd, weather_year=2021,
    filter_shading=True,          # uncomment to keep every neighbour panel
)
# Plot the IDF OBJECT (not the file) so what you see is exactly what will be
# simulated below — same object, same filtered shading.
hc.energy.plot_idf_geometry(idf,
                     output_path=f"{wd}/test6.png",
                     shading_opacity=0.6,
                     opacity=0.8,
                     include_shading=True,
                     title=_test_br)

# ---------------------------------------------------------------------------
# Occupant-behaviour SCENARIO ANALYSIS: run the building through EnergyPlus
# under five occupant-behaviour scenarios (random mix of user types; all
# unaware with random ventilation + erratic daytime HVAC use; unaware free-
# floating with random ventilation; aware free-floating with optimal-moment
# ventilation; all aware maintaining 19-22 C winter / 24-26 C summer). Returns
# global KPIs per thermal-zone use type (occupied zones only), SPLIT into the
# heating (Oct-Apr) and cooling (May-Sep) periods, using EN 16798-1 adaptive
# comfort and the social_ES-derived occupancy/conservation inputs when present
# (else defaults).
#
# Passing `idf=` reuses the model built above: every scenario runs on a COPY of it,
# so they share one geometry/envelope/shading by construction (not by trusting six
# rebuilds to agree), and the build cost is paid once. Only the behaviour-dependent
# wiring — archetype, setpoints, HVAC availability, ventilation — is re-applied per
# scenario. Drop `idf=` to have each scenario rebuild the model instead; the KPIs
# are identical either way.
#
# NOTE full-year runs get slower with more neighbour-shading surfaces (capped at
# build time by `max_shading` above); `run_period_months` can restrict the
# simulated period for a quick check (drop it for the true annual KPIs).
# `run_behavior_scenarios` takes the ONE building's frame and reads its reference
# off the `building_reference` field. That row is self-sufficient: `hc.merge`
# already wrote its census tract's aggregates onto it (`ctagg__*`, notably
# `ctagg__people_per_m2`), so the population-driven DHW occupancy survives the
# filtering down to a single building — no surrounding frame needed.
# `idf_path=` runs the model as it stands ON DISK, so anything you edit into the
# .idf between building it and running it is what gets simulated. build_energy_model
# writes a `{idf}.hcmeta.json` sidecar alongside carrying the zone -> use-type map,
# which the file format itself cannot hold; keep the two together (and update the
# sidecar if you rename a zone). Pass `idf=idf` instead to run the in-memory model.
kpis = hc.energy.run_behavior_scenarios(
    combined_gdf[combined_gdf["building_reference"] == _test_br],
    wd=wd, weather_year=2021,
    idf_path=idf.idfname,             # the model built + plotted above, from disk
    # run_period_months=[7],         # uncomment for a fast July-only smoke test
)
# Interactive HTML report + the KPI table as CSV. The report holds the summarised
# comparisons (demand, gains, delivered setpoint, discomfort, the indoor-temperature
# distribution, DHW) and the RAW hourly EnergyPlus series each scenario produced,
# including the worst cooling and heating weeks at hourly resolution. `output_dir`
# is where the scenario runner left its per-scenario eplusout.csv; `zone_areas`
# makes the building temperature floor-area weighted rather than a plain mean of
# unequal rooms.
# ONE PAGE PER USE: a dwelling, a shop and the parking box below them are not one
# building to average — the parking alone (no occupants, 1 ren/h) lifted the
# building air-change curve by a tenth and pulled the CO2 down towards the outdoor
# background, so the page described neither. `zone_uses` is what splits them; drop
# it and the sidecar beside the IDF is read instead. Returns {use: path}.
reports = hc.reporting.scenario_report(
    kpis,
    output_path_html=f"{wd}/{_test_br}_scenarios.html",
    output_path_csv=f"{wd}/{_test_br}_scenario_kpis.csv",
    output_dir=f"{wd}/energyplus/scenarios",
    building_reference=_test_br,
    zone_areas=hc.energy._zone_floor_areas(idf),
    zone_uses=hc.energy.zone_use_categories(idf),
    year=2021,                        # labels the hourly charts with the weekday
)
for _use, _path in reports.items():
    print(f"{_use} report written to {_path}")
print("\n=== Behaviour-scenario KPIs per use type ===")
print(kpis.to_string(index=False))
