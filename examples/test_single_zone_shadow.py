#!/usr/bin/env python3
"""Single-zone shadow test for zone 83411DF2884A / building 8341105DF2884A.

Mirrors what `_process_zone_worker` does, but only for one zone. The cached
per-zone result is deleted first so `process_zone` recomputes with the fix
applied to `utils.distance_from_centroid_to_polygons_by_orientation`
(the building_heights lookup was using enumerate's positional index instead
of the GeoSeries' real pandas index label, making every neighbour fail the
`relative_height > 0` gate and forcing all `building_shadows` to inf).
"""
import os
import warnings
import pandas as pd
import polars as pl
import geopandas as gpd
from shapely.geometry import LineString

import hypercadaster_ES as hc
from hypercadaster_ES import building_inference, mergers, utils

warnings.filterwarnings("ignore")

WD = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
CODE = "08900"
ZONE = "83411DF2884A"
BUILDING = "8341105DF2884A"

results_dir = f"{WD}/results"
cache_dir = f"{results_dir}/cache"
cadaster_dir = f"{WD}/cadaster"
open_street_dir = f"{WD}/os_maps"
DEM_raster_dir = f"{WD}/DEM_rasters"

# 1. Drop the stale zone cache (shard DB) so process_zone recomputes.
for ext in (".duckdb", "_sbr_results.pkl", "_br_results.pkl"):
    p = f"{cache_dir}/{ZONE}{ext}"
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed stale cache: {p}")

# 2. Load the prebuilt municipality auxiliaries from the central DuckDB.
# The aux pickles were replaced by aux_building_parts / aux_footprints tables
# in building_results.duckdb (see DESIGN_duckdb_migration.md).
print("Loading aux1/aux2 from DuckDB...")
from hypercadaster_ES import _db as _db_mod
_central = _db_mod.open_shard_read(_db_mod.db_path(results_dir))
try:
    aux_df = _db_mod.read_df_safe(
        _central, "aux_building_parts",
        where="cadaster_code = ?", params=[CODE],
    )
    footprint_df = _db_mod.read_df_safe(
        _central, "aux_footprints",
        where="cadaster_code = ?", params=[CODE],
    )
finally:
    _central.close()

gdf_global = gpd.GeoDataFrame(aux_df).set_geometry("building_part_geometry")
gdf_footprints_global = gpd.GeoDataFrame(footprint_df).set_geometry("building_footprint_by_floor")

# Mirror the production join: attach per-building elevation onto the global
# footprints frame so the shadow-casting gate in process_zone has real
# neighbour heights to compare against the current building.
if "elevation" not in gdf_footprints_global.columns and "elevation" in gdf_global.columns:
    elev_by_ref = gdf_global.groupby("building_reference")["elevation"].mean()
    gdf_footprints_global = gdf_footprints_global.merge(
        elev_by_ref.rename("elevation"),
        left_on="group", right_index=True, how="left",
    )
    print(f"  joined elevation onto footprints; non-null = {gdf_footprints_global['elevation'].notna().sum()}/{len(gdf_footprints_global)}")

# 3. Reconstruct the parcels gdf the way the pipeline does.
print("Reconstructing parcels_gdf...")
parcels_gdf = mergers.join_cadaster_parcel(None, cadaster_dir, [CODE])
parcels_gdf = parcels_gdf.set_geometry("parcel_geometry")
parcels_gdf["parcel_centroid"] = parcels_gdf.centroid
parcels_gdf["zone_reference"] = (
    parcels_gdf["building_reference"].str[0:5]
    + parcels_gdf["building_reference"].str[7:14]
)

# 4. Streets.
print("Loading streets...")
streets_gdf = utils.get_streets_open_street_maps(
    open_street_dir=open_street_dir,
    cadaster_code=CODE,
    crs=gdf_global.crs,
)
streets_gdf = streets_gdf[streets_gdf.geometry.apply(lambda g: isinstance(g, LineString))]

# 5. CAT slice for this zone (so building-spaces aggregation has something to chew on).
#    Use the same parser the real pipeline uses, then filter to this zone.
print("Loading CAT slice...")
buildings_CAT = None
CAT_files_dir = f"{cadaster_dir}/CAT_files"
if os.path.isdir(CAT_files_dir):
    refs_in_zone = list(
        gdf_global.loc[gdf_global["zone_reference"] == ZONE, "building_reference"].unique()
    )
    full_cat = building_inference.parse_horizontal_division_buildings_CAT_files(CODE, CAT_files_dir)
    buildings_CAT = full_cat.filter(
        pl.col("building_reference").is_in(refs_in_zone)
    ).to_pandas()
    print(f"  CAT rows in zone: {len(buildings_CAT)}")

# 6. Run process_zone for just this zone.
print(f"\nRunning process_zone({ZONE!r})...")
gdf_zone = gdf_global[gdf_global["zone_reference"] == ZONE].reset_index(drop=True)
print(f"  parts in zone: {len(gdf_zone)}")
print(f"  buildings in zone: {gdf_zone['building_reference'].nunique()}")

sbr, br = building_inference.process_zone(
    gdf_zone=gdf_zone,
    zone_reference=ZONE,
    gdf_footprints_global=gdf_footprints_global,
    parcels_gdf=parcels_gdf,
    streets_gdf=streets_gdf,
    buildings_CAT=buildings_CAT,
    results_dir=results_dir,
    plots=False,
    orientation_discrete_interval_in_degrees=5,
    floor_height=3.0,
    ground_floor_height=5.0,
    DEM_raster_dir=DEM_raster_dir,
)

print(f"\nprocess_zone returned: sbr={len(sbr) if sbr is not None else None} rows, br={len(br) if br is not None else None} rows")

# 7. Inspect the shadow data for the target building.
import numpy as np
br_row = br[br["building_reference"] == BUILDING].iloc[0]
shadows = br_row["br__shadows_at_distance"]
grid = br_row.get("br__terrain_grid", {})


def _count(d):
    if not isinstance(d, dict):
        return 0, 0
    finite = 0
    inf = 0
    for v in d.values():
        if isinstance(v, list):
            if any(np.isfinite(x) for x in v):
                finite += 1
            else:
                inf += 1
        elif isinstance(v, (int, float)):
            if np.isfinite(v):
                finite += 1
            else:
                inf += 1
    return finite, inf


fb, ib = _count(shadows)
print(f"\n=== {BUILDING} shadow summary AFTER fix ===")
print(f"  shadows orientations (combined building+terrain blocker per floor):")
print(f"    finite={fb}  all-inf={ib}  total={fb + ib}")
if isinstance(grid, dict) and grid.get("elev"):
    print(f"  terrain_grid: {grid['nx']}x{grid['ny']} cells of {grid['cell_size']:.0f}m")
if fb > 0:
    sample = {k: v for k, v in list(shadows.items())[:6]}
    print(f"  sample shadows entries: {sample}")

# 8. Plot.
print("\nGenerating plot...")
out_png = "/home/gmor/test_shadow_after_fix.png"
hc.utils.plot_building_shadows_analysis(br, BUILDING, output_path=out_png)
print(f"Wrote {out_png}")
