"""Building inference and analysis module for hypercadaster_ES.

This module provides advanced functionality for analyzing and inferring building
characteristics from Spanish cadastral data, including:

- Building parts processing and geometric analysis
- Floor footprint calculation and space inference
- Orientation analysis and street relationship detection
- CAT file parsing for detailed building information
- Building space aggregation and classification

Main functions:
    - process_building_parts(): Main orchestrator for building analysis pipeline
    - process_zone(): Detailed analysis for individual cadastral zones
    - parse_horizontal_division_buildings_CAT_files(): Parse CAT files for building data
    - aggregate_CAT_file_building_spaces(): Aggregate building space information

Note: This module contains computationally intensive functions that process
large amounts of geometric and cadastral data.
"""

# Core libraries
import os
import sys
import math
import multiprocessing
import re
import warnings
from datetime import date
from itertools import chain, zip_longest
import itertools
import random

# Data processing
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Geospatial libraries
import geopandas as gpd
from shapely.ops import linemerge, substring as _shp_substring
from shapely.geometry.polygon import orient
from shapely.geometry import (Polygon, LineString, LinearRing, MultiPolygon, MultiLineString,
                              GeometryCollection, Point)

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Internal modules
from hypercadaster_ES import utils
from hypercadaster_ES import downloaders

# Configure warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(
    "ignore",
    message=re.escape("FigureCanvasAgg is non-interactive, and thus cannot be shown")
)

# Pseudo-zone bookkeeping for neighbouring-municipality buildings. Such
# buildings are loaded only as shadow-casting context — they have no real
# cadastral zone in the municipality being analyzed and must NOT produce
# inferred result rows. We bin them spatially so chunking stays balanced.
NEARBY_ZONE_PREFIX = "_nb_"
NEARBY_GRID_SIZE_M = 200


def _split_segment_at_geom_boundaries(segment, geom_list, buffer=0.15, min_sub_len=0.05):
    """
    Split `segment` (a LineString) at points where any geometry in
    `geom_list` crosses in or out (buffered boundary intersection).
    Returns a list of sub-LineStrings of at least `min_sub_len` metres.
    When no split points are found the original segment is returned as a
    one-element list.
    """
    seg_len = segment.length
    if seg_len < 1e-6 or not geom_list:
        return [segment]
    params = set()
    for geom in geom_list:
        try:
            bnd = geom.buffer(buffer).boundary
            inter = segment.intersection(bnd)
        except Exception:
            continue
        if inter.is_empty:
            continue
        pts = list(inter.geoms) if hasattr(inter, 'geoms') else [inter]
        for pt in pts:
            try:
                t = segment.project(pt) / seg_len
            except Exception:
                continue
            margin = min_sub_len / seg_len
            if margin < t < 1.0 - margin:
                params.add(round(t, 6))
    if not params:
        return [segment]
    parts = []
    prev = 0.0
    for t in sorted(params):
        try:
            sub = _shp_substring(segment, prev, t, normalized=True)
            if sub.length >= min_sub_len:
                parts.append(sub)
        except Exception:
            pass
        prev = t
    try:
        sub = _shp_substring(segment, prev, 1.0, normalized=True)
        if sub.length >= min_sub_len:
            parts.append(sub)
    except Exception:
        pass
    return parts if parts else [segment]


def _outward_probe_point(segment, building_poly, dist=0.25):
    """
    Return a Point a short distance `dist` on the OUTWARD side of `segment`
    (the side facing away from `building_poly`'s interior), measured from the
    segment midpoint along its perpendicular.  Returns None when the outward
    side cannot be disambiguated.

    Used to decide whether a wall is truly adiabatic: only if a neighbour's
    footprint sits immediately behind the wall (outward side) is it shared.
    A footprint that merely runs near/parallel on the interior side must not
    flip the wall to adiabatic.
    """
    try:
        (x1, y1), (x2, y2) = segment.coords[0], segment.coords[-1]
    except Exception:
        return None
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    # Unit perpendicular.
    px, py = -dy / length, dx / length
    mid = segment.interpolate(0.5, normalized=True)
    cand_a = Point(mid.x + px * dist, mid.y + py * dist)
    cand_b = Point(mid.x - px * dist, mid.y - py * dist)
    try:
        inner = building_poly.buffer(0.01)
        a_in = inner.contains(cand_a)
        b_in = inner.contains(cand_b)
    except Exception:
        return None
    if a_in and not b_in:
        return cand_b
    if b_in and not a_in:
        return cand_a
    # Ambiguous (both inside a concave nook, or both outside) — caller falls
    # back to the looser coincidence test.
    return None


def _force_reset_code(results_dir, code, _db_mod):
    """Drop all central-DB rows + cache shards belonging to `code`.

    Used by `process_building_parts(force_update=True)`. Multi-municipality
    safe: other codes' rows in the central DB are untouched.
    """
    import glob as _glob_mod
    dbp = _db_mod.db_path(results_dir)
    cache_dir = f"{results_dir}/cache"

    deleted_shards = 0
    if os.path.isdir(cache_dir):
        # Building references begin with the 5-digit cadaster code, and
        # zone_reference = building_reference[0:5] + building_reference[7:14],
        # so every real shard for this code has a filename starting with
        # `{code}`. Pseudo-zones (NEARBY_ZONE_PREFIX) don't produce shards.
        for sp in _glob_mod.glob(f"{cache_dir}/{code}*.duckdb"):
            try:
                os.remove(sp); deleted_shards += 1
            except OSError:
                pass

    if os.path.exists(dbp):
        con = _db_mod.open_db(results_dir)
        try:
            # Also collect any zones in zone_status that don't match the
            # filename heuristic above, and remove their shard files too.
            try:
                extras = [r[0] for r in con.execute(
                    "SELECT zone_reference FROM zone_status WHERE cadaster_code = ?",
                    [code],
                ).fetchall()]
                for zr in extras:
                    sp = f"{cache_dir}/{zr}.duckdb"
                    if os.path.exists(sp):
                        try:
                            os.remove(sp); deleted_shards += 1
                        except OSError:
                            pass
            except Exception:
                pass
            for tbl in ("sbr_results", "br_results",
                        "aux_building_parts", "aux_footprints",
                        "_br_merged"):
                try:
                    con.execute(f'DROP TABLE IF EXISTS "main"."{tbl}"')
                except Exception:
                    pass
            try:
                con.execute(
                    "DELETE FROM zone_status WHERE cadaster_code = ?", [code]
                )
            except Exception:
                pass
        finally:
            con.close()
    sys.stderr.write(
        f"[force_update] cleared central DB rows and {deleted_shards} cache "
        f"shards for cadaster_code={code}\n"
    )


def _safe_nanmean(values):
    """np.nanmean that returns nan silently on empty / all-nan input."""
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


# Per-space CAT columns that leak into br_results in multi-building zones
# as all-NaN `_x` / `_y` pairs from an unintended merge. Strip them so the
# building-level output stays at the building grain.
_BR_PER_SPACE_LEAK_BASES = frozenset({
    "use_type", "floor_number", "building_space_floor_number",
    "building_space_reference", "building_space_reference_last_digits",
    "street_type", "street_name", "street_number1", "street_letter1",
    "street_number2", "street_letter2", "km",
    "building_space_block_name", "building_space_stair_name",
    "building_space_floor_name", "building_space_door_name",
    "building_space_year", "retrofitted", "building_space_retroffiting_year",
    "building_space_effective_year", "building_space_total_area",
    "building_space_area_without_communal",
    "building_space_area_balconies_terraces",
    "building_space_area_imputable_to_other_floors",
    "building_space_participation_rate", "building_space_use_type",
    "building_space_detailed_use_type", "building_space_typology",
    "building_space_total_area_without_communal", "building_spaces_considered",
    "building_space_communal_area", "building_space_area_with_communal",
    "building_space_typology_category", "building_space_typology_id",
    "building_space_age", "building_space_typology_category_booster",
    "building_space_typology_use", "building_space_typology_use_class",
    "building_space_typology_use_class_modality",
    "building_space_typology_use_level", "building_space_age_key",
    "construction_relative_economic_value",
    "age_correction_relative_economic_value",
    "building_space_relative_economic_value", "building_space_economic_value",
    "building_space_address", "building_space_inner_address",
    "building_space_floor_order_above_ground",
    "building_space_floor_order_below_ground", "building_space_floor_order",
    "building_floor_total_area", "building_total_area", "_typ_cat_int",
})


def _strip_per_space_leak_columns(df):
    if df is None or len(df.columns) == 0:
        return df
    to_drop = []
    for c in df.columns:
        base = c[:-2] if c.endswith("_x") or c.endswith("_y") else c
        if base in _BR_PER_SPACE_LEAK_BASES:
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


_BR_SPACES_LIST_COLS = (
    "br__building_spaces_reference",
    "br__building_spaces_postal_address",
    "br__building_spaces_inner_address",
    "br__building_spaces_floor_number",
    "br__building_spaces_category",
    "br__building_spaces_effective_year",
    "br__building_spaces_area_without_communal",
    "br__building_spaces_area_with_communal",
    "br__building_spaces_economic_value",
    "br__building_spaces_detailed_use_type",
)
_BR_SPACES_LIST_ADDITIVE = frozenset({
    "br__building_spaces_area_without_communal",
    "br__building_spaces_area_with_communal",
    "br__building_spaces_economic_value",
})


def _dedup_br_building_spaces_lists(df):
    """Collapse duplicate building-space entries inside the per-building
    br__building_spaces_* parallel lists.

    Dedup key is the triple (reference, postal_address, inner_address).
    Different (escalera, planta, puerta) tuples in CAT produce distinct
    inner_address values and are kept as separate entries, even when they
    share a building_space_reference. Only true duplicates (same triple)
    are folded: additive fields are summed, descriptive fields take the
    first occurrence.

    Each column is a {use_type: [values]} dict. Lists across columns are
    index-aligned per use_type.
    """
    if df is None or df.empty:
        return df
    list_cols = [c for c in _BR_SPACES_LIST_COLS if c in df.columns]
    if "br__building_spaces_reference" not in list_cols:
        return df

    has_postal = "br__building_spaces_postal_address" in list_cols
    has_inner = "br__building_spaces_inner_address" in list_cols

    def _fold_row(row):
        ref_dict = row["br__building_spaces_reference"]
        if not isinstance(ref_dict, dict):
            return row
        new_values = {c: {} for c in list_cols}
        for use_type, refs in ref_dict.items():
            if refs is None:
                for c in list_cols:
                    new_values[c][use_type] = refs
                continue
            refs_list = list(refs)
            per_col = {c: list(row[c].get(use_type, [])) if isinstance(row[c], dict) else []
                       for c in list_cols}
            postal_list = per_col.get("br__building_spaces_postal_address", []) if has_postal else []
            inner_list = per_col.get("br__building_spaces_inner_address", []) if has_inner else []

            def _at(lst, i):
                return lst[i] if i < len(lst) else None

            first_idx = {}
            order = []
            for i, r in enumerate(refs_list):
                key = (r, _at(postal_list, i), _at(inner_list, i))
                if key not in first_idx:
                    first_idx[key] = i
                    order.append(key)
                else:
                    j = first_idx[key]
                    for c in list_cols:
                        if c in _BR_SPACES_LIST_ADDITIVE:
                            try:
                                a = per_col[c][j]
                                b = per_col[c][i]
                                if a is None or (isinstance(a, float) and np.isnan(a)):
                                    a = 0.0
                                if b is None or (isinstance(b, float) and np.isnan(b)):
                                    b = 0.0
                                per_col[c][j] = float(a) + float(b)
                            except (IndexError, TypeError):
                                pass
            for c in list_cols:
                new_values[c][use_type] = [per_col[c][first_idx[k]] for k in order]
        for c in list_cols:
            row[c] = new_values[c]
        # Update the building space count to the deduped count.
        if "br__building_spaces" in row.index:
            bs = row["br__building_spaces"]
            if isinstance(bs, dict):
                row["br__building_spaces"] = {
                    use_type: len(new_values["br__building_spaces_reference"].get(use_type, []))
                    for use_type in bs
                }
        return row

    return df.apply(_fold_row, axis=1)


# Module-level state for the per-zone parallel worker. We run the pool under
# `fork`, so children inherit this dict via copy-on-write — no pickling, and
# the large shared geodata (gdf_footprints_global, parcels_gdf, streets_gdf,
# zone groupby, CAT lookup) stays a single physical copy across all workers.
# This is what keeps memory roughly flat in the number of workers; spawn/
# forkserver duplicated the state into each child and caused linear blow-up.
_ZONE_WORKER_STATE = {}

_ZONE_WORKER_INITIALIZED = False


def _init_zone_worker():
    """Per-worker one-time setup. Called lazily on the first task.

    - Limits BLAS/OpenMP threads to 1: fork-inherited thread pools in
      numpy/scipy/pandas/scikit-learn cause silent deadlocks. With many
      workers each running CPU-light geometric work, single-threaded BLAS
      is also a perf win (avoids 32×N thread contention).
    - Clears the DEM sampler cache so each worker opens its own DEM
      handle rather than reusing a parent-inherited (broken) one.

    Note: polars is intentionally NOT used from inside workers (all hot-path
    DataFrames are pandas) so the Rayon thread pool is never touched after
    fork, which is the original reason fork was unsafe.
    """
    global _ZONE_WORKER_INITIALIZED
    if _ZONE_WORKER_INITIALIZED:
        return
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    try:
        utils._DEM_SAMPLER_CACHE.clear()
    except Exception:
        pass
    _ZONE_WORKER_INITIALIZED = True


def _process_zone_worker(zone_reference):
    """Run process_zone for one cadastral zone.

    Only the zone key is passed across the fork boundary; the per-zone
    GeoDataFrame slice and CAT slice are looked up from the inherited
    `_ZONE_WORKER_STATE` so nothing big is pickled per task.

    We do NOT return the per-zone DataFrames back to the parent. process_zone
    already caches them to `{results_dir}/cache/{zone_reference}_{sbr,br}_results.pkl`
    on success, so the worker just reports an outcome tag and lets the parent
    re-read those files in batches at the end. This keeps the parent's heap
    flat across thousands of zones (the old code accumulated every zone's
    DataFrames in a Python list until the final concat, which was the OOM).
    """
    _init_zone_worker()
    s = _ZONE_WORKER_STATE
    try:
        gdf_zone = s["grouped"].get_group(zone_reference).reset_index(drop=True)
        cat_slice = s["zone_cat_lookup"].get(zone_reference)
        zone_plots = s["plots"] and (zone_reference in s["zones_to_plot"])
        sbr, br = process_zone(
            gdf_zone,
            zone_reference,
            s["gdf_footprints_global"],
            s["parcels_gdf"],
            s["streets_gdf"],
            cat_slice,
            s["results_dir"],
            zone_plots,
            s["orientation_discrete_interval_in_degrees"],
            s["floor_height"],
            s["ground_floor_height"],
            s["DEM_raster_dir"],
        )
        if sbr is None or br is None:
            return ("__EMPTY__", zone_reference)
        return ("__OK__", zone_reference)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return ("__ERR__", zone_reference, str(e), "\n".join(tb.splitlines()[-8:]))


def _spatially_bin_nearby_zones(gdf, grid_size_m=NEARBY_GRID_SIZE_M,
                                zone_prefix=NEARBY_ZONE_PREFIX):
    """Replace zone_reference on neighbouring/unknown rows with spatial-grid pseudo-zones.

    A row is treated as 'neighbour/unknown' if its `nearby_buildings` flag is
    True or its `zone_reference` is missing / "unknown" / "unassigned". Those
    rows get a pseudo-zone id of the form `f"{zone_prefix}{ix}_{iy}"` based on
    their centroid's `grid_size_m`-snapped position. Real cadastral zones are
    left untouched.
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    nearby_flag = gdf["nearby_buildings"].fillna(False).astype(bool) \
        if "nearby_buildings" in gdf.columns else pd.Series(False, index=gdf.index)
    zone_missing = gdf["zone_reference"].isna() | gdf["zone_reference"].isin(
        ["unknown", "unassigned"]
    ) if "zone_reference" in gdf.columns else pd.Series(True, index=gdf.index)

    mask = nearby_flag | zone_missing
    if not mask.any():
        return gdf

    geom = gdf["building_part_geometry"] if "building_part_geometry" in gdf.columns else gdf.geometry
    centroids = geom[mask].centroid
    ix = np.floor(centroids.x.values / grid_size_m).astype(int)
    iy = np.floor(centroids.y.values / grid_size_m).astype(int)
    pseudo = [f"{zone_prefix}{a}_{b}" for a, b in zip(ix, iy)]

    if "zone_reference" not in gdf.columns:
        gdf["zone_reference"] = "unknown"
    gdf.loc[mask, "zone_reference"] = pseudo

    n_pseudo = len(set(pseudo))
    sys.stderr.write(
        f"_spatially_bin_nearby_zones: assigned {int(mask.sum())} neighbour/unknown "
        f"building parts to {n_pseudo} spatial grid pseudo-zones "
        f"(prefix='{zone_prefix}', cell={grid_size_m}m)\n"
    )
    return gdf


# ==========================================
# MAIN BUILDING ANALYSIS PIPELINE
# ==========================================

def process_building_parts(code, building_part_gdf_, buildings_CAT, parcels_gdf, results_dir=None, cadaster_dir=None,
                           open_street_dir=None, plots=False, plot_zones_ratio=0.01,
                           orientation_discrete_interval_in_degrees=5, floor_height=3.0,
                           ground_floor_height=5.0, DEM_raster_dir=None,
                           num_workers=-1, nearby_buildings_buffer=80,
                           force_update=False, cleanup_shards=True):
    """
    Main function to process building parts and perform comprehensive building analysis.
    
    This function orchestrates the entire building inference pipeline including:
    - Building proximity detection and clustering
    - Floor footprint calculation  
    - Orientation analysis and street relationship detection
    - Building space classification and aggregation
    - Shadow analysis and environmental factors
    
    Parameters:
    -----------
    code : str
        Cadastral municipality code
    building_part_gdf_ : geopandas.GeoDataFrame
        Building parts geodataframe with geometric information
    buildings_CAT : polars.DataFrame or None
        CAT file data with detailed building information
    parcels_gdf : geopandas.GeoDataFrame
        Cadastral parcels geodataframe
    results_dir : str, optional
        Directory to save analysis results and cache files
    cadaster_dir : str, optional  
        Directory containing cadastral data files
    open_street_dir : str, optional
        Directory containing OpenStreetMap data
    plots : bool, default False
        Whether to generate visualization plots
    plot_zones_ratio : float, default 0.01
        Ratio of zones to randomly select for plotting when plots=True (0.01 = 1%, minimum 1 zone)
    orientation_discrete_interval_in_degrees : int, default 5
        Degree interval for orientation discretization
    num_workers : int, default calculated
        Number of parallel workers for processing
        
    Returns:
    --------
    ResultsHandle
        Lightweight handle pointing to the central DuckDB with sbr_results
        and br_results tables for the given cadaster_code. Use
        ``handle.br_to_pandas()`` or ``handle.sbr_to_pandas()`` to
        materialise, or pass the handle directly to the merger which
        consumes it via SQL.
    """

    from hypercadaster_ES import _db as _db_mod_early

    # `force_update=True`: wipe everything for THIS code so the run is a
    # clean recompute. Other municipalities in the same central DB are
    # untouched (all our tables are keyed by cadaster_code).
    if force_update:
        _force_reset_code(results_dir, code, _db_mod_early)

    # The pipeline below always runs. Each stage short-circuits internally
    # where it can: aux loads from DB if present, workers skip zones
    # already in zone_status, Phase A is a no-op when zone_status is
    # complete, Phase B reads from central. A fully-cached re-run only
    # pays aux-load + zone-grouping + Phase B read. The old "if rows
    # exist for this code, return them as-is" early-return is gone
    # because it prevented multi-municipality runs from filling in
    # newly-added zones.

    if True:

        os.makedirs(f"{results_dir}/cache", exist_ok=True)
        if plots:
            if results_dir is None:
                results_dir = "results"
            os.makedirs(f"{results_dir}/plots", exist_ok=True)

        # Aux frames are scratch state for one municipality at a time.
        # Stored in the central DuckDB, keyed by cadaster_code so the same
        # results_dir can serve multiple municipalities. Replaces the
        # legacy {code}_aux{1,2}.pkl files.
        from hypercadaster_ES import _db as _db_mod_aux
        _aux_con = _db_mod_aux.open_db(results_dir)
        try:
            def _aux_present(tbl):
                try:
                    return _aux_con.execute(
                        f'SELECT 1 FROM "main"."{tbl}" '
                        f'WHERE cadaster_code = ? LIMIT 1', [code],
                    ).fetchone() is not None
                except Exception:
                    return False

            def _aux_write(tbl, frame):
                df_w = frame.copy(deep=False)
                if "cadaster_code" in df_w.columns:
                    # Upstream (mergers.py) may already attach this column
                    # to the gdf. Overwrite in place so the value is
                    # authoritative and there's no duplicate column.
                    df_w["cadaster_code"] = code
                else:
                    df_w.insert(0, "cadaster_code", code)
                crs = getattr(frame, "crs", None)
                try:
                    _aux_con.execute(f'DESCRIBE "main"."{tbl}"').fetchall()
                    _aux_con.execute(
                        f'DELETE FROM "main"."{tbl}" WHERE cadaster_code = ?',
                        [code],
                    )
                    try:
                        _db_mod_aux.write_df_safe(_aux_con, tbl, df_w,
                                                   if_exists="append", crs=crs)
                    except Exception:
                        # Schema drift vs. an earlier write: nuke + recreate.
                        _aux_con.execute(f'DROP TABLE "main"."{tbl}"')
                        _db_mod_aux.write_df_safe(_aux_con, tbl, df_w,
                                                   if_exists="replace", crs=crs)
                except Exception:
                    _db_mod_aux.write_df_safe(_aux_con, tbl, df_w,
                                               if_exists="replace", crs=crs)

            def _aux_read(tbl, geom_col):
                df_r = _db_mod_aux.read_df_safe(
                    _aux_con, tbl,
                    where="cadaster_code = ?", params=[code],
                ).drop(columns=["cadaster_code"], errors="ignore")
                if geom_col in df_r.columns:
                    crs = _db_mod_aux.get_table_crs(
                        _aux_con, f"{tbl}.{geom_col}"
                    )
                    df_r = gpd.GeoDataFrame(df_r, geometry=geom_col, crs=crs)
                return df_r

            if not _aux_present("aux_building_parts"):
                # Neighbouring-municipality buildings (added for shadow-casting
                # context) have no valid zone_reference in the analyzed
                # municipality, so they all collapse into one giant chunk under
                # "unknown"/"unassigned". Tag those rows with a deterministic
                # 200 m spatial grid pseudo-zone so detect_close_buildings_parallel
                # produces many small bounded chunks.
                building_part_gdf_ = _spatially_bin_nearby_zones(
                    building_part_gdf_, grid_size_m=NEARBY_GRID_SIZE_M,
                    zone_prefix=NEARBY_ZONE_PREFIX,
                )

                gdf_global = utils.detect_close_buildings_parallel(gdf_building_parts=building_part_gdf_,
                                                             buffer_neighbours=nearby_buildings_buffer,
                                                             neighbours_column_name="nearby_buildings",
                                                             neighbours_id_column_name="building_reference",
                                                             num_workers=num_workers,
                                                             column_name_to_split="zone_reference")
                parcels_gdf["zone_reference"] = (
                    parcels_gdf["building_reference"].str[0:5] + parcels_gdf["building_reference"].str[7:14])
                gdf_parcels_global = utils.detect_close_parcels_parallel(gdf_parcels=parcels_gdf,
                                                                 buffer_neighbours=nearby_buildings_buffer*2,
                                                                 neighbours_column_name="nearby_parcels",
                                                                 neighbours_id_column_name="building_reference",
                                                                 num_workers=num_workers,
                                                                 column_name_to_split="zone_reference")
                gdf_global = gdf_global.merge(
                    gdf_parcels_global[["building_reference","nearby_parcels"]],
                    on="building_reference", how="left")
                gdf_global = gdf_global.set_geometry("building_part_geometry")
                _aux_write("aux_building_parts", gdf_global)
            else:
                gdf_global = _aux_read("aux_building_parts", "building_part_geometry")

            if not _aux_present("aux_footprints"):
                gdf_footprints_global = utils.calculate_floor_footprints(
                    gdf=gdf_global,
                    group_by="building_reference",
                    geometry_name="building_part_geometry",
                    only_exterior_geometry=True,
                    min_hole_area=0.5,
                    gap_tolerance=0.05,
                    chunk_size=500,
                    num_workers=num_workers)
                _aux_write("aux_footprints", gdf_footprints_global)
            else:
                gdf_footprints_global = _aux_read("aux_footprints", "geometry")
        finally:
            _aux_con.close()

        # Carry the per-building ground elevation into the global footprints
        # frame. The shadow gate in `process_zone` needs each neighbour's
        # roof elevation to compare against the current floor's mid-floor
        # eye level; without elevation here, every neighbour defaults to
        # ground=0 m, sits ~tens of metres below the analysed building, and
        # every potential shadow caster gets rejected by the `roof > eye`
        # gate (so `shadows_at_distance` is all-inf for every building).
        if "elevation" in gdf_global.columns and "elevation" not in gdf_footprints_global.columns:
            elev_by_ref = (
                gdf_global.groupby("building_reference")["elevation"].mean()
            )
            gdf_footprints_global = gdf_footprints_global.merge(
                elev_by_ref.rename("elevation"),
                left_on="group", right_index=True, how="left",
            )

        grouped = gdf_global.groupby("zone_reference")
        # Exclude spatial-grid pseudo-zones from result emission: those rows are
        # neighbouring-municipality buildings present only as shadow-casting
        # context. They are still available to process_zone via gdf_footprints_global.
        # Sort biggest-first so OOM (if it happens) hits early instead of after hours,
        # and so the long-tail of giant zones gets distributed across workers early
        # rather than stacking up at the end of the queue.
        _zone_sizes = grouped.size()
        zone_references = [z for z in _zone_sizes.sort_values(ascending=False).index
                           if not str(z).startswith(NEARBY_ZONE_PREFIX)]
        n_skipped_pseudo = len(grouped.groups) - len(zone_references)
        if n_skipped_pseudo:
            sys.stderr.write(
                f"Skipping {n_skipped_pseudo} neighbour-pseudo-zones from result emission; "
                f"{len(zone_references)} real zones will be processed\n"
            )

        # Skip zones whose results are already in the central DB. This
        # avoids re-computing zones that were aggregated and then had
        # their cache shards cleaned up (cleanup_shards=True). On resume,
        # we trust the central DB as the source of truth; only zones
        # missing from `zone_status` need fresh inference.
        try:
            from hypercadaster_ES import _db as _db_mod_pre
            _pre_con = _db_mod_pre.open_db(results_dir)
            try:
                already_in_central = _db_mod_pre.completed_zones(_pre_con, code)
            finally:
                _pre_con.close()
        except Exception:
            already_in_central = set()
        if already_in_central:
            before = len(zone_references)
            zone_references = [z for z in zone_references
                               if z not in already_in_central]
            skipped = before - len(zone_references)
            if skipped:
                sys.stderr.write(
                    f"Skipping {skipped} zones already aggregated in central DB; "
                    f"{len(zone_references)} zones will be (re)inferred\n"
                )

        # Random zone selection for plots when building_parts_plots is True
        if plots:
            total_zones = len(zone_references)
            # Calculate number of zones to plot: at least 1, or the specified ratio
            num_zones_to_plot = max(1, int(total_zones * plot_zones_ratio))
            if num_zones_to_plot < total_zones:
                # Randomly select zones for plotting
                zones_to_plot = set(random.sample(zone_references, num_zones_to_plot))
                print(f"Randomly selected {num_zones_to_plot} out of {total_zones} zones for plotting ({plot_zones_ratio*100:.1f}%)")
            else:
                # Plot all zones if ratio results in all zones
                zones_to_plot = set(zone_references)
                print(f"Plotting all {total_zones} zones")
        else:
            zones_to_plot = set()

        streets_gdf = utils.get_streets_open_street_maps(
            open_street_dir=open_street_dir,
            cadaster_code=code,
            crs=gdf_global.crs)
        streets_gdf = streets_gdf[streets_gdf.geometry.apply(lambda geom: isinstance(geom, LineString))]

        # All shared state lives in this module-level dict. Workers fork()ed
        # from this process inherit it via copy-on-write — no pickling, no
        # per-worker duplication.
        global _ZONE_WORKER_STATE
        _ZONE_WORKER_STATE = {
            "gdf_footprints_global": gdf_footprints_global,
            "parcels_gdf": parcels_gdf,
            "streets_gdf": streets_gdf,
            "grouped": grouped,
            "results_dir": results_dir,
            "plots": plots,
            "zones_to_plot": zones_to_plot,
            "orientation_discrete_interval_in_degrees": orientation_discrete_interval_in_degrees,
            "floor_height": floor_height,
            "ground_floor_height": ground_floor_height,
            "DEM_raster_dir": DEM_raster_dir,
        }

        # Build a per-zone CAT lookup once. For each zone, we keep only the
        # CAT rows whose building_reference appears in that zone. With Spanish
        # cadastre conventions every building belongs to exactly one zone, so
        # the slices are disjoint and small (typically <1 KB each).
        # Slices are materialised as pandas DataFrames so the parallel workers
        # never touch polars — that lets us run under `fork` without tripping
        # polars' Rayon thread pool, which is the main reason memory blew up
        # under spawn/forkserver (state had to be pickled into every worker).
        def _build_zone_cat_lookup():
            if buildings_CAT is None:
                return {zr: None for zr in zone_references}
            empty_pd = buildings_CAT.head(0).to_pandas()
            refs_in_zone = (gdf_global[gdf_global["zone_reference"].isin(zone_references)]
                            .groupby("zone_reference")["building_reference"]
                            .unique())
            lookup = {}
            for zr in zone_references:
                refs = refs_in_zone.get(zr, [])
                if len(refs) == 0:
                    lookup[zr] = empty_pd
                else:
                    lookup[zr] = buildings_CAT.filter(
                        pl.col("building_reference").is_in(list(refs))
                    ).to_pandas()
            return lookup

        zone_cat_lookup = _build_zone_cat_lookup()
        _ZONE_WORKER_STATE["zone_cat_lookup"] = zone_cat_lookup

        # Resolve worker count: -1 → all CPUs, 0/None → 1, else as given.
        if num_workers is None or num_workers == 0:
            effective_workers = 1
        elif num_workers < 0:
            effective_workers = multiprocessing.cpu_count()
        else:
            effective_workers = num_workers
        effective_workers = min(effective_workers, len(zone_references)) or 1

        # Pre-warm the DuckDB spatial extension in the parent BEFORE forking
        # workers. `INSTALL spatial` is a one-time download per machine; if
        # workers race on the first install we can get spurious failures.
        # After this call the extension is cached in ~/.duckdb/extensions/
        # and worker `LOAD spatial` calls are local-only.
        from hypercadaster_ES import _db as _db_mod
        _db_mod.open_db(results_dir).close()

        progress_bar = tqdm(total=len(zone_references), desc="Inferring building KPIs",
                            mininterval=1.0, leave=True, ncols=80)

        # Workers only report an outcome tag ("__OK__" / "__EMPTY__" / "__ERR__"),
        # never the DataFrames themselves — those are read back from each zone's
        # cache pickle after the pool finishes. This keeps the parent's heap
        # flat in the number of zones.
        ok_zone_refs = []

        def _record(r):
            tag = r[0] if isinstance(r, tuple) and r else None
            if tag == "__OK__":
                ok_zone_refs.append(r[1])
            elif tag == "__EMPTY__":
                pass  # zone had nothing to emit; no cache files written
            elif tag == "__ERR__":
                _, zr, msg, tail = r
                progress_bar.write(f"This cadastral zone failed: '{zr}'. Error: {msg}\n{tail}")
            progress_bar.update(1)

        if effective_workers == 1:
            # Sequential path (preserves original behavior when num_workers=1)
            for zr in zone_references:
                _record(_process_zone_worker(zr))
        else:
            # Parallel path under `fork`. Workers inherit `_ZONE_WORKER_STATE`
            # via copy-on-write so RAM stays roughly flat in the number of
            # workers (vs. linear blow-up with spawn/forkserver, which had to
            # pickle the full state into each child). Safe here because the
            # hot path is polars-free — all in-worker DataFrames are pandas,
            # so polars' Rayon thread pool is never touched post-fork.
            #
            # maxtasksperchild recycles workers periodically so per-process
            # allocator fragmentation (numpy/shapely/pandas) doesn't accumulate
            # over thousands of zones. 50 is a reasonable balance: enough that
            # fork overhead is amortised, low enough that RSS stays bounded.
            ctx = multiprocessing.get_context("fork")
            pool_kwargs = {
                "processes": effective_workers,
                "initializer": _init_zone_worker,
                "maxtasksperchild": 50,
            }
            with ctx.Pool(**pool_kwargs) as pool:
                for r in pool.imap_unordered(
                    _process_zone_worker, zone_references, chunksize=1
                ):
                    _record(r)

        progress_bar.close()
        # Release the large shared state before the disk-aggregation pass —
        # we no longer need grouped, the CAT lookup, or the global GDFs and
        # they together can be many GB on Barcelona-sized inputs.
        _ZONE_WORKER_STATE = {}
        del grouped, zone_cat_lookup, streets_gdf, gdf_global, gdf_footprints_global
        import gc as _gc
        _gc.collect()

        cache_dir = f"{results_dir}/cache"

        # ----------------------------------------------------------------
        # Phase A: drain per-zone shards into the central DB.
        # ATTACH/INSERT/DETACH per shard keeps the catalog small and avoids
        # holding all shard file handles open simultaneously. The central
        # DB ends up with two tables, sbr_results and br_results, with two
        # bookkeeping columns added (cadaster_code, zone_reference) on top
        # of whatever schema the shards carry.
        # ----------------------------------------------------------------
        from hypercadaster_ES import _db as _db_mod
        central_con = _db_mod.open_db(results_dir)
        # NOTE: we don't close central_con here — ownership is passed to
        # ResultsHandle which hands it to mergers.py.
        # Clean up any orphan rows left by a previous run that died
        # between an INSERT and the mark_zone for the same zone (no
        # transaction boundary covered that gap in earlier versions).
        # We use zone_status as the source of truth: any rows in the
        # result tables whose (cadaster_code, zone_reference) isn't
        # marked 'ok' are partial state and must be discarded so the
        # re-ingest of that zone doesn't duplicate rows.
        for _tbl in ("sbr_results", "br_results"):
            try:
                central_con.execute(
                    f'DELETE FROM "{_tbl}" '
                    f'WHERE cadaster_code = ? '
                    f'AND zone_reference NOT IN ('
                    f'    SELECT zone_reference FROM zone_status '
                    f'    WHERE cadaster_code = ? AND state = \'ok\''
                    f')',
                    [code, code],
                )
            except Exception:
                pass  # table may not exist yet on the very first run

        # Phase A short-circuit. If a previous run already aggregated
        # every expected zone (zone_status marks them all 'ok' and the
        # tables exist), skip straight to Phase B. Avoids redoing the
        # ~33 min ATTACH loop after a Phase B crash on Barcelona.
        def _phase_a_can_skip():
            try:
                have_tables = {
                    r[0] for r in central_con.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'main'"
                    ).fetchall()
                }
                if not {"sbr_results", "br_results"} <= have_tables:
                    return False
                done = _db_mod.completed_zones(central_con, code)
                return set(ok_zone_refs).issubset(done)
            except Exception:
                return False

        if _phase_a_can_skip():
            sys.stderr.write(
                f"Phase A: all {len(ok_zone_refs)} zones already aggregated "
                f"into central DB; skipping ATTACH loop\n"
            )
            n_aggregated = len(ok_zone_refs)
        else:
            # Incremental: skip zones already marked 'ok', and lazily
            # extend the central schema as new shards introduce new
            # CAT-derived columns. Per-zone schemas vary in this
            # codebase (process_zone's merges with CAT data produce
            # different column sets per zone), so the central table
            # is the union of every shard's columns rather than any
            # single shard's schema.
            done_zones = _db_mod.completed_zones(central_con, code)
            already_done = sum(1 for zr in ok_zone_refs if zr in done_zones)
            if already_done:
                sys.stderr.write(
                    f"Phase A: resuming with {already_done}/{len(ok_zone_refs)} "
                    f"zones already aggregated\n"
                )

            agg_bar = tqdm(total=len(ok_zone_refs), desc="Aggregating zone shards",
                           mininterval=1.0, leave=True, ncols=80)
            agg_bar.update(already_done)
            n_aggregated = already_done

            for zr in ok_zone_refs:
                if zr in done_zones:
                    continue
                sp = f"{cache_dir}/{zr}.duckdb"
                if not os.path.exists(sp):
                    agg_bar.write(f"Missing shard for zone '{zr}'; skipping")
                    agg_bar.update(1)
                    continue
                try:
                    central_con.execute(f"ATTACH '{sp}' AS shard (READ_ONLY)")
                    try:
                        # Transactional: either both result tables get
                        # this zone's rows AND mark_zone fires, or none
                        # does. No half-state for the next run to trip on.
                        central_con.execute("BEGIN TRANSACTION")
                        try:
                            bk = {"cadaster_code": code, "zone_reference": zr}
                            _db_mod.ingest_shard_table(
                                central_con,
                                shard_alias="shard",
                                shard_table="sbr_results",
                                central_table="sbr_results",
                                bookkeeping=bk,
                            )
                            _db_mod.ingest_shard_table(
                                central_con,
                                shard_alias="shard",
                                shard_table="br_results",
                                central_table="br_results",
                                bookkeeping=bk,
                            )
                            _db_mod.mark_zone(central_con, code, zr, "ok")
                            central_con.execute("COMMIT")
                            n_aggregated += 1
                        except Exception:
                            central_con.execute("ROLLBACK")
                            raise
                    finally:
                        central_con.execute("DETACH shard")
                except Exception as _agg_err:
                    agg_bar.write(
                        f"Shard aggregation failed for '{zr}' "
                        f"({type(_agg_err).__name__}: {_agg_err}); skipping"
                    )
                agg_bar.update(1)
                # Periodic CHECKPOINT flushes DuckDB's WAL and releases
                # transient memory so the parent's heap stays flat
                # across thousands of shards instead of growing
                # linearly until OOM.
                if (n_aggregated and n_aggregated % 200 == 0):
                    _db_mod.maybe_checkpoint(central_con)
            agg_bar.close()
            _db_mod.maybe_checkpoint(central_con)

        # Post-Phase-A cleanup: drop cache shards whose data is now
        # safely in the central DB. The worker-skip filter (above)
        # ensures future runs don't re-compute these zones. Failure
        # of an individual unlink is logged but not fatal.
        if cleanup_shards:
            try:
                has_central_data = central_con.execute(
                    'SELECT 1 FROM "main"."sbr_results" '
                    'WHERE cadaster_code = ? LIMIT 1', [code],
                ).fetchone() is not None
            except Exception:
                has_central_data = False

            if has_central_data:
                done_now = _db_mod.completed_zones(central_con, code)
                deleted = 0
                for zr in done_now:
                    sp = f"{cache_dir}/{zr}.duckdb"
                    if os.path.exists(sp):
                        try:
                            os.remove(sp); deleted += 1
                        except OSError as _e:
                            sys.stderr.write(
                                f"[cleanup_shards] could not remove {sp}: {_e}\n"
                            )
                if deleted:
                    sys.stderr.write(
                        f"[cleanup_shards] removed {deleted} aggregated shards "
                        f"from cache for cadaster_code={code}\n"
                    )
        # Do NOT close central_con here — pass ownership to ResultsHandle.
        # The caller (mergers.py) will close it after consuming the handle.

    # Phase B materialisation is gone — the central DB IS the artifact.
    # Return a lightweight handle; the caller (mergers.py) consumes it
    # via SQL instead of materialising full DataFrames. Pass the open
    # central_con so the handle reuses it and avoids the "different
    # configuration" DuckDB error.
    from hypercadaster_ES import _db as _db_mod_handle
    return _db_mod_handle.ResultsHandle(
        _db_mod_handle.db_path(results_dir), code,
        parent_con=central_con,
    )


# ==========================================
# ZONE-LEVEL ANALYSIS FUNCTIONS  
# ==========================================

def process_zone(gdf_zone, zone_reference, gdf_footprints_global, parcels_gdf, streets_gdf, buildings_CAT,
                 results_dir, plots, orientation_discrete_interval_in_degrees, floor_height,
                 ground_floor_height=5.0, DEM_raster_dir=None):
    """
    Perform detailed analysis for an individual cadastral zone.
    
    This function processes all buildings within a specific cadastral zone,
    analyzing their geometric properties, orientations, relationships with
    streets, and shadow patterns.
    
    Parameters:
    -----------
    gdf_zone : geopandas.GeoDataFrame
        Buildings within the specific zone to analyze
    zone_reference : str
        Unique identifier for the cadastral zone
    gdf_footprints_global : geopandas.GeoDataFrame
        Global floor footprints for shadow analysis
    parcels_gdf : geopandas.GeoDataFrame
        Cadastral parcels within the zone
    streets_gdf : geopandas.GeoDataFrame
        Street network data for orientation analysis
    buildings_CAT : polars.DataFrame or None
        CAT file building data
    results_dir : str
        Directory for saving cache and plot outputs
    plots : bool
        Whether to generate analysis plots
    orientation_discrete_interval_in_degrees : int
        Orientation discretization interval
        
    Returns:
    --------
    dict
        Comprehensive analysis results for the zone including building
        metrics, orientations, and environmental factors
    """

    try:
        shard_path = f"{results_dir}/cache/{zone_reference}.duckdb"

        # A previous run may have been killed (OOM) mid-write, leaving a
        # zero-byte / truncated shard on disk. Treat any unreadable shard as
        # "missing" and recompute, so we don't propagate corruption up the
        # stack. (Atomic .tmp → rename below makes new writes either
        # complete or absent.)
        def _cache_is_usable(path):
            if not os.path.exists(path):
                return False
            try:
                if os.path.getsize(path) == 0:
                    return False
            except OSError:
                return False
            return True

        cache_present = _cache_is_usable(shard_path)

        if not cache_present:

            # PDF setup for the zone if pdf_plots is True
            if plots:
                pdf = PdfPages(f"{results_dir}/plots/zone_{zone_reference}.pdf")

            # Calculation of the single building references and adjacent buildings
            gdf_zone = utils.detect_close_buildings(gdf_building_parts = gdf_zone,
                                   buffer_neighbours = 0.5,
                                   neighbours_column_name = "adjacent_buildings",
                                   neighbours_id_column_name = "single_building_reference",
                                   geometry_variable="building_part_geometry")

            # Calculation of the single building references and adjacent buildings
            gdf_zone = utils.detect_close_buildings(gdf_building_parts=gdf_zone,
                                              buffer_neighbours=1,
                                              neighbours_column_name="adjacent_parcels",
                                              neighbours_id_column_name="single_building_reference",
                                              geometry_variable="parcel_geometry")

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries(gdf_zone.building_part_geometry,
                                        labels=gdf_zone.n_floors_above_ground,
                                        clusters=gdf_zone.single_building_reference,
                                        ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            gdf_aux_footprints = utils.calculate_floor_footprints(
                gdf=gdf_zone,
                group_by="single_building_reference",
                geometry_name="building_part_geometry",
                min_hole_area=0.5,
                gap_tolerance=0.05,
                num_workers=1)

            gdf_aux_footprints_global = utils.calculate_floor_footprints(
                gdf=gdf_zone,
                geometry_name="building_part_geometry",
                min_hole_area=0.5,
                gap_tolerance=0.05,
                num_workers=1)

            # If no building in the zone has any positive floor count (e.g. a
            # ruins-only zone with n_floors_above_ground==0 everywhere),
            # there's nothing to analyse — skip cleanly instead of crashing
            # in the per-floor loop below.
            if gdf_aux_footprints.empty:
                if plots:
                    pdf.close()
                return None, None

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries(gdf_aux_footprints.geometry, clusters=gdf_aux_footprints.group, ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            # Detect all the patios in a zone
            patios_detected = {}
            for floor in range(gdf_aux_footprints.floor.max() + 1):
                patios_detected[floor] = (
                    utils.get_all_patios(gdf_aux_footprints_global[gdf_aux_footprints_global['floor']==floor].geometry))

            unique_patios_detected = utils.unique_polygons(list(itertools.chain(*patios_detected.values())),tolerance=0.1)

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries([i.exterior for i in gdf_zone.building_part_geometry] + unique_patios_detected, ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            # Close the PDF file for the current zone
            if plots:
                pdf.close()

            # DEM sampler for terrain-aware shadow analysis (shared across buildings in the zone)
            dem_sampler = None
            if DEM_raster_dir is not None:
                try:
                    dem_sampler = utils.make_dem_sampler(DEM_raster_dir, gdf_zone.crs)
                except Exception:
                    dem_sampler = None

            results_ = []

            for single_building_reference, building_gdf_item in gdf_zone.groupby('single_building_reference'):
                # grouped2 = gdf_zone.groupby("single_building_reference")
                # single_building_reference = list(grouped2.groups.keys())[0]
                # building_gdf_item = grouped2.get_group(single_building_reference).reset_index().drop(['index'], axis=1).copy()

                # PDF setup for the building if plots is True
                if plots:
                    pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}.pdf")

                # Obtain a general geometry of the building, considering all floors.
                building_geom = utils.union_geoseries_with_tolerance(building_gdf_item['building_part_geometry'],
                                                               gap_tolerance=0.05, resolution=16)
                parcel_geom = utils.union_geoseries_with_tolerance(building_gdf_item["parcel_geometry"],
                                                             gap_tolerance=0.5, resolution=0.5)

                # Some upstream rows carry NaN (float) instead of a comma-joined
                # string for these neighbour columns. Coerce non-strings to ""
                # so the .split(",") never sees a float.
                def _split_neighbour_column(series):
                    out = set()
                    for ids in series:
                        if not isinstance(ids, str) or not ids:
                            continue
                        for token in ids.split(","):
                            token = token.strip()
                            if token:
                                out.add(token)
                    return out

                # Extract and clean the set of neighbouring buildings
                adjacent_buildings_set = _split_neighbour_column(building_gdf_item['adjacent_buildings'])
                adjacent_buildings_set.discard(
                    single_building_reference)  # Safely remove the single_building_reference itself if present
                adjacent_buildings = sorted(adjacent_buildings_set)

                # Extract and clean the set of nearby buildings
                nearby_buildings_set = _split_neighbour_column(building_gdf_item['nearby_buildings'])
                nearby_buildings_set.discard(
                    single_building_reference.split("_")[0])  # Safely remove the single_building_reference itself
                nearby_buildings = sorted(nearby_buildings_set)

                # Extract and clean the set of nearby parcels
                nearby_parcels_set = _split_neighbour_column(building_gdf_item['nearby_parcels'])
                nearby_parcels_set.discard(
                    single_building_reference.split("_")[0]) # Safely remove the single_building_reference itself
                nearby_parcels = sorted(nearby_parcels_set)

                # Extract and clean the set of adjacent parcels
                adjacent_parcels_set = _split_neighbour_column(building_gdf_item['adjacent_parcels'])
                adjacent_parcels_set.discard(
                    single_building_reference)  # Safely remove the single_building_reference itself if present
                adjacent_parcels = sorted(adjacent_parcels_set)

                # Is there any premises in ground floor?
                premises = False
                premises_activity_typologies = []
                premises_names = []
                premises_last_revision = []
                if "number_of_ground_premises" in building_gdf_item.columns:
                    n_ground_premises = utils._coerce_scalar_float(
                        building_gdf_item.iloc[0]["number_of_ground_premises"], default=float("nan")
                    )
                    if np.isfinite(n_ground_premises) and n_ground_premises > 0:
                        premises = True
                        premises_activity_typologies = building_gdf_item.iloc[0]["ground_premises_activities"]
                        premises_names = building_gdf_item.iloc[0]["ground_premises_names"]
                        premises_last_revision = building_gdf_item.iloc[0]["ground_premises_last_revision"]

                # Filter building in study
                gdf_aux_footprint_building = gdf_aux_footprints[gdf_aux_footprints['group'] == single_building_reference]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(list(gdf_aux_footprint_building.geometry), ax=ax,
                                            title = "Building in analysis")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related buildings footprint
                gdf_aux_footprints_ = gdf_aux_footprints[gdf_aux_footprints['group'].isin(adjacent_buildings)]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(gdf_aux_footprints_.geometry, clusters=gdf_aux_footprints_.group, ax=ax,
                                            contextual_geometry = building_geom,
                                            title = "Adjacent buildings")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related buildings footprint
                gdf_aux_footprints_nearby = gdf_footprints_global[gdf_footprints_global['group'].isin(nearby_buildings)]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(gdf_aux_footprints_nearby.geometry, clusters=gdf_aux_footprints_nearby.group,
                                            ax=ax, contextual_geometry=building_geom,
                                            title="Nearby buildings")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related parcels
                related_parcels_geom = []
                nearby_related_parcels_geom = []
                related_union = None
                if len(adjacent_parcels)>0:
                    related_parcels_geom = gdf_zone.loc[gdf_zone["single_building_reference"].isin(adjacent_parcels), "parcel_geometry"]
                    related_parcels_geom = [utils.union_geoseries_with_tolerance(item,gap_tolerance=0.05, resolution=2) for item
                                            in list(related_parcels_geom.drop_duplicates())]
                if len(nearby_parcels)>0:
                    nearby_related_parcels_geom = parcels_gdf.loc[parcels_gdf['building_reference'].isin(nearby_parcels), "parcel_geometry"]
                    nearby_related_parcels_geom = [utils.union_geoseries_with_tolerance(item, gap_tolerance=0.05, resolution=2) for item
                       in list(nearby_related_parcels_geom.drop_duplicates())]

                # Detect which is the part of the parcel in contact with the street
                filtered_streets = streets_gdf[streets_gdf.geometry.distance(parcel_geom.centroid) <= 200]
                filtered_streets.reset_index(inplace=True)
                filtered_streets = filtered_streets[filtered_streets.element == "way"]

                if isinstance(parcel_geom, MultiPolygon):
                    # Shapely 2 dropped iteration on MultiPolygon — use .geoms.
                    # (The previous `list(parcel_geom)[0].geoms` was also wrong
                    # under Shapely 1, since Polygon has no `.geoms`.)
                    parcel_limits = list(parcel_geom.geoms)
                elif isinstance(parcel_geom, GeometryCollection):
                    # make_valid on degenerate inputs can return a mixed
                    # GeometryCollection (Polygon + LineString + Point). Only
                    # polygonal pieces have an `.exterior`, so filter the rest.
                    parcel_limits = [g for g in parcel_geom.geoms
                                     if isinstance(g, (Polygon, MultiPolygon))]
                    parcel_limits = list(chain.from_iterable(
                        g.geoms if isinstance(g, MultiPolygon) else [g]
                        for g in parcel_limits
                    ))
                elif isinstance(parcel_geom, Polygon):
                    parcel_limits = [parcel_geom]
                else:
                    # No polygonal area at all — nothing to compute against streets.
                    parcel_limits = []

                all_parcel_segments = []
                touching_street_and_non_adjacent_parcel_segments = []
                non_adjacent_parcel_segments = []

                for geom in parcel_limits:
                    filtered_streets = filtered_streets[~filtered_streets.geometry.intersects(geom)]
                    exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
                    #utils.plot_points_with_indices(exterior_coords)
                    for i in range(len(exterior_coords) - 1):
                        segment = LineString([exterior_coords[i], exterior_coords[i + 1]])
                        # utils.plot_polygons_group(related_parcels_geom,
                        #                     utils.shorten_linestring(segment, 0.1).buffer(2, cap_style="flat"),
                        #                     pdf_file="test2.pdf")
                        all_parcel_segments.append(segment)
                        if (len(related_parcels_geom)==0 or
                            not np.any([related_parcel_geom.intersects(
                                utils.shorten_linestring(segment, 0.1).buffer(2,cap_style="flat")) for
                            related_parcel_geom in related_parcels_geom])):
                            if utils.is_segment_in_contact_with_street(
                                    segment, streets = filtered_streets, max_distance=8, orientation_tolerance=15,
                                    orientation_discrete_interval_in_degrees=orientation_discrete_interval_in_degrees):
                                touching_street_and_non_adjacent_parcel_segments.append(segment)
                            else:
                                non_adjacent_parcel_segments.append(segment)
                street_parcel_limits = {"by_orientation": {}}

                if len(touching_street_and_non_adjacent_parcel_segments) > 0:
                    street_parcel_limits_segments = touching_street_and_non_adjacent_parcel_segments
                elif len(non_adjacent_parcel_segments) > 0:
                    street_parcel_limits_segments = non_adjacent_parcel_segments
                else:
                    street_parcel_limits_segments = all_parcel_segments

                for segment in street_parcel_limits_segments:
                    segment_orientation = str(utils.calculate_wall_outdoor_normal_orientation(
                            segment,
                            orientation_interval=orientation_discrete_interval_in_degrees))
                    if segment_orientation in street_parcel_limits["by_orientation"]:
                        street_parcel_limits["by_orientation"][segment_orientation] += segment.length
                    else:
                        street_parcel_limits["by_orientation"][segment_orientation] = segment.length
                if not street_parcel_limits["by_orientation"]:
                    # No parcel-edge segments at all (e.g. parcel_geom was a
                    # non-polygonal GeometryCollection or all segments were
                    # filtered out). Treat orientation as unknown and let
                    # downstream code propagate NaN safely.
                    street_parcel_limits["main_orientation"] = np.nan
                elif utils.detect_number_of_orientations(street_parcel_limits["by_orientation"])==1:
                    street_parcel_limits["main_orientation"] = utils.weighted_circular_mean(
                        degrees=[float(i) for i in street_parcel_limits["by_orientation"].keys()],
                        weights=[float(i) for i in street_parcel_limits["by_orientation"].values()])
                else:
                    main_orientation = max(street_parcel_limits["by_orientation"].items(), key=lambda item: item[1])[0]
                    street_parcel_limits["main_orientation"] = float(main_orientation)

                if plots:
                    pdf2 = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_streets.pdf")
                    fig, ax = plt.subplots(figsize=(15,15))
                    utils.plot_shapely_geometries(geometries=list(filtered_streets.geometry), ax=ax,
                                            contextual_geometry=[parcel_geom],
                                            contextual_geometry_options={
                                                "linestyle": "solid",
                                                "linewidth": 1.0,
                                                "color": "red",
                                                "alpha": 1,
                                            })
                    pdf2.savefig(fig)
                    fig, ax = plt.subplots(figsize=(15,15))
                    utils.plot_shapely_geometries(geometries=list(filtered_streets.geometry), ax=ax,
                                            contextual_geometry=street_parcel_limits_segments,
                                            contextual_geometry_options={
                                                "linestyle": "solid",
                                                "linewidth": 1.0,
                                                "color": "red",
                                                "alpha": 1,
                                            })
                    pdf2.savefig(fig)
                    plt.close(fig)
                    pdf2.close()

                floor_area = {}
                underground_floor_area = {}
                roof_area = {}
                patios_area = {}
                patios_n = {}
                perimeter = {}
                # Five-bucket wall classification (replaces the old
                # patios|adiabatic|air_contact scheme):
                #   main_facade_wall  — exterior wall facing open air (was air_contact_wall);
                #                       dict[orientation_str, length_per_floor]
                #   party_wall        — wall on a parcel boundary; per-orientation
                #   adiabatic_wall    — shared wall with a tall-enough neighbour (height-aware)
                #   terrain_contact_wall — wall against terrain (underground OR below DEM)
                #   patios_wall       — wall onto an interior patio
                main_facade_wall = {}
                party_wall = {}
                adiabatic_wall = {}
                terrain_contact_wall = {}
                patios_wall = {}
                shadows_at_distance = {}
                # Stay-empty defaults: terrain grid is the only terrain-related
                # field surfaced now. The combined per-floor `shadows_at_distance`
                # is populated inside the per-floor block below.
                terrain_grid = {}
                parcel_at_distance = {}
                main_facade_significant_orientations = []
                main_facade_significant_orientations_by_floor = {}
                party_wall_significant_orientations = []
                party_wall_significant_orientations_by_floor = {}
                # Per-sbr list of {geometry, type, floor, orientation_degrees,
                # length_m} for energy-simulation interop. Stored as opaque
                # list-of-dict; routed through HEAVY_BLOB_COLS so the bulk gdf
                # stays small.
                wall_segments_sbr = []
                n_floors = gdf_aux_footprint_building.floor.max() if gdf_aux_footprint_building.floor.max() >= 0 else np.nan
                n_underground_floors = -gdf_aux_footprint_building.floor.min() if gdf_aux_footprint_building.floor.min() < 0 else 0
                n_buildings = len(
                    gdf_zone[
                            gdf_zone["building_reference"] == building_gdf_item.iloc[0]["building_reference"]
                        ]["single_building_reference"].unique()
                )

                # Hoisted from the (later) shadow block so the wall classifier
                # can tell whether each segment sits below DEM (terrain_contact)
                # or behind a tall-enough adjacent neighbour (adiabatic).
                if 'elevation' in building_gdf_item.columns:
                    current_building_elevation = utils._coerce_scalar_float(
                        building_gdf_item['elevation'], 0.0)
                else:
                    current_building_elevation = 0.0

                # Per-adjacent-sbr roof elevation. `gdf_aux_footprints_` was
                # filtered above to the buildings touching ours; elevation
                # lives on `gdf_zone` rows keyed by single_building_reference.
                # Falls back to the analysed building's elevation when missing
                # (adjacent buildings are very close — better than 0 m).
                adjacent_elev_by_sbr = {}
                if "elevation" in gdf_zone.columns and len(adjacent_buildings) > 0:
                    adjacent_elev_by_sbr = dict(
                        gdf_zone[gdf_zone["single_building_reference"].isin(adjacent_buildings)]
                        .groupby("single_building_reference")["elevation"]
                        .first()
                    )
                adjacent_roof_elev = {}
                adj_footprint_all = {}   # {sbr: union of all floor footprints}
                from shapely.ops import unary_union as _uu_adj

                # Adjacent buildings (share a parcel boundary)
                if len(gdf_aux_footprints_) > 0:
                    _max_floor_per_adj = gdf_aux_footprints_.groupby("group")["floor"].max()
                    for _adj_sbr, _adj_max_f in _max_floor_per_adj.items():
                        adjacent_roof_elev[_adj_sbr] = utils.floor_base_elevation(
                            utils._coerce_scalar_float(
                                adjacent_elev_by_sbr.get(_adj_sbr, current_building_elevation),
                                0.0,
                            ),
                            int(_adj_max_f) + 1, ground_floor_height, floor_height,
                        )
                    for _adj_sbr, _adj_rows in gdf_aux_footprints_.groupby("group"):
                        try:
                            adj_footprint_all[_adj_sbr] = _uu_adj(_adj_rows.geometry.values)
                        except Exception:
                            pass

                # Nearby buildings (different parcel, same block — may still be
                # physically touching). Also included so the adiabatic check sees
                # walls that share a face with a neighbour from another parcel.
                if len(gdf_aux_footprints_nearby) > 0:
                    _nearby_elev_by_sbr = {}
                    _elev_src = gdf_footprints_global if "elevation" in gdf_footprints_global.columns else gdf_zone
                    _elev_col = "group" if _elev_src is gdf_footprints_global else "single_building_reference"
                    try:
                        _nearby_elev_by_sbr = dict(
                            _elev_src[_elev_src[_elev_col].isin(nearby_buildings)]
                            .groupby(_elev_col)["elevation"].first()
                        )
                    except Exception:
                        pass
                    _max_floor_nearby = gdf_aux_footprints_nearby.groupby("group")["floor"].max()
                    for _nbr_sbr, _nbr_max_f in _max_floor_nearby.items():
                        if _nbr_sbr not in adjacent_roof_elev:
                            adjacent_roof_elev[_nbr_sbr] = utils.floor_base_elevation(
                                utils._coerce_scalar_float(
                                    _nearby_elev_by_sbr.get(_nbr_sbr, current_building_elevation),
                                    0.0,
                                ),
                                int(_nbr_max_f) + 1, ground_floor_height, floor_height,
                            )
                    for _nbr_sbr, _nbr_rows in gdf_aux_footprints_nearby.groupby("group"):
                        if _nbr_sbr not in adj_footprint_all:
                            try:
                                adj_footprint_all[_nbr_sbr] = _uu_adj(_nbr_rows.geometry.values)
                            except Exception:
                                pass

                if n_underground_floors > 0:

                    for underground_floor in range(1,(n_underground_floors+1)):
                        underground_floor_area[(underground_floor)] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == -(underground_floor)]['geometry'].area.sum(),
                            2
                        )

                if n_floors is not np.nan:

                    for floor in range(n_floors + 1):
                        floor_area[floor] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                            2
                        )
                        if floor == n_floors:
                            roof_area[floor] = round(
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                                2
                            )
                        else:
                            roof_area[floor] = round(
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor][
                                    'geometry'].area.sum() -
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == (floor + 1)][
                                    'geometry'].area.sum(),
                                2
                            )
                        patios = utils.patios_in_the_building(
                            patios_geoms=patios_detected[floor],
                            building_geom=building_geom,
                            tolerance=0.8
                        )
                        patios_n[floor] = len(patios)
                        patios_area[floor] = round(sum([patio.area for patio in patios]), 2)
                        walls = gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry']
                        if isinstance(list(walls)[0], MultiPolygon):
                            walls = [walls_i for walls_i in list(walls)[0].geoms]
                        elif isinstance(list(walls)[0], Polygon):
                            walls = list(walls)

                        # Initialize the lengths for each type of wall contact and for each orientation
                        main_facade_wall[floor] = {direction: 0.0 for direction in [str(i) for i in
                                                                                    list(range(0, 360, orientation_discrete_interval_in_degrees))]}
                        party_wall[floor] = {direction: 0.0 for direction in [str(i) for i in
                                                                              list(range(0, 360, orientation_discrete_interval_in_degrees))]}
                        adiabatic_wall[floor] = 0.0
                        terrain_contact_wall[floor] = 0.0
                        patios_wall[floor] = 0.0
                        perimeter[floor] = round(sum([peri.exterior.length for peri in walls]), 2)

                        # Per-floor elevation thresholds.
                        # - eye_level_floor: mid-floor observer height, matches
                        #   the shadow block's `eye_level` exactly.
                        # - floor_top_elev: top-of-slab elevation, used by the
                        #   terrain_contact below-DEM test.
                        if floor == 0:
                            eye_level_floor = current_building_elevation + ground_floor_height / 2.0
                            floor_top_elev = current_building_elevation + ground_floor_height
                        else:
                            eye_level_floor = (current_building_elevation + ground_floor_height
                                               + (floor - 0.5) * floor_height)
                            floor_top_elev = (current_building_elevation + ground_floor_height
                                              + floor * floor_height)

                        # Base elevation of the current floor — used by the
                        # adiabatic test: a neighbour must reach at least this
                        # height (not just floor 0) to shelter this wall.
                        if floor == 0:
                            floor_base_elev = current_building_elevation
                        else:
                            floor_base_elev = (current_building_elevation
                                               + ground_floor_height
                                               + (floor - 1) * floor_height)

                        for geom in walls:

                            # Indoor patios — interior rings count as patios_wall
                            # length AND emit per-segment patio records so the
                            # wall_segments view stays comprehensive.
                            patios_wall[floor] += sum([item.length for item in list(geom.interiors)])
                            for ring in geom.interiors:
                                ring_coords = list(ring.coords)
                                for j in range(len(ring_coords) - 1):
                                    ring_seg = LineString([ring_coords[j], ring_coords[j + 1]])
                                    if ring_seg.length < 0.001:
                                        continue
                                    wall_segments_sbr.append({
                                        "geometry": ring_seg,
                                        "type": "patios",
                                        "floor": floor,
                                        "orientation_degrees": None,
                                        "length_m": round(ring_seg.length, 2),
                                    })

                            # Break down the exterior into segments.
                            # Precompute split-boundary geometries for this
                            # floor once (parcel edges + neighbour footprints)
                            # so we can split each wall segment wherever a
                            # neighbour starts or ends — giving finer per-type
                            # accounting rather than one label for a long wall.
                            _split_geoms = (
                                list(related_parcels_geom)
                                + list(adj_footprint_all.values())
                            )

                            exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
                            for i in range(len(exterior_coords) - 1):
                                raw_segment = LineString(
                                    [exterior_coords[i], exterior_coords[i + 1]])
                                if raw_segment.length < 0.001:
                                    continue

                                # Orientation is the same for all sub-segments
                                # (they are collinear).
                                segment_orientation = str(
                                    utils.calculate_wall_outdoor_normal_orientation(
                                        raw_segment,
                                        orientation_interval=orientation_discrete_interval_in_degrees))

                                sub_segments = _split_segment_at_geom_boundaries(
                                    raw_segment, _split_geoms)

                                for segment in sub_segments:
                                    segment_assigned = None

                                    # 1. PATIOS — segment intersects an interior
                                    # courtyard polygon.
                                    for patio in patios:
                                        if utils.segment_intersects_with_tolerance(
                                                segment, patio,
                                                buffer_distance=0.1,
                                                area_percentage_threshold=15
                                        ):
                                            patios_wall[floor] += round(segment.length, 2)
                                            if plots:
                                                fig, ax = plt.subplots()
                                                utils.plot_shapely_geometries(
                                                    geometries=[geom] + [segment],
                                                    title=f"Wall ID:{i}, floor: {floor},\n"
                                                          f"orientation: {segment_orientation},"
                                                          f"type: patios",
                                                    ax=ax,
                                                    contextual_geometry=building_geom)
                                                pdf.savefig(fig)
                                                plt.close(fig)
                                            segment_assigned = "patios"
                                            break

                                    # 2. TERRAIN_CONTACT — basement OR midpoint
                                    # below the floor's top-of-slab per DEM.
                                    if segment_assigned is None:
                                        is_terrain = False
                                        if floor < 0:
                                            is_terrain = True
                                        elif dem_sampler is not None:
                                            mid = segment.interpolate(0.5, normalized=True)
                                            try:
                                                terrain_elev = float(
                                                    dem_sampler([mid.x], [mid.y])[0]
                                                )
                                            except (TypeError, ValueError, Exception):
                                                terrain_elev = None
                                            if (terrain_elev is not None
                                                    and not math.isnan(terrain_elev)
                                                    and terrain_elev > floor_top_elev):
                                                is_terrain = True
                                        if is_terrain:
                                            terrain_contact_wall[floor] += round(segment.length, 2)
                                            if plots:
                                                fig, ax = plt.subplots()
                                                utils.plot_shapely_geometries(
                                                    geometries=[geom] + [segment],
                                                    title=f"Wall ID:{i}, floor: {floor},\n"
                                                          f"orientation: {segment_orientation},"
                                                          f"type: terrain_contact",
                                                    ax=ax,
                                                    contextual_geometry=building_geom)
                                                pdf.savefig(fig)
                                                plt.close(fig)
                                            segment_assigned = "terrain_contact"

                                    # 3. ADIABATIC — segment touches any floor of a
                                    # neighbour (adjacent or nearby) AND that
                                    # neighbour's roof is at or above this floor's
                                    # base elevation.  Checked BEFORE party_wall: a
                                    # tall-enough building physically behind the wall
                                    # means no heat exchange (adiabatic) regardless
                                    # of whether a parcel boundary also runs there.
                                    # Union of all neighbour floors (adj_footprint_all)
                                    # handles the case where a 2-floor neighbour must
                                    # be visible from floor 3+ of a taller building.
                                    # Per-segment splitting upstream ensures only the
                                    # portion actually backed by the neighbour gets
                                    # this label; the rest falls through to party_wall
                                    # or main_facade.
                                    if segment_assigned is None:
                                        # Probe just outside the wall: a wall is
                                        # only adiabatic if a neighbour footprint
                                        # actually lies on its outward side. When
                                        # the outward side can't be disambiguated
                                        # (probe is None) we fall back to the
                                        # looser coincidence test.
                                        _probe = _outward_probe_point(segment, geom)
                                        for adj_sbr, adj_union_geom in adj_footprint_all.items():
                                            if _probe is not None:
                                                try:
                                                    behind = adj_union_geom.buffer(0.05).contains(_probe)
                                                except Exception:
                                                    behind = False
                                                if not behind:
                                                    continue
                                            elif not utils.segment_intersects_with_tolerance(
                                                    segment, adj_union_geom,
                                                    buffer_distance=0.1,
                                                    area_percentage_threshold=15
                                            ):
                                                continue
                                            adj_top = adjacent_roof_elev.get(
                                                adj_sbr, float("-inf"))
                                            if adj_top >= floor_base_elev:
                                                adiabatic_wall[floor] += round(segment.length, 2)
                                                if plots:
                                                    fig, ax = plt.subplots()
                                                    utils.plot_shapely_geometries(
                                                        geometries=[geom] + [segment],
                                                        title=f"Wall ID: {i}, floor: {floor},\n"
                                                              f"orientation: {segment_orientation}, "
                                                              f"type: adiabatic",
                                                        ax=ax,
                                                        contextual_geometry=building_geom)
                                                    pdf.savefig(fig)
                                                    plt.close(fig)
                                                segment_assigned = "adiabatic"
                                                break

                                    # 4. PARTY_WALL — segment lies on a parcel
                                    # boundary with no tall-enough building behind it
                                    # (per-orientation; treated like an exterior wall
                                    # without windows).
                                    if segment_assigned is None and len(related_parcels_geom) > 0:
                                        for related_parcel in related_parcels_geom:
                                            if utils.segment_intersects_with_tolerance(
                                                    segment, related_parcel,
                                                    buffer_distance=0.1,
                                                    area_percentage_threshold=15
                                            ):
                                                party_wall[floor][segment_orientation] += round(segment.length, 2)
                                                if plots:
                                                    fig, ax = plt.subplots()
                                                    utils.plot_shapely_geometries(
                                                        geometries=[geom] + [segment],
                                                        title=f"Wall ID: {i}, floor: {floor},\n"
                                                              f"orientation: {segment_orientation}, "
                                                              f"type: party_wall",
                                                        ax=ax,
                                                        contextual_geometry=building_geom)
                                                    pdf.savefig(fig)
                                                    plt.close(fig)
                                                segment_assigned = "party_wall"
                                                break

                                    # 5. MAIN_FACADE — true exterior wall facing
                                    # open air.
                                    if segment_assigned is None:
                                        main_facade_wall[floor][segment_orientation] += round(segment.length, 2)
                                        if plots:
                                            fig, ax = plt.subplots()
                                            utils.plot_shapely_geometries(
                                                geometries=[geom] + [segment],
                                                title=f"Wall ID: {i}, floor: {floor},\n"
                                                      f"orientation: {segment_orientation}, "
                                                      f"type: main_facade",
                                                ax=ax,
                                                contextual_geometry=building_geom)
                                            pdf.savefig(fig)
                                            plt.close(fig)
                                        segment_assigned = "main_facade"

                                    # Per-segment record (energy-simulation interop).
                                    wall_segments_sbr.append({
                                        "geometry": segment,
                                        "type": segment_assigned,
                                        "floor": floor,
                                        "orientation_degrees": (
                                            float(segment_orientation)
                                            if segment_assigned in ("party_wall", "main_facade")
                                            else None
                                        ),
                                        "length_m": round(segment.length, 2),
                                    })

                    # Significant-orientation summaries: any direction where the
                    # per-floor wall length crosses 10 % of the floor perimeter.
                    # Same logic as the old air_contact pass, now applied to BOTH
                    # main_facade_wall and party_wall.
                    for floor, mf_walls in main_facade_wall.items():
                        threshold = 0.1 * perimeter[floor]
                        flagged = sorted({int(o) for o, l in mf_walls.items() if l > threshold})
                        main_facade_significant_orientations.extend(flagged)
                        main_facade_significant_orientations_by_floor[floor] = flagged
                    main_facade_significant_orientations = list(set(main_facade_significant_orientations))

                    for floor, pw_walls in party_wall.items():
                        threshold = 0.1 * perimeter[floor]
                        flagged = sorted({int(o) for o, l in pw_walls.items() if l > threshold})
                        party_wall_significant_orientations.extend(flagged)
                        party_wall_significant_orientations_by_floor[floor] = flagged
                    party_wall_significant_orientations = list(set(party_wall_significant_orientations))

                    # Close the PDF file for the current building
                    if plots:
                        pdf.close()

                    # if building_type == "Non-residential":
                    #     starting_residential_floor = np.nan
                    # elif premises:
                    #     starting_residential_floor = 1
                    #     if building_type == "Multi-family" and n_floors >= n_dwellings:
                    #         n_dwellings = n_dwellings - 1
                    # else:
                    #     if n_floors > 6 and building_type == "Multi-family":
                    #         starting_residential_floor = 1
                    #     else:
                    #         starting_residential_floor = 0
                    # floor_area_with_possible_residential_use = [floor_area[floor] if floor >= starting_residential_floor or starting_residential_floor is np.nan else 0.0 for floor in range(n_floors + 1)]

                if max(gdf_aux_footprint_building.floor.max(), gdf_aux_footprints_nearby.floor.max()) > 0:

                    # Combined per-floor "first horizontal blocker" analysis.
                    # For each (orientation, floor) we cast a horizontal ray at
                    # mid-floor eye level and return the distance to the first
                    # blocker, where a blocker is either:
                    #   - a nearby building polygon whose roof rises above the
                    #     eye level (line of sight blocked at the building's
                    #     ground-level edge — equivalent to a terrain step), or
                    #   - rising terrain that crosses the eye level
                    # whichever comes first. There is no separate "terrain
                    # shadow" output — it's folded into `shadows_at_distance`.
                    if plots:
                        pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_shadows.pdf")
                    else:
                        pdf = None

                    # `current_building_elevation` is hoisted above the wall
                    # loop so the wall classifier can see it; reused here.

                    # Pre-compute neighbour roof elevations (one per nearby
                    # building, not per-row). n_floors_above_ground per
                    # neighbour = max(footprint.floor) + 1; roof elevation =
                    # floor_base_elevation(elev, n_floors_above_ground).
                    if len(gdf_aux_footprints_nearby) > 0:
                        _max_floor_per_nb = (
                            gdf_aux_footprints_nearby.groupby("group")["floor"].max()
                        )
                        if "elevation" in gdf_aux_footprints_nearby.columns:
                            _elev_per_nb = (
                                gdf_aux_footprints_nearby.groupby("group")["elevation"].first()
                            )
                        else:
                            _elev_per_nb = pd.Series(dtype=float)
                        nearby_roof_elev = {
                            grp: utils.floor_base_elevation(
                                utils._coerce_scalar_float(_elev_per_nb.get(grp, 0.0), 0.0),
                                int(_max_floor_per_nb[grp]) + 1,
                                ground_floor_height, floor_height,
                            )
                            for grp in _max_floor_per_nb.index
                        }
                    else:
                        nearby_roof_elev = {}

                    # Sample the DEM profile ONCE per building (observer-
                    # independent); per-floor blocker derivation reuses it.
                    terrain_profile = {}
                    terrain_profile_distances = []
                    if dem_sampler is not None:
                        terrain_profile, terrain_profile_distances = (
                            utils.sample_terrain_profile_around_centroid(
                                centroid=building_geom.centroid,
                                dem_sampler=dem_sampler,
                                orientation_interval=orientation_discrete_interval_in_degrees,
                                max_distance=150.0,
                                sample_step=5.0,
                            )
                        )

                    # Per-neighbour, per-floor footprint table for the new
                    # `compute_floor_shadow_region`. Stepped buildings (wider
                    # podium, narrower tower) keep distinct shapes per floor;
                    # the shadow function picks the lowest neighbour-floor
                    # whose top exceeds the analyzed building's eye level
                    # and unions footprints from that floor upward.
                    nearby_buildings_data = []
                    if len(gdf_aux_footprints_nearby) > 0:
                        for grp, group_df in gdf_aux_footprints_nearby.groupby('group'):
                            floors_dict = {}
                            for _, row in group_df.iterrows():
                                g = row.geometry
                                if g is None or g.is_empty:
                                    continue
                                try:
                                    fi = int(row['floor'])
                                except (TypeError, ValueError):
                                    continue
                                floors_dict[fi] = g
                            if not floors_dict:
                                continue
                            nb_elev = utils._coerce_scalar_float(
                                group_df.iloc[0].get('elevation', 0.0), 0.0,
                            )
                            nearby_buildings_data.append({
                                'elev': float(nb_elev),
                                'floors': floors_dict,
                            })

                    max_floor_idx = max(gdf_aux_footprint_building.floor.max(),
                                        gdf_aux_footprints_nearby.floor.max())
                    for floor in range(max_floor_idx + 1):
                        # Mid-floor eye level. Floor 0 sits on the DEM with the
                        # ground-floor height; eye = DEM + g/2. Upper floors
                        # are floor_height tall; eye = DEM + g + (f - 0.5) * h.
                        if floor == 0:
                            eye_level = current_building_elevation + ground_floor_height / 2.0
                        else:
                            eye_level = (current_building_elevation + ground_floor_height
                                         + (floor - 0.5) * floor_height)

                        # Only consider neighbours that exist at this floor for
                        # ray intersections (their geometry is needed); the
                        # roof-vs-eye-level gate uses the precomputed roof.
                        floor_nearby = gdf_aux_footprints_nearby[
                            gdf_aux_footprints_nearby.floor == floor
                        ].copy()

                        building_heights = {}
                        for _, nearby_building in floor_nearby.iterrows():
                            grp = nearby_building.get("group")
                            nearby_top = nearby_roof_elev.get(
                                grp,
                                utils.floor_base_elevation(
                                    utils._coerce_scalar_float(
                                        nearby_building.get('elevation', 0.0), 0.0),
                                    int(nearby_building.get('floor', 0)) + 1,
                                    ground_floor_height, floor_height,
                                ),
                            )
                            building_heights[nearby_building.name] = {
                                'height': nearby_top,
                                'relative_height': nearby_top - eye_level,
                                'is_nearby': False,
                            }

                        # Building-blocker pass — single ray per orientation
                        # from the building's centroid. The same eye-level
                        # gate applies. The points-based variant remains
                        # available but the centroid version is simpler and
                        # deterministic (no random sampling noise).
                        building_shadow_result = utils.distance_from_centroid_to_polygons_by_orientation(
                            geom=building_geom.exterior if isinstance(building_geom, Polygon) else (
                                ([LinearRing(polygon.exterior.coords) for polygon in building_geom.geoms])),
                            other_polygons=floor_nearby.geometry,
                            centroid=building_geom.centroid,
                            orientation_interval=orientation_discrete_interval_in_degrees,
                            plots=plots,
                            pdf=pdf,
                            floor=floor,
                            max_distance=150,
                            current_height=eye_level,
                            building_heights=building_heights,
                        )
                        contour_by_orientation = building_shadow_result.get("contour", {})
                        building_dist = building_shadow_result.get("building_shadows", {})

                        # Terrain-blocker pass — distance to first DEM sample
                        # above eye level along each ray, relative to contour.
                        if terrain_profile:
                            terrain_dist, _ = utils.derive_terrain_shadow_from_profile(
                                profile=terrain_profile,
                                sample_distances=terrain_profile_distances,
                                contour_by_orientation=contour_by_orientation,
                                observer_height=eye_level,
                                clearance=0.0,
                            )
                        else:
                            terrain_dist = {k: np.inf for k in contour_by_orientation}

                        combined_shadows = {
                            k: float(min(building_dist.get(k, np.inf),
                                         terrain_dist.get(k, np.inf)))
                            for k in contour_by_orientation
                        }

                        # New: full plan-view multipolygon of the area blocked
                        # from this floor's eye level. Terrain shadow + the
                        # above-eye-level portion of each nearby building's
                        # per-floor footprints (stepped buildings project
                        # only their tower at high analyzed floors), with
                        # everything behind a closer terrain rise dropped.
                        floor_shadow_geom = utils.compute_floor_shadow_region(
                            building_geom=building_geom,
                            nearby_buildings=nearby_buildings_data,
                            dem_sampler=dem_sampler,
                            eye_level=eye_level,
                            ground_floor_height=ground_floor_height,
                            floor_height=floor_height,
                            max_distance=150.0,
                            orientation_interval=orientation_discrete_interval_in_degrees,
                        )

                        shadows_at_distance[floor] = {
                            "shadows": combined_shadows,
                            "contour": contour_by_orientation,
                            "shadows_geom": floor_shadow_geom,
                        }
                    if plots:
                        pdf.close()

                    # 5 m DEM grid around the building (bbox + 100 m margin)
                    # for contour-line rendering in the shadow plot.
                    terrain_grid = {}
                    if dem_sampler is not None:
                        terrain_grid = utils.sample_dem_grid_around_geometry(
                            geometry=building_geom,
                            dem_sampler=dem_sampler,
                            cell_size=5.0,
                            margin=100.0,
                        )

                if len(nearby_related_parcels_geom)>0:
                    # Street widths
                    if plots:
                        pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_street_widths.pdf")
                    else:
                        pdf = None
                    parcel_at_distance = utils.distance_from_points_to_polygons_by_orientation(
                        geom=street_parcel_limits_segments,
                        other_polygons=nearby_related_parcels_geom,
                        num_points_per_geom_element=10,
                        orientations_to_analyse=sorted(street_parcel_limits["by_orientation"].keys(), key=int),
                        force_one_orientation_per_geom_element=True,
                        plots=plots,
                        pdf=pdf,
                        max_distance=150
                    )
                    if plots:
                        pdf.close()

                street_parcel_limits["next_building_by_orientation"] = {}
                if len(street_parcel_limits["by_orientation"]) > 0 and len(shadows_at_distance)>0:
                    for i in street_parcel_limits["by_orientation"].keys():
                        street_parcel_limits["next_building_by_orientation"][i] = shadows_at_distance[0]["shadows"][i]

                street_parcel_limits["next_building_main_orientation"] = np.inf
                if street_parcel_limits["main_orientation"] is not np.nan and len(shadows_at_distance)>0:
                    street_parcel_limits["next_building_main_orientation"] = shadows_at_distance[0]["shadows"][
                        str(utils.discrete_orientation(
                            street_parcel_limits["main_orientation"],
                            orientation_discrete_interval_in_degrees))
                    ]

                # `parcel_at_distance` is the {distances-by-orientation,
                # representativity-by-orientation} tuple returned by
                # distance_from_points_to_polygons_by_orientation when nearby
                # parcels exist; otherwise it's still the initial empty dict.
                # The first element is the orientation→width mapping.
                if isinstance(parcel_at_distance, tuple) and len(parcel_at_distance) > 0:
                    widths_at_zero = parcel_at_distance[0]
                else:
                    widths_at_zero = {}
                street_parcel_limits["street_width_by_orientation"] = widths_at_zero
                filtered = {k: v for k, v in widths_at_zero.items() if np.isfinite(v)}
                if not filtered:
                    street_parcel_limits["street_width_main_orientation"] = np.inf
                else:
                    street_parcel_limits["main_orientation"] = max(filtered, key=filtered.get)
                    street_parcel_limits["street_width_main_orientation"] = widths_at_zero[
                            max(filtered, key=filtered.get)]

                # Store results
                results_.append({
                    'building_reference': single_building_reference.split("_")[0],
                    'single_building_reference': single_building_reference,
                    'zone_reference': zone_reference,
                    'sbr__n_buildings': n_buildings,
                    'sbr__detached': (np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0) < 1.0,
                    'br__parcel_orientations': street_parcel_limits["by_orientation"],
                    'br__parcel_main_orientation': street_parcel_limits["main_orientation"],
                    'br__next_building_by_orientation': street_parcel_limits["next_building_by_orientation"],
                    'br__next_building_main_orientation': street_parcel_limits["next_building_main_orientation"],
                    'br__street_width_by_orientation': street_parcel_limits["street_width_by_orientation"],
                    'br__street_width_main_orientation': street_parcel_limits["street_width_main_orientation"],
                    'br__exists_ground_commercial_premises': premises,
                    'br__ground_commercial_premises_names': premises_names,
                    'br__ground_commercial_premises_typology': premises_activity_typologies,
                    'br__ground_commercial_premises_last_revision': premises_last_revision,
                    'sbr__floors_above_ground': n_floors+1,
                    'sbr__floors_below_ground': n_underground_floors,
                    'sbr__below_ground_built_area_by_floor': [underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)] if n_underground_floors > 0 else [],
                    'sbr__below_ground_built_area': np.sum([underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)]) if n_underground_floors > 0 else 0.0,
                    'sbr__above_ground_built_area_by_floor': [floor_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                    'sbr__above_ground_built_area': np.sum([floor_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                    'sbr__above_ground_roof_area_by_floor': [roof_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                    'sbr__above_ground_roof_area': np.sum([roof_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                    'sbr__building_footprint_area': round(building_geom.area if isinstance(building_geom, Polygon)
                                                     else sum(polygon.area for polygon in building_geom.geoms) if isinstance(building_geom, MultiPolygon)
                                                     else 0.0, 2),
                    'sbr__building_footprint_geometry': building_geom.exterior if isinstance(building_geom, Polygon) else (
                                MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                    'sbr__building_footprint_by_floor': gdf_aux_footprint_building[["group","floor","geometry"]].to_dict(index=False, orient='split')['data'],
                    'sbr__patios_area_by_floor': [patios_area[floor] for floor in range(n_floors + 1)] if patios_area != {} else [],
                    'sbr__patios_number_by_floor': [patios_n[floor] for floor in range(n_floors + 1)] if patios_n != {} else [],
                    'sbr__walls_between_slabs': np.sum([perimeter[floor] for floor in
                                                         range(n_floors + 1)] if perimeter != {} else []),
                    'sbr__building_perimeter_by_floor': [perimeter[floor] for floor in range(n_floors + 1)] if perimeter != {} else [],
                    'sbr__main_facade_wall_by_floor': {key: [main_facade_wall[d][key] for d in main_facade_wall] for key in
                                         main_facade_wall[0]} if main_facade_wall != {} else {},
                    'sbr__main_facade_wall':  {key: np.sum([main_facade_wall[d][key] for d in main_facade_wall]) for key in
                                         main_facade_wall[0]} if main_facade_wall != {} else {},
                    'sbr__main_facade_wall_significant_orientations': main_facade_significant_orientations,
                    'sbr__main_facade_wall_significant_orientations_by_floor': main_facade_significant_orientations_by_floor,
                    'sbr__party_wall_by_floor': {key: [party_wall[d][key] for d in party_wall] for key in
                                         party_wall[0]} if party_wall != {} else {},
                    'sbr__party_wall': {key: np.sum([party_wall[d][key] for d in party_wall]) for key in
                                         party_wall[0]} if party_wall != {} else {},
                    'sbr__party_wall_significant_orientations': party_wall_significant_orientations,
                    'sbr__party_wall_significant_orientations_by_floor': party_wall_significant_orientations_by_floor,
                    'sbr__adiabatic_wall_by_floor': [adiabatic_wall[floor] for floor in range(n_floors + 1)] if adiabatic_wall != {} else [],
                    'sbr__adiabatic_wall': np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0,
                    'sbr__terrain_contact_wall_by_floor': [terrain_contact_wall[floor] for floor in range(n_floors + 1)] if terrain_contact_wall != {} else [],
                    'sbr__terrain_contact_wall': np.sum([terrain_contact_wall[floor] for floor in range(n_floors + 1)]) if terrain_contact_wall != {} else 0.0,
                    'sbr__patios_wall_by_floor': [patios_wall[floor] for floor in range(n_floors + 1)] if patios_wall != {} else [],
                    'sbr__patios_wall_total': np.sum([patios_wall[floor] for floor in range(n_floors + 1)]) if patios_wall != {} else 0.0,
                    # Per-segment wall records (energy-simulation interop).
                    # Heavy column — excluded from the bulk read path; fetch
                    # on-demand via `utils.get_heavy_columns_for_building`.
                    'sbr__wall_segments': wall_segments_sbr,
                    # Combined per-floor shadows (building + terrain blocker at
                    # mid-floor eye level). Shape: {orient: [floor0, floor1, ...]}.
                    'sbr__shadows_at_distance': {key: [shadows_at_distance[d]['shadows'][key] for d in shadows_at_distance] for key in
                                            shadows_at_distance[0]['shadows']} if shadows_at_distance != {} else {},
                    'sbr__building_contour_at_distance': {key: np.mean([shadows_at_distance[d]['contour'][key] for d in shadows_at_distance]) for key in
                                            shadows_at_distance[0]['contour']} if shadows_at_distance != {} else {},
                    # New: per-floor blocked-region multipolygon (plan view).
                    # Each key is a floor index, each value is a Polygon /
                    # MultiPolygon covering the area blocked from that floor's
                    # eye level within 150 m of the building.
                    'sbr__shadows': {f: shadows_at_distance[f].get('shadows_geom')
                                     for f in shadows_at_distance} if shadows_at_distance != {} else {},
                    # 5 m DEM grid (bbox + 100 m margin) for contour-line
                    # rendering in the shadow plot.
                    'sbr__terrain_grid': terrain_grid,
                })

            if len(results_) > 0:
                results_ = pd.DataFrame(results_)
                sbr_results_concat = pd.DataFrame()
                br_results_concat = pd.DataFrame()
                for building_reference in results_.building_reference.unique():
                    # building_reference = "0004701DF3800C"#"9505401DF2890F"

                    sbr_results = results_[results_.building_reference == building_reference]
                    br_agg_dict = {
                        'building_reference': [building_reference],
                        'br__detached': [np.any(sbr_results['sbr__detached'])],
                        'br__parcel_orientations': [sbr_results.iloc[0]['br__parcel_orientations']],
                        'br__parcel_main_orientation': [sbr_results.iloc[0]['br__parcel_main_orientation']],
                        'br__street_width_by_orientation': [sbr_results.iloc[0]['br__street_width_by_orientation']],
                        'br__street_width_main_orientation': [sbr_results.iloc[0]['br__street_width_main_orientation']],
                        'br__next_building_by_orientation': [sbr_results.iloc[0]['br__next_building_by_orientation']],
                        'br__next_building_main_orientation': [sbr_results.iloc[0]['br__next_building_main_orientation']],
                        'br__exists_ground_commercial_premises': [sbr_results.iloc[0]['br__exists_ground_commercial_premises']],
                        'br__ground_commercial_premises_names': [sbr_results.iloc[0]['br__ground_commercial_premises_names']],
                        'br__ground_commercial_premises_typology': [sbr_results.iloc[0]['br__ground_commercial_premises_typology']],
                        'br__ground_commercial_premises_last_revision': [sbr_results.iloc[0]['br__ground_commercial_premises_last_revision']],
                        'br__floors_above_ground': [max(sbr_results['sbr__floors_above_ground'])],
                        'br__floors_below_ground': [max(sbr_results['sbr__floors_below_ground'])],
                        'br__below_ground_built_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__below_ground_built_area_by_floor'], fillvalue=0)]],
                        'br__below_ground_built_area': [sum(sbr_results['sbr__below_ground_built_area'])],
                        'br__above_ground_built_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__above_ground_built_area_by_floor'], fillvalue=0)]],
                        'br__above_ground_built_area': [sum(sbr_results['sbr__above_ground_built_area'])],
                        'br__above_ground_roof_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__above_ground_roof_area_by_floor'], fillvalue=0)]],
                        'br__above_ground_roof_area': [sum(sbr_results['sbr__above_ground_roof_area'])],
                        'br__building_footprint_area': [sum(sbr_results['sbr__building_footprint_area'])],
                        'br__building_footprint_geometry': [linemerge(
                            [line for geom in sbr_results['sbr__building_footprint_geometry'].tolist() for line in
                             (geom.geoms if geom.geom_type == "MultiLineString" else [geom])])],
                        'br__building_footprint_by_floor':
                            [[item for sublist in sbr_results['sbr__building_footprint_by_floor'] for item in sublist]],
                        'br__patios_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_area_by_floor'], fillvalue=0)]],
                        'br__patios_number_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_number_by_floor'], fillvalue=0)]],
                        'br__walls_between_slabs': [sum(sbr_results['sbr__walls_between_slabs'])],
                        'br__building_perimeter_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__building_perimeter_by_floor'], fillvalue=0)]],
                        'br__main_facade_wall_by_floor': [dict(sorted({
                            k: [
                                sum(x)
                                for x in zip_longest(
                                    *(
                                        d.get(
                                            k,
                                            [0] * (
                                                max(
                                                    (len(v) for v in d.values() if isinstance(v, list)),
                                                    default=1,
                                                )
                                            )
                                        )
                                        for d in sbr_results["sbr__main_facade_wall_by_floor"]
                                    ),
                                    fillvalue=0,
                                )
                            ]
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__main_facade_wall_by_floor"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__main_facade_wall': [dict(sorted({
                            k: sum(
                                d.get(k, 0)
                                for d in sbr_results["sbr__main_facade_wall"]
                            )
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__main_facade_wall"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__main_facade_wall_significant_orientations': [
                            dict(sorted({
                                k: sum(
                                    d.get(k, 0)
                                    for d in sbr_results["sbr__main_facade_wall_significant_orientations"]
                                    if isinstance(d, dict)
                                )
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__main_facade_wall_significant_orientations"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__main_facade_wall_significant_orientations_by_floor': [
                            dict(sorted({
                                k: [
                                    sum(floor_vals)
                                    for floor_vals in zip_longest(
                                        *[
                                            d.get(k, []) for d in sbr_results["sbr__main_facade_wall_significant_orientations_by_floor"]
                                            if isinstance(d, dict)
                                        ],
                                        fillvalue=0
                                    )
                                ]
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__main_facade_wall_significant_orientations_by_floor"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__party_wall_by_floor': [dict(sorted({
                            k: [
                                sum(x)
                                for x in zip_longest(
                                    *(
                                        d.get(
                                            k,
                                            [0] * (
                                                max(
                                                    (len(v) for v in d.values() if isinstance(v, list)),
                                                    default=1,
                                                )
                                            )
                                        )
                                        for d in sbr_results["sbr__party_wall_by_floor"]
                                    ),
                                    fillvalue=0,
                                )
                            ]
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__party_wall_by_floor"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__party_wall': [dict(sorted({
                            k: sum(
                                d.get(k, 0)
                                for d in sbr_results["sbr__party_wall"]
                            )
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__party_wall"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__party_wall_significant_orientations': [
                            dict(sorted({
                                k: sum(
                                    d.get(k, 0)
                                    for d in sbr_results["sbr__party_wall_significant_orientations"]
                                    if isinstance(d, dict)
                                )
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__party_wall_significant_orientations"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__party_wall_significant_orientations_by_floor': [
                            dict(sorted({
                                k: [
                                    sum(floor_vals)
                                    for floor_vals in zip_longest(
                                        *[
                                            d.get(k, []) for d in sbr_results["sbr__party_wall_significant_orientations_by_floor"]
                                            if isinstance(d, dict)
                                        ],
                                        fillvalue=0
                                    )
                                ]
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__party_wall_significant_orientations_by_floor"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__adiabatic_wall_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__adiabatic_wall_by_floor'], fillvalue=0)]],
                        'br__adiabatic_wall': [sum(sbr_results['sbr__adiabatic_wall'])],
                        'br__terrain_contact_wall_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__terrain_contact_wall_by_floor'], fillvalue=0)]],
                        'br__terrain_contact_wall': [sum(sbr_results['sbr__terrain_contact_wall'])],
                        'br__patios_wall_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_wall_by_floor'], fillvalue=0)]],
                        'br__patios_wall_total': [sum(sbr_results['sbr__patios_wall_total'])],
                        # Per-segment wall records concatenated across sbrs.
                        # Heavy column — same fetch model as br__shadows.
                        'br__wall_segments': [sum(sbr_results['sbr__wall_segments'].tolist(), start=[])],
                        # Combined per-floor shadows (building + terrain blocker
                        # at mid-floor eye level). Per-orientation lists are
                        # averaged element-wise across sbrs (missing floors
                        # treated as inf = no blocker). We use np.mean directly,
                        # not _safe_nanmean, because inf here is a meaningful
                        # "no blocker in range" signal — collapsing it to nan
                        # would lose information at the building grain.
                        'br__shadows_at_distance': [dict(sorted({
                            k: [
                                float(np.mean(list(x)))
                                for x in zip_longest(
                                    *(
                                        d.get(k, []) for d in sbr_results["sbr__shadows_at_distance"]
                                        if isinstance(d, dict)
                                    ),
                                    fillvalue=np.inf,
                                )
                            ]
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__shadows_at_distance"]
                                    if isinstance(d, dict)
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))] if "sbr__shadows_at_distance" in sbr_results.columns else [{}],
                        'br__building_contour_at_distance': [dict(sorted({
                            k: np.mean(
                                list(
                                    d.get(k, 0)
                                    for d in sbr_results["sbr__building_contour_at_distance"]
                                )
                            )
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__building_contour_at_distance"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        # New: per-floor blocked-region multipolygon.
                        # Aggregate across sbrs of this building_reference
                        # by unioning per floor. Geometries from sbrs of the
                        # same building cover overlapping disks, and unary
                        # union handles that cleanly.
                        'br__shadows': [{
                            f: utils.union_geoseries_with_tolerance(
                                [d.get(f) for d in sbr_results["sbr__shadows"]
                                 if isinstance(d, dict)
                                 and d.get(f) is not None
                                 and not d.get(f).is_empty],
                                gap_tolerance=0.05, resolution=4,
                            ) if any(
                                isinstance(d, dict)
                                and d.get(f) is not None
                                and not d.get(f).is_empty
                                for d in sbr_results["sbr__shadows"]
                            ) else Polygon()
                            for f in sorted(set(chain.from_iterable(
                                d.keys() for d in sbr_results["sbr__shadows"]
                                if isinstance(d, dict)
                            )))
                        }] if "sbr__shadows" in sbr_results.columns else [{}],
                        # 5 m DEM grid for contour-line rendering in the
                        # shadow plot. SBRs of a building share centroid area,
                        # so the first non-empty grid is representative.
                        'br__terrain_grid': [next(
                            (g for g in sbr_results["sbr__terrain_grid"]
                             if isinstance(g, dict) and g.get("elev")),
                            {},
                        )] if "sbr__terrain_grid" in sbr_results.columns else [{}],
                    }
                    br_results = pd.DataFrame(br_agg_dict)
                    geoms_by_floor = list(sbr_results['sbr__building_footprint_by_floor'])
                    building_gdf_by_floor = pd.concat([gpd.GeoDataFrame(geom_by_floor) for geom_by_floor in geoms_by_floor])

                    if len(building_gdf_by_floor)>0:

                        building_gdf_by_floor.columns = ["single_building_reference", "floor_number", "geometry"]
                        building_gdf_by_floor = building_gdf_by_floor.set_geometry("geometry")
                        building_gdf_by_floor["area_single_building_reference"] = building_gdf_by_floor.area
                        building_aggregates = (
                            building_gdf_by_floor.groupby("floor_number").agg(
                                n_single_building_references=("single_building_reference", "nunique"),
                                area_building_reference=("area_single_building_reference", "sum")
                            ))
                        building_gdf_by_floor = building_gdf_by_floor.merge(building_aggregates, on="floor_number", how="left")
                        building_gdf_by_floor["building_reference"] = building_reference

                        # Consider the information from CAT files
                        if buildings_CAT is not None:

                            # Integrate all CAT file information. buildings_CAT
                            # is a pandas DataFrame here (sliced for this zone
                            # in process_building_parts so workers stay polars-
                            # free and can run under `fork` safely).
                            agg_CAT = aggregate_CAT_file_building_spaces(
                                building_CAT_df=buildings_CAT[
                                    buildings_CAT["building_reference"] == building_reference
                                ])

                            building_gdf_by_floor = building_gdf_by_floor.drop(columns="geometry")

                            # Reduce agg_CAT["by_floors"] to each single building reference
                            sbr_with_CAT = building_gdf_by_floor.merge(agg_CAT["by_floors"],
                                                                            on=["floor_number", "building_reference"])
                            def process_variable(df, variable, ops, multiplier=1, divider=1):
                                return {
                                    f"sbr__{variable}_by_floor": utils.agg_op(
                                        df,
                                        funcs=ops,
                                        grouping=["use_type","floor_number"],
                                        variable=f"br__{variable}_by_floor",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }
                            agg_list = [("building_spaces",[lambda x: round(x, 0)],"area_single_building_reference","area_building_reference"),
                                        ("area_with_communals",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("area_without_communals",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("communal_area",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("economic_value",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("mean_building_space_area_with_communals",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_area_without_communals",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_communal_area",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_effective_year",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_category",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_economic_value",[lambda x: round(x, 0)],1,1),
                                        ]
                            sbr_with_CAT_by_floors = (
                                sbr_with_CAT.groupby(["single_building_reference","building_reference"])
                                .apply(lambda df: pd.Series({
                                        key: value
                                        for agg_elem in agg_list
                                        for key, value in
                                        process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3]).items()
                                    }),
                                    include_groups=False)
                                .reset_index()
                            )

                            # Reduce agg_CAT["building"] to each single building reference
                            sbr_with_CAT = building_gdf_by_floor.merge(agg_CAT["building"],
                                                                            on=["building_reference"])
                            def process_variable(df, variable, summarise_op=None, digits=0, multiplier=1, divider=1):
                                return {
                                    f"sbr__{variable}": utils.agg_op(
                                        df,
                                        funcs=[summarise_op, lambda x: round(x, digits) if summarise_op is not None else
                                            [lambda x: round(x, digits)]],
                                        grouping="use_type",
                                        variable=f"br__{variable}",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }
                            agg_list = [("building_spaces",sum,0,"area_single_building_reference","area_building_reference"),
                                        ("area_with_communals",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("area_without_communals",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("communal_area",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("economic_value",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("mean_building_space_area_with_communals",np.mean,0,1,1),
                                        ("mean_building_space_area_without_communals",np.mean,0,1,1),
                                        ("mean_building_space_communal_area",np.mean,0,1,1),
                                        ("mean_building_space_effective_year",np.mean,0,1,1),
                                        ("mean_building_space_category",np.mean,0,1,1),
                                        ("mean_building_space_economic_value",np.mean,0,1,1)
                                        ]
                            sbr_with_CAT_summarised = (
                                sbr_with_CAT.groupby(["single_building_reference","building_reference"])
                                .apply(lambda df: pd.Series({
                                    key: value
                                    for agg_elem in agg_list
                                    for key, value in process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3],
                                                                       agg_elem[4]).items()
                                }),
                                include_groups=False)
                                .reset_index()
                            )

                            # Reduce agg_CAT to per-building br__* columns (shared
                            # with the no-inference CAT path in mergers.py).
                            br_with_CAT = _br_aggregate_from_agg_CAT(agg_CAT)

                            sbr_results = (sbr_results.
                                merge(sbr_with_CAT_by_floors, on=["building_reference","single_building_reference"], how="left").
                                merge(sbr_with_CAT_summarised, on=["building_reference","single_building_reference"], how="left"))

                            br_results = br_results.merge(br_with_CAT, on="building_reference", how="left")

                    sbr_results_concat = pd.concat([sbr_results_concat, sbr_results], ignore_index=True)
                    br_results_concat = pd.concat([br_results_concat, br_results], ignore_index=True)

                # Apply municipality-level cleanups WRITE-SIDE so the
                # central DB never holds the unclean version, and Phase B
                # avoids a second pass. Both transforms are idempotent —
                # safe if shards happen to get re-ingested.
                sbr_results_concat = _strip_per_space_leak_columns(sbr_results_concat)
                br_results_concat = _strip_per_space_leak_columns(br_results_concat)
                br_results_concat = _dedup_br_building_spaces_lists(br_results_concat)

                # Atomic write: stage to {shard}.tmp and rename, so a
                # SIGKILL between connect() and final flush either leaves
                # the prior shard intact (if any) or leaves no shard at
                # all. Never a corrupt one at the canonical path.
                from hypercadaster_ES import _db as _db_mod
                shard_tmp = shard_path + ".tmp"
                if os.path.exists(shard_tmp):
                    os.remove(shard_tmp)
                shard_con = _db_mod.open_shard(shard_tmp)
                try:
                    _db_mod.write_df_safe(shard_con, "sbr_results",
                                          sbr_results_concat, if_exists="replace")
                    _db_mod.write_df_safe(shard_con, "br_results",
                                          br_results_concat, if_exists="replace")
                finally:
                    shard_con.close()
                os.replace(shard_tmp, shard_path)

                return sbr_results_concat, br_results_concat

            else:
                return None, None

        else:
            # Resume path: shard already on disk from a prior run.
            from hypercadaster_ES import _db as _db_mod
            try:
                _resume_con = _db_mod.open_shard_read(shard_path)
                try:
                    return (_db_mod.read_df_safe(_resume_con, "sbr_results"),
                            _db_mod.read_df_safe(_resume_con, "br_results"))
                finally:
                    _resume_con.close()
            except Exception as cache_err:
                sys.stderr.write(
                    f"Corrupt shard for zone '{zone_reference}' ({cache_err}); "
                    f"deleting and recomputing.\n"
                )
                try:
                    os.remove(shard_path)
                except OSError:
                    pass
                return process_zone(
                    gdf_zone, zone_reference, gdf_footprints_global, parcels_gdf,
                    streets_gdf, buildings_CAT, results_dir, plots,
                    orientation_discrete_interval_in_degrees, floor_height,
                    ground_floor_height, DEM_raster_dir,
                )

    except Exception as e:
        # Error handling moved to calling function with tqdm.write() for clean progress bar
        raise e

# ==========================================
# CAT FILE PROCESSING FUNCTIONS
# ==========================================

def parse_horizontal_division_buildings_CAT_files(cadaster_code, CAT_files_dir):
    """
    Parse CAT files to extract horizontal division building information.
    
    This function processes CAT files (Spanish cadastral data format) to extract
    detailed building information including spaces, floors, usage types, and
    geometric properties for horizontal property divisions.
    
    Parameters:
    -----------
    cadaster_code : str
        Municipality cadastral code
    CAT_files_dir : str
        Directory containing CAT files
        
    Returns:
    --------
    polars.DataFrame
        Processed building data with inferred space types, areas,
        floor information, and usage classifications
    """

    combined_dfs = downloaders.parse_CAT_file(cadaster_code, CAT_files_dir, allowed_dataset_types=[14, 15])

    if combined_dfs is None:
        return None

    # Join the building space and building spaces detailed datasets. Order the items by floor.
    building_spaces_detailed = combined_dfs[14]
    building_spaces = combined_dfs[15]
    building_spaces = building_spaces.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference"),
        pl.concat_str(['space2_reference', 'space3_reference'], separator=""
                      ).alias("building_space_reference_last_digits")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference")
    )
    building_spaces_detailed = building_spaces_detailed.filter(
        pl.col("distribution_method_for_communal_areas") == "")
    floor_names = sorted(building_spaces_detailed["building_space_floor_name"].unique().to_list())
    df = pd.DataFrame({'floor_name': floor_names})
    df['order'] = df['floor_name'].apply(utils.classify_cadaster_floor_names)
    floor_names_sorted = list(df.sort_values(by='order').reset_index(drop=True).floor_name)

    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces.select(
            'building_space_reference', 'building_space_reference_last_digits', 'building_space_year',
            'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
            'building_space_total_area', 'building_space_participation_rate', 'building_space_use_type'),
        on = "building_space_reference").select(
        'building_reference', 'building_space_reference', 'building_space_reference_last_digits',
        'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
        'building_space_block_name', 'building_space_stair_name', 'building_space_floor_name', 'building_space_door_name',
        'building_space_year', 'retrofitted', 'building_space_retroffiting_year', 'building_space_effective_year',
        'building_space_total_area', 'building_space_area_without_communal', 'building_space_area_balconies_terraces',
        'building_space_area_imputable_to_other_floors', 'building_space_participation_rate',
        'building_space_use_type', 'building_space_detailed_use_type', 'building_space_typology'
    )
    order_mapping = {value: index for index, value in enumerate(floor_names_sorted)}
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_floor_name").map_elements(lambda x: order_mapping.get(x, -1),
                                                         return_dtype=pl.Int32).alias("custom_sort")
    )
    building_spaces_detailed = building_spaces_detailed.sort("custom_sort").drop("custom_sort").sort("building_space_reference")

    # Calculate areas and affected communal areas per building space
    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces_detailed.group_by("building_space_reference").agg(
            pl.col("building_space_area_without_communal").sum().alias("building_space_total_area_without_communal"),
            pl.len().alias("building_spaces_considered"),
        ),
        on = "building_space_reference"
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_participation_rate") / 1000000).alias("building_space_participation_rate"),
        pl.concat_str(['building_space_reference', 'building_space_reference_last_digits'], separator=""
                      ).alias("building_space_reference"),
        ((pl.col("building_space_total_area") - pl.col("building_space_total_area_without_communal")) *
         pl.col("building_space_area_without_communal") / pl.col("building_space_total_area_without_communal")).alias(
            "building_space_communal_area"))
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_communal_area") + pl.col("building_space_area_without_communal")).alias(
            "building_space_area_with_communal")
    )

    # Rename items of categorical columns (Typology, category, use...)
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_typology").str.slice(4,length=3).alias("building_space_typology_category"),
        pl.col("building_space_typology").str.head(4).alias("building_space_typology_id"),
        (pl.lit(date.today().year) - pl.col("building_space_effective_year")).alias("building_space_age")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        # Create booster column based on category
        pl.when(pl.col("building_space_typology_category") == "A")
        .then(1.5)
        .when(pl.col("building_space_typology_category") == "B")
        .then(1.3)
        .when(pl.col("building_space_typology_category") == "C")
        .then(1.15)
        .otherwise(1.0)
        .alias("building_space_typology_category_booster")
    ).with_columns(
        # Create booster column based on category
        pl.when(pl.col("building_space_typology_category") == "A")
        .then(1)
        .when(pl.col("building_space_typology_category") == "B")
        .then(1)
        .when(pl.col("building_space_typology_category") == "C")
        .then(1)
        .when(pl.col("building_space_typology_category") == "O")
        .then(1)
        .otherwise(pl.col("building_space_typology_category"))
        .alias("building_space_typology_category")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_typology_id").replace(
            {k: v.get("Use") for k, v in downloaders.building_space_typologies.items() if "Use" in v}).
            alias("building_space_typology_use"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClass") for k, v in downloaders.building_space_typologies.items() if "UseClass" in v}).
            alias("building_space_typology_use_class"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClassModality") for k, v in downloaders.building_space_typologies.items() if "UseClassModality" in v}).
            alias("building_space_typology_use_class_modality"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseLevel") for k, v in downloaders.building_space_typologies.items() if "UseLevel" in v}).
            alias("building_space_typology_use_level"),
        pl.when(pl.col("building_space_age") >= 90).
            then(pl.lit(90).cast(pl.Int16).cast(pl.Utf8)).
            otherwise(((pl.col("building_space_age") / 5).floor() * 5).cast(pl.Int16).cast(pl.Utf8)).
            alias("building_space_age_key"),
        pl.col("building_space_use_type").replace(downloaders.building_space_use_types).
            alias("building_space_use_type"),
        pl.col("building_space_detailed_use_type").replace(downloaders.building_space_detailed_use_types).
            alias("building_space_detailed_use_type")
    )
    building_space_age_value_dict = \
        {str(entry["Age"][0]): {k: v for k, v in entry.items() if k != "Age"} for entry in downloaders.building_space_age_value}

    # Helper function to safely convert category to integer index
    def safe_category_to_index(category):
        try:
            return int(category) - 1
        except (ValueError, TypeError):
            return 0  # Default to first index for non-numeric categories

    # Economical value coefficients of the constructions
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.struct(["building_space_typology_id", "building_space_typology_category"]).map_elements(
            lambda row:
                downloaders.building_space_typologies.get(
                    row["building_space_typology_id"], 
                    downloaders.building_space_typologies.get("0000", {"ConstructionValue": [1.0]})
                )["ConstructionValue"][
                    safe_category_to_index(row["building_space_typology_category"])], # Safe index conversion
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("construction_relative_economic_value"),
        pl.struct(["building_space_age_key", "building_space_typology_use_level",
                   "building_space_typology_category"]).map_elements(
            lambda row:
            building_space_age_value_dict.get(
                row["building_space_age_key"], {}
            ).get(
                row["building_space_typology_use_level"], [1.0]
            )[safe_category_to_index(row["building_space_typology_category"])] if len(
                building_space_age_value_dict.get(
                    row["building_space_age_key"], {}
                ).get(
                    row["building_space_typology_use_level"], [1.0]
                )
            ) > safe_category_to_index(row["building_space_typology_category"]) else 1.0,  # Safe index conversion
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("age_correction_relative_economic_value")
    )

    # Calculate relative economic value
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("construction_relative_economic_value") *
         pl.col("age_correction_relative_economic_value") *
         pl.col("building_space_typology_category_booster")).alias("building_space_relative_economic_value")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_relative_economic_value") * pl.col("building_space_area_with_communal")).alias("building_space_economic_value")
    )

    # Calculate the complete address
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.when(pl.col("street_number1") == "0000").then(pl.lit("")).otherwise(
            pl.col("street_number1").cast(pl.Int16))).alias("street_number1"),
        (pl.when(pl.col("street_number2") == "0000").then(pl.lit("")).otherwise(
            pl.col("street_number2").cast(pl.Int16))).alias("street_number2"),
        (pl.when(pl.col("km") == "00000").then(pl.lit("")).otherwise(pl.col("km").cast(pl.Int16))).alias("km")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.concat_str(["street_type", "street_name", "street_number1", "street_letter1", "street_number2",
                       "street_letter2", "km", "building_space_block_name", "building_space_stair_name"],
                      separator=" ").
        str.strip_chars().str.replace_all(r"\s+", " ").alias("building_space_address"),
        pl.concat_str(['building_space_block_name', 'building_space_stair_name', 'building_space_floor_name',
                       'building_space_door_name'],
                      separator=" ").
        str.strip_chars().str.replace_all(r"\s+", " ").alias("building_space_inner_address")
    )

    use_type_mapping = {
        "Restaurant 4 Stars": "Leisure and Hospitality",
        "Professional education": "Cultural",
        "Water treatment plants": "Singular building",  # or whichever fallback you prefer
        "Other office activities": "Offices",
        "Weaver's office": "Offices",
        "Sports facilities": "Sports facilities",
        "Bookstore commerce": "Commercial",
        "Food industry": "Industrial",
        "Parking in a household": "Warehouse - Parking",
        "Commerce": "Commercial",
        "Religious": "Religious",
        "Open terrace (100%)": "Urbanization and landscaping works, undeveloped land",
        "Public town hall (<20,000)": "Offices",
        "Hotel Cafe 3 Stars": "Leisure and Hospitality",
        "Hostel Standard 1": "Leisure and Hospitality",
        "Hotel 5 Stars": "Leisure and Hospitality",
        "Station": "Singular building",
        "Financial commerce": "Commercial",
        "Club": "Entertainment venues",
        "Printing commerce": "Commercial",
        "Restaurant 5 Stars": "Leisure and Hospitality",
        "Musician's office": "Offices",
        "Hotel Cafe 5 Stars": "Leisure and Hospitality",
        "Luxury apartments 3 Stars": "Residential",
        "Clinic": "Healthcare and Charity",
        "Swimming pool": "Sports facilities",
        "University education": "Cultural",
        "Other provincial union": "Singular building",
        "Sports complex": "Sports facilities",
        "Stadium": "Sports facilities",
        "Education": "Cultural",
        "Luxury apartments 1 Star": "Residential",
        "Personal/Home commerce": "Commercial",
        "Other outpatient clinic": "Healthcare and Charity",
        "Nursing office": "Healthcare and Charity",
        "Hostel Standard 3": "Leisure and Hospitality",
        "Wholesale commerce": "Commercial",
        "Jewelry commerce": "Commercial",
        "Hotel 4 Stars": "Leisure and Hospitality",
        "Public courthouse": "Offices",
        "Education (Library)": "Cultural",
        "Silos, solid storage": "Warehouse - Parking",
        "Museum education": "Cultural",
        "Liquid storage tanks": "Warehouse - Parking",
        "Porch (100%)": "Urbanization and landscaping works, undeveloped land",
        "Galleries commerce": "Commercial",
        "Casino (>20,000)": "Entertainment venues",
        "Shoe commerce": "Commercial",
        "Machinery industry": "Industrial",
        "Wood industry": "Industrial",
        "Petroleum industry": "Industrial",
        "Parking": "Warehouse - Parking",
        "Transformer hut": "Singular building",
        "Warehouse": "Warehouse - Parking",
        "Other uses": "Singular building",
        "Bazaar commerce": "Commercial",
        "Provincial court": "Offices",
        "Industry": "Industrial",
        "Other dispensary": "Healthcare and Charity",
        "Textile industry": "Industrial",
        "Metal industry": "Industrial",
        "Other rescue facilities": "Singular building",
        "Plumbing commerce": "Commercial",
        "Public town hall (>20,000)": "Offices",
        "Entertainment": "Entertainment venues",
        "Chemical industry": "Industrial",
        "Casino (<20,000)": "Entertainment venues",
        "Drugstore commerce": "Commercial",
        "Colonnade (50%)": "Urbanization and landscaping works, undeveloped land",
        "Agent office": "Offices",
        "Construction industry": "Industrial",
        "Pharmacy commerce": "Commercial",
        "Government building": "Offices",
        "Undeveloped land": "Urbanization and landscaping works, undeveloped land",
        "Gas storage": "Warehouse - Parking",
        "Butcher commerce": "Commercial",
        "Daycare": "Healthcare and Charity",
        "Residence": "Residential",
        "Institute education": "Cultural",
        "Medical/Law office": "Offices",
        "Other sanatorium": "Healthcare and Charity",
        "Furniture commerce": "Commercial",
        "Public": "Singular building",
        "Hotel 1 Star": "Leisure and Hospitality",
        "Hotel": "Leisure and Hospitality",
        "Religious chapel": "Religious",
        "Hostel Standard 2": "Leisure and Hospitality",
        "Housing": "Residential",
        "Restaurant 2 Stars": "Leisure and Hospitality",
        "Cinema": "Entertainment venues",
        "Hotel Cafe 2 Stars": "Leisure and Hospitality",
        "Restaurant 1 Star": "Leisure and Hospitality",
        "Retail commerce": "Commercial",
        "Hotel 3 Stars": "Leisure and Hospitality",
        "Entertainment hall": "Entertainment venues",
        "Auditorium": "Entertainment venues",
        "Salesperson office": "Offices",
        "Sports": "Sports facilities",
        "Hotel 2 Stars": "Leisure and Hospitality",
        "Electric industry": "Industrial",
        "Office": "Offices",
        "Watchmaking commerce": "Commercial",
        "Hotel Cafe 4 Stars": "Leisure and Hospitality",
        "Theater": "Entertainment venues",
        "Private garden (100%)": "Urbanization and landscaping works, undeveloped land",
        "Hotel Cafe 1 Star": "Leisure and Hospitality",
        "Restaurant 3 Stars": "Leisure and Hospitality",
        "Religious hermitage": "Religious",
        "Basic education": "Cultural",
        "Urbanization works": "Urbanization and landscaping works, undeveloped land",
        "Perfumery commerce": "Commercial",
        "Automobile commerce": "Commercial",
        "Hospital": "Healthcare and Charity",
        "Religious parish": "Religious",
        "Glass industry": "Industrial",
        "Public delegation": "Offices",
        "Fabric commerce": "Commercial",
        "Storage": "Warehouse - Parking",
        "Supermarket commerce": "Commercial",
        "Manufacturing industry": "Industrial",
        "Covered terrace (100%)": "Urbanization and landscaping works, undeveloped land",
        "Cultural house education": "Cultural",
    }
    mapper = pl.DataFrame([{'keys': x, 'values': y} for x, y in use_type_mapping.items()])

    building_spaces_detailed = building_spaces_detailed.with_columns(
        building_spaces_detailed["building_space_detailed_use_type"].
        to_frame("keys").join(mapper, on="keys", how="left").
        to_series(1).alias("building_space_inferred_use_type"))

    return building_spaces_detailed


def aggregate_CAT_file_building_spaces(building_CAT_df):
    """
    Aggregate building space information from CAT file data.

    Pandas implementation (called from inside the per-zone parallel worker —
    keeping it polars-free is what lets the worker pool run under `fork`
    without deadlocking on polars' Rayon thread pool).

    Parameters:
    -----------
    building_CAT_df : pandas.DataFrame
        Detailed building space data from CAT files (per-zone slice).

    Returns:
    --------
    dict[str, pandas.DataFrame]
        ``{"by_floors": ..., "building": ...}`` — both pandas DataFrames.
    """
    df = building_CAT_df.copy()

    # Raw classify outputs include fractional markers (0.5 for ground floor,
    # 0.25 increments for PR-prefixed floors) — these are *ordering* values
    # that the compaction step below converts to consecutive integers.
    def _classify_above_as_float(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        v = utils.classify_above_ground_floor_names(x)
        return np.nan if v is None else float(v)

    def _classify_below_as_float(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        v = utils.classify_below_ground_floor_names(x)
        return np.nan if v is None else float(v)

    df["building_space_floor_order_above_ground"] = (
        df["building_space_floor_name"].map(_classify_above_as_float).astype(float)
    )
    df["building_space_floor_order_below_ground"] = (
        df["building_space_floor_name"].map(_classify_below_as_float).astype(float)
    )

    # Map raw fractional ordering values to consecutive integers.
    above_unique = sorted(v for v in df["building_space_floor_order_above_ground"].dropna().unique())
    value_to_order_above = {val: i for i, val in enumerate(above_unique)}
    df["building_space_floor_order_above_ground"] = (
        df["building_space_floor_order_above_ground"]
        .map(lambda x: value_to_order_above.get(x) if pd.notna(x) else None)
        .astype("Int32")
    )

    below_unique = sorted(v for v in df["building_space_floor_order_below_ground"].dropna().unique())
    value_to_order_below = {val: -(i + 1) for i, val in enumerate(below_unique)}
    df["building_space_floor_order_below_ground"] = (
        df["building_space_floor_order_below_ground"]
        .map(lambda x: value_to_order_below.get(x) if pd.notna(x) else None)
        .astype("Int32")
    )

    # Join below/above orders into a single floor order.
    df["building_space_floor_order"] = df["building_space_floor_order_above_ground"].where(
        df["building_space_floor_order_above_ground"].notna(),
        df["building_space_floor_order_below_ground"],
    )

    # Per-address, compact non-consecutive floor orders into consecutive integers
    # while preserving the minimum value. Example: [3, 5, 7, 7, 5] → [3, 4, 5, 5, 4].
    grouped_addr = df.groupby("building_space_address")["building_space_floor_order"]
    dense_rank = grouped_addr.rank(method="dense").astype("Int32")
    min_per_addr = grouped_addr.transform("min").astype("Int32")
    df["building_space_floor_number"] = (dense_rank - 1 + min_per_addr).astype("Int32")

    # Building-floor and per-building total areas.
    df["building_floor_total_area"] = df.groupby("building_space_floor_number")[
        "building_space_area_with_communal"
    ].transform("sum")
    df["building_total_area"] = df.groupby("building_reference")[
        "building_space_area_with_communal"
    ].transform("sum")

    df["building_space_area_without_communal"] = df["building_space_area_without_communal"].round(2)
    df["building_space_area_with_communal"] = df["building_space_area_with_communal"].round(2)
    df["building_space_economic_value"] = df["building_space_economic_value"].round(2)

    df = df.sort_values(["building_space_address", "building_space_inner_address"], kind="stable")

    category_map = {"A": "1", "B": "2", "C": "3", "0": "1", "O": "1"}
    # Plain float (not nullable Int8/Float64) so that .mean() on an all-NA
    # group returns numpy NaN — Python's `round` chokes on pandas NAType but
    # handles float NaN fine, matching the polars null-propagation semantics.
    df["_typ_cat_int"] = pd.to_numeric(
        df["building_space_typology_category"].map(lambda v: category_map.get(v, v)),
        errors="coerce",
    ).astype(float)

    def _safe_round(x, digits):
        # Polars-style null propagation: pd.NA / None / NaN → NaN, no TypeError.
        if x is None or pd.isna(x):
            return float("nan")
        return round(float(x), digits)

    def _safe_div_round(num, den, digits):
        # Mirrors polars `(num / den).round(digits)` with null propagation:
        # divide by 0 or NaN denominator yields NaN, never warns.
        if num is None or pd.isna(num) or den is None or pd.isna(den) or float(den) == 0.0:
            return float("nan")
        return round(float(num) / float(den), digits)

    def _by_floors(group):
        floor_total_area = group["building_floor_total_area"].mean()
        out = {
            "building_reference": group["building_reference"].iloc[0],
            "br__building_spaces_by_floor": int(group["building_space_reference"].count()),
            "br__mean_building_space_area_with_communals_by_floor": _safe_round(group["building_space_area_with_communal"].mean(), 2),
            "br__mean_building_space_area_without_communals_by_floor": _safe_round(group["building_space_area_without_communal"].mean(), 2),
            "br__mean_building_space_communal_area_by_floor": _safe_round(group["building_space_communal_area"].mean(), 2),
            "br__mean_building_space_effective_year_by_floor": _safe_round(group["building_space_effective_year"].mean(), 1),
            "br__mean_building_space_category_by_floor": _safe_round(group["_typ_cat_int"].mean(), 1),
            "br__mean_building_space_economic_value_by_floor": _safe_round(group["building_space_economic_value"].mean(), 2),
            "br__area_with_communals_by_floor": group["building_space_area_with_communal"].sum(),
            "br__area_without_communals_by_floor": group["building_space_area_without_communal"].sum(),
            "br__communal_area_by_floor": group["building_space_communal_area"].sum(),
            "br__economic_value_by_floor": group["building_space_economic_value"].sum(),
            "br__relative_area_with_communals_by_floor": _safe_div_round(group["building_space_area_with_communal"].sum(), floor_total_area, 2) * 100,
            "br__relative_area_without_communals_by_floor": _safe_div_round(group["building_space_area_without_communal"].sum(), floor_total_area, 2) * 100,
            "br__relative_communal_area_by_floor": _safe_div_round(group["building_space_communal_area"].sum(), floor_total_area, 2) * 100,
        }
        return pd.Series(out)

    agg_CAT_by_floors = (
        df.groupby(["building_space_floor_number", "building_space_inferred_use_type"], dropna=False)
        .apply(_by_floors, include_groups=False)
        .reset_index()
        .rename(columns={
            "building_space_floor_number": "floor_number",
            "building_space_inferred_use_type": "use_type",
        })
        .sort_values("floor_number", kind="stable")
        .reset_index(drop=True)
    )

    def _by_building(group):
        total_area = group["building_total_area"].mean()
        # CAT splits a single space across the floors it occupies, so the
        # same building_space_reference shows up once per floor with split
        # area/economic value. Fold those back into one row per space:
        # additive fields summed, descriptive fields kept from the first row.
        additive = {
            "building_space_area_without_communal": "sum",
            "building_space_area_with_communal": "sum",
            "building_space_communal_area": "sum",
            "building_space_economic_value": "sum",
        }
        first_fields = {
            "building_space_address": "first",
            "building_space_inner_address": "first",
            "building_space_floor_number": "first",
            "_typ_cat_int": "first",
            "building_space_effective_year": "first",
            "building_space_detailed_use_type": "first",
        }
        agg_spec = {**additive, **first_fields}
        agg_spec = {c: how for c, how in agg_spec.items() if c in group.columns}
        dedup = (group.groupby("building_space_reference", sort=False, dropna=False)
                       .agg(agg_spec)
                       .reset_index())
        out = {
            "building_reference": group["building_reference"].iloc[0],
            "br__building_spaces": int(len(dedup)),
            "br__floors": sorted(pd.unique(dedup["building_space_floor_number"].dropna()).tolist()),
            "br__mean_building_space_area_with_communals": _safe_round(dedup["building_space_area_with_communal"].mean(), 2),
            "br__mean_building_space_area_without_communals": _safe_round(dedup["building_space_area_without_communal"].mean(), 2),
            "br__mean_building_space_communal_area": _safe_round(dedup["building_space_communal_area"].mean(), 2),
            "br__mean_building_space_effective_year": _safe_round(dedup["building_space_effective_year"].mean(), 1),
            "br__mean_building_space_category": _safe_round(dedup["_typ_cat_int"].mean(), 1),
            "br__mean_building_space_economic_value": _safe_round(dedup["building_space_economic_value"].mean(), 2),
            "br__area_with_communals": dedup["building_space_area_with_communal"].sum(),
            "br__area_without_communals": dedup["building_space_area_without_communal"].sum(),
            "br__communal_area": dedup["building_space_communal_area"].sum(),
            "br__economic_value": dedup["building_space_economic_value"].sum(),
            "br__relative_area_with_communals": _safe_div_round(dedup["building_space_area_with_communal"].sum(), total_area, 2) * 100,
            "br__relative_area_without_communals": _safe_div_round(dedup["building_space_area_without_communal"].sum(), total_area, 2) * 100,
            "br__relative_communal_area": _safe_div_round(dedup["building_space_communal_area"].sum(), total_area, 2) * 100,
            "br__building_spaces_reference": list(dedup["building_space_reference"]),
            "br__building_spaces_postal_address": list(dedup["building_space_address"]),
            "br__building_spaces_inner_address": list(dedup["building_space_inner_address"]),
            "br__building_spaces_floor_number": list(dedup["building_space_floor_number"]),
            "br__building_spaces_category": list(dedup["_typ_cat_int"]),
            "br__building_spaces_effective_year": list(dedup["building_space_effective_year"]),
            "br__building_spaces_area_without_communal": list(dedup["building_space_area_without_communal"]),
            "br__building_spaces_area_with_communal": list(dedup["building_space_area_with_communal"]),
            "br__building_spaces_economic_value": list(dedup["building_space_economic_value"]),
            "br__building_spaces_detailed_use_type": list(dedup["building_space_detailed_use_type"]),
        }
        return pd.Series(out)

    agg_CAT_building = (
        df.groupby("building_space_inferred_use_type", dropna=False)
        .apply(_by_building, include_groups=False)
        .reset_index()
        .rename(columns={"building_space_inferred_use_type": "use_type"})
    )

    return {
        "by_floors": agg_CAT_by_floors,
        "building": agg_CAT_building,
    }


def _br_aggregate_from_agg_CAT(agg_CAT):
    """Apply the br-level aggregation over the dict produced by
    ``aggregate_CAT_file_building_spaces``.

    Returns a DataFrame keyed by ``building_reference`` carrying the
    ``br__*_by_floor`` and ``br__*`` columns. Extracted from the inline
    block in ``process_zone`` so the no-inference CAT path in
    ``mergers.py`` can reuse exactly the same aggregation.
    """
    def _process_by_floors(df, variable, ops, multiplier=1, divider=1):
        return {
            f"br__{variable}_by_floor": utils.agg_op(
                df,
                funcs=ops,
                grouping=["use_type", "floor_number"],
                variable=f"br__{variable}_by_floor",
                multiplier=multiplier,
                divider=divider,
            )
        }

    floors_agg_list = [
        ("building_spaces", [lambda x: round(x, 2)], 1, 1),
        ("area_with_communals", [lambda x: round(x, 2)], 1, 1),
        ("area_without_communals", [lambda x: round(x, 2)], 1, 1),
        ("communal_area", [lambda x: round(x, 2)], 1, 1),
        ("economic_value", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_area_with_communals", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_area_without_communals", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_communal_area", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_effective_year", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_category", [lambda x: round(x, 2)], 1, 1),
        ("mean_building_space_economic_value", [lambda x: round(x, 2)], 1, 1),
    ]
    br_by_floors = (
        agg_CAT["by_floors"].groupby(["building_reference"])
        .apply(lambda df: pd.Series({
            key: value
            for agg_elem in floors_agg_list
            for key, value in _process_by_floors(df, agg_elem[0], agg_elem[1],
                                                 agg_elem[2], agg_elem[3]).items()
        }), include_groups=False)
        .reset_index()
    )

    def _process_building(df, variable, ops, multiplier=1, divider=1):
        return {
            f"br__{variable}": utils.agg_op(
                df,
                funcs=ops,
                grouping="use_type",
                variable=f"br__{variable}",
                multiplier=multiplier,
                divider=divider,
            )
        }

    building_agg_list = [
        ("building_spaces", [sum], 1, 1),
        ("area_with_communals", [sum], 1, 1),
        ("area_without_communals", [sum], 1, 1),
        ("communal_area", [sum], 1, 1),
        ("economic_value", [sum], 1, 1),
        ("mean_building_space_area_with_communals", [sum], 1, 1),
        ("mean_building_space_area_without_communals", [sum], 1, 1),
        ("mean_building_space_communal_area", [sum], 1, 1),
        ("mean_building_space_effective_year", [sum], 1, 1),
        ("mean_building_space_category", [sum], 1, 1),
        ("mean_building_space_economic_value", [sum], 1, 1),
        ('building_spaces_reference', [chain.from_iterable, list], 1, 1),
        ('building_spaces_postal_address', [chain.from_iterable, list], 1, 1),
        ('building_spaces_inner_address', [chain.from_iterable, list], 1, 1),
        ('building_spaces_floor_number', [chain.from_iterable, list], 1, 1),
        ('building_spaces_category', [chain.from_iterable, list], 1, 1),
        ('building_spaces_effective_year', [chain.from_iterable, list], 1, 1),
        ('building_spaces_area_without_communal', [chain.from_iterable, list], 1, 1),
        ('building_spaces_area_with_communal', [chain.from_iterable, list], 1, 1),
        ('building_spaces_economic_value', [chain.from_iterable, list], 1, 1),
        ('building_spaces_detailed_use_type', [chain.from_iterable, list], 1, 1),
    ]
    br_building = (
        agg_CAT["building"].groupby(["building_reference"])
        .apply(lambda df: pd.Series({
            key: value
            for agg_elem in building_agg_list
            for key, value in _process_building(df, agg_elem[0], agg_elem[1],
                                                 agg_elem[2], agg_elem[3]).items()
        }), include_groups=False)
        .reset_index()
    )

    return br_by_floors.merge(br_building, on="building_reference", how="outer")


def br_attributes_from_CAT(buildings_CAT):
    """Compute the CAT-derived ``br__*`` attributes for every building in
    ``buildings_CAT`` without running geometric building-parts inference.

    Used by ``hc.merge(building_parts_inference=False, use_CAT_files=True)``
    so the resulting GeoDataFrame still carries the rich per-building CAT
    aggregates (``br__building_spaces_*``, ``br__area_with_communals``,
    ``br__economic_value``, ...) that the full inference path produces.

    Parameters
    ----------
    buildings_CAT : polars.DataFrame or pandas.DataFrame
        Per-building-space CAT rows, the same frame produced by
        ``parse_horizontal_division_buildings_CAT_files``.

    Returns
    -------
    pandas.DataFrame keyed by ``building_reference`` with the same
    ``br__*_by_floor`` and ``br__*`` columns the inference path emits.
    """
    if buildings_CAT is None:
        return pd.DataFrame(columns=["building_reference"])
    if hasattr(buildings_CAT, "to_pandas"):
        df = buildings_CAT.to_pandas()
    else:
        df = buildings_CAT

    if df.empty:
        return pd.DataFrame(columns=["building_reference"])

    rows = []
    for _br, sub in df.groupby("building_reference", sort=False):
        agg_CAT = aggregate_CAT_file_building_spaces(building_CAT_df=sub)
        rows.append(_br_aggregate_from_agg_CAT(agg_CAT))

    if not rows:
        return pd.DataFrame(columns=["building_reference"])

    out = pd.concat(rows, ignore_index=True)
    out = _strip_per_space_leak_columns(out)
    out = _dedup_br_building_spaces_lists(out)
    return out