"""DuckDB persistence layer for hypercadaster_ES.

Replaces the per-municipality / per-zone pickle artifacts produced by
`building_inference.py` and `mergers.py` with a single DuckDB database
per `results_dir`. Geometry is stored as native GEOMETRY via the spatial
extension. See DESIGN_duckdb_migration.md for the overall design.

This module is foundations only: it owns connection lifecycle, schema
DDL for the structural tables, CRS sidecar storage, and helpers to move
GeoDataFrames in/out of the DB. Domain tables that grow columns over
time (sbr_results, br_results, buildings, ...) are extended lazily by
their respective callers using `write_gdf` / `append_gdf`.
"""

from __future__ import annotations

import os
import pickle as _pickle
import sys
from typing import Iterable, List, Optional, Tuple

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkb as _shapely_wkb
from shapely.geometry.base import BaseGeometry as _BaseGeometry


DB_FILENAME = "building_results.duckdb"

# Columns whose pickled BLOB payload is enormous on Barcelona-scale data
# (shadows are each ~11 GB on disk; wall_segments adds ~6 GB for ~150k
# sbrs × ~300 segments). Excluded from default Phase B materialisation;
# fetched on demand by `get_blob_for_building`. Wall segments are also
# how the per-building energy-simulation interop pulls (geometry, type,
# floor, orientation_degrees) records — see
# `building_inference.process_zone` for the schema.
HEAVY_BLOB_COLS = frozenset({
    "sbr__shadows", "sbr__shadows_at_distance", "sbr__terrain_grid",
    "br__shadows",  "br__shadows_at_distance",  "br__terrain_grid",
    "sbr__wall_segments", "br__wall_segments",
})

# The 12-column projection that `read_addresses_indicators` returns when
# the caller didn't run building-parts inference. Mirrors the strip-down
# in mergers.py around the `not building_parts_inference` branch.
BASIC_BUILDING_COLUMNS = (
    "gml_id", "building_status", "building_reference", "building_use",
    "building_geometry", "year_of_construction", "n_building_units",
    "n_dwellings", "n_floors_above_ground", "building_area",
    "zone_reference", "zone_type",
)


def db_path(results_dir: str) -> str:
    """Return the path to the central DuckDB file inside ``results_dir``."""
    return os.path.join(results_dir, DB_FILENAME)


def open_db(results_dir: str, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB file under results_dir.

    The spatial extension is loaded on every connection so GEOMETRY and
    ST_* are available without further setup. On a fresh machine the
    first INSTALL downloads the extension from DuckDB's repo; subsequent
    calls are no-ops.
    """
    os.makedirs(results_dir, exist_ok=True)
    con = duckdb.connect(db_path(results_dir), read_only=read_only)
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    _silence_progress_bar(con)
    if not read_only:
        ensure_schema(con)
    return con


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the structural tables. Domain tables are added lazily by
    their writers."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS _meta (
            key   VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS zone_status (
            cadaster_code   VARCHAR NOT NULL,
            zone_reference  VARCHAR NOT NULL,
            state           VARCHAR NOT NULL,
            finished_at     TIMESTAMP DEFAULT now(),
            error           VARCHAR,
            PRIMARY KEY (cadaster_code, zone_reference)
        )
    """)


# ---------------------------------------------------------------------------
# CAT-file attribute cache — the parsed per-space building table
# ---------------------------------------------------------------------------
# Reading + transforming the CAT files (parse_horizontal_division_buildings_
# CAT_files) is the dominant fixed cost of a no-inference CAT merge — ~1 h for a
# whole province. Its output (one row per building space, all primitive columns,
# no geometry) is cached here keyed by cadaster_code so re-runs skip the parse.
CAT_SPACES_TABLE = "cat_building_spaces"


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone() is not None


def cat_cache_read(results_dir: str, cadaster_code: str):
    """Return the cached CAT building-space frame for one municipality, or None.

    Reads back the exact per-space table produced by
    ``parse_horizontal_division_buildings_CAT_files`` and stored by
    :func:`cat_cache_write`, as a polars DataFrame (``cadaster_code`` column
    dropped). Returns ``None`` when the cache table or the municipality's rows are
    absent — or on any error, so a stale/corrupt cache degrades to a re-parse
    instead of breaking the merge.
    """
    path = db_path(results_dir)
    if not os.path.exists(path):
        return None
    con = None
    try:
        con = duckdb.connect(path, read_only=True)
        if not _table_exists(con, CAT_SPACES_TABLE):
            return None
        # Relation .pl(): duckdb 2.0 dropped .pl() on the DB-API connection.
        df = con.sql(
            f'SELECT * FROM "{CAT_SPACES_TABLE}" WHERE cadaster_code = ?',
            params=[str(cadaster_code)],
        ).pl()
        if df.height == 0:
            return None
        return df.drop("cadaster_code")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"CAT cache read failed for {cadaster_code} ({e}); re-parsing\n")
        return None
    finally:
        if con is not None:
            con.close()


def cat_cache_clear(results_dir: str, cadaster_code: Optional[str] = None) -> int:
    """Invalidate the CAT caches so the next merge re-parses and re-aggregates.

    Clears both CAT caches (``cat_building_spaces`` per-space parse and
    ``cat_building_attributes`` per-building aggregates): the rows for one
    ``cadaster_code`` (or every row / the whole tables when ``cadaster_code is
    None``). Returns the total number of rows removed across both tables (0 when
    neither exists). Use after replacing the CAT files, since the caches are
    otherwise reused regardless of ``force_update``.
    """
    path = db_path(results_dir)
    if not os.path.exists(path):
        return 0
    con = None
    removed = 0
    try:
        con = open_db(results_dir)
        for tbl in (CAT_SPACES_TABLE, CAT_ATTRS_TABLE):
            if not _table_exists(con, tbl):
                continue
            if cadaster_code is None:
                removed += con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
                con.execute(f'DROP TABLE "{tbl}"')
            else:
                removed += con.execute(
                    f'SELECT COUNT(*) FROM "{tbl}" WHERE cadaster_code = ?',
                    [str(cadaster_code)]).fetchone()[0]
                con.execute(
                    f'DELETE FROM "{tbl}" WHERE cadaster_code = ?',
                    [str(cadaster_code)])
        return int(removed)
    finally:
        if con is not None:
            con.close()


def cat_cache_write(results_dir: str, cadaster_code: str, buildings_cat) -> None:
    """Store one municipality's parsed CAT building-space frame in the central DB.

    Idempotent per ``cadaster_code``: any existing rows for the municipality are
    replaced (so a re-parse refreshes the cache). The table is created on first
    use and its schema extended (``ADD COLUMN``) if a later municipality carries a
    column not seen before. ``buildings_cat`` is the polars frame returned by
    ``parse_horizontal_division_buildings_CAT_files``. Any failure is swallowed
    with a warning — caching is an optimisation and must never break the merge.
    """
    import polars as pl

    if buildings_cat is None or getattr(buildings_cat, "height", 0) == 0:
        return
    code = str(cadaster_code)
    con = None
    try:
        df = buildings_cat
        if "cadaster_code" in df.columns:
            df = df.drop("cadaster_code")
        df = df.with_columns(pl.lit(code).alias("cadaster_code"))

        con = open_db(results_dir)
        con.register("_cat_cache_tmp", df.to_arrow())
        if not _table_exists(con, CAT_SPACES_TABLE):
            con.execute(
                f'CREATE TABLE "{CAT_SPACES_TABLE}" AS SELECT * FROM _cat_cache_tmp'
            )
        else:
            existing = {r[0] for r in con.execute(
                f'DESCRIBE "{CAT_SPACES_TABLE}"').fetchall()}
            for name, ty in ((r[0], r[1]) for r in
                             con.execute('DESCRIBE _cat_cache_tmp').fetchall()):
                if name not in existing:
                    con.execute(
                        f'ALTER TABLE "{CAT_SPACES_TABLE}" ADD COLUMN "{name}" {ty}')
            con.execute(
                f'DELETE FROM "{CAT_SPACES_TABLE}" WHERE cadaster_code = ?', [code])
            con.execute(
                f'INSERT INTO "{CAT_SPACES_TABLE}" BY NAME SELECT * FROM _cat_cache_tmp')
        con.unregister("_cat_cache_tmp")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"CAT cache write failed for {cadaster_code} ({e}); "
                         f"continuing without caching\n")
    finally:
        if con is not None:
            con.close()


# ---------------------------------------------------------------------------
# CAT per-building attribute cache — the expensive br_attributes_from_CAT output
# ---------------------------------------------------------------------------
# The real cost of a no-inference CAT merge is not the CAT parse (~1 s even for a
# 160k-space municipality) but ``br_attributes_from_CAT`` — a per-building
# aggregation that runs for minutes on a large municipality. Its output (one row
# per building, with many complex ``br__*`` dict/list columns) is cached here,
# keyed by cadaster_code, and reused on every run. Complex columns are pickled to
# BLOB by ``write_df_safe`` and rehydrated by ``read_df_safe``.
CAT_ATTRS_TABLE = "cat_building_attributes"


def cat_attrs_read(results_dir: str, cadaster_code: str):
    """Return the cached per-building CAT attribute frame for a municipality.

    The pandas frame produced by ``br_attributes_from_CAT`` (``cadaster_code``
    column dropped), or ``None`` when absent — or on any error, so a stale/corrupt
    cache degrades to a recompute instead of breaking the merge.
    """
    path = db_path(results_dir)
    if not os.path.exists(path):
        return None
    con = None
    try:
        con = open_shard_read(path)
        if not _table_exists(con, CAT_ATTRS_TABLE):
            return None
        df = read_df_safe(con, CAT_ATTRS_TABLE,
                          where="cadaster_code = ?", params=[str(cadaster_code)])
        if df is None or len(df) == 0:
            return None
        return df.drop(columns=["cadaster_code"])
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"CAT attrs cache read failed for {cadaster_code} "
                         f"({e}); recomputing\n")
        return None
    finally:
        if con is not None:
            con.close()


def cat_attrs_write(results_dir: str, cadaster_code: str, br_df) -> None:
    """Store one municipality's per-building CAT attribute frame in the central DB.

    Idempotent per ``cadaster_code`` (existing rows are replaced). ``br_df`` is the
    pandas frame returned by ``br_attributes_from_CAT``; its complex ``br__*``
    columns are pickled by ``write_df_safe``. Any failure is swallowed with a
    warning — caching must never break the merge.
    """
    if br_df is None or len(br_df) == 0:
        return
    code = str(cadaster_code)
    con = None
    try:
        df = br_df.copy()
        df["cadaster_code"] = code
        con = open_db(results_dir)
        if not _table_exists(con, CAT_ATTRS_TABLE):
            write_df_safe(con, CAT_ATTRS_TABLE, df, if_exists="replace")
        else:
            # Align to the existing column set/order so the positional INSERT in
            # write_df_safe(append) stays correct even if a municipality's frame
            # is ordered differently or misses a column.
            existing = [r[0] for r in con.execute(
                f'DESCRIBE "{CAT_ATTRS_TABLE}"').fetchall()]
            for c in existing:
                if c not in df.columns:
                    df[c] = None
            extra = [c for c in df.columns if c not in existing]
            if extra:
                # Unexpected new column(s): fall back to a full rewrite so nothing
                # is silently dropped (schema stays a superset).
                con.execute(
                    f'DELETE FROM "{CAT_ATTRS_TABLE}" WHERE cadaster_code = ?', [code])
                prev = read_df_safe(con, CAT_ATTRS_TABLE)
                combined = pd.concat([prev, df], ignore_index=True)
                write_df_safe(con, CAT_ATTRS_TABLE, combined, if_exists="replace")
            else:
                df = df[existing]
                con.execute(
                    f'DELETE FROM "{CAT_ATTRS_TABLE}" WHERE cadaster_code = ?', [code])
                write_df_safe(con, CAT_ATTRS_TABLE, df, if_exists="append")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"CAT attrs cache write failed for {cadaster_code} "
                         f"({e}); continuing without caching\n")
    finally:
        if con is not None:
            con.close()


# ---------------------------------------------------------------------------
# CRS sidecar — the spatial extension does not store per-table CRS itself
# ---------------------------------------------------------------------------

def set_table_crs(con: duckdb.DuckDBPyConnection, table: str, crs) -> None:
    """Persist a table's CRS (as EPSG) in the ``_meta`` sidecar table.

    No-ops when ``crs`` is None or has no resolvable EPSG code.
    """
    if crs is None:
        return
    try:
        epsg = crs.to_epsg()
    except AttributeError:
        from pyproj import CRS as _CRS
        epsg = _CRS.from_user_input(crs).to_epsg()
    if epsg is None:
        return
    con.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
        [f"crs:{table}", str(int(epsg))],
    )


def get_table_crs(con: duckdb.DuckDBPyConnection, table: str):
    """Return the stored ``pyproj.CRS`` for a table, or None if unset."""
    row = con.execute(
        "SELECT value FROM _meta WHERE key = ?", [f"crs:{table}"]
    ).fetchone()
    if row is None:
        return None
    from pyproj import CRS as _CRS
    return _CRS.from_epsg(int(row[0]))


# ---------------------------------------------------------------------------
# zone_status — replaces per-zone pickle existence checks
# ---------------------------------------------------------------------------

def mark_zone(con: duckdb.DuckDBPyConnection, cadaster_code: str,
              zone_reference: str, state: str,
              error: Optional[str] = None) -> None:
    """Upsert a zone's processing state into the ``zone_status`` ledger.

    Records ``state`` (e.g. ``'ok'`` / ``'error'``) and optional ``error`` text
    for ``(cadaster_code, zone_reference)``, refreshing ``finished_at``. This
    ledger drives resume-on-rerun (see :func:`completed_zones`).
    """
    con.execute(
        """
        INSERT INTO zone_status (cadaster_code, zone_reference, state, error)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (cadaster_code, zone_reference) DO UPDATE SET
            state = excluded.state,
            error = excluded.error,
            finished_at = now()
        """,
        [cadaster_code, zone_reference, state, error],
    )


def completed_zones(con: duckdb.DuckDBPyConnection, cadaster_code: str) -> set:
    """Return the set of ``zone_reference`` values marked ``'ok'`` for a
    municipality, so a re-run can skip already-processed zones."""
    rows = con.execute(
        "SELECT zone_reference FROM zone_status "
        "WHERE cadaster_code = ? AND state = 'ok'",
        [cadaster_code],
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# (Geo)DataFrame ↔ DuckDB
# ---------------------------------------------------------------------------

def _df_with_wkb(gdf: gpd.GeoDataFrame, geom_cols: Iterable[str]) -> pd.DataFrame:
    """Return a shallow-copied pandas DataFrame where each geometry column
    has been replaced by its WKB bytes, and complex object columns are
    pickled into BLOB bytes. DuckDB's ST_GeomFromWKB rehydrates geometry;
    `_maybe_unpickle` rehydrates the BLOB columns on read."""
    geom_set = set(geom_cols)
    df = pd.DataFrame(gdf)
    for col in geom_set:
        if col not in df.columns:
            continue
        df = df.assign(**{col: gpd.GeoSeries(df[col]).to_wkb()})
    # Re-pickle any object column whose sample value is a complex Python
    # object (dict, list, numpy scalar, etc.). Strings and bytes pass
    # through as-is; they're already DuckDB-compatible. This handles
    # round-trips through read_gdf which unpickles BLOB columns on read.
    for col in df.columns:
        if col in geom_set or df[col].dtype != object:
            continue
        sample = _first_non_null(df[col])
        if sample is None or isinstance(sample, (str, bytes, bytearray, memoryview)):
            continue
        df[col] = [None if _is_null(v) else _to_pickle_bytes(v) for v in df[col]]
    return df


def _geom_select_sql(columns: Iterable[str], geom_cols: Iterable[str]) -> str:
    parts = []
    geom_set = set(geom_cols)
    for c in columns:
        if c in geom_set:
            parts.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
        else:
            parts.append(f'"{c}"')
    return ", ".join(parts)


def write_gdf(con: duckdb.DuckDBPyConnection, table: str, gdf: gpd.GeoDataFrame,
              *, geom_cols: Iterable[str] = ("geometry",),
              if_exists: str = "replace",
              key_cols: Optional[Iterable[str]] = None) -> None:
    """Persist a GeoDataFrame to `table`.

    `if_exists`:
        - 'replace' — DROP + CREATE TABLE AS.
        - 'append'  — INSERT INTO an existing table, reconciling schema drift:
          columns the frame introduces are ADDed to the table; columns the
          table has but the frame lacks are filled NULL. This matters because
          different municipalities produce different column sets (e.g. CAT
          per-use-type area pivots), so a shared `buildings` table must be the
          union of all of them rather than the first writer's schema.
        - 'upsert' — requires `key_cols`. For rows whose key already exists,
          UPDATE only the columns this frame carries; INSERT the rest. Columns
          the table has but the frame lacks are left UNTOUCHED on existing rows.
          This is what lets a `building_parts_inference=False` re-run refresh
          the simple/CAT features while PRESERVING the complex inference
          features (shadows, walls, structural floors, …) computed by an
          earlier inference run — they simply aren't in the frame, so they keep
          their values instead of being nulled. Schema drift is reconciled the
          same way as 'append'. Creates the table when it doesn't exist yet.

    Geometry columns become native GEOMETRY. CRS is recorded in `_meta`.
    Any additional object column holding shapely geometries (e.g. the
    secondary geometries `read_gdf` decodes) is auto-detected and stored as
    GEOMETRY too, so it round-trips rather than being pickled to BLOB.
    """
    geom_set = set(geom_cols)
    for col in gdf.columns:
        if col in geom_set:
            continue
        sample = _first_non_null(gdf[col])
        if isinstance(sample, _BaseGeometry):
            geom_set.add(col)
    geom_cols = tuple(geom_set)
    df = _df_with_wkb(gdf, geom_cols)
    con.register("_tmp_write_gdf", df)

    def _table_exists() -> bool:
        try:
            con.execute(f'DESCRIBE "{table}"')
            return True
        except Exception:
            return False

    def _reconcile_schema():
        """ADD COLUMN for every frame column the table doesn't have yet."""
        existing = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
        tmp_types = {r[0]: str(r[1])
                     for r in con.execute('DESCRIBE _tmp_write_gdf').fetchall()}
        for c in df.columns:
            if c not in existing:
                _ty = 'GEOMETRY' if c in geom_set else tmp_types.get(c, 'VARCHAR')
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {_ty}')

    try:
        select_sql = _geom_select_sql(df.columns, geom_cols)
        if if_exists == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
            con.execute(
                f'CREATE TABLE "{table}" AS SELECT {select_sql} FROM _tmp_write_gdf'
            )
        elif if_exists == "append":
            _reconcile_schema()
            # Explicit column list → table columns absent from the frame get
            # NULL instead of a positional-count mismatch.
            col_list = ", ".join(f'"{c}"' for c in df.columns)
            con.execute(
                f'INSERT INTO "{table}" ({col_list}) '
                f'SELECT {select_sql} FROM _tmp_write_gdf'
            )
        elif if_exists == "upsert":
            keys = list(key_cols or ())
            if not keys or any(k not in df.columns for k in keys):
                raise ValueError(
                    "upsert requires key_cols present in the frame; got "
                    f"key_cols={key_cols!r}, frame columns missing "
                    f"{[k for k in keys if k not in df.columns]}"
                )
            if not _table_exists():
                con.execute(
                    f'CREATE TABLE "{table}" AS SELECT {select_sql} FROM _tmp_write_gdf'
                )
            else:
                _reconcile_schema()
                # Source rows with geometry decoded, used by both UPDATE and INSERT.
                src = f"(SELECT {select_sql} FROM _tmp_write_gdf) src"
                key_join = " AND ".join(f'"{table}"."{k}" = src."{k}"' for k in keys)
                set_cols = [c for c in df.columns if c not in keys]
                if set_cols:
                    set_clause = ", ".join(f'"{c}" = src."{c}"' for c in set_cols)
                    con.execute(
                        f'UPDATE "{table}" SET {set_clause} FROM {src} WHERE {key_join}'
                    )
                # Insert rows whose key isn't already present. Keys are plain
                # (non-geometry) columns, so correlate the table directly with
                # the raw _tmp_write_gdf rows.
                col_list = ", ".join(f'"{c}"' for c in df.columns)
                key_match = " AND ".join(
                    f'"{table}"."{k}" = _tmp_write_gdf."{k}"' for k in keys)
                con.execute(
                    f'INSERT INTO "{table}" ({col_list}) '
                    f'SELECT {select_sql} FROM _tmp_write_gdf '
                    f'WHERE NOT EXISTS (SELECT 1 FROM "{table}" WHERE {key_match})'
                )
        else:
            raise ValueError(
                f"if_exists must be 'replace', 'append' or 'upsert', "
                f"got {if_exists!r}"
            )
    finally:
        con.unregister("_tmp_write_gdf")
    if getattr(gdf, "crs", None) is not None:
        set_table_crs(con, table, gdf.crs)


# ---------------------------------------------------------------------------
# Per-zone shard DBs (step 2)
# ---------------------------------------------------------------------------

def open_shard(shard_path: str) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) a per-zone shard DB.

    Each cadastral zone gets its own DB file so the parallel inference
    workers have no write contention (DuckDB is single-writer per file).
    The parent later ATTACHes shards to the central DB and UNIONs.
    """
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    con = duckdb.connect(shard_path)
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    _silence_progress_bar(con)
    ensure_schema(con)
    return con


def open_shard_read(shard_path: str) -> duckdb.DuckDBPyConnection:
    """Open an existing shard read-only with the spatial extension loaded."""
    con = duckdb.connect(shard_path, read_only=True)
    con.execute("LOAD spatial")
    _silence_progress_bar(con)
    return con


def _silence_progress_bar(con: duckdb.DuckDBPyConnection) -> None:
    """Disable DuckDB's built-in progress bar.

    DuckDB renders a stdout progress bar for any query running longer
    than ~2s. With Barcelona-scale SELECT/INSERT/DELETE against the
    central DB, every Phase A/B operation trips that threshold, producing
    multiple unlabeled bars that duplicate our own tqdm output.
    """
    try:
        con.execute("SET enable_progress_bar = false")
    except Exception:
        pass
    # Cap DuckDB's working-set memory. Critically, this limit competes
    # with the *pandas* heap for the same process address space — the
    # consumer of read_df_safe materialises BLOB-pickled cells into
    # Python dicts that easily reach tens of GB. We pick a small DuckDB
    # cap on purpose so DuckDB spills to disk and leaves the bulk of
    # system RAM for pandas. Default: min(8 GB, system_ram/8), clamped
    # to >=2 GB. Override via HCDB_MEMORY_LIMIT (any DuckDB-accepted
    # size string, e.g. '4GB', '12GiB').
    limit = os.environ.get("HCDB_MEMORY_LIMIT")
    if not limit:
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            try:
                total_gb = (os.sysconf("SC_PAGE_SIZE")
                            * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
            except (AttributeError, ValueError):
                total_gb = 8.0
        target = max(2.0, min(8.0, total_gb / 8.0))
        limit = f"{int(target)}GB"
    try:
        con.execute(f"SET memory_limit = '{limit}'")
        # preserve_insertion_order=false lets DuckDB stream large
        # results without buffering the whole thing for ordering.
        con.execute("SET preserve_insertion_order = false")
    except Exception:
        pass


def get_blob_for_building(results_dir: str, cadaster_code: str,
                          building_reference: str, *,
                          table: str = "br_results",
                          columns: Optional[Iterable[str]] = None) -> dict:
    """Fetch heavy per-building BLOB cells on demand.

    Phase B's bulk materialisation omits HEAVY_BLOB_COLS (~22 GB of
    pickled shadow data on Barcelona) to keep the pandas peak in
    check. Use this to retrieve them for one building at a time —
    e.g. for `plot_building_shadows_analysis`.

    Returns a dict {column_name: unpickled_value}.
    """
    cols = list(columns) if columns is not None else list(HEAVY_BLOB_COLS)
    cols = [c for c in cols if (table.startswith("br") and c.startswith("br__"))
            or (table.startswith("sbr") and c.startswith("sbr__"))]
    if not cols:
        return {}
    select_sql = ", ".join(f'"{c}"' for c in cols)
    con = open_shard_read(db_path(results_dir))
    try:
        row = con.execute(
            f'SELECT {select_sql} FROM "main"."{table}" '
            f'WHERE cadaster_code = ? AND building_reference = ? LIMIT 1',
            [cadaster_code, building_reference],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return {c: None for c in cols}
    return {c: _maybe_unpickle(v) for c, v in zip(cols, row)}


def maybe_checkpoint(con: duckdb.DuckDBPyConnection) -> None:
    """Flush the WAL and release transient memory. Use periodically in
    long write loops (e.g. Phase A) to keep DuckDB's heap flat."""
    try:
        con.execute("CHECKPOINT")
    except Exception:
        pass


def read_df_safe_chunks(con: duckdb.DuckDBPyConnection, table: str, *,
                        where: Optional[str] = None,
                        params: Optional[list] = None,
                        vectors_per_chunk: int = 50,
                        exclude_cols: Optional[Iterable[str]] = None):
    """Streaming variant of `read_df_safe`.

    Yields successive pandas chunks (each ~2048 * vectors_per_chunk rows)
    with the same geometry / pickle decoding as the non-streaming
    version. Lets callers apply per-chunk transforms and avoid holding
    both the raw and transformed full frames in memory at the same time.
    """
    schema_rows = con.execute(f'DESCRIBE "main"."{table}"').fetchall()
    excluded = set(exclude_cols or ())
    cols = [r[0] for r in schema_rows if r[0] not in excluded]
    types = {r[0]: str(r[1]).upper() for r in schema_rows if r[0] not in excluded}

    select_parts = []
    geom_cols = []
    for c in cols:
        if types[c] == "GEOMETRY":
            geom_cols.append(c)
            select_parts.append(f'ST_AsWKB("{c}") AS "{c}"')
        else:
            select_parts.append(f'"{c}"')
    sql = f'SELECT {", ".join(select_parts)} FROM "main"."{table}"'
    if where:
        sql += f" WHERE {where}"

    result = con.execute(sql, params or [])
    while True:
        try:
            chunk = result.fetch_df_chunk(vectors_per_chunk)
        except Exception:
            break
        if chunk is None or chunk.empty:
            break
        for col in geom_cols:
            chunk[col] = gpd.GeoSeries.from_wkb([
                None if (b is None or b is pd.NA) else bytes(b)
                for b in chunk[col]
            ])
        for col in cols:
            if types[col] != "BLOB" or col in geom_cols:
                continue
            sample = _first_non_null(chunk[col])
            if sample is None:
                continue
            sb = bytes(sample) if not isinstance(sample, bytes) else sample
            if not (len(sb) >= 2 and sb[0] == 0x80):
                continue
            chunk[col] = [_maybe_unpickle(b) for b in chunk[col]]
        yield chunk
    # Do NOT call result.close(): DuckDB's Python API returns the same
    # connection object from con.execute(), so closing the "result"
    # would close the connection itself. Dropping the local reference
    # and letting the next con.execute() displace the cursor state is
    # the correct cleanup.
    del result


def ingest_shard_table(central_con: duckdb.DuckDBPyConnection, *,
                       shard_alias: str, shard_table: str,
                       central_table: str,
                       bookkeeping: dict) -> None:
    """Ingest one ATTACH'd shard's table into the central DB.

    Lazily extends the central schema (`ALTER TABLE ADD COLUMN`) for
    any column the shard introduces that isn't yet present centrally.
    Previously-ingested rows get NULL for new columns. The shard's row
    is INSERTed with explicit column projection so unrelated central
    columns also default to NULL for this row.

    Needed because per-zone shard schemas drift in this codebase:
    `process_zone`'s merges with CAT data produce different column sets
    per zone. Hard-coding "first shard's schema is canonical" caused
    every shard with extra columns to fail INSERT with a binder error.

    `bookkeeping` is a dict {column_name: value} to insert into every
    row from this shard (typically cadaster_code + zone_reference).
    Bookkeeping columns are typed as VARCHAR.
    """
    shard_typed = [
        (r[0], str(r[1])) for r in central_con.execute(
            f'DESCRIBE "{shard_alias}"."{shard_table}"'
        ).fetchall()
    ]
    # Drop any shard columns whose names collide with bookkeeping. The
    # bookkeeping values are authoritative (parent supplies them per
    # zone); a shard that happens to carry its own `zone_reference` /
    # `cadaster_code` column is by construction equal to those values
    # for every row, so dropping is lossless and avoids the DuckDB
    # "Duplicate column name in INSERT" binder error.
    shard_typed = [(n, t) for n, t in shard_typed if n not in bookkeeping]
    shard_names = [n for n, _ in shard_typed]
    # Record pre-resolution types so the SELECT builder knows whether to
    # call ST_AsWKB (needed for true GEOMETRY) or not (BLOB columns are
    # already WKB bytes and ST_AsWKB would fail with BLOB→GEOMETRY cast).
    original_shard_types = {n: t.upper() for n, t in shard_typed}

    # Fully qualify with "main" so DESCRIBE / ALTER don't get confused by
    # an identically-named table in the ATTACH'd shard.
    try:
        existing_rows = central_con.execute(
            f'DESCRIBE "main"."{central_table}"'
        ).fetchall()
        existing_types = {r[0]: str(r[1]).upper() for r in existing_rows}
        existing = set(existing_types.keys())
        central_exists = True
    except Exception:
        existing_types = {}
        existing = set()
        central_exists = False

    bookkeeping_cols = list(bookkeeping.keys())
    pickle_wrap_cols: set = set()

    if not central_exists:
        cols_ddl = [f'"{c}" VARCHAR' for c in bookkeeping_cols]
        cols_ddl += [f'"{n}" {t}' for n, t in shard_typed]
        central_con.execute(
            f'CREATE TABLE "main"."{central_table}" ({", ".join(cols_ddl)})'
        )
    else:
        for c in bookkeeping_cols:
            if c not in existing:
                central_con.execute(
                    f'ALTER TABLE "main"."{central_table}" '
                    f'ADD COLUMN "{c}" VARCHAR'
                )
        # Resolve type drift across shards (a real failure mode here:
        # process_zone's per-zone outputs sometimes have the same column
        # as a primitive in one zone and as a pickled dict/list in
        # another). Two cases handled losslessly via BLOB:
        #   - central=BLOB, shard=primitive → pickle-wrap incoming values
        #     during INSERT. Central type stays BLOB; new rows are
        #     pickled bytes that round-trip through _maybe_unpickle.
        #   - central=primitive, shard=BLOB → widen central column to
        #     BLOB via SQL (existing values encoded as their VARCHAR
        #     representation in bytes; on read, no 0x80 prefix so
        #     _maybe_unpickle returns them as raw bytes).
        # If the SQL widening fails (some primitive types can't be
        # encoded), we fall back to dropping the column for this shard
        # with a logged warning.
        _INT_TYPES = frozenset({
            "TINYINT", "SMALLINT", "INTEGER", "INT", "BIGINT", "HUGEINT",
            "INT8", "INT4", "INT2", "INT1", "UBIGINT", "UINTEGER",
        })
        _FLOAT_TYPES = frozenset({"FLOAT", "REAL", "DOUBLE", "FLOAT4", "FLOAT8"})
        _NUMERIC_TYPES = _INT_TYPES | _FLOAT_TYPES

        kept = []
        for n, t in shard_typed:
            if n not in existing_types:
                kept.append((n, t))
                continue
            ct = existing_types[n]
            st = t.upper()
            if ct == st:
                kept.append((n, t))
            elif ct == "BLOB":
                pickle_wrap_cols.add(n)
                kept.append((n, "BLOB"))
            elif st == "BLOB":
                # GEOMETRY columns in the shard appear as BLOB when accessed
                # via ATTACH (spatial extension metadata doesn't transfer).
                # Keep the central GEOMETRY; the INSERT path below uses
                # ST_AsWKB / ST_GeomFromWKB to bridge the gap.
                if ct == "GEOMETRY":
                    kept.append((n, "GEOMETRY"))
                else:
                    try:
                        central_con.execute(
                            f'ALTER TABLE "main"."{central_table}" '
                            f'ALTER COLUMN "{n}" SET DATA TYPE BLOB '
                            f'USING ENCODE(CAST("{n}" AS VARCHAR))'
                        )
                        existing_types[n] = "BLOB"
                        kept.append((n, "BLOB"))
                    except Exception as _wide_err:
                        sys.stderr.write(
                            f"[ingest_shard_table] could not widen '{n}' "
                            f"from {ct} to BLOB ({_wide_err}); dropping for "
                            f"this shard.\n"
                        )
            elif ct in _NUMERIC_TYPES and st in _NUMERIC_TYPES:
                # Numeric drift (e.g. BIGINT central vs DOUBLE shard).
                # DOUBLE can represent all BIGINT values for quantities like
                # floor counts without precision loss.
                if ct in _INT_TYPES:
                    # Widen central integer column to DOUBLE once.
                    try:
                        central_con.execute(
                            f'ALTER TABLE "main"."{central_table}" '
                            f'ALTER COLUMN "{n}" SET DATA TYPE DOUBLE'
                        )
                        existing_types[n] = "DOUBLE"
                        kept.append((n, "DOUBLE"))
                    except Exception as _ne:
                        sys.stderr.write(
                            f"[ingest_shard_table] type drift on '{n}' "
                            f"(central={ct}, shard={st}); could not widen "
                            f"({_ne}), dropping for this shard.\n"
                        )
                else:
                    # Central already floating — integer shard values fit.
                    kept.append((n, t))
            else:
                # Both sides non-BLOB, non-numeric, mismatched
                # (e.g. DOUBLE vs VARCHAR). Drop with a warning.
                sys.stderr.write(
                    f"[ingest_shard_table] type drift on '{n}' "
                    f"(central={ct}, shard={st}); dropping for this shard.\n"
                )
        shard_typed = kept
        shard_names = [n for n, _ in shard_typed]
        for n, t in shard_typed:
            if n not in existing:
                central_con.execute(
                    f'ALTER TABLE "main"."{central_table}" '
                    f'ADD COLUMN "{n}" {t}'
                )

    target_cols = bookkeeping_cols + shard_names
    target_sql = ", ".join(f'"{c}"' for c in target_cols)

    # Ensure spatial extension is loaded on the central connection so
    # GEOMETRY types from the ATTACH'd shard are resolved correctly.
    try:
        central_con.execute("LOAD spatial")
    except Exception:
        pass

    if not pickle_wrap_cols:
        # Fast path: pure SQL INSERT, no Python materialisation.
        # Split shard columns into GEOMETRY and non-GEOMETRY to handle
        # the ATTACH boundary type mismatch. GEOMETRY columns from the
        # shard can appear as BLOB to the central connection even when
        # DESCRIBE reports GEOMETRY (spatial extension metadata doesn't
        # transfer across ATTACH).
        # Also detect GEOMETRY columns by checking the central schema:
        # a shard column stored as BLOB that corresponds to a central
        # GEOMETRY column needs the same WKB round-trip treatment.
        geo_cols = {n for n, t in shard_typed if t.upper() == "GEOMETRY"}
        geo_cols |= {n for n in shard_names
                     if existing_types.get(n) == "GEOMETRY"}
        non_geo = [n for n in shard_names if n not in geo_cols]
        if geo_cols:
            pass

        if not geo_cols:
            # No geometry columns — straightforward INSERT.
            source_parts = ["?"] * len(bookkeeping_cols) + [f'"{n}"' for n in shard_names]
            source_sql = ", ".join(source_parts)
            central_con.execute(
                f'INSERT INTO "main"."{central_table}" ({target_sql}) '
                f'SELECT {source_sql} FROM "{shard_alias}"."{shard_table}"',
                list(bookkeeping.values()),
            )
        else:
            # Materialise shard into pandas. GEOMETRY columns need
            # ST_AsWKB wrapper so they come out as WKB bytes rather than
            # DuckDB GEOMETRY objects (which appear as BLOB across ATTACH).
            select_parts = []
            geo_set = geo_cols
            for n in shard_names:
                if n in geo_set:
                    # ST_AsWKB requires a GEOMETRY argument. If the shard
                    # stored the column as BLOB (WKB bytes — spatial metadata
                    # doesn't transfer via ATTACH), calling ST_AsWKB would
                    # trigger a failing implicit BLOB→GEOMETRY cast. Select
                    # as-is instead; the bytes are already WKB and
                    # ST_GeomFromWKB on INSERT handles the conversion.
                    if original_shard_types.get(n) == "GEOMETRY":
                        select_parts.append(f'ST_AsWKB("{n}") AS "{n}"')
                    else:
                        select_parts.append(f'"{n}"')
                else:
                    select_parts.append(f'"{n}"')
            df = central_con.execute(
                f'SELECT {", ".join(select_parts)} FROM "{shard_alias}"."{shard_table}"'
            ).fetchdf()
            # Ensure WKB bytes are proper Python bytes (not memoryview)
            for c in geo_cols:
                if c in df.columns:
                    df[c] = [bytes(x) if x is not None and not isinstance(x, bytes) else x
                             for x in df[c]]
            for c, v in reversed(list(bookkeeping.items())):
                df.insert(0, c, v)
            central_con.register("_tmp_ingest_all", df)
            try:
                # Wrap GEOMETRY columns with ST_GeomFromWKB on INSERT
                insert_select_parts = []
                for c in target_cols:
                    if c in geo_cols:
                        insert_select_parts.append(
                            f'ST_GeomFromWKB("{c}") AS "{c}"'
                        )
                    else:
                        insert_select_parts.append(f'"{c}"')
                central_con.execute(
                    f'INSERT INTO "main"."{central_table}" ({target_sql}) '
                    f'SELECT {", ".join(insert_select_parts)} FROM _tmp_ingest_all'
                )
            finally:
                central_con.unregister("_tmp_ingest_all")
    else:
        # Slow path: pickle-wrap the conflicting primitive columns so
        # they fit the central BLOB column. Materialise into pandas; each
        # shard is a single zone so the frame is bounded.
        # Geometry columns may coexist with pickle-wrap columns; handle
        # them the same way as the fast path (already-WKB BLOB selected
        # as-is, ST_GeomFromWKB applied on INSERT).
        slow_geo_cols = {n for n, t in shard_typed if t.upper() == "GEOMETRY"}
        slow_geo_cols |= {n for n in shard_names if existing_types.get(n) == "GEOMETRY"}
        select_parts_slow = []
        for n in shard_names:
            if n in slow_geo_cols and original_shard_types.get(n) == "GEOMETRY":
                select_parts_slow.append(f'ST_AsWKB("{n}") AS "{n}"')
            else:
                select_parts_slow.append(f'"{n}"')
        df = central_con.execute(
            f'SELECT {", ".join(select_parts_slow)} FROM "{shard_alias}"."{shard_table}"'
        ).fetchdf()
        for c in slow_geo_cols:
            if c in df.columns:
                df[c] = [bytes(x) if x is not None and not isinstance(x, bytes) else x
                         for x in df[c]]
        for c in pickle_wrap_cols:
            if c in df.columns:
                df[c] = [None if _is_null(v) else _to_pickle_bytes(v)
                         for v in df[c]]
        # Prepend the bookkeeping columns (broadcast as scalars).
        for c, v in reversed(list(bookkeeping.items())):
            df.insert(0, c, v)
        central_con.register("_tmp_ingest_wrap", df)
        try:
            insert_select_slow = []
            for c in target_cols:
                if c in slow_geo_cols:
                    insert_select_slow.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
                else:
                    insert_select_slow.append(f'"{c}"')
            central_con.execute(
                f'INSERT INTO "main"."{central_table}" ({target_sql}) '
                f'SELECT {", ".join(insert_select_slow)} FROM _tmp_ingest_wrap'
            )
        finally:
            central_con.unregister("_tmp_ingest_wrap")


def _is_null(v) -> bool:
    """True for None / pd.NA / NaN floats. Robust against scalars where
    pd.isna would broadcast (e.g. ndarrays in object columns)."""
    if v is None or v is pd.NA:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    return False


def _first_non_null(series: pd.Series):
    for v in series:
        if _is_null(v):
            continue
        return v
    return None


def _classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Bucket each column into (primitive, geometry, opaque).

    - primitive: pandas can hand it to DuckDB directly (numeric, str, bool,
      datetime, all-null object).
    - geometry: object column whose values are shapely objects.
    - opaque: object column with dicts / lists / tuples / sets / any other
      non-trivial python object. Persisted as pickled BLOB for lossless
      round-trip; step 6 will refine the heterogeneous ones to typed
      LIST<STRUCT> or JSON.
    """
    primitive: List[str] = []
    geometry: List[str] = []
    opaque: List[str] = []
    for col in df.columns:
        s = df[col]
        dtype_name = str(s.dtype)
        # GeoDataFrame's *active* geometry column has the 'geometry'
        # extension dtype, not object. Skipping the sample probe for
        # non-object dtypes used to misclassify it as primitive →
        # DuckDB stored it as VARCHAR via shapely's __repr__.
        if s.dtype != object and dtype_name != "geometry":
            primitive.append(col)
            continue
        sample = _first_non_null(s)
        if sample is None:
            primitive.append(col)
            continue
        # Shapely objects all expose __geo_interface__ and wkb_hex / wkb attrs.
        if hasattr(sample, "__geo_interface__") and hasattr(sample, "wkb"):
            geometry.append(col)
        elif isinstance(sample, str):
            primitive.append(col)
        elif isinstance(sample, (bytes, bytearray, memoryview)):
            primitive.append(col)  # already binary; let DuckDB store as BLOB
        else:
            opaque.append(col)
    return primitive, geometry, opaque


def _to_pickle_bytes(v):
    if v is None:
        return None
    # Skip NaN floats (pandas-imputed nulls in object columns)
    if isinstance(v, float) and v != v:
        return None
    return _pickle.dumps(v, protocol=_pickle.HIGHEST_PROTOCOL)


def write_df_safe(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame,
                  *, if_exists: str = "replace",
                  crs=None) -> None:
    """Persist a pandas DataFrame to `table` with lossless round-trip.

    Geometry columns become native GEOMETRY (via ST_GeomFromWKB on WKB).
    Other complex object columns (dicts / lists / arbitrary objects) are
    pickled into BLOB columns. Read back with `read_df_safe`.

    `if_exists`:
        - 'replace' — DROP + CREATE.
        - 'append'  — INSERT INTO existing table; schemas must match.

    `crs` (optional): if provided, written to `_meta` for each geometry
    column (all geometry cols in a single frame share the same CRS).
    """
    primitive, geometry, opaque = _classify_columns(df)
    geom_set, opq_set = set(geometry), set(opaque)
    # Build a fresh plain pandas DataFrame. We can't just `df.copy()` and
    # reassign columns: a geopandas GeoDataFrame's setitem re-coerces
    # bytes back into the 'geometry' extension dtype, which DuckDB then
    # rejects in con.register with "Data type 'geometry' not recognized".
    new_data = {}
    for col in df.columns:
        if col in geom_set:
            # GeoSeries.to_wkb handles None entries (returns None).
            new_data[col] = gpd.GeoSeries(df[col]).to_wkb()
        elif col in opq_set:
            new_data[col] = [_to_pickle_bytes(v) for v in df[col]]
        else:
            src = df[col]
            # Preserve pandas extension dtypes (e.g. StringDtype, Int64) by
            # keeping the Series — `to_numpy()` would collapse an all-NA string
            # column back to object, which DuckDB then infers as INTEGER,
            # causing VARCHAR-vs-INTEGER type drift against the central table.
            if hasattr(src, "dtype") and pd.api.types.is_extension_array_dtype(src.dtype):
                new_data[col] = src.reset_index(drop=True)
            else:
                new_data[col] = src.to_numpy() if hasattr(src, "to_numpy") else list(src)
    df2 = pd.DataFrame(new_data)

    con.register("_tmp_write_df_safe", df2)
    try:
        select_parts = []
        for c in df2.columns:
            if c in geometry:
                select_parts.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
            else:
                select_parts.append(f'"{c}"')
        select_sql = ", ".join(select_parts)
        if if_exists == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
            con.execute(
                f'CREATE TABLE "{table}" AS SELECT {select_sql} FROM _tmp_write_df_safe'
            )
        elif if_exists == "append":
            con.execute(
                f'INSERT INTO "{table}" SELECT {select_sql} FROM _tmp_write_df_safe'
            )
        else:
            raise ValueError(
                f"if_exists must be 'replace' or 'append', got {if_exists!r}"
            )
    finally:
        con.unregister("_tmp_write_df_safe")
    if crs is not None:
        for col in geometry:
            set_table_crs(con, f"{table}.{col}", crs)


def read_df_safe(con: duckdb.DuckDBPyConnection, table: str, *,
                 where: Optional[str] = None,
                 params: Optional[list] = None,
                 exclude_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Inverse of `write_df_safe`.

    GEOMETRY columns are unwrapped to shapely; BLOB columns whose first
    bytes look like pickled data are rehydrated via pickle.loads.

    `exclude_cols` skips those column names entirely (no SELECT, no
    decode) — useful for omitting heavy BLOB columns from a bulk read.
    """
    schema_rows = con.execute(f'DESCRIBE "{table}"').fetchall()
    # DESCRIBE returns: (column_name, column_type, null, key, default, extra)
    excluded = set(exclude_cols or ())
    cols = [r[0] for r in schema_rows if r[0] not in excluded]
    types = {r[0]: str(r[1]).upper() for r in schema_rows if r[0] not in excluded}

    select_parts = []
    geom_cols = []
    for c in cols:
        if types[c] == "GEOMETRY":
            geom_cols.append(c)
            select_parts.append(f'ST_AsWKB("{c}") AS "{c}"')
        else:
            select_parts.append(f'"{c}"')
    sql = f'SELECT {", ".join(select_parts)} FROM "{table}"'
    if where:
        sql += f" WHERE {where}"
    df = con.execute(sql, params or []).fetchdf()

    for col in geom_cols:
        # Use GeoSeries.from_wkb so the column lands as a proper geopandas
        # GeoSeries (dtype 'geometry'). Element-wise shapely.wkb.loads
        # returns valid Polygon/Point objects but geopandas 1.x's
        # from_shapely path rejects them when later wrapped in GeoDataFrame.
        df[col] = gpd.GeoSeries.from_wkb([
            None if (b is None or b is pd.NA) else bytes(b)
            for b in df[col]
        ])
    # BLOB columns may be opaque-pickled. Decide PER ROW: if the bytes
    # look like pickle (start with 0x80) and successfully unpickle, use
    # the rehydrated value; otherwise leave as raw bytes. Per-row
    # robustness matters because a column may end up mixed across
    # shards, and a single coincidentally-0x80 raw value used to crash
    # the whole read.
    for col in cols:
        if types[col] != "BLOB" or col in geom_cols:
            continue
        sample = _first_non_null(df[col])
        if sample is None:
            continue
        sb = bytes(sample) if not isinstance(sample, bytes) else sample
        if not (len(sb) >= 2 and sb[0] == 0x80):
            continue  # no row in the probe likely pickled; cheap-out
        df[col] = [_maybe_unpickle(b) for b in df[col]]
    return df


def _maybe_unpickle(b):
    """Try to unpickle bytes; on any failure (or non-pickle), return raw."""
    if b is None or b is pd.NA:
        return None
    bb = bytes(b) if not isinstance(b, bytes) else b
    if len(bb) < 2 or bb[0] != 0x80:
        return bb
    try:
        return _pickle.loads(bb)
    except Exception:
        # Bytes happened to start with 0x80 but aren't a valid pickle —
        # most likely raw binary data that was stored without our
        # serialisation wrapper. Return the raw bytes; caller can
        # decide what to do.
        return bb


def validate_shard_vs_pickle(shard_path: str, sbr_pkl: str, br_pkl: str,
                              *, atol: float = 1e-9) -> dict:
    """Compare a per-zone shard DB against the parallel pickle pair.

    Returns a dict summarising mismatches. Empty dict means full agreement.
    Intended for manual spot-checks during step 2 validation; not used
    by the runtime path.
    """
    report = {}
    sbr_pk = pd.read_pickle(sbr_pkl)
    br_pk = pd.read_pickle(br_pkl)
    con = duckdb.connect(shard_path, read_only=True)
    try:
        con.execute("LOAD spatial")
        sbr_db = read_df_safe(con, "sbr_results")
        br_db = read_df_safe(con, "br_results")
    finally:
        con.close()

    def _cmp(name, pk, db):
        diffs = {}
        if len(pk) != len(db):
            diffs["row_count"] = (len(pk), len(db))
        if set(pk.columns) != set(db.columns):
            diffs["columns"] = {
                "only_in_pkl": sorted(set(pk.columns) - set(db.columns)),
                "only_in_db":  sorted(set(db.columns) - set(pk.columns)),
            }
        if diffs:
            return diffs
        # Sort by all primary-key-ish columns we know about; fall back to
        # row order if none present.
        keys = [c for c in ("building_reference", "single_building_reference")
                if c in pk.columns]
        if keys:
            pk = pk.sort_values(keys).reset_index(drop=True)
            db = db.sort_values(keys).reset_index(drop=True)
        mismatched_cols = []
        for col in pk.columns:
            a, b = pk[col], db[col]
            if a.dtype.kind in "fc":  # float / complex tolerance
                if not ((a.isna() == b.isna()).all()
                        and ((a.fillna(0) - b.fillna(0)).abs() <= atol).all()):
                    mismatched_cols.append(col)
            else:
                # Element-wise equality with shapely-aware comparison
                ok = True
                for x, y in zip(a, b):
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        if y is None or (isinstance(y, float) and pd.isna(y)):
                            continue
                        ok = False; break
                    if hasattr(x, "equals") and hasattr(y, "equals"):
                        if not x.equals(y):
                            ok = False; break
                    elif x != y:
                        ok = False; break
                if not ok:
                    mismatched_cols.append(col)
        if mismatched_cols:
            diffs["mismatched_columns"] = mismatched_cols
        return diffs

    sbr_diff = _cmp("sbr_results", sbr_pk, sbr_db)
    if sbr_diff:
        report["sbr_results"] = sbr_diff
    br_diff = _cmp("br_results", br_pk, br_db)
    if br_diff:
        report["br_results"] = br_diff
    return report


# ---------------------------------------------------------------------------
# Geometry-specific gdf helpers (kept separate from write_df_safe for
# callers that know they have a GeoDataFrame with a designated geometry col)
# ---------------------------------------------------------------------------

def read_gdf(con: duckdb.DuckDBPyConnection, table: str, *,
             where: Optional[str] = None, params: Optional[list] = None,
             geom_col: str = "geometry") -> gpd.GeoDataFrame:
    """Read a DuckDB table back as a GeoDataFrame.

    The designated geometry column is materialised from GEOMETRY → WKB →
    shapely via `GeoSeries.from_wkb` and set as the active geometry. CRS is
    restored from `_meta`.

    Object columns are also decoded so the returned frame carries usable
    Python values rather than opaque ``bytearray``s:

    * Other GEOMETRY-typed columns (e.g. ``br__building_footprint_geometry``)
      → shapely geometries (plain object column, not the active geometry).
    * Pickled BLOBs (0x80 prefix) → their original Python object.
    * Text BLOBs (valid UTF-8, e.g. a date string widened to BLOB by shard
      type-drift) → ``str``.

    `write_gdf` round-trips all of these back (it auto-detects shapely
    columns as GEOMETRY and re-pickles complex objects).
    """
    sql = f'SELECT * REPLACE (ST_AsWKB("{geom_col}") AS "{geom_col}") FROM "{table}"'
    if where:
        sql += f" WHERE {where}"
    df = con.execute(sql, params or []).fetchdf()
    # DuckDB BLOB → pandas object column, with NULLs surfacing as pd.NA.
    # GeoSeries.from_wkb accepts None for missing, so coerce here.
    df[geom_col] = gpd.GeoSeries.from_wkb([
        None if (b is None or b is pd.NA) else bytes(b)
        for b in df[geom_col]
    ])

    # Columns DuckDB reports as GEOMETRY (other than geom_col) come back as
    # raw WKB bytes — decode them to shapely so callers get real geometries.
    try:
        geom_typed = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()
                      if str(r[1]).upper() == 'GEOMETRY'}
    except Exception:
        geom_typed = set()

    def _wkb_or_none(b):
        if b is None or b is pd.NA:
            return None
        try:
            return _shapely_wkb.loads(bytes(b))
        except Exception:
            return b  # leave undecodable bytes untouched

    def _text_or_bytes(b):
        if b is None or b is pd.NA:
            return None
        try:
            return bytes(b).decode('utf-8')
        except Exception:
            return b  # genuine binary stays as bytes

    for col in df.columns:
        if col == geom_col or df[col].dtype != object:
            continue
        if col in geom_typed:
            df[col] = [_wkb_or_none(v) for v in df[col]]
            continue
        sample = _first_non_null(df[col])
        if sample is None or not isinstance(sample, (bytes, bytearray, memoryview)):
            continue
        sb = bytes(sample) if not isinstance(sample, bytes) else sample
        if len(sb) >= 2 and sb[0] == 0x80:
            df[col] = [_maybe_unpickle(v) for v in df[col]]   # pickled object
        else:
            # Non-pickle BLOB: decode to str if it is valid UTF-8 text (e.g. a
            # date string that shard type-drift widened to BLOB), else keep raw.
            try:
                sb.decode('utf-8')
            except Exception:
                continue
            df[col] = [_text_or_bytes(v) for v in df[col]]
    return gpd.GeoDataFrame(df, geometry=geom_col, crs=get_table_crs(con, table))


# ---------------------------------------------------------------------------
# ResultsHandle — lightweight wrapper around a central DB + cadaster_code
# ---------------------------------------------------------------------------

class ResultsHandle:
    """Cheap wrapper around a central DuckDB path and cadaster_code.

    Opens a new short-lived connection per query rather than holding one
    open — keeps the handle small and safe to pass across functions.
    Materialisation (pandas frames) happens explicitly via ``to_pandas()``
    so the caller sees the cost.

    If a ``parent_con`` is provided (e.g. from mergers.py that already
    holds a read-write connection), queries reuse that connection to
    avoid "can't open same DB with different config" errors.
    """

    __slots__ = ("db_path", "cadaster_code", "_parent_con")

    def __init__(self, db_path: str, cadaster_code: str,
                 parent_con: Optional[duckdb.DuckDBPyConnection] = None) -> None:
        self.db_path = db_path
        self.cadaster_code = cadaster_code
        self._parent_con = parent_con

    # -- relation helpers ---------------------------------------------------

    def _con(self) -> duckdb.DuckDBPyConnection:
        if self._parent_con is not None:
            return self._parent_con
        con = duckdb.connect(self.db_path, read_only=True)
        con.execute("LOAD spatial")
        _silence_progress_bar(con)
        return con

    def sbr_relation(self) -> duckdb.DuckDBPyRelation:
        """Return a lazy DuckDB relation for sbr_results filtered by code."""
        con = self._con()
        return con.execute(
            'SELECT * FROM "main"."sbr_results" WHERE cadaster_code = ?',
            [self.cadaster_code],
        )

    def br_relation(self) -> duckdb.DuckDBPyRelation:
        """Return a lazy DuckDB relation for br_results filtered by code."""
        con = self._con()
        return con.execute(
            'SELECT * FROM "main"."br_results" WHERE cadaster_code = ?',
            [self.cadaster_code],
        )

    # -- batched iteration --------------------------------------------------

    def sbr_iter_batches(self, batch_size: int = 5000):
        """Yield successive pandas chunks for sbr_results."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            yield from read_df_safe_chunks(
                con, "sbr_results",
                where="cadaster_code = ?", params=[self.cadaster_code],
                exclude_cols=HEAVY_BLOB_COLS,
                vectors_per_chunk=batch_size // 2048 or 1,
            )
        finally:
            if own_con:
                con.close()

    def br_iter_batches(self, batch_size: int = 5000):
        """Yield successive pandas chunks for br_results."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            yield from read_df_safe_chunks(
                con, "br_results",
                where="cadaster_code = ?", params=[self.cadaster_code],
                exclude_cols=HEAVY_BLOB_COLS,
                vectors_per_chunk=batch_size // 2048 or 1,
            )
        finally:
            if own_con:
                con.close()

    # -- full materialisation (escape hatch) --------------------------------

    def sbr_to_pandas(self) -> pd.DataFrame:
        """Materialise all sbr rows for this code into a single DataFrame."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            return read_df_safe(
                con, "sbr_results",
                where="cadaster_code = ?", params=[self.cadaster_code],
                exclude_cols=HEAVY_BLOB_COLS,
            )
        finally:
            if own_con:
                con.close()

    def br_to_pandas(self) -> pd.DataFrame:
        """Materialise all br rows for this code into a single DataFrame."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            return read_df_safe(
                con, "br_results",
                where="cadaster_code = ?", params=[self.cadaster_code],
                exclude_cols=HEAVY_BLOB_COLS,
            )
        finally:
            if own_con:
                con.close()

    # -- row count ----------------------------------------------------------

    def sbr_count(self) -> int:
        """Number of ``sbr_results`` rows for this handle's municipality."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            row = con.execute(
                'SELECT count(*) FROM "main"."sbr_results" WHERE cadaster_code = ?',
                [self.cadaster_code],
            ).fetchone()
            return row[0] if row else 0
        finally:
            if own_con:
                con.close()

    def br_count(self) -> int:
        """Number of ``br_results`` rows for this handle's municipality."""
        con = self._con()
        own_con = con is not self._parent_con
        try:
            row = con.execute(
                'SELECT count(*) FROM "main"."br_results" WHERE cadaster_code = ?',
                [self.cadaster_code],
            ).fetchone()
            return row[0] if row else 0
        finally:
            if own_con:
                con.close()
