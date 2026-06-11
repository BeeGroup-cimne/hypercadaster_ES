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
# CRS sidecar — the spatial extension does not store per-table CRS itself
# ---------------------------------------------------------------------------

def set_table_crs(con: duckdb.DuckDBPyConnection, table: str, crs) -> None:
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
              if_exists: str = "replace") -> None:
    """Persist a GeoDataFrame to `table`.

    `if_exists`:
        - 'replace' — DROP + CREATE TABLE AS.
        - 'append'  — INSERT INTO an existing table with a compatible schema.

    Geometry columns become native GEOMETRY. CRS is recorded in `_meta`.
    """
    geom_cols = tuple(geom_cols)
    df = _df_with_wkb(gdf, geom_cols)
    con.register("_tmp_write_gdf", df)
    try:
        select_sql = _geom_select_sql(df.columns, geom_cols)
        if if_exists == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
            con.execute(
                f'CREATE TABLE "{table}" AS SELECT {select_sql} FROM _tmp_write_gdf'
            )
        elif if_exists == "append":
            con.execute(
                f'INSERT INTO "{table}" SELECT {select_sql} FROM _tmp_write_gdf'
            )
        else:
            raise ValueError(
                f"if_exists must be 'replace' or 'append', got {if_exists!r}"
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

    The geometry column is materialised from GEOMETRY → WKB → shapely
    via `GeoSeries.from_wkb`. CRS is restored from `_meta`. BLOB
    columns whose bytes look like pickled data are unpickled (same
    heuristic as `read_df_safe`); other GEOMETRY columns stay as raw
    WKB bytes so they don't acquire the geopandas `geometry` dtype
    (which would cause DuckDB `con.register` failures in `write_gdf`).
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
    # Unpickle BLOB columns whose content is a pickled Python object.
    # Only the designated geom_col is converted to GeoSeries; all other
    # bytes/bytearray columns stay as bytes unless they look like pickle.
    for col in df.columns:
        if col == geom_col or df[col].dtype != object:
            continue
        sample = _first_non_null(df[col])
        if sample is None or not isinstance(sample, (bytes, bytearray, memoryview)):
            continue
        sb = bytes(sample) if not isinstance(sample, bytes) else sample
        if len(sb) < 2 or sb[0] != 0x80:
            continue
        df[col] = [_maybe_unpickle(v) for v in df[col]]
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
