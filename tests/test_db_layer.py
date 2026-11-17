"""Tests for the DuckDB persistence layer (`_db.py`) across duckdb 1.5 and 2.x.

duckdb 2.0 changed two connection semantics this layer depends on:

* a partially-read ``execute()`` result (e.g. after ``fetchone()``) blocks
  ``register()`` / ``sql()`` on the same connection with ResourceInUseException;
* a failed statement inside an explicit transaction aborts it — later statements
  raise and ``COMMIT`` silently rolls back — so "probe with try/except" patterns
  are unsafe inside ``BEGIN TRANSACTION``.

Each test below exercises one call sequence the inference / merge paths use, so a
regression on either duckdb major shows up offline.
"""
import duckdb
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from hypercadaster_ES import _db


def _gdf(refs, *, extra=None):
    data = {
        "cadaster_code": ["00900"] * len(refs),
        "building_reference": list(refs),
        "n_floors": [i + 1 for i in range(len(refs))],
        "attrs": [{"use": "Residential", "i": i} for i in range(len(refs))],
        "geometry": [box(i, i, i + 1, i + 1) for i in range(len(refs))],
    }
    data.update(extra or {})
    return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:25831")


# ---------------------------------------------------------------------------
# GeoDataFrame round-trips
# ---------------------------------------------------------------------------

def test_write_gdf_replace_append_and_read_roundtrip(tmp_path):
    con = _db.open_db(str(tmp_path))
    try:
        _db.write_gdf(con, "buildings", _gdf(["A", "B"]), if_exists="replace")
        _db.write_gdf(con, "buildings",
                      _gdf(["C"], extra={"new_col": ["x"]}), if_exists="append")
        out = _db.read_gdf(con, "buildings").sort_values("building_reference")
    finally:
        con.close()

    assert list(out["building_reference"]) == ["A", "B", "C"]
    assert out.crs.to_epsg() == 25831
    assert out.geometry.iloc[2].equals(box(0, 0, 1, 1))
    assert out["attrs"].iloc[1] == {"use": "Residential", "i": 1}
    assert out["new_col"].isna().sum() == 2 and out["new_col"].iloc[2] == "x"


def test_write_gdf_upsert_creates_then_preserves_untouched_columns(tmp_path):
    con = _db.open_db(str(tmp_path))
    keys = ("cadaster_code", "building_reference")
    try:
        # First upsert must create the table (no table-exists probe may fail).
        _db.write_gdf(con, "buildings",
                      _gdf(["A", "B"], extra={"inferred": [10.0, 20.0]}),
                      if_exists="upsert", key_cols=keys)
        # Second upsert lacks 'inferred': existing values must survive.
        refreshed = _gdf(["B", "C"])
        refreshed["n_floors"] = [99, 7]
        _db.write_gdf(con, "buildings", refreshed,
                      if_exists="upsert", key_cols=keys)
        out = _db.read_gdf(con, "buildings").set_index("building_reference")
    finally:
        con.close()

    assert sorted(out.index) == ["A", "B", "C"]
    assert out.loc["B", "n_floors"] == 99
    assert out.loc["B", "inferred"] == 20.0
    assert pd.isna(out.loc["C", "inferred"])


def test_write_gdf_upsert_inside_transaction(tmp_path):
    con = _db.open_db(str(tmp_path))
    keys = ("cadaster_code", "building_reference")
    try:
        con.execute("BEGIN TRANSACTION")
        _db.write_gdf(con, "buildings", _gdf(["A"]),
                      if_exists="upsert", key_cols=keys)
        con.execute("COMMIT")
        n = con.execute('SELECT count(*) FROM "buildings"').fetchall()[0][0]
    finally:
        con.close()
    assert n == 1


def test_write_df_safe_roundtrip_and_chunks(tmp_path):
    df = pd.DataFrame({
        "k": list(range(5000)),
        "obj": [{"v": i} for i in range(5000)],
        "geom": [Point(i, i) for i in range(5000)],
    })
    con = _db.open_db(str(tmp_path))
    try:
        _db.write_df_safe(con, "t", df, if_exists="replace")
        back = _db.read_df_safe(con, "t").sort_values("k").reset_index(drop=True)
        chunks = list(_db.read_df_safe_chunks(con, "t", vectors_per_chunk=1))
    finally:
        con.close()

    assert back["obj"].iloc[42] == {"v": 42}
    assert back["geom"].iloc[42].equals(Point(42, 42))
    assert sum(len(c) for c in chunks) == 5000
    assert len(chunks) > 1


def test_abandoned_chunk_stream_does_not_block_connection(tmp_path):
    df = pd.DataFrame({"k": list(range(10000))})
    con = _db.open_db(str(tmp_path))
    try:
        _db.write_df_safe(con, "t", df, if_exists="replace")
        gen = _db.read_df_safe_chunks(con, "t", vectors_per_chunk=1)
        next(gen)
        gen.close()  # consumer stops early → partially-read result
        # register()/sql() must still work on the same connection.
        con.register("_probe", pd.DataFrame({"a": [1]}))
        assert con.sql("SELECT a FROM _probe").fetchall() == [(1,)]
        con.unregister("_probe")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Single-row reads must not leave the connection with an unfinished result
# ---------------------------------------------------------------------------

def test_single_row_reads_leave_connection_usable(tmp_path):
    con = _db.open_db(str(tmp_path))
    try:
        _db.write_gdf(con, "buildings", _gdf(["A"]))
        _db.mark_zone(con, "00900", "Z1", "ok")
        assert _db.get_table_crs(con, "buildings").to_epsg() == 25831
        con.register("_after_crs", pd.DataFrame({"a": [1]}))
        con.unregister("_after_crs")
        assert _db._table_exists(con, "buildings")
        assert not _db._table_exists(con, "nope")
        con.register("_after_exists", pd.DataFrame({"a": [1]}))
        con.unregister("_after_exists")
        assert _db.completed_zones(con, "00900") == {"Z1"}
    finally:
        con.close()


def test_results_handle_count_then_register_on_parent_con(tmp_path):
    """mergers.py: handle.br_count() then central_con.register(...) on the
    same (parent) connection."""
    con = _db.open_db(str(tmp_path))
    try:
        _db.write_df_safe(con, "br_results", pd.DataFrame({
            "cadaster_code": ["00900", "00900"],
            "building_reference": ["A", "B"],
            "br__shadows": [{"s": 1}, {"s": 2}],
            "br__height": [3.0, 6.0],
        }))
        h = _db.ResultsHandle(_db.db_path(str(tmp_path)), "00900", parent_con=con)
        assert h.br_count() == 2
        con.register("_building_slim", pd.DataFrame({"a": [1]}))
        con.unregister("_building_slim")
        df = h.br_to_pandas()
        assert "br__shadows" not in df.columns  # heavy BLOBs excluded
        assert sorted(df["br__height"]) == [3.0, 6.0]
        assert h.br_relation().aggregate("count(*)").fetchall() == [(2,)]
    finally:
        con.close()

    blobs = _db.get_blob_for_building(str(tmp_path), "00900", "B",
                                      columns=["br__shadows"])
    assert blobs == {"br__shadows": {"s": 2}}


# ---------------------------------------------------------------------------
# Phase A: shard ingestion inside BEGIN TRANSACTION
# ---------------------------------------------------------------------------

def _make_shard(path, sbr_rows, br_rows):
    con = _db.open_shard(str(path))
    try:
        _db.write_df_safe(con, "sbr_results", pd.DataFrame(sbr_rows))
        _db.write_df_safe(con, "br_results", pd.DataFrame(br_rows))
    finally:
        con.close()


def _ingest(central, shard_path, zone):
    """Mirror of building_inference's Phase A per-shard transaction."""
    central.execute(f"ATTACH '{shard_path}' AS shard (READ_ONLY)")
    try:
        central.execute("BEGIN TRANSACTION")
        try:
            bk = {"cadaster_code": "00900", "zone_reference": zone}
            for tbl in ("sbr_results", "br_results"):
                _db.ingest_shard_table(central, shard_alias="shard",
                                       shard_table=tbl, central_table=tbl,
                                       bookkeeping=bk)
            _db.mark_zone(central, "00900", zone, "ok")
            central.execute("COMMIT")
        except Exception:
            central.execute("ROLLBACK")
            raise
    finally:
        central.execute("DETACH shard")


def test_phase_a_ingest_creates_and_extends_central_tables(tmp_path):
    s1, s2 = tmp_path / "z1.duckdb", tmp_path / "z2.duckdb"
    _make_shard(s1,
                {"single_building_reference": ["A_1"], "sbr__h": [3.0],
                 "sbr__geom": [box(0, 0, 1, 1)]},
                {"building_reference": ["A"], "br__floors": [1],
                 "br__dict": [{"a": 1}]})
    # Second zone: a new column, integer→double drift and primitive→BLOB drift.
    _make_shard(s2,
                {"single_building_reference": ["B_1"], "sbr__h": [4.5],
                 "sbr__geom": [box(1, 1, 2, 2)], "sbr__new": ["n"]},
                {"building_reference": ["B"], "br__floors": [2.5],
                 "br__dict": ["plain"]})

    central = _db.open_db(str(tmp_path / "central"))
    try:
        _ingest(central, s1, "Z1")  # central tables don't exist yet
        _ingest(central, s2, "Z2")
        assert _db.completed_zones(central, "00900") == {"Z1", "Z2"}
        sbr = _db.read_df_safe(central, "sbr_results").sort_values(
            "single_building_reference").reset_index(drop=True)
        br = _db.read_df_safe(central, "br_results").sort_values(
            "building_reference").reset_index(drop=True)
    finally:
        central.close()

    assert list(sbr["zone_reference"]) == ["Z1", "Z2"]
    assert sbr["sbr__geom"].iloc[1].equals(box(1, 1, 2, 2))
    assert pd.isna(sbr["sbr__new"].iloc[0]) and sbr["sbr__new"].iloc[1] == "n"
    assert list(br["br__floors"]) == [1.0, 2.5]
    assert br["br__dict"].iloc[0] == {"a": 1} and br["br__dict"].iloc[1] == "plain"


def test_phase_a_widens_primitive_central_column_to_blob(tmp_path):
    s1, s2 = tmp_path / "z1.duckdb", tmp_path / "z2.duckdb"
    _make_shard(s1, {"single_building_reference": ["A_1"], "sbr__v": ["text"]},
                {"building_reference": ["A"]})
    _make_shard(s2, {"single_building_reference": ["B_1"], "sbr__v": [{"d": 1}]},
                {"building_reference": ["B"]})
    central = _db.open_db(str(tmp_path / "central"))
    try:
        _ingest(central, s1, "Z1")
        _ingest(central, s2, "Z2")  # VARCHAR central vs pickled-BLOB shard
        types = {r[0]: r[1] for r in
                 central.execute('DESCRIBE "sbr_results"').fetchall()}
        rows = central.execute(
            'SELECT single_building_reference, sbr__v FROM "sbr_results" '
            'ORDER BY 1').fetchall()
    finally:
        central.close()
    assert types["sbr__v"] == "BLOB"
    assert bytes(rows[0][1]) == b"text"
    assert _db._maybe_unpickle(rows[1][1]) == {"d": 1}


def test_phase_a_failed_shard_rolls_back_cleanly(tmp_path):
    good, bad = tmp_path / "good.duckdb", tmp_path / "bad.duckdb"
    _make_shard(good, {"single_building_reference": ["A_1"]},
                {"building_reference": ["A"]})
    _make_shard(bad, {"single_building_reference": ["B_1"]},
                {"building_reference": ["B"]})
    con = duckdb.connect(str(bad))
    con.execute("DROP TABLE br_results")  # second ingest of the zone will fail
    con.close()

    central = _db.open_db(str(tmp_path / "central"))
    try:
        _ingest(central, good, "Z1")
        with pytest.raises(Exception):
            _ingest(central, bad, "Z2")
        # Z2's sbr rows were rolled back with the failed transaction and the
        # connection is usable for the next zone.
        assert _db.completed_zones(central, "00900") == {"Z1"}
        n = central.execute('SELECT count(*) FROM "sbr_results"').fetchall()[0][0]
        assert n == 1
    finally:
        central.close()


# ---------------------------------------------------------------------------
# CAT caches
# ---------------------------------------------------------------------------

def test_cat_attrs_cache_roundtrip_and_refresh(tmp_path):
    rd = str(tmp_path)
    df = pd.DataFrame({"building_reference": ["A", "B"],
                       "br__uses": [{"Residential": 1}, {"Commercial": 2}]})
    _db.cat_attrs_write(rd, "00900", df)
    _db.cat_attrs_write(rd, "00900", df.iloc[:1])  # refresh replaces rows
    _db.cat_attrs_write(rd, "08019", df)
    out = _db.cat_attrs_read(rd, "00900")
    assert list(out["building_reference"]) == ["A"]
    assert out["br__uses"].iloc[0] == {"Residential": 1}
    assert _db.cat_cache_clear(rd, "00900") == 1
    assert _db.cat_attrs_read(rd, "00900") is None
    assert len(_db.cat_attrs_read(rd, "08019")) == 2


# ---------------------------------------------------------------------------
# Mixed-type columns (shard type drift) — decoding must not depend on row order
# ---------------------------------------------------------------------------

def test_mixed_str_float_column_roundtrips(tmp_path):
    """A column holding str in some rows and float in others (as
    br__parcel_main_orientation does across zones) used to crash write_gdf
    or reach DuckDB half-coerced, depending on which value came first."""
    for values in (["165", 125.0, None], [125.0, "165", None]):
        gdf = _gdf(["A", "B", "C"], extra={"orient": values})
        con = _db.open_db(str(tmp_path / str(values[0])))
        try:
            _db.write_gdf(con, "buildings", gdf)
            out = _db.read_gdf(con, "buildings").set_index("building_reference")
            _db.write_gdf(con, "buildings2", out.reset_index())  # re-write
            back = _db.read_df_safe(con, "buildings").set_index("building_reference")
        finally:
            con.close()
        assert out.loc["A", "orient"] == values[0]
        assert out.loc["B", "orient"] == values[1]
        assert pd.isna(out.loc["C", "orient"])
        assert back.loc["A", "orient"] == values[0]


@pytest.mark.parametrize("text_first", [True, False])
def test_widened_blob_column_decodes_every_row(tmp_path, text_first):
    """A VARCHAR column widened to BLOB by ingest_shard_table holds
    ENCODE()d text from early shards and pickles from later ones."""
    import pickle
    text = ("A", b"165")
    pick = ("B", pickle.dumps(125.0, protocol=pickle.HIGHEST_PROTOCOL))
    rows = [text, pick] if text_first else [pick, text]
    con = _db.open_db(str(tmp_path))
    try:
        con.execute('CREATE TABLE t (building_reference VARCHAR, v BLOB, '
                    'geometry GEOMETRY)')
        for ref, blob in rows:
            con.execute("INSERT INTO t VALUES (?, ?, ST_Point(0, 0))", [ref, blob])
        gdf = _db.read_gdf(con, "t").set_index("building_reference")
        df = _db.read_df_safe(con, "t").set_index("building_reference")
        chunks = pd.concat(list(_db.read_df_safe_chunks(con, "t"))
                           ).set_index("building_reference")
    finally:
        con.close()
    assert gdf.loc["A", "v"] == "165" and gdf.loc["B", "v"] == 125.0
    # read_df_safe keeps non-pickle BLOBs raw but must still unpickle the rest.
    for frame in (df, chunks):
        assert frame.loc["B", "v"] == 125.0
        assert bytes(frame.loc["A", "v"]) == b"165"
