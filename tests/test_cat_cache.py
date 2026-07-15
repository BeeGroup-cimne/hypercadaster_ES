"""Tests for the CAT-file attribute cache in the central DuckDB.

The parsed per-space CAT table (produced by
``parse_horizontal_division_buildings_CAT_files``) is cached in
``cat_building_spaces`` keyed by ``cadaster_code`` so a re-run skips the slow
CAT parse. These tests cover the read/write round-trip fidelity, per-municipality
isolation, idempotent refresh and graceful degradation.
"""
import pandas as pd
import polars as pl
import pytest

from hypercadaster_ES import _db


def _sample_frame(refs):
    """A frame resembling buildings_CAT: mixed dtypes, nulls, no geometry."""
    n = len(refs)
    return pl.DataFrame({
        "building_reference": refs,
        "building_space_reference": [f"{r}0001" for r in refs],
        "building_space_total_area": [float(i) * 10.0 for i in range(n)],
        "building_space_year": [1990 + i for i in range(n)],
        "building_space_inferred_use_type": (["Residential", "Commercial"] * n)[:n],
        "building_space_area_without_communal": [None if i % 2 else float(i)
                                                 for i in range(n)],
    })


def test_cat_cache_roundtrip_preserves_schema_and_values(tmp_path):
    results_dir = str(tmp_path)
    df = _sample_frame(["A1", "A2", "A3"])

    assert _db.cat_cache_read(results_dir, "00900") is None  # nothing yet
    _db.cat_cache_write(results_dir, "00900", df)
    out = _db.cat_cache_read(results_dir, "00900")

    assert out is not None
    # cadaster_code is an internal key, not part of the returned frame
    assert "cadaster_code" not in out.columns
    # same columns, dtypes and values (column order is not significant)
    assert set(out.columns) == set(df.columns)
    cols = sorted(df.columns)
    assert out.select(cols).equals(df.select(cols))


def test_cat_cache_isolates_municipalities(tmp_path):
    results_dir = str(tmp_path)
    _db.cat_cache_write(results_dir, "00900", _sample_frame(["A1", "A2"]))
    _db.cat_cache_write(results_dir, "25120", _sample_frame(["B1", "B2", "B3"]))

    a = _db.cat_cache_read(results_dir, "00900")
    b = _db.cat_cache_read(results_dir, "25120")
    assert sorted(a["building_reference"].to_list()) == ["A1", "A2"]
    assert sorted(b["building_reference"].to_list()) == ["B1", "B2", "B3"]


def test_cat_cache_write_is_idempotent_refresh(tmp_path):
    results_dir = str(tmp_path)
    _db.cat_cache_write(results_dir, "00900", _sample_frame(["A1", "A2", "A3"]))
    # re-parse of the same municipality replaces its rows (no duplication)
    _db.cat_cache_write(results_dir, "00900", _sample_frame(["A1"]))
    out = _db.cat_cache_read(results_dir, "00900")
    assert out.height == 1
    assert out["building_reference"].to_list() == ["A1"]


def test_cat_cache_read_missing_db_returns_none(tmp_path):
    # No DuckDB file at all -> graceful None, never raises.
    assert _db.cat_cache_read(str(tmp_path / "nope"), "00900") is None


def test_cat_cache_write_empty_is_noop(tmp_path):
    results_dir = str(tmp_path)
    _db.cat_cache_write(results_dir, "00900", None)
    _db.cat_cache_write(results_dir, "00900", _sample_frame([]))
    assert _db.cat_cache_read(results_dir, "00900") is None


# ---- per-building attribute cache (br_attributes_from_CAT output) -----------

def _attrs_frame(refs):
    """A frame resembling br_attributes_from_CAT: primitives + complex dict/list
    columns (the ``br__*`` per-use-type aggregates) that must round-trip."""
    n = len(refs)
    return pd.DataFrame({
        "building_reference": refs,
        "br__area_with_communals": [float(i) * 100.0 for i in range(n)],
        "br__building_spaces": [3 + i for i in range(n)],
        "br__building_spaces_by_floor": [{0: 2, 1: 1 + i} for i in range(n)],
        "br__use_types": [["Residential", "Commercial"][: (i % 2) + 1] for i in range(n)],
    })


def test_cat_attrs_roundtrip_preserves_complex_columns(tmp_path):
    results_dir = str(tmp_path)
    df = _attrs_frame(["A1", "A2", "A3"])

    assert _db.cat_attrs_read(results_dir, "00900") is None
    _db.cat_attrs_write(results_dir, "00900", df)
    out = _db.cat_attrs_read(results_dir, "00900")

    assert out is not None
    assert "cadaster_code" not in out.columns
    assert set(out.columns) == set(df.columns)
    a = df.sort_values("building_reference").reset_index(drop=True)
    b = out[df.columns].sort_values("building_reference").reset_index(drop=True)
    # complex dict/list columns survive the pickle round-trip
    assert list(b["br__building_spaces_by_floor"]) == list(a["br__building_spaces_by_floor"])
    assert list(b["br__use_types"]) == list(a["br__use_types"])
    assert b["br__area_with_communals"].equals(a["br__area_with_communals"])


def test_cat_attrs_isolation_and_idempotent_refresh(tmp_path):
    results_dir = str(tmp_path)
    _db.cat_attrs_write(results_dir, "00900", _attrs_frame(["A1", "A2"]))
    _db.cat_attrs_write(results_dir, "25120", _attrs_frame(["B1"]))
    # refresh 00900 with fewer rows -> replaced, not duplicated
    _db.cat_attrs_write(results_dir, "00900", _attrs_frame(["A1"]))

    a = _db.cat_attrs_read(results_dir, "00900")
    b = _db.cat_attrs_read(results_dir, "25120")
    assert a["building_reference"].tolist() == ["A1"]
    assert b["building_reference"].tolist() == ["B1"]


def test_cat_cache_clear_removes_both_caches(tmp_path):
    results_dir = str(tmp_path)
    _db.cat_cache_write(results_dir, "00900", _sample_frame(["A1", "A2"]))
    _db.cat_attrs_write(results_dir, "00900", _attrs_frame(["A1", "A2"]))

    removed = _db.cat_cache_clear(results_dir, "00900")
    assert removed == 4  # 2 space rows + 2 attribute rows
    assert _db.cat_cache_read(results_dir, "00900") is None
    assert _db.cat_attrs_read(results_dir, "00900") is None
