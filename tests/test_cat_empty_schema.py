"""Empty CAT slices must still carry the full aggregation schema.

Buildings with no CAT record (rural parcels, undeveloped "solares") reach
``aggregate_CAT_file_building_spaces`` with an empty frame. pandas'
``groupby(...).apply(...)`` skips the aggregation function entirely in that
case and passes the *input* columns through, so without an explicit empty-input
schema the downstream ``br__`` aggregation raises ``KeyError`` and takes the
whole cadastral zone down with it.
"""
import pandas as pd

from hypercadaster_ES import building_inference as bi


def _cat_frame(ref="A1", n=4):
    """One building over four floors — the per-building slice the aggregation
    expects (it groups by use type, not by building)."""
    return pd.DataFrame({
        "building_reference": [ref] * n,
        "building_space_reference": [f"{ref}S{i}" for i in range(n)],
        "building_space_address": [f"ADDR_{ref}"] * n,
        "building_space_inner_address": ["1", "2", "3", "4"][:n],
        "building_space_floor_name": ["00", "01", "02", "03"][:n],
        "building_space_floors": [1.0] * n,
        "building_space_area_with_communal": [80.0, 90.0, 70.0, 60.0][:n],
        "building_space_area_without_communal": [70.0, 80.0, 60.0, 50.0][:n],
        "building_space_communal_area": [10.0] * n,
        "building_space_economic_value": [100000.0, 110000.0, 90000.0, 80000.0][:n],
        "building_space_effective_year": [1990.0, 1990.0, 2001.0, 2001.0][:n],
        "building_space_typology_category": ["A", "A", "B", "B"][:n],
        "building_space_category": ["A", "A", "B", "B"][:n],
        "building_space_detailed_use_type": ["Residential"] * n,
        "building_space_inferred_use_type": ["Residential"] * n,
        "building_space_related_area": [0.0] * n,
    })


def test_empty_input_matches_populated_schema():
    populated = bi.aggregate_CAT_file_building_spaces(_cat_frame())
    empty = bi.aggregate_CAT_file_building_spaces(_cat_frame().iloc[0:0])

    for key in ("by_floors", "building"):
        assert len(empty[key]) == 0
        # The declared empty schema must not drift from the real output.
        assert list(empty[key].columns) == list(populated[key].columns), key


def test_empty_input_survives_br_aggregation():
    """The regression: this used to raise KeyError('br__building_spaces_by_floor')."""
    empty = bi.aggregate_CAT_file_building_spaces(pd.DataFrame())
    out = bi._br_aggregate_from_agg_CAT(empty)

    assert len(out) == 0
    assert "building_reference" in out.columns
    assert "br__building_spaces_by_floor" in out.columns

    populated = bi._br_aggregate_from_agg_CAT(
        bi.aggregate_CAT_file_building_spaces(_cat_frame()))
    assert set(out.columns) == set(populated.columns)


def test_none_input_is_treated_as_empty():
    agg = bi.aggregate_CAT_file_building_spaces(None)
    assert len(agg["by_floors"]) == 0 and len(agg["building"]) == 0


def test_populated_aggregation_still_works():
    """A building with CAT data aggregates as before, and a mix of buildings
    with and without CAT data reduces to the with-CAT ones only."""
    agg = bi.aggregate_CAT_file_building_spaces(_cat_frame("A1"))
    out = bi._br_aggregate_from_agg_CAT(agg)
    assert list(out["building_reference"]) == ["A1"]
    # four spaces on a single use type
    assert [sum(v.values()) for v in out["br__building_spaces"]] == [4]

    empty = bi.aggregate_CAT_file_building_spaces(_cat_frame("A2").iloc[0:0])
    combined = bi._br_aggregate_from_agg_CAT({
        key: pd.concat([agg[key], empty[key]], ignore_index=True)
        for key in ("by_floors", "building")
    })
    assert list(combined["building_reference"]) == ["A1"]
