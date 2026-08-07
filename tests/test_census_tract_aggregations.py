"""Offline unit tests for the ``ctagg__*`` census-tract aggregations
(``mergers.join_census_tract_aggregations``, wired into ``hc.merge``).

These are the tract-level quantities the energy simulations need but cannot
derive from one building — above all the occupancy density that drives the DHW
demand. The point of computing them at merge time is that a frame later filtered
down to a single building still carries its tract's context, so the tests below
check both the arithmetic and that degradation is a NaN, never a wrong number.
"""
import numpy as np
import pandas as pd
import pytest

from hypercadaster_ES import mergers

PREFIX = mergers.CTAGG_PREFIX


def _frame(buildings, population=None, census_dwellings=None, tract="0801901001"):
    """``buildings`` is a list of per-building dwelling-area lists."""
    n = len(buildings)
    data = {
        "building_reference": [f"B{i}" for i in range(n)],
        "census_tract_code": [tract] * n,
        "br__building_spaces_reference": [
            {"Residential": [f"B{i}_{j}" for j in range(len(areas))]}
            for i, areas in enumerate(buildings)],
        "br__building_spaces_area_without_communal": [
            {"Residential": list(areas)} for areas in buildings],
        "br__building_spaces_floors": [
            {"Residential": [[0]] * len(areas)} for areas in buildings],
    }
    if population is not None:
        data["so__population_2025"] = [float(population)] * n
    if census_dwellings is not None:
        data["so__dwellings_main_2021"] = [float(census_dwellings)] * n
    return pd.DataFrame(data)


def test_counts_areas_and_densities():
    # 3 buildings, 6 dwellings, 600 m², 18 people → 0.03 people/m², 3 per dwelling.
    gdf = _frame([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
                 population=18, census_dwellings=6)
    out = mergers.join_census_tract_aggregations(gdf, verbose=False)
    row = out.iloc[0]
    assert row[f"{PREFIX}buildings"] == 3
    assert row[f"{PREFIX}residential_buildings"] == 3
    assert row[f"{PREFIX}residential_area_m2"] == pytest.approx(600.0)
    assert row[f"{PREFIX}dwellings"] == 6
    assert row[f"{PREFIX}population"] == pytest.approx(18.0)
    assert row[f"{PREFIX}people_per_m2"] == pytest.approx(0.03)
    assert row[f"{PREFIX}people_per_dwelling"] == pytest.approx(3.0)
    assert row[f"{PREFIX}dwelling_coverage"] == pytest.approx(1.0)
    assert bool(row[f"{PREFIX}complete_tract"]) is True
    # Every building of the tract carries the same tract-level values.
    for col in (f"{PREFIX}people_per_m2", f"{PREFIX}dwellings"):
        assert out[col].nunique() == 1


def test_every_row_carries_it_so_one_row_survives_filtering():
    # This is the whole reason the aggregation happens at merge time: the density
    # is a property of the tract, and a single row must keep it.
    gdf = _frame([[100.0]] * 10, population=30, census_dwellings=10)
    out = mergers.join_census_tract_aggregations(gdf, verbose=False)
    one = out[out["building_reference"] == "B7"]
    assert len(one) == 1
    assert one.iloc[0][f"{PREFIX}people_per_m2"] == pytest.approx(0.03)
    assert one.iloc[0][f"{PREFIX}residential_area_m2"] == pytest.approx(1000.0)


def test_partial_tract_publishes_no_density():
    # A frame holding 2 of the tract's 100 census dwellings would otherwise report
    # 30 people over 200 m² = 0.15 people/m², five times the truth. Counts are
    # still emitted; the densities are NaN, and the tract is flagged incomplete.
    gdf = _frame([[100.0], [100.0]], population=30, census_dwellings=100)
    row = mergers.join_census_tract_aggregations(gdf, verbose=False).iloc[0]
    assert row[f"{PREFIX}dwellings"] == 2
    assert row[f"{PREFIX}residential_area_m2"] == pytest.approx(200.0)
    assert row[f"{PREFIX}dwelling_coverage"] == pytest.approx(0.02)
    assert bool(row[f"{PREFIX}complete_tract"]) is False
    assert np.isnan(row[f"{PREFIX}people_per_m2"])
    assert np.isnan(row[f"{PREFIX}people_per_dwelling"])


def test_without_the_census_yardstick_the_frame_is_trusted():
    # No so__dwellings_main column → nothing to check coverage against, so the
    # frame is assumed to hold the tract (what a whole-municipality merge does).
    gdf = _frame([[100.0], [100.0]], population=6)
    row = mergers.join_census_tract_aggregations(gdf, verbose=False).iloc[0]
    assert np.isnan(row[f"{PREFIX}dwelling_coverage"])
    assert bool(row[f"{PREFIX}complete_tract"]) is True
    assert row[f"{PREFIX}people_per_m2"] == pytest.approx(0.03)


def test_without_population_only_the_counts_are_emitted():
    gdf = _frame([[100.0], [100.0]])
    row = mergers.join_census_tract_aggregations(gdf, verbose=False).iloc[0]
    assert row[f"{PREFIX}dwellings"] == 2
    assert row[f"{PREFIX}residential_area_m2"] == pytest.approx(200.0)
    assert np.isnan(row[f"{PREFIX}population"])
    assert np.isnan(row[f"{PREFIX}people_per_m2"])


def test_tracts_are_aggregated_independently():
    a = _frame([[100.0], [100.0]], population=6, tract="0801901001")
    b = _frame([[50.0]], population=5, tract="0801901002")
    out = mergers.join_census_tract_aggregations(pd.concat([a, b], ignore_index=True),
                                                 verbose=False)
    by_tract = out.groupby("census_tract_code")[f"{PREFIX}people_per_m2"].first()
    assert by_tract["0801901001"] == pytest.approx(0.03)     # 6 / 200
    assert by_tract["0801901002"] == pytest.approx(0.10)     # 5 / 50


def test_non_residential_space_is_not_counted():
    # Parking/commercial floor area must not dilute the dwelling density.
    gdf = _frame([[100.0]], population=3)
    gdf.at[0, "br__building_spaces_reference"] = {
        "Residential": ["B0_1"], "Warehouse - Parking": ["B0_P"]}
    gdf.at[0, "br__building_spaces_area_without_communal"] = {
        "Residential": [100.0], "Warehouse - Parking": [900.0]}
    gdf.at[0, "br__building_spaces_floors"] = {
        "Residential": [[0]], "Warehouse - Parking": [[-1]]}
    row = mergers.join_census_tract_aggregations(gdf, verbose=False).iloc[0]
    assert row[f"{PREFIX}residential_area_m2"] == pytest.approx(100.0)
    assert row[f"{PREFIX}dwellings"] == 1
    assert row[f"{PREFIX}people_per_m2"] == pytest.approx(0.03)


def test_missing_inputs_are_a_no_op():
    # No census tract, or no CAT building spaces → the frame comes back untouched.
    plain = pd.DataFrame({"building_reference": ["A"]})
    assert list(mergers.join_census_tract_aggregations(plain, verbose=False).columns) \
        == ["building_reference"]
    no_spaces = pd.DataFrame({"building_reference": ["A"],
                              "census_tract_code": ["0801901001"]})
    out = mergers.join_census_tract_aggregations(no_spaces, verbose=False)
    assert not [c for c in out.columns if c.startswith(PREFIX)]


def test_rerunning_replaces_rather_than_suffixes():
    # Re-merging must overwrite the columns, not produce _x/_y pairs.
    gdf = _frame([[100.0], [100.0]], population=6)
    once = mergers.join_census_tract_aggregations(gdf, verbose=False)
    twice = mergers.join_census_tract_aggregations(once, verbose=False)
    assert sorted(c for c in twice.columns if c.startswith(PREFIX)) == \
        sorted(c for c in once.columns if c.startswith(PREFIX))
    assert not [c for c in twice.columns if c.endswith(("_x", "_y"))]
    assert len(twice) == len(once)


def test_compacted_list_tract_codes_are_unwrapped():
    # After address compaction a census_tract_code can arrive as a 1-element list.
    gdf = _frame([[100.0], [100.0]], population=6)
    gdf["census_tract_code"] = [["0801901001"], ["0801901001"]]
    row = mergers.join_census_tract_aggregations(gdf, verbose=False).iloc[0]
    assert row[f"{PREFIX}buildings"] == 2
    assert row[f"{PREFIX}people_per_m2"] == pytest.approx(0.03)
