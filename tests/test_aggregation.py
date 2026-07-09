"""Grid and H3 aggregation of building-level GeoDataFrames."""
import geopandas as gpd
import pytest

import hypercadaster_ES as hc


def test_grid_two_cells(building_gdf):
    grid = hc.mergers.aggregate_to_grid(building_gdf, cell_size_m=100)
    # ten buildings split into exactly two 100 m cells (6 + 4)
    assert len(grid) == 2
    assert set(grid["n_buildings"]) == {6, 4}
    # every building is accounted for
    assert grid["n_buildings"].sum() == len(building_gdf)
    # dwellings are summed (default agg)
    assert grid["n_dwellings"].sum() == building_gdf["n_dwellings"].sum()
    assert grid.crs == building_gdf.crs
    assert (grid.geometry.geom_type == "Polygon").all()


def test_grid_requires_projected_crs(building_gdf):
    wgs = building_gdf.copy()
    wgs["building_centroid"] = wgs["building_centroid"].to_crs("EPSG:4326")
    with pytest.raises(ValueError):
        hc.mergers.aggregate_to_grid(wgs, cell_size_m=100)


def test_grid_empty_input(building_gdf):
    empty = building_gdf.iloc[0:0]
    out = hc.mergers.aggregate_to_grid(empty)
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 0


def test_grid_custom_aggregation(building_gdf):
    grid = hc.mergers.aggregate_to_grid(
        building_gdf, cell_size_m=100,
        aggregations={"building_area": "max"},
    )
    assert grid["building_area"].max() == building_gdf["building_area"].max()


def test_h3_aggregation(building_gdf):
    h3 = pytest.importorskip("h3")  # noqa: F841
    out = hc.mergers.aggregate_to_h3(building_gdf, resolution=10)
    assert out["n_buildings"].sum() == len(building_gdf)
    assert out.crs.to_epsg() == 4326
    assert (out.geometry.geom_type == "Polygon").all()
    assert "h3_id" in out.columns
