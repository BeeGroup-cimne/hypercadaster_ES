"""Theoretical single building space for buildings absent from the CAT file.

Rural/agricultural buildings exist in the INSPIRE Building layer but have no
CAT construction record, so they have no building spaces and no thermal zones.
``add_theoretical_CAT_spaces`` gives each one a single space carrying the
building's main usage and official built area.
"""
import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon

from hypercadaster_ES import building_inference as bi


def _square(x0, y0, side):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


@pytest.fixture
def cat():
    """A CAT frame knowing only about building B1."""
    return pl.DataFrame({
        "building_reference": ["B1", "B1"],
        "building_space_reference": ["B1S1", "B1S2"],
        "building_space_address": ["ADDR1", "ADDR1"],
        "building_space_inner_address": ["1", "2"],
        "building_space_floor_name": ["00", "01"],
        "building_space_year": pl.Series([1990, 1990], dtype=pl.Int16),
        "building_space_effective_year": pl.Series([1990, 1990], dtype=pl.Int16),
        "building_space_total_area": pl.Series([50.0, 50.0], dtype=pl.Float32),
        "building_space_total_area_without_communal": pl.Series([45.0, 45.0], dtype=pl.Float32),
        "building_space_area_with_communal": pl.Series([50.0, 50.0], dtype=pl.Float32),
        "building_space_area_without_communal": pl.Series([45.0, 45.0], dtype=pl.Float32),
        "building_space_communal_area": pl.Series([5.0, 5.0], dtype=pl.Float32),
        "building_spaces_considered": pl.Series([1, 1], dtype=pl.UInt32),
        "building_space_use_type": ["V", "V"],
        "building_space_typology_category": ["A", "A"],
        "building_space_detailed_use_type": ["Housing", "Housing"],
        "building_space_inferred_use_type": ["Residential", "Residential"],
        "building_space_economic_value": [1000.0, 1000.0],
        "cadaster_code": ["25900", "25900"],
    })


@pytest.fixture
def gdf():
    """B1 has CAT data; B2 (agriculture) and B3 (no usage) do not."""
    return gpd.GeoDataFrame({
        "building_reference": ["B1", "B2", "B3"],
        "building_use": ["1_residential", "2_agriculture", None],
        "building_area": [100.0, 108.0, 240.0],
        "year_of_construction": [1990, 1996, None],
        "building_geometry": [_square(0, 0, 10), _square(50, 0, 12), _square(100, 0, 20)],
    }, geometry="building_geometry", crs="EPSG:25831")


def test_adds_one_space_per_building_missing_from_CAT(cat, gdf):
    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")

    assert out.height == cat.height + 2  # B2 and B3 only
    added = out.filter(pl.col("building_reference").is_in(["B2", "B3"]))
    assert sorted(added["building_reference"]) == ["B2", "B3"]
    # the building IS its only space
    assert added.filter(pl.col("building_reference") == "B2")["building_space_reference"][0] == "B2"


def test_preserves_CAT_schema_and_leaves_known_buildings_untouched(cat, gdf):
    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")

    assert out.columns == cat.columns
    assert out.schema == cat.schema
    # B1's real CAT rows are unchanged
    assert out.filter(pl.col("building_reference") == "B1").equals(cat)


def test_usage_is_mapped_from_the_building_and_defaults_to_unknown(cat, gdf):
    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")

    def use(ref):
        return out.filter(pl.col("building_reference") == ref)["building_space_inferred_use_type"][0]

    assert use("B2") == "Agricultural"   # 2_agriculture
    assert use("B3") == "Unknown"        # building_use is null


def test_area_comes_from_the_official_built_area_all_private(cat, gdf):
    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")
    b2 = out.filter(pl.col("building_reference") == "B2")

    assert b2["building_space_area_with_communal"][0] == pytest.approx(108.0)
    assert b2["building_space_area_without_communal"][0] == pytest.approx(108.0)
    assert b2["building_space_communal_area"][0] == 0.0
    assert b2["building_space_effective_year"][0] == 1996


def test_area_falls_back_to_the_footprint_when_the_cadaster_area_is_missing(cat, gdf):
    gdf.loc[gdf["building_reference"] == "B2", "building_area"] = None
    gdf.loc[gdf["building_reference"] == "B3", "building_area"] = 0.0

    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")
    area = {r: out.filter(pl.col("building_reference") == r)["building_space_area_with_communal"][0]
            for r in ("B2", "B3")}
    assert area["B2"] == pytest.approx(12 * 12)  # footprint of B2
    assert area["B3"] == pytest.approx(20 * 20)  # footprint of B3


def test_theoretical_space_yields_one_thermal_zone(cat, gdf):
    """The point of the exercise: these buildings become modellable."""
    out = bi.add_theoretical_CAT_spaces(cat, gdf, "25900")
    sl = out.filter(pl.col("building_reference") == "B2").to_pandas()

    br = bi._br_aggregate_from_agg_CAT(bi.aggregate_CAT_file_building_spaces(sl))
    zones = br["br__thermal_zones"].iloc[0]

    assert len(zones) == 1
    assert zones[0]["use_type"] == "Agricultural"
    assert zones[0]["reference"] == "B2"
    assert zones[0]["floors"] == [0]
    assert zones[0]["area_with_communals"] == pytest.approx(108.0)


def test_noop_without_CAT_or_buildings(cat, gdf):
    assert bi.add_theoretical_CAT_spaces(None, gdf, "25900") is None
    empty = gdf.iloc[0:0]
    assert bi.add_theoretical_CAT_spaces(cat, empty, "25900").height == cat.height
    # every building already has CAT data -> nothing to add
    only_b1 = gdf[gdf["building_reference"] == "B1"]
    assert bi.add_theoretical_CAT_spaces(cat, only_b1, "25900").height == cat.height
