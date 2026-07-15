"""BuildingParts that declare no floors at all still get a storey.

Taken literally, ``n_floors_above_ground == 0`` yields ``range(0)`` — no floor
footprints — so the building drops out of the per-floor analysis and, further
downstream, out of the CAT integration (which is gated on a non-empty
per-floor frame). Such a part still has a real footprint, so it is read as a
single storey; the CAT floor count then rebuilds the rest.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from hypercadaster_ES import utils


def _square(x0=0.0, y0=0.0, side=10.0):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def _parts(rows):
    """rows: list of (ref, n_above, n_below, geometry)."""
    gdf = gpd.GeoDataFrame({
        "single_building_reference": [r[0] for r in rows],
        "n_floors_above_ground": [r[1] for r in rows],
        "n_floors_below_ground": [r[2] for r in rows],
        "building_part_geometry": [r[3] for r in rows],
    }, geometry="building_part_geometry", crs="EPSG:25831")
    return gdf


def _footprints(rows):
    return utils.calculate_floor_footprints(
        gdf=_parts(rows), group_by="single_building_reference",
        geometry_name="building_part_geometry", min_hole_area=0.5,
        gap_tolerance=0.05, num_workers=1)


def test_building_with_no_floors_at_all_becomes_one_storey():
    fp = _footprints([("B1", 0, 0, _square(side=10.0))])
    assert sorted(fp["floor"]) == [0]
    assert fp.geometry.area.sum() == pytest.approx(100.0, rel=0.01)


def test_normal_building_is_unaffected():
    fp = _footprints([("B1", 3, 0, _square(side=10.0))])
    assert sorted(fp["floor"]) == [0, 1, 2]


def test_zero_floor_part_inside_a_real_building_stays_a_canopy():
    """A 0-floor part alongside parts that have floors is a canopy/porch: it
    must not become a storey, nor contribute to the floor footprints."""
    fp = _footprints([
        ("B1", 0, 0, _square(x0=20, side=10.0)),   # canopy, detached area
        ("B1", 2, 0, _square(side=10.0)),
    ])
    assert sorted(fp["floor"]) == [0, 1]
    # only the 2-storey part's 100 m² counts on each floor — not the canopy
    for _, row in fp.iterrows():
        assert row.geometry.area == pytest.approx(100.0, rel=0.01)


def test_underground_only_structure_gains_no_ground_floor():
    fp = _footprints([("B1", 0, 2, _square(side=10.0))])
    assert 0 not in set(fp["floor"])
    assert sorted(fp["floor"]) == [-2, -1]


def test_cat_floor_count_is_replicated_onto_the_synthesized_storey():
    """The point of reading it as one storey: _augment_building_floors_from_cat
    can then rebuild the floors CAT reports, buffered to their CAT area."""
    from hypercadaster_ES import building_inference as bi

    fp = _footprints([("B1", 0, 0, _square(side=10.0))])
    assert sorted(fp["floor"]) == [0]

    # CAT reports three above-ground floors for this building
    cat_area = {0: 100.0, 1: 100.0, 2: 64.0}
    out = bi._augment_building_floors_from_cat(fp, "B1", cat_area, share=1.0)

    assert sorted(out["floor"]) == [0, 1, 2]
    areas = dict(zip(out["floor"], out.geometry.area))
    assert areas[1] == pytest.approx(100.0, rel=0.05)
    assert areas[2] == pytest.approx(64.0, rel=0.05)
