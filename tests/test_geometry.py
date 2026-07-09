"""Pure-geometry helpers behind the energy footprint engine.

These are area-conserving polygon operations, so they can be property-tested
with synthetic shapes and no cadastral data.
"""
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from hypercadaster_ES import utils
from hypercadaster_ES import energy


def _square(side=20.0):
    return Polygon([(0, 0), (side, 0), (side, side), (0, side)])


def test_subdivide_weighted_conserves_area_and_count():
    poly = _square(20.0)
    weights = [1.0, 1.0, 2.0]
    regs = utils._subdivide_polygon_weighted_oriented(poly, weights)
    regs = [r for r in regs if r is not None and not r.is_empty]
    assert len(regs) == len(weights)
    total = unary_union(regs).area
    assert total == pytest.approx(poly.area, rel=0.02)


def test_subdivide_weighted_areas_track_weights():
    poly = _square(30.0)
    weights = [1.0, 3.0]
    regs = utils._subdivide_polygon_weighted_oriented(poly, weights)
    areas = sorted(r.area for r in regs)
    # larger weight -> larger region (roughly 3x)
    assert areas[1] > areas[0]
    assert areas[1] / areas[0] == pytest.approx(3.0, rel=0.15)


def test_carve_central_rect_hits_target_area():
    poly = _square(20.0)          # 400 m2
    core, remainder = utils._carve_central_rect(poly, 100.0)
    assert core is not None and not core.is_empty
    assert core.area == pytest.approx(100.0, rel=0.1)
    assert remainder.area == pytest.approx(poly.area - core.area, rel=0.05)


def test_carve_end_block_hits_target_area():
    poly = Polygon([(0, 0), (40, 0), (40, 10), (0, 10)])  # 400 m2, long axis x
    block, remainder = energy._carve_end_block(poly, 100.0, 0.0)
    assert block is not None and not block.is_empty
    assert block.area == pytest.approx(100.0, rel=0.15)


def test_rationalize_floors_preserves_keys():
    fp = {0: _square(20.0), 1: _square(18.0), 2: _square(16.0)}
    out = energy._rationalize_floors(fp)
    assert set(out.keys()) == set(fp.keys())
    for g in out.values():
        assert g.is_valid
        assert g.area > 0


def test_rationalize_floors_empty():
    assert energy._rationalize_floors({}) == {}
