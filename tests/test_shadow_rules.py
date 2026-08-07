"""Shadow-analysis conventions: the >=5-floors ground-floor rule and the 75%
terrain-coverage threshold for terrain contour shadows."""
import math

import pytest
from shapely.geometry import LineString, Polygon

from hypercadaster_ES import building_inference as bi
from hypercadaster_ES import utils


def _dem_const(val):
    return lambda xs, ys: [val] * len(xs)


def test_wall_buried_area_fraction():
    seg = LineString([(0, 0), (10, 0)])          # storey elev 0..3 (base 0, top 3)
    assert bi._wall_buried_area_fraction(seg, _dem_const(5.0), 0.0, 3.0) == pytest.approx(1.0)
    assert bi._wall_buried_area_fraction(seg, _dem_const(0.0), 0.0, 3.0) == pytest.approx(0.0)
    assert bi._wall_buried_area_fraction(seg, _dem_const(1.5), 0.0, 3.0) == pytest.approx(0.5)


def test_terrain_contact_needs_75pct_buried():
    # Terrain fully buries the wall over the first ~75% of its length, exposed on
    # the rest → mean buried area fraction ≈ 0.75 → classified terrain_contact.
    seg = LineString([(0, 0), (100, 0)])

    def dem_75(xs, ys):
        return [5.0 if x <= 75 else 0.0 for x in xs]

    def dem_50(xs, ys):
        return [5.0 if x <= 50 else 0.0 for x in xs]

    assert bi._wall_buried_area_fraction(seg, dem_75, 0.0, 3.0) >= bi.TERRAIN_CONTACT_MIN_FRACTION
    assert bi._wall_buried_area_fraction(seg, dem_50, 0.0, 3.0) < bi.TERRAIN_CONTACT_MIN_FRACTION


def test_effective_ground_floor_height_rule():
    # Taller ground floor only for buildings with >= 5 floors above ground.
    assert bi._effective_ground_floor_height(5, 5.0, 3.0) == 5.0
    assert bi._effective_ground_floor_height(8, 5.0, 3.0) == 5.0
    assert bi._effective_ground_floor_height(4, 5.0, 3.0) == 3.0
    assert bi._effective_ground_floor_height(1, 5.0, 3.0) == 3.0
    assert bi._effective_ground_floor_height(None, 5.0, 3.0) == 3.0


def test_industrial_agricultural_force_tall_ground_floor():
    # Industrial and agricultural buildings always get the taller ground floor,
    # even single-storey (machinery / vehicle clearance).
    assert utils.is_tall_ground_floor_use("3_industrial")
    assert utils.is_tall_ground_floor_use("2_agriculture")
    assert not utils.is_tall_ground_floor_use("1_residential")
    assert not utils.is_tall_ground_floor_use(None)
    # 1-floor industrial / agricultural → 5 m; 1-floor residential → 3 m.
    assert utils.effective_ground_floor_height(1, 5.0, 3.0, "3_industrial") == 5.0
    assert utils.effective_ground_floor_height(1, 5.0, 3.0, "2_agriculture") == 5.0
    assert utils.effective_ground_floor_height(1, 5.0, 3.0, "1_residential") == 3.0
    assert utils.effective_ground_floor_height(6, 5.0, 3.0, "1_residential") == 5.0
    # The building_inference wrapper honours the same use override.
    assert bi._effective_ground_floor_height(1, 5.0, 3.0, "3_industrial") == 5.0
    assert bi._effective_ground_floor_height(1, 5.0, 3.0, "1_residential") == 3.0


def _building():
    return Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])


def _dem_high_fraction(fraction):
    """DEM sampler with terrain above eye level in ~``fraction`` of directions."""
    cutoff = 360.0 * fraction

    def _sampler(xs, ys):
        out = []
        for x, y in zip(xs, ys):
            ang = math.degrees(math.atan2(x, y)) % 360.0   # 0=N, 90=E
            out.append(100.0 if ang < cutoff else -10.0)
        return out
    return _sampler


def test_terrain_shadow_ignored_below_75pct_coverage():
    bld = _building()
    # Terrain rises in only ~25% of directions → no terrain contour shadow.
    r = utils.compute_floor_shadow_region(
        bld, [], _dem_high_fraction(0.25), eye_level=2.5, max_distance=100.0)
    assert r.area < 1.0


def test_terrain_shadow_kept_above_75pct_coverage():
    bld = _building()
    # Terrain rises in ~90% of directions → terrain contour shadow present.
    r = utils.compute_floor_shadow_region(
        bld, [], _dem_high_fraction(0.90), eye_level=2.5, max_distance=100.0)
    assert r.area > 1000.0


def test_terrain_coverage_threshold_is_tunable():
    bld = _building()
    dem = _dem_high_fraction(0.5)     # 50% coverage
    # Default 0.75 → dropped; lower the threshold to 0.4 → kept.
    assert utils.compute_floor_shadow_region(
        bld, [], dem, eye_level=2.5, max_distance=100.0).area < 1.0
    assert utils.compute_floor_shadow_region(
        bld, [], dem, eye_level=2.5, max_distance=100.0,
        terrain_coverage_threshold=0.4).area > 1000.0
