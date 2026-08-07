"""Floorplan partition of building spaces into per-floor thermal zones.

Exercises ``energy._partition_floor`` and ``energy.synthesize_space_geometry``
on synthetic buildings (no cadastral data needed), asserting the guarantees the
partition is designed to provide:

  #1 the ground-floor communal core reaches the building entrance;
  #2 the floor tiles exactly (Σ zone areas ≈ footprint) after the appendage
     area-correction — communal stays the authoritative value;
  #3 every Housing zone keeps ≥ 3 m of main-façade frontage.
"""
import math

import pytest
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

from hypercadaster_ES import energy


def _rect(w, h):
    return Polygon([(0, 0), (w, 0), (w, h), (0, h)])


def _long_facades(w, h):
    """Main-façade lines on the two long (y=0 and y=h) sides of a w×h rect."""
    return LineString([(0, 0), (w, 0)]).union(LineString([(0, h), (w, h)]))


def _facade_contact(poly, facade, tol=0.25):
    """Length of ``poly``'s boundary lying on the façade line."""
    return poly.boundary.intersection(facade.buffer(tol)).length


def test_partition_tiles_floor_exactly_and_fronts_facade():
    floor = _rect(24.0, 12.0)                 # 288 m²
    facade = _long_facades(24.0, 12.0)
    present = [
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 60.0, "reference": "A"},
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 60.0, "reference": "B"},
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 60.0, "reference": "C"},
    ]
    communal = 20.0
    priv = sum(s["area"] for s in present)
    appends = max(0.0, floor.area - priv - communal)
    targets = {id(s): s["area"] + (s["area"] / priv) * appends for s in present}

    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, energy._facade_angle(facade, 0.0), facade,
        None, communal, 0.0, (12.0, -1.0), (12.0, 0.0), True,
        energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)

    # Exact tiling: dwellings + communal ≈ footprint.
    tiled = sum(p.area for p in assign.values() if p is not None) + (
        comm.area if comm is not None else 0.0)
    assert tiled == pytest.approx(floor.area, rel=0.02)

    # Every Housing zone fronts the façade with ≥ 3 m and isn't flagged deficient.
    assert not deficit
    for s in present:
        poly = assign[id(s)]
        assert poly is not None and not poly.is_empty
        assert _facade_contact(poly, facade) >= energy._MIN_FACADE_FRONTAGE - 0.1

    # #1 ground-floor communal core reaches the entrance point.
    assert comm is not None
    assert comm.distance(Point(12.0, 0.0)) <= energy._CORRIDOR_WIDTH


def test_too_many_dwellings_flags_facade_deficit():
    # 6 dwellings sharing a single 10 m façade cannot all get 3 m (6*3 > 10).
    floor = _rect(10.0, 8.0)
    facade = LineString([(0, 0), (10, 0)])           # façade only on the bottom
    present = [{"use_type": "Residential", "kind": "occupied", "floors": [0],
                "area": 12.0, "reference": f"D{i}"} for i in range(6)]
    targets = {id(s): s["area"] for s in present}
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, energy._facade_angle(facade, 0.0), facade,
        None, 0.0, 0.0, None, None, False,
        energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    assert deficit                                   # some zones flagged


def test_anchored_mode_places_houses_near_their_address():
    # A 40 m terrace of 4 row houses, addresses along the street (y=0) at
    # x = 5, 15, 25, 35 → each house region should sit at its own address.
    floor = _rect(40.0, 10.0)
    present = []
    for i, x in enumerate([5.0, 15.0, 25.0, 35.0]):
        present.append({"use_type": "Residential", "kind": "occupied",
                        "floors": [0], "area": 100.0, "reference": f"H{i}",
                        "anchor": Point(x, 0.0)})
    targets = {id(s): 100.0 for s in present}
    assert energy._anchors_are_distributed(present, floor)
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, 0.0, None, None, 0.0, 0.0, None, None,
        False, energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    # Each house's region contains its own address, and areas tile exactly.
    for s in present:
        poly = assign[id(s)]
        assert poly is not None and not poly.is_empty
        assert poly.buffer(0.01).contains(Point(s["anchor"].x, 0.5))
    assert sum(p.area for p in assign.values() if p is not None) == pytest.approx(
        floor.area, rel=0.02)


def test_shared_address_stays_in_corridor_mode():
    # Apartment block: all units share the single entrance point → NOT anchored.
    floor = _rect(20.0, 12.0)
    present = [{"use_type": "Residential", "kind": "occupied", "floors": [0],
                "area": 60.0, "reference": f"U{i}", "anchor": Point(10.0, 0.0)}
               for i in range(4)]
    assert not energy._anchors_are_distributed(present, floor)


def test_partition_by_anchors_component_aware():
    # Two detached blocks 50 m apart; two houses each, addressed to their block.
    from shapely.geometry import MultiPolygon
    left = _rect(10.0, 10.0)
    right = Polygon([(60, 0), (70, 0), (70, 10), (60, 10)])
    poly = MultiPolygon([left, right])
    present = [
        {"reference": "L1", "anchor": Point(2, 5)},
        {"reference": "L2", "anchor": Point(8, 5)},
        {"reference": "R1", "anchor": Point(62, 5)},
        {"reference": "R2", "anchor": Point(68, 5)},
    ]
    targets = {id(s): 50.0 for s in present}
    regs = energy._partition_by_anchors(poly, present, targets)
    # Left houses land in the left block, right houses in the right block.
    for s in present:
        r = regs[id(s)]
        assert r is not None and not r.is_empty
        assert r.buffer(0.01).contains(s["anchor"])


def test_pool_reconstruction_for_coarsely_geocoded_houses():
    # A row of 12 houses; the CAT per-space address geocoded only 2 of them, the
    # other 10 collapsed to one building fallback point. The building's own address
    # pool (postal_addresses_location) has 13 spread points → anchors are rebuilt
    # from the pool so the houses spread out instead of clustering.
    fallback = Point(0.0, -5.0)
    refs = {"Residential": [f"H{i}" for i in range(12)]}
    # per-space location: houses 0 and 11 real, the other 10 collapsed to fallback
    locs = {"Residential": [Point(1.0, 0.0)] + [fallback] * 10 + [Point(39.0, 0.0)]}
    pool = [Point(float(x), 0.0) for x in range(1, 40, 3)]   # 13 spread points
    row = {
        "br__building_spaces_reference": refs,
        "br__building_spaces_postal_address_location": locs,
        "postal_addresses_location": pool,
        "br__building_footprint_by_floor": [["s", 0, _rect(40.0, 10.0)]],
    }
    spaces = [{"reference": r} for r in refs["Residential"]]
    resolved = energy._resolve_space_anchors(row, spaces)
    pts = [resolved[id(s)] for s in spaces]
    assert all(p is not None for p in pts)
    # The 6 previously-fallback houses now sit at distinct, spread locations.
    distinct = {(round(p.x, 1), round(p.y, 1)) for p in pts}
    assert len(distinct) >= 6


def test_pool_reconstruction_skipped_when_geocoding_is_good():
    # When most spaces already have distinct real addresses, the pool must NOT
    # override them (that would inject noise into a clean layout).
    refs = {"Residential": [f"H{i}" for i in range(6)]}
    locs = {"Residential": [Point(float(5 * i), 0.0) for i in range(6)]}  # all real
    pool = [Point(float(x), 0.0) for x in range(0, 60, 2)]
    row = {
        "br__building_spaces_reference": refs,
        "br__building_spaces_postal_address_location": locs,
        "postal_addresses_location": pool,
        "br__building_footprint_by_floor": [["s", 0, _rect(40.0, 10.0)]],
    }
    spaces = [{"reference": r} for r in refs["Residential"]]
    resolved = energy._resolve_space_anchors(row, spaces)
    for s, want in zip(spaces, locs["Residential"]):
        assert resolved[id(s)].equals(want)


def test_building_units_splits_multiple_buildings():
    fp = {
        0: unary_union([_rect(20.0, 10.0),
                        Polygon([(60, 0), (80, 0), (80, 10), (60, 10)])]),
        1: unary_union([_rect(20.0, 10.0),
                        Polygon([(60, 0), (80, 0), (80, 10), (60, 10)])]),
    }
    units = energy._building_units(fp)
    assert len(units) == 2


def test_assign_spaces_fills_every_building():
    # Two buildings; spaces with no trustworthy address must still fill BOTH
    # (area-fill), never pile into just one.
    left = _rect(20.0, 10.0)
    right = Polygon([(60, 0), (80, 0), (80, 10), (60, 10)])
    units = [left, right]
    unit_cap = [left.area, right.area]
    spaces = [{"reference": f"S{i}", "area": 40.0} for i in range(6)]
    info = {s["reference"]: {"loc": None, "postal": None, "inner": None}
            for s in spaces}
    assign = energy._assign_spaces_to_units(spaces, units, unit_cap, info)
    used_units = set(assign.values())
    assert used_units == {0, 1}                       # both buildings got spaces


def test_multi_building_synthesis_covers_all_buildings():
    left = _rect(20.0, 10.0)                            # 200 m²
    right = Polygon([(60, 0), (80, 0), (80, 12), (60, 12)])  # 240 m²
    both = unary_union([left, right])
    tz, refs, locs = [], [], []
    # 3 dwellings addressed to the left building, 3 to the right.
    for i, (x, y) in enumerate([(5, 5), (10, 5), (15, 5),
                                (65, 6), (70, 6), (75, 6)]):
        ref = f"D{i}"
        tz.append({"zone_id": f"Residential|{ref}|F0", "kind": "occupied",
                   "use_type": "Residential", "reference": ref, "floors": [0],
                   "area_without_communals": 60.0, "area_with_communals": 60.0,
                   "detailed_use_types": []})
        refs.append(ref)
        locs.append(Point(float(x), float(y)))
    row = {
        "building_reference": "M", "cadaster_code": "00000",
        "br__building_footprint_by_floor": [["s", 0, both]],
        "br__thermal_zones": tz,
        "br__building_spaces_reference": {"Residential": refs},
        "br__building_spaces_postal_address_location": {"Residential": locs},
        "br__communal_area_by_floor": {},
    }
    zones = energy.synthesize_space_geometry(row, cat_shrink=False)
    units = energy._building_units(energy._footprints_by_floor(row))
    assert len(units) == 2
    # Every building gets at least one dwelling polygon on floor 0.
    per_unit = {}
    for z in zones:
        p = (z.get("floor_polys") or {}).get(0)
        if p is not None and not p.is_empty and z.get("kind") == "occupied":
            per_unit.setdefault(energy._unit_index_for(p, units), 0)
            per_unit[energy._unit_index_for(p, units)] += 1
    assert set(per_unit) == {0, 1}
    # Both buildings tile ~fully.
    covered = sum(z["floor_polys"][0].area for z in zones
                  if 0 in z.get("floor_polys", {}))
    assert covered == pytest.approx(both.area, rel=0.03)


def test_specific_postal_ignores_no_number_addresses():
    assert energy._specific_postal("AVENIDA DEVESA, 10") == "AVENIDA DEVESA, 10"
    assert energy._specific_postal("CALLE X, S-N") is None
    assert energy._specific_postal("CALLE X, S/N") is None
    assert energy._specific_postal("CALLE SIN NUMERO") is None
    assert energy._specific_postal(None) is None


def test_parking_grouped_with_housing_by_postal():
    # A dwelling and its garage share a postal address (differ only in inner) and
    # must land in the SAME building even though the garage's own point is missing.
    left = _rect(20.0, 10.0)
    right = Polygon([(60, 0), (80, 0), (80, 10), (60, 10)])
    units = [left, right]
    unit_cap = [left.area, right.area]
    spaces = [
        {"reference": "H", "area": 60.0},          # dwelling in the RIGHT building
        {"reference": "G", "area": 30.0},          # its garage, no own location
        {"reference": "X", "area": 50.0},          # unrelated, no address
    ]
    info = {
        "H": {"loc": Point(70, 5), "postal": "AV DEVESA, 10", "inner": "10 1 1"},
        "G": {"loc": None, "postal": "AV DEVESA, 10", "inner": "10 0 1"},
        "X": {"loc": None, "postal": None, "inner": None},
    }
    assign = energy._assign_spaces_to_units(spaces, units, unit_cap, info)
    assert assign[id(spaces[0])] == assign[id(spaces[1])]   # dwelling + garage together
    assert assign[id(spaces[0])] == 1                       # in the right building


def test_parking_block_fronts_a_main_facade():
    # A parking block must front a main façade wide enough for the vehicle door.
    work = Polygon([(0, 0), (30, 0), (30, 10), (0, 10)])
    facade = LineString([(0, 0), (30, 0)])                  # façade on the bottom
    block, rem = energy._carve_parking_block(work, 60.0, 0.0, facade, None, 3.0)
    assert block is not None and not block.is_empty
    frontage = block.boundary.intersection(facade.buffer(0.4)).length
    assert frontage >= 3.0


def test_all_garage_floor_fronts_facade():
    # A floor of only garages tiles them along the façade so each gets frontage.
    floor = _rect(30.0, 10.0)
    facade = LineString([(0, 0), (30, 0)])
    present = [{"use_type": "Warehouse - Parking", "kind": "non_occupied",
                "floors": [0], "area": 40.0, "reference": f"P{i}", "anchor": None}
               for i in range(4)]
    targets = {id(s): 40.0 for s in present}
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, 0.0, facade, None, 0.0, 0.0, None, None,
        False, energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    for s in present:
        poly = assign[id(s)]
        assert poly is not None and not poly.is_empty
        assert poly.boundary.intersection(facade.buffer(0.3)).length >= 3.0 - 0.1


def test_house_columns_stack_garage_under_dwelling():
    # Attached house: a Warehouse (floor 0) and a Residential (floors 1-2) share
    # one reference → they form a single vertical column (garage under dwelling),
    # so the ground-floor garage sits directly below the upper dwelling.
    fp = _rect(24.0, 10.0)
    ws = []
    for fl in (0, 1, 2):
        ws += [{"geometry": LineString([(0, 0), (24, 0)]), "type": "main_facade", "floor": fl},
               {"geometry": LineString([(0, 10), (24, 10)]), "type": "main_facade", "floor": fl}]
    tz = []
    for i in range(4):
        ref = f"H{i}"
        tz.append({"zone_id": f"R|{ref}", "kind": "occupied", "use_type": "Residential",
                   "reference": ref, "floors": [1, 2], "area_without_communals": 60.0,
                   "area_with_communals": 60.0, "detailed_use_types": []})
        tz.append({"zone_id": f"W|{ref}", "kind": "non_occupied",
                   "use_type": "Warehouse - Parking", "reference": ref, "floors": [0],
                   "area_without_communals": 30.0, "area_with_communals": 30.0,
                   "detailed_use_types": []})
    row = {
        "building_reference": "HR", "cadaster_code": "0",
        "br__building_footprint_by_floor": [["s", 0, fp], ["s", 1, fp], ["s", 2, fp]],
        "br__thermal_zones": tz, "br__wall_segments": ws,
        "br__communal_area_by_floor": {}, "br__parcel_main_orientation": 0.0,
    }
    zones = energy.synthesize_space_geometry(row, cat_shrink=False)
    by_ref = {}
    for z in zones:
        if z.get("reference"):
            by_ref.setdefault(z["reference"], []).append(z)
    # For each house, the floor-0 garage overlaps (stacks under) the floor-1 dwelling.
    for ref, zs in by_ref.items():
        g = next((z for z in zs if z["kind"] == "non_occupied"), None)
        d = next((z for z in zs if z["kind"] == "occupied"), None)
        if not g or not d:
            continue
        gp = g["floor_polys"].get(0)
        dp = d["floor_polys"].get(1)
        assert gp is not None and dp is not None
        assert gp.intersection(dp).area > 0.7 * min(gp.area, dp.area)


def test_label_unit_routes_single_street_but_not_ambiguous():
    # Two buildings. Street "CARRER A" is entirely in building 1; "CARRER B" spans
    # both. A space's postal should route to building 1 for CARRER A, and to None
    # (→ area-fill) for the ambiguous CARRER B.
    good_labels = ["AVENIDA DEVESA, 2", "AVENIDA DEVESA, 4", "AVENIDA DEVESA, 6",
                   "CALLE BASSA, 1", "CALLE BASSA, 3"]
    good_units = [1, 1, 1, 0, 1]
    # AVENIDA DEVESA is entirely in building 1 → route there even for S-N.
    assert energy._label_unit("AVENIDA DEVESA, S-N", good_labels, good_units) == 1
    assert energy._label_unit("AVENIDA DEVESA, 8", good_labels, good_units) == 1
    # CALLE BASSA spans buildings 0 and 1 evenly → ambiguous → area-fill.
    assert energy._label_unit("CALLE BASSA, S-N", good_labels, good_units) is None
    # A different street with no geocoded parcel address → no match.
    assert energy._label_unit("PLAZA MAYOR, 1", good_labels, good_units) is None


def test_street_type_stripped_in_harmonisation():
    from hypercadaster_ES import downloaders, utils
    # The type prefix (CALLE / URBANIZACION / AVENIDA) is dropped so the street
    # name drives the match — both of these reduce to the same "DEVESA PARK" stem.
    assert downloaders.strip_street_type("CALLE URBANIZACION DEVESA PARK, S-N") == "DEVESA PARK, S-N"
    assert downloaders.strip_street_type("URBANIZACION DEVESA PARK, 14") == "DEVESA PARK, 14"
    assert downloaders.strip_street_type("AVENIDA DEVESA") == "DEVESA"
    # A CAT address whose type differs from the parcel label still harmonises.
    labels = ["URBANIZACION DEVESA PARK, 14", "CALLE BASSA, 147"]
    assert utils._best_address_match_index("CALLE URBANIZACION DEVESA PARK, S-N", labels) == 0


def test_assign_fills_all_buildings_proportionally():
    # Spaces with unresolved addresses must spread across ALL buildings by
    # capacity, never pile into the largest — and never leave a building empty.
    from shapely.geometry import Polygon as _P
    units = [_P([(0, 0), (10, 0), (10, 10), (0, 10)]),          # cap 100
             _P([(20, 0), (60, 0), (60, 10), (20, 10)]),        # cap 400
             _P([(70, 0), (80, 0), (80, 10), (70, 10)])]        # cap 100
    unit_cap = [u.area for u in units]
    spaces = [{"reference": f"S{i}", "area": 20.0} for i in range(20)]
    info = {s["reference"]: {"loc": None, "postal": "CALLE X, S-N", "inner": None}
            for s in spaces}
    assign = energy._assign_spaces_to_units(spaces, units, unit_cap, info,
                                            label_info=([], []))
    from collections import Counter
    per = Counter(assign.values())
    assert set(per) == {0, 1, 2}                    # every building got spaces
    assert per[1] > per[0] and per[1] > per[2]      # the big one got proportionally more


def test_widths_with_minimums():
    # Feasible: two items need >= 3 m, all widths sum to the span.
    w, d = energy._widths_with_minimums(20.0, [1.0, 1.0, 1.0], [3.0, 0.0, 3.0])
    assert sum(w) == pytest.approx(20.0)
    assert w[0] >= 3.0 - 1e-9 and w[2] >= 3.0 - 1e-9
    assert not any(d)
    # Infeasible: minimums don't fit → scaled to fit, both flagged deficient.
    w, d = energy._widths_with_minimums(4.0, [1.0, 1.0], [3.0, 3.0])
    assert sum(w) == pytest.approx(4.0)
    assert all(d)


def test_is_warehouse():
    assert energy._is_warehouse({"use_type": "Warehouse - Parking"})
    assert energy._is_warehouse({"use_type": "Garage"})
    assert not energy._is_warehouse({"use_type": "Residential"})
    assert not energy._is_warehouse({"use_type": "Commercial"})


def test_ground_floor_access_min_facade_for_non_warehouse():
    # A narrow reference (small dwelling) shares the block with a big warehouse.
    # By area alone the dwelling column would be < 3 m; the access rule widens it.
    # The warehouse also needs 3 m of façade but, being 220 m², already has plenty.
    fp = _rect(30.0, 8.0)
    ws = [{"geometry": LineString([(0, 0), (30, 0)]), "type": "main_facade", "floor": 0},
          {"geometry": LineString([(0, 8), (30, 8)]), "type": "main_facade", "floor": 0}]
    tz = [
        {"zone_id": "R|H|F0", "kind": "occupied", "use_type": "Residential",
         "reference": "H", "floors": [0], "area_without_communals": 12.0,
         "area_with_communals": 12.0, "detailed_use_types": []},
        {"zone_id": "W|W|F0", "kind": "non_occupied", "use_type": "Warehouse - Parking",
         "reference": "W", "floors": [0], "area_without_communals": 220.0,
         "area_with_communals": 220.0, "detailed_use_types": []},
    ]
    row = {
        "building_reference": "AC", "cadaster_code": "0",
        "br__building_footprint_by_floor": [["s", 0, fp]],
        "br__thermal_zones": tz, "br__wall_segments": ws,
        "br__communal_area_by_floor": {}, "br__parcel_main_orientation": 0.0,
    }
    zones = energy.synthesize_space_geometry(row, cat_shrink=False)
    facade = LineString([(0, 0), (30, 0)]).union(LineString([(0, 8), (30, 8)]))
    h = next(z for z in zones if z.get("reference") == "H")
    hp = h["floor_polys"][0]
    front = hp.boundary.intersection(facade.buffer(0.25)).length / 2
    assert front >= energy._MIN_FACADE_FRONTAGE - 0.2   # dwelling got its 3 m access
    assert not h.get("facade_deficit")
    w = next(z for z in zones if z.get("reference") == "W")
    assert not w.get("facade_deficit")                  # warehouse never flagged


def test_concave_floor_forces_central_communal_and_shares_by_area():
    # A C-shaped floor (20×12 rect with a right-side bite) whose two dwellings
    # OVER-SUBSCRIBE the footprint (120 + 100 > 172 available). The forced central
    # communal core must be reserved and the rest shared IN PROPORTION TO AREA,
    # tiling the footprint EXACTLY — no lobe dropped, no communal sliver. This is
    # the regression for Alcarràs 3444707BG9034S.
    floor = Polygon([(0, 0), (20, 0), (20, 12), (0, 12)]).difference(
        Polygon([(8, 4), (20, 4), (20, 8), (8, 8)]))       # C opening to the right
    assert floor.area < 0.85 * floor.convex_hull.area      # genuinely concave
    present = [
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 120.0, "reference": "A"},
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 100.0, "reference": "B"},
    ]
    communal = 20.0
    targets = {id(s): s["area"] for s in present}
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, 0.0, None, None, communal, 0.0, None, None,
        False, energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)

    # Communal forced to its authoritative area, compact and near the centroid.
    assert comm is not None
    assert comm.area == pytest.approx(communal, rel=0.05)
    assert comm.distance(floor.centroid) <= 3.0

    # Every space present (no dropped lobe) and the floor tiles EXACTLY.
    a, b = assign[id(present[0])], assign[id(present[1])]
    assert a is not None and not a.is_empty and b is not None and not b.is_empty
    tiled = a.area + b.area + comm.area
    assert tiled == pytest.approx(floor.area, rel=0.02)

    # Every zone is a SINGLE connected polygon (easy to model as one thermal zone),
    # never a scattered MultiPolygon — even on this concave floor.
    for z in (a, b, comm):
        assert z.geom_type == "Polygon"

    # The remaining ring is shared in proportion to area (120 : 100) — the
    # connected bisection keeps the proportion exact.
    assert a.area / b.area == pytest.approx(120.0 / 100.0, rel=0.06)


def test_communal_core_kept_off_main_facade():
    # The forced communal core must sit INTERIOR — not in contact with the main
    # façade — so the whole façade stays available to the dwellings.
    floor = _rect(20.0, 12.0)
    facade = _long_facades(20.0, 12.0)
    present = [
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 120.0, "reference": "A"},
        {"use_type": "Residential", "kind": "occupied", "floors": [0],
         "area": 100.0, "reference": "B"},
    ]
    targets = {id(s): s["area"] for s in present}
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, energy._facade_angle(facade, 0.0), facade,
        None, 20.0, 0.0, None, None, False,
        energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    assert comm is not None and comm.geom_type == "Polygon"
    assert comm.distance(facade) > 0.3                # clear of the main façade
    for s in present:                                 # dwellings still front it
        assert _facade_contact(assign[id(s)], facade) >= energy._MIN_FACADE_FRONTAGE


def test_parking_needs_facade_frontage():
    # Parking/warehouse spaces are NOT exempt: each needs ≥ 3 m of main façade for
    # its vehicle door. Six garages on a 10 m façade cannot all get it → all flagged;
    # two garages with room to spare are not flagged.
    floor = _rect(10.0, 8.0)
    facade = LineString([(0, 0), (10, 0)])
    crowded = [{"use_type": "Warehouse - Parking", "kind": "non_occupied",
                "floors": [0], "area": 12.0, "reference": f"P{i}"} for i in range(6)]
    _, _, deficit = energy._partition_floor(
        floor, floor, crowded, {id(s): 12.0 for s in crowded},
        energy._facade_angle(facade, 0.0), facade, None, 0.0, 0.0, None, None,
        False, energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    assert len(deficit) == len(crowded)               # every garage flagged

    roomy = [{"use_type": "Warehouse - Parking", "kind": "non_occupied",
              "floors": [0], "area": 40.0, "reference": f"Q{i}"} for i in range(2)]
    _, _, deficit2 = energy._partition_floor(
        floor, floor, roomy, {id(s): 40.0 for s in roomy},
        energy._facade_angle(facade, 0.0), facade, None, 0.0, 0.0, None, None,
        False, energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    assert not deficit2                                # both have their 3 m


def test_ground_floor_commercial_and_parking_both_front_facade():
    # A communal-less ground floor (shop + parking) is split NORMAL to the façade so
    # BOTH front the street, with areas proportional to their CAT areas and each a
    # single connected polygon. Regression for 3444707BG9034S floor 0 (commercial
    # was landlocked behind the parking).
    floor = _rect(20.0, 10.0)
    facade = LineString([(0, 0), (20, 0)])
    present = [
        {"use_type": "Commercial", "kind": "occupied", "floors": [0],
         "area": 110.0, "reference": "C"},
        {"use_type": "Warehouse - Parking", "kind": "non_occupied", "floors": [0],
         "area": 90.0, "reference": "P"},
    ]
    targets = {id(s): s["area"] for s in present}
    assign, comm, deficit = energy._partition_floor(
        floor, floor, present, targets, energy._facade_angle(facade, 0.0), facade,
        None, 0.0, 0.0, None, (10.0, -1.0), True,
        energy._MIN_FACADE_FRONTAGE, energy._CORRIDOR_WIDTH)
    c, p = assign[id(present[0])], assign[id(present[1])]
    assert c.geom_type == "Polygon" and p.geom_type == "Polygon"
    assert _facade_contact(c, facade, tol=0.3) >= energy._MIN_FACADE_FRONTAGE  # shop fronts street
    assert _facade_contact(p, facade, tol=0.3) >= energy._MIN_FACADE_FRONTAGE  # parking door
    assert not deficit
    assert c.area / p.area == pytest.approx(110.0 / 90.0, rel=0.05)   # proportional


def test_synthesize_space_geometry_end_to_end():
    fp = _rect(24.0, 12.0)
    ws = []
    for fl in (0, 1):
        ws += [
            {"geometry": LineString([(0, 0), (24, 0)]), "type": "main_facade", "floor": fl},
            {"geometry": LineString([(0, 12), (24, 12)]), "type": "main_facade", "floor": fl},
            {"geometry": LineString([(0, 0), (0, 12)]), "type": "party_wall", "floor": fl},
            {"geometry": LineString([(24, 0), (24, 12)]), "type": "party_wall", "floor": fl},
        ]
    tz = [
        {"zone_id": "Residential|A|F0-1", "kind": "occupied", "use_type": "Residential",
         "reference": "A", "floors": [0, 1], "area_without_communals": 70.0,
         "area_with_communals": 80.0, "detailed_use_types": []},
        {"zone_id": "Residential|B|F0", "kind": "occupied", "use_type": "Residential",
         "reference": "B", "floors": [0], "area_without_communals": 60.0,
         "area_with_communals": 68.0, "detailed_use_types": []},
        {"zone_id": "Residential|C|F1", "kind": "occupied", "use_type": "Residential",
         "reference": "C", "floors": [1], "area_without_communals": 60.0,
         "area_with_communals": 68.0, "detailed_use_types": []},
        {"zone_id": "Parking|P|F0", "kind": "non_occupied", "use_type": "Parking",
         "reference": "P", "floors": [0], "area_without_communals": 40.0,
         "area_with_communals": 40.0, "detailed_use_types": []},
        {"zone_id": "communal_core", "kind": "communal", "use_type": "Communal",
         "floors": [0, 1], "area": 44.0},
    ]
    row = {
        "building_reference": "T", "cadaster_code": "00000",
        "br__building_footprint_by_floor": [["s", 0, fp], ["s", 1, fp]],
        "br__thermal_zones": tz,
        "br__wall_segments": ws,
        "br__communal_area_by_floor": {"Communal": {0: 20.0, 1: 24.0}},
        "br__parcel_main_orientation": 0.0,
        "postal_addresses_location": [Point(12.0, 0.0)],
    }
    # cat_shrink off → tile the gross footprint (no walled-footprint synthesis).
    zones = energy.synthesize_space_geometry(row, cat_shrink=False)
    assert zones
    kinds = {z.get("kind") for z in zones}
    assert "communal" in kinds and "occupied" in kinds

    # Each floor tiles exactly to the 288 m² footprint.
    for fl in (0, 1):
        tiled = sum(z["floor_polys"][fl].area for z in zones
                    if fl in z.get("floor_polys", {}))
        assert tiled == pytest.approx(fp.area, rel=0.02)

    # Communal core exists on both floors and reaches the ground-floor entrance.
    comm = next(z for z in zones if z.get("kind") == "communal")
    assert 0 in comm["floor_polys"] and 1 in comm["floor_polys"]
    assert comm["floor_polys"][0].distance(Point(12.0, 0.0)) <= energy._CORRIDOR_WIDTH + 1e-6
