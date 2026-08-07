"""Non-manifold cell complex for building spaces (``hypercadaster_ES.topology``).

Pure-geometry tests on synthetic prisms — no cadastral data, no topologicpy
(the module is an independent MIT reimplementation of the concept).
"""
import math

import pytest
from shapely.geometry import Polygon

from hypercadaster_ES import topology as T


def _rect(x0, y0, x1, y1, z0, z1, name, floor, floor_bc="Ground"):
    return {"name": name, "poly": Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
            "z0": z0, "z1": z1, "floor": floor, "floor_bc": floor_bc}


def test_no_topologic_import():
    # The module must not import the AGPL topologic kernel (only mention it in the
    # docstring explaining why). Check for real import statements + runtime modules.
    import re
    import sys
    import hypercadaster_ES.topology as mod
    src = open(mod.__file__).read()
    assert not re.search(r"^\s*(import|from)\s+topologic", src, re.MULTILINE)
    assert "topologicpy" not in sys.modules and "topologic_core" not in sys.modules


def test_shared_wall_is_one_interior_face_with_two_cells():
    # Two rooms sharing the x=5 edge.
    prisms = [_rect(0, 0, 5, 10, 0, 3, "A", 0), _rect(5, 0, 10, 10, 0, 3, "B", 0)]
    cx = T.CellComplex.from_prisms(prisms)
    iw = [f for f in cx.interior_walls() if {f.cell_a, f.cell_b} == {"A", "B"}]
    assert len(iw) == 1
    assert iw[0].geometry.length == pytest.approx(10.0, abs=0.1)
    assert cx.graph.has_edge("A", "B")           # adjacency recorded


def test_roof_equals_uncovered_top():
    # One room, plus a smaller room stacked on half of it → roof only over the
    # uncovered half.
    prisms = [_rect(0, 0, 10, 10, 0, 3, "A", 0, "Ground"),
              _rect(0, 0, 5, 10, 3, 6, "B", 1, "Outdoors")]
    cx = T.CellComplex.from_prisms(prisms)
    roof_a = sum(f.geometry.area for f in cx.roofs() if f.cell_a == "A")
    assert roof_a == pytest.approx(50.0, abs=1.0)     # the half not under B
    # B fully covers its half of A → a floor/ceiling slab pair, no A-roof there.
    slabs = [f for f in cx.faces if f.kind == "slab" and {f.cell_a, f.cell_b} == {"A", "B"}]
    assert sum(f.geometry.area for f in slabs) == pytest.approx(50.0, abs=1.0)


def test_exterior_walls_trace_the_perimeter():
    prisms = [_rect(0, 0, 5, 10, 0, 3, "A", 0), _rect(5, 0, 10, 10, 0, 3, "B", 0)]
    cx = T.CellComplex.from_prisms(prisms)
    ext = sum(f.geometry.length for f in cx.exterior_walls())
    assert ext == pytest.approx(40.0, abs=0.2)        # perimeter of the 10x10 union


def test_snap_resolves_overlap_watertight():
    # A and B overlap by 0.02 m; snapping must split the shared edge cleanly so
    # their areas sum to the footprint with no double-count.
    a = Polygon([(0, 0), (5.02, 0), (5.02, 10), (0, 10)])
    b = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    snapped = T.snap_floor_polys({"A": {0: a}, "B": {0: b}})
    total = snapped["A"][0].area + snapped["B"][0].area
    assert total == pytest.approx(100.0, abs=0.5)
    assert snapped["A"][0].intersection(snapped["B"][0]).area < 0.01


def test_adjacency_graph_only_links_touching_spaces():
    # A touches B; C is detached far away → no edge to A or B.
    prisms = [_rect(0, 0, 5, 10, 0, 3, "A", 0), _rect(5, 0, 10, 10, 0, 3, "B", 0),
              _rect(40, 0, 50, 10, 0, 3, "C", 0)]
    cx = T.CellComplex.from_prisms(prisms)
    assert cx.graph.has_edge("A", "B")
    assert not cx.graph.has_edge("A", "C")
    assert not cx.graph.has_edge("B", "C")
