"""Lightweight non-manifold CELL COMPLEX for buildings.

An independent, MIT-licensed reimplementation of the *cell-complex concept* used
by topology kernels for building modelling: each SPACE is a Cell (3-cell), each
shared WALL / SLAB is a Face (2-cell) referenced once by its two adjacent cells,
and the spaces form an explicit adjacency graph. By construction this gives a
watertight, gap-free space subdivision, interior faces shared *exactly* (what
EnergyPlus needs for valid adjacency), a correct roof (= a top face with no cell
above), and a per-building adjacency graph.

This module deliberately uses NONE of ``topologicpy``'s code and does not import
it: ``topologicpy`` / ``topologic_core`` are AGPL-licensed, and hypercadaster_ES
is MIT. Only permissively-licensed libraries are used here — ``shapely`` (BSD),
``networkx`` (BSD) and ``numpy``. The ideas (planar arrangement, non-manifold
cell complex, space-adjacency graph) are standard computational geometry / CAD
concepts, implemented here from scratch.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, polygonize, unary_union

_SNAP = 2      # coordinate rounding (decimals) used to node the arrangement


def _clean(poly):
    """buffer(0)-repair a polygon and keep its dominant piece."""
    if poly is None:
        return None
    try:
        poly = poly.buffer(0)
    except Exception:
        return poly
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area) if not poly.is_empty else None
    return poly if (poly is not None and not poly.is_empty) else None


def _polys(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return []
    return [g for g in (list(geom.geoms) if hasattr(geom, "geoms") else [geom])
            if g is not None and not g.is_empty and hasattr(g, "exterior")]


# --------------------------------------------------------------------------- #
# 2-D planar arrangement → watertight per-space faces + shared edges
# --------------------------------------------------------------------------- #
def _arrange_floor(items):
    """Planar arrangement of one floor.

    ``items`` = list of ``(key, polygon)`` (a key may repeat — its pieces are
    merged). Returns ``(snapped, walls)``:
      * ``snapped``: ``{key: Polygon}`` — the space regions rebuilt from the
        arrangement's minimal faces, so neighbours share EXACT edges and there
        are no gaps/overlaps/slivers.
      * ``walls``: list of ``(key_a, key_b_or_None, LineString)`` — one entry per
        shared 1-cell (interior wall, two keys) or boundary 1-cell (exterior wall,
        ``key_b is None``).
    """
    polys = [(k, p) for k, p in items if p is not None and not p.is_empty]
    if not polys:
        return {}, []
    # Node every boundary together, then split into minimal faces.
    boundary = unary_union([p.boundary for _k, p in polys])
    minimal = [mf for mf in polygonize(boundary) if mf.area > 1e-6]
    if not minimal:
        return {k: _clean(p) for k, p in polys}, []

    # Assign each minimal face to the space whose region covers it (nearest on
    # a tie / for a gap face).
    face_key = {}
    grouped = defaultdict(list)
    for mf in minimal:
        rp = mf.representative_point()
        best, bestd = None, float("inf")
        for k, p in polys:
            if p.covers(rp):
                best = k
                break
            d = p.distance(rp)
            if d < bestd:
                bestd, best = d, k
        face_key[id(mf)] = best
        grouped[best].append(mf)
    snapped = {}
    for k, fs in grouped.items():
        # Keep MULTIPART regions intact (e.g. a scattered communal core) — a
        # dominant-piece _clean would silently drop area.
        g = unary_union(fs)
        try:
            g = g.buffer(0)
        except Exception:
            pass
        if g is not None and not g.is_empty:
            snapped[k] = g

    # Edge → incident minimal faces. An edge shared by two faces of DIFFERENT
    # spaces is an interior wall; an edge on only one face is an exterior wall;
    # an edge between two faces of the SAME space is internal (dissolved).
    seg_owners = defaultdict(list)   # segkey -> list of (key, (p0, p1))
    for mf in minimal:
        k = face_key[id(mf)]
        cc = list(mf.exterior.coords)
        for i in range(len(cc) - 1):
            a, b = cc[i][:2], cc[i + 1][:2]
            if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-6:
                continue
            ra = (round(a[0], _SNAP), round(a[1], _SNAP))
            rb = (round(b[0], _SNAP), round(b[1], _SNAP))
            seg_owners[tuple(sorted((ra, rb)))].append((k, (a, b)))

    grouped_walls = defaultdict(list)   # (key_a, key_b_or_None) -> [LineString]
    for _sk, lst in seg_owners.items():
        keys = {k for k, _ in lst}
        p0, p1 = lst[0][1]
        seg = LineString([p0, p1])
        if len(lst) == 1:                        # boundary edge → exterior wall
            grouped_walls[(lst[0][0], None)].append(seg)
        elif len(keys) >= 2:                     # two spaces → interior wall
            a, b = sorted(k for k in keys if k is not None)[:2]
            grouped_walls[(a, b)].append(seg)
        # else: same space on both sides → internal, ignore

    walls = []
    for (ka, kb), segs in grouped_walls.items():
        u = unary_union(segs)
        merged = linemerge(u) if u.geom_type == "MultiLineString" else u
        for ls in (list(merged.geoms) if hasattr(merged, "geoms") else [merged]):
            if ls is not None and not ls.is_empty and ls.length > 0.05:
                walls.append((ka, kb, ls))
    return snapped, walls


def snap_floor_polys(space_floor_map):
    """Rebuild ``{key: {floor: Polygon}}`` through the per-floor planar arrangement
    so the spaces tile each floor watertight (shared exact edges, no slivers or
    gaps). Disjoint buildings on a floor are handled naturally (their faces just
    don't touch). Keys/floors that produce no face are dropped for that floor.
    """
    by_floor = defaultdict(list)
    for key, fmap in space_floor_map.items():
        if not isinstance(fmap, dict):
            continue
        for fl, poly in fmap.items():
            for piece in _polys(poly):
                by_floor[fl].append((key, piece))
    out = defaultdict(dict)
    for fl, items in by_floor.items():
        snapped, _walls = _arrange_floor(items)
        for key, poly in snapped.items():
            out[key][fl] = poly
    return {k: dict(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Cells / faces / complex
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    """One space on one floor (a 3-cell = extruded face)."""
    name: str                 # unique zone name (matches the EnergyPlus zone)
    space_key: str            # space identity (reference / zone_id)
    floor: int
    z0: float
    z1: float
    poly: Polygon
    use_type: Optional[str] = None
    kind: Optional[str] = None
    floor_bc: str = "Ground"


@dataclass
class Face:
    """A shared 2-cell. ``kind`` ∈ wall_interior / wall_exterior / slab / roof /
    floor. ``cell_b`` is None for exterior faces. ``bc`` is the EnergyPlus
    boundary condition (Surface / Outdoors / Ground)."""
    kind: str
    geometry: object          # LineString (wall) or Polygon (horizontal)
    z0: float
    z1: float
    cell_a: str
    cell_b: Optional[str]
    bc: str


class CellComplex:
    """Non-manifold cell complex of a building: cells (spaces), faces (shared
    walls / slabs / roofs), and a ``networkx`` adjacency graph."""

    def __init__(self, cells, faces, graph):
        self.cells = cells        # list[Cell]
        self.faces = faces        # list[Face]
        self.graph = graph        # networkx.Graph

    # -- construction -------------------------------------------------------- #
    @classmethod
    def from_prisms(cls, prisms, slab_eps=0.25):
        """Build the complex from ``prisms`` = list of dicts
        ``{name, poly, z0, z1, floor, floor_bc, space_key?, use_type?, kind?}``.
        ``poly`` is the 2-D footprint; the prism is ``poly × [z0, z1]``.
        """
        cells = [Cell(name=p["name"], space_key=p.get("space_key", p["name"]),
                      floor=int(p["floor"]), z0=float(p["z0"]), z1=float(p["z1"]),
                      poly=_clean(p["poly"]), use_type=p.get("use_type"),
                      kind=p.get("kind"), floor_bc=p.get("floor_bc", "Ground"))
                 for p in prisms]
        cells = [c for c in cells if c.poly is not None]
        by_name = {c.name: c for c in cells}
        faces = []

        # Vertical walls per floor from the planar arrangement.
        by_floor = defaultdict(list)
        for c in cells:
            by_floor[c.floor].append(c)
        for fl, cs in by_floor.items():
            _snapped, walls = _arrange_floor([(c.name, c.poly) for c in cs])
            for ka, kb, seg in walls:
                a = by_name.get(ka)
                if a is None:
                    continue
                if kb is None:
                    faces.append(Face("wall_exterior", seg, a.z0, a.z1, ka, None,
                                      "Outdoors"))
                else:
                    faces.append(Face("wall_interior", seg, a.z0, a.z1, ka, kb,
                                      "Surface"))

        # Horizontal slabs / roofs / ground floors (vertical adjacency).
        z_top = defaultdict(list)     # z level -> cells whose z1 == level
        z_bot = defaultdict(list)     # z level -> cells whose z0 == level
        for c in cells:
            z_top[round(c.z1, 3)].append(c)
            z_bot[round(c.z0, 3)].append(c)
        for c in cells:
            # Bottom face: split by cells directly below (their z1 == c.z0).
            rem = c.poly
            for q in z_top.get(round(c.z0, 3), []):     # q is directly below c
                inter = _clean(c.poly.intersection(q.poly))
                if inter is not None and inter.area > slab_eps:
                    faces.append(Face("slab", inter, c.z0, c.z0, q.name, c.name,
                                      "Surface"))
                    rem = _clean(rem.difference(q.poly)) if rem is not None else None
            if rem is not None and rem.area > slab_eps:
                faces.append(Face("floor", rem, c.z0, c.z0, c.name, None, c.floor_bc))
            # Top face: whatever is not covered by a cell above is roof.
            rem = c.poly
            for q in z_bot.get(round(c.z1, 3), []):     # q is directly above c
                inter = _clean(c.poly.intersection(q.poly))
                if inter is not None and inter.area > slab_eps and rem is not None:
                    rem = _clean(rem.difference(q.poly))
            if rem is not None and rem.area > slab_eps:
                faces.append(Face("roof", rem, c.z1, c.z1, c.name, None, "Outdoors"))

        graph = nx.Graph()
        for c in cells:
            graph.add_node(c.name, space_key=c.space_key, floor=c.floor,
                           use_type=c.use_type, kind=c.kind, area=c.poly.area)
        for f in faces:
            if f.cell_b is None:
                continue
            area = f.geometry.length * (f.z1 - f.z0) if f.kind.startswith("wall") \
                else f.geometry.area
            if graph.has_edge(f.cell_a, f.cell_b):
                graph[f.cell_a][f.cell_b]["area"] += area
            else:
                graph.add_edge(f.cell_a, f.cell_b, kind=f.kind, area=area)
        return cls(cells, faces, graph)

    # -- convenience --------------------------------------------------------- #
    def interior_walls(self):
        return [f for f in self.faces if f.kind == "wall_interior"]

    def exterior_walls(self):
        return [f for f in self.faces if f.kind == "wall_exterior"]

    def roofs(self):
        return [f for f in self.faces if f.kind == "roof"]

    def summary(self):
        from collections import Counter
        return {"cells": len(self.cells),
                "faces": dict(Counter(f.kind for f in self.faces)),
                "adjacencies": self.graph.number_of_edges()}
