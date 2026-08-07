"""EnergyPlus model generation for inferred buildings.

Turns the per-building geometric/CAT attributes produced by the inference
(``br__*`` columns) into a 3-D, per-space thermal model and exports it as an
EnergyPlus **IDF** via ``eppy`` + ``geomeppy`` (both MIT — no AGPL coupling).

Pipeline (see ``build_energy_model``):
  1. Footprint rationalization — simplify the GROUND floor, then snap the other
     floors to the ground-floor node set so vertical surfaces align across
     floors (needed for clean surface adjacency).
  2. Per-space subdivision (``synthesize_space_geometry`` / ``_partition_floor``)
     — a cadastral reference may hold SEVERAL physical buildings; the spaces are
     distributed across ALL of them (``_building_units`` / ``_assign_spaces_to_
     units``, by shared postal address then area-fill) so every building gets
     geometry. Within each building each floor is split: open-air pools use their
     real OtherConstruction geometry; parking/warehouse are placed FRONTING a main
     façade wide enough for the vehicle door (``_carve_parking_block``); the
     communal area is then FORCED as a compact CENTRAL core and the occupied
     Housing/premise spaces share the surrounding ring IN PROPORTION TO THEIR AREA
     as one contiguous wing each (``_carve_central_rect`` + ``_subdivide_ring_
     angular``). The ring tiles the footprint EXACTLY on every floor — the spaces
     scale to fill precisely what the forced core leaves, so no floor area is
     dropped (even on C-/U-shaped floors) and the communal never degenerates into a
     thin corridor sliver. On the ground floor the communal core is connected to
     the building entrance by a corridor (``_CORRIDOR_WIDTH``).
  3. IDF assembly — one EnergyPlus zone per (space, floor) prism with explicit
     surfaces; inter-zone adjacency is wired BY CONSTRUCTION from the region
     partition (``_build_surfaces``; geomeppy's intersector is unreliable on
     synthetic geometry); exterior boundary conditions are refined from the
     inferred wall typing; entrance doors (``_add_entry_doors``) then realistic
     equally-spaced square windows sized to the WWR (``_add_windows``).
     All defaults live in ``energy_config.yaml`` (see ``configure``).
  4. Systems (``add_energy_systems``, on by default) — constructions/materials,
     schedules, internal loads (people/lights/equipment/infiltration) and
     ideal-loads HVAC + thermostats by use category, plus simulation control
     and outputs. Constructions follow a **Spanish archetype template**
     (``building_archetype``): the year of construction maps to a thermal-code
     period (pre-1980 / NBE-CT-79 1980-2006 / CTE 2007-2013 / CTE post-2013)
     whose target envelope U-values drive the insulation thickness of each
     construction. The result is a runnable annual model; an EnergyPlus engine
     and an EPW weather file are needed to actually simulate.

This module is meant to be called **on demand per building**
(``hc.build_energy_model``), never for every building in a bulk merge.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
import warnings
import zlib

import numpy as np
from shapely import wkb as _wkb
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, snap

from hypercadaster_ES import utils

# Use types whose zones are not occupied by people and are placed as a compact
# end block rather than wrapped around the floor.
_PARKING_USE_KEYS = ("parking", "warehouse", "storage", "garage", "agricultur")


def _is_warehouse(s):
    """A warehouse/parking/storage/garage space. Like every occupiable space it
    still needs ``_MIN_FACADE_FRONTAGE`` of main façade — for its vehicle door
    rather than a pedestrian entrance — so it is NOT exempt from the frontage rule;
    this classifier is kept for callers that place such spaces differently."""
    ut = str(s.get("use_type", "")).lower()
    return any(k in ut for k in _PARKING_USE_KEYS)


# Floorplan partition constants (see ``_partition_floor``):
#   every Housing space must hold at least this much MAIN-FAÇADE frontage (m) so
#   it has an exterior wall for its access/windows; the ground-floor communal
#   core is connected to the building entrance by a corridor of this width (m).
_MIN_FACADE_FRONTAGE = 3.0
_CORRIDOR_WIDTH = 1.0

# --------------------------------------------------------------------------- #
# Configuration — all EnergyPlus model defaults live in energy_config.yaml
# (windows, materials, constructions, insulation, archetypes, loads, doors,
# storey heights, simulation control). Override by editing that file, pointing
# the HC_ENERGY_CONFIG env var at your own copy, or calling ``configure(path)``.
# --------------------------------------------------------------------------- #
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "energy_config.yaml")


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path=None):
    """Load the EnergyPlus defaults from YAML (``path`` → ``HC_ENERGY_CONFIG``
    → packaged ``energy_config.yaml``). Missing/invalid files fall back to the
    packaged defaults; an explicit ``path`` is merged over them."""
    import yaml
    cfg = {}
    try:
        with open(_CONFIG_PATH) as fh:                 # packaged baseline
            cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        sys.stderr.write(f"  energy: could not read {_CONFIG_PATH} ({e})\n")
    extra = path or os.environ.get("HC_ENERGY_CONFIG")
    if extra:
        try:
            with open(extra) as fh:
                cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
        except Exception as e:
            sys.stderr.write(f"  energy: could not read config {extra} ({e})\n")
    return cfg


def _apply_config(cfg):
    """Derive the module-level constants used throughout from a config dict."""
    g = globals()
    fl = cfg.get("floors", {})
    g["_FLOOR_HEIGHT"] = float(fl.get("floor_height", 3.0))
    g["_GROUND_FLOOR_HEIGHT"] = float(fl.get("ground_floor_height", 5.0))
    g["_GROUND_FLOOR_MIN_FLOORS"] = int(fl.get("ground_floor_min_floors", 5))

    geo = cfg.get("geometry", {})
    g["_CAT_AREA_SHRINK"] = bool(geo.get("cat_area_shrink", True))
    g["_BUFFER_DISTANCE"] = float(geo.get("buffer_distance", 6.0))

    win = cfg.get("windows", {})
    g["_DEFAULT_WWR"] = float(win.get("default_wwr", 0.2))
    g["_WWR_BOUNDS"] = tuple(win.get("wwr_bounds", (0.05, 0.95)))
    g["_WIN_AREA_PER"] = float(win.get("area_per_window", 3.0))
    g["_WIN_SILL"] = float(win.get("sill_height", 0.9))
    g["_WIN_HEAD_MARGIN"] = float(win.get("head_margin", 0.3))
    g["_WIN_EDGE_GAP"] = float(win.get("edge_gap", 0.4))
    g["_WIN_MIN_SIZE"] = float(win.get("min_size", 0.4))
    g["_WIN_REVEAL"] = float(win.get("reveal_depth", 0.3))
    g["_WIN_INFILTRATION"] = dict(win.get("infiltration", {}))
    g["_WWR_BY_USE"] = dict(win.get("wwr_by_use", {}))
    g["_SHADING"] = dict(cfg.get("shading", {}) or {})
    nb = dict(g["_SHADING"].get("neighbours", {}) or {})
    g["_SHADE_NEIGHBOURS"] = nb
    g["_FILTER_SHADING"] = bool(nb.get("filter", True))
    g["_MAX_SHADING"] = nb.get("max_panels", 200)

    dr = cfg.get("doors", {})
    md, pd_ = dr.get("main", {}), dr.get("parking", {})
    g["_MAIN_DOOR_SIZE"] = (float(md.get("width", 1.0)), float(md.get("height", 2.1)))
    g["_PARKING_DOOR_SIZE"] = (float(pd_.get("width", 3.0)), float(pd_.get("height", 2.1)))

    g["_MATERIALS"] = {n: tuple(v) for n, v in cfg.get("materials", {}).items()}
    ins = cfg.get("insulation", {})
    g["_INS_CONDUCTIVITY"] = float(ins.get("conductivity", 0.034))
    g["_RSI"] = dict(ins.get("rsi", {"wall": 0.13, "roof": 0.10, "floor": 0.17}))
    g["_RSE"] = dict(ins.get("rse", {"wall": 0.04, "roof": 0.04, "floor": 0.04}))
    g["_CONSTRUCTION_RECIPES"] = {
        n: (c.get("u_class"), list(c.get("layers", [])))
        for n, c in cfg.get("constructions", {}).items()}
    g["_ARCHETYPE_ENVELOPE"] = {k: dict(v) for k, v in
                                cfg.get("archetype_envelope", {}).items()}
    loads = cfg.get("loads", {}) or {}
    # The per-use load rows; the sibling scalars in the block are settings, not uses.
    g["_LOAD_LIBRARY"] = {k: dict(v) for k, v in loads.items()
                          if isinstance(v, dict)}
    g["_RES_OCC_FROM_CENSUS"] = bool(loads.get("residential_occupancy_from_census",
                                               True))
    g["_ACT_SLEEPING"] = float(loads.get("residential_activity_sleeping", 72))
    g["_OCC_AREA_WEIGHT"] = float(loads.get("occupancy_area_weight", 0.5))
    g["_CO2"] = dict(cfg.get("co2", {}) or {})
    g["_NONRES_VENT"] = dict(cfg.get("nonresidential_ventilation", {}) or {})
    g["_HS3_VENT"] = dict(cfg.get("hs3_ventilation", {}) or {})
    g["_COMFORT_ZONE"] = dict(cfg.get("comfort_zone", {}) or {})
    seas = cfg.get("seasons", {}) or {}
    g["_SEASON_SMOOTH_DAYS"] = int(seas.get("smoothing_days", 7))
    g["_INBETWEEN_TEMP_LO"] = float(seas.get("in_between_low_c", 16.0))
    g["_INBETWEEN_TEMP_HI"] = float(seas.get("in_between_high_c", 24.0))
    g["_HVAC"] = dict(cfg.get("hvac", {}) or {})
    g["_SCHEDULES"] = cfg.get("schedules", {}) or {}
    g["_RESIDENTIAL_BEHAVIOR"] = cfg.get("residential_behavior", {}) or {}
    g["_RES_CTE"] = dict(cfg.get("residential_cte", {}) or {})
    g["_INCOME_SETPOINTS"] = dict(cfg.get("income_setpoints", {}) or {})
    g["_RES_DHW"] = dict(cfg.get("residential_dhw", {}) or {})
    g["_SIM"] = cfg.get("simulation", {})


def configure(path=None):
    """Reload the EnergyPlus defaults, optionally merging a user YAML on top.
    Returns the active config dict."""
    g = globals()
    g["_CFG"] = _load_config(path)
    _apply_config(g["_CFG"])
    return g["_CFG"]


def get_config():
    """Return the active EnergyPlus defaults config dict."""
    return _CFG


_CFG = _load_config()
_apply_config(_CFG)

# Sentinel for keyword defaults that must follow the ACTIVE config rather than
# freeze its value at import time (``configure()`` can reload it), and whose
# "off" value is a meaningful None -- e.g. ``max_shading=None`` means "no cap",
# so None cannot double as "ask the config".
_FROM_CONFIG = object()

_WINDOW_CONSTRUCTION = "HC_Window"
_MAIN_DOOR_CONSTRUCTION = "HC_MainDoor"
_PARKING_DOOR_CONSTRUCTION = "HC_ParkingDoor"

_idd_initialized = False


def _default_idd_path():
    """Locate an ``Energy+.idd``: an env override, a system EnergyPlus, or the
    one bundled with eppy (preferred V9.2)."""
    env = os.environ.get("HC_ENERGYPLUS_IDD")
    if env and os.path.exists(env):
        return env
    for p in ("/usr/local/EnergyPlus*/Energy+.idd", "/opt/EnergyPlus*/Energy+.idd"):
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    import eppy
    idds = glob.glob(os.path.join(os.path.dirname(eppy.__file__),
                                  "resources", "iddfiles", "Energy+V9_2_0.idd"))
    if not idds:
        idds = sorted(glob.glob(os.path.join(os.path.dirname(eppy.__file__),
                                             "resources", "iddfiles", "Energy+V*.idd")))
    if not idds:
        raise FileNotFoundError(
            "No Energy+.idd found. Set HC_ENERGYPLUS_IDD or install EnergyPlus.")
    return idds[-1]


_idd_version = "9.2"


def _parse_idd_version(idd_path):
    """Read the ``!IDD_Version X.Y.Z`` header → ``"X.Y"`` (the IDF Version must
    match the EnergyPlus engine that will run it)."""
    try:
        with open(idd_path) as fh:
            for _ in range(10):
                line = fh.readline()
                if "IDD_Version" in line:
                    v = line.split("IDD_Version")[1].strip().split()[0]
                    parts = v.split(".")
                    return ".".join(parts[:2]) if len(parts) >= 2 else v
    except Exception:
        pass
    return "9.2"


def _ensure_idd(idd_path=None):
    global _idd_initialized, _idd_version
    from geomeppy import IDF
    if idd_path is None:
        idd_path = _default_idd_path()
    _idd_version = _parse_idd_version(idd_path)
    if not _idd_initialized:
        IDF.setiddname(idd_path)
        _idd_initialized = True
    return IDF


# --------------------------------------------------------------------------- #
# Row → geometry/attribute extraction
# --------------------------------------------------------------------------- #
def _as_geom(g):
    if isinstance(g, (bytes, bytearray, memoryview)):
        try:
            return _wkb.loads(bytes(g))
        except Exception:
            return None
    return g


def _footprints_by_floor(row):
    """{floor_int: union Polygon} from ``br__building_footprint_by_floor``
    (records ``[sbr, floor, geometry]``)."""
    recs = row.get("br__building_footprint_by_floor")
    if not isinstance(recs, (list, tuple)) or not recs:
        return {}
    by_floor = {}
    for rec in recs:
        try:
            fl = int(rec[1])
            g = _as_geom(rec[2])
        except Exception:
            continue
        if g is None or getattr(g, "is_empty", True):
            continue
        by_floor.setdefault(fl, []).append(g)
    out = {}
    for fl, gs in by_floor.items():
        try:
            u = unary_union(gs)
        except Exception:
            u = gs[0]
        if u is not None and not u.is_empty:
            out[fl] = u
    return out


def _building_spaces(row, pool_use_types=("Sports facilities",)):
    """List of space dicts: {use_type, detailed, floors, area, reference, kind}.

    Derived from the parallel ``br__building_spaces_*`` dicts (keyed by use_type).
    ``kind`` is occupied / non_occupied (parking) / pool (set later from
    OtherConstruction) / communal handled separately.
    """
    refs = row.get("br__building_spaces_reference") or {}
    flrs = row.get("br__building_spaces_floors") or {}
    areas = (row.get("br__building_spaces_area_without_communal")
             or row.get("br__building_spaces_area_with_communal") or {})
    det = row.get("br__building_spaces_detailed_use_type") or {}
    spaces = []
    if not isinstance(refs, dict):
        return spaces
    for use, ref_list in refs.items():
        if not isinstance(ref_list, (list, tuple)):
            continue
        fl_list = flrs.get(use, [])
        ar_list = areas.get(use, [])
        du_list = det.get(use, [])
        for i, ref in enumerate(ref_list):
            span = fl_list[i] if i < len(fl_list) else []
            try:
                span = sorted({int(f) for f in span}) if isinstance(span, (list, tuple)) else [int(span)]
            except (TypeError, ValueError):
                span = []
            try:
                area = float(ar_list[i]) if i < len(ar_list) else 0.0
            except (TypeError, ValueError):
                area = 0.0
            ulow = str(use).lower()
            kind = "non_occupied" if any(k in ulow for k in _PARKING_USE_KEYS) else "occupied"
            spaces.append({
                "reference": ref,
                "use_type": use,
                "detailed": du_list[i] if i < len(du_list) else None,
                "floors": span,
                "area": area,
                "kind": kind,
            })
    return spaces


def _zones_from_thermal_zones(row):
    """Geometry-record skeletons built from the AUTHORITATIVE ``br__thermal_zones``
    (``building_inference._synthesize_thermal_zones``), so the EnergyPlus zones
    stay 1:1 with that list rather than being independently re-derived.

    Returns the same record shape as ``_building_spaces`` (reference / use_type /
    detailed / floors / area / kind) plus the thermal-zone identity carried
    through for naming and traceability (zone_id / n_spaces / duplex /
    area_with_communals / area_without_communals). ``area`` is the space's
    PRIVATE area (without communal) used for the per-floor area-weighted
    subdivision — the communal area is placed separately as the vertical core,
    so it must not be double-counted here. The communal core zone is skipped
    (handled by ``_communal_area_by_floor`` during synthesis). Returns ``[]``
    when the column is absent so callers fall back to ``_building_spaces``.
    """
    tzs = row.get("br__thermal_zones")
    if not isinstance(tzs, (list, tuple)) or not tzs:
        return []
    spaces = []
    for z in tzs:
        if not isinstance(z, dict):
            continue
        if z.get("kind") == "communal" or z.get("zone_id") == "communal_core":
            continue                                     # carved separately
        span = z.get("floors")
        if not span and z.get("floor") is not None:
            span = [z["floor"]]
        try:
            floors = sorted({int(f) for f in (span or [])})
        except (TypeError, ValueError):
            floors = []
        anc, awc = z.get("area_without_communals"), z.get("area_with_communals")
        try:
            area = float(anc if anc is not None else (awc if awc is not None else 0.0))
        except (TypeError, ValueError):
            area = 0.0
        det = z.get("detailed_use_types") or []
        spaces.append({
            "zone_id": z.get("zone_id"),
            "reference": z.get("reference"),
            "use_type": z.get("use_type"),
            "detailed": det[0] if det else None,
            "floors": floors,
            "area": area,
            "kind": z.get("kind") or "occupied",
            "n_spaces": z.get("n_spaces", 1),
            "duplex": bool(z.get("duplex")),
            "area_with_communals": awc,
            "area_without_communals": anc,
        })
    return spaces


def _communal_zone_id(row):
    """``zone_id`` of the authoritative communal core in ``br__thermal_zones``
    (so the synthesized communal geometry keeps the same identity). Falls back
    to ``"communal_core"``."""
    tzs = row.get("br__thermal_zones")
    if isinstance(tzs, (list, tuple)):
        for z in tzs:
            if isinstance(z, dict) and (z.get("kind") == "communal"
                                        or z.get("zone_id") == "communal_core"):
                return z.get("zone_id") or "communal_core"
    return "communal_core"


def _pool_geoms(row):
    out = []
    ocs = row.get("br__other_constructions")
    if isinstance(ocs, (list, tuple)):
        for o in ocs:
            if isinstance(o, dict) and o.get("nature") == "openAirPool":
                g = _as_geom(o.get("geometry"))
                if g is not None and not g.is_empty:
                    out.append(g)
    return out


def _communal_area_by_floor(row):
    """{floor_int: total communal area} summed across uses."""
    cbf = row.get("br__communal_area_by_floor") or {}
    out = {}
    if isinstance(cbf, dict):
        for _use, fl_map in cbf.items():
            if not isinstance(fl_map, dict):
                continue
            for fl, a in fl_map.items():
                try:
                    out[int(fl)] = out.get(int(fl), 0.0) + float(a)
                except (TypeError, ValueError):
                    pass
    return out


# --------------------------------------------------------------------------- #
# Footprint rationalization
# --------------------------------------------------------------------------- #
def _clean(poly):
    if poly is None:
        return None
    try:
        poly = poly.buffer(0)
    except Exception:
        return poly
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)  # keep the dominant piece
    return poly if (poly is not None and not poly.is_empty) else None


def _mp_clean(poly):
    """Like ``_clean`` but PRESERVES every piece of a MultiPolygon (no
    dominant-piece collapse). Used where a partition slice/wedge may legitimately
    be disconnected (a dwelling wrapping a patio in a C-/U-shaped floor) — keeping
    all pieces means the floor area is never silently dropped."""
    if poly is None:
        return None
    try:
        poly = poly.buffer(0)
    except Exception:
        return poly
    return poly if (poly is not None and not poly.is_empty) else None


def _largest_piece(poly):
    """The single largest-area connected Polygon of a (Multi)Polygon (or None)."""
    ps = _polys(poly)
    return max(ps, key=lambda g: g.area) if ps else None


def _carve_central_connected(poly, target_area, angle=0.0, origin=(0.0, 0.0)):
    """Carve a CONNECTED central core of area ≈ ``target_area``. Like
    ``utils._carve_central_rect`` but centred on a point guaranteed INSIDE the
    polygon and sized by binary search on the LARGEST connected component of the
    box∩polygon, so a notched/C-shaped floor still yields a single-Polygon core
    (never a pair of lobes straddling the patio). Returns ``(core, remainder)``."""
    from shapely.affinity import rotate as _rot
    from shapely.geometry import box as _box
    if poly is None or getattr(poly, "is_empty", True) or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    rp = _rot(poly, -angle, origin=origin)
    c = rp.centroid
    if not rp.contains(c):                      # centroid in the notch → snap inside
        c = rp.representative_point()
    cx, cy = c.x, c.y
    minx, miny, maxx, maxy = rp.bounds
    W, H = maxx - minx, maxy - miny
    aspect = (W / H) if H > 1e-9 else 1.0
    rh0 = (target_area / aspect) ** 0.5 if aspect > 0 else target_area ** 0.5
    rw0 = aspect * rh0

    def _largest_at(scale):
        rect = _box(cx - scale * rw0 / 2, cy - scale * rh0 / 2,
                    cx + scale * rw0 / 2, cy + scale * rh0 / 2)
        try:
            return _largest_piece(rp.intersection(rect))
        except Exception:
            return None

    lo, hi = 0.0, 4.0
    for _ in range(30):
        s = (lo + hi) / 2.0
        lp = _largest_at(s)
        if (lp.area if lp is not None else 0.0) < target_area:
            lo = s
        else:
            hi = s
    lp = _largest_at((lo + hi) / 2.0)
    if lp is None or lp.is_empty:
        return None, poly
    core = _mp_clean(_rot(lp, angle, origin=origin))
    if core is None:
        return None, poly
    return core, _mp_clean(poly.difference(core))


def _carve_interior_core(poly, target_area, facade_lines, angle, setback):
    """Carve a connected central core of area ≈ ``target_area`` that does NOT touch
    the main façade — the communal stair/lift core sits INTERIOR so the whole
    façade is left to the occupiable/parking spaces (which each need frontage). The
    core is carved from ``poly`` minus a façade-adjacent strip, trying the widest
    feasible ``setback`` (down to a minimal clearance) and falling back to an
    unconstrained central core only when the floor is too shallow to keep it clear.
    Returns ``(core, remainder)`` with ``remainder = poly − core`` (the full ring,
    façade band included)."""
    if poly is None or getattr(poly, "is_empty", True) or target_area <= 0:
        return None, poly
    trials = sorted({setback, setback / 2.0, 0.6}, reverse=True) \
        if facade_lines is not None else []
    for sb in trials:
        try:
            region = _largest_piece(_mp_clean(poly.difference(facade_lines.buffer(sb))))
        except Exception:
            region = None
        if region is None or region.area < target_area * 1.02:
            continue
        core, _ = _carve_central_connected(
            region, target_area, angle=angle,
            origin=(region.centroid.x, region.centroid.y))
        if (core is not None and not core.is_empty
                and core.distance(facade_lines) > 0.05):
            return core, _mp_clean(poly.difference(core))
    core, _ = _carve_central_connected(
        poly, target_area, angle=angle, origin=(poly.centroid.x, poly.centroid.y))
    if core is None or core.is_empty:
        return None, poly
    return core, _mp_clean(poly.difference(core))


def _cut_at_angle(poly, frac, angle):
    """Split ``poly`` by ONE straight line into ``(a, b)`` where ``a`` carries
    ~``frac`` of the area. The cut is perpendicular to ``angle`` (deg); ``a`` is the
    low side along the ``angle`` direction. ``a`` and ``b`` are complementary
    (their union is ``poly``) so no area is lost."""
    from shapely.affinity import rotate as _rot
    from shapely.geometry import box as _box
    origin = (poly.centroid.x, poly.centroid.y)
    rp = _rot(poly, -angle, origin=origin)
    minx, miny, maxx, maxy = rp.bounds
    if maxx - minx <= 1e-9:
        return None, None
    target = rp.area * frac
    lo, hi = minx, maxx
    for _ in range(32):
        c = (lo + hi) / 2.0
        if rp.intersection(_box(minx, miny, c, maxy)).area < target:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2.0
    a = _mp_clean(_rot(rp.intersection(_box(minx, miny, c, maxy)), angle, origin=origin))
    b = _mp_clean(_rot(rp.intersection(_box(c, miny, maxx, maxy)), angle, origin=origin))
    return a, b


def _bisect_connected(poly, spaces, weights, prefer_angle=0.0):
    """Partition ``poly`` among ``spaces`` into pieces that are BOTH ~proportional
    to ``weights`` in AREA and each a SINGLE CONNECTED polygon — the two properties
    a good thermal zone needs. Recursive area-balanced bisection: at each step the
    spaces are split into two groups of ≈ equal weight and ``poly`` is cut by a
    straight line whose ANGLE is chosen so BOTH halves come out connected
    (``prefer_angle`` — normal to the main façade — is tried first, so ground-floor
    pieces still front the street). Because every cut is complementary the areas
    stay exactly proportional and the floor tiles exactly. Returns ``{id: Polygon}``.
    """
    result = {}

    def _rec(sub, sps, wts):
        if sub is None or getattr(sub, "is_empty", True) or not sps:
            return
        if len(sps) == 1:
            result[id(sps[0])] = sub
            return
        order = sorted(range(len(sps)), key=lambda i: -wts[i])
        sps = [sps[i] for i in order]
        wts = [wts[i] for i in order]
        tot = float(sum(wts)) or 1.0
        acc, m = 0.0, 1
        for i, w in enumerate(wts):
            acc += w
            if acc >= tot / 2.0:
                m = i + 1
                break
        m = max(1, min(len(sps) - 1, m))
        frac = sum(wts[:m]) / tot
        maj = _major_axis_angle(sub)
        cands = [prefer_angle, prefer_angle + 90.0, maj, maj + 90.0] \
            + list(range(0, 180, 15))
        best, best_n = None, 1 << 30
        for ang in cands:
            a, b = _cut_at_angle(sub, frac, ang)
            if a is None or b is None or a.is_empty or b.is_empty:
                continue
            n = len(_polys(a)) + len(_polys(b))
            if n < best_n:
                best_n, best = n, (a, b)
            if n == 2:                           # both halves already connected
                break
        if best is None:                         # degenerate — keep whole piece
            result[id(sps[0])] = sub
            return
        a, b = best
        _rec(a, sps[:m], wts[:m])
        _rec(b, sps[m:], wts[m:])

    _rec(poly, list(spaces), [max(float(w), 1e-6) for w in weights])
    return result


def _rationalize_floors(fp_by_floor, simplify_tol=0.3, snap_tol=0.6):
    """Simplify the ground floor, then snap every other floor to the
    ground-floor boundary/nodes so coincident vertical edges share nodes."""
    if not fp_by_floor:
        return {}
    out = {}
    gf_key = 0 if 0 in fp_by_floor else min(fp_by_floor)
    gf = _clean(fp_by_floor[gf_key])
    if gf is None:
        return {f: _clean(g) for f, g in fp_by_floor.items()}
    gf = gf.simplify(simplify_tol, preserve_topology=True)
    gf = _clean(gf) or gf
    out[gf_key] = gf
    for fl, g in fp_by_floor.items():
        if fl == gf_key:
            continue
        g = _clean(g)
        if g is None:
            continue
        g = g.simplify(simplify_tol, preserve_topology=True)
        try:
            g = snap(g, gf, snap_tol)          # align shared edges to GF nodes
            g = _clean(g) or g
        except Exception:
            pass
        out[fl] = g
    return out


# --------------------------------------------------------------------------- #
# Principal-axis end block (parking / non-occupied)
# --------------------------------------------------------------------------- #
def _major_axis_angle(poly):
    """Angle (deg, CCW from +x — i.e. an ``affinity.rotate`` angle) of the
    building's principal (longest) axis via PCA of the exterior vertices.
    ``affinity.rotate(poly, -_major_axis_angle(poly))`` aligns that axis to +x,
    so an end-block cut perpendicular to x slices cleanly ALONG the major axis."""
    try:
        xy = np.asarray(poly.exterior.coords[:-1], dtype=float)
        xy = xy - xy.mean(axis=0)
        cov = np.cov(xy.T)
        w, v = np.linalg.eigh(cov)
        major = v[:, int(np.argmax(w))]
        return math.degrees(math.atan2(major[1], major[0]))  # math angle from +x
    except Exception:
        return 0.0


def _parcel_main_angle(row):
    """Building/zone axis (deg, CCW from +x for ``affinity.rotate``) from the
    inferred ``br__parcel_main_orientation`` — a compass bearing (0=N, 90=E),
    possibly bytearray/pickle-encoded. Converts bearing→math angle (90−bearing).
    Returns None if unavailable so callers fall back to PCA."""
    v = row.get("br__parcel_main_orientation")
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        try:
            import pickle as _pkl
            v = _pkl.loads(b)
        except Exception:
            try:
                v = b.decode("utf-8", "ignore")
            except Exception:
                return None
    try:
        bearing = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bearing):
        return None
    return 90.0 - bearing            # compass bearing → math angle from +x


def _carve_end_block(poly, target_area, major_angle, toward=None):
    """Carve a block of ~``target_area`` at ONE END of the building along its
    principal axis (full minor-axis width → never splits the remainder on both
    sides). Returns (block, remainder).

    End choice **prioritizes the NORTHERN extreme** (parking/non-occupied space
    is best kept away from the sun, leaving the southern side for occupied
    zones); the nearest street (``toward``) is only a tie-break when the
    principal axis runs ~E–W and neither end is clearly more northern.
    """
    from shapely import affinity
    if poly is None or poly.is_empty or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    cx, cy = poly.centroid.x, poly.centroid.y
    rot = affinity.rotate(poly, -major_angle, origin=(cx, cy))  # major axis → x
    frac = target_area / poly.area
    # Both candidate end blocks (the small ~target piece at each end).
    a_lo, b_lo = utils._split_polygon_by_area_fraction(rot, frac)         # a_lo = low-x end
    a_hi, b_hi = utils._split_polygon_by_area_fraction(rot, 1.0 - frac)   # b_hi = high-x end
    if a_lo is None or b_hi is None:
        return None, poly

    def _rot_back(g):
        return _clean(affinity.rotate(g, major_angle, origin=(cx, cy))) if g is not None else None

    low_block, low_rem = _rot_back(a_lo), _rot_back(b_lo)
    high_block, high_rem = _rot_back(b_hi), _rot_back(a_hi)

    def _cy(g):
        return g.centroid.y if (g is not None and not g.is_empty) else -1e18
    if low_block is None and high_block is None:
        return None, poly
    dy = _cy(high_block) - _cy(low_block)
    if abs(dy) > 1.0:                       # one end is clearly more northern
        pick_high = dy > 0
    elif toward is not None:                # E–W axis → tie-break toward street
        def _dot(g):
            if g is None or g.is_empty:
                return -1e18
            return (g.centroid.x - cx) * toward[0] + (g.centroid.y - cy) * toward[1]
        pick_high = _dot(high_block) > _dot(low_block)
    else:
        pick_high = False
    return (high_block, high_rem) if pick_high else (low_block, low_rem)


def _facade_frontage(block, facade_lines, tol=0.4):
    """Length of ``block``'s boundary lying on a main-façade line — the width of
    street frontage available for a door on that block."""
    if block is None or getattr(block, "is_empty", True) or facade_lines is None:
        return 0.0
    try:
        return block.boundary.intersection(facade_lines.buffer(tol)).length
    except Exception:
        return 0.0


def _carve_facade_strip(poly, target_area, facade_lines, major_fallback, door_w):
    """Carve a ~``target_area`` block as a STRIP along the dominant main façade, so
    it fronts the street over the full building width (≥ ``door_w``). Used as the
    parking fallback when neither building end offers enough façade for the vehicle
    door. Returns (block, remainder) or (None, poly)."""
    from shapely import affinity
    ang = _facade_angle(facade_lines, fallback=major_fallback)
    cx, cy = poly.centroid.x, poly.centroid.y
    rp = affinity.rotate(poly, -ang, origin=(cx, cy))
    fr = (affinity.rotate(facade_lines, -ang, origin=(cx, cy))
          if facade_lines is not None else None)
    bot, top = _facade_side_lengths(rp, fr)
    if max(bot, top) < door_w:
        return None, poly                              # no façade to sit on
    rest = max(rp.area - target_area, 1.0)
    band = (_hbands(rp, [target_area, rest])[0] if bot >= top
            else _hbands(rp, [rest, target_area])[1])
    band = _clean(band)
    if band is None or band.is_empty:
        return None, poly
    block = _clean(affinity.rotate(band, ang, origin=(cx, cy)))
    if block is None or block.is_empty:
        return None, poly
    rem = _clean(poly.difference(block))
    return block, rem


def _carve_parking_block(poly, target_area, major_angle, facade_lines,
                         toward, door_w):
    """Carve a parking / non-occupied block that FRONTS a main façade wide enough
    for a vehicle door (≥ ``door_w``), so ``_add_entry_doors`` can place it.

    Prefers the building END with ≥ ``door_w`` of main-façade frontage (tie-break
    to the more northern one, keeping parking away from the sun); if neither end
    offers enough street frontage, falls back to a strip ALONG the façade. Returns
    (block, remainder)."""
    from shapely import affinity
    if poly is None or poly.is_empty or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    if facade_lines is None:                            # no façade info → old behaviour
        return _carve_end_block(poly, target_area, major_angle, toward)
    cx, cy = poly.centroid.x, poly.centroid.y
    rot = affinity.rotate(poly, -major_angle, origin=(cx, cy))
    frac = target_area / poly.area
    a_lo, b_lo = utils._split_polygon_by_area_fraction(rot, frac)
    a_hi, b_hi = utils._split_polygon_by_area_fraction(rot, 1.0 - frac)

    def _back(g):
        return _clean(affinity.rotate(g, major_angle, origin=(cx, cy))) if g is not None else None

    cands = []
    if a_lo is not None:
        cands.append((_back(a_lo), _back(b_lo)))
    if b_hi is not None:
        cands.append((_back(b_hi), _back(a_hi)))
    scored = []
    for blk, rem in cands:
        if blk is None or blk.is_empty:
            continue
        f = _facade_frontage(blk, facade_lines)
        cyv = blk.centroid.y
        scored.append((f >= door_w, cyv, f, blk, rem))
    if scored:
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)  # qualify, north, frontage
        if scored[0][0]:                                # an end fronts the street
            return scored[0][3], scored[0][4]
    blk, rem = _carve_facade_strip(poly, target_area, facade_lines, major_angle, door_w)
    if blk is not None and not blk.is_empty:
        return blk, rem
    if scored:                                          # best available end
        return scored[0][3], scored[0][4]
    return _carve_end_block(poly, target_area, major_angle, toward)


def _nearest_street_dir(row):
    sw = row.get("br__street_width_by_orientation")
    if not isinstance(sw, dict) or not sw:
        return None
    finite = {}
    for o, w in sw.items():
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if math.isfinite(wv) and wv > 0:
            finite[o] = wv
    if not finite:
        return None
    best = min(finite, key=finite.get)
    ang = math.radians(float(best))
    return (math.sin(ang), math.cos(ang))


# --------------------------------------------------------------------------- #
# Per-space geometry synthesis
# --------------------------------------------------------------------------- #
def _facade_lines_by_floor(row):
    """``{floor_int: unioned LineString/MultiLineString}`` of the exterior
    MAIN-FAÇADE wall segments, from ``br__wall_segments`` (records
    ``{geometry, type, floor}``). Only ``type == 'main_facade'`` segments qualify
    as the frontage a Housing space may claim — party/patio/terrain edges don't.
    """
    segs = row.get("br__wall_segments")
    out = {}
    if not isinstance(segs, (list, tuple)):
        return out
    tmp = {}
    for seg in segs:
        if not isinstance(seg, dict) or str(seg.get("type")) != "main_facade":
            continue
        g = _as_geom(seg.get("geometry"))
        if g is None or getattr(g, "is_empty", True):
            continue
        try:
            fl = int(seg.get("floor", 0))
        except (TypeError, ValueError):
            fl = 0
        tmp.setdefault(fl, []).append(g)
    for fl, gs in tmp.items():
        try:
            out[fl] = unary_union(gs)
        except Exception:
            out[fl] = gs[0]
    return out


def _geom_segments(geom):
    """Yield ``(x0, y0, x1, y1)`` for each edge of a (Multi)LineString."""
    if geom is None:
        return
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for g in geoms:
        try:
            cs = list(g.coords)
        except Exception:
            continue
        for i in range(len(cs) - 1):
            yield cs[i][0], cs[i][1], cs[i + 1][0], cs[i + 1][1]


def _facade_angle(lines, fallback=0.0):
    """Length-weighted dominant orientation (deg, math angle from +x, folded to
    a direction) of the façade ``lines``. Rotating a polygon by ``-`` this makes
    the façade run ~horizontally (along +x), so vertical guillotine cuts give
    each slice a face on the façade. Falls back to ``fallback`` when empty."""
    sx = sy = 0.0
    for x0, y0, x1, y1 in _geom_segments(lines):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        th = math.atan2(dy, dx)
        sx += L * math.cos(2 * th)
        sy += L * math.sin(2 * th)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return fallback
    return math.degrees(0.5 * math.atan2(sy, sx))


def _facade_side_lengths(rot_poly, rot_lines, tol=0.75):
    """Façade length lying near the BOTTOM (min-y) and TOP (max-y) edges of a
    poly already rotated so the façade runs along x. Returns ``(bottom, top)``,
    used to decide where the communal corridor goes so dwellings keep the façade.
    """
    if rot_lines is None:
        return 0.0, 0.0
    minx, miny, maxx, maxy = rot_poly.bounds
    bot = top = 0.0
    for x0, y0, x1, y1 in _geom_segments(rot_lines):
        L = math.hypot(x1 - x0, y1 - y0)
        ymid = (y0 + y1) / 2.0
        if abs(ymid - miny) <= tol:
            bot += L
        elif abs(ymid - maxy) <= tol:
            top += L
    return bot, top


def _water_fill_widths(L, weights, min_w):
    """Split length ``L`` into widths ~proportional to ``weights`` but with each
    at least ``min_w`` (water-filling: pin the deficient ones to ``min_w`` and
    re-distribute the rest). Returns ``(widths, deficits)`` where ``deficits[i]``
    is True when i had to be clamped (or nobody can reach ``min_w``: too many
    spaces for the available façade)."""
    n = len(weights)
    if n == 0:
        return [], []
    if n * min_w >= L:                      # infeasible — everyone below minimum
        return [L / n] * n, [True] * n
    widths = [0.0] * n
    deficits = [False] * n
    free = list(range(n))
    free_L = float(L)
    free_w = float(sum(weights)) or 1.0
    changed = True
    while changed and free:
        changed = False
        for i in list(free):
            prop = (free_L * weights[i] / free_w) if free_w > 1e-12 else free_L / len(free)
            if prop < min_w:
                widths[i] = min_w
                deficits[i] = True
                free.remove(i)
                free_L -= min_w
                free_w -= weights[i]
                changed = True
    for i in free:
        widths[i] = (free_L * weights[i] / free_w) if free_w > 1e-12 else free_L / max(1, len(free))
    return widths, deficits


def _widths_with_minimums(span, weights, minimums):
    """Distribute ``span`` across items with widths ~proportional to ``weights``
    but each at least ``minimums[i]``. Used for column widths where every column
    holding an accessible (non-warehouse) space needs a minimum façade frontage.
    When the minimums don't all fit they are scaled down to fit and every item
    that wanted a minimum is flagged deficient. Returns ``(widths, deficient)``."""
    n = len(weights)
    if n == 0:
        return [], []
    tot_min = sum(minimums)
    if tot_min <= span:
        free = span - tot_min
        tw = float(sum(weights)) or 1.0
        return [minimums[i] + free * weights[i] / tw for i in range(n)], [False] * n
    scale = span / tot_min if tot_min > 0 else 0.0
    widths = [minimums[i] * scale if minimums[i] > 0 else 0.0 for i in range(n)]
    leftover = span - sum(widths)
    tw0 = sum(weights[i] for i in range(n) if minimums[i] <= 0)
    if leftover > 1e-9 and tw0 > 0:
        for i in range(n):
            if minimums[i] <= 0:
                widths[i] += leftover * weights[i] / tw0
    return widths, [minimums[i] > 0 for i in range(n)]


def _slice_band_rot(band, weights, min_frontage):
    """Vertical (constant-x) guillotine cuts of a band already rotated so the
    façade runs along x: each slice spans the full band depth and fronts the
    façade with width = its slice width (water-filled to ``min_frontage``).
    Returns ``(polys, deficits)`` aligned to ``weights``."""
    from shapely.geometry import box as _box
    minx, miny, maxx, maxy = band.bounds
    L = maxx - minx
    widths, deficits = _water_fill_widths(L, weights, min_frontage)
    polys, cum = [], minx
    for w in widths:
        polys.append(_clean(band.intersection(_box(cum, miny, cum + w, maxy))))
        cum += w
    return polys, deficits


def _hbands(poly, areas):
    """Stack of horizontal (constant-y) sub-bands of ``poly`` from the BOTTOM up,
    the i-th carrying ~``areas[i]`` (the last band takes the remainder). Cut
    positions are binary-searched on a horizontal line, so each band spans the
    full width and — for the façade-fronting bands — touches the façade running
    along x. Returns a list aligned to ``areas`` (entries may be None)."""
    from shapely.geometry import box as _box
    minx, miny, maxx, maxy = poly.bounds
    bands, y0 = [], miny
    for a in areas[:-1]:
        lo, hi = y0, maxy
        for _ in range(34):
            yc = (lo + hi) / 2.0
            if poly.intersection(_box(minx, y0, maxx, yc)).area < a:
                lo = yc
            else:
                hi = yc
        yc = (lo + hi) / 2.0
        bands.append(_clean(poly.intersection(_box(minx, y0, maxx, yc))))
        y0 = yc
    bands.append(_clean(poly.intersection(_box(minx, y0, maxx, maxy))))
    return bands


def _entrance_corridor(communal_poly, floor_poly, door_xy, width, assign):
    """Ground floor only: connect the communal core to the building entrance.
    Snap ``door_xy`` to the floor boundary, and if the communal core doesn't
    already reach it, union in a ``width``-wide corridor from the core to that
    point. Where the corridor crosses a dwelling, the dwelling keeps its dominant
    (single-polygon) piece and the SEVERED piece(s) are folded into the communal
    core — so total area is conserved with no gap and no multi-part zones (the
    IDF builder keeps only a zone's dominant piece). Returns
    ``(communal_poly, assign)``."""
    from shapely.geometry import Point, LineString
    try:
        dp = Point(door_xy)
        bnd = floor_poly.boundary
        bpt = bnd.interpolate(bnd.project(dp))          # entrance on the façade
        if communal_poly.distance(bpt) <= width * 0.5:
            return communal_poly, assign                # already reaches the door
        cb = communal_poly.boundary
        qpt = cb.interpolate(cb.project(bpt))           # nearest core boundary pt
        corr = LineString([(qpt.x, qpt.y), (bpt.x, bpt.y)]).buffer(
            width / 2.0, cap_style=2)
        corr = corr.intersection(floor_poly)
        if corr is None or corr.is_empty:
            return communal_poly, assign
        extra = [corr]                                  # pieces to fold into core
        for k, poly in list(assign.items()):
            if poly is None or getattr(poly, "is_empty", True) or not poly.intersects(corr):
                continue
            diff = poly.difference(corr)
            parts = list(diff.geoms) if hasattr(diff, "geoms") else [diff]
            parts = [p for p in parts if p is not None and not p.is_empty and p.area > 1e-6]
            if not parts:                               # dwelling fully consumed
                assign[k] = None
                extra.append(poly)
                continue
            parts.sort(key=lambda p: p.area, reverse=True)
            assign[k] = _clean(parts[0])                # dominant piece stays
            extra.extend(parts[1:])                     # severed slivers → core
        # Keep the communal core MULTIPART — it may be scattered (the corridor plus
        # void strips); a dominant-piece _clean would drop most of its area.
        merged = unary_union([communal_poly] + extra)
        try:
            merged = merged.buffer(0)
        except Exception:
            pass
        return ((merged if (merged is not None and not merged.is_empty)
                 else communal_poly), assign)
    except Exception:
        return communal_poly, assign


def _space_anchor_by_reference(row):
    """``{building_space_reference: Point}`` from the per-space postal-address
    location, so a synthesized space can be placed near its REAL address.
    ``br__building_spaces_postal_address_location`` is a dict keyed by use type
    whose lists parallel ``br__building_spaces_reference``."""
    refs = row.get("br__building_spaces_reference") or {}
    locs = row.get("br__building_spaces_postal_address_location") or {}
    out = {}
    if not isinstance(refs, dict) or not isinstance(locs, dict):
        return out
    for use, ref_list in refs.items():
        if not isinstance(ref_list, (list, tuple)):
            continue
        loc_list = locs.get(use, [])
        for i, ref in enumerate(ref_list):
            p = _as_geom(loc_list[i]) if i < len(loc_list) else None
            if p is not None and not getattr(p, "is_empty", True):
                out[ref] = p
    return out


def _points_pca_angle(pts):
    """Principal-axis angle (deg, math from +x) of a set of points, or None."""
    if len(pts) < 2:
        return None
    xy = np.asarray([[p.x, p.y] for p in pts], dtype=float)
    xy = xy - xy.mean(axis=0)
    try:
        w, v = np.linalg.eigh(np.cov(xy.T))
        major = v[:, int(np.argmax(w))]
        return math.degrees(math.atan2(major[1], major[0]))
    except Exception:
        return None


def _building_address_pool(row, footprint=None, buffer_m=8.0):
    """Distinct building-level postal-address points (``postal_addresses_location``)
    — the true per-dwelling entrance locations. Optionally kept to those within
    ``buffer_m`` of the footprint. Used to reconstruct anchors for spaces whose
    own address is a non-informative fallback (see ``_resolve_space_anchors``)."""
    pal = row.get("postal_addresses_location")
    if pal is None:
        return []
    seq = (list(pal) if hasattr(pal, "__iter__") and not hasattr(pal, "geom_type")
           else [pal])
    pts, seen = [], set()
    for g in seq:
        g = _as_geom(g)
        if g is None or getattr(g, "is_empty", True) or not hasattr(g, "x"):
            continue
        key = (round(g.x, 1), round(g.y, 1))
        if key in seen:
            continue
        seen.add(key)
        pts.append(g)
    if footprint is not None and pts:
        near = [p for p in pts if footprint.distance(p) <= buffer_m]
        pts = near or pts
    return pts


def _resolve_space_anchors(row, spaces):
    """``{id(space): Point}`` address anchor per space, robust to coarse geocoding.

    Starts from the per-space match (``_space_anchor_by_reference``). Spaces whose
    address is a FALLBACK — a single location shared by many spaces (e.g. the
    ``S-N``/no-number CAT addresses all collapsed to the building point) — carry
    no real position, so they are re-seeded from the building's own
    ``postal_addresses_location`` pool (the actual dwelling entrances), assigned in
    street order. This only kicks in when that pool is rich enough to be a row of
    houses, so apartment blocks (one clustered entrance) are left untouched and
    fall through to the corridor layout."""
    from collections import Counter
    from shapely.affinity import rotate as _rot
    per = _space_anchor_by_reference(row)
    base = {id(s): per.get(s.get("reference")) for s in spaces}
    n = len(spaces)
    cnt = Counter((round(p.x, 1), round(p.y, 1))
                  for p in base.values() if p is not None)
    fallback = {k for k, c in cnt.items() if c > max(3, 0.15 * n)}

    def _is_fallback(p):
        return p is None or (round(p.x, 1), round(p.y, 1)) in fallback

    need = [s for s in spaces if _is_fallback(base[id(s)])]
    # Only reconstruct when geocoding LARGELY failed (most spaces are fallback).
    # When the per-space addresses are mostly good, trust them — spreading a few
    # fallback spaces from the pool would only inject noise into a clean layout.
    if len(need) < 0.5 * n:
        return base
    fp = _footprints_by_floor(row)
    foot = unary_union(list(fp.values())) if fp else None
    used = {(round(p.x, 1), round(p.y, 1)) for i, p in base.items()
            if not _is_fallback(p)}
    avail = [p for p in _building_address_pool(row, foot)
             if (round(p.x, 1), round(p.y, 1)) not in used]
    # Only reconstruct for a genuine row of houses: the pool must hold MANY
    # distinct, SPREAD address points. Apartment blocks (one clustered entrance)
    # fail this and fall through to the corridor layout. Several spaces may map to
    # one point (a house and its garage share an address) → cycling is fine.
    if len(avail) < 8:
        return base
    axs, ays = [p.x for p in avail], [p.y for p in avail]
    adiag = math.hypot(max(axs) - min(axs), max(ays) - min(ays))
    pdiag = 1.0
    if foot is not None:
        mnx, mny, mxx, mxy = foot.bounds
        pdiag = math.hypot(mxx - mnx, mxy - mny) or 1.0
    if adiag / pdiag < 0.25:
        return base
    ang = _points_pca_angle(avail) or 0.0
    ox = float(np.mean([p.x for p in avail]))
    oy = float(np.mean([p.y for p in avail]))
    avail_sorted = sorted(avail, key=lambda p: _rot(p, -ang, origin=(ox, oy)).x)
    need_sorted = sorted(need, key=lambda s: str(s.get("reference")))
    m = len(avail_sorted)
    for i, s in enumerate(need_sorted):
        base[id(s)] = avail_sorted[min(int(i * m / len(need_sorted)), m - 1)]
    return base


def _anchors_are_distributed(spaces, poly, min_frac=0.5, spread_ratio=0.25):
    """True when the spaces' address anchors are DISTINCT and SPREAD across the
    footprint — the "many detached/row houses under one reference" case, where
    each space should sit at its own location rather than in a shared corridor
    layout. False for apartment blocks (all units share the entrance point).
    """
    pts = []
    for s in spaces:
        a = s.get("anchor")
        if a is not None and not getattr(a, "is_empty", True):
            pts.append((round(a.x, 1), round(a.y, 1)))
    if len(spaces) < 2 or len(pts) < 2:
        return False
    distinct = set(pts)
    if len(distinct) < 2 or len(distinct) < min_frac * len(spaces):
        return False
    xs = [p[0] for p in distinct]
    ys = [p[1] for p in distinct]
    adiag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    minx, miny, maxx, maxy = poly.bounds
    pdiag = math.hypot(maxx - minx, maxy - miny) or 1.0
    return (adiag / pdiag) >= spread_ratio


def _bisect_by_anchors(poly, spaces, targets, angle=0.0, origin=(0.0, 0.0)):
    """Recursive coordinate bisection of a SINGLE connected ``poly`` among
    ``spaces`` so each lands near its ``anchor``: at each step split the piece
    across its longer axis at the target-area-balanced boundary, lower-anchor
    spaces to the low side. Clean rectangular regions in address order, areas ≈
    ``targets``. Returns ``{id(space): Polygon}``."""
    from shapely.affinity import rotate as _rot
    rot = abs(angle) > 1e-9
    rp = _rot(poly, -angle, origin=origin) if rot else poly
    c = rp.centroid
    anchors = {}
    for s in spaces:
        a = s.get("anchor")
        if a is not None and not getattr(a, "is_empty", True):
            ar = _rot(a, -angle, origin=origin) if rot else a
            anchors[id(s)] = (ar.x, ar.y)
        else:
            anchors[id(s)] = (c.x, c.y)          # no address → piece centre

    result = {}

    def _w(s):
        return max(targets.get(id(s), 0.0), 1e-6)

    def _rec(sub, subs):
        if sub is None or getattr(sub, "is_empty", True) or not subs:
            return
        if len(subs) == 1:
            result[id(subs[0])] = _clean(sub)
            return
        minx, miny, maxx, maxy = sub.bounds
        axis = 0 if (maxx - minx) >= (maxy - miny) else 1
        subs_sorted = sorted(subs, key=lambda s: anchors[id(s)][axis])
        tot = sum(_w(s) for s in subs_sorted)
        acc, m = 0.0, 1
        for i, s in enumerate(subs_sorted):
            acc += _w(s)
            if acc >= tot / 2.0:
                m = i + 1
                break
        m = max(1, min(len(subs_sorted) - 1, m))
        frac = sum(_w(s) for s in subs_sorted[:m]) / tot
        # _split_polygon_by_area_fraction cuts across the longer bbox axis, which
        # is `axis` here → low piece holds the lower-anchor spaces.
        a, b = utils._split_polygon_by_area_fraction(sub, frac)
        if a is None or getattr(a, "is_empty", True):
            result[id(subs_sorted[0])] = _clean(sub)
            return
        _rec(a, subs_sorted[:m])
        _rec(b, subs_sorted[m:])

    _rec(rp, list(spaces))
    if rot:
        return {k: (_clean(_rot(v, angle, origin=origin)) if v is not None else None)
                for k, v in result.items()}
    return result


def _anchor_axis_angle(spaces, fallback_poly):
    """Cut axis for the anchored layout: the PRINCIPAL DIRECTION of the spaces'
    address anchors (the street line the houses sit along), so the guillotine
    cuts run across that line and tile the houses in address order. Falls back to
    the footprint's own major axis when there are too few distinct anchors."""
    pts, seen = [], set()
    for s in spaces:
        a = s.get("anchor")
        if a is not None and not getattr(a, "is_empty", True):
            key = (round(a.x, 1), round(a.y, 1))
            if key not in seen:
                seen.add(key)
                pts.append(a)
    ang = _points_pca_angle(pts)
    return ang if ang is not None else _major_axis_angle(fallback_poly)


def _partition_by_anchors(poly, spaces, targets, angle=0.0, origin=(0.0, 0.0)):
    """Place each of ``spaces`` near its ``anchor`` address. Component-aware: when
    ``poly`` is several disconnected footprints (detached blocks / a terrace plus
    outbuildings), each space is first assigned to the NEAREST component so it
    can't be smeared across the gap between them, then each component is bisected
    among its own spaces (``_bisect_by_anchors``). The cut axis is derived from the
    anchors themselves (``_anchor_axis_angle``), so ``angle`` here is unused.
    Returns ``{id(space): Polygon}``.
    """
    comps = [c for c in (list(poly.geoms) if hasattr(poly, "geoms") else [poly])
             if c is not None and not c.is_empty and c.area > 0.5]
    if len(comps) <= 1:
        ang = _anchor_axis_angle(spaces, poly)
        return _bisect_by_anchors(poly, spaces, targets, ang, origin)
    buckets = {i: [] for i in range(len(comps))}
    for s in spaces:
        a = s.get("anchor")
        if a is not None and not getattr(a, "is_empty", True):
            j = min(range(len(comps)), key=lambda i: comps[i].distance(a))
        else:                                    # no address → largest component
            j = max(range(len(comps)), key=lambda i: comps[i].area)
        buckets[j].append(s)
    result = {}
    for i, comp in enumerate(comps):
        subs = buckets[i]
        if not subs:
            continue
        ang = _anchor_axis_angle(subs, comp)
        result.update(_bisect_by_anchors(
            comp, subs, targets, ang, (comp.centroid.x, comp.centroid.y)))
    return result


def _polys(geom):
    """Non-empty polygon components of a (Multi)Polygon as a list."""
    if geom is None:
        return []
    return [g for g in (list(geom.geoms) if hasattr(geom, "geoms") else [geom])
            if g is not None and not getattr(g, "is_empty", True)]


def _building_units(fp_by_floor):
    """The distinct PHYSICAL BUILDINGS under a cadastral reference: connected
    components of the footprint unioned across every floor. A reference can hold
    several detached/semi-detached blocks — each is a unit that must receive its
    own building spaces (else only the largest building would get geometry)."""
    geoms = []
    for g in fp_by_floor.values():
        geoms.extend(_polys(g))
    if not geoms:
        return []
    u = unary_union(geoms)
    comps = [g for g in _polys(u) if g.area > 1.0]
    return comps or _polys(u)


def _unit_index_for(comp, units):
    """Index of the building unit that ``comp`` (a per-floor footprint piece)
    belongs to — the unit it overlaps most."""
    best, bi = -1.0, 0
    for i, u in enumerate(units):
        try:
            a = u.intersection(comp).area
        except Exception:
            a = 0.0
        if a > best:
            best, bi = a, i
    return bi


def _space_info_by_reference(row):
    """``{reference: {loc, postal, inner}}`` per building space — the address
    LOCATION (Point), POSTAL address string and INNER address (portal/stair/door).
    All three are dicts keyed by use type, parallel to
    ``br__building_spaces_reference``."""
    refs = row.get("br__building_spaces_reference") or {}
    locs = row.get("br__building_spaces_postal_address_location") or {}
    posts = row.get("br__building_spaces_postal_address") or {}
    inners = row.get("br__building_spaces_inner_address") or {}
    out = {}
    if not isinstance(refs, dict):
        return out

    def _get(d, use, i):
        lst = d.get(use, []) if isinstance(d, dict) else []
        return lst[i] if i < len(lst) else None

    for use, rl in refs.items():
        if not isinstance(rl, (list, tuple)):
            continue
        for i, ref in enumerate(rl):
            p = _as_geom(_get(locs, use, i))
            out[ref] = {
                "loc": p if (p is not None and not getattr(p, "is_empty", True)) else None,
                "postal": _get(posts, use, i),
                "inner": _get(inners, use, i),
            }
    return out


def _building_label_units(row, units):
    """Relate each PARCEL address to a building: for every
    ``postal_addresses_with_extensions`` label that has a well-geocoded location
    (i.e. not the shared fallback point reused by many), the unit nearest that
    location. Returns parallel lists ``(labels, unit_indices)``. This is the
    "which building is related to each address location" map; a space is then
    routed by matching its own postal address to these labels."""
    labels = (row.get("postal_addresses_with_extensions")
              or row.get("postal_addresses_clean"))
    locs = row.get("postal_addresses_location")
    if not isinstance(labels, (list, tuple)) or not isinstance(locs, (list, tuple)) \
            or not units:
        return [], []
    from collections import Counter
    pts = [_as_geom(p) for p in locs]
    cnt = Counter((round(p.x, 1), round(p.y, 1))
                  for p in pts if p is not None and hasattr(p, "x"))
    fallback = {k for k, c in cnt.items() if c > max(3, 0.15 * len(pts))}
    good_labels, good_units = [], []
    for i, lab in enumerate(labels):
        p = pts[i] if i < len(pts) else None
        if p is None or not hasattr(p, "x") or not isinstance(lab, str):
            continue
        if (round(p.x, 1), round(p.y, 1)) in fallback:
            continue
        # Only NUMBERED parcel addresses give a per-dwelling street distribution;
        # an ``S-N`` label is a single point for a whole street and would pull all
        # of that street's spaces to one building, so skip it.
        if _specific_postal(lab) is None:
            continue
        u = min(range(len(units)), key=lambda k: units[k].distance(p))
        good_labels.append(lab)
        good_units.append(u)
    return good_labels, good_units


def _label_unit(postal, good_labels, good_units, min_ratio=0.66,
                margin=0.1, dominance=0.6):
    """Building (unit) the space's ``postal`` most likely belongs to, from the
    well-geocoded parcel addresses. Fuzzy-matches (``difflib``, like
    ``utils._best_address_match_index``) and takes the whole family of similar
    addresses (within ``margin`` of the best ratio — i.e. the same street): if
    that family is DOMINATED by one building it is returned, else None (an
    ambiguous street that spans several buildings is left to area-fill so its
    spaces spread instead of piling into one)."""
    import difflib
    from collections import Counter
    from hypercadaster_ES import downloaders

    def _street(s):
        # Street NAME only (type prefix stripped, number dropped): so many spaces
        # sharing one ``S-N`` postal match the whole street family, and a differing
        # type prefix (CALLE vs URBANIZACION) doesn't block the match.
        return downloaders.strip_street_type(str(s).split(",")[0]) if s else ""

    a = _street(postal)
    if not a or not good_labels:
        return None
    ratios = [difflib.SequenceMatcher(None, a, _street(lab)).ratio()
              for lab in good_labels]
    best = max(ratios)
    if best < min_ratio:
        return None
    fam = Counter(u for r, u in zip(ratios, good_units) if r >= best - margin)
    if not fam:
        return None
    top, cnt = fam.most_common(1)[0]
    return top if cnt / sum(fam.values()) >= dominance else None


def _assign_spaces_to_units(spaces, units, unit_cap, info, label_info=None):
    """Assign each building space to a physical building (unit), using the per-space
    address as far as it is trustworthy and AREA-FILL for the rest so every
    existing building gets spaces.

    A space's address LOCATION is trusted unless it is a shared fallback point (a
    single coordinate reused by many spaces — the ``S-N``/no-number CAT addresses
    collapsed to the building point). Spaces with a trustworthy location are grouped
    by their POSTAL / INNER address (so a dwelling and its garage/annex at the same
    address stay in the same building) and placed in the unit nearest that group.
    The remaining (untrustworthy-address) spaces are area-filled into the units with
    the most remaining capacity. Returns ``{id(space): unit_index}``."""
    from collections import Counter, defaultdict
    from shapely.geometry import Point
    n = len(units)
    remaining = list(unit_cap)
    assign = {}

    def _stot(s):
        try:
            return max(float(s.get("area") or 0.0), 1.0)
        except (TypeError, ValueError):
            return 1.0

    nsp = len(spaces)
    cnt = Counter()
    for s in spaces:
        p = info.get(s.get("reference"), {}).get("loc")
        if p is not None:
            cnt[(round(p.x, 1), round(p.y, 1))] += 1
    fallback = {k for k, c in cnt.items() if c > max(3, 0.15 * nsp)}

    def _loc(s):
        p = info.get(s.get("reference"), {}).get("loc")
        if p is None or (round(p.x, 1), round(p.y, 1)) in fallback:
            return None
        return p

    good_labels, good_units = label_info or ([], [])

    def _postal(s):
        return info.get(s.get("reference"), {}).get("postal")

    # Group by SPECIFIC postal address (street + number). A dwelling and its
    # warehouse/parking share the same postal — they differ only in the inner
    # address — so grouping on postal keeps them in the SAME building. Spaces with
    # no specific postal (``S-N``) are grouped by their (street-only) postal so
    # same-street spaces still travel together.
    groups, singles = defaultdict(list), []
    for s in spaces:
        key = _specific_postal(_postal(s))
        if key is not None:
            groups[key].append(s)
        else:
            singles.append(s)

    def _fill_ratio_unit():
        # Building with the most remaining capacity RELATIVE to its size, so every
        # building fills proportionally and none is left empty (a plain max-remaining
        # would pile everything into the largest buildings first).
        return max(range(n), key=lambda i: remaining[i] / max(unit_cap[i], 1.0))

    def _place(grp):
        # 1) trustworthy per-space location → nearest building.
        pts = [p for p in (_loc(s) for s in grp) if p is not None]
        if pts:
            a = Point(sum(p.x for p in pts) / len(pts),
                      sum(p.y for p in pts) / len(pts))
            order = sorted(range(n), key=lambda i: units[i].distance(a))
            return next((i for i in order if remaining[i] > 0), order[0])
        # 2) else relate the space's postal to a well-geocoded parcel address and
        #    take THAT address's building (the user's address→building idea).
        for s in grp:
            u = _label_unit(_postal(s), good_labels, good_units)
            if u is not None:
                return u
        # 3) else area-fill so no building is left empty.
        return _fill_ratio_unit()

    for _key, grp in sorted(groups.items(),
                            key=lambda kv: -sum(_stot(s) for s in kv[1])):
        u = _place(grp)
        for s in grp:
            assign[id(s)] = u
        remaining[u] -= sum(_stot(s) for s in grp)

    # Singles that share a street label go together; otherwise area-fill.
    single_groups = defaultdict(list)
    for s in singles:
        single_groups[_postal(s) or id(s)].append(s)
    for _key, grp in sorted(single_groups.items(),
                            key=lambda kv: -sum(_stot(s) for s in kv[1])):
        # Keep same-street singles together only when their street resolves to one
        # building; otherwise split them across buildings by proportional fill so a
        # big ``S-N`` street (spanning several buildings) doesn't pile into one.
        u = _place(grp)
        located = any(_loc(s) is not None for s in grp)
        street_resolved = _label_unit(_postal(grp[0]), good_labels, good_units) is not None
        if located or street_resolved:
            for s in grp:
                assign[id(s)] = u
            remaining[u] -= sum(_stot(s) for s in grp)
        else:
            for s in sorted(grp, key=_stot, reverse=True):
                v = _fill_ratio_unit()
                assign[id(s)] = v
                remaining[v] -= _stot(s)
    return assign


def _specific_postal(postal):
    """Normalised postal address when it carries a real STREET NUMBER, else None.
    ``S-N`` / ``S/N`` (sin número) and number-less strings are not specific — they
    can't relate a dwelling to its garage — so they return None."""
    if not postal:
        return None
    s = str(postal).strip()
    low = s.lower()
    if "s-n" in low or "s/n" in low or low.endswith(" sn"):
        return None
    if not any(ch.isdigit() for ch in s):
        return None
    return s


def _partition_floor(floor_poly, gross_poly, present, targets, facade_angle,
                     facade_lines, pool_union, communal_area, major_angle,
                     street_dir, door_xy, is_ground, min_frontage, corridor_w):
    """Partition one floor into per-space polygons (requirements #1/#3/#4).

    Two layouts. When the spaces have distinct, spread-out address anchors (many
    detached/row houses under one reference — ``_anchors_are_distributed``) each
    space is placed near its OWN address via ``_partition_by_anchors``. Otherwise
    the apartment layout: open-air pools (real geometry) → parking/non-occupied
    end blocks (fronting a main façade for the vehicle door) → the communal area
    is FORCED as a compact CENTRAL core (``_carve_central_rect``) → the remaining
    ring is shared among the occupied spaces IN PROPORTION TO THEIR AREA as one
    contiguous wing each (``_subdivide_ring_angular``) → floor-0 corridor from the
    core to the entrance. The ring tiles the footprint EXACTLY (spaces scale to
    fill whatever the forced core leaves — the cadastral footprint, not the CAT
    constructed area, is the physical constraint), so no floor area is dropped or
    over-filled even on C-/U-shaped floors.

    ``present`` are the space dicts on this floor; ``targets`` maps ``id(space)``
    to its per-floor area (the sharing weight). Returns ``(assign, communal_poly,
    deficit)`` where ``assign`` maps ``id(space)`` to a (Multi)Polygon and
    ``deficit`` is the set of ``id(space)`` that could not reach ``min_frontage``
    of main façade."""
    from shapely.affinity import rotate as _rot
    assign, deficit = {}, set()
    work = floor_poly

    pool_spaces, noc, occ = [], [], []
    for s in present:
        if pool_union is not None and str(s.get("use_type", "")).lower().startswith("sport"):
            pool_spaces.append(s)
        elif s.get("kind") == "non_occupied":
            noc.append(s)
        else:
            occ.append(s)

    # 1) pools — real OtherConstruction geometry (clipped to the gross floor).
    for s in pool_spaces:
        try:
            pg = _clean(pool_union.intersection(gross_poly)) or _clean(pool_union)
        except Exception:
            pg = _clean(pool_union)
        if pg is not None and not pg.is_empty:
            assign[id(s)] = pg
            s["kind"] = "non_occupied"
            s["is_pool"] = True
            if work is not None:
                work = _clean(work.difference(pg))

    if work is None or getattr(work, "is_empty", True):
        return assign, None, deficit

    # Location-anchored layout applies to OCCUPIED houses only (many detached/row
    # houses addressed individually). Non-occupied garages are laid out so each
    # FRONTS a main façade — a compact street-fronting block when dwellings share
    # the floor, or (all-garage floor) as the band spaces themselves — so every
    # parking gets ≥ ``min_frontage`` of façade for its vehicle door.
    anchor_occ = bool(occ) and _anchors_are_distributed(occ, work)

    comm_val = communal_area if (communal_area and communal_area > 0.5) else 0.0

    def _flag_frontage(sid, poly):
        # Every space (dwelling AND parking/warehouse) needs ≥ min_frontage of main
        # façade — dwellings for access/windows, garages for the vehicle door. Only
        # assessable when the façade geometry is known (a missing ``facade_lines``
        # means "unknown", not "no façade", so don't false-flag).
        if (poly is not None and not poly.is_empty and facade_lines is not None
                and _facade_frontage(poly, facade_lines) < min_frontage):
            deficit.add(sid)

    # 2) FLOORS WITHOUT A COMMUNAL CORE (ground-floor shops + parking, all-garage
    #    basements, any communal-less floor): EVERY space fronts the main façade, so
    #    partition the whole floor among ALL spaces into proportional, connected
    #    pieces cut NORMAL to the façade (``prefer_angle=facade_angle``) — shops and
    #    garages each reach the street. No central core here.
    if not anchor_occ and (comm_val <= 0.5 or not occ):
        band = occ + noc
        if not band:
            return assign, None, deficit
        regs = _bisect_connected(
            work, band, [targets.get(id(s), 0.0) for s in band], facade_angle)
        for s in band:
            reg = regs.get(id(s))
            if reg is not None and not reg.is_empty:
                assign[id(s)] = reg
                _flag_frontage(id(s), reg)
        return assign, None, deficit

    # 3) RESIDENTIAL floor with a communal core. Parking (if any) is carved first as
    #    a façade-fronting block (its vehicle door reaches the street); the dwellings
    #    then tile the rest.
    for s in noc:
        if work is None or work.is_empty:
            break
        block, work = _carve_parking_block(
            work, targets.get(id(s), 0.0), major_angle, facade_lines,
            street_dir, min_frontage)
        if block is not None and not block.is_empty:
            assign[id(s)] = _clean(block)
            _flag_frontage(id(s), assign[id(s)])

    if work is None or getattr(work, "is_empty", True):
        return assign, None, deficit

    # Occupied houses with spread addresses → place each near its own address.
    if anchor_occ:
        core_poly, remainder = None, work
        if comm_val > 0.5:
            cx, cy = work.centroid.x, work.centroid.y
            core, rem = utils._carve_central_rect(
                work, comm_val, angle=major_angle, origin=(cx, cy))
            core = _clean(core)
            if core is not None and not core.is_empty:
                core_poly, remainder = core, _clean(rem)
        if remainder is not None and not remainder.is_empty:
            regs = _partition_by_anchors(
                remainder, occ, targets, angle=major_angle,
                origin=(remainder.centroid.x, remainder.centroid.y))
            for s in occ:
                p = regs.get(id(s))
                if p is not None and not p.is_empty:
                    assign[id(s)] = p
        if is_ground and core_poly is not None and door_xy is not None:
            core_poly, assign = _entrance_corridor(
                core_poly, floor_poly, door_xy, corridor_w, assign)
        return assign, core_poly, deficit

    # 4) Apartment layout: FORCE the communal area as a compact INTERIOR core (kept
    #    OFF the façade), then share the surrounding ring among the dwellings IN
    #    PROPORTION TO THEIR AREA as single CONNECTED pieces (``_bisect_connected``).
    #    Every zone is one connected polygon and the ring tiles the footprint EXACTLY
    #    — the cadastral footprint (not the CAT constructed area, which can over/
    #    under-subscribe it) is the physical constraint, so nothing is dropped or
    #    over-filled, even on a C-/U-shaped floor.
    comm = min(comm_val, 0.6 * work.area)      # never let the core swallow the floor
    core_poly, ring = None, work
    if comm > 0.5:
        core, rem = _carve_interior_core(work, comm, facade_lines, facade_angle,
                                         min_frontage)
        if core is not None and not core.is_empty and rem is not None and not rem.is_empty:
            core_poly, ring = core, rem
    communal_poly = core_poly

    regs = _bisect_connected(
        ring, occ, [targets.get(id(s), 0.0) for s in occ], facade_angle)
    for s in occ:
        reg = regs.get(id(s))
        if reg is not None and not reg.is_empty:
            assign[id(s)] = reg
            _flag_frontage(id(s), reg)

    # 5) floor-0 entrance corridor from the communal core to the door.
    if is_ground and communal_poly is not None and door_xy is not None:
        communal_poly, assign = _entrance_corridor(
            communal_poly, floor_poly, door_xy, corridor_w, assign)

    return assign, communal_poly, deficit


def _layout_building(ufp, unit_spaces, facade_lines, communal_by_floor, info,
                     door_xy, ground_floor, min_frontage, corridor_w, pool_union):
    """Tile ONE physical building into vertical space COLUMNS, cut PERPENDICULAR to
    its main façade and ALIGNED across floors.

    A dwelling and its garage/store share one ``building_space_reference``, so all
    spaces of a reference form a single column (garage under the dwelling). Columns
    tile the building's façade extent by area, at the SAME absolute cut positions on
    every floor — so party walls stack vertically (clean roofs) and every division
    is normal to the façade. Column widths are proportional to each reference's
    built area plus its share of the communal area (the ``optimal`` area-based
    distribution). Returns ``(assign, communal_polys, deficit)`` where ``assign``
    maps ``id(space) → {floor: Polygon}`` and ``communal_polys`` is ``{floor: Polygon}``.
    """
    from shapely.affinity import rotate as _rot
    from shapely.geometry import box as _box
    from collections import OrderedDict
    assign, communal_polys, deficit = {}, {}, set()
    comm_pieces = {}
    floors = sorted(f for f in ufp if ufp.get(f) is not None and not ufp[f].is_empty)
    if not floors:
        return assign, communal_polys, deficit

    # Pools use their real geometry and are removed from the tiled footprint.
    pool_spaces = [s for s in unit_spaces if pool_union is not None
                   and str(s.get("use_type", "")).lower().startswith("sport")]
    fp_use = {f: ufp[f] for f in floors}
    for s in pool_spaces:
        for fl in s["floors"]:
            if fl not in fp_use:
                continue
            try:
                pg = _clean(pool_union.intersection(fp_use[fl])) or _clean(pool_union)
            except Exception:
                pg = _clean(pool_union)
            if pg is not None and not pg.is_empty:
                assign.setdefault(id(s), {})[fl] = pg
                s["kind"] = "non_occupied"
                s["is_pool"] = True
                fp_use[fl] = _clean(fp_use[fl].difference(pg))
    col_spaces = [s for s in unit_spaces if s not in pool_spaces]
    if not col_spaces:
        return assign, communal_polys, deficit

    # Layout choice. COLUMN (attached/semi-detached houses): a building_space
    # reference spans floors — its garage(s) + dwelling stack into ONE vertical
    # column cut normal to the façade, aligned across floors. PER-FLOOR
    # (apartments): different dwellings are stacked on each floor, so references
    # do NOT repeat across floors — each floor is laid out on its own (double-
    # loaded corridor). Columns only apply when references largely repeat.
    from collections import defaultdict as _dd
    refs_by_floor = _dd(set)
    for s in col_spaces:
        for fl in s["floors"]:
            if fl in fp_use:
                refs_by_floor[fl].add(s.get("reference"))
    all_refs = {s.get("reference") for s in col_spaces}
    max_pf = max((len(v) for v in refs_by_floor.values()), default=0)
    column_like = max_pf > 0 and len(all_refs) <= 1.5 * max_pf

    if not column_like:
        for fl in floors:
            comp = fp_use.get(fl)
            present = [s for s in col_spaces if fl in s["floors"]]
            if comp is None or comp.is_empty or not present:
                continue
            major = _major_axis_angle(comp)
            fangle = _facade_angle(facade_lines, fallback=major)
            occ_ids = {id(s) for s in present if s.get("kind") == "occupied"}
            base = {id(s): max(float(s.get("area") or 0.0) / max(1, len(s["floors"])), 0.0)
                    for s in present}
            priv = sum(v for k, v in base.items() if k in occ_ids)
            other = sum(v for k, v in base.items() if k not in occ_ids)
            cc = communal_by_floor.get(fl, 0.0)
            appends = max(0.0, max(0.0, comp.area - priv - other) - cc)
            targets = {sid: (base[sid] + (base[sid] / priv) * appends
                             if (sid in occ_ids and priv > 1e-9) else base[sid])
                       for sid in base}
            a_fl, c_fl, d_fl = _partition_floor(
                comp, comp, present, targets, fangle, facade_lines, pool_union, cc,
                major, None, door_xy, fl == ground_floor, min_frontage, corridor_w)
            for s in present:
                p = a_fl.get(id(s))
                if p is not None and not getattr(p, "is_empty", True):
                    assign.setdefault(id(s), {})[fl] = p
                if id(s) in d_fl:
                    deficit.add(id(s))
            if c_fl is not None and not c_fl.is_empty:
                comm_pieces.setdefault(fl, []).append(c_fl)
        for fl, pieces in comm_pieces.items():
            g = unary_union(pieces)
            if g is not None and not g.is_empty:
                communal_polys[fl] = g
        return assign, communal_polys, deficit

    fp_union = unary_union([g for g in fp_use.values() if g is not None and not g.is_empty])
    if fp_union is None or fp_union.is_empty:
        return assign, communal_polys, deficit
    ang = _facade_angle(facade_lines, fallback=_major_axis_angle(fp_union))
    ox, oy = fp_union.centroid.x, fp_union.centroid.y

    def _projx(pt):
        return _rot(pt, -ang, origin=(ox, oy)).x

    # One column per building_space_reference (dwelling + its garage/store).
    cols = OrderedDict()
    for s in col_spaces:
        cols.setdefault(s.get("reference") or ("_", id(s)), []).append(s)

    # Column weight = mean per-floor built area; order key = address projection.
    # ``needs_frontage``: EVERY space column (dwellings AND parking/warehouses) needs
    # at least ``min_frontage`` of main façade — dwellings for access/windows,
    # garages/stores for their vehicle door. Only the communal core is exempt (its
    # access is the entrance corridor), and it is kept OFF the façade.
    items = []            # [order_key, weight, spaces_or_None(communal), needs_frontage]
    for _key, mem in cols.items():
        area = sum(float(s.get("area") or 0.0) for s in mem)
        n_fl = len({f for s in mem for f in s["floors"]}) or 1
        projs = [_projx(s["anchor"]) for s in mem
                 if s.get("anchor") is not None and not getattr(s["anchor"], "is_empty", True)]
        items.append([(sum(projs) / len(projs)) if projs else None,
                      max(area / n_fl, 1e-6), mem, True])
    tot_comm = sum(float(v or 0.0) for v in communal_by_floor.values())
    if tot_comm > 0.5:
        items.append([None, tot_comm / max(1, len(floors)), None, False])

    mid = _projx(fp_union.centroid)
    for it in items:
        if it[0] is None:
            it[0] = mid
    items.sort(key=lambda it: it[0])

    # Absolute x cut positions over the building extent (same on every floor).
    # Columns needing façade access get >= min_frontage of width.
    minx, _, maxx, _ = _rot(fp_union, -ang, origin=(ox, oy)).bounds
    span = maxx - minx
    widths, wdef = _widths_with_minimums(
        span, [it[1] for it in items],
        [min_frontage if it[3] else 0.0 for it in items])
    cuts, cum = [], minx
    for it, w, dfl in zip(items, widths, wdef):
        cuts.append((cum, cum + w, w, it[2], dfl))
        cum += w

    for fl in floors:
        rp = _rot(fp_use[fl], -ang, origin=(ox, oy))
        if rp is None or rp.is_empty:
            continue
        _, fminy, _, fmaxy = rp.bounds
        for xlo, xhi, w, mem, col_deficit in cuts:
            strip = _clean(rp.intersection(_box(xlo, fminy - 1, xhi, fmaxy + 1)))
            if strip is None or strip.is_empty:
                continue
            back_strip = _clean(_rot(strip, ang, origin=(ox, oy)))
            if back_strip is None or back_strip.is_empty:
                continue
            here = [] if mem is None else [s for s in mem if fl in s["floors"]]
            if mem is None or not here:
                # Communal column, OR a strip with no space of this reference on
                # this floor (a ground-floor strip below an upper-only dwelling) →
                # the building's communal/void. Kept as separate pieces (unioned
                # at the end without collapsing).
                comm_pieces.setdefault(fl, []).append(back_strip)
                continue
            # Every space (dwelling OR parking/warehouse) needs >= min_frontage of
            # main façade — for access/windows or the vehicle door; flag it when the
            # column can't provide it (col_deficit) or its own strip doesn't reach
            # the façade.
            if len(here) == 1:
                assign.setdefault(id(here[0]), {})[fl] = back_strip
                if (col_deficit or (facade_lines is not None
                        and _facade_frontage(back_strip, facade_lines) < min_frontage)):
                    deficit.add(id(here[0]))
            else:                                    # >1 space on a floor → sub-slice
                wts = [max(float(s.get("area") or 0.0) / max(1, len(s["floors"])), 1e-6)
                       for s in here]
                polys, defs = _slice_band_rot(strip, wts, min_frontage)
                for s, poly, _dfl in zip(here, polys, defs):
                    poly = _clean(_rot(poly, ang, origin=(ox, oy))) if poly is not None else None
                    if poly is not None and not poly.is_empty:
                        assign.setdefault(id(s), {})[fl] = poly
                        if (col_deficit or (facade_lines is not None
                                and _facade_frontage(poly, facade_lines) < min_frontage)):
                            deficit.add(id(s))

    for fl, pieces in comm_pieces.items():
        g = _clean(unary_union(pieces)) if len(pieces) == 1 else unary_union(pieces)
        if g is not None and not g.is_empty:
            communal_polys[fl] = g

    # Ground-floor entrance corridor from the communal column to the door.
    if ground_floor in communal_polys and door_xy is not None and ground_floor in fp_use:
        flat = {sid: m[ground_floor] for sid, m in assign.items() if ground_floor in m}
        cp, flat = _entrance_corridor(
            communal_polys[ground_floor], fp_use[ground_floor], door_xy, corridor_w, flat)
        communal_polys[ground_floor] = cp
        for sid, poly in flat.items():
            if sid in assign:
                assign[sid][ground_floor] = poly
    return assign, communal_polys, deficit


def synthesize_space_geometry(row, simplify_tol=0.3, cat_shrink=None,
                              buffer_distance=None):
    """Per-space region polygons per floor.

    Returns ``list[dict]`` of zone records:
    ``{key, reference, use_type, detailed, kind, floor_polys: {floor: Polygon}}``
    where one record = one building space (so a duplex spans several floors).
    Housing zones that could not secure ``_MIN_FACADE_FRONTAGE`` on a floor carry
    ``facade_deficit=True``.

    A cadastral reference may hold SEVERAL physical buildings; the spaces are
    distributed across all of them (``_building_units`` / ``_assign_spaces_to_
    units``) and each building is laid out by ``_layout_building`` in one of two
    modes: COLUMN (attached/semi-detached houses — a ``building_space_reference``
    spans floors, so its garage + dwelling stack into one vertical column cut
    NORMAL to the building's own main façade, aligned across floors → clean stacked
    walls/roofs); or PER-FLOOR (apartments — different dwellings on each floor →
    ``_partition_floor``'s double-loaded-corridor layout). ``cat_shrink`` /
    ``buffer_distance`` are accepted for API compatibility but the zones now tile
    the gross per-building footprints. Housing zones below ``_MIN_FACADE_FRONTAGE``
    of real façade contact carry ``facade_deficit=True``.
    """
    fp_raw = _footprints_by_floor(row)
    if not fp_raw:
        return []
    # Zone identity comes from the authoritative ``br__thermal_zones`` when
    # present (so the EnergyPlus zones are the same set); fall back to the
    # independent CAT-based derivation only when it is absent.
    spaces = _zones_from_thermal_zones(row) or _building_spaces(row)
    if not spaces:
        return []

    # A cadastral reference can hold SEVERAL physical buildings (detached / semi-
    # detached blocks). Each is a "unit"; the building spaces must be distributed
    # across ALL of them, otherwise the footprint collapses to its largest piece
    # and only that one building ever gets geometry.
    units = _building_units(fp_raw)
    multi = len(units) > 1

    # Address info drives (a) which BUILDING a space lands in — ``_assign_spaces_
    # to_units`` (multi only) — and (b) the WITHIN-building layout via anchors.
    info = _space_info_by_reference(row)
    if multi:
        from collections import Counter as _Counter
        _cnt = _Counter()
        for s in spaces:
            p = info.get(s.get("reference"), {}).get("loc")
            if p is not None:
                _cnt[(round(p.x, 1), round(p.y, 1))] += 1
        _fb = {k for k, c in _cnt.items() if c > max(3, 0.15 * len(spaces))}
        for s in spaces:
            p = info.get(s.get("reference"), {}).get("loc")
            s["anchor"] = (p if (p is not None
                                 and (round(p.x, 1), round(p.y, 1)) not in _fb) else None)
        unit_cap = [0.0] * len(units)
        for _fl, _geom in fp_raw.items():
            for _c in _polys(_geom):
                unit_cap[_unit_index_for(_c, units)] += _c.area
        unit_of_space = _assign_spaces_to_units(
            spaces, units, unit_cap, info, label_info=_building_label_units(row, units))
    else:
        anchor_by_id = _resolve_space_anchors(row, spaces)
        for s in spaces:
            s["anchor"] = anchor_by_id.get(id(s))
        unit_of_space = {id(s): 0 for s in spaces}

    pools = _pool_geoms(row)
    pool_union = unary_union(pools) if pools else None
    communal = _communal_area_by_floor(row)
    facade_by_floor = _facade_lines_by_floor(row)
    all_facade = (unary_union([g for g in facade_by_floor.values()])
                  if facade_by_floor else None)
    door_xy = _address_xy(row)

    for s in spaces:
        s["floor_polys"] = {}
    _core_id = _communal_zone_id(row)
    communal_zone = {"zone_id": _core_id, "reference": _core_id,
                     "use_type": "Communal", "detailed": None, "kind": "communal",
                     "floors": [], "floor_polys": {}}
    communal_pieces = {}

    def _base(s):
        return max(float(s.get("area") or 0.0) / max(1, len(s["floors"])), 0.0)

    def _is_pool(s):
        return pool_union is not None and str(s["use_type"]).lower().startswith("sport")

    # Per-building rationalized footprints — snapping each building's upper floors
    # to its ground floor keeps the floors ALIGNED, which the column layout relies
    # on for clean, stacked walls/roofs.
    unit_fp_raw = {u: {} for u in range(len(units))}
    for _fl, _geom in fp_raw.items():
        for _c in _polys(_geom):
            _u = _unit_index_for(_c, units)
            _prev = unit_fp_raw[_u].get(_fl)
            unit_fp_raw[_u][_fl] = (unary_union([_prev, _c]) if _prev is not None else _c)
    unit_fp = {u: _rationalize_floors(m, simplify_tol) for u, m in unit_fp_raw.items()}

    floors_all = sorted(fp_raw)
    ground_floor = floors_all[0] if floors_all else 0
    floor_priv = {fl: sum(_base(s) for s in spaces if fl in s["floors"]
                          and s["kind"] == "occupied" and not _is_pool(s))
                  for fl in floors_all}

    for u in range(len(units)):
        ufp = {f: g for f, g in unit_fp[u].items() if g is not None and not g.is_empty}
        u_spaces = [s for s in spaces if unit_of_space.get(id(s)) == u]
        if not ufp or not u_spaces:
            continue
        # This building's share of each floor's communal (pro-rata by occupied area).
        unit_comm = {}
        for fl in ufp:
            up = sum(_base(s) for s in u_spaces if fl in s["floors"]
                     and s["kind"] == "occupied" and not _is_pool(s))
            unit_comm[fl] = (communal.get(fl, 0.0) * up / floor_priv[fl]
                             if floor_priv.get(fl, 0.0) > 1e-9 else 0.0)
        # This building's main-façade lines (drive the perpendicular cut axis).
        ufp_union = unary_union(list(ufp.values()))
        ufacade = None
        if all_facade is not None:
            try:
                cl = all_facade.intersection(ufp_union.buffer(2.0))
                ufacade = cl if not getattr(cl, "is_empty", True) else None
            except Exception:
                ufacade = None

        a_u, c_u, d_u = _layout_building(
            ufp, u_spaces, ufacade, unit_comm, info, door_xy, ground_floor,
            _MIN_FACADE_FRONTAGE, _CORRIDOR_WIDTH, pool_union)

        for s in u_spaces:
            m = a_u.get(id(s))
            if not m:
                continue
            for fl, poly in m.items():
                if poly is not None and not getattr(poly, "is_empty", True):
                    s["floor_polys"][fl] = poly
            if id(s) in d_u:
                s["facade_deficit"] = True
        for fl, poly in c_u.items():
            if poly is not None and not getattr(poly, "is_empty", True):
                communal_pieces.setdefault(fl, []).append(poly)

    # Communal spans every building's core; keep the pieces separate (a union then
    # ``_clean`` would drop all but the largest and lose most of the communal area).
    for fl, pieces in communal_pieces.items():
        g = unary_union(pieces)
        if g is not None and not g.is_empty:
            communal_zone["floor_polys"][fl] = g
            communal_zone["floors"].append(fl)

    out = [s for s in spaces if s["floor_polys"]]
    if communal_zone["floor_polys"]:
        out.append(communal_zone)

    # Snap the tiled floors through the planar arrangement of the cell complex
    # (topology.py) so neighbouring spaces share EXACT edges — no slivers, gaps or
    # overlaps. Keyed by zone index so a dwelling and its garage (same reference,
    # different floors) stay distinct.
    try:
        from hypercadaster_ES import topology as _topo
        snapped = _topo.snap_floor_polys({i: z["floor_polys"] for i, z in enumerate(out)})
        for i, z in enumerate(out):
            snp = snapped.get(i, {})
            for fl in list(z["floor_polys"].keys()):
                if fl in snp and snp[fl] is not None and not snp[fl].is_empty:
                    z["floor_polys"][fl] = snp[fl]
    except Exception as _e:
        sys.stderr.write(f"  cell-complex snap skipped ({_e})\n")
    return out


# --------------------------------------------------------------------------- #
# z levels + IDF assembly
# --------------------------------------------------------------------------- #
def _floor_z(f, ground_floor_height, floor_height):
    """(z_bottom, z_top) for floor index ``f``, ground at z=0."""
    if f == 0:
        return 0.0, ground_floor_height
    if f > 0:
        z0 = ground_floor_height + (f - 1) * floor_height
        return z0, z0 + floor_height
    return f * floor_height, (f + 1) * floor_height  # basements below 0


def _lines(geom):
    """Yield LineStrings from a possibly-collection geometry."""
    if geom is None or getattr(geom, "is_empty", True):
        return
    gt = geom.geom_type
    if gt == "LineString":
        yield geom
    elif gt in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            yield from _lines(g)


def _wall_quad_outward(x0, y0, x1, y1, z0, z1, poly=None, eps=0.05):
    """Vertices of a wall panel, wound so its normal points OUT of ``poly``.

    A wall segment can be walked in either direction, and the two orders give
    opposite normals. EnergyPlus expects an exterior surface counterclockwise seen
    from outside, and everything hung off a façade downstream — balconies, awnings,
    exterior blinds, window reveals — is positioned along that normal, so a
    consistently wrong winding puts all of it inside the dwelling.

    The room's own plan polygon decides: step off the segment midpoint along the
    candidate normal, and if that lands INSIDE the room, take the other order. This
    is settled at build time, from the geometry that defines the room, rather than
    guessed downstream from centroids or envelopes — neither of which survives a
    concave or wrapped-around zone.
    """
    a = [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
    if poly is None:
        return a
    dx, dy = x1 - x0, y1 - y0
    h = math.hypot(dx, dy)
    if h < 1e-12:
        return a
    # Normal of order ``a`` (right-hand rule on its first two edges).
    nx, ny = dy / h, -dx / h
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    probe = max(1e-3, min(eps, h / 4.0))
    try:
        from shapely.geometry import Point as _Pt
        if poly.contains(_Pt(mx + nx * probe, my + ny * probe)):
            return [(x1, y1, z0), (x0, y0, z0), (x0, y0, z1), (x1, y1, z1)]
    except Exception:  # noqa: BLE001 - winding stays as-is if the test cannot run
        pass
    return a


def _add_surface(idf, name, stype, zone, verts, bc, bc_obj=""):
    o = idf.newidfobject(
        "BUILDINGSURFACE:DETAILED", Name=name, Surface_Type=stype,
        Construction_Name="", Zone_Name=zone,
        Outside_Boundary_Condition=bc, Outside_Boundary_Condition_Object=bc_obj,
        Sun_Exposure="SunExposed" if bc == "Outdoors" else "NoSun",
        Wind_Exposure="WindExposed" if bc == "Outdoors" else "NoWind")
    o.setcoords(verts)
    return o


def _build_surfaces(idf, prisms, eps=0.05):
    """Build all zones + surfaces for a building, wiring inter-zone adjacency
    BY CONSTRUCTION from the region partition.

    ``prisms`` = list of dicts ``{name, poly, z0, z1, floor, floor_bc}``. Each
    prism is one EnergyPlus zone. Interior walls/slabs are generated from the
    geometric overlap (shared edge in plan for walls, plan-overlap for slabs)
    between zones, so paired surfaces have identical coordinates → valid E+
    adjacency. The non-shared remainder is exterior (Outdoors/Ground).
    """
    from collections import defaultdict
    for p in prisms:
        p["poly"] = orient(_clean(p["poly"]), sign=1.0)
        idf.newidfobject("ZONE", Name=p["name"])

    # ---- Vertical walls (per floor) -------------------------------------
    by_floor = defaultdict(list)
    for p in prisms:
        by_floor[p["floor"]].append(p)

    # Adjacency tolerance: two zones are treated as sharing a wall when one
    # zone's boundary runs within ``adj_tol`` of the other. This catches
    # near-coincident partitions (e.g. the eroded communal core vs the
    # surrounding zones) whose edges don't overlap EXACTLY — otherwise both
    # sides would wrongly become Outdoors and receive windows.
    adj_tol = 0.4
    for fl, ps in by_floor.items():
        # Interior walls: shared edges between each region pair. The shared edge
        # is taken from ONE zone's boundary and used for BOTH paired surfaces
        # (identical coords → valid E+ adjacency), so a small mismatch between
        # the two zones' edges is closed rather than left as an exterior gap.
        shared_per_zone = defaultdict(list)  # name -> list of shared LineStrings
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if not ps[i]["poly"].buffer(adj_tol).intersects(ps[j]["poly"]):
                    continue
                inter = ps[i]["poly"].boundary.intersection(
                    ps[j]["poly"].buffer(adj_tol))
                z0, z1 = ps[i]["z0"], ps[i]["z1"]
                k = 0
                for seg in _lines(inter):
                    if seg.length < eps:
                        continue
                    coords = list(seg.coords)
                    for c in range(len(coords) - 1):
                        (x0, y0), (x1, y1) = coords[c][:2], coords[c + 1][:2]
                        if math.hypot(x1 - x0, y1 - y0) < eps:
                            continue
                        na = f"{ps[i]['name']}_iw{ps[j]['name']}_{k}"
                        nb = f"{ps[j]['name']}_iw{ps[i]['name']}_{k}"
                        va = _wall_quad_outward(x0, y0, x1, y1, z0, z1,
                                                ps[i]["poly"], eps)
                        _add_surface(idf, na, "Wall", ps[i]["name"], va,
                                     "Surface", nb)
                        _add_surface(idf, nb, "Wall", ps[j]["name"],
                                     list(reversed(va)), "Surface", na)
                        k += 1
                    shared_per_zone[ps[i]["name"]].append(seg)
                    shared_per_zone[ps[j]["name"]].append(seg)
        # Exterior walls: boundary edges not on any shared segment. The shared
        # buffer uses ``adj_tol`` so a neighbour's slightly-offset edge is still
        # recognised as interior (never emitted as an exterior wall/window).
        from shapely.geometry import Point
        # Union of ALL zone footprints on this floor. Prisms are oriented CCW, so
        # an exterior-ring edge's OUTWARD normal is (dy, -dx). If the point just
        # OUTSIDE an edge lands inside the union (another zone is there), the edge
        # is an interior wall (adjacency missed it, e.g. the eroded communal
        # core running alongside a zone) → make it ADIABATIC, never a windowed
        # façade. A true façade/patio/setback edge points to empty space.
        all_u = unary_union([p["poly"] for p in ps]).buffer(-0.02)
        for p in ps:
            shared = shared_per_zone.get(p["name"], [])
            shared_u = unary_union([s.buffer(adj_tol) for s in shared]) if shared else None
            own = p["poly"]
            z0, z1 = p["z0"], p["z1"]
            wi = 0
            # The exterior ring plus every interior ring (patio void): patio
            # walls must exist so the zone stays enclosed now that slabs/roofs
            # correctly exclude the voids.
            rings = [list(p["poly"].exterior.coords)]
            rings += [list(h.coords) for h in p["poly"].interiors]
            for ring in rings:
                for c in range(len(ring) - 1):
                    (x0, y0), (x1, y1) = ring[c], ring[c + 1]
                    ex, ey = x1 - x0, y1 - y0
                    L_e = math.hypot(ex, ey)
                    if L_e < 0.1:
                        continue  # skip near-zero edges → no degenerate wall
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    if shared_u is not None and shared_u.contains(Point(mx, my)):
                        continue  # this edge is an interior (shared) wall
                    # Outward = the side NOT inside this zone; if that side lands
                    # inside another zone the edge is interior → Adiabatic.
                    nx, ny = ey / L_e, -ex / L_e
                    pa = Point(mx + nx * 0.3, my + ny * 0.3)
                    pb = Point(mx - nx * 0.3, my - ny * 0.3)
                    outside = pb if own.contains(pa) else pa
                    bc = "Adiabatic" if all_u.contains(outside) else "Outdoors"
                    _add_surface(idf, f"{p['name']}_ew{wi}", "Wall", p["name"],
                                 _wall_quad_outward(x0, y0, x1, y1, z0, z1, own, eps),
                                 bc)
                    wi += 1

    # ---- Horizontal slabs (floor / ceiling / roof) ----------------------
    # For each prism: bottom face and top face, split by overlap with the
    # prisms directly below/above into interior (Surface) + exterior parts.
    eps_a = 0.25  # m^2 area tolerance
    for p in prisms:
        below = [q for q in prisms if abs(q["z1"] - p["z0"]) < 1e-6]
        top_neigh = [q for q in prisms if abs(q["z0"] - p["z1"]) < 1e-6]

        # Bottom face
        rem = p["poly"]
        for q in below:
            inter = _clean(p["poly"].intersection(q["poly"]))
            if inter is not None and inter.area > eps_a:
                nm = f"{p['name']}_flrTo_{q['name']}"
                om = f"{q['name']}_clgTo_{p['name']}"
                _add_horizontal(idf, nm, "Floor", p["name"], inter, p["z0"], "Surface", om, down=True)
                _add_horizontal(idf, om, "Ceiling", q["name"], inter, p["z0"], "Surface", nm, down=False)
                rem = _clean(rem.difference(q["poly"])) if rem is not None else None
        if rem is not None and rem.area > eps_a:
            bc = p["floor_bc"]
            _add_horizontal(idf, f"{p['name']}_flrExt", "Floor", p["name"], rem, p["z0"], bc, "", down=True)

        # Top face (only the part not covered by a zone above is roof)
        rem = p["poly"]
        for q in top_neigh:
            inter = _clean(p["poly"].intersection(q["poly"]))
            if inter is not None and inter.area > eps_a:
                rem = _clean(rem.difference(q["poly"])) if rem is not None else None
                # adjacency already added as the upper zone's floor↔ceiling above
        if rem is not None and rem.area > eps_a:
            _add_horizontal(idf, f"{p['name']}_roof", "Roof", p["name"], rem, p["z1"], "Outdoors", "", down=False)


def _build_surfaces_topological(idf, prisms, eps=0.05):
    """Build all zones + surfaces from the CELL COMPLEX (``topology.CellComplex``).

    Every wall/slab/roof is a Face of the complex — interior faces know their two
    cells, so paired surfaces get identical coordinates by construction (valid
    EnergyPlus adjacency) with no pairwise buffering or tolerance probing. Mirrors
    the output contract of ``_build_surfaces`` (ZONE per prism, paired interior
    walls, floor/ceiling slabs, roofs, exterior walls)."""
    from hypercadaster_ES import topology as _topo
    for p in prisms:
        p["poly"] = orient(_clean(p["poly"]), sign=1.0)
        idf.newidfobject("ZONE", Name=p["name"])
    cx = _topo.CellComplex.from_prisms(prisms)

    wi = [0]
    # Plan polygon of each zone, so every wall can be wound to face OUT of the room
    # it belongs to (see ``_wall_quad_outward``).
    zone_poly = {p["name"]: p.get("poly") for p in prisms}

    def _wall_quads(line, z0, z1, zone_a, zone_b=None):
        coords = list(line.coords)
        for c in range(len(coords) - 1):
            (x0, y0), (x1, y1) = coords[c][:2], coords[c + 1][:2]
            if math.hypot(x1 - x0, y1 - y0) < eps:
                continue
            k = wi[0]; wi[0] += 1
            va = _wall_quad_outward(x0, y0, x1, y1, z0, z1,
                                    zone_poly.get(zone_a), eps)
            if zone_b is not None:
                na, nb = f"{zone_a}_iw{k}", f"{zone_b}_iw{k}"
                # The paired face is the same quad seen from the other room, so it
                # is the mirror of this one: identical coordinates, reversed winding.
                _add_surface(idf, na, "Wall", zone_a, va, "Surface", nb)
                _add_surface(idf, nb, "Wall", zone_b, list(reversed(va)),
                             "Surface", na)
            else:
                _add_surface(idf, f"{zone_a}_ew{k}", "Wall", zone_a, va, "Outdoors")

    hz = [0]
    for f in cx.faces:
        if f.kind == "wall_interior":
            _wall_quads(f.geometry, f.z0, f.z1, f.cell_a, f.cell_b)
        elif f.kind == "wall_exterior":
            _wall_quads(f.geometry, f.z0, f.z1, f.cell_a)
        elif f.kind == "slab":                 # cell_a below, cell_b above
            k = hz[0]; hz[0] += 1
            nm, om = f"{f.cell_b}_flr{k}", f"{f.cell_a}_clg{k}"
            _add_horizontal(idf, nm, "Floor", f.cell_b, f.geometry, f.z0, "Surface", om, down=True)
            _add_horizontal(idf, om, "Ceiling", f.cell_a, f.geometry, f.z0, "Surface", nm, down=False)
        elif f.kind == "roof":
            k = hz[0]; hz[0] += 1
            _add_horizontal(idf, f"{f.cell_a}_roof{k}", "Roof", f.cell_a,
                            f.geometry, f.z0, "Outdoors", "", down=False)
        elif f.kind == "floor":
            k = hz[0]; hz[0] += 1
            _add_horizontal(idf, f"{f.cell_a}_flrExt{k}", "Floor", f.cell_a,
                            f.geometry, f.z0, f.bc, "", down=True)
    return cx


def _zones_to_prisms(zones, ox, oy, eff_gfh, floor_height):
    """``zones`` (``synthesize_space_geometry`` output) → prism dicts for the cell
    complex / IDF. One prism per (space, floor, polygon-piece); pools excluded.
    Returns ``(prisms, zone_meta)`` where ``zone_meta[name.upper()]`` is the space
    dict. Shared by ``build_energy_model`` and ``building_cell_complex``."""
    from shapely import affinity
    prisms, zone_meta, used, zi = [], {}, set(), 0
    for s in zones:
        if s.get("is_pool"):
            continue
        base = _safe(s.get("reference") or s.get("zone_id") or s.get("use_type"))
        for fl, poly in sorted(s["floor_polys"].items()):
            if poly is None or getattr(poly, "is_empty", True):
                continue
            z0, z1 = _floor_z(fl, eff_gfh, floor_height)
            for piece in _polys(poly.buffer(0) if poly.geom_type == "Polygon" else poly):
                p = _clean(piece)
                if p is None or p.area < 1.0:
                    continue
                p = affinity.translate(p, xoff=-ox, yoff=-oy)
                zname = f"{base}_F{fl}"
                while zname.upper() in used:
                    zname = f"{base}_F{fl}_{zi}"
                used.add(zname.upper())
                prisms.append({"name": zname, "poly": p, "z0": z0, "z1": z1,
                               "floor": fl,
                               "floor_bc": "Ground" if fl <= 0 else "Outdoors",
                               "space_key": base, "use_type": s.get("use_type"),
                               "kind": s.get("kind")})
                zone_meta[zname.upper()] = s
                zi += 1
    return prisms, zone_meta


def building_cell_complex(gdf, building_reference, wd=None,
                          floor_height=None, ground_floor_height=None,
                          simplify_tol=0.3):
    """Non-manifold CELL COMPLEX (``topology.CellComplex``) of one building —
    cells = spaces, faces = shared walls/slabs/roofs, plus a ``networkx`` space
    adjacency graph (``.graph``). A permissive "building intelligence" artifact
    (the concept popularised by AGPL topologicpy, reimplemented here MIT).

    ``wd`` auto-fetches the heavy BLOB columns from the central DuckDB, like the
    ``plot_building_*`` helpers.
    """
    from hypercadaster_ES import topology as _topo
    if floor_height is None:
        floor_height = _FLOOR_HEIGHT
    if ground_floor_height is None:
        ground_floor_height = _GROUND_FLOOR_HEIGHT
    sub = gdf[gdf["building_reference"] == building_reference]
    if sub.empty:
        raise ValueError(f"building {building_reference} not found")
    row = sub.iloc[0].copy()

    heavy = ["br__thermal_zones", "br__building_footprint_by_floor", "br__wall_segments"]
    missing = [c for c in heavy if c not in row.index or row.get(c) is None]
    if wd is not None and missing:
        cc = row.get("cadaster_code")
        if cc is not None:
            try:
                for k, v in utils.get_heavy_columns_for_building(
                        wd, str(cc), building_reference, table="br_results",
                        columns=missing).items():
                    row[k] = v
            except Exception as _e:
                sys.stderr.write(f"  heavy-col fetch failed: {_e}\n")
    for _c in list(row.index):                 # decode WKB / pickled BLOBs
        v = row[_c]
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            try:
                row[_c] = _wkb.loads(b, include_srid=True)
            except Exception:
                try:
                    import pickle as _pkl
                    row[_c] = _pkl.loads(b)
                except Exception:
                    pass

    zones = synthesize_space_geometry(row, simplify_tol=simplify_tol)
    if not zones:
        raise ValueError(f"no space geometry for {building_reference}")
    fp = _footprints_by_floor(row)
    if fp:
        base = fp.get(0) or next(iter(fp.values()))
        ox, oy = base.centroid.x, base.centroid.y
    else:
        ox, oy = 0.0, 0.0
    try:
        n_ag = int(row.get("br__floors_above_ground"))
    except (TypeError, ValueError):
        n_ag = len([f for f in fp if f >= 0]) if fp else 1
    eff_gfh = utils.effective_ground_floor_height(
        n_ag, ground_floor_height, floor_height, row.get("building_use"))
    prisms, _meta = _zones_to_prisms(zones, ox, oy, eff_gfh, floor_height)
    return _topo.CellComplex.from_prisms(prisms)


def _clean_ring(coords, area_min=0.05, tol=1e-4):
    """Drop consecutive-duplicate and colinear vertices from a ring and reject
    degenerate/sliver polygons. Colinear leading vertices make EnergyPlus fail
    with 'Invalid dot product' (no computable surface normal), so they must go."""
    pts = []
    for x, y in coords:
        if not pts or abs(x - pts[-1][0]) > 1e-7 or abs(y - pts[-1][1]) > 1e-7:
            pts.append((x, y))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return []
    keep = []
    m = len(pts)
    for i in range(m):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % m]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol:
            keep.append(b)
    if len(keep) < 3:
        return []
    try:
        if Polygon(keep).area < area_min:
            return []
    except Exception:
        return []
    return keep


def _triangulate_polygon(sub):
    """Delaunay triangles clipped to ``sub`` (respects holes); list of rings."""
    from shapely.ops import triangulate
    out = []
    pieces = []
    for tri in triangulate(sub):
        clip = sub.intersection(tri)
        geoms = (clip.geoms if clip.geom_type in
                 ("MultiPolygon", "GeometryCollection") else [clip])
        for g in geoms:
            if getattr(g, "geom_type", "") == "Polygon" and g.area > 1e-6:
                pieces.append(g)
    if pieces and abs(sum(p.area for p in pieces) - sub.area) <= 0.02 * sub.area + 0.5:
        for g in pieces:
            if g.interiors:                          # hole inside a triangle
                for t in triangulate(g):
                    if g.contains(t.representative_point()):
                        out.append(list(t.exterior.coords)[:-1])
            else:
                out.append(list(g.exterior.coords)[:-1])
        return out
    return [list(sub.exterior.coords)[:-1]]          # safe fallback


def _horizontal_rings(poly, max_verts=20):
    """Return simple (hole-free) coordinate rings whose union is exactly
    ``poly``. Polygons with interior rings (patios, or the annular roof/slab
    remainder where another zone sits above part of a floor) — or with too many
    vertices for one EnergyPlus surface — are TRIANGULATED (Delaunay clipped to
    the polygon) so the surface never spans its holes and never overflows the
    per-surface vertex limit. Prevents slab/roof area outside the building
    volume. Rings are simplified first to drop colinear/duplicate artefacts;
    the same input polygon always yields the same rings, so stacked floor↔ceiling
    adjacency stays coordinate-identical."""
    if poly is None or getattr(poly, "is_empty", True):
        return []
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    raw = []
    for sub in polys:
        try:
            sub = sub.simplify(0.05, preserve_topology=True)
        except Exception:
            pass
        if sub.is_empty or sub.geom_type != "Polygon" or sub.area <= 1e-6:
            continue
        ext = list(sub.exterior.coords)[:-1]
        # Emit as a single surface only if it is hole-free, small enough AND
        # convex; otherwise triangulate. Non-convex casting surfaces make
        # EnergyPlus shadowing fail ('surfaces ... non-convex').
        try:
            convex = abs(sub.convex_hull.area - sub.area) <= 1e-6 * sub.area + 1e-9
        except Exception:
            convex = False
        if not sub.interiors and len(ext) <= max_verts and convex:
            raw.append(ext)
        else:
            raw.extend(_triangulate_polygon(sub))
    # Remove colinear/duplicate vertices and drop degenerate slivers so every
    # emitted surface has a valid normal (avoids E+ 'Invalid dot product').
    cleaned = []
    for r in raw:
        cr = _clean_ring(r)
        if cr:
            cleaned.append(cr)
    return cleaned


def _add_horizontal(idf, name, stype, zone, poly, z, bc, bc_obj, down):
    """Add a horizontal surface (floor/ceiling/roof) for `poly` at height `z`.
    `down`=True → normal points down (floor): vertices CW in plan; else up.
    Polygons with holes (or too many vertices) are triangulated so the surface
    excludes the voids. When that splits an inter-zone adjacency into several
    pieces, the paired surface's BC object is suffixed identically — both sides
    triangulate the SAME polygon deterministically, so piece i pairs with piece
    i (keeping floor↔ceiling adjacency resolvable)."""
    poly = orient(_clean(poly), sign=1.0)
    if poly is None:
        return
    rings = [r for r in _horizontal_rings(poly) if len(r) >= 3]
    multi = len(rings) > 1
    for n, ring in enumerate(rings):
        seq = list(reversed(ring)) if down else ring
        sname = f"{name}_{n}" if multi else name
        sbc_obj = f"{bc_obj}_{n}" if (multi and bc_obj) else bc_obj
        _add_surface(idf, sname, stype, zone,
                     [(x, y, z) for (x, y) in seq], bc, sbc_obj)


def _seed_idf(idf_cls):
    seed = (f"Version,{_idd_version};\n"
            "Building, Building, 0, , , , , , ;\n"
            "GlobalGeometryRules, UpperLeftCorner, Counterclockwise, World;\n")
    path = os.path.join("/tmp", "_hc_energy_seed.idf")
    with open(path, "w") as fh:
        fh.write(seed)
    return idf_cls(path)


def build_energy_model(gdf, building_reference, wd=None, output_path=None,
                       floor_height=None, ground_floor_height=None,
                       wwr=None, wwr_model=None, simplify_tol=0.3,
                       idd_path=None, geometry_only=False, weather_year=None,
                       epw_path=None, era5_dir=None, epw_dir=None,
                       cat_shrink=None, buffer_distance=None,
                       output_frequencies=None, residential_behavior=None,
                       residential_period=None, behavior_seed=0,
                       filter_shading=None, max_shading=_FROM_CONFIG):
    """Build an EnergyPlus IDF for one building and return the ``geomeppy.IDF``.

    Parameters
    ----------
    gdf : (Geo)DataFrame carrying the ``br__*`` inference columns.
    building_reference : str
    output_path : str, optional — where to write the ``.idf``.
    floor_height, ground_floor_height : storey heights (m).
    wwr : default window-to-wall ratio for exterior façades (used when no
        ``wwr_model`` is supplied).
    wwr_model : optional WWR predictor — a float override, a callable
        ``f(features)→ratio``, an estimator with ``.predict``, or a path to a
        serialised (joblib/pickle) estimator. When given, the per-building ratio
        is predicted from the building's characteristics via ``building_wwr``
        (features in ``_wwr_features``); result clamped to ``_WWR_BOUNDS``.
    simplify_tol : ground-floor simplification tolerance (m).
    geometry_only : if True, stop after geometry/zones/surfaces/windows; if
        False (default) also add constructions, schedules, internal loads,
        ideal-loads HVAC and simulation control via ``add_energy_systems`` —
        making the IDF a runnable annual model (needs an EnergyPlus engine +
        EPW weather file to actually simulate).
    weather_year : int, optional — if given (and not ``geometry_only``), attach
        the ERA5-Land EPW for that natural year at the ERA5-Land cell closest to
        this building, so ``idf.run()`` simulates with real local weather. The
        EPW is cached per (cell, year) and reused across municipalities sharing
        the cell (coastal centroids resolve to the nearest cell with data).
    epw_path : str, optional — attach this EPW directly (overrides
        ``weather_year`` resolution).
    era5_dir, epw_dir : ERA5-Land GRIB/parquet and EPW cache directories;
        default ``{wd}/weather/era5land`` and ``{wd}/weather/epw``.
    filter_shading : drop NEIGHBOUR-shading panels that cannot shade this
        building — below its horizon from every vantage, in the N/NE/NW sector
        (no direct sun in the northern hemisphere), or hidden behind a nearer
        panel (``_filter_relevant_shading``). The building's own self-shading
        (``CatShade_*``, balconies, awnings, reveals) is always kept. ``None``
        (default) follows ``shading.neighbours.filter`` in the config.
    max_shading : hard cap on the neighbour panels surviving the filter, largest
        first (E+ runtime grows with panel count); ``None`` = no cap, default
        follows ``shading.neighbours.max_panels``.

    Both are applied **here, as the model is built**, so the ``.idf`` written to
    ``output_path``, the geometry :func:`plot_idf_geometry` draws and the panels
    EnergyPlus actually simulates are one and the same set. The returned IDF is
    marked ``_hc_shading_filtered`` so :func:`run_behavior_scenarios` does not
    redo the work.

    Storey heights, the ground-floor rule, window placement, materials, loads
    and other defaults come from ``energy_config.yaml`` (see ``configure``).
    """
    if floor_height is None:
        floor_height = _FLOOR_HEIGHT
    if ground_floor_height is None:
        ground_floor_height = _GROUND_FLOOR_HEIGHT
    if wwr is None:
        wwr = _DEFAULT_WWR
    if cat_shrink is None:
        cat_shrink = _CAT_AREA_SHRINK
    if buffer_distance is None:
        buffer_distance = _BUFFER_DISTANCE
    IDF = _ensure_idd(idd_path)

    sub = gdf[gdf["building_reference"] == building_reference]
    if sub.empty:
        raise ValueError(f"building {building_reference} not found")
    row = sub.iloc[0].copy()

    # Heavy columns (neighbour obstructions + typed wall segments) are excluded
    # from the bulk frame — fetch them on demand from the central DuckDB. The
    # wall segments (``br__wall_segments``: geometry + type + floor) drive the
    # exterior-wall boundary conditions and the anchoring of adiabatic/party
    # walls, so they must be present before synthesis / CAT shrink.
    def _missing(v):
        return v is None or (hasattr(v, "__len__") and len(v) == 0)
    if wd is not None:
        need = [c for c in ("br__shadows", "br__wall_segments")
                if _missing(row.get(c))]
        cc = row.get("cadaster_code")
        if need and cc is not None:
            try:
                fetched = utils.get_heavy_columns_for_building(
                    wd, str(cc), building_reference, columns=need)
                for k, v in (fetched or {}).items():
                    row[k] = v
            except Exception as _e:
                sys.stderr.write(f"  heavy cols: fetch failed ({_e})\n")
    shadows = row.get("br__shadows")

    # Zones are tiled over the shrunk single volumetry INSIDE synthesis (so the
    # 3-D plot and this model share one geometry); no post-hoc clipping here.
    zones = synthesize_space_geometry(row, simplify_tol=simplify_tol,
                                      cat_shrink=cat_shrink,
                                      buffer_distance=buffer_distance)
    if not zones:
        raise ValueError(f"no space geometry could be synthesized for {building_reference}")

    # Local origin = building footprint centroid. EnergyPlus shading math loses
    # ~0.5 m precision at UTM magnitudes (Y≈4.6e6 in float), so all geometry is
    # shifted to be local; UTM is north-up, so orientation is preserved.
    from shapely import affinity
    _fp = _footprints_by_floor(row)
    if _fp:
        _base = _fp.get(0) or next(iter(_fp.values()))
        ox, oy = _base.centroid.x, _base.centroid.y
    else:
        ox, oy = 0.0, 0.0

    idf = _seed_idf(IDF)

    # A taller ground floor applies to buildings with >= 5 above-ground floors OR to
    # industrial/agricultural buildings (machinery/vehicle clearance); other smaller
    # buildings use the regular ``floor_height`` for the ground storey too.
    try:
        _n_floors_ag = int(row.get("br__floors_above_ground"))
    except (TypeError, ValueError):
        _n_floors_ag = len([f for f in _fp if f >= 0]) if _fp else 1
    eff_gfh = utils.effective_ground_floor_height(
        _n_floors_ag, ground_floor_height, floor_height, row.get("building_use"))

    # The zones were already tiled over the shrunk footprint in synthesis; here
    # we only need the leftover setback (gross − walled) per floor, emitted as
    # self-shading, from the SAME shared volumetry.
    cat_remainders = {}
    if cat_shrink:
        try:
            _w, _g, cat_remainders, _dy = _walled_footprint_by_floor(row, buffer_distance)
        except Exception as _e:
            sys.stderr.write(f"  cat-shrink: skipped ({_e})\n")

    # One EnergyPlus zone per (space, floor) prism; the thermal-zone identity
    # (``br__thermal_zones`` reference/zone_id) is kept in the name so a
    # multi-floor dwelling → stacked zones sharing that identity with inter-floor
    # adjacency. Single-zone-per-dwelling merging is a later refinement.
    # One prism per (space, floor, piece); pools excluded. A zone's floor polygon
    # can be MULTIPART (a strip crossing a bent footprint) → one prism per piece so
    # the full area is covered (a single ``_clean`` would drop all but the largest).
    prisms, zone_meta = _zones_to_prisms(zones, ox, oy, eff_gfh, floor_height)

    # Build zones + surfaces from the non-manifold CELL COMPLEX (topology.py):
    # (``zone_meta`` — {ZONE_NAME: space record} — is also attached to the returned
    # IDF as ``_hc_zone_meta`` so scenario tooling can map zones → use type.)
    # spaces are cells, shared walls/slabs are faces referenced once by their two
    # cells — so interior adjacency is exact by construction. Falls back to the
    # legacy pairwise ``_build_surfaces`` if the complex can't be built.
    try:
        _build_surfaces_topological(idf, prisms)
    except Exception as _e:
        sys.stderr.write(f"  cell-complex surfaces failed ({_e}); using legacy\n")
        _build_surfaces(idf, prisms)
    # Refine exterior boundary conditions from the inferred wall typing. Party
    # walls become windowless Outdoors façades (their names are returned so
    # windows/doors skip them).
    party_walls, main_facade_walls = _refine_exterior_bcs(
        idf, row, zone_meta, floor_height, eff_gfh, origin=(ox, oy))
    # Neighbour shading obstructions (same as the 3-D plot's shade panels).
    n_shade = _add_shading(idf, shadows, ox, oy, floor_height, eff_gfh)
    # Footprint area not covered by the (shrunk) walled zones → self-shading.
    if cat_remainders:
        _add_cat_shading(idf, cat_remainders, ox, oy, eff_gfh, floor_height)

    # Resolve the per-building window-to-wall ratio (constant default, or a
    # predictor via ``wwr_model``).
    _model = _load_wwr_model(wwr_model)
    eff_wwr = building_wwr(row, model=_model if _model is not None else wwr,
                           default=_DEFAULT_WWR)

    # Entrance doors FIRST so windows can be placed clear of them: a main
    # pedestrian door at the façade nearest the address, and a vehicle door on
    # each parking zone. Anchor in the same local frame as the geometry.
    try:
        addr = _address_xy(row)
        anchor = (addr[0] - ox, addr[1] - oy) if addr else (0.0, 0.0)
        _add_entry_doors(idf, anchor, zone_meta, skip_walls=party_walls)
    except Exception as _e:
        sys.stderr.write(f"  doors: could not place entrance doors ({_e})\n")

    # Realistic windows: equally-spaced square panes sized to meet the WWR
    # (~one per ``area_per_window`` m² of glazing), skipping parking façades,
    # party walls and door positions.
    try:
        _add_windows(idf, eff_wwr, zone_meta, skip_walls=party_walls,
                     main_facade_walls=main_facade_walls)
    except Exception as _e:
        sys.stderr.write(f"  windows: could not place windows ({_e})\n")

    # Residential main-façade BALCONIES (floors ≥1, sized by construction period)
    # and window AWNINGS on E/S/W openings (per income + conservation state) — the
    # balconies also raise their wall's windows to the configured door height.
    try:
        _add_facade_shading(idf, row, zone_meta, main_facade_walls)
    except Exception as _e:
        sys.stderr.write(f"  facade shading: {_e}\n")

    # Reveal/recess shades around every window and door perimeter (inner-aligned
    # Spanish glazing → the wall thickness shades the opening).
    try:
        _add_subsurface_reveals(idf, _WIN_REVEAL)
    except Exception as _e:
        sys.stderr.write(f"  reveals: could not add window/door reveals ({_e})\n")

    if not geometry_only:
        # Constructions + schedules + loads + ideal-loads HVAC + sim control.
        try:
            _elev = float(row.get("elevation"))
            if not math.isfinite(_elev):
                _elev = 200.0
        except (TypeError, ValueError):
            _elev = 200.0
        # Resolve the Spanish archetype (thermal-code period × dominant use) →
        # envelope U-values driving the construction insulation thicknesses.
        dom_use = _dominant_use_type(zones)
        arch = building_archetype(row.get("year_of_construction"), dom_use)

        # Attach the ERA5-Land EPW so idf.run() uses real local weather. The
        # EPW is cached per (ERA5-Land cell, year); coastal centroids resolve to
        # the nearest cell with data. Explicit epw_path overrides resolution.
        #
        # BEFORE add_energy_systems, not after: the occupant engine and the
        # persiana control derive their SEASONS from the weather file (a day is
        # heating / cooling / in-between by its smoothed mean outdoor temperature),
        # falling back to a fixed month split when no EPW is attached. Building the
        # systems first meant the model was wired on the month fallback and, since
        # the schedule writers no-op on a name that already exists, that fallback
        # then survived every scenario re-wiring.
        if epw_path is None and weather_year is not None:
            try:
                from hypercadaster_ES import weather as _weather
                _e5 = era5_dir or (os.path.join(wd, "weather", "era5land") if wd else None)
                _ed = epw_dir or (os.path.join(wd, "weather", "epw") if wd else None)
                lon, lat = _building_lonlat(row, crs=getattr(gdf, "crs", None) or "EPSG:25831")
                if _e5 and _ed and lon is not None:
                    epw_path = _weather.municipality_epw(
                        _e5, _ed, lat, lon, int(weather_year),
                        elevation=_elev, update=False)
            except Exception as _e:
                sys.stderr.write(f"  weather: could not attach EPW ({_e})\n")
        if epw_path:
            idf.epw = epw_path

        # Constructions + schedules + loads + ideal-loads HVAC + the occupant
        # model + sim control — now with the weather attached, so the seasons are
        # the ones the KPIs will report.
        add_energy_systems(idf, zone_meta, elevation=_elev,
                           envelope=arch["envelope"], archetype=arch,
                           output_frequencies=output_frequencies, row=row,
                           residential_behavior=residential_behavior,
                           residential_period=residential_period,
                           behavior_seed=behavior_seed, wd=wd)

    # Neighbour-shading filter + cap, BEFORE the model is written or returned, so
    # the .idf on disk, the 3-D plot and the simulated context are the same panels.
    # Runs after every surface exists (the filter tests panels against the
    # building's own surfaces) and after the façade shading, which it must not
    # touch — it only ever removes the ``Shade_F*`` neighbour panels.
    _apply_shading_filter(idf, filter_shading=filter_shading,
                          max_shading=max_shading,
                          label=building_reference, verbose=False)

    # Default IDF location: {wd}/energyplus/{building_reference}.idf
    if output_path is None and wd is not None:
        out_dir = os.path.join(wd, "energyplus")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{building_reference}.idf")
    if output_path:
        if not output_path.lower().endswith(".idf"):
            output_path += ".idf"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        idf.saveas(output_path)
        # …and the metadata the file itself cannot carry, so the model can be
        # reloaded (or hand-edited and reloaded) and still simulated.
        _write_idf_metadata(output_path, zone_meta,
                            building_reference=building_reference,
                            shading_filtered=True,
                            epw=getattr(idf, "epw", None) or None)
    try:
        idf._hc_zone_meta = zone_meta          # zones → space record (for scenarios)
    except Exception:
        pass
    return idf


def _safe(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:24]


def _address_xy(row):
    """(x, y) of the building's address point in the source CRS (EPSG:25831),
    used to anchor entrance doors. ``postal_addresses_location`` is a list of
    Points; falls back to the footprint centroid. Returns None if nothing usable."""
    addr = row.get("postal_addresses_location")
    pt = None
    if addr is not None:
        seq = list(addr) if hasattr(addr, "__iter__") and not hasattr(addr, "geom_type") else [addr]
        for g in seq:
            if g is not None and not getattr(g, "is_empty", True):
                pt = g
                break
    if pt is None:
        fp = _footprints_by_floor(row)
        if fp:
            base = fp.get(0) or next(iter(fp.values()))
            pt = base.centroid
    if pt is None:
        return None
    try:
        return float(pt.x), float(pt.y)
    except Exception:
        return None


def _building_lonlat(row, crs="EPSG:25831"):
    """(lon, lat) WGS84 of a building row's geometry, for weather-cell lookup.
    ``crs`` should be the source frame's CRS (cadastral data is EPSG:25831 /
    ETRS89 UTM 31N in Catalonia). Returns (None, None) if no usable geometry."""
    geom = row.get("building_geometry")
    if geom is None or getattr(geom, "is_empty", True):
        for c in ("geometry", "br__building_footprint_geometry"):
            geom = row.get(c)
            if geom is not None and not getattr(geom, "is_empty", True):
                break
    if geom is None:
        return None, None
    try:
        import geopandas as gpd
        pt = gpd.GeoSeries([geom.centroid], crs=crs).to_crs(4326).iloc[0]
        return float(pt.x), float(pt.y)
    except Exception:
        return None, None


def _dominant_use_type(zones):
    """Dominant occupied use type across spaces, weighted by floor area —
    drives the building archetype's use category."""
    def _tally(only_occupied):
        by_use = {}
        for s in zones:
            kind = s.get("kind")
            if only_occupied:
                if kind and kind != "occupied":
                    continue
            elif kind == "communal":
                continue
            ut = s.get("use_type")
            area = 0.0
            for poly in (s.get("floor_polys") or {}).values():
                try:
                    area += float(poly.area)
                except Exception:
                    pass
            by_use[ut] = by_use.get(ut, 0.0) + area
        return by_use

    # A building with no occupied zones at all (a barn, a standalone garage) is
    # still characterised by the use it does have — falling through to None here
    # would resolve it to the generic "other" archetype instead.
    by_use = _tally(only_occupied=True) or _tally(only_occupied=False)
    if not by_use:
        return None
    return max(by_use, key=by_use.get)


def _typed_wall_segments(row):
    """{floor: list[(LineString, type)]} from ``br__wall_segments`` (the full
    per-floor wall description: geometry + type in {main_facade, party_wall,
    adiabatic, terrain_contact})."""
    segs = row.get("br__wall_segments")
    if not isinstance(segs, (list, tuple)):
        return {}
    out = {}
    for s in segs:
        if not isinstance(s, dict):
            continue
        g = _as_geom(s.get("geometry"))
        t = s.get("type")
        if g is None or getattr(g, "is_empty", True) or not t:
            continue
        try:
            fl = int(s.get("floor", 0))
        except (TypeError, ValueError):
            fl = 0
        out.setdefault(fl, []).append((g, t))
    return out


def _refine_exterior_bcs(idf, row, zone_meta, floor_height, ground_floor_height,
                         origin=(0.0, 0.0)):
    """Refine exterior wall boundary conditions from the typed
    ``br__wall_segments`` (matched per floor by proximity to each wall's base):
      * ``adiabatic`` (abuts a taller/equal neighbour) → Adiabatic, no heat
        transfer, no windows;
      * ``party_wall`` (shared with an adjacent building at a similar interior
        temperature) → Adiabatic, no heat transfer, window/door-free (returned so
        windows/doors skip it); gets the HC_PartyWall construction via the
        Adiabatic BC mapping;
      * ``terrain_contact`` → Ground;
      * ``main_facade`` (or unmatched) → left Outdoors (gets windows).
    ``origin`` (ox, oy) matches the local-origin shift of the surfaces. Returns
    ``(party_wall_names, main_facade_names)`` — the latter are the street-facing
    exterior walls (used for balconies + commercial glazing)."""
    from shapely import affinity
    from shapely.geometry import LineString
    party_names = set()
    main_facade_names = set()
    typed_raw = _typed_wall_segments(row)
    if not typed_raw:
        return party_names
    ox, oy = origin
    # Buffer each typed segment for hit-testing, in the surfaces' local origin.
    typed = {}
    for fl, items in typed_raw.items():
        typed[fl] = [(affinity.translate(g, xoff=-ox, yoff=-oy).buffer(0.6), t)
                     for g, t in items]
    for srf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if srf.Surface_Type != "Wall" or srf.Outside_Boundary_Condition.lower() != "outdoors":
            continue
        wb = _wall_base(srf)          # geomeppy reorders verts → find base edge
        if wb is None:
            continue
        (x0, y0), (x1, y1), z0, ztop = wb
        zmid = (z0 + ztop) / 2.0
        fl = _floor_at_z(zmid, ground_floor_height, floor_height)
        cands = typed.get(fl) or []
        if not cands:
            continue
        wall_line = LineString([(x0, y0), (x1, y1)])
        wl = wall_line.length or 1.0
        # Sum overlap per type — a single wall can span many short segments, so
        # accumulate and take the dominant type (avoids missing walls that are
        # covered by several sub-segments each shorter than the wall).
        overlaps = {}
        for buf, t in cands:
            if wall_line.intersects(buf):
                overlaps[t] = overlaps.get(t, 0.0) + wall_line.intersection(buf).length
        best_t = max(overlaps, key=overlaps.get) if overlaps else None
        if best_t is not None and overlaps[best_t] < 0.3 * wl:
            best_t = None                             # no dominant type
        if best_t in (None, "main_facade"):
            if best_t == "main_facade":
                main_facade_names.add(srf.Name)       # street-facing → balconies
            continue                                  # stays Outdoors + windows
        if best_t == "party_wall":
            srf.Outside_Boundary_Condition = "Adiabatic"   # neighbour at similar T
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
            srf.Construction_Name = "HC_PartyWall"    # distinct from adiabatic below
            party_names.add(srf.Name)                 # no windows/doors
        elif best_t == "adiabatic":
            srf.Outside_Boundary_Condition = "Adiabatic"
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
            srf.Construction_Name = "HC_IntWall"      # internal partition (not party)
        elif best_t == "terrain_contact":
            srf.Outside_Boundary_Condition = "Ground"
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
    return party_names, main_facade_names


def _add_shading(idf, shadows, ox, oy, floor_height, ground_floor_height, eps=0.05):
    """Add neighbour-obstruction shading surfaces from ``br__shadows`` — the
    SAME per-floor shadow boundary the 3-D plot draws. Each boundary segment
    (excluding the analysis buffer-circle, detected via the radial gap, exactly
    as the plot does) becomes a vertical ``Shading:Building:Detailed`` panel
    spanning its floor, in the model's local origin (``ox, oy``)."""
    if not isinstance(shadows, dict) or not shadows:
        return 0
    radii = []
    for poly in shadows.values():
        if poly is None or getattr(poly, "is_empty", True):
            continue
        for g in (list(poly.geoms) if hasattr(poly, "geoms") else [poly]):
            if hasattr(g, "exterior"):
                for x, y, *_ in g.exterior.coords:
                    radii.append(math.hypot(x - ox, y - oy))
    if not radii:
        return 0
    R = max(radii)
    cut = R * 0.98
    outer = sorted(r for r in radii if r > 0.85 * R)
    best_gap, best_at = 0.0, None
    for i in range(len(outer) - 1):
        gp = outer[i + 1] - outer[i]
        if gp > best_gap:
            best_gap, best_at = gp, outer[i]
    if best_at is not None and best_gap >= 3.0:
        cut = best_at + 1e-6

    n = 0
    for floor, poly in sorted(shadows.items()):
        if poly is None or getattr(poly, "is_empty", True):
            continue
        try:
            fl = int(floor)
        except (TypeError, ValueError):
            continue
        z0, z1 = _floor_z(fl, ground_floor_height, floor_height)
        for g in (list(poly.geoms) if hasattr(poly, "geoms") else [poly]):
            if not hasattr(g, "exterior"):
                continue
            ring = list(g.exterior.coords)
            for c in range(len(ring) - 1):
                x0, y0 = ring[c][:2]
                x1, y1 = ring[c + 1][:2]
                if (math.hypot(x0 - ox, y0 - oy) > cut
                        or math.hypot(x1 - ox, y1 - oy) > cut):
                    continue  # buffer-circle artefact (not a real obstruction)
                if math.hypot(x1 - x0, y1 - y0) < eps:
                    continue
                o = idf.newidfobject("SHADING:BUILDING:DETAILED",
                                     Name=f"Shade_F{fl}_{n}")
                o.setcoords([(x0 - ox, y0 - oy, z0), (x1 - ox, y1 - oy, z0),
                             (x1 - ox, y1 - oy, z1), (x0 - ox, y0 - oy, z1)])
                n += 1
    return n


def _floor_at_z(z, ground_floor_height, floor_height):
    if z < 0:
        return int(math.floor(z / floor_height))
    if z < ground_floor_height:
        return 0
    return 1 + int((z - ground_floor_height) // floor_height)


# --------------------------------------------------------------------------- #
# CAT-area shrink: walls follow the CAT built area, the rest becomes shading
# --------------------------------------------------------------------------- #
def _cat_area_by_floor(row):
    """Target walled area per floor (m²) = building-spaces area of that floor PLUS
    its communal share, i.e. the sum of ``area_with_communals`` per floor, summed
    over uses, EXCLUDING open-air pools (their ``Sports facilities`` area is
    OtherConstruction, not walled). This is the restriction the CAT shrink erodes
    the footprint toward.

    Prefers the direct per-floor total ``br__area_with_communals_by_floor``;
    falls back to reconstructing it from
    ``br__mean_building_space_area_with_communals_by_floor`` × space count
    (``br__building_spaces_by_floor``) when the direct column is absent."""
    drop_sports = len(_pool_geoms(row)) > 0
    out = {}

    direct = row.get("br__area_with_communals_by_floor")
    if isinstance(direct, dict) and direct:
        for use, by_fl in direct.items():
            if drop_sports and _use_category(use) == "sports":
                continue
            if not isinstance(by_fl, dict):
                continue
            for fl, a in by_fl.items():
                try:
                    out[int(fl)] = out.get(int(fl), 0.0) + float(a)
                except (TypeError, ValueError):
                    continue
        return out

    mean = row.get("br__mean_building_space_area_with_communals_by_floor")
    if not isinstance(mean, dict):
        return {}
    counts = row.get("br__building_spaces_by_floor")
    for use, by_fl in mean.items():
        if drop_sports and _use_category(use) == "sports":
            continue
        if not isinstance(by_fl, dict):
            continue
        cnt_use = counts.get(use, {}) if isinstance(counts, dict) else {}
        for fl, m in by_fl.items():
            try:
                c = float(cnt_use.get(fl, 1) or 1) if isinstance(cnt_use, dict) else 1.0
                out[int(fl)] = out.get(int(fl), 0.0) + float(m) * c
            except (TypeError, ValueError):
                continue
    return out


def _fixed_wall_union_by_floor(row):
    """{floor: union of adiabatic + party_wall segment geometries} from the typed
    ``br__wall_segments`` — the walls abutting/facing a neighbour, which must stay
    ANCHORED to the footprint boundary (never moved by the CAT shrink)."""
    typed = _typed_wall_segments(row)
    out = {}
    for fl, items in typed.items():
        geoms = [g for g, t in items if t in ("party_wall", "adiabatic")]
        if geoms:
            out[fl] = unary_union(geoms)
    return out


def _walled_outline(footprint, d, fixed_union=None):
    """Footprint eroded inward by ``d`` (mitre join), but with the band along the
    ``fixed_union`` edges (adiabatic/party walls) restored to the footprint
    boundary so those walls don't move. Returns a cleaned polygon (or None)."""
    eroded = _clean(footprint.buffer(-d, join_style=2))
    if eroded is None or eroded.is_empty:
        return None
    if fixed_union is not None and not fixed_union.is_empty:
        band = footprint.difference(eroded)              # the d-wide perimeter ring
        restore = band.intersection(fixed_union.buffer(d + 0.5))
        if not restore.is_empty:
            eroded = _clean(unary_union([eroded, restore]))
    if eroded is None:
        return None
    return _clean(eroded.simplify(0.05, preserve_topology=True))


def _solve_erosion(poly, target_area, cap, fixed_union=None, tol=0.5, iters=40):
    """Smallest inward buffer distance d∈[0, cap] (m) whose anchored walled
    outline (``_walled_outline``) has area ≈ ``target_area``. Walled area is
    monotone-decreasing in d, so binary search. Returns ``cap`` if even the max
    erosion cannot reach the target."""
    if cap <= 0:
        return 0.0

    def _area(d):
        w = _walled_outline(poly, d, fixed_union)
        return w.area if w is not None else 0.0

    if _area(cap) > target_area:
        return cap
    lo, hi = 0.0, cap
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        a = _area(mid)
        if abs(a - target_area) < tol:
            return mid
        if a > target_area:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _walled_footprint_by_floor(row, buffer_distance):
    """The SINGLE shrunk volumetry shared by the 3-D plot and the EnergyPlus
    model. Aligns the walled area to the CAT built area per floor with a NEGATIVE
    BUFFER (inward wall offset): adiabatic/party/terrain walls (from
    ``br__wall_segments``) and patio rings stay ANCHORED to the footprint
    boundary; only the free façades (main_facade / patio) move inward. When the
    building has any such fixed edge the block is NOT shifted; otherwise the
    freestanding block is shifted north by min(erosion)/2 (≤ erosion, so it stays
    inside the footprint), uniformly across floors so they stay aligned.

    Returns ``(walled_by_floor, gross_by_floor, remainders, dy)``:
    ``walled_by_floor[fl]`` the shrunk+shifted footprint (only for floors that
    were actually eroded), ``gross_by_floor[fl]`` the rationalized gross
    footprint, ``remainders[fl]`` the setback shade (gross − walled, pools
    removed), ``dy`` the north shift. Independent of the synthesized zones so
    every consumer converges on the same geometry."""
    from shapely import affinity
    from shapely.geometry import LineString
    fp = _rationalize_floors(_footprints_by_floor(row))
    cat = _cat_area_by_floor(row)
    pools = _pool_geoms(row)
    pool_u = unary_union(pools) if pools else None
    fixed_by_floor = _fixed_wall_union_by_floor(row)      # party + adiabatic

    def _floor_fixed(fl, gross):
        """Anchored boundaries for a floor: party/adiabatic walls PLUS the
        footprint's interior rings (patios) — a negative buffer would otherwise
        GROW the patios inward, wiping out the surrounding thermal zones."""
        parts = []
        fx = fixed_by_floor.get(fl)
        if fx is not None and not fx.is_empty:
            parts.append(fx)
        for r in gross.interiors:                        # patio voids
            parts.append(LineString(r.coords))
        return unary_union(parts) if parts else None

    # Pass 1 — per-floor anchored erosion outline (None where CAT not lower).
    plan = {}                                            # fl -> (gross, walled|None, d)
    any_fixed = False
    for fl in sorted(fp):
        gross = fp.get(fl)
        if gross is None or gross.is_empty:
            continue
        ca = cat.get(fl)
        fx = _floor_fixed(fl, gross)
        if fx is not None:
            any_fixed = True
        if not ca or ca <= 0 or ca >= gross.area:
            plan[fl] = (gross, None, 0.0)                # CAT not lower → no shrink
            continue
        d = _solve_erosion(gross, ca, buffer_distance, fixed_union=fx)
        if d < 0.05:                                     # negligible → no shrink
            plan[fl] = (gross, None, 0.0)
            continue
        walled = _walled_outline(gross, d, fixed_union=fx)
        plan[fl] = (gross, walled, d if (walled and not walled.is_empty) else 0.0)

    gross_by_floor = {fl: g for fl, (g, _w, _d) in plan.items()}
    if not plan:
        return {}, gross_by_floor, {}, 0.0
    # Freestanding buildings (no anchored wall/patio): shift north to leave a
    # southern setback; min(d)/2 keeps the shift inside every floor's erosion
    # headroom and uniform across floors. Anchored buildings are not shifted.
    eroded = [d for (_g, w, d) in plan.values() if w is not None and d > 0]
    dy = 0.0 if (any_fixed or not eroded) else min(eroded) / 2.0

    walled_by_floor, remainders = {}, {}
    for fl, (gross, walled, _d) in plan.items():
        if walled is None or walled.is_empty:
            continue                                     # no erosion on this floor
        walled_sh = affinity.translate(walled, yoff=dy) if dy else walled
        walled_by_floor[fl] = walled_sh
        # Shades use the FULL footprint minus the (shifted) walled area.
        rem = _clean(gross.difference(walled_sh.buffer(0.01)))
        if rem is not None and pool_u is not None:
            rem = _clean(rem.difference(pool_u))
        if rem is not None and not rem.is_empty and rem.area > 0.5:
            remainders[fl] = rem
    return walled_by_floor, gross_by_floor, remainders, dy


def _lines_from_boundary(geom):
    """Yield LineStrings from a polygon boundary or a difference thereof
    (handles LinearRing, LineString, and Multi* / GeometryCollection)."""
    from shapely.geometry import LineString
    if geom is None or getattr(geom, "is_empty", True):
        return
    gt = geom.geom_type
    if gt in ("LineString", "LinearRing"):
        yield LineString(geom.coords)
    elif gt in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            yield from _lines_from_boundary(g)


def shrunk_wall_segments(row, buffer_distance=None, cat_shrink=None):
    """``br__wall_segments`` transformed to the shrunk single volumetry used by
    both the 3-D plot (``utils.plot_building_3d_geometry`` Layer 3) and the
    EnergyPlus model, so the plotted envelope coincides with the thermal zones.

    Party/adiabatic/terrain_contact walls stay ANCHORED (unchanged); the free
    main_facade/patio walls are replaced by the boundary of the shrunk (walled)
    footprint where it does not run along an anchored wall. Floors that are not
    eroded keep their original segments. Returns the same record shape as
    ``br__wall_segments`` ({geometry, type, floor}); returns the input unchanged
    when shrinking is disabled or unavailable."""
    segs_in = row.get("br__wall_segments")
    if not isinstance(segs_in, (list, tuple)) or not segs_in:
        return segs_in
    do_shrink = _CAT_AREA_SHRINK if cat_shrink is None else cat_shrink
    bd = _BUFFER_DISTANCE if buffer_distance is None else buffer_distance
    if not do_shrink:
        return list(segs_in)
    try:
        walled_by_floor, _gross, _rem, _dy = _walled_footprint_by_floor(row, bd)
    except Exception:
        return list(segs_in)
    if not walled_by_floor:
        return list(segs_in)
    typed = _typed_wall_segments(row)                     # {fl: [(LineString, type)]}
    anchored_types = ("party_wall", "adiabatic", "terrain_contact")
    out = []
    seen_floors = set()
    for fl, items in typed.items():
        seen_floors.add(fl)
        walled = walled_by_floor.get(fl)
        anchored = [g for g, t in items if t in anchored_types]
        for g, t in items:
            if t in anchored_types:
                out.append({"geometry": g, "type": t, "floor": fl})
        if walled is None or walled.is_empty:
            for g, t in items:                            # floor not eroded → keep free walls
                if t not in anchored_types:
                    out.append({"geometry": g, "type": t, "floor": fl})
            continue
        bnd = walled.boundary
        fx = unary_union(anchored) if anchored else None
        if fx is not None and not fx.is_empty:
            bnd = bnd.difference(fx.buffer(0.5))
        for ls in _lines_from_boundary(bnd):
            if ls.length > 0.1:
                out.append({"geometry": ls, "type": "main_facade", "floor": fl})
    # Eroded floors with no typed segments → the whole outline is free façade.
    for fl, walled in walled_by_floor.items():
        if fl in seen_floors or walled is None or walled.is_empty:
            continue
        for ls in _lines_from_boundary(walled.boundary):
            if ls.length > 0.1:
                out.append({"geometry": ls, "type": "main_facade", "floor": fl})
    return out


def _add_cat_shading(idf, remainders, ox, oy, ground_floor_height, floor_height):
    """Emit the footprint-minus-walled remainder of each floor as a horizontal
    overhang at the TOP of that floor (z1) — it represents the slab/balcony of
    the floor above projecting over the setback, which shades that floor's
    windows. Placing it at the floor top (not the bottom slab) avoids a
    meaningless shade lying on the ground at z=0."""
    from shapely import affinity
    n = 0
    for fl, rem in sorted(remainders.items()):
        _z0, z1 = _floor_z(fl, ground_floor_height, floor_height)
        rem_local = affinity.translate(rem, xoff=-ox, yoff=-oy)
        for ring in _horizontal_rings(rem_local):
            o = idf.newidfobject("SHADING:BUILDING:DETAILED",
                                 Name=f"CatShade_F{fl}_{n}")
            o.setcoords([(x, y, z1) for (x, y) in ring])
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Entrance doors (main pedestrian + parking vehicle); sizes from config
# (``_MAIN_DOOR_SIZE`` / ``_PARKING_DOOR_SIZE``).
# --------------------------------------------------------------------------- #
def _zone_is_parking(zone_meta, zname):
    """True for zones that get a vehicle door instead of a pedestrian one:
    parking/warehouse, and agricultural sheds (which are entered by vehicle)."""
    s = zone_meta.get(zname.upper())
    return bool(s) and _use_category(s.get("use_type")) in ("parking", "agricultural")


def _wall_base(wall):
    """(P0, P1, z0, z_top) of a vertical wall's ground edge from its coords."""
    coords = [(float(x), float(y), float(z)) for x, y, z in wall.coords]
    z0 = min(c[2] for c in coords)
    ztop = max(c[2] for c in coords)
    base = [c for c in coords if abs(c[2] - z0) < 1e-6]
    if len(base) < 2:
        return None
    return (base[0][0], base[0][1]), (base[1][0], base[1][1]), z0, ztop


def _wall_outward_normal(wall, ref=None):
    """Unit OUTWARD horizontal normal (nx, ny) of a vertical wall and its outward
    BEARING (deg, 0=N/90=E).

    The direction comes from the surface WINDING, which is authoritative: an
    EnergyPlus exterior surface is ordered counterclockwise seen from outside, so
    the vertex cross product already points outdoors, and that is the same
    convention the windows on the wall are built with.

    ``ref=(x, y)`` is only a FALLBACK, used when the winding cannot decide (a
    degenerate or non-planar polygon): the normal is then forced to point away
    from that reference point. It used to be applied unconditionally, on the
    assumption that the winding was unreliable — but flipping against a zone
    centroid is what is actually unreliable, because the centroid of an L-shaped
    or wrapped-around zone can sit on the far side of its own wall. Measured on
    the reference building, the centroid rule disagreed with the windows' own
    outward normals on 46 of 49 walls, which put the balconies, awnings and
    exterior blinds INSIDE the dwellings, where they shade nothing.
    """
    c = [(float(x), float(y), float(z)) for x, y, z in wall.coords]
    if len(c) < 3:
        return None
    # Newell's method: robust for slightly non-planar polygons, where a single
    # cross product of the first two edges can pick up a spurious direction.
    nx = ny = nz = 0.0
    for i, (x0, y0, z0) in enumerate(c):
        x1, y1, z1 = c[(i + 1) % len(c)]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    h = math.hypot(nx, ny)
    if h < 1e-9:
        return None                       # horizontal surface: no façade direction
    nx, ny = nx / h, ny / h
    if ref is not None and _degenerate_winding(c):
        mx = sum(p[0] for p in c) / len(c)
        my = sum(p[1] for p in c) / len(c)
        if nx * (mx - ref[0]) + ny * (my - ref[1]) < 0:
            nx, ny = -nx, -ny
    return nx, ny, math.degrees(math.atan2(nx, ny)) % 360.0


def _degenerate_winding(coords, tol=1e-6):
    """Whether a polygon's winding cannot be trusted to give an outward normal —
    it is collapsed to a line or has (near-)zero area."""
    n = len(coords)
    if n < 3:
        return True
    ax = ay = az = 0.0
    for i, (x0, y0, z0) in enumerate(coords):
        x1, y1, z1 = coords[(i + 1) % n]
        ax += y0 * z1 - z0 * y1
        ay += z0 * x1 - x0 * z1
        az += x0 * y1 - y0 * x1
    return math.sqrt(ax * ax + ay * ay + az * az) / 2.0 < tol


def _building_income(row):
    """Mean net income (€/yr) for the building's area from the social_ES layer
    (``so__*income*`` columns), or None when unavailable."""
    for c in (list(row.index) if hasattr(row, "index") else []):
        cl = str(c).lower()
        if cl.startswith("so__") and ("income" in cl or "renta" in cl):
            try:
                v = float(row.get(c))
                if math.isfinite(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return None


def _income_setpoint_offset(income, cfg=None):
    """Heating/cooling setpoint offsets (°C) driven by household income, per the
    RETABIT sensitivity finding (Mont Lecocq et al. 2025, §5/Fig. 14: lower income →
    lower heating setpoint / higher cooling setpoint). Linear ramp — the full
    ``heating_offset_max`` / ``cooling_offset_max`` at/below ``income_floor``, zero
    at/above ``reference_income``. Returns ``(heat_off, cool_off)``; ``(0.0, 0.0)``
    when disabled. A ``None`` income falls back to
    ``residential_behavior.default_income``."""
    cfg = _INCOME_SETPOINTS if cfg is None else cfg
    if not cfg or not cfg.get("enabled", False):
        return (0.0, 0.0)
    ref = float(cfg.get("reference_income", 30000))
    floor = float(cfg.get("income_floor", 12000))
    hmax = float(cfg.get("heating_offset_max", 0.0))
    cmax = float(cfg.get("cooling_offset_max", 0.0))
    if not (isinstance(income, (int, float)) and math.isfinite(income)):
        rb = globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {}
        income = float(rb.get("default_income", 22000))
    income = float(income)
    if ref <= floor:                        # degenerate config → step, no ramp
        frac = 0.0 if income >= ref else 1.0
    else:                                   # 0 at/above ref, 1 at/below floor
        frac = max(0.0, min(1.0, (ref - income) / (ref - floor)))
    return (round(hmax * frac, 2) or 0.0, round(cmax * frac, 2) or 0.0)


def _building_conservation(row):
    """Dominant dwelling conservation status from the Essential-Characteristics
    conservation proportions (see :func:`_conservation_proportion`), or None when
    unavailable."""
    best, bestp = None, 0.0
    for status in _CONSERVATION_INFIL_FACTOR:
        p = _conservation_proportion(row, status)
        if p is not None and p > bestp:
            best, bestp = status, p
    return best


def _add_facade_shading(idf, row, zone_meta, main_facade_walls):
    """Residential MAIN-FAÇADE balconies (floors ≥ 1) + window AWNINGS (reqs #3/#4).

    The exterior persiana is NOT built here — it is a ``WindowMaterial:Shade``
    layer driven by a ``WindowShadingControl`` (``_add_persiana_controls``).

    Balconies: a horizontal slab at each upper residential main-façade wall's floor
    level, its projection depth and along-façade length growing with the
    construction period (old → short/shallow, modern → deep/full-width); the wall's
    windows are raised to the configured balcony-door height (2 m). Awnings: a small
    overhang (+ outer drop) over each E/SE/S/SW/W window WITHOUT a balcony, adopted
    with a probability driven by area income and dwelling conservation state — drawn
    ONCE PER FAÇADE COLUMN (``shading.awning.adoption_scope``), so the stack of
    windows at one façade XY is awninged consistently on every floor instead of
    flickering storey by storey; on a balcony wall the awning sits at the balcony's
    outer edge instead. All are
    ``Shading:Building:Detailed`` panels. Config: ``shading`` in energy_config.yaml.
    """
    cfg = _SHADING or {}
    bcfg = cfg.get("balcony", {}) or {}
    acfg = cfg.get("awning", {}) or {}
    period = _construction_period(row.get("year_of_construction"))
    bp = (bcfg.get("by_period", {}) or {}).get(period, {})
    depth = float(bp.get("depth", 0.0) or 0.0)
    length_fraction = float(bp.get("length_fraction", 0.0) or 0.0)
    win_h = float(bcfg.get("window_height", 2.0) or 2.0)
    balcony_on = bool(bcfg.get("enabled", True)) and depth > 0 and length_fraction > 0

    # Awning adoption = clamp(income / income_reference) × conservation weight.
    inc = _building_income(row) or float(acfg.get("default_income", 22000))
    cons = _building_conservation(row) or acfg.get("default_conservation", "normal")
    cw = (acfg.get("conservation_weight", {}) or {}).get(cons, 0.6)
    adoption = min(1.0, inc / float(acfg.get("income_reference", 30000) or 30000)) * float(cw)
    awning_on = bool(acfg.get("enabled", True))
    # WHAT the adoption draw is made on. "column" (default) draws once per façade
    # bay, so a stack of windows at the same façade XY gets the same answer on
    # every floor; "window" is the old per-window draw (visibly random up the
    # façade); "building" is all-or-nothing for the whole building.
    scope = str(acfg.get("adoption_scope", "column") or "column").lower()
    win_col = _window_columns(idf) if scope == "column" else {}
    bkey = str(row.get("building_reference") or getattr(idf, "idfname", "") or "b")

    def _awning_key(win_name):
        if scope == "building":
            return bkey
        return win_col.get(win_name, win_name)
    a_len = float(acfg.get("length", acfg.get("depth", 1.5)) or 1.5)
    # Awning slope, measured FROM THE VERTICAL: 90° is a horizontal canopy, 0° a
    # fabric hanging flat down the glazing. ``inclination_deg`` (the old key, an
    # angle below the HORIZONTAL) is still honoured so an existing config keeps
    # meaning what it did.
    if acfg.get("inclination_from_vertical_deg") is not None:
        a_from_vert = math.radians(float(acfg["inclination_from_vertical_deg"]))
    else:
        a_from_vert = math.radians(90.0 - float(acfg.get("inclination_deg", 30) or 30))
    a_hor = a_len * math.sin(a_from_vert)   # outward horizontal reach
    a_ver = a_len * math.cos(a_from_vert)   # vertical drop at the outer edge
    sun_hw = 67.5                       # N/NE/NW get no sun → no awning

    # Building reference (footprint convex hull) so a balcony is only placed where it
    # projects OUTSIDE the building — i.e. onto a street, not into a patio — and the
    # per-ZONE centroid to orient every outward normal robustly (street AND patio).
    from shapely.geometry import MultiPoint, Point
    zc = _zone_centroids(idf)
    bpts = []
    for w in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if w.Surface_Type.lower() == "wall":
            for x, y, _z in w.coords:
                bpts.append((float(x), float(y)))
    if not bpts:
        return 0, 0, 0
    bcx = sum(p[0] for p in bpts) / len(bpts)
    bcy = sum(p[1] for p in bpts) / len(bpts)
    try:
        hull = MultiPoint(bpts).convex_hull
    except Exception:
        hull = None
    # Top floor (highest balcony level) — its main-façade windows have no balcony
    # above, so they get a horizontal awning of the balcony's depth instead.
    bz = [_wall_base(w)[2] for w in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
          if w.Name in main_facade_walls and w.Surface_Type.lower() == "wall"
          and _wall_base(w) is not None]
    top_z0 = max(bz) if bz else None

    # Awnings + blinds are OPERABLE: they start CLEAR (fully transmitting → no
    # effect); the scenario occupant model deploys them per behaviour/season/time.
    clear = _sched_compact(idf, "SHADE_CLEAR", "Fraction",
                           ["For: AllDays", "Until: 24:00,1.0"])
    windows_by_wall = {}
    for w in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if str(getattr(w, "Surface_Type", "")).lower() == "window":
            windows_by_wall.setdefault(w.Building_Surface_Name, []).append(w)

    def _shade(name, coords, transmit=clear):
        o = idf.newidfobject("SHADING:BUILDING:DETAILED", Name=name)
        o.setcoords(coords)
        o.Transmittance_Schedule_Name = transmit
        return o

    n_balc = n_awn = 0
    # Balconies + awnings + blinds are RESIDENTIAL only (req #1). Iterate every
    # residential exterior wall (blinds go on all their windows; balconies only on
    # the street main façade; awnings on E/S/W windows without a balcony).
    for wall in list(idf.idfobjects["BUILDINGSURFACE:DETAILED"]):
        if (wall.Surface_Type.lower() != "wall"
                or wall.Outside_Boundary_Condition.lower() != "outdoors"):
            continue
        s = zone_meta.get(str(wall.Zone_Name).upper())
        if not s or _resolve_use_category(s) != "residential":
            continue
        wb = _wall_base(wall)
        # OUTWARD normal = away from the wall's OWN zone (robust for street & patio).
        # Oriented against the wall's own ZONE FLOOR (then the hull, then the
        # centroid): everything below hangs off this vector, so a flipped normal
        # builds the balcony, awning and blind inside the dwelling.
        nrm = _oriented_outward_normal(
            wall, hull=hull, ref=zc.get(zone_name := str(wall.Zone_Name), (bcx, bcy)))
        if wb is None or nrm is None:
            continue
        (x0, y0), (x1, y1), z0, ztop = wb
        nx, ny, bearing = nrm
        mx0, my0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        zone = wall.Zone_Name
        orient = _facade_orient(bearing)
        wins = windows_by_wall.get(wall.Name, [])
        # Balcony only on a STREET-facing main façade above the ground floor: it must
        # be a typed main-façade wall AND project OUTSIDE the footprint (not a patio).
        street = (hull is None or not hull.buffer(0.2).contains(
            Point(mx0 + nx * (depth + 0.3), my0 + ny * (depth + 0.3))))
        has_balcony = (wall.Name in main_facade_walls and balcony_on and z0 > 0.5
                       and street)

        if has_balcony:
            half = length_fraction * L / 2.0
            cxm, cym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            il = (cxm - ux * half, cym - uy * half)
            ir = (cxm + ux * half, cym + uy * half)
            ol = (il[0] + nx * depth, il[1] + ny * depth)
            orr = (ir[0] + nx * depth, ir[1] + ny * depth)
            _shade(f"Balcony_{wall.Name}",
                   [(il[0], il[1], z0), (ir[0], ir[1], z0),
                    (orr[0], orr[1], z0), (ol[0], ol[1], z0)], transmit="")  # opaque slab
            n_balc += 1
            for w in wins:                          # → 2 m balcony doors
                _reshape_window_height(w, ux, uy, z0, win_h)

        is_top = has_balcony and top_z0 is not None and abs(z0 - top_z0) < 0.1
        for w in wins:
            c = [(float(x), float(y), float(z)) for x, y, z in w.coords]
            zt = max(p[2] for p in c)
            zbt = min(p[2] for p in c)
            us = [(p[0] - x0) * ux + (p[1] - y0) * uy for p in c]
            a, b = min(us), max(us)
            la = (x0 + ux * a, y0 + uy * a); lb = (x0 + ux * b, y0 + uy * b)
            # The exterior persiana is NOT a panel pasted in front of the glazing:
            # a coincident surface only removes the beam its own shadow catches
            # and lets the sky diffuse straight through, so a fully closed one
            # still passed about half the solar. It is a WindowMaterial:Shade
            # layer driven by a WindowShadingControl instead
            # (``_add_persiana_controls``), which attenuates beam and diffuse
            # alike and enters the window heat balance.
            if not (awning_on and _sun_facing(bearing, sun_hw)):
                continue
            if is_top:
                # Top floor has no balcony above → a horizontal awning of the
                # balcony's projection depth over the window.
                oa = (la[0] + nx * depth, la[1] + ny * depth)
                ob = (lb[0] + nx * depth, lb[1] + ny * depth)
                _shade(f"Awn_{zone}_{n_awn}",
                       [(la[0], la[1], zt), (lb[0], lb[1], zt),
                        (ob[0], ob[1], zt), (oa[0], oa[1], zt)])
                n_awn += 1
            elif not has_balcony and _adopted(_awning_key(w.Name), adoption):
                # Retractable inclined awning (length a_len, slope a_incl) OUTWARD.
                oa = (la[0] + nx * a_hor, la[1] + ny * a_hor)
                ob = (lb[0] + nx * a_hor, lb[1] + ny * a_hor)
                _shade(f"Awn_{zone}_{n_awn}",
                       [(la[0], la[1], zt), (lb[0], lb[1], zt),
                        (ob[0], ob[1], zt - a_ver), (oa[0], oa[1], zt - a_ver)])
                n_awn += 1
    return n_balc, n_awn


def _facade_orient(bearing):
    """Coarse façade orientation from an outward bearing (deg, 0=N): E/S/W/N."""
    b = bearing % 360.0
    if 45.0 <= b < 135.0:
        return "E"
    if 135.0 <= b < 225.0:
        return "S"
    if 225.0 <= b < 315.0:
        return "W"
    return "N"


def _sun_facing(bearing, halfwidth=67.5):
    """True when a façade bearing (deg, 0=N) is NOT within ``halfwidth`` of North
    (i.e. faces E/SE/S/SW/W and so receives direct sun in the northern hemisphere)."""
    return min(bearing % 360.0, 360.0 - (bearing % 360.0)) > halfwidth


def _window_columns(idf, tol=0.4):
    """``{window_name: column_key}`` — group every fenestration surface into the
    FAÇADE COLUMN (vertical bay) it belongs to, i.e. by its PLAN position only,
    with the floor level ignored.

    Windows are named after their host wall (``{wall}_win{i}``) and every floor
    has its own walls, so hashing a window's NAME draws independently on each
    storey: the stack of windows sitting at one façade XY came out with awnings
    on some floors and not others, which reads as random on the 3-D model. A
    real façade does the opposite — an awning belongs to the bay, and a bay is
    the same opening repeated up the building — so the adoption draw is made
    once per column here and shared by every floor.

    Positions are clustered with a ``tol`` (m) snap rather than rounded to a
    grid: the stacked windows land on identical coordinates by construction, but
    a grid boundary would still split two that differ by a float ulp.
    """
    cell = max(float(tol), 1e-6)
    cells = {}          # (gi, gj) -> [(key, rx, ry), ...] representatives
    out = {}
    for sub in sorted(idf.idfobjects["FENESTRATIONSURFACE:DETAILED"],
                      key=lambda s: str(s.Name)):
        c = [(float(x), float(y)) for x, y, _z in sub.coords]
        if not c:
            continue
        cx = sum(p[0] for p in c) / len(c)
        cy = sum(p[1] for p in c) / len(c)
        gi, gj = int(math.floor(cx / cell)), int(math.floor(cy / cell))
        key = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for k, rx, ry in cells.get((gi + di, gj + dj), ()):
                    if math.hypot(cx - rx, cy - ry) <= cell:
                        key = k
                        break
                if key:
                    break
            if key:
                break
        if key is None:
            key = f"col|{cx:.2f}|{cy:.2f}"
            cells.setdefault((gi, gj), []).append((key, cx, cy))
        out[str(sub.Name)] = key
    return out


def _adopted(name, probability):
    """Deterministic per-element adoption at rate ``probability`` (0–1), hashed on
    ``name`` so a fraction ≈ ``probability`` of windows/balconies get an awning.
    Uses a STABLE hash (zlib.crc32) — Python's builtin ``hash`` is salted per process
    (PYTHONHASHSEED), which would make the random_mix behaviour split and the awning/
    blind adoption change every run."""
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return (zlib.crc32(str(name).encode("utf-8")) & 0xffff) / 65535.0 < probability


def _reshape_window_height(win, ux, uy, floor_z0, height, sill=0.1):
    """Reshape a window to a ``height``-tall balcony door: keep its along-wall
    centre and width, set its bottom at ``floor_z0 + sill``."""
    c = [(float(x), float(y), float(z)) for x, y, z in win.coords]
    # along-wall extent (relative to the first vertex)
    ox_, oy_ = c[0][0], c[0][1]
    us = [(p[0] - ox_) * ux + (p[1] - oy_) * uy for p in c]
    a, b = min(us), max(us)
    la = (ox_ + ux * a, oy_ + uy * a)
    lb = (ox_ + ux * b, oy_ + uy * b)
    zb = floor_z0 + sill
    zt = zb + height
    win.setcoords([(la[0], la[1], zb), (lb[0], lb[1], zb),
                   (lb[0], lb[1], zt), (la[0], la[1], zt)])


def _seg_point_distance(p0, p1, q):
    """2-D distance from point q to segment p0→p1 (and the projection param t)."""
    (x0, y0), (x1, y1), (qx, qy) = p0, p1, q
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(qx - x0, qy - y0), 0.0
    t = ((qx - x0) * dx + (qy - y0) * dy) / L2
    tc = max(0.0, min(1.0, t))
    px, py = x0 + tc * dx, y0 + tc * dy
    return math.hypot(qx - px, qy - py), t


def _add_door_on_wall(idf, wall, anchor_xy, width, height, name, eps=0.03):
    """Add an opaque Door subsurface to ``wall``, centred on the base point
    nearest ``anchor_xy`` and inset so it stays strictly within the wall."""
    wb = _wall_base(wall)
    if wb is None:
        return None
    p0, p1, z0, ztop = wb
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    wall_h = ztop - z0
    if L <= 2 * eps or wall_h <= 2 * eps:
        return None
    ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
    w = min(width, L - 2 * eps)
    h = min(height, wall_h - 2 * eps)
    # Centre the door on the projection of the anchor onto the base, clamped so
    # it fits inside [eps, L-eps].
    _d, t = _seg_point_distance(p0, p1, anchor_xy)
    centre = max(0.0, min(1.0, t)) * L
    s = max(eps, min(L - eps - w, centre - w / 2))
    zb = z0 + eps
    a = (p0[0] + ux * s, p0[1] + uy * s)
    b = (p0[0] + ux * (s + w), p0[1] + uy * (s + w))
    o = idf.newidfobject(
        "FENESTRATIONSURFACE:DETAILED", Name=name, Surface_Type="Door",
        Construction_Name="", Building_Surface_Name=wall.Name)
    o.setcoords([(a[0], a[1], zb), (b[0], b[1], zb),
                 (b[0], b[1], zb + h), (a[0], a[1], zb + h)])
    return o


def _remove_windows_on(idf, wall_names):
    """Delete Window subsurfaces whose base surface is in ``wall_names``."""
    for w in list(idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]):
        if (w.Surface_Type.lower() == "window"
                and w.Building_Surface_Name in wall_names):
            idf.removeidfobject(w)


def _subtract_intervals(lo, hi, forbidden):
    """Free sub-intervals of [lo, hi] after removing the ``forbidden`` ones."""
    if hi <= lo:
        return []
    fs = sorted((max(lo, a), min(hi, b)) for a, b in forbidden if b > lo and a < hi)
    free, cur = [], lo
    for a, b in fs:
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        free.append((cur, hi))
    return free


def _distribute_panes(free, n, s, gap):
    """Allocate ``n`` panes of width ``s`` across free intervals (min pitch
    s+gap), balancing density; returns a per-interval count list."""
    pitch = s + gap
    caps = [max(0, int((b - a + gap) // pitch)) for a, b in free]
    alloc = [0] * len(free)
    for _ in range(n):
        best, best_score = -1, -1.0
        for i, (a, b) in enumerate(free):
            if alloc[i] < caps[i]:
                score = (b - a) / (alloc[i] + 1)
                if score > best_score:
                    best, best_score = i, score
        if best < 0:
            break
        alloc[best] += 1
    return alloc


def _add_windows(idf, wwr, zone_meta, skip_walls=None, main_facade_walls=None):
    """Place realistic windows: equally-spaced SQUARE panes on each exterior
    (Outdoors) wall of a non-parking zone, sized so the wall's glazing meets its
    WWR — roughly one pane per ``area_per_window`` m² of glazing, with the
    residual spread equally across that wall's panes (all panes on a wall are the
    same size). Panes sit at the configured sill height and never overlap a door
    or the wall ends. ``skip_walls`` (e.g. party walls) get no windows. Interior
    walls never reach here — they are Surface (paired) or Adiabatic, not
    Outdoors. Returns the number of windows created."""
    aprw, sill = _WIN_AREA_PER, _WIN_SILL
    head, gap, min_s = _WIN_HEAD_MARGIN, _WIN_EDGE_GAP, _WIN_MIN_SIZE
    skip_walls = skip_walls or set()
    main_facade_walls = main_facade_walls or set()
    wwr_by_use = _WWR_BY_USE or {}
    comm_main_only = bool((_SHADING or {}).get("commercial_main_facade_only", True))
    if wwr <= 0:
        return 0
    doors_on = {}
    for d in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if d.Surface_Type.lower() == "door":
            doors_on.setdefault(d.Building_Surface_Name, []).append(d)

    count = 0
    for wall in list(idf.idfobjects["BUILDINGSURFACE:DETAILED"]):
        if (wall.Surface_Type.lower() != "wall"
                or wall.Outside_Boundary_Condition.lower() != "outdoors"):
            continue
        if wall.Name in skip_walls:
            continue                                   # party wall → no windows
        if _zone_is_parking(zone_meta, wall.Zone_Name):
            continue                                   # no windows on parking
        # Per-use glazing: commercial premises are far less glazed and (optionally)
        # only on their street main façade.
        cat = _resolve_use_category(zone_meta.get(str(wall.Zone_Name).upper()) or {})
        if cat == "commercial" and comm_main_only and wall.Name not in main_facade_walls:
            continue
        wwr_w = wwr * float(wwr_by_use.get(cat, 1.0))
        if wwr_w <= 0:
            continue
        wb = _wall_base(wall)
        if wb is None:
            continue
        p0, p1, z0, ztop = wb
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        H = ztop - z0
        Hv = H - sill - head                           # vertical room for a pane
        if L <= 0 or Hv < min_s:
            continue
        ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L

        forbidden = []
        for d in doors_on.get(wall.Name, []):
            us = [(x - p0[0]) * ux + (y - p0[1]) * uy for x, y, _z in d.coords]
            forbidden.append((min(us) - gap, max(us) + gap))
        free = [iv for iv in _subtract_intervals(gap, L - gap, forbidden)
                if iv[1] - iv[0] >= min_s]
        if not free:
            continue

        glaz = wwr_w * L * H
        if glaz < min_s * min_s:
            continue
        n = max(1, round(glaz / aprw))
        if glaz / n > Hv * Hv:      # square would be taller than the room →
            n = max(n, math.ceil(glaz / (Hv * Hv)))   # more, smaller panes
        s = min(math.sqrt(glaz / n), Hv)
        # Fit the panes; shrink the count if they don't fit, growing the squares
        # so the surviving panes still carry the wall's target glazing.
        for _ in range(12):
            n_fit = sum(_distribute_panes(free, n, s, gap))
            if n_fit >= n:
                s2 = min(Hv, math.sqrt(glaz / n))
                if sum(_distribute_panes(free, n, s2, gap)) >= n:
                    s = s2
                break
            if n_fit <= 0:
                n = 0
                break
            n = n_fit
            s = min(Hv, math.sqrt(glaz / n))
        if n <= 0 or s < min_s:
            continue

        alloc = _distribute_panes(free, n, s, gap)
        wi = 0
        for (a, b), k in zip(free, alloc):
            if k <= 0:
                continue
            seg = (b - a) / k
            for j in range(k):
                c = a + (j + 0.5) * seg
                u0, u1 = c - s / 2, c + s / 2
                ax, ay = p0[0] + ux * u0, p0[1] + uy * u0
                bx, by = p0[0] + ux * u1, p0[1] + uy * u1
                zb = z0 + sill
                o = idf.newidfobject(
                    "FENESTRATIONSURFACE:DETAILED", Name=f"{wall.Name}_win{wi}",
                    Surface_Type="Window", Construction_Name="",
                    Building_Surface_Name=wall.Name)
                o.setcoords([(ax, ay, zb), (bx, by, zb),
                             (bx, by, zb + s), (ax, ay, zb + s)])
                wi += 1
                count += 1
    return count


def _zone_centroids(idf):
    """{zone_name: (cx, cy)} — the horizontal centroid of each zone's surfaces,
    used to orient an exterior wall's OUTWARD normal (away from its own zone), which
    is robust for street AND patio walls unlike the raw surface winding."""
    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        z = str(getattr(s, "Zone_Name", ""))
        for x, y, _z in s.coords:
            a = acc[z]
            a[0] += float(x); a[1] += float(y); a[2] += 1
    return {z: (a[0] / a[2], a[1] / a[2]) for z, a in acc.items() if a[2]}


def _oriented_outward_normal(wall, hull=None, ref=None, **_ignored):
    """The wall's OUTWARD ``(nx, ny, bearing)``, straight from the surface winding.

    Since ``_wall_quad_outward`` winds every wall against its own room's plan
    polygon at BUILD time, the winding is outward by construction and nothing
    downstream has to second-guess it. ``hull``/``ref`` are accepted and ignored;
    they are the remains of two heuristics that were measured and rejected —
    orienting against a zone centroid inverted 46 of 49 walls, and against the
    zone's floor surfaces inverted all of them.
    """
    return _wall_outward_normal(wall, ref=ref)


def _add_subsurface_reveals(idf, depth):
    """Add a recess/reveal shade around the perimeter of every window and door:
    Spanish glazing usually aligns to the INNER wall face, so the wall thickness
    forms a ~``depth`` reveal that shades the opening on all four sides. Each
    edge becomes a thin Shading:Building:Detailed fin projecting ``depth`` along
    the host wall's OUTWARD normal (oriented away from the wall's zone)."""
    if depth <= 0:
        return 0
    wall_by_name = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    zc = _zone_centroids(idf)
    n = 0
    for sub in list(idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]):
        st = sub.Surface_Type.lower()
        if st not in ("window", "door", "glassdoor", "glazeddoor"):
            continue
        wall = wall_by_name.get(sub.Building_Surface_Name)
        if wall is None:
            continue
        nrm = _wall_outward_normal(wall, ref=zc.get(str(wall.Zone_Name)))
        if nrm is None:
            continue
        off = np.array([nrm[0], nrm[1], 0.0]) * depth   # outward, horizontal
        coords = [np.array([float(x), float(y), float(z)]) for x, y, z in sub.coords]
        m = len(coords)
        for i in range(m):
            a, b = coords[i], coords[(i + 1) % m]
            o = idf.newidfobject("SHADING:BUILDING:DETAILED",
                                 Name=f"{sub.Name}_rev{i}")
            o.setcoords([tuple(a), tuple(b), tuple(b + off), tuple(a + off)])
            n += 1
    return n


def _add_entry_doors(idf, address_xy, zone_meta,
                     main_size=_MAIN_DOOR_SIZE, parking_size=_PARKING_DOOR_SIZE,
                     skip_walls=None):
    """Add a main pedestrian door (nearest ground-floor occupied façade to the
    address) and, for each parking zone, a vehicle door (≥3.0×2.1 m) on its
    nearest ground-floor façade. Called BEFORE window placement so windows are
    laid out clear of the doors. ``skip_walls`` (e.g. party walls) get no door.
    Returns the set of wall names with a door."""
    skip_walls = skip_walls or set()
    # Candidate exterior ground-floor walls, split by parking vs occupied.
    occupied, parking = [], {}
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if s.Surface_Type.lower() != "wall":
            continue
        if s.Outside_Boundary_Condition.lower() != "outdoors":
            continue
        if s.Name in skip_walls:                # party wall → no door
            continue
        wb = _wall_base(s)
        if wb is None or abs(wb[2]) > 0.5:      # ground floor only (z0 ≈ 0)
            continue
        if _zone_is_parking(zone_meta, s.Zone_Name):
            parking.setdefault(s.Zone_Name.upper(), []).append((s, wb))
        else:
            occupied.append((s, wb))

    door_walls = set()

    def _nearest(cands):
        best, bestd = None, None
        for s, wb in cands:
            d, _t = _seg_point_distance(wb[0], wb[1], address_xy)
            if bestd is None or d < bestd:
                best, bestd = s, d
        return best

    # Main pedestrian door.
    if occupied:
        w = _nearest(occupied)
        if w is not None and _add_door_on_wall(
                idf, w, address_xy, main_size[0], main_size[1],
                f"{w.Name}_MainDoor") is not None:
            door_walls.add(w.Name)

    # One vehicle door per parking zone (prefer a wall wide enough for ≥3.0 m).
    for zname, walls in parking.items():
        wide = [(s, wb) for s, wb in walls
                if math.hypot(wb[1][0] - wb[0][0], wb[1][1] - wb[0][1])
                >= parking_size[0]]
        w = _nearest(wide or walls)
        if w is not None and _add_door_on_wall(
                idf, w, address_xy, parking_size[0], parking_size[1],
                f"{w.Name}_ParkingDoor") is not None:
            door_walls.add(w.Name)

    return door_walls


# =========================================================================== #
# EnergyPlus systems: constructions, schedules, loads, HVAC, sim control.
#
# All numbers below are reasonable defaults (CTE / EnergyPlus-example order of
# magnitude) intended to be TUNED. They make the IDF a runnable annual model
# (ideal-loads HVAC for conditioned zones) once paired with an EnergyPlus
# engine and an EPW weather file.
# =========================================================================== #

def _use_category(use_type):
    u = str(use_type or "").lower()
    if any(k in u for k in ("parking", "warehouse", "storage", "garage")):
        return "parking"
    # Kept out of the "parking" bucket on purpose: an agricultural building is
    # equally unconditioned, but it must not inherit parking's vehicle doors
    # (``_is_parking_space`` keys off this category).
    if "agricultur" in u:
        return "agricultural"
    if "communal" in u:
        return "communal"
    if any(k in u for k in ("sport", "pool")):
        return "sports"
    if any(k in u for k in ("office", "tertiary", "administ")):
        return "office"
    if any(k in u for k in ("commercial", "retail", "shop", "store")):
        return "commercial"
    if any(k in u for k in ("residential", "housing", "dwelling")):
        return "residential"
    return "other"


def _resolve_use_category(space, building_use=None):
    """Use category of a building space, ALWAYS preferring the CAT-file usage when
    available (the space's ``use_type`` / detailed use come from the CAT-derived
    ``br__thermal_zones``); only when neither is present does it fall back to the
    cadastral ``building_use``."""
    for key in ("use_type", "detailed"):
        v = space.get(key) if hasattr(space, "get") else None
        if v:
            return _use_category(v)
    return _use_category(building_use)


def zone_use_categories(idf):
    """``{ZONE_NAME (upper): use category}`` for a model built by
    :func:`build_energy_model` — 'residential', 'commercial', 'office', 'parking' …

    The IDF itself cannot hold a zone's USE, so the builder keeps it on
    ``idf._hc_zone_meta`` (and in the ``.hcmeta.json`` sidecar). Pass the result to
    :func:`hypercadaster_ES.reporting.scenario_report` so it can report each use on
    its own, instead of averaging a dwelling with the parking box below it.
    """
    zm = getattr(idf, "_hc_zone_meta", None) or {}
    return {str(z).upper(): _resolve_use_category(s)
            for z, s in zm.items() if s}


def _sched(idf, name, limits, fields):
    o = idf.newidfobject("SCHEDULE:COMPACT", Name=name,
                         Schedule_Type_Limits_Name=limits)
    o.obj.extend([str(f) for f in fields])
    return name


def _steps_to_fields(profile, day_key="AllDays"):
    """Convert a config day profile → Schedule:Compact ``For:``/``Until:`` fields.

    ``profile`` is either a flat list of ``[until "HH:MM", value]`` steps (applied
    ``For: AllDays``) or a ``{weekdays:[...], weekends:[...]}`` split. Returns the
    list of fields following the ``Through: 12/31`` header."""
    def _block(day, steps):
        out = [f"For: {day}"]
        for until, val in steps:
            out += [f"Until: {until}", val]
        return out
    if isinstance(profile, dict):
        fields = []
        if profile.get("weekdays"):
            fields += _block("Weekdays", profile["weekdays"])
        if profile.get("weekends"):
            fields += _block("AllOtherDays", profile["weekends"])
        elif profile.get("weekdays"):
            fields += ["For: AllOtherDays", "Until: 24:00", 0.0]
        return fields
    return _block(day_key, profile or [])


def _setpoint_fields(cat, p, kind):
    """Setpoint (heating|cooling) day fields for a category: an explicit
    ``schedules.setpoints.<cat>.<kind>`` profile if configured, else built from
    the category's ``heat``/``heat_sb`` (or ``cool``/``cool_sb``) scalars with a
    07:00–23:00 occupied band."""
    sp = (_SCHEDULES.get("setpoints", {}) or {}).get(cat, {})
    prof = sp.get(kind)
    if prof:
        return _steps_to_fields(prof)
    if kind == "heating":
        occ, sb = p.get("heat", 20), p.get("heat_sb", 17)
    else:
        occ, sb = p.get("cool", 25), p.get("cool_sb", 27)
    return _steps_to_fields([["07:00", sb], ["23:00", occ], ["24:00", sb]])


def _add_schedules(idf):
    """ScheduleTypeLimits + Schedule:Compact for internal-gains usage, setpoints
    and activity, one set per use category — profiles come from
    ``schedules.occupancy`` / ``schedules.setpoints`` in the config (see
    ``_SCHEDULES``), with sensible fallbacks so an empty config still runs."""
    existing = {o.Name for o in idf.idfobjects["SCHEDULETYPELIMITS"]}
    for nm, lo, hi, ntype, unit in (
            ("Fraction", 0.0, 1.0, "Continuous", ""),
            ("Temperature", -60.0, 200.0, "Continuous", "Temperature"),
            ("AnyNumber", None, None, "Continuous", ""),
    ):
        if nm in existing:
            continue
        kw = dict(Name=nm, Numeric_Type=ntype)
        if lo is not None:
            kw["Lower_Limit_Value"] = lo
            kw["Upper_Limit_Value"] = hi
        if unit:
            kw["Unit_Type"] = unit
        idf.newidfobject("SCHEDULETYPELIMITS", **kw)

    # Internal-gains usage fraction profiles per ``occ`` name (config-driven).
    _occ_default = {
        "OCC_RESIDENTIAL": [["07:00", 1.0], ["16:00", 0.3], ["23:00", 0.9], ["24:00", 1.0]],
        "OCC_OFFICE": {"weekdays": [["08:00", 0.0], ["18:00", 0.9], ["24:00", 0.1]],
                       "weekends": [["24:00", 0.05]]},
        "OCC_COMMERCIAL": [["09:00", 0.0], ["21:00", 0.8], ["24:00", 0.1]],
        "OCC_LOW": [["24:00", 0.1]],
    }
    occ_cfg = dict(_occ_default)
    occ_cfg.update(_SCHEDULES.get("occupancy", {}) or {})
    for name, profile in occ_cfg.items():
        _sched(idf, name, "Fraction", ["Through: 12/31"] + _steps_to_fields(profile))
    _sched(idf, "ALWAYS_ON", "Fraction",
           ["Through: 12/31", "For: AllDays", "Until: 24:00", 1.0])

    # Per-category setpoint + activity schedules (conditioned categories only).
    for cat, p in _LOAD_LIBRARY.items():
        if not p.get("conditioned"):
            continue
        _sched(idf, f"HEAT_SP_{cat.upper()}", "Temperature",
               ["Through: 12/31"] + _setpoint_fields(cat, p, "heating"))
        _sched(idf, f"COOL_SP_{cat.upper()}", "Temperature",
               ["Through: 12/31"] + _setpoint_fields(cat, p, "cooling"))
        _sched(idf, f"ACT_{cat.upper()}", "AnyNumber",
               ["Through: 12/31", "For: AllDays", "Until: 24:00", p["activity"]])


def _construction_period(year):
    """Map a year of construction to a Spanish thermal-regulation period:
    pre-1980 (no code), 1980-2006 (NBE-CT-79), 2007-2013 (CTE DB-HE 2006),
    post-2013 (CTE DB-HE 2013/2019)."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "1980_2006"          # most common stock when unknown
    if y < 1980:
        return "pre1980"
    if y < 2007:
        return "1980_2006"
    if y < 2014:
        return "2007_2013"
    return "post2013"


def building_archetype(year=None, use_type=None):
    """Resolve a building archetype from year of construction (Spanish thermal
    code period) and dominant use. Returns a dict with ``period``,
    ``use_category``, the period's ``envelope`` (target U-values + glazing) and
    the period's window ``infiltration`` (frame/glazing type + ACH).

    Agricultural buildings are the exception to the period rule: a shed is
    uninsulated whatever year it was built, so it takes the ``uninsulated``
    envelope instead of its period's."""
    period = _construction_period(year)
    cat = _use_category(use_type)
    env_key = "uninsulated" if (cat == "agricultural"
                                and "uninsulated" in _ARCHETYPE_ENVELOPE) else period
    return {
        "period": period,
        "use_category": cat,
        "envelope": dict(_ARCHETYPE_ENVELOPE[env_key]),
        "infiltration": _window_infiltration(period),
    }


def _window_infiltration(period):
    """Window air-infiltration for a construction ``period``: a dict with the
    ``type`` (glazing + frame technology) and ``air_changes_per_hour``. Falls
    back to the 1980-2006 entry, then a neutral default."""
    inf = _WIN_INFILTRATION.get(period)
    if not inf:
        inf = _WIN_INFILTRATION.get("1980_2006")
    if not inf:
        return {"type": "unknown", "air_changes_per_hour": 0.6}
    return {"type": inf.get("type", "unknown"),
            "air_changes_per_hour": float(inf.get("air_changes_per_hour", 0.6))}


# --------------------------------------------------------------------------- #
# Window-to-wall ratio (default constant now; pluggable predictor later)
# --------------------------------------------------------------------------- #
# Feature names a future WWR model is expected to consume, extracted from a
# building row by ``_wwr_features``. Kept explicit so the serialised model and
# this extractor stay in sync.
_WWR_FEATURE_KEYS = (
    "use_type", "year_of_construction", "n_floors_above_ground",
    "n_dwellings", "n_building_units", "footprint_area", "elevation",
)


def _wwr_features(row):
    """Best-effort feature dict for a WWR predictor, tolerant of missing cols."""
    def g(*names):
        for n in names:
            v = row.get(n)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return v
        return None

    fp = _footprints_by_floor(row)
    area = None
    if fp:
        base = fp.get(0) or next(iter(fp.values()))
        try:
            area = float(base.area)
        except Exception:
            area = None
    return {
        "use_type": g("building_use", "br__building_use_type", "use_type"),
        "year_of_construction": g("year_of_construction"),
        "n_floors_above_ground": g("br__floors_above_ground",
                                   "n_floors_above_ground"),
        "n_dwellings": g("n_dwellings"),
        "n_building_units": g("n_building_units"),
        "footprint_area": area,
        "elevation": g("elevation"),
    }


def building_wwr(row, model=None, default=_DEFAULT_WWR):
    """Resolve the window-to-wall ratio for a building.

    ``model`` may be:
      * ``None`` → return ``default`` (current behaviour);
      * a ``float`` → used directly (manual override);
      * a callable → ``model(features_dict)`` returning a ratio;
      * an estimator with ``.predict`` → called on a 1-row DataFrame of
        ``_wwr_features`` and the first output used.
    The result is clamped to ``_WWR_BOUNDS``. Any failure falls back to
    ``default`` so model issues never break IDF generation.

    A serialised predictor (to be provided) plugs in here unchanged."""
    if isinstance(model, (int, float)) and not isinstance(model, bool):
        return _clamp_wwr(float(model), default)
    if model is None:
        return _clamp_wwr(default, default)
    feats = _wwr_features(row)
    try:
        if callable(model):
            return _clamp_wwr(float(model(feats)), default)
        if hasattr(model, "predict"):
            import pandas as pd
            X = pd.DataFrame([feats], columns=list(_WWR_FEATURE_KEYS))
            pred = model.predict(X)
            val = float(pred[0] if hasattr(pred, "__len__") else pred)
            return _clamp_wwr(val, default)
    except Exception as _e:
        sys.stderr.write(f"  wwr: predictor failed ({_e}); using default\n")
    return _clamp_wwr(default, default)


def _clamp_wwr(value, default):
    lo, hi = _WWR_BOUNDS
    if value is None or not math.isfinite(value):
        value = default
    return max(lo, min(hi, value))


def _load_wwr_model(model):
    """Accept a WWR model as an already-loaded object, a float, or a path to a
    serialised estimator (joblib/pickle). Returns something ``building_wwr``
    understands, or ``None``."""
    if model is None or isinstance(model, (int, float)) or callable(model) \
            or hasattr(model, "predict"):
        return model
    if isinstance(model, str) and os.path.exists(model):
        try:
            import joblib
            return joblib.load(model)
        except Exception:
            import pickle
            with open(model, "rb") as fh:
                return pickle.load(fh)
    return None


def _fixed_resistance(layers, u_class):
    """Sum of fixed-layer + surface-film resistances (excludes the INS layer)."""
    r = _RSI.get(u_class, 0.13) + _RSE.get(u_class, 0.04)
    for lyr in layers:
        if lyr == "INS":
            continue
        _, th, k, _, _ = _MATERIALS[lyr]
        r += th / k
    return r


def _add_constructions(idf, envelope):
    """Build materials and constructions whose insulation thickness is solved to
    hit the archetype target U-values, plus a simple-glazing window."""
    have_mat = {o.Name for o in idf.idfobjects["MATERIAL"]}
    for name, (rough, th, k, rho, cp) in _MATERIALS.items():
        if name in have_mat:
            continue
        idf.newidfobject("MATERIAL", Name=name, Roughness=rough, Thickness=th,
                         Conductivity=k, Density=rho, Specific_Heat=cp,
                         Thermal_Absorptance=0.9, Solar_Absorptance=0.7,
                         Visible_Absorptance=0.7)
    have_con = {o.Name for o in idf.idfobjects["CONSTRUCTION"]}
    for cname, (u_class, layers) in _CONSTRUCTION_RECIPES.items():
        if cname in have_con:
            continue
        resolved = []
        for lyr in layers:
            if lyr != "INS":
                resolved.append(lyr)
                continue
            # Solve insulation thickness: U = 1 / (R_fixed + d_ins / k_ins).
            u_target = envelope.get(f"{u_class}_U") if u_class else None
            r_fixed = _fixed_resistance(layers, u_class)
            d_ins = _INS_CONDUCTIVITY * (1.0 / u_target - r_fixed) if u_target else 0.0
            if d_ins < 0.005:      # fixed layers already meet target → omit
                continue
            d_ins = min(d_ins, 0.40)
            mname = f"{cname}_INS"
            if mname not in {o.Name for o in idf.idfobjects["MATERIAL"]}:
                idf.newidfobject(
                    "MATERIAL", Name=mname, Roughness="Smooth",
                    Thickness=round(d_ins, 4), Conductivity=_INS_CONDUCTIVITY,
                    Density=35, Specific_Heat=1400, Thermal_Absorptance=0.9,
                    Solar_Absorptance=0.6, Visible_Absorptance=0.6)
            resolved.append(mname)
        kw = {"Name": cname, "Outside_Layer": resolved[0]}
        for i, lyr in enumerate(resolved[1:], start=2):
            kw[f"Layer_{i}"] = lyr
        idf.newidfobject("CONSTRUCTION", **kw)
    if _WINDOW_CONSTRUCTION not in have_con:
        idf.newidfobject(
            "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", Name="HC_Glazing",
            UFactor=envelope.get("window_U", 2.8),
            Solar_Heat_Gain_Coefficient=envelope.get("window_shgc", 0.6),
            Visible_Transmittance=envelope.get("window_vt", 0.7))
        idf.newidfobject("CONSTRUCTION", Name=_WINDOW_CONSTRUCTION,
                         Outside_Layer="HC_Glazing")


# The exterior roller blind ("persiana") as a real EnergyPlus shade LAYER, plus the
# schedule names its WindowShadingControl is driven by. One control per zone lists
# that zone's windows; `_apply_occupant_control` rewires the schedule per occupant
# behaviour, so the same objects serve every scenario.
_PERSIANA_MATERIAL = "HC_Persiana"
_PERSIANA_CONTROL_SCHED = "PERSIANA_ALLOW"      # season gate, rewired per behaviour
_PERSIANA_NEVER_SCHED = "PERSIANA_NEVER"        # constant 0: never deployed
_PERSIANA_NIGHT_SCHED = "PERSIANA_NIGHT"       # night-insulation season gate
# EnergyPlus refuses two DISSIMILAR WindowShadingControls on one window
# (``CheckWindowShadingControlSimilarForWindow``), so the night shutters and the
# daytime solar control cannot be separate objects. One control type carries both:
# shade at night always, and by day only when cooling is called for AND the sun on
# the glazing is strong. That gives the winter behaviour asked for without a
# second schedule — in the heating season nothing is cooling, so the day half
# never fires and the glazing stays clear, while the night half still brings the
# shutters down.
_PERSIANA_SOLAR_CONTROL_TYPE = "OnIfHighSolarOnWindow"
_PERSIANA_NIGHT_CONTROL_TYPE = "OnNightAndOnDayIfCoolingAndHighSolarOnWindow"


def _add_persiana_material(idf):
    """The ``WindowMaterial:Shade`` an exterior persiana is made of.

    A shade LAYER attenuates the beam and the sky diffuse alike, and its air gap
    and long-wave properties enter the window's heat balance. A coincident
    ``Shading:Building:Detailed`` panel cannot do either — it only removes the
    beam its own shadow happens to catch — which is why a fully closed persiana
    modelled that way still let about half the solar through.
    """
    cfg = (globals().get("_SHADING", {}) or {}).get("persiana", {}) or {}
    if _PERSIANA_MATERIAL.upper() in {str(o.Name).upper()
                                      for o in idf.idfobjects.get("WINDOWMATERIAL:SHADE", [])}:
        return _PERSIANA_MATERIAL
    idf.newidfobject(
        "WINDOWMATERIAL:SHADE", Name=_PERSIANA_MATERIAL,
        Solar_Transmittance=float(cfg.get("solar_transmittance", 0.05)),
        Solar_Reflectance=float(cfg.get("solar_reflectance", 0.55)),
        Visible_Transmittance=float(cfg.get("visible_transmittance", 0.05)),
        Visible_Reflectance=float(cfg.get("visible_reflectance", 0.55)),
        Infrared_Hemispherical_Emissivity=float(cfg.get("thermal_emissivity", 0.9)),
        Infrared_Transmittance=float(cfg.get("thermal_transmittance", 0.0)),
        Thickness=float(cfg.get("thickness", 0.003)),
        Conductivity=float(cfg.get("conductivity", 0.15)),
        Shade_to_Glass_Distance=float(cfg.get("distance_to_glass", 0.05)),
        Top_Opening_Multiplier=0.5, Bottom_Opening_Multiplier=0.5,
        LeftSide_Opening_Multiplier=0.5, RightSide_Opening_Multiplier=0.5,
        Airflow_Permeability=float(cfg.get("air_permeability", 0.05)))
    return _PERSIANA_MATERIAL


def _add_persiana_controls(idf, zone_meta, season_by_doy=None):
    """One ``WindowShadingControl`` per RESIDENTIAL zone, over all its windows.

    The control is ``OnIfHighSolarOnWindow`` **and** scheduled, which in
    EnergyPlus means the persiana closes only when the schedule allows it *and*
    the incident solar exceeds the setpoint. The schedule is the season gate: 0
    through the HEATING season — a Spanish household wants that sun, so the
    persiana stays fully retracted — and 1 through the cooling and in-between
    seasons. ``_apply_occupant_control`` swaps in a per-behaviour schedule so the
    occupants who never touch them keep them open all year.

    Returns the number of controls added.
    """
    cfg = (globals().get("_SHADING", {}) or {}).get("persiana", {}) or {}
    if not cfg.get("enabled", True):
        return 0
    material = _add_persiana_material(idf)
    _persiana_allow_schedule(idf, season_by_doy)
    _sched_compact(idf, _PERSIANA_NEVER_SCHED, "Fraction",
                   ["For: AllDays", "Until: 24:00,0"])
    setpoint = float(cfg.get("solar_setpoint_w_m2", 140))

    windows_by_zone = {}
    walls = {str(s.Name): str(s.Zone_Name) for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    for w in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if str(getattr(w, "Surface_Type", "")).lower() != "window":
            continue
        z = walls.get(str(w.Building_Surface_Name))
        if z:
            windows_by_zone.setdefault(z, []).append(str(w.Name))

    def _control(zone, wins, name, ctype, sched, sp, seq):
        o = idf.newidfobject(
            "WINDOWSHADINGCONTROL", Name=name, Zone_Name=zone,
            Shading_Control_Sequence_Number=seq,
            Shading_Type="ExteriorShade",
            Shading_Control_Type=ctype,
            Schedule_Name=sched,
            Setpoint=sp,
            Shading_Control_Is_Scheduled="Yes",
            Glare_Control_Is_Active="No",
            Shading_Device_Material_Name=material,
            Type_of_Slat_Angle_Control_for_Blinds="FixedSlatAngle",
            Multiple_Surface_Control_Type="Sequential")
        for i, wn in enumerate(wins, start=1):
            try:
                setattr(o, f"Fenestration_Surface_{i}_Name", wn)
            except Exception:  # noqa: BLE001 - more windows than the IDD allows
                break
        return o

    n = 0
    for zone, wins in windows_by_zone.items():
        s = zone_meta.get(zone.upper())
        if not s or _resolve_use_category(s) != "residential":
            continue                       # persianas are a dwelling feature
        # ONE control per zone; ``_apply_occupant_control`` swaps its TYPE and
        # schedule per occupant, which is how the night shutters are given to the
        # behaviours that use them without a second (illegal) control object.
        _control(zone, wins, f"Persiana_{zone}", _PERSIANA_SOLAR_CONTROL_TYPE,
                 _PERSIANA_CONTROL_SCHED, setpoint, 1)
        n += 1
    return n


def _persiana_allow_schedule(idf, season_by_doy=None, name=_PERSIANA_CONTROL_SCHED):
    """The season gate: 0 every HEATING day (persianas and awnings fully open, the
    sun is welcome), 1 through the cooling and in-between seasons."""
    on = ["For: AllDays", "Until: 24:00,1"]
    off = ["For: AllDays", "Until: 24:00,0"]
    if season_by_doy:
        return _seasonal_day_schedule(idf, name, season_by_doy,
                                      {"heating": off, "cooling": on,
                                       "in_between": on})
    # No EPW yet: fall back to the month split the rest of the model uses.
    if name.upper() in {str(o.Name).upper()
                        for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        return name
    fields = []
    for through, active in (("3/31", False), ("5/31", True), ("9/30", True),
                            ("11/30", True), ("12/31", False)):
        fields.append(f"Through: {through}")
        fields += on if active else off
    _sched(idf, name, "Fraction", fields)
    return name


# Indoor CO2. EnergyPlus derives a person's CO2 generation from the ACTIVITY
# LEVEL (``People.Carbon_Dioxide_Generation_Rate`` is m³/s per WATT), so the
# sleeping metabolic rate below produces proportionally less CO2 than a waking
# hour without a second schedule — one number moves the heat and the CO2 together.
_CO2_OUTDOOR_SCHED = "CO2_OUTDOOR"
_ACT_RESIDENTIAL_NIGHT = "ACT_RESIDENTIAL_NIGHT"


def _add_co2_balance(idf):
    """Switch on the contaminant balance so every zone reports its CO2 in ppm.

    CO2 is the air-RENEWAL indicator: occupants raise it, infiltration and open
    windows flush it, so it reads the census occupancy and the occupant's
    ventilation behaviour at the same time. Returns the number of People objects
    given a generation rate, or 0 when disabled.
    """
    cfg = globals().get("_CO2", {}) or {}
    if not cfg.get("enabled", True):
        return 0
    ppm = float(cfg.get("outdoor_ppm", 420))
    if _CO2_OUTDOOR_SCHED.upper() not in {str(o.Name).upper()
                                          for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        _sched(idf, _CO2_OUTDOOR_SCHED, "AnyNumber",
               ["Through: 12/31", "For: AllDays", "Until: 24:00", ppm])
    if not idf.idfobjects.get("ZONEAIRCONTAMINANTBALANCE", []):
        idf.newidfobject("ZONEAIRCONTAMINANTBALANCE",
                         Carbon_Dioxide_Concentration="Yes",
                         Outdoor_Carbon_Dioxide_Schedule_Name=_CO2_OUTDOOR_SCHED)
    rate = float(cfg.get("generation_rate_m3_s_w", 3.82e-8))
    n = 0
    for o in idf.idfobjects.get("PEOPLE", []):
        try:
            o.Carbon_Dioxide_Generation_Rate = rate
            n += 1
        except Exception:  # noqa: BLE001 - an older IDD without the field
            break
    return n


def _month_blocks(profile_by_month, limits="Temperature"):
    """Schedule:Compact fields from ``{month: [day fields]}``, consecutive months
    with the same profile merged into one ``Through`` block."""
    out, m = [], 1
    while m <= 12:
        prof = profile_by_month.get(m)
        end = m
        while end < 12 and profile_by_month.get(end + 1) == prof:
            end += 1
        out.append(f"Through: {end}/{_MONTH_DAYS[end - 1]}")
        out += prof or ["For: AllDays", "Until: 24:00,0"]
        m = end + 1
    return out


def _add_cte_thermostat(idf):
    """The normative CTE DB-HE thermostat, and the availability that goes with it.

    CTE DB-HE Apéndice D divides the year by CALENDAR DATE, with **no gap**: every
    hour is either heating-enabled or cooling-enabled. The "in-between season"
    this model gates the occupant scenarios with — mild weather, nobody touches
    the HVAC — is a description of behaviour, not a rule, and has no place in the
    normative case. So the CTE zones are available all year and the service that
    is out of its season is made unreachable by its setpoint instead (EnergyPlus
    offers no way to disable one side of a HVACTemplate thermostat).

    Within its season the standard also prescribes a shape, not a constant: a
    heating night setback, and a tighter cooling consigna for the occupied
    evening. Values come from ``residential_cte.setpoints``.

    Returns ``(thermostat_name, availability_schedule_name)``.
    """
    cfg = globals().get("_RES_CTE", {}) or {}
    seasons = cfg.get("seasons", {}) or {}
    sp = cfg.get("setpoints", {}) or {}
    heat_months = set(seasons.get("heating_months", [1, 2, 3, 4, 5, 10, 11, 12]))
    cool_months = set(seasons.get("cooling_months", [6, 7, 8, 9]))
    h = dict(sp.get("heating", {}) or {})
    c = dict(sp.get("cooling", {}) or {})
    off_h = float(sp.get("off_heating", -50))
    off_c = float(sp.get("off_cooling", 100))

    h_night, h_day = float(h.get("night", 17)), float(h.get("day", 20))
    h_from = int(h.get("day_from", 7))
    c_day, c_eve = float(c.get("day", 27)), float(c.get("evening", 25))
    c_from = int(c.get("evening_from", 16))

    heat_on = ["For: AllDays", f"Until: {h_from:02d}:00,{h_night}",
               f"Until: 24:00,{h_day}"]
    heat_off = ["For: AllDays", f"Until: 24:00,{off_h}"]
    cool_on = ["For: AllDays", f"Until: {c_from:02d}:00,{c_day}",
               f"Until: 24:00,{c_eve}"]
    cool_off = ["For: AllDays", f"Until: 24:00,{off_c}"]

    name = "TSTAT_CTE"
    if name.upper() not in {str(o.Name).upper()
                            for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}:
        _sched(idf, "SP_CTE_HEAT", "Temperature",
               _month_blocks({m: (heat_on if m in heat_months else heat_off)
                              for m in range(1, 13)}))
        _sched(idf, "SP_CTE_COOL", "Temperature",
               _month_blocks({m: (cool_on if m in cool_months else cool_off)
                              for m in range(1, 13)}))
        idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=name,
                         Heating_Setpoint_Schedule_Name="SP_CTE_HEAT",
                         Cooling_Setpoint_Schedule_Name="SP_CTE_COOL")
    avail = _sched_compact(idf, "AVAIL_CTE", "Fraction",
                           ["For: AllDays", "Until: 24:00,1"])
    return name, avail


def _add_night_activity_schedule(idf, cat="residential"):
    """Residential metabolic rate with the NIGHT band asleep.

    The fallback for a building with no Time Use Survey agenda: the night hours
    (:data:`_NIGHT_HOURS`) take the sleeping rate, the rest the waking one. With
    an agenda, ``_apply_occupant_control`` builds the same schedule off the
    survey's own Sleeping hours instead, which is the better source.
    """
    p = (globals().get("_LOAD_LIBRARY", {}) or {}).get(cat, {})
    awake = float(p.get("activity", 110))
    asleep = float(globals().get("_ACT_SLEEPING", 72))
    night = sorted(_NIGHT_HOURS)
    fields, prev = ["For: AllDays"], None
    for h in range(1, 25):
        v = asleep if h in _NIGHT_HOURS else awake
        if prev is not None and abs(v - prev) < 1e-9:
            fields[-1] = f"Until: {h:02d}:00,{v}"
        else:
            fields.append(f"Until: {h:02d}:00,{v}")
        prev = v
    _sched_compact(idf, _ACT_RESIDENTIAL_NIGHT, "AnyNumber", fields)
    return _ACT_RESIDENTIAL_NIGHT


def _construction_for(surface):
    st = surface.Surface_Type.lower()
    bc = surface.Outside_Boundary_Condition.lower()
    if st == "wall":
        if bc == "adiabatic":
            # Preserve the party (HC_PartyWall) vs internal-adiabatic (HC_IntWall)
            # distinction stamped by ``_refine_exterior_bcs`` (both are thermally
            # adiabatic — same layers — but kept apart for the geometry view).
            existing = (getattr(surface, "Construction_Name", "") or "").strip()
            return existing if existing in ("HC_PartyWall", "HC_IntWall") else "HC_PartyWall"
        return {"outdoors": "HC_ExtWall", "ground": "HC_GroundWall",
                "surface": "HC_IntWall"}.get(bc, "HC_ExtWall")
    if st == "floor":
        return {"ground": "HC_GroundFloor", "surface": "HC_IntFloor",
                "outdoors": "HC_ExposedFloor"}.get(bc, "HC_GroundFloor")
    if st in ("roof",):
        return "HC_Roof"
    if st in ("ceiling",):
        return "HC_IntFloor"
    return "HC_ExtWall"


def _assign_constructions(idf):
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        s.Construction_Name = _construction_for(s)
    for w in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        st = w.Surface_Type.lower()
        if st == "door":
            w.Construction_Name = (_PARKING_DOOR_CONSTRUCTION
                                   if "parkingdoor" in w.Name.lower()
                                   else _MAIN_DOOR_CONSTRUCTION)
        else:
            w.Construction_Name = _WINDOW_CONSTRUCTION


def _set_first(obj, candidates, value):
    """Set the first field in `candidates` that exists on `obj` (handles field
    renames across EnergyPlus versions)."""
    for f in candidates:
        if f in obj.fieldnames:
            setattr(obj, f, value)
            return f
    return None


def _first_value(obj, candidates, default=None):
    """Read back the first field in `candidates` that exists on `obj` and carries a
    value — the getter half of :func:`_set_first`."""
    for f in candidates:
        if obj is not None and f in obj.fieldnames:
            v = getattr(obj, f, None)
            if v not in (None, ""):
                return v
    return default


# Field-name candidates that vary across EnergyPlus versions.
_F_ZONE = ("Zone_or_ZoneList_or_Space_or_SpaceList_Name",
           "Zone_or_ZoneList_Name", "Zone_Name")
_F_PPL_AREA = ("People_per_Floor_Area", "People_per_Zone_Floor_Area")
_F_W_AREA = ("Watts_per_Floor_Area", "Watts_per_Zone_Floor_Area")
_F_MIX_ZONE = ("Zone_or_Space_Name", "Zone_Name")
_F_MIX_SRC = ("Source_Zone_or_Space_Name", "Source_Zone_Name")


def _add_interfloor_connections(idf, zone_meta, flow=0.05):
    """Connect the stacked per-floor zones of a multi-floor dwelling with a
    ZoneCrossMixing air exchange — modelling the open stair/lift shaft that
    thermally couples the floors of a duplex/triplex. Only conditioned zones of
    the SAME building space on consecutive floors are linked. Returns the count."""
    from collections import defaultdict
    groups = defaultdict(list)
    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        if not s:
            continue
        p = _LOAD_LIBRARY.get(_use_category(s.get("use_type")), _LOAD_LIBRARY["other"])
        if not p.get("conditioned"):
            continue
        groups[id(s)].append(zone.Name)

    def _flr(name):
        try:
            return int(name.rsplit("_F", 1)[1])
        except (IndexError, ValueError):
            return 0

    n = 0
    for names in groups.values():
        if len(names) < 2:
            continue
        names = sorted(set(names), key=_flr)
        for a, b in zip(names, names[1:]):
            if _flr(b) - _flr(a) != 1:
                continue                        # only directly stacked floors
            o = idf.newidfobject(
                "ZONECROSSMIXING", Name=f"Stair_{a}_{b}",
                Schedule_Name="ALWAYS_ON",
                Design_Flow_Rate_Calculation_Method="Flow/Zone",
                Design_Flow_Rate=flow)
            _set_first(o, _F_MIX_ZONE, a)
            _set_first(o, _F_MIX_SRC, b)
            n += 1
    return n


def _add_loads(idf, zone_meta, window_infil_ach=None, residential_ppl_per_m2=None):
    """People / Lights / ElectricEquipment / Infiltration per zone, by use.
    Conditioned zones take their infiltration from the window technology of the
    construction period (``window_infil_ach``); non-conditioned zones
    (parking/communal/sports) keep their use-based ``infil_ach``.

    ``residential_ppl_per_m2`` overrides the ``loads.residential.ppl_per_m2``
    default for RESIDENTIAL zones with the building's census-derived occupancy
    (:func:`_residential_occupancy_density`), so the metabolic gains describe the
    people the census counts there rather than a national average."""
    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        cat = _use_category(s["use_type"]) if s else "other"
        p = _LOAD_LIBRARY.get(cat, _LOAD_LIBRARY["other"])
        occ = p["occ"]
        ppl = p["ppl_per_m2"]
        if cat == "residential" and residential_ppl_per_m2:
            ppl = float(residential_ppl_per_m2)
        if ppl > 0:
            o = idf.newidfobject(
                "PEOPLE", Name=f"{zone.Name}_People",
                Number_of_People_Schedule_Name=occ,
                Number_of_People_Calculation_Method="People/Area",
                Activity_Level_Schedule_Name=f"ACT_{cat.upper()}"
                if p.get("conditioned") else "ACT_OFFICE")
            _set_first(o, _F_ZONE, zone.Name)
            _set_first(o, _F_PPL_AREA, ppl)
        if p["lights"] > 0:
            o = idf.newidfobject(
                "LIGHTS", Name=f"{zone.Name}_Lights", Schedule_Name=occ,
                Design_Level_Calculation_Method="Watts/Area")
            _set_first(o, _F_ZONE, zone.Name)
            _set_first(o, _F_W_AREA, p["lights"])
        if p["equip"] > 0:
            o = idf.newidfobject(
                "ELECTRICEQUIPMENT", Name=f"{zone.Name}_Equip", Schedule_Name=occ,
                Design_Level_Calculation_Method="Watts/Area")
            _set_first(o, _F_ZONE, zone.Name)
            _set_first(o, _F_W_AREA, p["equip"])
        ach = (window_infil_ach if (p.get("conditioned") and window_infil_ach)
               else p["infil_ach"])
        o = idf.newidfobject(
            "ZONEINFILTRATION:DESIGNFLOWRATE", Name=f"{zone.Name}_Infil",
            Schedule_Name="ALWAYS_ON",
            Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
            Air_Changes_per_Hour=ach)
        _set_first(o, _F_ZONE, zone.Name)


def _add_hvac(idf, zone_meta):
    """Ideal-loads HVAC + thermostat for conditioned zones (one thermostat
    template per conditioned category); non-occupied zones stay free-floating."""
    made_thermostats = set()
    areas = _zone_floor_areas(idf)
    hcfg = globals().get("_HVAC", {}) or {}
    q_heat = float(hcfg.get("max_heating_w_per_m2", 100))
    q_cool = float(hcfg.get("max_cooling_w_per_m2", 80))
    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        cat = _use_category(s["use_type"]) if s else "other"
        p = _LOAD_LIBRARY.get(cat, _LOAD_LIBRARY["other"])
        if not p.get("conditioned"):
            continue
        tstat = f"TSTAT_{cat.upper()}"
        if tstat not in made_thermostats:
            idf.newidfobject(
                "HVACTEMPLATE:THERMOSTAT", Name=tstat,
                Heating_Setpoint_Schedule_Name=f"HEAT_SP_{cat.upper()}",
                Cooling_Setpoint_Schedule_Name=f"COOL_SP_{cat.upper()}")
            made_thermostats.add(tstat)
        iz = idf.newidfobject(
            "HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM",
            Zone_Name=zone.Name, Template_Thermostat_Name=tstat)
        # Finite HVAC: cap heating/cooling capacity from the zone floor area so the
        # ideal-loads system is a realistically-sized plant, not an infinite one.
        a = float(areas.get(zone.Name.upper(), 0.0))
        if a > 0:
            iz.Heating_Limit = "LimitCapacity"
            iz.Maximum_Sensible_Heating_Capacity = round(q_heat * a, 1)
            iz.Cooling_Limit = "LimitCapacity"
            iz.Maximum_Total_Cooling_Capacity = round(q_cool * a, 1)


# --------------------------------------------------------------------------- #
# Residential occupant-behaviour archetypes (unaware / aware / always_on /
# passive) + occupancy periods — assigned per dwelling from census-tract demographics
# (random, seeded) or fixed. All settings in energy_config.yaml.
# --------------------------------------------------------------------------- #
def _residential_demographics(row):
    """(average_household_size, population_mean_age) for a building row from the
    sociodemographic layer, or (None, None) when unavailable."""
    def _num(*names):
        for n in names:
            v = row.get(n) if hasattr(row, "get") else None
            try:
                f = float(v)
                if math.isfinite(f):
                    return f
            except (TypeError, ValueError):
                pass
        return None
    return (_num("br__average_household_size"),
            _num("br__population_mean_age"))


def _bucket_name(value, buckets, key, default="default"):
    """Pick the first bucket whose ``key`` threshold ≥ ``value`` (buckets ordered
    ascending in the config); ``default`` when value missing or no bucket."""
    if value is None or not buckets:
        return default
    for name, spec in buckets.items():
        thr = spec.get(key) if isinstance(spec, dict) else None
        try:
            if thr is not None and value <= float(thr):
                return name
        except (TypeError, ValueError):
            continue
    return list(buckets)[-1]


def _weighted_choice(weights, rng):
    """Weighted random key from ``{key: weight}`` using ``rng`` (a random.Random)."""
    items = [(k, float(v)) for k, v in (weights or {}).items()
             if _safe_float(v) and float(v) > 0]
    if not items:
        return None
    tot = sum(w for _k, w in items)
    r = rng.random() * tot
    acc = 0.0
    for k, w in items:
        acc += w
        if r <= acc:
            return k
    return items[-1][0]


def _safe_float(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _assign_residential_user(reference, household_size, mean_age,
                             fixed_behavior=None, fixed_period=None, seed=0):
    """Resolve the occupant BEHAVIOUR of one residential dwelling — fixed when
    given, else a demographic-weighted deterministic random draw (seeded by the
    dwelling reference so a rebuild is reproducible). Weights come from
    ``residential_behavior.assignment.behavior_weights``, bucketed by the section's
    mean age.

    Returns ``(behavior, None)``. The second element used to be a per-dwelling
    OCCUPANCY PERIOD, drawn from ``occupancy_periods`` by household size. It is
    gone: the occupant engine reads occupancy from the census tract's INE Time Use
    Survey agenda, or from one stylised residential profile when the survey does
    not reach the tract, so a per-dwelling period changed nothing in any KPI. The
    tuple shape is kept because it is what ``idf._hc_behavior`` carries.
    ``fixed_period`` is accepted and ignored, for callers written against the old
    signature."""
    import random
    rb = _RESIDENTIAL_BEHAVIOR
    behaviors = rb.get("behaviors", {})
    asg = rb.get("assignment", {})
    if fixed_behavior and fixed_behavior in behaviors:
        return fixed_behavior, None
    ab = _bucket_name(mean_age, asg.get("age_buckets", {}), "max_mean_age")
    bw = (asg.get("behavior_weights", {}).get(ab)
          or asg.get("behavior_weights", {}).get("default", {}))
    behavior = (_weighted_choice(bw, random.Random(f"{seed}|{reference}|b"))
                or next(iter(behaviors), "aware"))
    return behavior, None


def _assign_residential_behavior(idf, zone_meta, row, fixed_behavior=None,
                                 fixed_period=None, seed=0):
    """Assign each RESIDENTIAL zone its occupant BEHAVIOUR archetype and return
    ``{zone_name: (behavior, None)}`` — the map ``build_energy_model`` puts on
    ``idf._hc_behavior`` and every downstream pass reads.

    It AUTHORS NOTHING. Setpoints, HVAC availability, occupancy and ventilation are
    all written by ``_wire_occupant_model`` → ``_apply_occupant_control``, which is
    the single occupant engine. This function used to write its own thermostats,
    availability schedules and ``ZoneVentilation`` objects from a parallel set of
    config keys (``behaviors.*.heating``/``cooling``/``hvac_availability``/
    ``ventilation_ach``, ``ventilation_schedules``, ``occupancy_periods``) — and
    every one of them was then deleted or overwritten before a single KPI was
    computed, so the config had two behaviour models of which only one was ever
    simulated. Those keys are gone.

    No-op when the behaviour engine is disabled or the building has no dwellings.
    """
    rb = _RESIDENTIAL_BEHAVIOR
    if not rb or not rb.get("enabled", True) or not rb.get("behaviors", {}):
        return {}
    hh, mean_age = _residential_demographics(row)
    return {zone.Name: _assign_residential_user(
                s.get("reference") or zone.Name, hh, mean_age,
                fixed_behavior, fixed_period, seed)
            for zone in idf.idfobjects["ZONE"]
            for s in [zone_meta.get(zone.Name.upper())]
            if s and _use_category(s.get("use_type")) == "residential"}


def _wire_occupant_model(idf, zone_meta, row, seed=0, time_use=None,
                         sp_offset=None):
    """THE occupant model, in one pass — used by ``build_energy_model`` and by
    every scenario of ``run_behavior_scenarios``, so the .idf that is written,
    plotted and simulated all carry the same behaviour wiring.

    Reads the per-dwelling archetype off ``idf._hc_behavior`` and authors, for
    every occupied zone: the occupancy schedules and metabolic profile, the
    season-gated HVAC availability, the thermostats (adaptive for aware/passive,
    fixed bands for the others, the RITE/RD 486-2022 restriction for the tertiary
    zones) and the whole ventilation model — airings, night purge, CTE DB-HS3
    continuous ventilation and the mechanical outdoor air of the premises.

    ``sp_offset`` is the income-driven setpoint shift; computed from ``row`` when
    not supplied. Idempotent: re-applying it to a model that has been through it
    reuses the schedules by name and re-authors only the per-zone assignment,
    which is what lets a scenario rewire a COPY of the built model.
    """
    if sp_offset is None:
        sp_offset = _income_setpoint_offset(_building_income(row))
    # ``row`` is normally a pandas Series — never truth-test it (``row or {}``
    # raises "the truth value of a Series is ambiguous"), just ask for .get.
    year = row.get("year_of_construction") if hasattr(row, "get") else None
    _apply_occupant_control(idf, zone_meta, seed=seed, sp_offset=sp_offset,
                            time_use=time_use, year_of_construction=year)
    res_zones = {z for z, s in zone_meta.items()
                 if _resolve_use_category(s) == "residential"}
    # AWARE and PASSIVE condition to the EN 16798-1 ADAPTIVE comfort band, whose
    # occupied/away split reads the SAME occupancy mask that gated their window
    # openings — the census tract's Time Use Survey week where it reaches the
    # building, the stylised profile otherwise. _apply_occupant_control recorded
    # which one applies to each dwelling; the zones are grouped by it so a building
    # whose survey covers only some behaviours still gets one thermostat per mask
    # rather than one mask imposed on all of them.
    aware_zones = {z.upper() for z, (b, _p) in
                   (getattr(idf, "_hc_behavior", {}) or {}).items()
                   if b in ("aware", "passive")} & res_zones
    masks = getattr(idf, "_hc_occ_masks", None) or {"STY": _stylised_occupancy_mask()}
    tags = getattr(idf, "_hc_occ_tag", None) or {}
    by_tag = {}
    for z in aware_zones:
        by_tag.setdefault(tags.get(z, "STY"), set()).add(z)
    for tag, zones in sorted(by_tag.items()):
        _apply_adaptive_setpoints(idf, zones, sp_offset=sp_offset, tag=tag,
                                  occ_by_day=masks.get(tag))
    # Non-residential zones: one normative user, independent of the archetype.
    _apply_nonresidential_setpoints(idf, zone_meta)


# EnergyPlus Output:Variable reporting-frequency names, keyed by config aliases.
_FREQ_ALIASES = {
    "hour": "Hourly", "hourly": "Hourly", "day": "Daily", "daily": "Daily",
    "month": "Monthly", "monthly": "Monthly", "year": "Annual",
    "annual": "Annual", "runperiod": "RunPeriod", "timestep": "Timestep",
}


def _resolve_output_frequencies(frequencies=None):
    """Normalise the requested output frequencies to a list of EnergyPlus
    reporting-frequency names (from ``build_energy_model`` or
    ``simulation.output_frequencies``)."""
    if frequencies is None:
        frequencies = _SIM.get("output_frequencies", ["monthly"])
    if isinstance(frequencies, str):
        frequencies = [frequencies]
    return [str(f).lower() for f in frequencies]


def _add_outputs(idf, frequencies=None):
    """Add ``Output:Variable`` requests per reporting frequency. Fine
    frequencies (hourly/daily) emit COMFORT (air temperature + relative humidity)
    AND thermal DEMAND per zone; coarse frequencies (monthly/annual) emit thermal
    demand per zone only. Variable names + the fine/coarse split come from
    ``simulation.outputs`` in the config."""
    out_cfg = _SIM.get("outputs", {}) or {}
    comfort = list(out_cfg.get("comfort",
                   ["Zone Mean Air Temperature", "Zone Air Relative Humidity"]))
    demand = list(out_cfg.get("demand",
                  ["Zone Ideal Loads Supply Air Total Heating Energy",
                   "Zone Ideal Loads Supply Air Total Cooling Energy"]))
    fine = {str(f).lower()
            for f in out_cfg.get("fine_frequencies", ["hourly", "daily"])}
    seen = set()
    for key in _resolve_output_frequencies(frequencies):
        ep_freq = _FREQ_ALIASES.get(key, "Monthly")
        # Comfort (temp + RH) for every zone at fine frequency; demand always.
        variables = (comfort + demand) if key in fine else list(demand)
        for var in variables:
            if (var, ep_freq) in seen:
                continue
            seen.add((var, ep_freq))
            idf.newidfobject("OUTPUT:VARIABLE", Key_Value="*",
                             Variable_Name=var, Reporting_Frequency=ep_freq)


def _add_sim_control(idf, latitude=None, longitude=None, time_zone=None,
                     elevation=None, output_frequencies=None):
    loc = _SIM.get("location", {})
    if latitude is None:
        latitude = loc.get("latitude", 41.62)
    if longitude is None:
        longitude = loc.get("longitude", 0.63)
    if time_zone is None:
        time_zone = loc.get("time_zone", 1.0)
    if elevation is None:
        elevation = loc.get("elevation", 200.0)
    if not idf.idfobjects["SIMULATIONCONTROL"]:
        idf.newidfobject(
            "SIMULATIONCONTROL",
            Do_Zone_Sizing_Calculation="No", Do_System_Sizing_Calculation="No",
            Do_Plant_Sizing_Calculation="No",
            Run_Simulation_for_Sizing_Periods="No",
            Run_Simulation_for_Weather_File_Run_Periods="Yes")
    if not idf.idfobjects["TIMESTEP"]:
        idf.newidfobject("TIMESTEP",
                         Number_of_Timesteps_per_Hour=int(_SIM.get("timestep_per_hour", 4)))
    if not idf.idfobjects["RUNPERIOD"]:
        idf.newidfobject("RUNPERIOD", Name="FullYear", Begin_Month=1,
                         Begin_Day_of_Month=1, End_Month=12, End_Day_of_Month=31)
    if not idf.idfobjects["SITE:LOCATION"]:
        idf.newidfobject("SITE:LOCATION", Name="Site", Latitude=latitude,
                         Longitude=longitude, Time_Zone=time_zone,
                         Elevation=elevation)
    _add_outputs(idf, output_frequencies)
    idf.newidfobject("OUTPUT:METER", Key_Name="Electricity:Facility",
                     Reporting_Frequency="Monthly")
    if not idf.idfobjects["OUTPUTCONTROL:TABLE:STYLE"]:
        idf.newidfobject("OUTPUTCONTROL:TABLE:STYLE", Column_Separator="HTML")
    idf.newidfobject("OUTPUT:TABLE:SUMMARYREPORTS",
                     Report_1_Name="AllSummaryAndMonthly")


def add_energy_systems(idf, zone_meta, latitude=None, longitude=None,
                       time_zone=None, elevation=None, envelope=None,
                       archetype=None, output_frequencies=None, row=None,
                       residential_behavior=None, residential_period=None,
                       behavior_seed=0, wd=None):
    """Add constructions, schedules, loads, ideal-loads HVAC and simulation
    control to a geometry-only IDF, making it a runnable annual model.

    ``envelope`` is the archetype envelope dict (target U-values + glazing) from
    ``building_archetype(...)["envelope"]``; if omitted it defaults to the
    1980-2006 (NBE-CT-79) period. ``archetype`` carries the period, used here to
    set the conditioned-zone air infiltration from the window technology.
    ``output_frequencies`` selects the Output:Variable reporting frequencies
    (defaults to ``simulation.output_frequencies`` in the config). ``row`` (the
    building attributes) + ``residential_behavior`` / ``behavior_seed`` drive the
    residential occupant-behaviour archetypes (``_assign_residential_behavior``);
    when ``residential_behavior`` is None each dwelling draws one at random from
    the section's demographics. The assignment is then wired into a complete,
    simulable occupant model by ``_wire_occupant_model`` — the same pass every
    scenario runs. ``wd`` is the hypercadaster working directory, needed only to
    reach the census tract's INE Time Use Survey occupancy agenda; without it the
    stylised residential profile is used.

    ``residential_period`` is accepted and IGNORED. It used to pick a per-dwelling
    occupancy period, which nothing has read since occupancy moved to the Time Use
    Survey agenda / the stylised profile."""
    if envelope is None:
        envelope = dict(_ARCHETYPE_ENVELOPE["1980_2006"])
    period = archetype["period"] if archetype else "1980_2006"
    win_infil = _window_infiltration(period)["air_changes_per_hour"]
    _add_constructions(idf, envelope)
    _assign_constructions(idf)
    _add_schedules(idf)
    # Residential metabolic gains follow the building's census-tract occupancy
    # (ctagg__people_per_m2 from hc.merge) when it carries one — the same head
    # count the DHW demand uses — instead of the national ppl_per_m2 default.
    # Indoor CO2 balance + the CO2 generation rate on every People object.
    try:
        _add_co2_balance(idf)
    except Exception as _e:  # noqa: BLE001 - the thermal model stands without it
        sys.stderr.write(f"  co2: could not add the contaminant balance ({_e})\n")
    # Exterior persianas: a real shade layer + a per-zone WindowShadingControl,
    # gated OFF through the heating season (see _add_persiana_controls).
    try:
        _add_persiana_controls(idf, zone_meta,
                               season_by_doy=_season_by_doy(getattr(idf, "epw", None)))
    except Exception as _e:  # noqa: BLE001 - the model is still valid without them
        sys.stderr.write(f"  persiana: could not add shading control ({_e})\n")
    res_ppl = None
    if row is not None and globals().get("_RES_OCC_FROM_CENSUS", True):
        res_ppl = _residential_occupancy_density(row)
    _add_loads(idf, zone_meta, window_infil_ach=win_infil,
               residential_ppl_per_m2=res_ppl)
    _add_hvac(idf, zone_meta)
    # Residential occupant behaviour. The archetype is assigned per dwelling and
    # kept on the IDF (so scenario tooling can target e.g. the AWARE dwellings),
    # then the ONE occupant engine authors everything that follows from it:
    # occupancy, setpoints, HVAC availability and the whole ventilation model.
    # The model returned here is therefore a COMPLETE, simulable one — the same
    # wiring every scenario runs, not a half-built shell the scenarios replace.
    if row is not None:
        idf._hc_behavior = _assign_residential_behavior(
            idf, zone_meta, row, fixed_behavior=residential_behavior,
            fixed_period=residential_period, seed=behavior_seed)
        _wire_occupant_model(idf, zone_meta, row, seed=behavior_seed,
                             time_use=_building_time_use_agenda(wd, row))
        # Conservation-status envelope leakage. Applied HERE and only here: it
        # scales the infiltration objects in place, so a second pass would square
        # the factor -- and it is a property of the FABRIC, identical in every
        # scenario, which is why the scenario loop inherits it through the copy.
        _apply_infiltration_factor(idf, _conservation_infiltration_factor(row))
    _add_interfloor_connections(idf, zone_meta)   # stair/lift coupling of duplexes
    _add_sim_control(idf, latitude, longitude, time_zone, elevation,
                     output_frequencies=output_frequencies)
    return idf


# Colours for the IDF viewer, keyed by the fine-grained surface CATEGORY
# (surface type + subtype from the boundary condition / construction). Unknown
# categories fall back to a generated colour.
_IDF_VIEW_COLORS = {
    "Wall: exterior":       "lightyellow",
    "Wall: party":          "mediumpurple",
    "Wall: adiabatic":      "slategray",
    "Wall: ground":         "peru",
    "Wall: interior":       "khaki",
    "Roof":                 "firebrick",
    "Floor: exposed (air)": "teal",
    "Floor: ground":        "dimgray",
    "Floor: interior":      "silver",
    "Ceiling: interior":    "gainsboro",
    "Window":               "cornflowerblue",
    "Door: main":           "saddlebrown",
    "Door: parking":        "sienna",
    "Shading":              "darkolivegreen",
}


def _surface_category(s):
    """Fine-grained category label for an IDF surface — the surface TYPE plus a
    SUBTYPE from its boundary condition (walls: exterior/party-adiabatic/ground/
    interior; floors: exposed-to-air/ground/interior) or construction (doors:
    main/parking). Drives the per-category legend groups in ``plot_idf_geometry``
    so every type and subtype can be toggled independently."""
    st = getattr(s, "Surface_Type", None)
    if st is None:
        return "Shading"                       # Shading:*:Detailed has no type
    st = st.lower()
    bc = (getattr(s, "Outside_Boundary_Condition", "") or "").lower()
    if st == "wall":
        if bc == "adiabatic":
            # Party walls and adiabatic walls share the Adiabatic boundary
            # condition; tell them apart by construction (party → HC_PartyWall).
            # A party wall abuts the neighbour (its outer face sees outdoor air),
            # while an adiabatic wall is a purely internal partition.
            con = (getattr(s, "Construction_Name", "") or "").lower()
            return "Wall: party" if "party" in con else "Wall: adiabatic"
        return "Wall: " + {"outdoors": "exterior", "ground": "ground",
                           "surface": "interior"}.get(bc, "exterior")
    if st == "floor":
        return "Floor: " + {"ground": "ground", "outdoors": "exposed (air)",
                            "surface": "interior"}.get(bc, "interior")
    if st == "roof":
        return "Roof"
    if st == "ceiling":
        return "Ceiling: interior"
    if st == "window":
        return "Window"
    if st == "door":
        con = (getattr(s, "Construction_Name", "") or "").lower()
        return "Door: parking" if "parking" in con else "Door: main"
    return st.title()


def _idf_surfaces_by_category(idf, include_shading=True):
    """Map each fine-grained surface category (``_surface_category``) → list of
    polygons (each a list of (x,y,z) verts), using geomeppy's accessors."""
    from geomeppy import view_geometry as vg
    from collections import OrderedDict
    out = OrderedDict()
    for s in vg._get_surfaces(idf):
        cat = _surface_category(s)
        if cat == "Shading" and not include_shading:
            continue
        coords = vg.getcoords(s)
        if coords and len(coords) >= 3:
            out.setdefault(cat, []).append(
                [(float(x), float(y), float(z)) for x, y, z in coords])
    return out


def plot_idf_geometry(idf, output_path=None, show=False, opacity=0.8,
                      include_shading=True, title=None, shading_opacity=0.2,
                      **_ignored):
    """Render the 3-D geometry of an EnergyPlus IDF — zone walls/roofs/floors,
    windows and neighbour shadings — as an INTERACTIVE plotly scene (rotate /
    zoom / toggle layers), mirroring ``utils.plot_building_3d_geometry``.

    Parameters
    ----------
    idf : a ``geomeppy``/``eppy`` IDF object, or a path to an ``.idf`` file.
    output_path : optional path to save the figure; written as interactive
        ``.html`` (any other extension is coerced to ``.html``).
    show : if True and no ``output_path``, open the figure (``fig.show``).
    include_shading : draw ``Shading:*:Detailed`` neighbour obstructions.
    opacity : fill opacity of the building surfaces (default 0.8, so a surface
        seen through one or more others reads as clearly blurred/dimmed).
    shading_opacity : fill opacity of the neighbour ``Shading`` panels (default
        0.2) — kept well below ``opacity`` so the context obstructions don't
        compete with the building itself.
    title : figure title.

    Returns the plotly ``Figure``.

    Party walls (``Wall: party``) and adiabatic partitions (``Wall: adiabatic``)
    are separate legend groups/colours: both share the Adiabatic boundary
    condition, but a party wall abuts the neighbour building (its outer face sees
    outdoor air) whereas an adiabatic wall is a purely internal partition.
    """
    import plotly.graph_objects as go

    if isinstance(idf, str):
        IDF = _ensure_idd()
        idf = IDF(idf)

    polys_by_cat = _idf_surfaces_by_category(idf, include_shading=include_shading)

    # Stable fallback palette for categories not in _IDF_VIEW_COLORS.
    _fallback = ["#8dd3c7", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
                 "#bc80bd", "#ffed6f", "#fccde5", "#d9d9d9"]
    traces = []
    for ci, (cat, polys) in enumerate(polys_by_cat.items()):
        if not polys:
            continue
        color = _IDF_VIEW_COLORS.get(cat, _fallback[ci % len(_fallback)])
        # ---- filled surfaces: one Mesh3d per category (fan-triangulated) ----
        xs, ys, zs, i, j, k = [], [], [], [], [], []
        ex, ey, ez = [], [], []           # edge polylines (None-separated)
        for poly in polys:
            base = len(xs)
            for (x, y, z) in poly:
                xs.append(x); ys.append(y); zs.append(z)
            for t in range(1, len(poly) - 1):
                i.append(base); j.append(base + t); k.append(base + t + 1)
            for (x, y, z) in poly + [poly[0]]:
                ex.append(x); ey.append(y); ez.append(z)
            ex.append(None); ey.append(None); ez.append(None)
        # Neighbour shading is context — render it much more transparent than the
        # building surfaces so it never competes with them.
        is_shading = cat == "Shading"
        cat_opacity = shading_opacity if is_shading else opacity
        traces.append(go.Mesh3d(
            x=xs, y=ys, z=zs, i=i, j=j, k=k,
            color=color, opacity=cat_opacity, flatshading=True,
            name=cat, legendgroup=cat, showlegend=True,
            hoverinfo="name"))
        # ---- crisp edges: one Scatter3d per category, shares its legend group -
        edge_col = "rgba(80,80,80,0.25)" if is_shading else "rgba(40,40,40,0.6)"
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=edge_col, width=1 if is_shading else 2),
            legendgroup=cat, showlegend=False, hoverinfo="skip"))

    fig = go.Figure(data=traces)
    # True 1:1:1 metre scale (aspectmode='data') like plot_building_3d_geometry.
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x (m)"),
            yaxis=dict(title="y (m)"),
            zaxis=dict(title="Height (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        legend=dict(title="Surface", x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=40, t=40),
    )

    if output_path:
        ext = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else ""
        if ext != "html":
            output_path = (output_path.rsplit(".", 1)[0] + ".html"
                           if "." in output_path else output_path + ".html")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path)
        print(f"IDF geometry plot saved to {output_path}")
    elif show:
        fig.show()
    return fig


# =========================================================================== #
# Occupant-behaviour SCENARIO ANALYSIS  → EnergyPlus runs → per-use-type KPIs
# =========================================================================== #
# Each scenario builds (and runs) the building's EnergyPlus model under a
# specific occupant-behaviour assumption, and — for the conditioned (HVAC)
# scenarios — a paired FREE-FLOATING twin (same fabric/occupancy/ventilation, no
# HVAC) so the "comfort without HVAC" KPIs are available for every scenario.
#
# Comfort uses the EN 16798-1 ADAPTIVE model (Category II): the operative
# temperature is "comfortable" when it lies within ±3 K of the comfort
# temperature Tc = 0.33·θrm + 18.8, where θrm is the running-mean outdoor
# temperature (α = 0.8). θrm is clamped to [10, 30] °C (the model's validity
# range) so heating-season hours keep the θrm=10 band.
_ADAPTIVE_ALPHA = 0.8
_ADAPTIVE_CAT_BAND = {"I": 2.0, "II": 3.0, "III": 4.0}
_KWH_PER_J = 1.0 / 3.6e6
# NIGHT = 22:00–07:59, DAY = 08:00–21:59. In the E+ hourly CSV the label is the
# END hour of the interval, so 22:00–23:00→23 … 07:00–08:00→8 are the night hours.
_NIGHT_HOURS = {23, 24, 1, 2, 3, 4, 5, 6, 7, 8}

# Indoor-temperature percentiles reported per season, over the OCCUPIED hours.
# The spread is what a mean hides: two dwellings can share an average and differ
# entirely in how often they sit at the extremes.
_TEMPERATURE_PERCENTILES = (10, 20, 50, 80, 90)

# …reported alongside the extremes. The percentiles describe the bulk of the
# hours; min and max are what a compliance argument actually turns on — the one
# freezing night or the one overheated afternoon never reaches p10/p90.
_TEMPERATURE_EXTREMES = ("min", "max")

# Indoor CO2 percentiles reported per season over the OCCUPIED hours. Fewer steps
# than the temperature ones: what matters is the typical level and the peak, not
# the shape of the low tail (an empty dwelling always decays to outdoor).
_CO2_PERCENTILES = (50, 90)
# Air-change percentiles, reported like the temperature ones. The TOTAL rate is
# the headline (it is what renews the air, whatever route it came by); the two
# routes are reported as means beside it, since what matters about them is the
# split, not the shape of their distributions.
#
# NO MEDIAN (p50) here, deliberately, and none anywhere else on the outdoor-air
# rates: an air change rate is a FLOW, and what a dwelling actually renews over a
# season is the sum of its hourly flows — the mean. The median of a series that is
# 0 for twenty hours and 4 ACH for one is 0, which reads as a dwelling that never
# opens a window and is wrong about the only thing the number is asked. The centre
# of these distributions is therefore the mean (``total_mean``); the percentiles
# stay to show the shape around it.
_ACH_PERCENTILES = (10, 20, 80, 90)
# Seasons are defined per DAY by the daily-mean outdoor temperature: the IN-BETWEEN
# season (no HVAC for any user) is any day whose mean is within this band; below →
# HEATING, above → COOLING. 16–24 °C is the range over which a free-running dwelling
# (helped by internal gains) sits closest to the EN 16798-1 adaptive comfort band;
# colder days clearly need heating and warmer ones cooling. (Month sets are a
# fallback when no weather is available.)
_INBETWEEN_TEMP_LO = 16.0
_INBETWEEN_TEMP_HI = 24.0
# Season classification is smoothed over a rolling window of DAYS so a single warm/
# cold day doesn't flip the season — seasons stay continuous. A day's season is
# decided from the daily mean outdoor temperature averaged over itself and the
# ±3 days around it, i.e. a centred WEEK. A week is the shortest window that
# tracks a real change of season without following weather: a 3-day mean still
# turns a warm spell in March into "cooling" for a few days, and the season a
# result is reported under then depends on which days happened to be mild.
# Configurable as ``seasons.smoothing_days``.
_SEASON_SMOOTH_DAYS = 7
_HEATING_SEASON_MONTHS = {12, 1, 2, 3}
_COOLING_SEASON_MONTHS = {6, 7, 8, 9}
_INBETWEEN_SEASON_MONTHS = {4, 5, 10, 11}
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)   # non-leap


def _season_of_temp(tmean, lo=None, hi=None):
    """Season of a day from its (smoothed) mean outdoor temperature (°C)."""
    if lo is None:
        lo = float(globals().get("_INBETWEEN_TEMP_LO", 16.0))
    if hi is None:
        hi = float(globals().get("_INBETWEEN_TEMP_HI", 24.0))
    if tmean < lo:
        return "heating"
    if tmean > hi:
        return "cooling"
    return "in_between"


def _smooth_days(values, window=None):
    """Centered rolling mean of a daily series (a list of per-day values) over a
    ``window``-day window — so the season classification only changes when the
    temperature TREND does, not on a single anomalous day (edges use fewer days).
    ``None`` takes the configured window (:data:`_SEASON_SMOOTH_DAYS`, a week)."""
    if window is None:
        window = int(globals().get("_SEASON_SMOOTH_DAYS", 7))
    import numpy as _np
    v = _np.asarray(values, dtype=float)
    n, half = len(v), window // 2
    out = _np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = v[lo:hi].mean()
    return out


def _calendar_day_index(df):
    """Chronological calendar-day index (0,0,…,1,1,…) for each row, incrementing on
    each (month, day) change. Falls back to ``index // 24`` when the frame has no
    month/day columns (bare test frames)."""
    import numpy as _np
    if "month" not in df or "day" not in df:
        return (df.index // 24).values if hasattr(df.index, "values") else \
            _np.arange(len(df)) // 24
    md = (df["month"].astype(int).values * 100 + df["day"].astype(int).values)
    if len(md) == 0:
        return _np.zeros(0, int)
    return _np.concatenate([[0], (md[1:] != md[:-1]).astype(int)]).cumsum()


def _doy_to_mmdd(doy):
    """(month, day) for a 1-based day-of-year on a non-leap calendar."""
    m = 0
    while doy > _MONTH_DAYS[m]:
        doy -= _MONTH_DAYS[m]
        m += 1
    return m + 1, doy


def _epw_daily_mean_temps(epw_path):
    """{(month, day): mean dry-bulb °C} for each day in an EPW file, or None."""
    if not epw_path or not os.path.exists(epw_path):
        return None
    days = {}
    try:
        with open(epw_path) as f:
            for _ in range(8):                          # skip the 8 header lines
                f.readline()
            for line in f:
                p = line.split(",")
                if len(p) < 7:
                    continue
                try:
                    key = (int(p[1]), int(p[2]))
                    days.setdefault(key, []).append(float(p[6]))
                except ValueError:
                    continue
    except OSError:
        return None
    return {k: sum(v) / len(v) for k, v in days.items() if v} or None


def _season_by_doy(epw_path):
    """{doy(1-365): season} from the EPW daily-mean temperatures, SMOOTHED over a
    3-day window so seasons stay continuous, or None."""
    dm = _epw_daily_mean_temps(epw_path)
    if not dm:
        return None
    order = [dm.get((m, d), 15.0) for m in range(1, 13)
             for d in range(1, _MONTH_DAYS[m - 1] + 1)]
    smoothed = _smooth_days(order)
    return {doy: _season_of_temp(t) for doy, t in enumerate(smoothed, 1)}


def _seasonal_day_schedule(idf, name, season_by_doy, profile_by_season, limits="Fraction"):
    """Build a Schedule:Compact where each DAY follows the day-profile for its
    (temperature-derived) season — consecutive same-season days grouped into
    ``Through`` ranges."""
    if name.upper() in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        return name
    off = ["For: AllDays", "Until: 24:00,0"]
    fields, doy = [], 1
    while doy <= 365:
        cur = season_by_doy.get(doy, "in_between")
        end = doy
        while end < 365 and season_by_doy.get(end + 1, "in_between") == cur:
            end += 1
        mm, dd = _doy_to_mmdd(end)
        fields.append(f"Through: {mm}/{dd}")
        fields += profile_by_season.get(cur, off)
        doy = end + 1
    _sched(idf, name, limits, fields)
    return name
# Only these use categories are conditioned (have HVAC) and enter the per-use-type
# KPI aggregation. Any other use (singular/cultural/religious/unknown, parking,
# agricultural, communal, sports…) is NOT thermally modelled with HVAC — and a
# building whose dominant use is none of these has no thermal energy model at all.
_OCCUPIED_USE_CATEGORIES = ("residential", "office", "commercial")
_MODELABLE_USE_CATEGORIES = ("residential", "office", "commercial")

# The default six scenarios. A scenario is an ARCHETYPE ASSIGNMENT and nothing
# else — everything that follows from it is authored by the one occupant engine
# (``_wire_occupant_model``), so two scenarios differ only where their spec does.
#
#   behavior           fixed archetype for every dwelling, or None for the
#                      demographic-weighted draw
#   free_floating      strip the HVAC → the unconditioned twin of its parent
#   mix_aware_unaware  each dwelling independently aware or unaware, 50/50
#   residential_only   skip the scenario for a building with no dwellings
#
# ``random_ventilation``, ``random_hvac_use`` and ``seasonal_setpoints`` used to
# live here; nothing has read them since the behaviour model was unified, and the
# stochastic ventilation they named is now part of the unaware archetype itself.
# ``period`` is gone with the per-dwelling occupancy period (see
# ``_assign_residential_user``).
_DEFAULT_SCENARIOS = {
    "random_mix":    {"behavior": None,      "free_floating": False,
                      "mix_aware_unaware": True, "residential_only": True},
    "unaware":       {"behavior": "unaware", "free_floating": False},
    "unaware_ff":    {"behavior": "unaware", "free_floating": True},
    "aware_ff":      {"behavior": "aware",   "free_floating": True},
    "aware":         {"behavior": "aware",   "free_floating": False},
    "cte":           {"behavior": "cte",     "free_floating": False},
}
_WINTER_MONTHS = {1, 2, 3, 4, 10, 11, 12}     # cooling capped tighter (19–22 band)


# Infiltration multiplier by dwelling conservation status (worse conservation →
# leakier envelope). Keyed on the Essential-Characteristics conservation
# proportions from social_ES; absent → factor 1.0 (default envelope).
_CONSERVATION_INFIL_FACTOR = {"excellent": 0.85, "very_good": 0.9, "good": 1.0,
                              "normal": 1.1, "fair": 1.3, "bad": 1.6}

# The column the social layer emits for a conservation class — social_ES joins the
# class to its variable with "~", and those names are now kept verbatim. The two
# underscore forms are the pre-1.2.4 spellings, still read so a frame merged by an
# older release (or one carrying only the short name) keeps driving the envelope.
_CONSERVATION_COLUMNS = ("ec__main_dwellings_conservation_status~{}",
                         "ec__main_dwellings_conservation_status_{}",
                         "ec__conservation_{}")


def _conservation_proportion(row, status):
    """Proportion (%) of the building's main dwellings in conservation ``status``
    (social_ES Essential Characteristics), or None when not published."""
    if not hasattr(row, "get"):
        return None
    for template in _CONSERVATION_COLUMNS:
        try:
            p = float(row.get(template.format(status)))
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and p >= 0:
            return p
    return None


def _conservation_infiltration_factor(row):
    """Envelope infiltration multiplier from the building's dwelling conservation
    status (``ec__main_dwellings_conservation_status~{excellent…bad}`` proportion
    columns, social_ES Essential Characteristics). Proportion-weighted; returns
    1.0 when the columns are absent (default envelope)."""
    num, den = 0.0, 0.0
    for status, fac in _CONSERVATION_INFIL_FACTOR.items():
        p = _conservation_proportion(row, status)
        if p is not None:
            num += fac * p
            den += p
    return (num / den) if den > 1e-9 else 1.0


def _apply_infiltration_factor(idf, factor):
    """Scale every zone's design infiltration air-changes by ``factor`` (no-op at
    1.0). Models the conservation-status effect on envelope air-tightness."""
    if abs(factor - 1.0) < 1e-6:
        return
    for o in idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", []):
        try:
            o.Air_Changes_per_Hour = float(o.Air_Changes_per_Hour) * factor
        except (TypeError, ValueError):
            pass


def _strip_hvac(idf):
    """Make ``idf`` FREE-FLOATING: remove the ideal-loads HVAC templates (and
    their thermostats) so EnergyPlus reports the unconditioned zone temperatures.
    Natural ventilation, loads and schedules are kept."""
    for key in ("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", "HVACTEMPLATE:THERMOSTAT"):
        for o in list(idf.idfobjects.get(key, [])):
            idf.removeidfobject(o)


# Name prefix of the NEIGHBOUR (context) shading panels, the only ones the filter
# and the cap may touch. Everything else on the model is the building's OWN
# shading -- ``CatShade_*`` footprint remainders, ``Balcony_*``, ``Awn_*``,
# ``Blind_*`` and ``*_rev*`` window reveals -- and is part of the building, not
# context detail to be budgeted away.
_NEIGHBOUR_SHADE_PREFIX = "Shade_F"


def _neighbour_shades(idf):
    """The neighbour (context) shading panels of ``idf``."""
    return [o for o in idf.idfobjects.get("SHADING:BUILDING:DETAILED", [])
            if str(getattr(o, "Name", "")).startswith(_NEIGHBOUR_SHADE_PREFIX)]


def _apply_shading_filter(idf, filter_shading=None, max_shading=_FROM_CONFIG,
                          label="", verbose=False):
    """Filter and cap the NEIGHBOUR-shading panels of ``idf``, once.

    ``filter_shading`` / ``max_shading`` default to ``shading.neighbours.filter``
    and ``.max_panels`` in the config (``None`` for the latter genuinely means "no
    cap", hence the ``_FROM_CONFIG`` sentinel). Sets ``idf._hc_shading_filtered``
    so a second caller — ``run_behavior_scenarios`` on an IDF the generator already
    filtered — skips the work instead of re-running an occlusion cull that would
    remove nothing. Returns ``(n_before, n_after)``.
    """
    if getattr(idf, "_hc_shading_filtered", False):
        n = len(_neighbour_shades(idf))
        return (n, n)
    if filter_shading is None:
        filter_shading = globals().get("_FILTER_SHADING", True)
    if max_shading is _FROM_CONFIG:
        max_shading = globals().get("_MAX_SHADING", 200)
    nb = globals().get("_SHADE_NEIGHBOURS", {}) or {}

    n0 = len(_neighbour_shades(idf))
    if filter_shading:
        _filter_relevant_shading(
            idf,
            sun_exclude_halfwidth=float(nb.get("sun_exclude_halfwidth", 67.5)),
            occlusion=bool(nb.get("occlusion", True)))
    _trim_shading(idf, max_shading)
    n1 = len(_neighbour_shades(idf))
    try:
        idf._hc_shading_filtered = True
    except Exception:  # noqa: BLE001 - marking is an optimisation, not a contract
        pass
    if verbose and n1 != n0:
        print(f"    shading panels {n0} → {n1}"
              + (f"  [{label}]" if label else ""))
    return (n0, n1)


def _trim_shading(idf, keep, name_prefix=_NEIGHBOUR_SHADE_PREFIX):
    """Cap the number of NEIGHBOUR shading panels at ``keep`` (the largest by
    bounding-box area) — neighbour-shading detail dominates E+ runtime, and the
    scenario KPIs are compared under the SAME (trimmed) context. ``None`` keeps
    them all.

    Only ``name_prefix`` panels are eligible, and the cap counts only those: the
    building's OWN shading — balconies, awnings, exterior blinds, window reveals,
    footprint remainders — is part of the model, not context detail. Trimming by
    size across everything removed the blinds and awnings first (they are the
    smallest panels there are), which silently disabled the whole operable-shading
    behaviour: with no ``Blind_``/``Awn_`` objects left,
    ``_apply_occupant_control`` had nothing to deploy and every occupant type
    returned identical results.
    """
    if not keep:
        return
    shades = [o for o in idf.idfobjects.get("SHADING:BUILDING:DETAILED", [])
              if str(getattr(o, "Name", "")).startswith(name_prefix)]
    if len(shades) <= keep:
        return
    from geomeppy import view_geometry as vg

    def _bbox_area(o):
        try:
            cs = vg.getcoords(o)
            xs = [c[0] for c in cs]; ys = [c[1] for c in cs]; zs = [c[2] for c in cs]
            return (max(xs) - min(xs) + 1e-6) * (max(zs) - min(zs) + max(ys) - min(ys) + 1e-6)
        except Exception:
            return 0.0
    shades.sort(key=_bbox_area, reverse=True)
    for o in shades[keep:]:
        idf.removeidfobject(o)


def _shade_irrelevant(coords, cx, cy, min_bz, sun_exclude_halfwidth=67.5):
    """Whether a shading panel (``coords`` = list of (x,y,z)) cannot affect a
    building whose centroid is ``(cx, cy)`` and whose lowest surface point is
    ``min_bz``: True if it sits at/below the building's horizon everywhere, OR if
    its bearing from the centroid is within ``sun_exclude_halfwidth`` of due NORTH
    (no direct sun from the N/NE/NW in the northern hemisphere)."""
    if not coords:
        return True
    if max(c[2] for c in coords) <= min_bz + 1e-6:          # below the horizon
        return True
    scx = sum(c[0] for c in coords) / len(coords)
    scy = sum(c[1] for c in coords) / len(coords)
    bearing = math.degrees(math.atan2(scx - cx, scy - cy)) % 360.0   # 0=N, 90=E
    return min(bearing, 360.0 - bearing) <= sun_exclude_halfwidth


def _shade_angular_box(coords, cx, cy, cz):
    """Angular footprint of a shading panel as SEEN FROM the vantage ``(cx,cy,cz)``:
    ``(distance, az_min, az_max, el_min, el_max)`` in degrees (azimuth 0=N/90=E,
    elevation from the horizontal). Distance is the horizontal range to its
    centroid (nearer shades occlude farther ones)."""
    azs, els = [], []
    for x, y, z in coords:
        hd = math.hypot(x - cx, y - cy)
        azs.append(math.degrees(math.atan2(x - cx, y - cy)) % 360.0)
        els.append(math.degrees(math.atan2(z - cz, hd)) if hd > 1e-6
                   else (90.0 if z > cz else -90.0))
    scx = sum(c[0] for c in coords) / len(coords)
    scy = sum(c[1] for c in coords) / len(coords)
    return (math.hypot(scx - cx, scy - cy),
            min(azs), max(azs), min(els), max(els))


def _occlusion_cull(boxes, az_bin=1.0, el_bin=1.0):
    """Indices of the angular ``boxes`` (from ``_shade_angular_box``) to KEEP after
    view-based occlusion culling: sweeping from NEAREST to farthest, a panel is
    dropped when its whole azimuth×elevation footprint is already covered by nearer
    panels (so the building can't see any of it → it adds no shadow)."""
    import numpy as _np
    if not boxes:
        return set()
    az_lo = min(b[1] for b in boxes)
    el_lo = min(b[3] for b in boxes)
    naz = max(1, int(math.ceil((max(b[2] for b in boxes) - az_lo) / az_bin)) + 1)
    nel = max(1, int(math.ceil((max(b[4] for b in boxes) - el_lo) / el_bin)) + 1)
    covered = _np.zeros((naz, nel), dtype=bool)
    keep = set()
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][0]):   # nearest first
        _d, a0, a1, e0, e1 = boxes[i]
        ia0 = max(0, int((a0 - az_lo) / az_bin))
        ia1 = min(naz, max(int(math.ceil((a1 - az_lo) / az_bin)), ia0 + 1))
        ie0 = max(0, int((e0 - el_lo) / el_bin))
        ie1 = min(nel, max(int(math.ceil((e1 - el_lo) / el_bin)), ie0 + 1))
        sub = covered[ia0:ia1, ie0:ie1]
        if sub.all():
            continue                                # fully hidden behind nearer shades
        keep.add(i)
        covered[ia0:ia1, ie0:ie1] = True
    return keep


def _filter_relevant_shading(idf, sun_exclude_halfwidth=67.5,
                             name_prefix=_NEIGHBOUR_SHADE_PREFIX,
                             occlusion=True):
    """Drop NEIGHBOUR-shading panels (``Shade_F*`` from ``_add_shading``) that
    cannot shade the building being simulated — keeping the building's own self-
    shading (``CatShade_*`` / ``*_rev*``). Tests, using the building's surfaces and
    centroid:

    1. **Not visible above any horizon** — a panel whose highest point is at or
       below the building's LOWEST surface point sits below it from every vantage,
       so it can never cast a shadow onto it.
    2. **Northern sector (NE/N/NW)** — in the northern hemisphere (Spain) the sun is
       always in the southern sky, so an obstruction whose bearing from the building
       centroid is within ``sun_exclude_halfwidth`` (67.5°) of due NORTH never
       blocks direct beam radiation.
    3. **Occluded by a nearer panel** (``occlusion``) — a panel entirely hidden
       behind nearer panels as seen from the building adds no shadow of its own
       (``_occlusion_cull``, view-based from the building centroid).

    Returns the number of panels removed."""
    from geomeppy import view_geometry as vg
    xs, ys, zs = [], [], []
    for s in idf.idfobjects.get("BUILDINGSURFACE:DETAILED", []):
        try:
            for x, y, z in vg.getcoords(s):
                xs.append(x); ys.append(y); zs.append(z)
        except Exception:
            continue
    if not xs:
        return 0
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    min_bz, cz = min(zs), sum(zs) / len(zs)
    removed, survivors = 0, []
    for o in list(idf.idfobjects.get("SHADING:BUILDING:DETAILED", [])):
        if not str(getattr(o, "Name", "")).startswith(name_prefix):
            continue                               # keep self-shading (CatShade/reveals)
        try:
            cs = vg.getcoords(o)
        except Exception:
            continue
        if not cs:
            continue
        if _shade_irrelevant(cs, cx, cy, min_bz, sun_exclude_halfwidth):
            idf.removeidfobject(o); removed += 1
        else:
            survivors.append((o, cs))
    if occlusion and survivors:
        # A panel is occluded only if it is hidden from EVERY part of the building
        # ("from any part…"), so test several vantages — the centroid plus the
        # footprint bounding-box corners — and KEEP a panel visible from any one.
        vantages = [(cx, cy, cz),
                    (min(xs), min(ys), cz), (min(xs), max(ys), cz),
                    (max(xs), min(ys), cz), (max(xs), max(ys), cz)]
        coords_list = [cs for _o, cs in survivors]
        keep = set()
        for vx, vy, vz in vantages:
            keep |= _occlusion_cull([_shade_angular_box(cs, vx, vy, vz)
                                     for cs in coords_list])
        for i, (o, _cs) in enumerate(survivors):
            if i not in keep:
                idf.removeidfobject(o); removed += 1
    return removed


_WEEKDAY_NAMES_EPLUS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday")


def _epw_year(idf):
    """The calendar year of the attached EPW, from its filename, or ``None``.

    The ERA5-Land EPWs this model builds are named ``…_{lat}_{lon}_{year}.epw``,
    so the year the weather actually belongs to is recoverable without parsing the
    file."""
    path = str(getattr(idf, "epw", "") or "")
    years = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", os.path.basename(path))
    return int(years[-1]) if years else None


def _set_run_period(idf, months, year=None):
    """Set the RunPeriod's calendar, including WHICH DAY OF THE WEEK it starts on.

    ``months`` (1-based, contiguous) restricts the run for a fast partial-year
    check; ``None`` keeps the full year.

    The start weekday matters more than it looks. Left blank, EnergyPlus assumes
    1 January is a **Sunday**, so every ``For: Saturday``/``For: Sunday`` block —
    the Time Use Survey's weekly occupancy agenda above all — lands on the wrong
    calendar day. With 2021 weather (1 January was a Friday) the whole week ran
    two days out, which showed up as the household's weekend occupancy peak
    appearing on a Thursday. It is set here from the weather year.
    """
    rp = idf.idfobjects.get("RUNPERIOD", [])
    if not rp:
        return
    year = year or _epw_year(idf)
    if year:
        import datetime as _dt
        start_month = min(months) if months else 1
        try:
            wd = _dt.date(int(year), int(start_month), 1).weekday()
            rp[0].Day_of_Week_for_Start_Day = _WEEKDAY_NAMES_EPLUS[wd]
        except (ValueError, TypeError):
            pass
    if not months:
        return
    lo, hi = min(months), max(months)
    rp[0].Begin_Month = lo
    rp[0].Begin_Day_of_Month = 1
    rp[0].End_Month = hi
    rp[0].End_Day_of_Month = 28 if hi == 2 else (30 if hi in (4, 6, 9, 11) else 31)


# Transmitted-solar output variable, by EnergyPlus generation. Radiant enclosures
# were split from thermal zones in 23.1, which renamed the zone-level window solar
# gain; asking for the wrong one silently reports nothing (a warning in
# eplusout.err and no column in the CSV), so the name is chosen from the IDD.
_SOLAR_GAIN_VARIABLE_MODERN = "Enclosure Windows Total Transmitted Solar Radiation Energy"
_SOLAR_GAIN_VARIABLE_LEGACY = "Zone Windows Total Transmitted Solar Radiation Energy"
_SOLAR_ENCLOSURE_RENAMED_IN = (23, 1)


def _solar_gain_variable(idf=None):
    """The transmitted-solar output-variable name this EnergyPlus understands."""
    version = getattr(idf, "idd_version", None)
    try:
        if tuple(version[:2]) < _SOLAR_ENCLOSURE_RENAMED_IN:
            return _SOLAR_GAIN_VARIABLE_LEGACY
    except (TypeError, IndexError):
        pass
    return _SOLAR_GAIN_VARIABLE_MODERN


def _add_scenario_outputs(idf):
    """Request the hourly variables the scenario KPIs need — operative
    temperature, ideal-loads heating/cooling energy, total internal gains,
    transmitted solar gains, occupant count and the two thermostat setpoints per
    zone, plus outdoor dry-bulb — dropping the model's default (coarser) output
    set so the CSV stays lean."""
    for o in list(idf.idfobjects.get("OUTPUT:VARIABLE", [])):
        idf.removeidfobject(o)
    per_zone = ["Zone Operative Temperature",
                "Zone Ideal Loads Supply Air Total Heating Energy",
                "Zone Ideal Loads Supply Air Total Cooling Energy",
                "Zone Total Internal Total Heating Energy",
                # Solar entering through the glazing — reported separately from the
                # internal gains, which are the people/lights/equipment ones only.
                # EnergyPlus renamed this from "Zone …" to "Enclosure …" when
                # radiant enclosures were split from zones (23.1); request the name
                # this IDD knows, so the other one does not warn in eplusout.err.
                _solar_gain_variable(idf),
                "Zone People Occupant Count",
                # The setpoints actually in force each hour: the scenarios move
                # them (adaptive band, income offset, setback, CTE), so the figure
                # a run was driven by is not readable from the config alone.
                "Zone Thermostat Heating Setpoint Temperature",
                "Zone Thermostat Cooling Setpoint Temperature",
                # Indoor CO2 — how renewed the air is. Occupants raise it, leakage
                # and open windows flush it.
                "Zone Air CO2 Concentration",
                # The other half of the indoor state: comfort is a point in
                # (temperature, humidity), not a temperature. This is the
                # psychrometric ordinate the report's comfort chart plots the
                # operative temperature against.
                "Zone Mean Air Humidity Ratio",
                # …and the outdoor air doing the flushing, split by route. Both
                # ZoneInfiltration and ZoneVentilation draw OUTDOOR air, so both
                # enter the contaminant balance at the outdoor concentration —
                # these say how much of it each scenario actually gets.
                "Zone Infiltration Standard Density Air Change Rate",
                "Zone Ventilation Standard Density Air Change Rate"]
    for var in per_zone:
        idf.newidfobject("OUTPUT:VARIABLE", Key_Value="*", Variable_Name=var,
                         Reporting_Frequency="Hourly")
    idf.newidfobject("OUTPUT:VARIABLE", Key_Value="Environment",
                     Variable_Name="Site Outdoor Air Drybulb Temperature",
                     Reporting_Frequency="Hourly")


def _zone_floor_areas(idf):
    """{ZONE_NAME (upper): floor area m²} from the zone's Floor surfaces."""
    from geomeppy import view_geometry as vg
    out = {}
    for s in idf.idfobjects.get("BUILDINGSURFACE:DETAILED", []):
        if str(getattr(s, "Surface_Type", "")).lower() != "floor":
            continue
        try:
            cs = vg.getcoords(s)
            a = abs(sum(cs[i][0] * cs[(i + 1) % len(cs)][1]
                        - cs[(i + 1) % len(cs)][0] * cs[i][1]
                        for i in range(len(cs)))) / 2.0
        except Exception:
            a = 0.0
        z = str(getattr(s, "Zone_Name", "")).upper()
        out[z] = out.get(z, 0.0) + a
    return out


def _zone_volumes(idf):
    """{ZONE_NAME (upper): volume m³} — floor area × the zone's z extent. The zones
    are prismatic (one storey of one dwelling), so the product is exact."""
    areas = _zone_floor_areas(idf)
    span = {}
    for s in idf.idfobjects.get("BUILDINGSURFACE:DETAILED", []):
        z = str(getattr(s, "Zone_Name", "")).upper()
        if not z:
            continue
        try:
            zs = [float(c[2]) for c in s.coords]
        except Exception:
            continue
        lo, hi = span.get(z, (min(zs), max(zs)))
        span[z] = (min(lo, min(zs)), max(hi, max(zs)))
    return {z: a * max(0.0, span.get(z, (0.0, 0.0))[1] - span.get(z, (0.0, 0.0))[0])
            for z, a in areas.items()}


def _sched_compact(idf, name, limits, fields):
    if name.upper() not in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        _sched(idf, name, limits, ["Through: 12/31"] + fields)
    return name


def _running_mean_by_doy(epw_path, alpha=0.8):
    """EN 16798-1 running-mean outdoor temperature θrm per day-of-year, from the
    EPW daily means: θrm(d) = (1−α)·θed−1 + α·θrm(d−1), α=0.8. Returns {doy: θrm}."""
    dm = _epw_daily_mean_temps(epw_path)
    if not dm:
        return None
    order = [dm.get((m, d)) for m in range(1, 13)
             for d in range(1, _MONTH_DAYS[m - 1] + 1)]
    prev = next((t for t in order if t is not None), 15.0)
    out = {}
    for i, t in enumerate(order, 1):
        if t is None:
            t = prev
        prev = (1 - alpha) * t + alpha * prev
        out[i] = prev
    return out


def _daily_value_schedule(idf, name, doy_values, limits="Temperature"):
    """Schedule:Compact of one constant value per day, grouping consecutive
    equal-valued days into ``Through`` ranges."""
    if name.upper() in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        return name
    fields, doy = [], 1
    while doy <= 365:
        v = doy_values[doy]
        end = doy
        while end < 365 and doy_values[end + 1] == v:
            end += 1
        mm, dd = _doy_to_mmdd(end)
        fields += [f"Through: {mm}/{dd}", "For: AllDays", f"Until: 24:00,{v}"]
        doy = end + 1
    _sched(idf, name, limits, fields)
    return name


# Residential occupancy-aware setpoint day blocks: the (adaptive) setpoint applies
# while the dwelling is occupied and sets BACK to ``setback`` when it is empty —
# weekdays 09:00–14:00 and 16:00–19:00, mirroring _OCCUPANCY_PROFILES["residential"];
# weekends/holidays stay occupied all day.
def _spans_subtract(spans, cut):
    """``spans`` minus ``cut`` — both lists of ``(start, end)`` minute spans."""
    out = []
    for lo, hi in _merged_spans(list(spans)):
        pieces = [(lo, hi)]
        for clo, chi in _merged_spans(list(cut)):
            nxt = []
            for a, b in pieces:
                if chi <= a or clo >= b:
                    nxt.append((a, b))
                    continue
                if a < clo:
                    nxt.append((a, clo))
                if chi < b:
                    nxt.append((chi, b))
            pieces = nxt
        out += pieces
    return _merged_spans(out)


# Schedule:Compact day types, in the order EnergyPlus names them. The unaware
# night opening picks whole nights of the WEEK out of this list: a per-day random
# subset would need a Through block for each of the 365 days, in every dwelling of
# every building, and the whole point of the number is the SHARE of nights open,
# which a weekly pattern delivers at a fraction of the schedule size.
_WEEK_DAY_TYPES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday")


def _minutes_to_hhmm(m):
    """Minutes past midnight → the ``"HH:MM"`` a Schedule:Compact ``Until:`` wants."""
    m = max(0, min(24 * 60, int(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


def _merged_spans(spans):
    """Sorted ``(start, end)`` minute spans with the overlapping ones merged, so a
    window profile built from them can never step backwards in time."""
    out = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def _pair(value, default):
    """A config ``[lo, hi]`` range, also accepting a bare scalar as ``[v, v]``."""
    try:
        if isinstance(value, (list, tuple)):
            lo, hi = float(value[0]), float(value[-1])
        else:
            lo = hi = float(value)
    except (TypeError, ValueError, IndexError):
        lo, hi = float(default[0]), float(default[-1])
    return (min(lo, hi), max(lo, hi))


def _timestep_minutes():
    """Length of one simulation timestep, in minutes. A Schedule:Compact ``Until:``
    between two timesteps is rounded by EnergyPlus, so every airing this module
    writes is snapped to this grid — the config then says what actually runs."""
    try:
        n = int((globals().get("_SIM", {}) or {}).get("timestep_per_hour", 4))
    except (TypeError, ValueError):
        n = 4
    return max(1, int(round(60.0 / max(1, n))))


def _weekly_night_days(rng, fraction):
    """``fraction`` of the nights of a week, as whole day-type names drawn from
    ``rng``.

    Randomised rounding: a dwelling gets ``floor(f·7)`` nights, plus one more with
    probability equal to the remainder — so each dwelling gets a whole number of
    nights while the MEAN across the building is exactly the configured share.
    Whole nights of the WEEK rather than of the year keep the schedule small: a
    per-day subset needs a ``Through:`` block for each of the 365 days, in every
    dwelling of every building.
    """
    f = max(0.0, min(1.0, float(fraction or 0.0)))
    if f <= 0:
        return []
    exact = f * len(_WEEK_DAY_TYPES)
    k = int(exact) + (1 if rng.random() < (exact - int(exact)) else 0)
    k = max(0, min(len(_WEEK_DAY_TYPES), k))
    if k == 0:
        return []
    return sorted(rng.sample(_WEEK_DAY_TYPES, k), key=_WEEK_DAY_TYPES.index)


def _split_day_blocks(fields):
    """A Schedule:Compact day profile → ``{day_type: [Until lines]}`` over the
    seven weekdays plus ``AllOtherDays``, expanding the collective selectors
    (``AllDays``, ``Weekdays``, ``Weekends``, ``Holiday``, ``AllOtherDays``) so a
    single day can be rewritten without disturbing the rest."""
    groups = {"ALLDAYS": list(_WEEK_DAY_TYPES) + ["AllOtherDays"],
              "WEEKDAYS": list(_WEEK_DAY_TYPES[:5]),
              "WEEKENDS": list(_WEEK_DAY_TYPES[5:]),
              "WEEKEND": list(_WEEK_DAY_TYPES[5:]),
              "HOLIDAY": ["AllOtherDays"], "HOLIDAYS": ["AllOtherDays"],
              "ALLOTHERDAYS": ["AllOtherDays"]}
    out, block, targets = {}, None, []
    for f in fields:
        text = str(f)
        if text.upper().startswith("FOR:"):
            block, targets = [], []
            for token in text.split(":", 1)[1].replace(",", " ").split():
                targets += groups.get(token.upper(), [token.capitalize()])
            for t in targets:
                out.setdefault(t, block)         # the SAME list, shared
        elif block is not None:
            block.append(text)
    # Anything the profile never named repeats the last-resort block.
    fallback = out.get("AllOtherDays") or next(iter(out.values()), ["Until: 24:00,0"])
    for day in list(_WEEK_DAY_TYPES) + ["AllOtherDays"]:
        out.setdefault(day, fallback)
    return {d: list(v) for d, v in out.items()}


def _join_day_blocks(by_day):
    """The inverse of ``_split_day_blocks``: day types sharing a profile are
    regrouped into one ``For:`` block, ``AllOtherDays`` last."""
    fields, seen = [], []
    for day in _WEEK_DAY_TYPES:
        prof = by_day[day]
        for names, p in seen:
            if p == prof:
                names.append(day)
                break
        else:
            seen.append(([day], prof))
    for names, prof in seen:
        fields += [f"For: {' '.join(names)}"] + list(prof)
    return fields + ["For: AllOtherDays"] + list(by_day["AllOtherDays"])


def _spans_intersect(a, b):
    """The overlap of two lists of ``(start, end)`` minute spans."""
    out = []
    for lo, hi in _merged_spans(list(a)):
        for clo, chi in _merged_spans(list(b)):
            s, e = max(lo, clo), min(hi, chi)
            if e > s:
                out.append((s, e))
    return _merged_spans(out)


def _periods_to_spans(periods):
    """``[["HH:MM", "HH:MM"], …]`` → ``(start, end)`` minute spans."""
    out = []
    for a, b in (periods or []):
        try:
            lo = int(str(a).split(":")[0]) * 60 + int(str(a).split(":")[1])
            hi = int(str(b).split(":")[0]) * 60 + int(str(b).split(":")[1])
        except (TypeError, ValueError, IndexError):
            continue
        out.append((lo, hi))
    return _merged_spans(out)


def _profile_positive_spans(fields):
    """A Schedule:Compact day profile → ``{day_type: [(start, end)]}``, the minutes
    it holds a value above zero. Used to read an OCCUPANCY profile back as "the
    hours somebody is home", so an airing can be gated on it."""
    out = {}
    for day, lines in _split_day_blocks(fields).items():
        spans, cursor = [], 0
        for f in lines:
            try:
                clock = str(f).split()[1].split(",")[0]
                end = int(clock.split(":")[0]) * 60 + int(clock.split(":")[1])
                value = float(str(f).rsplit(",", 1)[1])
            except (ValueError, IndexError):
                continue
            if value > 0 and end > cursor:
                spans.append((cursor, end))
            cursor = end
        out[day] = _merged_spans(spans)
    return out


def _gated_opening_fields(spans, occ_by_day, inside=1.0, outside=0.0):
    """Window-opening day profile GATED ON OCCUPANCY: ``spans`` intersected, per
    day type, with the minutes somebody is home (``occ_by_day`` from
    ``_profile_positive_spans``).

    Nobody opens a window in a dwelling they are not in. Ungated, the airings ran
    on the clock alone — the aware occupant's 08:30 hygiene airing and the unaware
    occupant's random bursts happened just the same on a weekday when the flat was
    empty from nine to two, ventilating (and, in winter, cooling) nothing.

    Swap ``inside``/``outside`` for the complement — the "HVAC may not run while a
    window is open" availability, which has to be gated identically or it would
    idle the system for airings that no longer happen.
    """
    return _join_day_blocks(
        {day: _spans_to_until(_spans_intersect(spans, occ), inside, outside)
         for day, occ in occ_by_day.items()})


def _placed_opening_spans(band, duration_min, occ_by_day):
    """An airing of ``duration_min`` placed INSIDE ``band``, at the first moment of
    it the dwelling is occupied — per day type.

    "Air for a quarter of an hour some time between 13:00 and 14:00" is a band plus
    a duration, not a fixed clock time, and it has to be read that way once the
    openings are gated on occupancy: pinned to 13:00 sharp the airing would simply
    never happen on a weekday whose occupancy profile has the household out until
    two. Placing it inside the band lets the same instruction survive the gate.

    A day type with no occupied minute in the band gets no airing — the gate wins,
    which is the point of having it. Returns ``{day_type: [(start, end)]}``.
    """
    step = _timestep_minutes()
    out = {}
    for day, occ in occ_by_day.items():
        chosen = []
        for lo, hi in _spans_intersect(band, occ):
            start = -(-lo // step) * step            # round up onto the grid
            if start + duration_min <= hi:
                chosen = [(start, start + duration_min)]
                break
        out[day] = chosen
    return out


def _spans_to_until(spans, value, other):
    """``Until:`` lines holding ``value`` inside each minute span and ``other``
    between them, collapsing runs so equal neighbours do not repeat."""
    out, cursor = [], 0
    for lo, hi in _merged_spans(list(spans)):
        if lo > cursor:
            out.append(f"Until: {lo // 60:02d}:{lo % 60:02d},{other}")
        out.append(f"Until: {hi // 60:02d}:{hi % 60:02d},{value}")
        cursor = hi
    if cursor < 24 * 60:
        out.append(f"Until: 24:00,{other}")
    return out


# THE stylised occupancy of a Spanish dwelling: away for the working morning and
# the late afternoon, home the rest of the time; at home all day at weekends. It is
# the FALLBACK occupancy for everything the occupant does — setpoints, HVAC
# availability, and every window opening — used whenever the census tract's INE
# Time Use Survey agenda is unavailable. One definition, so the thermostat and the
# airings can never disagree about when somebody is in.
_AWARE_OCCUPIED_SPANS = {
    "Weekdays": [(0, 9 * 60), (14 * 60, 16 * 60), (19 * 60, 24 * 60)],
    "AllOtherDays": [(0, 24 * 60)],
}


def _stylised_occupancy_mask():
    """``_AWARE_OCCUPIED_SPANS`` as a ``{day_type: [(start, end)]}`` mask over the
    seven weekdays plus ``AllOtherDays`` — the shape every gate here consumes."""
    weekdays = _AWARE_OCCUPIED_SPANS["Weekdays"]
    other = _AWARE_OCCUPIED_SPANS["AllOtherDays"]
    mask = {d: list(weekdays) for d in _WEEK_DAY_TYPES[:5]}
    mask.update({d: list(other) for d in _WEEK_DAY_TYPES[5:]})
    mask["AllOtherDays"] = list(other)
    return mask


def _night_setback_spans():
    """The SLEEPING band as ``(start, end)`` minute spans, from
    ``residential_behavior.aware_night_setback``; empty when disabled.

    A household asleep is not a household holding 21 °C. The band wraps midnight,
    so it comes back as two spans (23:00–24:00 and 00:00–07:00 by default)."""
    cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
           .get("aware_night_setback", {}) or {})
    if not cfg.get("enabled", True):
        return []
    try:
        lo, hi = (str(t) for t in (cfg.get("hours") or ["23:00", "07:00"]))
        lo = int(lo.split(":")[0]) * 60 + int(lo.split(":")[1])
        hi = int(hi.split(":")[0]) * 60 + int(hi.split(":")[1])
    except (TypeError, ValueError, IndexError):
        return [(0, 7 * 60), (23 * 60, 24 * 60)]
    if lo == hi:
        return []
    return [(lo, hi)] if lo < hi else [(0, hi), (lo, 24 * 60)]


def _preconditioning_lead():
    """Minutes the AWARE/PASSIVE system starts BEFORE the dwelling is occupied
    (``residential_behavior.aware_preconditioning``); 0 when disabled."""
    cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
           .get("aware_preconditioning", {}) or {})
    if not cfg.get("enabled", True):
        return 0
    try:
        return max(0, int(cfg.get("lead_minutes", 60)))
    except (TypeError, ValueError):
        return 60


def _setback_day_fields(value, setback, night_spans=(), lead=None,
                        occ_by_day=None):
    """Day profile of an adaptive setpoint: ``value`` while the dwelling is lived
    in, ``setback`` when it is empty — and ``setback`` through ``night_spans`` too,
    whoever is home. Nobody holds a comfort setpoint while asleep.

    ``lead`` PRECONDITIONS: every comfort period starts that many minutes early, so
    the system is already working when the household walks in (and when it wakes)
    rather than starting from a setback dwelling at the moment somebody arrives.
    That is what a thermostat with a schedule on it actually does, and without it
    the aware occupant's first hour home was always spent recovering.
    """
    lead = _preconditioning_lead() if lead is None else lead
    by_day = {}
    for day, spans in (occ_by_day or _stylised_occupancy_mask()).items():
        awake = _spans_subtract(spans, night_spans or [])
        if lead:
            awake = _merged_spans([(max(0, lo - lead), hi) for lo, hi in awake])
        by_day[day] = _spans_to_until(awake, value, setback)
    return _join_day_blocks(by_day)


def _daily_setback_schedule(idf, name, doy_values, setback, night_spans=(),
                            occ_by_day=None):
    """Schedule:Compact of the daily (adaptive) setpoint with an unoccupied SETBACK,
    grouping consecutive equal-valued days into ``Through`` ranges. ``night_spans``
    additionally sets the value back through the sleeping hours."""
    if name.upper() in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        return name
    fields, doy = [], 1
    while doy <= 365:
        v = doy_values[doy]
        end = doy
        while end < 365 and doy_values[end + 1] == v:
            end += 1
        mm, dd = _doy_to_mmdd(end)
        fields.append(f"Through: {mm}/{dd}")
        fields += _setback_day_fields(v, setback, night_spans,
                                      occ_by_day=occ_by_day)
        doy = end + 1
    _sched(idf, name, "Temperature", fields)
    return name


def _confine_setpoints(heat, cool, z_lo, z_hi, deadband=1.0):
    """Confine a ``(heating, cooling)`` setpoint pair to ``[z_lo, z_hi]`` — the
    extended comfort field's temperature window — keeping at least ``deadband``
    between them.

    BOTH ends are confined, and the heating setpoint is then held below the
    cooling one. Flooring the heating setpoint while only capping the cooling one
    is not enough: in a hot week the adaptive heating setpoint (Tc − edge) climbs
    past the field's ceiling, and EnergyPlus treats the resulting inversion as a
    FATAL error ("Effective heating set-point higher than effective cooling
    set-point"), which is exactly what it did.
    """
    if z_lo is None or z_hi is None:
        return heat, cool
    heat = min(max(heat, z_lo), z_hi)
    cool = min(max(cool, z_lo), z_hi)
    return min(heat, cool - deadband), cool


def _apply_adaptive_setpoints(idf, res_zone_names, cat_band=3.0, margin=1.0,
                              sp_offset=(0.0, 0.0), occ_by_day=None, tag="STY"):
    """EN 16798-1 ADAPTIVE thermostat for the aware residential zones: the setpoints
    TRACK the daily comfort temperature Tc = 0.33·θrm + 18.8 (θrm = running-mean
    outdoor temperature, clamped 10–30 °C). Heating setpoint = Tc − (band − margin);
    cooling setpoint = Tc + (band − margin) — i.e. the Category-``cat_band`` adaptive
    band pulled ``margin`` °C inward so the operative temperature stays inside it.
    When the dwelling is UNOCCUPIED the setpoints set BACK to config
    ``hvac.aware_setback`` (default 16 °C heating / 30 °C cooling); the HVAC is left
    available around the clock in the heating/cooling seasons (season-gated ALWAYS-on)
    so that setback is actually held rather than the zone free-floating. Falls back
    to a fixed band when no weather is attached.

    ``sp_offset`` is an income-driven ``(heating, cooling)`` shift (°C, from
    ``_income_setpoint_offset``) applied to the OCCUPIED adaptive setpoints when
    'aware' is in ``income_setpoints.apply_to`` (the unoccupied setback is left as-is)."""
    rm = _running_mean_by_doy(getattr(idf, "epw", None))
    if not rm:
        return _apply_seasonal_setpoints(idf, res_zone_names)
    edge = max(0.0, cat_band - margin)
    _isp = globals().get("_INCOME_SETPOINTS", {}) or {}
    ho, co = sp_offset if "aware" in set(_isp.get("apply_to", []) or []) else (0.0, 0.0)
    # The adaptive band CONFINED to the extended comfort field: the setpoint still
    # tracks the running-mean outdoor temperature, but stops at the field's edge.
    # Clamped BEFORE the income offset, never after — confining the comfort target
    # must not quietly erase the energy-poverty underheating the offset models.
    z_lo, z_hi = _comfort_zone_setpoint_limits()
    dead = float(((globals().get("_COMFORT_ZONE", {}) or {})
                  .get("setpoints", {}) or {}).get("min_deadband_c", 1.0) or 0.0)
    heat, cool = {}, {}
    for d, trm in rm.items():
        tc = 0.33 * min(30.0, max(10.0, trm)) + 18.8
        h_sp, c_sp = _confine_setpoints(tc - edge, tc + edge, z_lo, z_hi, dead)
        heat[d] = round((h_sp + ho) * 2) / 2          # 0.5 °C steps → good grouping
        cool[d] = round((c_sp + co) * 2) / 2
        # The income offsets only ever widen the band, but the guard is cheap and
        # a inverted thermostat is a fatal error rather than a bad number.
        cool[d] = max(cool[d], heat[d] + dead)
    hcfg = globals().get("_HVAC", {}) or {}
    sb = hcfg.get("aware_setback", {}) or {}
    sb_h, sb_c = float(sb.get("heating", 16)), float(sb.get("cooling", 30))
    # NIGHT SETBACK, on the HEATING schedule only. A sleeping household does not
    # hold 21 °C — it turns the heating down and puts a blanket on, which is what
    # every Spanish dwelling does and what the occupied/away split alone could not
    # express (the dwelling is at its FULLEST at 03:00, so it read as occupied and
    # held the full adaptive setpoint all night).
    #
    # Cooling is deliberately left alone: in the cooling season this occupant is
    # already purging the night with the windows open and the HVAC switched off by
    # AVAIL_AWARE_WINDOWS, so a cooling setback there would be a second, redundant
    # gate on hours that are not conditioned anyway.
    night_spans = _night_setback_spans()
    h = _daily_setback_schedule(idf, f"SP_ADAPT_HEAT_{tag}", heat, sb_h, night_spans,
                                occ_by_day=occ_by_day)
    c = _daily_setback_schedule(idf, f"SP_ADAPT_COOL_{tag}", cool, sb_c,
                                occ_by_day=occ_by_day)
    tstat = f"TSTAT_ADAPTIVE_{tag}"
    if tstat.upper() not in {o.Name.upper() for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}:
        idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat,
                         Heating_Setpoint_Schedule_Name=h,
                         Cooling_Setpoint_Schedule_Name=c)
    # Season-gated 24 h availability (authored by _apply_occupant_control) keeps the
    # system on so the unoccupied setback is enforced; in-between stays off. It is
    # the variant that also switches OFF while the windows are open — this occupant
    # airs on purpose, and heating or cooling into an open dwelling would pay for
    # the air it just decided to throw away. Falls back to the plain always-on
    # schedule for a model authored before that schedule existed.
    have = {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}
    always = "AVAIL_ALWAYS_SEAS" if "AVAIL_ALWAYS_SEAS" in have else None
    for iz in idf.idfobjects.get("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", []):
        if str(iz.Zone_Name).upper() in res_zone_names:
            iz.Template_Thermostat_Name = tstat
            # _apply_occupant_control already gave this zone the windows-open
            # variant, built from the occupancy mask that drives THIS dwelling.
            # Only fall back to the plain always-on schedule when it did not (a
            # model authored before that schedule existed).
            current = str(getattr(iz, "System_Availability_Schedule_Name", "") or "")
            if always and not current.upper().startswith(_AVAIL_AWARE_OPEN):
                iz.System_Availability_Schedule_Name = always
    # Temperature is only half the field — the humidity bounds go on the same zones.
    _apply_comfort_zone_humidity_control(
        idf, {str(z).upper() for z in res_zone_names})


def _occ_setpoint_fields(profile, value, setback):
    """Turn an occupancy fraction profile (``For:``/``Until: HH:MM,frac`` blocks) into a
    setpoint Schedule:Compact body: ``value`` while occupied (frac > 0), ``setback``
    when empty."""
    out = []
    for f in profile:
        if f.startswith("Until:"):
            head, frac = f.rsplit(",", 1)
            out.append(f"{head},{value if float(frac) > 0 else setback}")
        else:                                    # For: … day selector
            out.append(f)
    return out


def _apply_nonresidential_setpoints(idf, zone_meta):
    """Single NORMATIVE user for the non-residential (office / commercial / tertiary)
    zones — no aware/unaware split. Setpoints follow the RITE / RD 486-2022 restriction
    for tertiary buildings (config ``hvac.nonresidential``): heating ≤ 19 °C, cooling
    ≥ 27 °C while occupied, setting BACK to ``hvac.nonresidential_setback`` (16/30) when
    empty. The HVAC is left available around the clock in the heating/cooling seasons
    (season-gated ALWAYS-on) so the setback is held; in-between stays off."""
    hcfg = globals().get("_HVAC", {}) or {}
    nr = hcfg.get("nonresidential", {}) or {}
    heat, cool = float(nr.get("heating", 19)), float(nr.get("cooling", 27))
    sb = hcfg.get("nonresidential_setback", {}) or {}
    sb_h, sb_c = float(sb.get("heating", 16)), float(sb.get("cooling", 30))
    vent = hcfg.get("nonresidential_ventilation_ls_per_person", {}) or {}
    ideal = {str(o.Zone_Name).upper(): o
             for o in idf.idfobjects.get("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", [])}
    always = "AVAIL_ALWAYS_SEAS" if "AVAIL_ALWAYS_SEAS".upper() in {
        o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])} else None
    tstats = {o.Name.upper() for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}
    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        if not s:
            continue
        cat = _resolve_use_category(s)
        if cat == "residential" or cat not in _OCCUPIED_USE_CATEGORIES:
            continue
        prof = _OCCUPANCY_PROFILES.get(cat, _OCCUPANCY_PROFILES["office"])
        tstat = f"TSTAT_NORM_{cat.upper()}"
        if tstat.upper() not in tstats:
            hs = _sched_compact(idf, f"SP_NORM_HEAT_{cat.upper()}", "Temperature",
                                _occ_setpoint_fields(prof, heat, sb_h))
            cs = _sched_compact(idf, f"SP_NORM_COOL_{cat.upper()}", "Temperature",
                                _occ_setpoint_fields(prof, cool, sb_c))
            idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat,
                             Heating_Setpoint_Schedule_Name=hs,
                             Cooling_Setpoint_Schedule_Name=cs)
            tstats.add(tstat.upper())
        iz = ideal.get(zone.Name.upper())
        if iz is not None:
            iz.Template_Thermostat_Name = tstat
            if always:
                iz.System_Availability_Schedule_Name = always
            # RITE IT 1.1.4.2 minimum outdoor air, per person (l/s → m³/s); the HVAC
            # supplies it whenever occupied (per-person → 0 when the zone is empty).
            ls = vent.get(cat)
            if ls:
                _set_first(iz, ("Outdoor_Air_Method",), "Flow/Person")
                _set_first(iz, ("Outdoor_Air_Flow_Rate_per_Person",),
                           round(float(ls) / 1000.0, 6))


def _apply_seasonal_setpoints(idf, res_zone_names, winter=(21.5, 23), summer=(24, 26)):
    """Fixed-band fallback (no weather): heat to the winter lower bound; cool to the
    winter upper bound in winter/shoulder and the summer upper bound in summer."""
    heat = _sched_compact(idf, "SP_AWARE_HEAT", "Temperature",
                          ["For: AllDays", f"Until: 24:00,{winter[0]}"])
    cool = "SP_AWARE_COOL_SEASONAL"
    if cool.upper() not in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        _sched(idf, cool, "Temperature",
               ["Through: 5/31", "For: AllDays", f"Until: 24:00,{winter[1]}",
                "Through: 9/30", "For: AllDays", f"Until: 24:00,{summer[1]}",
                "Through: 12/31", "For: AllDays", f"Until: 24:00,{winter[1]}"])
    tstat = "TSTAT_AWARE_SEASONAL"
    if tstat.upper() not in {o.Name.upper() for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}:
        idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat,
                         Heating_Setpoint_Schedule_Name=heat,
                         Cooling_Setpoint_Schedule_Name=cool)
    for iz in idf.idfobjects.get("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", []):
        if str(iz.Zone_Name).upper() in res_zone_names:
            iz.Template_Thermostat_Name = tstat


# Fixed occupancy fraction profiles (Schedule:Compact day blocks, no leading
# ``Through``) per use category — reqs #7/#8/#9.
_OCCUPANCY_PROFILES = {
    "residential": [
        "For: Weekdays",
        "Until: 07:00,1.0", "Until: 09:00,1.0", "Until: 14:00,0.0",
        "Until: 16:00,0.5", "Until: 19:00,0.0", "Until: 24:00,1.0",
        "For: Weekends Holiday",
        "Until: 09:00,1.0", "Until: 23:00,0.5", "Until: 24:00,1.0",
        "For: AllOtherDays", "Until: 24:00,1.0"],
    "office": [
        "For: Weekdays", "Until: 08:00,0.0", "Until: 20:00,1.0", "Until: 24:00,0.0",
        "For: AllOtherDays", "Until: 24:00,0.0"],
    "commercial": [
        "For: Weekdays Saturday", "Until: 07:00,0.0", "Until: 22:00,1.0",
        "Until: 24:00,0.0",
        "For: AllOtherDays", "Until: 24:00,0.0"],
}
# CTE DB-HE "perfiles de uso" residential 24-h fixed fractions (all days equal).
# Occupancy sensible = 2.15 W/m² × these → 2.15 / 0.54 / 1.08 (paper Table 25).
_CTE_OCC_PROFILE = ["For: AllDays", "Until: 07:00,1.0", "Until: 15:00,0.25",
                    "Until: 23:00,0.5", "Until: 24:00,1.0"]
# Lighting/appliances = 4.40 W/m² × these → 0.44 / 1.32 / 2.20 / 4.40 W/m² (paper).
_CTE_GAIN_PROFILE = ["For: AllDays", "Until: 07:00,0.1", "Until: 19:00,0.3",
                     "Until: 20:00,0.5", "Until: 23:00,1.0", "Until: 24:00,0.1"]
# "HVAC available when someone is home" day profile — GENERATED from the stylised
# occupancy so it cannot drift from the mask the setpoints and the airings use.
# It was previously written out by hand, which is exactly how a model ends up
# conditioning for one occupancy and ventilating for another.
_AVAIL_PRESENT = _join_day_blocks(
    {d: _spans_to_until(s, 1, 0) for d, s in _stylised_occupancy_mask().items()})
# Unaware "leaves the HVAC on": available all waking hours.
_AVAIL_LEFTON = ["For: AllDays", "Until: 07:00,0", "Until: 24:00,1"]
_AVAIL_ALWAYS = ["For: AllDays", "Until: 24:00,1"]
# Season-gated availability of the AWARE/PASSIVE thermostat: on around the clock so
# the unoccupied setback is held, EXCEPT while the occupant has the windows open.
# Authored by _apply_occupant_control (which reads the airing config) and picked up
# by _apply_adaptive_setpoints, which owns those zones' thermostats.
_AVAIL_AWARE_OPEN = "AVAIL_AWARE_WINDOWS"
# Natural-ventilation wind coefficients: EnergyPlus computes the rate as
# ``design × schedule × (A + C·V)``. Defaults now DELIVER the nominal rate
# (A=1, C=0) — see ``residential_behavior.ventilation_wind``. The previous
# (0.15, 0.20) meant an open window nominally at 4 ren/h delivered 0.6 in calm air
# and reached its nominal only at 4.25 m/s, so the air change reported bore no
# resemblance to the number configured for it.
_VENT_A, _VENT_C = 1.0, 0.0


# --------------------------------------------------------------------------- #
# Measured occupancy agenda — INE Time Use Survey (social_ES)
# --------------------------------------------------------------------------- #
# ``social.time_use_household_occupancy`` returns the dominant household occupancy
# STATE at each of the 168 hours of the week ('Sleeping' / 'WithActivity' /
# 'NonOccupied', hour 0 = Monday 00:00) for the building's census tract. The
# helpers below turn that week into the Schedule:Compact day blocks the scenario
# runs need — one profile per weekday, so the survey's weekday/weekend contrast
# survives instead of being averaged away.
_WEEK_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday")


def _week_values_to_fields(values, dp=3):
    """Schedule:Compact fields for 168 hourly VALUES (hour 0 = Monday 00:00),
    one ``For:`` block per weekday with consecutive equal hours collapsed."""
    return _week_states_to_fields(range(168), lambda i: round(float(values[i]), dp))


def _week_states_to_fields(states, value_of):
    """Schedule:Compact fields for a 168-hour week: one ``For:`` block per weekday
    with ``value_of(state)`` per hour, consecutive equal hours collapsed into a
    single ``Until:``. Holidays and any other day type repeat Sunday."""
    fields = []
    sunday = None
    for day, name in enumerate(_WEEK_DAY_NAMES):
        block, previous = [], None
        for hour in range(24):
            value = round(float(value_of(states[day * 24 + hour])), 3)
            if previous is not None and abs(value - previous) < 1e-9:
                block[-1] = f"Until: {hour + 1:02d}:00,{value}"   # extend the run
            else:
                block.append(f"Until: {hour + 1:02d}:00,{value}")
            previous = value
        fields += [f"For: {name}"] + block
        if name == "Sunday":
            sunday = block
    return fields + ["For: AllOtherDays"] + (sunday or ["Until: 24:00,0.0"])


def _time_use_schedules(idf, agenda, cfg):
    """Author the survey-driven residential schedules for one building and return
    ``(occupancy, aware_availability, unaware_availability)`` schedule names, or
    ``None`` when the agenda or the configuration is unusable.

    Three schedules come out of the same week of states:

    * occupancy — ``state_fractions`` per state, driving PEOPLE. This is how many
      of the household are HOME, so a sleeping hour is 1.0: the body at rest is
      handled by the activity level, which also lowers its CO2 generation;
    * gains — ``state_gain_fractions``, driving LIGHTS and EQUIPMENT, which follow
      what the household is doing rather than how many are in: a sleeping dwelling
      is fully occupied and almost fully dark, and an empty one still runs its
      fridge;
    * aware availability — 1 while someone is in the dwelling
      (``present_states``), which is what "condition only when home" means;
    * unaware availability — 1 through every waking hour (``awake_states``),
      at home or not, which is what "leaves the HVAC on" means.

    The two availability profiles are still season-gated by the caller, so no
    behaviour conditions in the in-between season.
    """
    if not agenda or not cfg or not cfg.get("enabled", True):
        return None
    states = agenda.get("states")
    if not states or len(states) < 168:
        return None
    fractions = dict(cfg.get("state_fractions", {}) or {})
    gains = dict(cfg.get("state_gain_fractions", {}) or fractions)
    present = set(cfg.get("present_states", []) or [])
    awake = set(cfg.get("awake_states", []) or [])
    if not fractions or not present or not awake:
        return None

    # The survey publishes both the dominant state per hour and the SHARE of
    # households in each. Prefer the shares: the dominant state is an argmax, so a
    # Sunday where 58-84 % of households are home reads as 100 % occupied for all
    # 24 hours — a far harder profile than the survey actually measured, and one
    # that showed up as an implausible weekend CO2 peak. The argmax is still used
    # for the two binary decisions (HVAC availability), where a yes/no is wanted.
    shares = agenda.get("shares") or {}
    use_shares = all(len(shares.get(k, ())) >= 168
                     for k in ("Sleeping", "WithActivity", "NonOccupied"))

    def _share(state, h):
        return float(shares[state][h]) / 100.0

    def _by_share(h, weights):
        """Household-share-weighted blend of a per-state value at hour ``h``."""
        return sum(_share(k, h) * float(weights.get(k, 0.0))
                   for k in ("Sleeping", "WithActivity", "NonOccupied"))

    threshold = float(cfg.get("availability_share_threshold", 0.5))
    res = (globals().get("_LOAD_LIBRARY", {}) or {}).get("residential", {})
    w_awake = float(res.get("activity", 110))
    w_asleep = float(globals().get("_ACT_SLEEPING", 72))
    # Name the schedules after the survey cell, not the building: every building of
    # a tract's (community, income band) shares one week, so a whole-municipality
    # run authors a handful of schedules rather than one per dwelling.
    tag = (f"{agenda.get('autonomous_community_code', 'XX')}_"
           f"{agenda.get('household_income_group', 0)}")
    if use_shares:
        occ_v = [round(_by_share(h, fractions), 3) for h in range(168)]
        gain_v = [round(_by_share(h, gains), 3) for h in range(168)]
        # Metabolic rate of those AT HOME, split between asleep and awake.
        act_v = []
        for h in range(168):
            home = _share("Sleeping", h) + _share("WithActivity", h)
            act_v.append(round(
                (_share("Sleeping", h) * w_asleep + _share("WithActivity", h) * w_awake)
                / home, 1) if home > 1e-6 else w_awake)
        present_v = [1.0 if (_share("Sleeping", h) + _share("WithActivity", h))
                     >= threshold else 0.0 for h in range(168)]
        awake_v = [1.0 if (_share("WithActivity", h) + _share("NonOccupied", h))
                   >= threshold else 0.0 for h in range(168)]
    else:                                    # only the dominant state is available
        occ_v = [fractions.get(s, 0.0) for s in states]
        gain_v = [gains.get(s, 0.0) for s in states]
        act_v = [w_asleep if s == "Sleeping" else w_awake for s in states]
        present_v = [1.0 if s in present else 0.0 for s in states]
        awake_v = [1.0 if s in awake else 0.0 for s in states]

    occ = _sched_compact(idf, f"SC_OCC_TUS_{tag}", "Fraction",
                         _week_values_to_fields(occ_v))
    # Lights and appliances follow what the household is DOING, not how many of
    # them are home: a sleeping dwelling is fully occupied and almost fully dark.
    gain = _sched_compact(idf, f"SC_GAIN_TUS_{tag}", "Fraction",
                          _week_values_to_fields(gain_v))
    # Metabolic rate follows the same week: the survey knows which hours the
    # household is ASLEEP, and a sleeping body emits less heat and less CO2 (in
    # EnergyPlus the CO2 generation is per watt of activity, so one schedule
    # carries both).
    act = _sched_compact(idf, f"ACT_TUS_{tag}", "AnyNumber",
                         _week_values_to_fields(act_v, dp=1))
    return (occ, gain, _week_values_to_fields(present_v),
            _week_values_to_fields(awake_v), act)


def _building_time_use_agenda(wd, row, verbose=False):
    """The census tract's weekly occupancy agenda for one building row, or ``None``
    when the survey does not reach it (no ``wd``, no census tract, a tract the
    income atlas does not cover, or ``social_ES`` unavailable) — every caller
    falls back to the stylised residential profile."""
    rb = globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {}
    cfg = rb.get("time_use_occupancy", {}) or {}
    if not cfg.get("enabled", False) or not wd or row is None:
        return None
    try:
        from hypercadaster_ES import social
        agenda = social.time_use_household_occupancy(wd, row.get("census_tract_code"))
    except Exception as e:  # noqa: BLE001 - the stylised profile is a valid fallback
        if verbose:
            print(f"  time-use occupancy unavailable ({e}); using the fixed profile")
        return None
    if agenda and verbose:
        print(f"  time-use occupancy: autonomous community "
              f"{agenda['autonomous_community_code']}, household income group "
              f"{agenda['household_income_group']} "
              f"({agenda.get('household_income_group_label')})")
    return agenda


def _windows_to_fields(periods, inside=1.0, outside=0.0):
    """Schedule:Compact day fields that hold ``inside`` within each
    ``["HH:MM", "HH:MM"]`` window and ``outside`` between them — the shape of an
    "open the windows now" profile.

    Swap the two (``inside=0, outside=1``) for its COMPLEMENT: the "HVAC is not
    allowed to run while a window is open" availability that shadows every airing.
    """
    def _mins(t):
        h, m = str(t).split(":")[:2]
        return int(h) * 60 + int(m)

    spans = sorted((_mins(a), _mins(b)) for a, b in (periods or []) if a and b)
    fields, cursor = ["For: AllDays"], 0
    for lo, hi in spans:
        if hi <= lo or hi <= cursor:           # empty, or inside an earlier window
            continue
        if lo > cursor:
            fields.append(f"Until: {lo // 60:02d}:{lo % 60:02d},{outside}")
        fields.append(f"Until: {hi // 60:02d}:{hi % 60:02d},{inside}")
        cursor = hi
    if cursor < 24 * 60:
        fields.append(f"Until: 24:00,{outside}")
    return fields


def _forgotten_night_availability(base_fields, days):
    """``base_fields`` with the listed day types held ON around the clock — the
    UNAWARE occupant who went to bed without switching the heating off."""
    if not days:
        return list(base_fields)
    by_day = _split_day_blocks(base_fields)
    for day in days:
        by_day[day] = ["Until: 24:00,1"]
    return _join_day_blocks(by_day)


def _unaware_day_schedule(idf, zone_name, rng, occ_by_day=None, season_by_doy=None):
    """Opening profile for the UNAWARE occupant's DAYTIME airings, or None when the
    behaviour is switched off.

    ``residential_behavior.unaware_day_ventilation`` gives the number of airings a
    day (``openings_per_day``) and how long each one lasts (``duration_minutes``),
    both as inclusive ranges drawn per DWELLING from ``rng`` (already seeded by the
    zone name). This occupant runs no ventilation strategy, but it does open the
    windows: SHORT, DISCRETE airings — a handful of quarter-hours, not a window
    left ajar through the day. They start on the hour, at most one per hour, inside
    the ``hours`` band.

    Two things narrow them:

    * ``occ_by_day`` gates every airing on the dwelling being OCCUPIED. The start
      hours are also DRAWN from the occupied hours of a representative weekday, so
      the configured 1–3 openings a day survive the gate instead of being trimmed
      to nothing on a flat that is empty from nine to two.
    * ``heating_factor`` keeps only that share of them through the HEATING season —
      a cold day is a day you open the window less, and this occupant is careless,
      not immune to the weather. Randomised rounding again, so the share is exact
      across the building rather than in each dwelling.

    Durations are snapped to the simulation timestep, since EnergyPlus rounds a
    sub-timestep ``Until:`` anyway.
    """
    cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
           .get("unaware_day_ventilation", {}) or {})
    n_lo, n_hi = _pair(cfg.get("openings_per_day"), (1, 3))
    d_lo, d_hi = _pair(cfg.get("duration_minutes"), (15, 30))
    band = cfg.get("hours") or ["08:00", "22:00"]
    try:
        h0, h1 = (int(str(t).split(":")[0]) for t in (band[0], band[-1]))
    except (TypeError, ValueError, IndexError):
        h0, h1 = 8, 22
    hours = list(range(max(0, h0), max(h0 + 1, min(24, h1))))
    # Snap the duration range onto the timestep grid; at least one timestep, so a
    # config asking for less than a timestep still opens the window rather than
    # silently doing nothing.
    step = _timestep_minutes()
    durations = [d for d in range(step, 24 * 60 + 1, step) if d_lo <= d <= d_hi]
    if not durations:
        durations = [max(step, int(round(d_lo / step)) * step)]
    # Draw the start hours from the hours somebody is HOME on a representative
    # weekday. Drawing across the whole band and letting the gate trim afterwards
    # would quietly turn "1-3 airings a day" into "however many happened to land
    # while the flat was occupied", which is not what the config says.
    if occ_by_day:
        at_home = occ_by_day.get("Monday") or occ_by_day.get("AllOtherDays") or []
        home_hours = [h for h in hours
                      if _spans_intersect([(h * 60, h * 60 + 60)], at_home)]
        hours = home_hours or hours
    n = int(rng.randint(int(round(n_lo)), int(round(n_hi))))
    n = max(0, min(n, len(hours)))
    if n == 0 or not hours:
        return None
    spans = _merged_spans([(h * 60, h * 60 + rng.choice(durations))
                           for h in sorted(rng.sample(hours, n))])
    # HEATING keeps only a share of them (config heating_factor).
    try:
        factor = float(cfg.get("heating_factor", 1.0))
    except (TypeError, ValueError):
        factor = 1.0
    factor = max(0.0, min(1.0, factor))
    exact = len(spans) * factor
    k = int(exact) + (1 if rng.random() < (exact - int(exact)) else 0)
    heat_spans = spans[:max(0, min(len(spans), k))]

    def _fields(sel):
        if occ_by_day:
            return _gated_opening_fields(sel, occ_by_day)
        return _windows_to_fields([(_minutes_to_hhmm(lo), _minutes_to_hhmm(hi))
                                   for lo, hi in sel])

    name = f"VENT_RND_{zone_name}"
    if season_by_doy and abs(factor - 1.0) > 1e-9:
        return _seasonal_day_schedule(idf, name, season_by_doy,
                                      {"heating": _fields(heat_spans),
                                       "in_between": _fields(spans),
                                       "cooling": _fields(spans)})
    return _sched_compact(idf, name, "Fraction", _fields(spans))


def _unaware_night_hvac_availability(idf, zone_name, seed, avail, base_fields,
                                     fallback):
    """HVAC availability of one UNAWARE dwelling, with the nights it FORGOT to
    switch the system off.

    The unaware occupant leaves the HVAC on through every waking hour and it goes
    off overnight — but not always: ``residential_behavior.unaware_night_hvac.
    nights_fraction`` of the nights it simply keeps running at the setpoint,
    because nobody remembered to turn it off before going to bed. That is the
    archetype's defining failure, and modelling it as never-happens made the
    unaware dwelling look better overnight than the careful one.

    Drawn per dwelling (seeded by the zone name), on its own random stream so the
    forgotten nights are independent of the nights the same occupant sleeps with
    the windows open — the two are uncoordinated, which is rather the point.
    Returns ``fallback`` (the shared schedule) when the behaviour is switched off.
    """
    import random as _rnd
    cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
           .get("unaware_night_hvac", {}) or {})
    try:
        frac = float(cfg.get("nights_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        return fallback
    if frac <= 0 or not base_fields:
        return fallback
    rng = _rnd.Random(zlib.crc32(f"{zone_name}|{seed}|hvac_night".encode("utf-8"))
                      & 0xffffffff)
    days = _weekly_night_days(rng, frac)
    if not days:
        return fallback
    return avail(f"AVAIL_UNAWARE_NIGHT_{zone_name}",
                 _forgotten_night_availability(base_fields, days))


def _unaware_night_schedule(idf, zone_name, rng, season_by_doy,
                            occ_by_day=None):
    """Opening profile for the UNAWARE occupant's NIGHT window opening, or None
    when the behaviour is switched off (``nights_fraction`` 0, or a draw of zero
    nights for this dwelling).

    ``residential_behavior.unaware_night_ventilation.nights_fraction`` is the share
    of the nights on which this occupant sleeps with the windows open. It is
    realised as whole nights of the week, drawn from ``rng`` (already seeded by the
    zone name, so the draw is deterministic per dwelling): a fraction of 0.3 gives
    each dwelling either 2 or 3 nights a week, chosen so that the fraction comes
    out right ACROSS the building rather than being rounded in every dwelling.

    The opening is unattended — no indoor or outdoor temperature gate — because
    the household is asleep; only the SEASONS listed in the config are excluded.
    """
    cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
           .get("unaware_night_ventilation", {}) or {})
    try:
        frac = float(cfg.get("nights_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    days = _weekly_night_days(rng, frac)
    if not days:
        return None
    # Gated on occupancy like every other opening, though it rarely bites: the
    # window is opened on the way to bed, and asleep is the one state that is
    # unambiguously "at home".
    spans = _periods_to_spans(cfg.get("periods", [["00:00", "08:00"]]))
    shut = ["Until: 24:00,0.0"]
    by_day = {}
    for day in list(_WEEK_DAY_TYPES) + ["AllOtherDays"]:
        if day not in days:
            by_day[day] = shut
            continue
        at_home = (occ_by_day or {}).get(day)
        by_day[day] = _spans_to_until(
            _spans_intersect(spans, at_home) if occ_by_day else spans, 1.0, 0.0)
    fields = _join_day_blocks(by_day)
    name = f"VENT_UNAWARE_NIGHT_{zone_name}"
    seasons = [s for s in (cfg.get("seasons") or ["cooling", "in_between"])
               if s in ("heating", "cooling", "in_between")]
    if not seasons:
        return None
    if season_by_doy and set(seasons) != {"heating", "cooling", "in_between"}:
        return _seasonal_day_schedule(idf, name, season_by_doy,
                                      {s: fields for s in seasons})
    return _sched_compact(idf, name, "Fraction", fields)


def _season_gated_avail(idf, name, day_profile):
    """A Schedule:Compact that follows ``day_profile`` in the HEATING (Dec–Mar) and
    COOLING (Jun–Sep) seasons but is 0 in the IN-BETWEEN season (Apr–May, Oct–Nov)
    — no HVAC there, for any user (req #5)."""
    if name.upper() in {o.Name.upper() for o in idf.idfobjects.get("SCHEDULE:COMPACT", [])}:
        return name
    off = ["For: AllDays", "Until: 24:00,0"]
    fields = []
    for through, active in (("3/31", True), ("5/31", False), ("9/30", True),
                            ("11/30", False), ("12/31", True)):
        fields.append(f"Through: {through}")
        fields += day_profile if active else off
    _sched(idf, name, "Fraction", fields)
    return name


def _add_ventilation(idf, zone_name, name, sched, ach, min_indoor=None,
                     min_outdoor=None, delta=None, fixed=False):
    """Add a natural ``ZoneVentilation:DesignFlowRate`` with optional indoor/outdoor
    temperature gates.

    The delivered rate is ``ach × schedule × (constant + velocity × wind)``, with
    the bracket from ``residential_behavior.ventilation_wind`` — 1.0 + 0.0 by
    default, so an open window delivers the ``ach`` it was configured with.
    ``fixed=True`` forces that regardless of config (the CTE night ventilation,
    which the standard specifies as a flat rate)."""
    _w = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
          .get("ventilation_wind", {}) or {})
    ct, vt = ((1.0, 0.0) if fixed
              else (float(_w.get("constant", _VENT_A)),
                    float(_w.get("velocity", _VENT_C))))
    o = idf.newidfobject(
        "ZONEVENTILATION:DESIGNFLOWRATE", Name=name, Schedule_Name=sched,
        Design_Flow_Rate_Calculation_Method="AirChanges/Hour", Air_Changes_per_Hour=ach,
        Ventilation_Type="Natural", Constant_Term_Coefficient=ct,
        Temperature_Term_Coefficient=0.0, Velocity_Term_Coefficient=vt,
        Velocity_Squared_Term_Coefficient=0.0)
    _set_first(o, _F_ZONE, zone_name)
    if min_indoor is not None:
        _set_first(o, ("Minimum_Indoor_Temperature",), min_indoor)
    if min_outdoor is not None:
        _set_first(o, ("Minimum_Outdoor_Temperature",), min_outdoor)
    if delta is not None:
        _set_first(o, ("Delta_Temperature",), delta)
    return o


def _apply_occupant_control(idf, zone_meta, seed=0, sp_offset=(0.0, 0.0),
                            time_use=None, year_of_construction=None):
    """Author the OCCUPANCY schedules (per use category), the season-gated HVAC
    AVAILABILITY (by occupant behaviour) and the per-behaviour VENTILATION for
    every occupied zone — the reqs #1/#2/#5/#6/#7/#8/#9 behaviour model. Uses the
    per-dwelling behaviour assignment on ``idf._hc_behavior`` (aware / unaware /
    always_on / passive).

    ``sp_offset`` is an income-driven ``(heating, cooling)`` setpoint shift (°C, from
    ``_income_setpoint_offset``) applied to the UNAWARE fixed band when 'unaware' is
    in ``income_setpoints.apply_to``. The normative CTE band is never shifted.

    ``time_use`` is the building's census-tract weekly occupancy agenda from the INE
    Time Use Survey (``_building_time_use_agenda``). When present, the behaviours
    listed in ``residential_behavior.time_use_occupancy.apply_to`` take their
    occupancy and their HVAC availability from that measured week instead of the
    stylised residential profile; the others, and every behaviour when the agenda
    is missing, keep the fixed profiles.

    ``year_of_construction`` decides whether the dwellings carry the CTE DB-HS3
    CONTINUOUS ventilation (``hs3_ventilation``). That one is deliberately NOT a
    behaviour: a ventilation system is part of the BUILDING, so every occupant of a
    post-CTE dwelling gets it, the normative ``cte`` user included."""
    import random as _r
    behavior = getattr(idf, "_hc_behavior", {}) or {}

    # 1) Occupancy schedules per use category → People / Lights / Equipment.
    #    Distinct names ("SC_OCC_*") so they REPLACE (not collide with) the base
    #    config occupancy — the scenario profiles genuinely drop to 0 when the space
    #    is empty, so the "occupied hours" the comfort KPIs use are real.
    occ = {c: _sched_compact(idf, f"SC_OCC_{c.upper()}", "Fraction", f)
           for c, f in _OCCUPANCY_PROFILES.items()}
    people = {o.Name: o for o in idf.idfobjects["PEOPLE"]}
    lights = {o.Name: o for o in idf.idfobjects["LIGHTS"]}
    equips = {o.Name: o for o in idf.idfobjects["ELECTRICEQUIPMENT"]}
    ideal = {str(o.Zone_Name).upper(): o
             for o in idf.idfobjects["HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM"]}
    # Remove the generic/random ventilation from the base build; behaviour-specific
    # ventilation is (re)authored below.
    for o in list(idf.idfobjects.get("ZONEVENTILATION:DESIGNFLOWRATE", [])):
        idf.removeidfobject(o)

    # Seasons are TEMPERATURE-based (in-between = daily mean 18–26 °C) from the EPW,
    # so HVAC/ventilation/awnings follow the actual weather rather than fixed months.
    sbd = _season_by_doy(getattr(idf, "epw", None))
    clr = ["For: AllDays", "Until: 24:00,1.0"]
    off = ["For: AllDays", "Until: 24:00,0"]

    def _avail(name, prof):
        # HVAC available (following ``prof``) on heating/cooling days, OFF in-between.
        if sbd:
            return _seasonal_day_schedule(
                idf, name, sbd, {"heating": prof, "cooling": prof, "in_between": off})
        return _season_gated_avail(idf, name, prof)      # month fallback (no EPW)

    av_present = _avail("AVAIL_PRESENT_SEAS", _AVAIL_PRESENT)
    av_lefton = _avail("AVAIL_LEFTON_SEAS", _AVAIL_LEFTON)
    av_always = _avail("AVAIL_ALWAYS_SEAS", _AVAIL_ALWAYS)
    # Survey-driven residential agenda (INE Time Use Survey via social_ES): the
    # occupancy schedule plus the two availability profiles built from the same
    # week — presence for 'aware', waking hours for 'unaware'. ``tu_behaviors`` is
    # the set of behaviours it replaces the stylised profile for; empty when the
    # agenda is unavailable, which leaves everything on the fixed profiles.
    _rb = globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {}
    _tu_cfg = _rb.get("time_use_occupancy", {}) or {}
    tu = _time_use_schedules(idf, time_use, _tu_cfg)
    tu_behaviors, tu_occ, tu_present, tu_lefton = set(), None, av_present, av_lefton
    tu_act = tu_gain = awake_fields = None
    # Fallback metabolic profile for residential zones with no survey agenda: the
    # night band asleep, the rest awake.
    night_act = _add_night_activity_schedule(idf)
    if tu:
        tag = (f"{(time_use or {}).get('autonomous_community_code', 'XX')}_"
               f"{(time_use or {}).get('household_income_group', 0)}")
        tu_occ, tu_gain, present_fields, awake_fields, tu_act = tu
        tu_present = _avail(f"AVAIL_TUS_PRESENT_{tag}", present_fields)
        tu_lefton = _avail(f"AVAIL_TUS_LEFTON_{tag}", awake_fields)
        tu_behaviors = set(_tu_cfg.get("apply_to", []) or [])
    av_office = _avail("AVAIL_OFFICE_SEAS", [
        "For: Weekdays", "Until: 08:00,0", "Until: 20:00,1", "Until: 24:00,0",
        "For: AllOtherDays", "Until: 24:00,0"])
    av_commercial = _avail("AVAIL_COMMERCIAL_SEAS", [
        "For: Weekdays Saturday", "Until: 07:00,0", "Until: 22:00,1", "Until: 24:00,0",
        "For: AllOtherDays", "Until: 24:00,0"])
    # --- The AIRINGS, gated on OCCUPANCY ----------------------------------- #
    # Nobody opens a window in a dwelling they are not in. Every opening below is
    # intersected, per day type, with the hours somebody is home: ungated they ran
    # on the clock alone, so the aware occupant's 08:30 hygiene airing and the
    # unaware occupant's random bursts happened just the same on a weekday when the
    # flat was empty from nine to two — ventilating, and in winter cooling, nobody.
    #
    # Which occupancy mask applies depends on whether the survey agenda drives that
    # dwelling, so the schedules are built once per SOURCE (at most two per
    # building) and picked per zone.
    _hyg_cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
                .get("hygiene_ventilation", {}) or {})
    _mid_cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
                .get("midday_ventilation", {}) or {})
    _np_cfg = ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
               .get("night_purge", {}) or {})
    # One open-window rate for every airing: what separates the behaviours is when
    # and for how long they open, not how wide.
    open_ach = float((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
                     .get("opening_ach", 4.0))
    hyg_ach = float(_hyg_cfg.get("ach", open_ach))
    midday_min_outdoor = float(_mid_cfg.get("min_outdoor_c", 12.0))
    _hyg_mid_p = _hyg_cfg.get("periods", [["08:00", "08:30"], ["20:00", "20:30"]])
    # HEATING moves the hygiene airing to the middle of the day and shortens it to
    # a quarter of an hour. Airing at 08:30 and again at 20:30 in January opens the
    # dwelling at the two coldest hours it is awake for; the warmest hour of a
    # winter day is the one a household actually picks.
    # Given as a BAND plus a duration, not a fixed clock time: gated on occupancy,
    # an airing pinned to 13:00 sharp would never fire on a weekday whose profile
    # has the household out until two. ``heating_periods`` overrides with explicit
    # windows if you want the airing at a fixed time whatever the occupancy.
    _hyg_heat_p = _hyg_cfg.get("heating_periods")
    _hyg_heat_band = _periods_to_spans([_hyg_cfg.get("heating_window")
                                        or ["13:00", "14:00"]])
    _hyg_heat_dur = int(_hyg_cfg.get("heating_duration_minutes", 15) or 15)
    # The evening airing is dropped on cooling days: the night purge already opens
    # the dwelling from midnight, and a 20:00 opening on a hot evening brings in
    # air warmer than the room.
    _hyg_cool_p = _hyg_cfg.get("cooling_periods") or _hyg_mid_p
    _mid_p = _mid_cfg.get("periods", [["14:00", "14:15"]])
    _night_p = _np_cfg.get("periods", [["00:00", "08:00"]])

    _airing_cache = {}

    def _airings(tag, occ_by_day):
        """``(hygiene, midday, night_purge, hvac_availability)`` schedule names for
        one occupancy mask. Everything is assembled as per-day-type SPANS first, so
        the availability can be the exact complement of the openings that survived
        the gate — gate them differently and the HVAC would be idled for airings
        that no longer happen."""
        if tag in _airing_cache:
            return _airing_cache[tag]

        def _gate(periods):
            spans = _periods_to_spans(periods)
            return {d: _spans_intersect(spans, occ) for d, occ in occ_by_day.items()}

        def _fields(by_day, inside=1.0, outside=0.0):
            return _join_day_blocks({d: _spans_to_until(v, inside, outside)
                                     for d, v in by_day.items()})

        def _both(a, b):
            return {d: _merged_spans(list(a[d]) + list(b[d])) for d in a}

        hyg_heat = (_gate(_hyg_heat_p) if _hyg_heat_p else
                    _placed_opening_spans(_hyg_heat_band, _hyg_heat_dur, occ_by_day))
        hyg_mid, hyg_cool = _gate(_hyg_mid_p), _gate(_hyg_cool_p)
        mid, ngt = _gate(_mid_p), _gate(_night_p)
        if sbd:
            hyg = _seasonal_day_schedule(
                idf, f"VENT_HYGIENE_{tag}", sbd,
                {"heating": _fields(hyg_heat), "in_between": _fields(hyg_mid),
                 "cooling": _fields(hyg_cool)})
            # In the HEATING and IN-BETWEEN seasons, not only heating: the mild
            # season is when opening a window is cheapest, and the occupant who
            # cares about air is exactly the one who does it.
            midday = _seasonal_day_schedule(
                idf, f"VENT_MIDDAY_{tag}", sbd,
                {"heating": _fields(mid), "in_between": _fields(mid)})
            night = _seasonal_day_schedule(idf, f"VENT_NIGHT_{tag}", sbd,
                                           {"cooling": _fields(ngt)})
            # in_between is left out on purpose: no HVAC for any user that season.
            avail = _seasonal_day_schedule(
                idf, f"{_AVAIL_AWARE_OPEN}_{tag}", sbd,
                {"heating": _fields(_both(hyg_heat, mid), inside=0, outside=1),
                 "cooling": _fields(_both(hyg_cool, ngt), inside=0, outside=1)})
        else:
            hyg = _sched_compact(idf, f"VENT_HYGIENE_{tag}", "Fraction",
                                 _fields(hyg_mid))
            midday = _season_gated_avail(idf, f"VENT_MIDDAY_{tag}", _fields(mid))
            night = _sched_compact(idf, f"VENT_NIGHT_{tag}", "Fraction",
                                   _fields(ngt))
            avail = _season_gated_avail(
                idf, f"{_AVAIL_AWARE_OPEN}_{tag}",
                _fields(_both(hyg_mid, mid), inside=0, outside=1))
        _airing_cache[tag] = (hyg, midday, night, avail)
        return _airing_cache[tag]

    # THE occupancy masks — the survey week when the agenda reaches this building's
    # census tract, the stylised profile otherwise. Everything the occupant does
    # reads one of these two and nothing else: the thermostat's occupied/away band,
    # the HVAC availability, and every window opening. Which one applies is decided
    # per dwelling below and recorded on the IDF, so the passes that run after this
    # one (the adaptive thermostat) can ask rather than guess.
    occ_mask_stylised = _stylised_occupancy_mask()
    occ_mask_survey = (_profile_positive_spans(present_fields) if tu else None)
    idf._hc_occ_masks = {"STY": occ_mask_stylised}
    if occ_mask_survey:
        idf._hc_occ_masks["TUS"] = occ_mask_survey
    idf._hc_occ_tag = {}

    # Operable shade transmittance schedules (1 = clear/retracted, <1 = deployed).
    clear = _sched_compact(idf, "SHADE_CLEAR", "Fraction", clr)
    awn_prof = ["For: AllDays", "Until: 08:00,1.0", "Until: 20:00,0.0",
                "Until: 24:00,1.0"]                       # cooling day: deployed 08–20
    if sbd:
        # Deployed on cooling AND in-between days, fully retracted through the
        # heating season: in a Spanish winter the sun on the glazing is wanted, so
        # nothing that shades it operates then.
        awn_deploy = _seasonal_day_schedule(
            idf, "AWN_DEPLOY", sbd,
            {"heating": clr, "cooling": awn_prof, "in_between": awn_prof})
    else:
        awn_deploy = _sched_compact(idf, "AWN_DEPLOY", "Fraction", awn_prof)
    # Persiana (WindowShadingControl) season gate, on the same rule as the awnings.
    # A DISTINCT name from the build-time one: that schedule is authored before the
    # EPW is attached, so it can only use the month split, while the seasons the
    # KPIs report are derived from the weather. Sharing the name would silently
    # keep the month version (the schedule writers no-op on an existing name) and
    # leave the persianas operating on days the results call heating.
    persiana_allow = (_persiana_allow_schedule(idf, sbd, name="PERSIANA_ALLOW_SEAS")
                      if sbd else _persiana_allow_schedule(idf, None))
    persiana_never = _sched_compact(idf, _PERSIANA_NEVER_SCHED, "Fraction",
                                    ["For: AllDays", "Until: 24:00,0"])
    persiana_always = _sched_compact(idf, "PERSIANA_ALWAYS", "Fraction",
                                     ["For: AllDays", "Until: 24:00,1"])
    persiana_by_zone = {}
    for o in idf.idfobjects.get("WINDOWSHADINGCONTROL", []):
        persiana_by_zone.setdefault(str(o.Zone_Name).upper(), []).append(o)
    _night_on = bool((((globals().get("_SHADING", {}) or {}).get("persiana", {}) or {})
                      .get("night_insulation", {}) or {}).get("enabled", True))
    # Index the operable shades by their zone (name = Blind_{zone}_{orient}_… /
    # Awn_{zone}_… / BalconyAwn_{zone}_…).
    shades_by_zone = {}
    for o in idf.idfobjects.get("SHADING:BUILDING:DETAILED", []):
        nm = str(o.Name)
        for pre in ("Blind_", "Awn_", "BalconyAwn_"):
            if nm.startswith(pre):
                z = nm[len(pre):].rsplit("_", 2)[0] if pre == "Blind_" \
                    else nm[len(pre):].rsplit("_", 1)[0]
                shades_by_zone.setdefault(z.upper(), []).append((pre, o))
                break
    # Fixed thermostat for unaware dwellings: a constant band (config hvac.unaware,
    # default 21 °C heating / 25 °C cooling), held whenever the HVAC is available.
    hcfg = globals().get("_HVAC", {}) or {}
    un = hcfg.get("unaware", {}) or {}
    uh, uc = float(un.get("heating", 21)), float(un.get("cooling", 25))
    # Income-driven setpoint shift for the UNAWARE band (gated by apply_to).
    _isp = globals().get("_INCOME_SETPOINTS", {}) or {}
    if "unaware" in set(_isp.get("apply_to", []) or []):
        uh += float(sp_offset[0]); uc += float(sp_offset[1])
    tstat_unaware = "TSTAT_UNAWARE"
    if tstat_unaware.upper() not in {o.Name.upper() for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}:
        _sched_compact(idf, "SP_UNAWARE_HEAT", "Temperature", ["For: AllDays", f"Until: 24:00,{uh}"])
        _sched_compact(idf, "SP_UNAWARE_COOL", "Temperature", ["For: AllDays", f"Until: 24:00,{uc}"])
        idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat_unaware,
                         Heating_Setpoint_Schedule_Name="SP_UNAWARE_HEAT",
                         Cooling_Setpoint_Schedule_Name="SP_UNAWARE_COOL")
    # Fixed thermostat for the ALWAYS_ON archetype (config hvac.always_on). Not in
    # the default scenario set, but reachable through residential_behavior= and a
    # custom scenarios= dict, and it needs its own band: it is the only behaviour
    # that neither adapts (aware/passive) nor follows the unaware one.
    ao = hcfg.get("always_on", {}) or {}
    ah, ac = float(ao.get("heating", 21)), float(ao.get("cooling", 24))
    tstat_always_on = "TSTAT_ALWAYS_ON"
    if tstat_always_on.upper() not in {o.Name.upper()
                                       for o in idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])}:
        _sched_compact(idf, "SP_ALWAYSON_HEAT", "Temperature",
                       ["For: AllDays", f"Until: 24:00,{ah}"])
        _sched_compact(idf, "SP_ALWAYSON_COOL", "Temperature",
                       ["For: AllDays", f"Until: 24:00,{ac}"])
        idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat_always_on,
                         Heating_Setpoint_Schedule_Name="SP_ALWAYSON_HEAT",
                         Cooling_Setpoint_Schedule_Name="SP_ALWAYSON_COOL")

    # CTE residential user (Spanish standards) — constant 21/26 setpoints (no
    # setback), CTE DB-HE internal-gain profiles, fixed 4 ren/h summer-night
    # ventilation, and shadow control at opaque fraction 0.6 (τ=0.4). Config
    # ``residential_cte``.
    cte_cfg = globals().get("_RES_CTE", {}) or {}
    cte_sp = cte_cfg.get("setpoints", {}) or {}
    cte_light_w = float(cte_cfg.get("lighting_w_m2", 4.4))
    cte_equip_w = float(cte_cfg.get("appliances_w_m2", 4.4))
    cte_vent_ach = float(cte_cfg.get("night_ventilation_ach", 4.0))
    cte_h0, cte_h1 = cte_cfg.get("night_ventilation_hours", [1, 8])
    cte_tau = float(cte_cfg.get("shade_transmittance", 0.4))
    tstat_cte, av_cte = _add_cte_thermostat(idf)
    cte_occ = _sched_compact(idf, "SC_OCC_CTE", "Fraction", _CTE_OCC_PROFILE)
    cte_gain = _sched_compact(idf, "SC_GAIN_CTE", "Fraction", _CTE_GAIN_PROFILE)
    # Shadow control ≈ radiation > 140 W/m² → deploy (approximated as daytime).
    # CTE DB-HE specifies the solar protection by INCIDENT RADIATION, all year —
    # not "deployed whenever it is daylight". That radiation trigger is what the
    # persiana's WindowShadingControl does (OnIfHighSolarOnWindow, ungated for
    # this case), so the awnings are left RETRACTED here rather than pinned at
    # tau=cte_tau through every daylight hour of the year. Doing both applied the
    # normative protection twice and, being unconditional, shaded the glazing on
    # midwinter mornings when no control would have closed anything.
    #
    # ``SHADE_CTE`` is still authored at ``cte_tau`` for a caller who wants the
    # old blanket-daytime behaviour back (assign it in place of ``clear`` below).
    _sched_compact(idf, "SHADE_CTE", "Fraction",
                   ["For: AllDays", "Until: 08:00,1.0",
                    f"Until: 20:00,{cte_tau}", "Until: 24:00,1.0"])
    cte_shade = clear
    # Fixed 4 ren/h natural ventilation on summer (cooling-season) nights.
    cte_vent_prof = ["For: AllDays", f"Until: {int(cte_h0):02d}:00,0.0",
                     f"Until: {int(cte_h1):02d}:00,1.0", "Until: 24:00,0.0"]
    if sbd:
        cte_night = _seasonal_day_schedule(idf, "VENT_CTE_NIGHT", sbd,
                                           {"cooling": cte_vent_prof})
    else:
        cte_night = _sched_compact(idf, "VENT_CTE_NIGHT", "Fraction", cte_vent_prof)

    # CTE DB-HS3 continuous ventilation of the dwellings — the same system for every
    # occupant, so it is sized here from the geometry and the head count the model
    # already houses, not from the behaviour.
    hs3_on = _hs3_applies(year_of_construction)
    hs3_sched = (_sched_compact(idf, "VENT_HS3", "Fraction",
                                ["For: AllDays", "Until: 24:00,1.0"])
                 if hs3_on else None)
    hs3_areas = _zone_floor_areas(idf) if hs3_on else {}
    hs3_volumes = _zone_volumes(idf) if hs3_on else {}

    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        if not s:
            continue
        cat = _resolve_use_category(s)
        if cat not in _OCCUPIED_USE_CATEGORIES:
            continue
        prof = occ.get(cat, occ["office"])
        for suffix, attr, table in (("_People", "Number_of_People_Schedule_Name", people),
                                    ("_Lights", "Schedule_Name", lights),
                                    ("_Equip", "Schedule_Name", equips)):
            o = table.get(f"{zone.Name}{suffix}")
            if o is not None:
                setattr(o, attr, prof)

        zu = zone.Name.upper()
        if cat != "residential":
            # Office/commercial: HVAC available in the heating/cooling seasons only
            # (never in-between), following the use's occupancy hours.
            iz = ideal.get(zu)
            if iz is not None:
                iz.System_Availability_Schedule_Name = (
                    av_commercial if cat == "commercial" else av_office)
            # Mechanical outdoor air, on the OCCUPANCY schedule rather than on
            # HVAC availability: premises are ventilated whenever they are open,
            # including the mild season when the heating and cooling are gated
            # off. Added here because the ventilation objects from the base build
            # were removed above.
            _add_mechanical_ventilation(idf, zu, cat, prof)
            continue                                # behaviour model is residential
        beh = (behavior.get(zone.Name) or behavior.get(zu) or ("aware", None))[0]
        # Survey-driven agenda for the behaviours it is configured for: the
        # dwelling's people/lights/equipment follow the measured week instead of
        # the stylised residential profile.
        survey_driven = bool(tu_occ) and beh in tu_behaviors
        # Which OCCUPANCY MASK gates this dwelling's window openings — the survey
        # week when it drives this behaviour, the stylised presence profile
        # otherwise. ``(tag, mask)``; the tag names the shared schedules.
        occ_source = (("TUS", occ_mask_survey) if (survey_driven and occ_mask_survey)
                      else ("STY", occ_mask_stylised))
        idf._hc_occ_tag[zu] = occ_source[0]
        if survey_driven:
            for suffix, attr, table, sched in (
                    ("_People", "Number_of_People_Schedule_Name", people, tu_occ),
                    ("_Lights", "Schedule_Name", lights, tu_gain),
                    ("_Equip", "Schedule_Name", equips, tu_gain)):
                o = table.get(f"{zone.Name}{suffix}")
                if o is not None:
                    setattr(o, attr, sched)
        # Metabolic rate: the survey week when it drives this dwelling, the
        # night-band fallback otherwise. Never for the normative cte case, whose
        # gains are the standard's own fixed profile.
        #
        # This is DELIBERATE and has been queried once already — do not "fix" it.
        # The cte People therefore sit at a flat 110 W all night while the
        # behaviour scenarios drop to the sleeping 72 W, and since EnergyPlus
        # derives CO2 from watts of metabolic rate, the normative case exhales
        # ~1.53x more at 03:00 for the same head count. Its indoor-CO2 profile is
        # consequently shaped quite differently from aware/unaware (the weekly
        # totals differ by only ~2 %; it is the diurnal shape that inverts). That
        # is a property of the STANDARD, which prescribes a fixed occupancy gain
        # and says nothing about sleeping metabolism, not a modelling slip: the
        # cte user is kept exactly as the law defines it, and the cost is that its
        # CO2 is not directly comparable with the behaviour scenarios'.
        if beh != "cte":
            po = people.get(f"{zone.Name}_People")
            if po is not None:
                po.Activity_Level_Schedule_Name = (
                    tu_act if (survey_driven and tu_act) else night_act)
        iz = ideal.get(zu)
        if iz is not None:
            if beh == "cte":
                # All year, no in-between gate: the standard never stops
                # conditioning, it only changes which service its calendar allows.
                iz.System_Availability_Schedule_Name = av_cte
                iz.Template_Thermostat_Name = tstat_cte
            else:
                # 'aware' conditions while someone is home, 'unaware' through every
                # waking hour — both read off the survey week when it drives this
                # behaviour, off the stylised profiles otherwise.
                present = tu_present if survey_driven else av_present
                lefton = tu_lefton if survey_driven else av_lefton
                if beh == "unaware":
                    # …and on a share of the nights this occupant simply FORGETS,
                    # leaving the HVAC running at its setpoint until morning. Drawn
                    # per dwelling (config unaware_night_hvac), so a building has
                    # both kinds of night in it rather than one blanket rule.
                    lefton = _unaware_night_hvac_availability(
                        idf, zu, seed, _avail,
                        awake_fields if survey_driven else _AVAIL_LEFTON,
                        lefton)
                iz.System_Availability_Schedule_Name = (
                    lefton if beh == "unaware" else
                    av_always if beh == "always_on" else present)
                if beh == "unaware":
                    iz.Template_Thermostat_Name = tstat_unaware
                elif beh == "always_on":
                    iz.Template_Thermostat_Name = tstat_always_on

        if beh == "cte":
            # CTE 24-h fixed internal-gain profiles (occupancy sensible × CTE_OCC;
            # lighting/appliances 4.40 W/m² × CTE_GAIN → 0.44/1.32/2.20/4.40).
            for suffix, attr, table, watts in (
                    ("_People", "Number_of_People_Schedule_Name", people, None),
                    ("_Lights", "Schedule_Name", lights, cte_light_w),
                    ("_Equip", "Schedule_Name", equips, cte_equip_w)):
                o = table.get(f"{zone.Name}{suffix}")
                if o is None:
                    continue
                setattr(o, attr, cte_occ if suffix == "_People" else cte_gain)
                if watts is not None:
                    _set_first(o, _F_W_AREA, watts)
            # The occupancy density is deliberately NOT reset here: every scenario
            # houses the same people, from the census-tract density the build
            # applied. Only the behaviour differs, which is the whole point of
            # comparing them — a cte case with 3.5 people per dwelling against the
            # census's 1 would differ for two reasons at once.

        # Operable AWNINGS: aware and passive deploy them by day in the cooling
        # and in-between seasons; everyone else leaves them retracted, the
        # normative case included (its solar protection is the radiation-triggered
        # persiana, not a permanently lowered awning). They are always transparent
        # through the heating season, whoever lives there.
        for pre, o in shades_by_zone.get(zu, []):
            if beh == "cte":
                o.Transmittance_Schedule_Name = cte_shade
            elif beh in ("aware", "passive"):
                o.Transmittance_Schedule_Name = awn_deploy
            else:
                o.Transmittance_Schedule_Name = clear
        # Exterior PERSIANAS: the WindowShadingControl closes them when the sun on
        # the window exceeds the setpoint, but only while its schedule allows —
        # which is never in the heating season, and never at all for the occupants
        # who do not operate them.
        # The occupant behaviours never shade in the heating season; the normative
        # cte case follows CTE DB-HE, whose shadow control is by radiation all
        # year, so it gets the ungated schedule.
        for ctrl in persiana_by_zone.get(zu, []):
            if beh == "cte":
                # Normative: radiation control all year, no night rule.
                ctrl.Shading_Control_Type = _PERSIANA_SOLAR_CONTROL_TYPE
                ctrl.Schedule_Name = persiana_always
            elif beh in ("aware", "passive"):
                # Shutters down at night — a Spanish household closes them, and a
                # closed persiana plus its air gap cuts the loss through the
                # glazing. By day the control only fires when cooling is called
                # for, so the heating season keeps its solar gain.
                ctrl.Shading_Control_Type = (_PERSIANA_NIGHT_CONTROL_TYPE
                                             if _night_on
                                             else _PERSIANA_SOLAR_CONTROL_TYPE)
                ctrl.Schedule_Name = (persiana_always if _night_on
                                      else persiana_allow)
            else:
                ctrl.Schedule_Name = persiana_never

        # CTE DB-HS3: a post-CTE dwelling is ventilated CONTINUOUSLY, whoever lives
        # in it. Added before the behaviour-driven airings below, which are what the
        # occupant does ON TOP of the system the building came with.
        if hs3_on:
            area = float(hs3_areas.get(zu, 0.0))
            po = people.get(f"{zone.Name}_People")
            occupants = area * float(_first_value(po, _F_PPL_AREA) or 0.0) if po else 0.0
            _add_hs3_ventilation(idf, zu, occupants, area,
                                 hs3_volumes.get(zu, 0.0), hs3_sched)

        if beh == "cte":
            # CTE natural ventilation: FIXED 4 ren/h on summer nights 01:00–08:00.
            _add_ventilation(idf, zu, f"{zu}_CTENight", cte_night, cte_vent_ach,
                             fixed=True)
        elif beh in ("aware", "passive"):
            # Daily hygiene airing, midday airing when the outdoor air is mild, and
            # the summer NIGHT PURGE (req #1/#2).
            #
            # The night purge carries NO temperature gate. The occupant opens the
            # window on the way to bed and it stays open until morning: the
            # household is asleep, so nothing reopens or recloses it when the room
            # crosses a threshold at 04:00. It used to be gated on the zone being
            # ≥22 °C with a 1 K indoor-outdoor delta, which made it a thermostat
            # instead of a habit — and because a free-floating dwelling drifts
            # warmer than a conditioned one, that gate opened the windows of
            # 'aware_ff' for more hours than those of 'aware'. Those two scenarios
            # must differ ONLY by the HVAC being off, which is now the case.
            #
            # All three are gated on OCCUPANCY: this occupant airs the dwelling
            # they are in, not the one they left at nine.
            hyg, midday, night, avail_open = _airings(*occ_source)
            _add_ventilation(idf, zu, f"{zu}_Hyg", hyg, hyg_ach)
            _add_ventilation(idf, zu, f"{zu}_MidHeat", midday, open_ach,
                             min_outdoor=midday_min_outdoor)
            _add_ventilation(idf, zu, f"{zu}_NightPurge", night, open_ach)
            # …and while any of them is open the HVAC does not run. Set here (not
            # only in _apply_adaptive_setpoints) because the right schedule depends
            # on which occupancy mask drives this dwelling.
            if iz is not None:
                iz.System_Availability_Schedule_Name = avail_open
        else:
            # Unaware: a few SHORT random airings through the day, regardless of
            # the HVAC state (req #6) — 1–3 openings of 15–30 min, drawn per
            # dwelling (config unaware_day_ventilation), and halved in the heating
            # season. Discrete quarter-hours, not a window left ajar all day, and
            # only in the hours somebody is home to open one.
            rng = _r.Random(zlib.crc32(f"{zu}|{seed}".encode("utf-8")) & 0xffffffff)
            rs = _unaware_day_schedule(idf, zu, rng, occ_source[1], sbd)
            if rs:
                _add_ventilation(idf, zu, f"{zu}_RndVent", rs, open_ach)
            # …and, on a share of the nights, the windows are simply left open
            # until morning (config unaware_night_ventilation.nights_fraction).
            # Unattended like the aware night purge — the household is asleep,
            # which is also the one state that is unambiguously "at home".
            ns = _unaware_night_schedule(idf, zu, rng, sbd, occ_source[1])
            if ns:
                _add_ventilation(idf, zu, f"{zu}_UnawareNight", ns, open_ach)


def _add_mechanical_ventilation(idf, zone_name, use_category, schedule):
    """RITE mechanical outdoor air for a NON-RESIDENTIAL zone.

    Sized per occupant (``nonresidential_ventilation.litres_per_second_person``,
    the RITE IDA categories) and driven by the zone's OCCUPANCY schedule, so it
    delivers air whenever the premises are open — not only when the HVAC happens
    to be running. That distinction is the whole point: the shoulder season is
    exactly when the heating and cooling are off and a full shop would otherwise
    be left with nothing but its own leakage.

    ``Ventilation_Type=Intake`` marks it as fan-driven outdoor air, so it enters
    the contaminant balance at the outdoor CO2 concentration like any other
    outdoor-air route. Returns the object, or ``None`` when disabled.
    """
    cfg = globals().get("_NONRES_VENT", {}) or {}
    if not cfg.get("enabled", True):
        return None
    rates = cfg.get("litres_per_second_person", {}) or {}
    ls = rates.get(use_category, rates.get("other"))
    if not ls or float(ls) <= 0:
        return None
    name = f"{zone_name}_MechVent"
    if name.upper() in {str(o.Name).upper()
                        for o in idf.idfobjects.get("ZONEVENTILATION:DESIGNFLOWRATE", [])}:
        return None
    o = idf.newidfobject(
        "ZONEVENTILATION:DESIGNFLOWRATE", Name=name, Schedule_Name=schedule,
        Design_Flow_Rate_Calculation_Method="Flow/Person",
        Flow_Rate_per_Person=float(ls) / 1000.0,      # l/s·person → m³/s·person
        Ventilation_Type="Intake",
        Constant_Term_Coefficient=1.0, Temperature_Term_Coefficient=0.0,
        Velocity_Term_Coefficient=0.0, Velocity_Squared_Term_Coefficient=0.0)
    _set_first(o, _F_ZONE, zone_name)
    return o


def _hs3_applies(year_of_construction):
    """True when a dwelling of this construction year must carry a CTE DB-HS3
    continuous-ventilation system (``hs3_ventilation.from_year``; the CTE became
    obligatory on 29-09-2006, so 2007 is the first year that had to comply).

    An UNKNOWN year is treated as non-compliant: the Spanish stock is dominated by
    dwellings older than the CTE, and inventing a ventilation system that a
    building may not have would flatter its air quality on no evidence."""
    cfg = globals().get("_HS3_VENT", {}) or {}
    if not cfg.get("enabled", True):
        return False
    try:
        return int(year_of_construction) >= int(cfg.get("from_year", 2007))
    except (TypeError, ValueError):
        return False


def _add_hs3_ventilation(idf, zone_name, occupants, floor_area_m2, volume_m3,
                         schedule):
    """CTE DB-HS3 CONTINUOUS ventilation of one dwelling, for every occupant.

    Sized on DB-HS3 Tabla 2.1 as ``max(admission, extraction)`` — the occupant term
    (dormitorios + salas) against the wet-room term (baños + cocina) — since the two
    are the two ends of one airflow and the system is built to the larger. Config:
    ``hs3_ventilation``.

    ``Ventilation_Type=Intake`` marks it as fan-driven outdoor air, like the RITE
    ventilation of the premises, so it enters the contaminant balance at the outdoor
    CO2 concentration. It runs on an always-on schedule: "continuous" is the whole
    point of HS3, and it is the one air-renewal route that does NOT depend on
    whether the occupant opens anything. Returns the object, or ``None``.
    """
    cfg = globals().get("_HS3_VENT", {}) or {}
    ls = max(float(occupants or 0.0) * float(cfg.get("l_s_per_occupant", 8.0) or 0.0),
             float(floor_area_m2 or 0.0) * float(cfg.get("l_s_per_m2", 0.55) or 0.0))
    if ls <= 0:
        return None
    flow = ls / 1000.0                                   # l/s → m³/s
    if volume_m3 and volume_m3 > 0:
        lo = float(cfg.get("min_ach", 0.0) or 0.0) * volume_m3 / 3600.0
        hi = cfg.get("max_ach")
        flow = max(flow, lo)
        if hi:
            flow = min(flow, float(hi) * volume_m3 / 3600.0)
    name = f"{zone_name}_HS3"
    if name.upper() in {str(o.Name).upper()
                        for o in idf.idfobjects.get("ZONEVENTILATION:DESIGNFLOWRATE", [])}:
        return None
    o = idf.newidfobject(
        "ZONEVENTILATION:DESIGNFLOWRATE", Name=name, Schedule_Name=schedule,
        Design_Flow_Rate_Calculation_Method="Flow/Zone", Design_Flow_Rate=flow,
        Ventilation_Type="Intake",
        Constant_Term_Coefficient=1.0, Temperature_Term_Coefficient=0.0,
        Velocity_Term_Coefficient=0.0, Velocity_Squared_Term_Coefficient=0.0)
    _set_first(o, _F_ZONE, zone_name)
    return o


def _first_present(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v not in (None, ""):
                return v
    return None


def _read_eplus_csv(csv_path):
    """Load an EnergyPlus ``eplusout.csv`` into a DataFrame with added integer
    ``month``, ``day`` and ``hour`` (1–24, end-of-interval) parsed from its
    Date/Time column. Returns None if the file is missing/empty."""
    import pandas as _pd
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    df = _pd.read_csv(csv_path)
    dt = df.columns[0]
    months, days, hours = [], [], []
    for v in df[dt].astype(str):
        # " 01/21  10:00:00"
        try:
            date_part, time_part = v.split()[0], v.split()[1]
            mm, dd = date_part.split("/")[:2]
            months.append(int(mm)); days.append(int(dd))
            hours.append(int(time_part.split(":")[0]))
        except Exception:
            months.append(0); days.append(0); hours.append(0)
    df["month"] = months
    df["day"] = days
    df["hour"] = hours
    return df


def _running_mean_outdoor(df, outdoor_col):
    """Per-row running-mean outdoor temperature θrm (EN 16798-1): the daily mean
    outdoor temps smoothed by θrm(d) = (1-α)·θ(d-1) + α·θrm(d-1), α=0.8, broadcast
    back to the hourly rows. Clamped to [10, 30] °C (model validity range)."""
    import pandas as _pd
    import numpy as _np
    day = _calendar_day_index(df)
    daily = _pd.Series(df[outdoor_col].values).groupby(day).mean()
    rm, prev = {}, daily.iloc[0] if len(daily) else 20.0
    for d, tday in daily.items():
        prev = (1 - _ADAPTIVE_ALPHA) * tday + _ADAPTIVE_ALPHA * prev
        rm[d] = min(30.0, max(10.0, prev))
    return _pd.Series(day).map(rm).values


def _adaptive_out_of_comfort(top, theta_rm, cat="II"):
    """Boolean array: operative temperature ``top`` outside the EN 16798-1
    Category ``cat`` adaptive band around Tc = 0.33·θrm + 18.8."""
    import numpy as _np
    band = _ADAPTIVE_CAT_BAND.get(cat, 3.0)
    tc = 0.33 * _np.asarray(theta_rm) + 18.8
    top = _np.asarray(top, dtype=float)
    return (top > tc + band) | (top < tc - band)


# --------------------------------------------------------------------------- #
# Psychrometrics + the EXTENDED comfort zone
# --------------------------------------------------------------------------- #
# Comfort is a region in (temperature, humidity), not an interval on a
# thermometer: 26 °C at 8 g/kg and 26 °C at 16 g/kg are different rooms to be in.
# The adaptive band above answers a different question — was the TEMPERATURE
# acceptable for this climate — and both are reported, because a dwelling can pass
# one and fail the other and that difference is the finding.
#
# Sea-level pressure. The zone is a nomogram, not a site measurement: a 200 m site
# moves the RH curves by well under a curve spacing, and pinning one pressure keeps
# two buildings' charts comparable.
_P_ATM = 101325.0


def _p_sat(t_c):
    """Saturation vapour pressure over water (Pa) — Magnus form, well under 0.1 %
    error across the −20…+50 °C an indoor comfort chart covers. It is an
    ambient-range fit, not a steam table; do not read it at 100 °C."""
    import numpy as _np
    t = _np.asarray(t_c, dtype=float)
    return 610.94 * _np.exp(17.625 * t / (t + 243.04))


def _humidity_ratio(t_c, rh, p=_P_ATM):
    """Humidity ratio (kg/kg) at dry-bulb ``t_c`` and relative humidity ``rh`` (0–1)."""
    import numpy as _np
    pw = _np.clip(_np.asarray(rh, dtype=float), 0.0, 1.0) * _p_sat(t_c)
    return 0.621945 * pw / _np.maximum(p - pw, 1.0)


def _relative_humidity(t_c, w, p=_P_ATM):
    """Relative humidity (0–1) back out of a (temperature, humidity ratio) point."""
    import numpy as _np
    w = _np.maximum(_np.asarray(w, dtype=float), 0.0)
    pw = w * p / (0.621945 + w)
    return _np.clip(pw / _p_sat(t_c), 0.0, 1.5)


def _moist_air_enthalpy(t_c, w):
    """Specific enthalpy of moist air (kJ per kg of DRY air) — the h of the Mollier
    h-x diagram: ``1.006·t + w·(2501 + 1.86·t)``."""
    import numpy as _np
    t = _np.asarray(t_c, dtype=float)
    w = _np.asarray(w, dtype=float)
    return 1.006 * t + w * (2501.0 + 1.86 * t)


def _comfort_zone_humidity_limits(t_c, cfg=None):
    """``(w_lo, w_hi)`` in kg/kg — the comfort zone's humidity bounds at each
    temperature, as the INTERSECTION of every constraint that applies there.

    The extended (Mollier h-x) comfort field is a polygon, not a rectangle: the
    relative-humidity curves, the absolute-humidity cap and the enthalpy lines each
    bind over a different stretch of the temperature range, and which one binds is
    exactly what gives the field its shape. Rather than hard-code vertices that
    would silently stop meaning anything the moment a bound is edited, every bound
    is evaluated at every temperature and the tightest wins — so the polygon
    re-derives itself from ``comfort_zone`` in the config.

    Returns ``w_hi < w_lo`` where the constraints leave nothing, which is how the
    caller detects the ends of the field.
    """
    import numpy as _np
    c = dict(cfg or globals().get("_COMFORT_ZONE", {}) or {})
    t = _np.asarray(t_c, dtype=float)
    lo = _np.full(t.shape, 0.0)
    hi = _np.full(t.shape, _np.inf)
    rh_lo, rh_hi = c.get("rh_min"), c.get("rh_max")
    if rh_lo is not None:
        lo = _np.maximum(lo, _humidity_ratio(t, float(rh_lo) / 100.0))
    if rh_hi is not None:
        hi = _np.minimum(hi, _humidity_ratio(t, float(rh_hi) / 100.0))
    if c.get("humidity_ratio_max_g_kg") is not None:
        hi = _np.minimum(hi, float(c["humidity_ratio_max_g_kg"]) / 1000.0)
    if c.get("humidity_ratio_min_g_kg") is not None:
        lo = _np.maximum(lo, float(c["humidity_ratio_min_g_kg"]) / 1000.0)
    # Enthalpy bounds are lines in (t, w): w = (h − 1.006·t) / (2501 + 1.86·t).
    for key, side in (("enthalpy_max_kj_kg", "hi"), ("enthalpy_min_kj_kg", "lo")):
        if c.get(key) is None:
            continue
        w_h = (float(c[key]) - 1.006 * t) / (2501.0 + 1.86 * t)
        if side == "hi":
            hi = _np.minimum(hi, w_h)
        else:
            lo = _np.maximum(lo, w_h)
    return lo, hi


def comfort_zone_polygon(cfg=None, steps=97):
    """``(t, w_lo, w_hi)`` tracing the extended comfort zone, temperatures in °C
    and humidity ratios in kg/kg — the region drawn behind the report's
    psychrometric chart. Temperatures where the constraints leave nothing are
    dropped, so the returned arrays are the field's real extent."""
    import numpy as _np
    c = dict(cfg or globals().get("_COMFORT_ZONE", {}) or {})
    t = _np.linspace(float(c.get("t_min_c", 20.0)), float(c.get("t_max_c", 26.0)),
                     int(steps))
    lo, hi = _comfort_zone_humidity_limits(t, c)
    keep = hi > lo
    return t[keep], lo[keep], hi[keep]


def _outside_comfort_zone(t_op, w, cfg=None):
    """Boolean array: the (operative temperature, humidity ratio) state is OUTSIDE
    the extended comfort zone — too cold, too warm, too dry, too humid, or over the
    enthalpy line. NaN humidity is treated as outside, never as passing."""
    import numpy as _np
    c = dict(cfg or globals().get("_COMFORT_ZONE", {}) or {})
    t = _np.asarray(t_op, dtype=float)
    w = _np.asarray(w, dtype=float)
    lo, hi = _comfort_zone_humidity_limits(t, c)
    out = (t < float(c.get("t_min_c", 20.0))) | (t > float(c.get("t_max_c", 26.0)))
    out |= ~_np.isfinite(w) | (w < lo) | (w > hi)
    return out


def _comfort_zone_setpoint_limits(cfg=None):
    """``(heat_min, cool_max)`` °C — the temperature window a thermostat must stay
    inside to keep the zone in the extended comfort field, or ``(None, None)`` when
    the field is not steering the setpoints (``comfort_zone.setpoints``).

    The margin is taken off BOTH ends because the thermostat holds the zone AIR
    temperature while the comfort criterion is written on the OPERATIVE one, and
    the two differ by the radiant half of the room.
    """
    c = dict(cfg or globals().get("_COMFORT_ZONE", {}) or {})
    sp = dict(c.get("setpoints", {}) or {})
    if not c.get("enabled", True) or not sp.get("enabled", True):
        return None, None
    m = float(sp.get("margin_c", 0.5) or 0.0)
    lo = float(c.get("t_min_c", 20.0)) + m
    hi = float(c.get("t_max_c", 26.0)) - m
    return (lo, hi) if hi > lo else (None, None)


def _apply_comfort_zone_humidity_control(idf, zone_names, cfg=None):
    """Put the comfort field's HUMIDITY bounds on the ideal-loads systems of
    ``zone_names``.

    Temperature is only half of the field. Without this the setpoints hold the
    dwelling between 20 and 26 °C and its humidity still wanders out of the zone —
    which is most of what the psychrometric chart was showing.

    The bounds go on the TEMPLATE's own setpoint fields, not on a hand-written
    ``ZoneControl:Humidistat``: ExpandObjects writes one humidistat per zone from
    those fields, and a second one authored here is rejected as a duplicate
    (``ZoneControl:Humidistat has a duplicate field``, and the run dies).

    ``comfort_zone.setpoints.humidity_control``: ``dehumidification`` (default)
    drives the upper bound only, which is what a real cooling coil does and all a
    Spanish dwelling has the equipment for; ``full`` adds humidification against
    the lower bound; ``none`` leaves humidity uncontrolled. Returns the number of
    zones wired.

    ``humidity_setpoint`` decides WHICH upper bound: ``field`` (default) uses the
    field's EFFECTIVE ceiling at the cooling setpoint, ``rh_max`` the bare
    ``rh_max``. They are not the same number and the difference matters — a
    humidistat can only hold a relative humidity, but the field's ceiling is the
    tightest of the RH, absolute-humidity and enthalpy bounds, and above ~22.3 °C
    that is the ENTHALPY line. Set at ``rh_max`` = 65 %, the humidistat leaves the
    dwelling sitting outside the very zone the setpoints are supposed to deliver;
    at the cooling setpoint of 25.5 °C the field's real ceiling is nearer 47 % RH.
    """
    c = dict(cfg or globals().get("_COMFORT_ZONE", {}) or {})
    sp = dict(c.get("setpoints", {}) or {})
    mode = str(sp.get("humidity_control", "dehumidification") or "none").lower()
    if (not zone_names or mode == "none" or not c.get("enabled", True)
            or not sp.get("enabled", True)):
        return 0
    rh_lo = float(c.get("rh_min", 30) or 0.0)
    rh_hi = float(c.get("rh_max", 65) or 100.0)
    if str(sp.get("humidity_setpoint", "field")).lower() == "field":
        _lo, _hi = _comfort_zone_setpoint_limits(c)
        if _hi is not None:
            import numpy as _np
            _w_hi = float(_comfort_zone_humidity_limits(_np.array([_hi]), c)[1][0])
            if _w_hi > 0:
                rh_hi = min(rh_hi,
                            round(float(_relative_humidity(_hi, _w_hi)) * 100.0, 1))
    humidify = mode == "full"
    n = 0
    for iz in idf.idfobjects.get("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", []):
        if str(iz.Zone_Name).upper() not in zone_names:
            continue
        iz.Dehumidification_Control_Type = "Humidistat"
        iz.Dehumidification_Setpoint = rh_hi
        iz.Humidification_Control_Type = "Humidistat" if humidify else "None"
        iz.Humidification_Setpoint = rh_lo
        n += 1
    return n


def _time_use_occupied_mask(agenda, cfg=None):
    """The census tract's Time Use Survey week as a ``{day_type: [(start, end)]}``
    occupancy mask — the hours the household is HOME (``present_states``). None
    when the survey does not reach this building."""
    states = (agenda or {}).get("states")
    if not states or len(states) < 168:
        return None
    cfg = cfg or ((globals().get("_RESIDENTIAL_BEHAVIOR", {}) or {})
                  .get("time_use_occupancy", {}) or {})
    present = set(cfg.get("present_states", ["Sleeping", "WithActivity"]) or [])
    return _profile_positive_spans(
        _week_states_to_fields(states, lambda s: 1.0 if s in present else 0.0))


def _hourly_occupied(df, mask, year=None):
    """Hourly boolean over an EnergyPlus run: is somebody HOME, per a
    ``{day_type: spans}`` occupancy mask read against the calendar.

    The KPI "occupied hours" have to be the SAME set of hours in every scenario or
    the comparisons in the report answer different questions per column — a
    percentage of discomfort over one occupancy is not comparable with the same
    percentage over another. Taking them from each run's own People output made
    exactly that mistake, so the mask is now imposed from outside the run.
    """
    import numpy as _np
    import datetime as _dt
    n = len(df)
    if not mask:
        return _np.ones(n, bool)
    # 7 day types x 24 hours, resolved once.
    table = _np.zeros((8, 24), bool)
    order = list(_WEEK_DAY_TYPES) + ["AllOtherDays"]
    for i, day in enumerate(order):
        for lo, hi in (mask.get(day) or []):
            for h in range(24):
                if lo <= h * 60 < hi:
                    table[i, h] = True
    months = df["month"].to_numpy(int)
    days = df["day"].to_numpy(int)
    hours = _np.clip(df["hour"].to_numpy(int) - 1, 0, 23)   # 1..24 → 0..23
    out = _np.zeros(n, bool)
    cache = {}
    for i in range(n):
        key = (months[i], days[i])
        wd = cache.get(key)
        if wd is None:
            try:
                wd = _dt.date(int(year or 2021), key[0], key[1]).weekday()
            except (ValueError, TypeError):
                wd = 7                                  # AllOtherDays
            cache[key] = wd
        out[i] = table[wd, hours[i]]
    return out


def _zone_column(df, zone, variable):
    """The E+ CSV column for ``zone``'s ``variable`` (case-insensitive), or None.

    The output key is usually the zone name (``ZONE:Var``) but for ideal-loads
    variables ExpandObjects reports under the system name (``ZONE IDEAL LOADS AIR
    SYSTEM:Var``), so a key that *starts with* the zone name (at a space boundary)
    also matches."""
    z = zone.upper()
    var = variable.upper()
    for c in df.columns:
        cu = c.upper()
        key = cu.split(":", 1)[0].strip()
        if (key == z or key.startswith(z + " ")) and var in cu:
            return c
    return None


_SEASON_NAMES = ("heating", "cooling", "in_between")


# Days in each month of a non-leap year, to walk the calendar when collapsing a
# set of dates into contiguous runs.
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _next_calendar_day(month, day):
    """The day after ``(month, day)``, wrapping 31/12 to 01/01."""
    if day < _MONTH_LENGTHS[month - 1]:
        return (month, day + 1)
    return (1, 1) if month == 12 else (month + 1, 1)


def _date_runs(dates):
    """Collapse a set of ``(month, day)`` pairs into contiguous ``"DDMM/DDMM"``
    runs, ordered through the calendar.

    A season classified from daily temperatures is normally several separate
    stretches — a Spanish heating season runs from autumn into spring, so it shows
    up as two runs at either end of the year rather than one. Reporting only the
    first and last date would describe it as spanning January to December.
    """
    if not dates:
        return []
    ordered = sorted(dates)
    runs, start, prev = [], ordered[0], ordered[0]
    for d in ordered[1:]:
        if d == _next_calendar_day(*prev):
            prev = d
            continue
        runs.append((start, prev))
        start = prev = d
    runs.append((start, prev))
    return [f"{a[1]:02d}{a[0]:02d}/{b[1]:02d}{b[0]:02d}" for a, b in runs]


def _hourly_season_mask(main_df, out_col):
    """Per-hour season (heating/cooling/in_between) from each DAY's mean outdoor
    temperature SMOOTHED over a 3-day window (so seasons stay continuous), plus the
    day COUNT and the calendar SPAN per season.

    Returns ``(season_mask{name: bool-array}, season_days, season_ranges)``, where
    ``season_ranges[name]`` lists the CONTIGUOUS date runs of that season as
    ``"DDMM/DDMM"`` strings, e.g. ``["0101/0104", "0111/3112"]`` for a heating
    season that wraps the turn of the year. Seasons are temperature-derived, so
    they genuinely break into several runs — a single first/last pair would report
    a heating season as spanning the whole year.
    """
    import numpy as _np
    import pandas as _pd
    n = len(main_df)
    if out_col is None:
        return ({s: _np.zeros(n, bool) for s in _SEASON_NAMES},
                {s: 0 for s in _SEASON_NAMES},
                {s: [] for s in _SEASON_NAMES})
    day = _calendar_day_index(main_df)
    day_mean = _pd.Series(main_df[out_col].values).groupby(day).mean()
    smoothed = _smooth_days(day_mean.values)                 # chronological day order
    season_of_day = {d: _season_of_temp(t)
                     for d, t in zip(day_mean.index, smoothed)}
    season_hourly = _np.array([season_of_day[d] for d in day])
    mask = {s: (season_hourly == s) for s in _SEASON_NAMES}
    days = {s: 0 for s in _SEASON_NAMES}
    for t in smoothed:
        days[_season_of_temp(t)] += 1
    # Contiguous date runs of each season, read off the rows themselves so they
    # follow the simulated run period (a July-only run reports July dates, not the
    # full year).
    months = main_df["month"].values
    dom = main_df["day"].values
    ranges = {s: _date_runs({(int(mo), int(dd))
                             for mo, dd in zip(months[mask[s]], dom[mask[s]])
                             if mo and dd})
              for s in _SEASON_NAMES}
    return mask, days, ranges


def _zone_scenario_kpis(main_df, zone_meta, areas, comfort_cat="II",
                        occupied_mask=None, year=None):
    """Per-occupied-zone KPI records for one scenario, one record PER SEASON
    (heating / cooling / in-between — classified by each day's MEAN outdoor
    temperature, in-between = 18–26 °C): the zone's HVAC thermal demand and internal
    gains in that season, and the % of OCCUPIED hours out of comfort split into DAY
    (08:00–21:59) and NIGHT (22:00–07:59). Comfort is EN 16798-1 adaptive; per-m²
    metrics use the zone's floor area. Each record also carries ``season_days`` (the
    number of days of the year in that season).

    ``occupied_mask`` fixes WHICH HOURS are the occupied ones — the census tract's
    Time Use Survey week (a ``{day_type: spans}`` mask), ``"all"`` for a scenario
    that carries no occupancy profile at all, or None to fall back to the run's own
    People output. It is imposed from outside the run on purpose: a discomfort or
    CO2 percentage taken over one occupancy is not comparable with the same
    percentage over another, and reading each scenario's own People count made
    every column of the comparison answer a slightly different question."""
    import numpy as _np
    out_col = _zone_column(main_df, "Environment",
                           "Site Outdoor Air Drybulb Temperature")
    theta_rm = (_running_mean_outdoor(main_df, out_col)
                if out_col else _np.full(len(main_df), 20.0))
    # Resolved once for the whole run, not per zone: every dwelling of a building
    # shares the census tract's agenda.
    imposed_occ = (_hourly_occupied(main_df, occupied_mask, year)
                   if isinstance(occupied_mask, dict) else None)
    night = main_df["hour"].isin(_NIGHT_HOURS).values
    day = ~night
    _co2_thresholds = [float(t) for t in
                       (globals().get("_CO2", {}) or {}).get("thresholds", [1000, 1400])]
    season_mask, season_days, season_ranges = _hourly_season_mask(main_df, out_col)
    records = []
    for zname_u, s in zone_meta.items():
        cat = _resolve_use_category(s)
        if cat not in _OCCUPIED_USE_CATEGORIES:
            continue
        area = areas.get(zname_u, 0.0)
        if area <= 1e-6:
            continue
        # Comfort is only meaningful while someone is there, so every discomfort
        # percentage below is taken over the OCCUPIED hours alone. An occupied-use
        # zone with no occupant-count output is a modelling gap, not an
        # always-occupied zone: reporting all its hours as if they were occupied
        # would quietly answer a different question, so it yields NaN instead.
        has_occ = True
        if isinstance(occupied_mask, dict):
            occ = imposed_occ                       # the survey week, every zone
        elif occupied_mask == "all":
            occ = _np.ones(len(main_df), bool)      # no occupancy profile (cte)
        else:
            occ_c = _zone_column(main_df, zname_u, "Zone People Occupant Count")
            if occ_c:
                occ = main_df[occ_c].values > 0.0
            else:
                occ, has_occ = _np.zeros(len(main_df), bool), False
                sys.stderr.write(
                    f"  comfort: no occupant count reported for {zname_u}; its "
                    f"out-of-comfort percentages are left empty (they are defined "
                    f"over occupied hours only)\n")
        top_c = _zone_column(main_df, zname_u, "Zone Operative Temperature")
        heat_c = _zone_column(main_df, zname_u, "Ideal Loads Supply Air Total Heating")
        cool_c = _zone_column(main_df, zname_u, "Ideal Loads Supply Air Total Cooling")
        gain_c = _zone_column(main_df, zname_u, "Zone Total Internal Total Heating")
        # Either spelling, so a CSV produced by an older EnergyPlus still reads.
        sol_c = (_zone_column(main_df, zname_u,
                              "Enclosure Windows Total Transmitted Solar Radiation")
                 or _zone_column(main_df, zname_u,
                                 "Zone Windows Total Transmitted Solar Radiation"))
        co2_c = _zone_column(main_df, zname_u, "Zone Air CO2 Concentration")
        inf_c = _zone_column(main_df, zname_u,
                             "Zone Infiltration Standard Density Air Change Rate")
        ven_c = _zone_column(main_df, zname_u,
                             "Zone Ventilation Standard Density Air Change Rate")
        sph_c = _zone_column(main_df, zname_u, "Zone Thermostat Heating Setpoint")
        spc_c = _zone_column(main_df, zname_u, "Zone Thermostat Cooling Setpoint")
        n = len(main_df)
        heat = main_df[heat_c].values if heat_c else _np.zeros(n)
        cool = main_df[cool_c].values if cool_c else _np.zeros(n)
        hvac = heat + cool
        gains = main_df[gain_c].values if gain_c else _np.zeros(n)
        solar = main_df[sol_c].values if sol_c else _np.zeros(n)
        ooc = (_adaptive_out_of_comfort(main_df[top_c].values, theta_rm, comfort_cat)
               if top_c else None)
        # The EXTENDED comfort zone — the (temperature, humidity) field, not the
        # temperature band. Needs the humidity output, so it is NaN on a run made
        # before it was requested rather than silently reading as comfortable.
        hum_c = _zone_column(main_df, zname_u, "Zone Mean Air Humidity Ratio")
        ooc_zone = (_outside_comfort_zone(main_df[top_c].values,
                                          main_df[hum_c].values)
                    if (top_c and hum_c
                        and (globals().get("_COMFORT_ZONE", {}) or {}).get("enabled", True))
                    else None)

        def _pct(mask, flags=None):
            flags = ooc if flags is None else flags
            if flags is None or mask.sum() == 0:
                return _np.nan
            return 100.0 * float(flags[mask].mean())

        def _setpoint_while_running(smask):
            """Mean thermostat setpoint over the hours the HVAC actually ran, using
            the HEATING setpoint in the hours it heated and the COOLING one in the
            hours it cooled.

            One number per row, because a row is one season: reporting a heating
            and a cooling setpoint side by side left one of them meaningless in
            every row. Averaging the schedule instead of the running hours would
            mix in the setback and the free-floating hours, which is not a
            temperature the building was ever held at."""
            vals = []
            for col, served in ((sph_c, heat), (spc_c, cool)):
                if not col:
                    continue
                running = smask & (served > 0.0)
                if not running.any():
                    continue
                v = main_df[col].values[running]
                v = v[_np.isfinite(v)]
                if v.size:
                    vals.append((float(v.sum()), int(v.size)))
            tot = sum(n for _v, n in vals)
            return round(sum(v for v, _n in vals) / tot, 2) if tot else _np.nan

        def _co2_stats(mask):
            """Indoor CO2 over the masked hours: the typical level, the peak, and
            how much of the time it sat above each threshold.

            Reported over OCCUPIED hours only, like the comfort KPIs — CO2 in an
            empty dwelling decays to outdoor and says nothing about whether the
            people in it had enough fresh air."""
            keys = ([f"p{p}" for p in _CO2_PERCENTILES] + ["mean", "max"]
                    + [f"pct_above_{int(t)}" for t in _co2_thresholds])
            empty = {k: _np.nan for k in keys}
            if co2_c is None or not mask.any():
                return empty
            v = main_df[co2_c].values[mask]
            v = v[_np.isfinite(v)]
            if not v.size:
                return empty
            out = {f"p{p}": round(float(_np.percentile(v, p)), 0)
                   for p in _CO2_PERCENTILES}
            out["mean"] = round(float(v.mean()), 0)
            out["max"] = round(float(v.max()), 0)
            for t in _co2_thresholds:
                out[f"pct_above_{int(t)}"] = round(100.0 * float((v > t).mean()), 1)
            return out

        def _ach_stats(mask):
            """Outdoor-air change rate over the masked hours, by route and total.

            Both ``ZoneInfiltration`` and ``ZoneVentilation`` draw OUTDOOR air in
            EnergyPlus, so both dilute the indoor CO2 towards the outdoor
            concentration; their sum is the renewal rate the CO2 KPIs answer to.
            The total gets a MEAN plus percentiles around it (the spread is the
            point — a dwelling sealed for twenty hours and flung open for one
            renews as much air as one left ajar, and reads very differently); the
            routes get means, where only the split matters. No median: on a flow
            that is zero most hours it reports 0 and says nothing (see
            ``_ACH_PERCENTILES``).
            """
            keys = ([f"total_{k}" for k in
                     [f"p{p}" for p in _ACH_PERCENTILES] + ["mean", "min", "max"]]
                    + ["infiltration_mean", "ventilation_mean"])
            empty = {k: _np.nan for k in keys}
            if not mask.any() or (inf_c is None and ven_c is None):
                return empty
            inf = (main_df[inf_c].values[mask] if inf_c
                   else _np.zeros(int(mask.sum())))
            ven = (main_df[ven_c].values[mask] if ven_c
                   else _np.zeros(int(mask.sum())))
            inf = _np.nan_to_num(inf, nan=0.0)
            ven = _np.nan_to_num(ven, nan=0.0)
            tot = inf + ven
            if not tot.size:
                return empty
            out = {f"total_p{p}": round(float(_np.percentile(tot, p)), 3)
                   for p in _ACH_PERCENTILES}
            out["total_mean"] = round(float(tot.mean()), 3)
            out["total_min"] = round(float(tot.min()), 3)
            out["total_max"] = round(float(tot.max()), 3)
            out["infiltration_mean"] = round(float(inf.mean()), 3)
            out["ventilation_mean"] = round(float(ven.mean()), 3)
            return out

        def _temperature_stats(mask):
            """Indoor operative-temperature percentiles AND extremes over the
            masked hours, keyed ``p10``… plus ``min``/``max``."""
            keys = [f"p{p}" for p in _TEMPERATURE_PERCENTILES] + list(_TEMPERATURE_EXTREMES)
            empty = {k: _np.nan for k in keys}
            if top_c is None or not mask.any():
                return empty
            v = main_df[top_c].values[mask]
            v = v[_np.isfinite(v)]
            if not v.size:
                return empty
            out = {f"p{p}": round(float(_np.percentile(v, p)), 2)
                   for p in _TEMPERATURE_PERCENTILES}
            out["min"] = round(float(v.min()), 2)
            out["max"] = round(float(v.max()), 2)
            return out

        for season, smask in season_mask.items():
            n_season = int(smask.sum())
            records.append({
                "zone": zname_u, "use_type": s.get("use_type"), "use_category": cat,
                "season": season, "season_days": season_days[season],
                "floor_area_m2": area,
                "season_ranges": ", ".join(season_ranges[season]) or None,
                "thermal_demand_kwh": float(hvac[smask].sum()) * _KWH_PER_J,
                "internal_gains_kwh": float(gains[smask].sum()) * _KWH_PER_J,
                "solar_gains_kwh": float(solar[smask].sum()) * _KWH_PER_J,
                # The setpoint as DELIVERED, over the hours the HVAC ran.
                "avg_setpoint_c": _setpoint_while_running(smask),
                "hvac_hours": int((smask & (hvac > 0.0)).sum()),
                "pct_discomfort_day": _pct(occ & smask & day),
                "pct_discomfort_night": _pct(occ & smask & night),
                # …and against the EXTENDED comfort zone, which also has to be
                # passed on humidity. A dwelling can sit inside the adaptive
                # temperature band and outside this one, and that gap is the
                # reason both are reported.
                "pct_discomfort_zone_day": _pct(occ & smask & day, ooc_zone),
                "pct_discomfort_zone_night": _pct(occ & smask & night, ooc_zone),
                # Indoor operative temperature the occupants actually saw: over
                # OCCUPIED hours only, split day/night like the discomfort above.
                **{f"t_{k}_day": v for k, v in
                   _temperature_stats(occ & smask & day).items()},
                **{f"t_{k}_night": v for k, v in
                   _temperature_stats(occ & smask & night).items()},
                # Indoor CO2 — how renewed the air was while people were in it.
                **{f"co2_{k}_day": v for k, v in
                   _co2_stats(occ & smask & day).items()},
                **{f"co2_{k}_night": v for k, v in
                   _co2_stats(occ & smask & night).items()},
                # Outdoor air renewing the zone — what the CO2 above answers to.
                **{f"ach_{k}_day": v for k, v in
                   _ach_stats(occ & smask & day).items()},
                **{f"ach_{k}_night": v for k, v in
                   _ach_stats(occ & smask & night).items()},
                # The denominator those two percentages are taken over, so the
                # basis of the comfort KPIs is visible rather than implied.
                # ``None`` when the zone reports no occupancy at all: such a zone
                # has no comfort figures either, so it must not drag the reported
                # occupied share of its use type down as if it were empty.
                "occupied_hours": int((occ & smask).sum()) if has_occ else None,
                "season_hours": n_season,
            })
    return records


def _report_nonresidential_once(rows, scenario, nonres_source):
    """Keep every residential row, and emit the NON-residential ones only for the
    ``nonres_source`` run, relabelled :data:`NONRESIDENTIAL_SCENARIO_LABEL`.

    The behaviour scenarios describe residents; office and commercial premises run
    the same normative user (RITE / RD 486-2022) in all of them, so repeating them
    per scenario invites reading a thermal-coupling artefact — their demand does
    drift with what the dwellings around them do — as an occupant effect on the
    shop. ``nonres_source=None`` disables the collapsing (per-scenario reporting).
    """
    if nonres_source is None:
        return rows
    out = []
    for r in rows:
        if _use_category(r.get("use_type")) == "residential":
            out.append(r)
        elif scenario == nonres_source:
            out.append({**r, "scenario": NONRESIDENTIAL_SCENARIO_LABEL})
    return out


def _pct_occupied(recs):
    """Share (%) of the season's hours that were OCCUPIED across ``recs`` — the
    denominator the out-of-comfort percentages are taken over. Zones reporting no
    occupancy at all are left out of both sides, so they neither contribute
    comfort figures nor deflate this one; ``NaN`` when none of them reports any."""
    import numpy as _np
    known = [r for r in recs if r.get("occupied_hours") is not None]
    total = sum(r.get("season_hours", 0) for r in known)
    if not known or total <= 0:
        return _np.nan
    return round(100.0 * sum(r["occupied_hours"] for r in known) / total, 1)


def _aggregate_scenario_kpis(records, scenario):
    """Area-weighted aggregation of per-zone KPI records by (USE TYPE × SEASON) for
    one scenario → list of dict rows (per-m² seasonal demand/gains, area-weighted
    day/night comfort percentages, and the occupied share those rest on)."""
    import numpy as _np
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["use_type"], r["season"])].append(r)
    _order = {"heating": 0, "cooling": 1, "in_between": 2}
    rows = []
    for (use, season), recs in sorted(
            by_key.items(), key=lambda kv: (kv[0][0], _order.get(kv[0][1], 9))):
        A = sum(r["floor_area_m2"] for r in recs) or 1.0

        def _wmean(key):
            # ``.get`` rather than ``[]``: a KPI added later (the comfort-zone
            # columns) must not make the aggregator blow up on a record built
            # before it existed — it reports NaN, like any other missing figure.
            vals = [(r.get(key), r["floor_area_m2"]) for r in recs
                    if r.get(key) == r.get(key) and r.get(key) is not None]
            w = sum(a for _v, a in vals)
            return round(sum(v * a for v, a in vals) / w, 1) if w else _np.nan
        def _hmean(key, hours_key):
            """Mean weighted by the HOURS each zone actually ran — a zone that
            conditioned for three hours must not weigh the same as one that ran
            all season."""
            vals = [(r[key], r.get(hours_key, 0)) for r in recs
                    if r.get(key) == r.get(key) and r.get(hours_key, 0) > 0]
            w = sum(h for _v, h in vals)
            return round(sum(v * h for v, h in vals) / w, 2) if w else _np.nan

        rows.append({
            "scenario": scenario, "use_type": use, "season": season,
            "season_days": recs[0].get("season_days"),
            "season_ranges": recs[0].get("season_ranges"),
            "n_zones": len(recs), "floor_area_m2": round(A, 1),
            "thermal_demand_kwh_m2": round(
                sum(r["thermal_demand_kwh"] for r in recs) / A, 2),
            "internal_gains_kwh_m2": round(
                sum(r["internal_gains_kwh"] for r in recs) / A, 2),
            "solar_gains_kwh_m2": round(
                sum(r["solar_gains_kwh"] for r in recs) / A, 2),
            "avg_setpoint_c": _hmean("avg_setpoint_c", "hvac_hours"),
            # Hours the average zone ran, the weight behind that setpoint.
            "hvac_hours": _wmean("hvac_hours"),
            # Percentiles cannot be pooled exactly across zones without the raw
            # hours, so each zone's percentile is area-weighted — the same
            # treatment the discomfort percentages get.
            **{f"t_{k}_{part}": _wmean(f"t_{k}_{part}")
               for part in ("day", "night")
               for k in [f"p{p}" for p in _TEMPERATURE_PERCENTILES]
               + list(_TEMPERATURE_EXTREMES)},
            **{k: _wmean(k) for k in recs[0]
               if str(k).startswith(("co2_", "ach_"))},
            "pct_hours_out_of_comfort_day": _wmean("pct_discomfort_day"),
            "pct_hours_out_of_comfort_night": _wmean("pct_discomfort_night"),
            "pct_hours_out_of_comfort_zone_day": _wmean("pct_discomfort_zone_day"),
            "pct_hours_out_of_comfort_zone_night": _wmean("pct_discomfort_zone_night"),
            # Share of the season's hours the two percentages above are taken
            # over: they describe OCCUPIED hours, and this says how many those
            # were. 100 % means the profile never empties the zone (the CTE
            # user, by construction); a low value means a mostly-empty zone whose
            # comfort figures rest on few hours.
            "pct_occupied_hours": _pct_occupied(recs),
        })
    return rows


# The occupant-behaviour scenarios are a RESIDENTIAL construct: office and
# commercial premises have one normative user in every one of them (the RITE /
# RD 486-2022 tertiary restriction, ``_apply_nonresidential_setpoints``). Their
# KPIs are therefore reported ONCE, under this scenario label, rather than
# repeated under each residential behaviour as if the shop had six of its own.
NONRESIDENTIAL_SCENARIO_LABEL = "normative"

# Where those single rows are taken from. The premises are thermally coupled to
# the dwellings around them (shared floors/ceilings), so their demand does shift
# a little with what the residents do even though their own user never changes —
# the reported figure has to come from ONE run, and the honest choice is the one
# where the rest of the building is normative too.
NONRESIDENTIAL_SOURCE_SCENARIO = "cte"


# --------------------------------------------------------------------------- #
# Domestic hot water (DHW) end-use — CTE DB-HE HE4, census-tract occupancy
# --------------------------------------------------------------------------- #
_DHW_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_DHW_RHO_CP = 4.186        # kJ/(L·K) = ρ_water[kg/L] × Cp_water[kJ/(kg·K)]

# Plausible band for the census-tract occupancy (people per m² of dwelling floor
# area), used to detect a tract whose buildings are only partly present in the
# frame. The CTE default is 0.028 (≈2.8 people per 100 m²); the bounds are set
# wide enough to accept any real Spanish tract — 0.005 is a nearly empty one,
# 0.10 is ten people in a 100 m² flat — so anything outside them is a truncated
# denominator, not an occupancy.
_DHW_PLAUSIBLE_RATE = (0.005, 0.10)


def _scalar_code(v):
    """Normalise a census ``census_tract_code`` that may be a 1-element list/array (after
    address compaction) to a plain hashable scalar, or ``None`` — mirrors
    ``social._scalar`` so tract keys match across the social_ES join."""
    if isinstance(v, (list, tuple, np.ndarray)):
        v = v[0] if len(v) else None
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _population_column(gdf):
    """Name of the social_ES population-COUNT column (``so__population_<year>``),
    the most recent year when several are present; ``None`` when absent. Excludes
    the derived ``so__pct_*``/age columns."""
    best, best_year = None, -1
    for c in list(getattr(gdf, "columns", [])):
        m = re.fullmatch(r"so__population(?:_(\d{4}))?", str(c).lower())
        if not m:
            continue
        yr = int(m.group(1)) if m.group(1) else 0
        if yr >= best_year:
            best, best_year = c, yr
    return best


def _residential_household_areas(row):
    """Non-communal floor areas (m²) of the building's RESIDENTIAL (Housing)
    building spaces — one entry per dwelling. Uses ``_building_spaces`` (which
    prefers ``area_without_communal``)."""
    out = []
    for sp in _building_spaces(row):
        if _use_category(sp.get("use_type")) != "residential":
            continue
        try:
            a = float(sp.get("area"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and a > 0:
            out.append(a)
    return out


def _census_tract_person_rate(gdf, verbose=False):
    """``{census_tract_code → people per m²}`` of residential non-communal floor area —
    the social_ES tract population divided by the summed Housing area of ALL
    buildings in the tract. Empty when DHW is disabled or the population/tract data
    is unavailable (callers then fall back to the CTE-default density).

    The denominator is only right when ``gdf`` holds the WHOLE tract. Handed a
    subset — one building, say — the area is too small and the rate comes out
    proportionally too high, which would silently multiply the DHW demand rather
    than fail. Rates outside :data:`_DHW_PLAUSIBLE_RATE` (people per m² of
    dwelling) are therefore dropped as evidence of a partial frame, and those
    tracts fall back to the CTE-default density.
    """
    cfg = globals().get("_RES_DHW", {}) or {}
    if not cfg.get("enabled", False):
        return {}
    if gdf is None or "census_tract_code" not in list(getattr(gdf, "columns", [])):
        return {}
    pop_col = _population_column(gdf)
    if not pop_col:
        return {}
    area_by, pop_by = {}, {}
    for _, row in gdf.iterrows():
        sec = _scalar_code(row.get("census_tract_code"))
        if sec is None:
            continue
        area_by[sec] = area_by.get(sec, 0.0) + sum(_residential_household_areas(row))
        if sec not in pop_by:
            try:
                p = float(row.get(pop_col))
                if math.isfinite(p) and p > 0:
                    pop_by[sec] = p
            except (TypeError, ValueError):
                pass
    lo, hi = _DHW_PLAUSIBLE_RATE
    rates, rejected = {}, []
    for sec, a in area_by.items():
        if a <= 1e-6 or sec not in pop_by:
            continue
        rate = pop_by[sec] / a
        if lo <= rate <= hi:
            rates[sec] = rate
        else:
            rejected.append((sec, rate))
    if rejected and verbose:
        sec, rate = rejected[0]
        print(f"  census-tract occupancy: {rate:.3f} people/m² for tract {sec} is "
              f"outside {lo}–{hi} — the frame holds only part of the tract, so the "
              f"CTE-default density is used for DHW (pass context_gdf= for the "
              f"population-driven figure)")
    return rates


_CTAGG_PEOPLE_PER_M2 = "ctagg__people_per_m2"
_CTAGG_PEOPLE_PER_DWELLING = "ctagg__people_per_dwelling"


def _row_tract_rate(row):
    """The building's own census-tract occupancy density (people per m² of
    dwelling), from the ``ctagg__people_per_m2`` column ``hc.merge`` writes onto
    every building — or ``None`` when the merge did not produce it.

    This is what lets a ONE-BUILDING frame carry its tract's context: the density
    is a property of the whole tract, computed at merge time while every building
    of it was in hand, so it no longer has to be re-derived from whatever subset
    of the frame the caller happens to still be holding."""
    try:
        v = float(row.get(_CTAGG_PEOPLE_PER_M2))
    except (TypeError, ValueError):
        return None
    return v if (math.isfinite(v) and v > 0) else None


def _row_people_per_dwelling(row):
    """The census-tract mean household size (``ctagg__people_per_dwelling``), or
    ``None`` — the per-DWELLING half of the occupancy estimate."""
    try:
        v = float(row.get(_CTAGG_PEOPLE_PER_DWELLING))
    except (TypeError, ValueError):
        return None
    return v if (math.isfinite(v) and v > 0) else None


def _dwelling_occupants(row, tract_rate):
    """People per household, blending the two census estimates.

    ``population / tract dwellings`` gives every home the tract's average
    household whatever its size; ``population / tract residential area × this
    dwelling's area`` scales with size but takes it too literally. Household size
    does grow with dwelling size, far more slowly than floor area does, so the two
    are averaged (weight ``loads.occupancy_area_weight``)::

        people = max(1, round((1-w)·people_per_dwelling + w·people_per_m²·area))

    Falls back to whichever half is available. ``max(1, …)`` keeps a small
    dwelling from being modelled as empty. Returns ``(total_people,
    residential_area)``; ``total_people`` is ``None`` when neither estimate nor
    any residential space is available."""
    areas = _residential_household_areas(row)
    total_area = sum(areas)
    rate = _row_tract_rate(row)
    if rate is None and isinstance(tract_rate, dict):
        rate = tract_rate.get(_scalar_code(row.get("census_tract_code")))
    per_dwelling = _row_people_per_dwelling(row)
    if not areas or (rate is None and per_dwelling is None):
        return None, total_area
    w = float(globals().get("_OCC_AREA_WEIGHT", 0.5))
    if rate is None:                       # only the per-dwelling estimate
        w = 0.0
    elif per_dwelling is None:             # only the per-area one
        w = 1.0
    total = 0
    for a in areas:
        est = (1.0 - w) * (per_dwelling or 0.0) + w * (rate or 0.0) * a
        total += max(1, int(round(est)))
    return total, total_area


def _residential_occupancy_density(row, tract_rate=None):
    """The building's RESIDENTIAL occupancy density (people per m²) implied by its
    census tract, or ``None`` when the tract occupancy is unavailable.

    Built from the same per-dwelling head count as the DHW demand
    (:func:`_dwelling_occupants`: ``max(1, round(rate × dwelling_area))``, so a
    dwelling is never modelled as empty) and expressed back as a density over the
    building's residential floor area — which is the unit an EnergyPlus ``People``
    object takes. Going through the head count rather than passing the tract rate
    straight through matters for small dwellings, where the min-1 floor lifts the
    effective density above the tract average.

    Used for the PEOPLE internal gains, so the metabolic heat, the DHW demand and
    the household size all follow one census population. Clamped to
    :data:`_DHW_PLAUSIBLE_RATE` as a last guard against a nonsense rate reaching
    the loads.
    """
    people, area = _dwelling_occupants(row, tract_rate)
    if not people or area <= 1e-6:
        return None
    density = people / area
    lo, hi = _DHW_PLAUSIBLE_RATE
    if not (lo <= density <= hi):
        return None
    return density


def _cte_default_occupants(row):
    """People per household from the CTE-default occupancy density
    (``loads.residential.ppl_per_m2``): ``max(1, round(density × household_area))``
    summed over the building's Housing spaces. Returns ``(total_people, area)``."""
    dens = float((globals().get("_LOAD_LIBRARY", {}).get("residential", {}) or {})
                 .get("ppl_per_m2", 0.028))
    areas = _residential_household_areas(row)
    return (sum(max(1, int(round(dens * a))) for a in areas), sum(areas))


def _dhw_demand_kwh(occupants, cfg=None):
    """Annual DHW thermal demand (kWh) for ``occupants`` people, CTE DB-HE HE4:
    Σ_month occupants × L/person/day × days × ρ·Cp × (T_ref − T_inlet_month) / 3600.
    0 when disabled or no occupants."""
    cfg = globals().get("_RES_DHW", {}) if cfg is None else cfg
    if not cfg or not cfg.get("enabled", False) or not occupants:
        return 0.0
    lpp = float(cfg.get("litres_per_person_day", 28))
    t_ref = float(cfg.get("reference_temp", 60))
    inlet = list(cfg.get("inlet_temp_monthly") or []) or [10.0] * 12
    kwh = 0.0
    for m in range(12):
        ti = float(inlet[m] if m < len(inlet) else inlet[-1])
        dT = max(0.0, t_ref - ti)
        kwh += occupants * lpp * _DHW_DAYS_IN_MONTH[m] * _DHW_RHO_CP * dT / 3600.0
    return kwh


def _dhw_demand_kwh_by_season(occupants, season_by_doy, run_period_months=None,
                              cfg=None):
    """Annual DHW thermal demand (kWh) split by the EPW's per-day season, honoring
    ``run_period_months`` (a day counts only if its month is simulated). Returns
    ``{"heating":…, "cooling":…, "in_between":…}`` (zeros when disabled / no
    occupants / no season map). Each day contributes ``occupants × L/day × ρ·Cp ×
    (T_ref − T_inlet_month) / 3600`` to whichever season it falls in — DHW is used
    year-round, so it appears in every season that is simulated."""
    out = {"heating": 0.0, "cooling": 0.0, "in_between": 0.0}
    cfg = globals().get("_RES_DHW", {}) if cfg is None else cfg
    if not cfg or not cfg.get("enabled", False) or not occupants or not season_by_doy:
        return out
    lpp = float(cfg.get("litres_per_person_day", 28))
    t_ref = float(cfg.get("reference_temp", 60))
    inlet = list(cfg.get("inlet_temp_monthly") or []) or [10.0] * 12
    months = set(run_period_months) if run_period_months else None
    doy = 1
    for m in range(1, 13):
        ti = float(inlet[m - 1] if m - 1 < len(inlet) else inlet[-1])
        day_kwh = occupants * lpp * _DHW_RHO_CP * max(0.0, t_ref - ti) / 3600.0
        for _ in range(_MONTH_DAYS[m - 1]):
            if months is None or m in months:
                s = season_by_doy.get(doy, "in_between")
                out[s] = out.get(s, 0.0) + day_kwh
            doy += 1
    return out


def _building_dhw_occupants(row, tract_rate):
    """DHW occupant count for the building's residential (Housing) spaces.

    ONE head count, used by every scenario: the census-tract density
    (``population / the tract's residential floor area``) applied to each
    dwelling's own area. The occupancy is a property of the building's
    neighbourhood, not of the occupant archetype living in it — giving the
    normative case the CTE-default density instead put 3.5 people in each dwelling
    against the census's 1, so its demand differed from the behaviour scenarios
    for two reasons at once and the comparison no longer isolated behaviour.

    The CTE default density remains the FALLBACK for a building whose tract has no
    census population. Returns ``{"real": people, "cte": people}`` (both the same;
    the keys are kept so callers need not care where the number came from), or
    ``None`` when DHW is disabled or the building has no residential area."""
    cfg = globals().get("_RES_DHW", {}) or {}
    if not cfg.get("enabled", False):
        return None
    cte_people, area = _cte_default_occupants(row)
    if area <= 0:
        return None
    people, _ = _dwelling_occupants(row, tract_rate)
    if not people:                         # no tract/population data → CTE fallback
        people = cte_people
    return {"real": people, "cte": people}


def _augment_dhw_by_season(rows, dhw_occ, spec, season_by_doy, run_period_months,
                           n_dwellings=None):
    """Add ``dhw_demand_kwh_m2`` and ``people_per_dwelling`` to each RESIDENTIAL
    per-season row in ``rows`` (in place).

    ``dhw_demand_kwh_m2`` is the season's DHW demand over the row's floor area.
    ``people_per_dwelling`` is the occupancy the run was actually driven by,
    reported so the demand can be read against the household it assumes. Both come
    from the SAME head count, and both follow the scenario: the normative cte one
    uses the CTE-default density, every other scenario the census-tract occupancy.
    No-op when there is no DHW/occupancy."""
    if not dhw_occ:
        return rows
    people = dhw_occ["cte" if spec.get("behavior") == "cte" else "real"]
    dbs = _dhw_demand_kwh_by_season(people, season_by_doy, run_period_months)
    ppd = (round(people / n_dwellings, 2)
           if (n_dwellings and people is not None) else None)
    for r in rows:
        if _use_category(r.get("use_type")) == "residential" and r.get("floor_area_m2"):
            r["dhw_demand_kwh_m2"] = round(dbs.get(r["season"], 0.0)
                                           / r["floor_area_m2"], 2)
            if n_dwellings:
                r["n_dwellings"] = int(n_dwellings)
            if ppd is not None:
                r["people_per_dwelling"] = ppd
    return rows


# Sidecar written beside a generated .idf. The zone -> use-type map lives on the
# IDF object as ``_hc_zone_meta``, which a file cannot carry: EnergyPlus has no
# object to hold it and eppy drops stray comments on save. Persisting it means an
# .idf can be reloaded — or hand-edited and reloaded — and still be simulated
# through ``run_behavior_scenarios(idf_path=…)``.
_IDF_META_SUFFIX = ".hcmeta.json"

# The parts of a space record the scenario tooling reads. Kept to plain strings so
# the sidecar is legible and diffable, and so a user can correct it by hand after
# renaming a zone.
_IDF_META_FIELDS = ("use_type", "detailed", "reference", "kind")


def idf_metadata_path(idf_path):
    """Path of the sidecar that carries an IDF's zone metadata."""
    return f"{idf_path}{_IDF_META_SUFFIX}"


def _write_idf_metadata(idf_path, zone_meta, building_reference=None,
                        shading_filtered=False, epw=None):
    """Persist what the scenario runner needs but the IDF cannot hold."""
    import json
    payload = {
        "building_reference": building_reference,
        "shading_filtered": bool(shading_filtered),
        "epw": epw,
        "zone_meta": {z: {k: (None if (v or {}).get(k) is None else str((v or {})[k]))
                          for k in _IDF_META_FIELDS}
                      for z, v in (zone_meta or {}).items()},
    }
    try:
        with open(idf_metadata_path(idf_path), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - the model is still usable without it
        sys.stderr.write(f"  idf metadata: could not write sidecar ({e})\n")


def _read_idf_metadata(idf_path):
    """The sidecar beside ``idf_path``, or ``None``."""
    import json
    path = idf_metadata_path(idf_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"  idf metadata: could not read {path} ({e})\n")
        return None


def load_energy_model(idf_path, epw_path=None, idd_path=None):
    """Load an ``.idf`` written by :func:`build_energy_model` back into a usable
    model — the eppy IDF plus the zone metadata the scenario tooling needs.

    Edit the file in between and the edits are what gets simulated: this is the
    hand-tuning path, where a model is generated, adjusted by hand or by another
    tool, and then run through the occupant scenarios unchanged in every other
    respect.

    The zone → use-type map comes from the sidecar
    (``{idf}.hcmeta.json``). Renaming a zone in the IDF without renaming it there
    leaves that zone unmapped, and it is skipped rather than guessed at.
    """
    IDF = _ensure_idd(idd_path)
    idf = IDF(idf_path)
    meta = _read_idf_metadata(idf_path) or {}
    zone_meta = {str(z).upper(): dict(v) for z, v in (meta.get("zone_meta") or {}).items()}
    if zone_meta:
        idf._hc_zone_meta = zone_meta
    if meta.get("shading_filtered"):
        idf._hc_shading_filtered = True
    epw = epw_path or meta.get("epw")
    if epw:
        idf.epw = epw
    if not zone_meta:
        sys.stderr.write(
            f"  idf: no {_IDF_META_SUFFIX} beside {idf_path}; zone use types are "
            f"unknown, so nothing will be simulated. Regenerate the model with "
            f"build_energy_model (which writes it) or supply the sidecar.\n")
    return idf


def _copy_idf(idf):
    """An independent copy of ``idf``, so a scenario's destructive rewiring cannot
    leak into the next one.

    Round-trips through the IDF text rather than ``copy.deepcopy``: eppy objects
    hold a reference to the shared IDD tables, which deep-copying would duplicate
    (hundreds of MB) or choke on. The ``_hc_*`` bookkeeping and the attached EPW
    are plain attributes, invisible to the text form, so they are carried over
    explicitly.
    """
    import io as _io
    clone = type(idf)(_io.StringIO(idf.idfstr()))
    for attr in ("epw", "iddname", "_hc_zone_meta", "_hc_shading_filtered"):
        if hasattr(idf, attr):
            try:
                setattr(clone, attr, getattr(idf, attr))
            except Exception:  # noqa: BLE001 - a read-only attr is not fatal
                pass
    return clone


def _one_building(gdf, building_reference=None):
    """Resolve ``(row, building_reference)`` from a building frame.

    ``gdf`` is normally the ONE building to simulate, and its reference is read
    off the ``building_reference`` field rather than passed alongside. A frame
    holding several buildings is still accepted when ``building_reference`` says
    which one; without it, an ambiguous frame is an error rather than a silent
    "first row wins".
    """
    if "building_reference" not in getattr(gdf, "columns", []):
        raise ValueError("gdf has no 'building_reference' column")
    if building_reference is not None:
        sub = gdf[gdf["building_reference"] == building_reference]
        if sub.empty:
            raise ValueError(f"building {building_reference} not found in gdf")
        return sub.iloc[0], str(building_reference)
    refs = gdf["building_reference"].dropna().unique()
    if len(refs) == 0:
        raise ValueError("gdf holds no building_reference")
    if len(refs) > 1:
        raise ValueError(
            f"gdf holds {len(refs)} buildings; pass the single building to "
            f"simulate (e.g. gdf[gdf['building_reference'] == ref]) or name it "
            f"with building_reference=")
    ref = str(refs[0])
    return gdf[gdf["building_reference"] == ref].iloc[0], ref


def run_behavior_scenarios(gdf, wd=None, scenarios=None,
                           weather_year=None, epw_path=None, output_dir=None,
                           max_shading=_FROM_CONFIG, run_period_months=None,
                           comfort_category="II", seed=0, verbose=True,
                           filter_shading=None, idf=None, idf_path=None,
                           building_reference=None, context_gdf=None,
                           nonresidential_reporting="normative"):
    """Run the occupant-behaviour SCENARIOS for one building through EnergyPlus and
    return global KPIs per thermal-zone USE TYPE (occupied zones only).

    For every scenario (see ``_DEFAULT_SCENARIOS``) the building's model is built
    (or copied from ``idf``) and simulated once (the ``*_ff`` scenarios ARE the
    free-floating cases, so no separate twin is run). Occupancy follows fixed
    per-use-type schedules
    (residential / office / commercial), HVAC is gated OFF in the in-between season
    (Apr–May, Oct–Nov), and ventilation follows each behaviour (aware hygiene +
    mild-midday + summer night purge; unaware random daytime bursts plus a share
    of the nights with the windows open) — ``_apply_occupant_control``.

    KPIs per use type × season (area-weighted):

    ============================== =============================================
    ``season_days``                days of the year in the season …
    ``season_start`` / ``_end``    … and the ``MM-DD`` calendar span they fall in
    ``thermal_demand_kwh_m2``      HVAC demand (a row is one season, so the
                                   heating/cooling split would be redundant)
    ``internal_gains_kwh_m2``      people + lights + equipment
    ``solar_gains_kwh_m2``         solar transmitted through the glazing
    ``avg_setpoint_c``             mean setpoint over the hours the HVAC was
                                   RUNNING — heating setpoint in the hours it
                                   heated, cooling setpoint in the hours it
                                   cooled (not the schedule mean: setback and
                                   free-floating hours are excluded)
    ``hvac_hours``                 hours it ran, the weight behind that mean
    ``pct_hours_out_of_comfort_``  EN 16798-1 adaptive discomfort over OCCUPIED
    ``day`` / ``night``            hours, split at 22:00–07:59 — a TEMPERATURE
                                   criterion, relative to this climate
    ``pct_hours_out_of_comfort_``  …and the same hours against the EXTENDED comfort
    ``zone_day`` / ``_zone_night`` zone (``comfort_zone``: the Mollier h-x field of
                                   EN ISO 7730 / DIN 1946-2 / ASHRAE 55), which
                                   also has to be passed on humidity and enthalpy.
                                   NaN without the humidity output
    ``pct_occupied_hours``         the occupied share those percentages rest on
    ``ach_total_{mean,min,max}_*``,  OUTDOOR-air change rate (ACH) renewing the
    ``ach_total_p{10,20,80,90}_*``,  zone, split into the two routes that carry
    ``ach_{infiltration,``           it — this is what the CO2 below answers to.
    ``ventilation}_mean_*``          Never a median: on a flow that is 0 most
                                     hours it reports 0 and says nothing
    ``co2_p50_day`` / ``_p90`` /   indoor CO2 (ppm) over the OCCUPIED hours, and
    ``co2_max_*``,                 the share of them above each threshold — how
    ``co2_pct_above_{ppm}_*``      renewed the air was while people were in it
    ``t_p{10,20,50,80,90}_day``    indoor operative-temperature percentiles over
    ``t_{min,max}_day`` / ``…_night``  the OCCUPIED hours, plus the extremes the
                                   percentiles by definition never reach, day
                                   and night
    ``n_dwellings``                dwellings in the building (residential) …
    ``people_per_dwelling``        … the occupancy each one was given …
    ``dhw_demand_kwh_m2``          … and the hot-water demand it produces
    ============================== =============================================

    People counts come from the sociodemographic layer (``br__average_household_
    size``) and dwelling conservation status from the Essential-Characteristics
    ``ec__*`` columns when present, else model defaults (see ``build_energy_model``
    / the residential-behaviour engine).

    Parameters
    ----------
    gdf : the building to simulate — normally a ONE-ROW frame, whose
        ``building_reference`` field names it. A larger frame is accepted only
        together with ``building_reference``; otherwise it is an error, so a
        multi-building frame can never be silently reduced to its first row.
    wd : REQUIRED — wall typing, shadows and the EPW are fetched through it.
    building_reference : which building of ``gdf`` to simulate. Only needed when
        ``gdf`` holds more than one.
    context_gdf : the surrounding buildings (municipality or census tract),
        optional and normally unnecessary. The census-tract DHW occupancy is the
        tract population over the residential floor area of every building in the
        tract, which one row cannot supply — but ``hc.merge`` computes it once and
        writes it onto each building as ``ctagg__people_per_m2``, so a one-row
        frame from a current merge already carries it. This is the fallback for a
        frame merged before that existed (or with
        ``census_tract_aggregations=False``): pass the municipality frame and the
        rate is recomputed from it. With neither, the CTE-default occupancy
        density is used.
    scenarios : dict of scenario_name→spec, default ``_DEFAULT_SCENARIOS``.
    weather_year / epw_path : weather for the run (an EPW is required to simulate).
    idf_path : path to an ``.idf`` written by :func:`build_energy_model`,
        optional. The model is loaded from disk (with its ``.hcmeta.json``
        sidecar) and used exactly as ``idf`` would be — so a model can be
        generated, **edited by hand or by another tool**, and then run through the
        scenarios with the edits intact. Takes precedence over ``idf``.
    idf : an already-built ``geomeppy.IDF`` from :func:`build_energy_model`,
        optional. When given, each scenario works on a **copy** of it instead of
        rebuilding the model from ``gdf`` — the geometry, windows, constructions
        and neighbour shading are then provably the ones you inspected and
        plotted, and the (dominant) build cost is paid once rather than once per
        scenario. Its ``epw`` is reused unless ``weather_year``/``epw_path`` is
        given here. Behaviour-dependent rewiring — occupant archetype, setpoints,
        HVAC availability, ventilation — is re-applied per scenario on the copy,
        so a single IDF drives every scenario without them contaminating one
        another.
    filter_shading, max_shading : neighbour-shading filter and cap, as in
        :func:`build_energy_model` — which is where they are now applied. They
        are forwarded when this function builds the model itself; an ``idf``
        that arrives already filtered is left alone (it carries
        ``_hc_shading_filtered``), so what you plotted is what is simulated.
    run_period_months : optional [months] to simulate a partial year (fast checks);
        None = full annual run (needed for the true annual KPIs).
    comfort_category : EN 16798-1 category ("I"/"II"/"III").
    nonresidential_reporting : ``"normative"`` (default) or ``"per_scenario"``.
        The behaviour scenarios describe RESIDENTS; office and commercial premises
        run the same normative user (RITE / RD 486-2022 tertiary restriction) in
        every one of them. ``"normative"`` therefore reports them **once**, under
        the scenario label ``"normative"``, taken from the
        :data:`NONRESIDENTIAL_SOURCE_SCENARIO` run (``"cte"``, i.e. the one where
        the dwellings around them are normative too; the first conditioned
        scenario if ``cte`` is not being run). ``"per_scenario"`` restores the old
        behaviour of emitting them under every scenario name.

        Those per-scenario figures were never six different tertiary users — the
        premises are thermally coupled to the dwellings through shared
        floors/ceilings, so their demand shifts a little with what the residents
        do while their own setpoints and schedules never change. Reporting them
        once avoids reading that coupling as an occupant effect on the shop.

    Returns a ``pandas.DataFrame``: one row per scenario × use type × season for
    the residential zones, plus ONE set of ``"normative"`` rows for any
    non-residential use.
    """
    import pandas as _pd
    # Pre-1.2.4 signature was (gdf, building_reference, wd, scenarios, …). Its
    # third positional was a path where the current one is a scenarios dict, which
    # makes the old order unambiguous — accept it once, loudly, instead of failing
    # deep inside with a confusing message.
    if isinstance(wd, str) and isinstance(scenarios, str):
        warnings.warn(
            "run_behavior_scenarios(gdf, building_reference, wd, …) is "
            "deprecated: pass the ONE building's frame and let its "
            "building_reference field name it — "
            "run_behavior_scenarios(gdf_one_building, wd, …).",
            DeprecationWarning, stacklevel=2)
        building_reference, wd, scenarios = wd, scenarios, None
    row, building_reference = _one_building(gdf, building_reference)
    if idf_path is not None:
        # The hand-tuning path: whatever is in the file is what gets simulated.
        idf = load_energy_model(idf_path, epw_path=epw_path)
    scenarios = scenarios or _DEFAULT_SCENARIOS
    # Which run the single non-residential row set comes from: the normative one
    # if it is being run, else the first scenario that still has HVAC (a
    # free-floating run reports zero demand, which is not a normative figure).
    nonres_source = None
    if nonresidential_reporting == "normative":
        conditioned = [n for n, sp in scenarios.items() if not sp.get("free_floating")]
        nonres_source = (NONRESIDENTIAL_SOURCE_SCENARIO
                         if NONRESIDENTIAL_SOURCE_SCENARIO in scenarios
                         else (conditioned[0] if conditioned
                               else next(iter(scenarios), None)))
    output_dir = output_dir or (os.path.join(wd, "energyplus", "scenarios")
                                if wd else "scenarios")
    os.makedirs(output_dir, exist_ok=True)

    def _log(m):
        if verbose:
            print(m)

    def _prepare(model):
        _add_scenario_outputs(model)
        # Normally a no-op: build_energy_model already filtered and marked the
        # model, so the simulated context is the one that was written and plotted.
        # This still covers an IDF built by an older release or hand-assembled.
        n0, n1 = _apply_shading_filter(model, filter_shading=filter_shading,
                                       max_shading=max_shading)
        if verbose and n1 != n0:
            _log(f"    shading panels {n0} → {n1}")
        _set_run_period(model, run_period_months, year=weather_year)

    def _run(model, tag):
        d = os.path.join(output_dir, f"{building_reference}_{tag}")
        os.makedirs(d, exist_ok=True)
        try:
            model.run(output_directory=d, expandobjects=True, readvars=True,
                      verbose="q")
        except Exception as e:
            _log(f"    run {tag} failed: {e}")
        return _read_eplus_csv(os.path.join(d, "eplusout.csv"))

    all_rows = []
    # Only residential/office/commercial buildings get a thermal model (req #6): if
    # NONE of the building's spaces is a conditioned use, there is nothing to model.
    has_residential = True
    cats = {_resolve_use_category(z, row.get("building_use"))
            for z in (row.get("br__thermal_zones") or [])
            if isinstance(z, dict)}
    if cats and not (cats & set(_MODELABLE_USE_CATEGORIES)):
        _log(f"{building_reference}: no conditioned (residential/office/commercial) "
             f"use — no thermal model.")
        return _pd.DataFrame()
    # ``random_mix`` (the aware/unaware dwelling split) is a residential-only
    # scenario — skip it for buildings with no residential use.
    if cats:
        has_residential = "residential" in cats
    infil_factor = _conservation_infiltration_factor(row)
    if verbose and abs(infil_factor - 1.0) > 1e-6:
        _log(f"conservation-status infiltration factor: {infil_factor:.2f}")
    # Income-driven residential setpoint offset (per building; skips normative CTE).
    sp_offset = _income_setpoint_offset(_building_income(row))
    if verbose and (abs(sp_offset[0]) > 1e-6 or abs(sp_offset[1]) > 1e-6):
        _log(f"income setpoint offset (heat/cool °C): "
             f"{sp_offset[0]:+.2f}/{sp_offset[1]:+.2f}")
    # DHW occupant counts (per building; constant across scenarios). "real"
    # occupancy (census-tract population) feeds the unaware/aware scenarios; the
    # normative cte scenario — and the fallback when population data is missing —
    # uses the CTE-default density. The annual demand is split across the simulated
    # seasons per building (below), keyed by the EPW day→season map.
    # The census-tract occupancy needs EVERY building of the tract as its
    # denominator. hc.merge computes it once and writes it onto each building
    # (``ctagg__people_per_m2``), so the row usually carries it and no frame scan
    # is needed. Otherwise fall back to deriving it from ``context_gdf`` (or from
    # ``gdf`` itself); a rate from a partial frame is rejected as implausible
    # there, dropping to the CTE-default density rather than inflating the DHW.
    if _row_tract_rate(row) is not None:
        tract_rate = {}
    else:
        tract_rate = _census_tract_person_rate(
            context_gdf if context_gdf is not None else gdf, verbose=verbose)
    dhw_occ = _building_dhw_occupants(row, tract_rate)
    # Dwelling count of the building, so the occupancy the run assumes can be
    # reported per dwelling next to the demand it produces.
    n_dwellings = len(_residential_household_areas(row))
    # Measured weekly occupancy agenda of the building's census tract (INE Time Use
    # Survey via social_ES) — constant across scenarios, and None when the survey
    # does not reach this tract, in which case the stylised profiles are used.
    time_use = _building_time_use_agenda(wd, row, verbose=verbose)
    # …and the same agenda read as an occupancy MASK, which fixes the "occupied
    # hours" every comfort / CO2 / air-change KPI below is taken over.
    tus_occupied = _time_use_occupied_mask(time_use)
    if verbose and not tus_occupied:
        _log("occupied hours: no Time Use Survey agenda for this tract — the KPIs "
             "fall back to each run's own People output")
    dhw_season_by_doy = None                # cached from the EPW on first scenario
    for name, spec in scenarios.items():
        if spec.get("residential_only") and not has_residential:
            _log(f"[scenario] {name}: skipped (no residential use)")
            continue
        _log(f"[scenario] {name}")
        scenario_idf = os.path.join(output_dir, f"{building_reference}_{name}.idf")
        if idf is not None:
            # Work on a copy of the model the caller built and inspected, and
            # re-apply only the behaviour-dependent wiring. Every scenario then
            # shares one geometry/envelope/shading by construction, instead of
            # relying on the builder being deterministic across six rebuilds.
            model = _copy_idf(idf)
            if epw_path:
                model.epw = epw_path
            model._hc_behavior = _assign_residential_behavior(
                model, {k: v for k, v in getattr(model, "_hc_zone_meta", {}).items() if v},
                row,
                fixed_behavior=spec.get("behavior"),
                fixed_period=spec.get("period"), seed=seed)
        else:
            model = build_energy_model(
                gdf, building_reference, wd=wd, weather_year=weather_year,
                epw_path=epw_path,
                residential_behavior=spec.get("behavior"),
                residential_period=spec.get("period"), behavior_seed=seed,
                filter_shading=filter_shading, max_shading=max_shading,
                output_path=scenario_idf)
        zone_meta = {k: v for k, v in getattr(model, "_hc_zone_meta", {}).items() if v}
        areas = _zone_floor_areas(model)
        res_zones = {z for z, s in zone_meta.items()
                     if _resolve_use_category(s) == "residential"}
        # random_mix: each dwelling is randomly UNAWARE (50 %) or AWARE (50 %).
        if spec.get("mix_aware_unaware"):
            model._hc_behavior = {z: ("aware" if _adopted(z, 0.5) else "unaware", None)
                                  for z in res_zones}
        # cte: every dwelling is the CTE (Spanish-standards) user. "cte" is not a
        # residential_behavior config archetype, so assign it directly here.
        elif spec.get("behavior") == "cte":
            model._hc_behavior = {z: ("cte", None) for z in res_zones}
        # THE occupant engine — the same call ``build_energy_model`` makes, so a
        # scenario is its archetype assignment and nothing else. Occupancy, HVAC
        # availability, thermostats (adaptive for aware/passive), the tertiary
        # normative user and the whole ventilation model all come from here.
        #
        # The conservation-status infiltration factor is NOT re-applied: it scales
        # the infiltration objects in place, so a second pass would square it. It
        # is fabric, identical across scenarios, and rides along in the copy.
        _wire_occupant_model(model, zone_meta, row, seed=seed, time_use=time_use,
                             sp_offset=sp_offset)
        _prepare(model)

        if spec.get("free_floating"):
            _strip_hvac(model)
        # Save AFTER the scenario rewiring, so the .idf kept next to the results is
        # the one that ran — not the pre-scenario model it used to be.
        try:
            model.saveas(scenario_idf)
        except Exception as _e:  # noqa: BLE001 - an unwritable copy must not stop the run
            _log(f"    could not save {scenario_idf}: {_e}")
        main_df = _run(model, f"{name}")
        if main_df is None:
            _log(f"    no results for {name}; skipping")
            continue
        # The OCCUPIED HOURS every comfort / CO2 / air-change KPI is taken over.
        # One definition for the whole comparison — the census tract's Time Use
        # Survey week — so the columns of the report answer the same question.
        # The normative 'cte' case is the exception: it carries no occupancy
        # profile of its own, so its KPIs are taken over ALL hours.
        recs = _zone_scenario_kpis(
            main_df, zone_meta, areas, comfort_category, year=weather_year,
            occupied_mask=("all" if spec.get("behavior") == "cte"
                           else (tus_occupied or None)))
        rows = _aggregate_scenario_kpis(recs, name)
        rows = _report_nonresidential_once(rows, name, nonres_source)
        if dhw_occ:                            # residential DHW, split per season
            if dhw_season_by_doy is None:
                dhw_season_by_doy = _season_by_doy(getattr(model, "epw", None)) or {}
            _augment_dhw_by_season(rows, dhw_occ, spec, dhw_season_by_doy,
                                   run_period_months, n_dwellings=n_dwellings)
        all_rows.extend(rows)
    return _pd.DataFrame(all_rows)
