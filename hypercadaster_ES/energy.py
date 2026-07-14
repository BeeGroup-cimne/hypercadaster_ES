"""EnergyPlus model generation for inferred buildings.

Turns the per-building geometric/CAT attributes produced by the inference
(``br__*`` columns) into a 3-D, per-space thermal model and exports it as an
EnergyPlus **IDF** via ``eppy`` + ``geomeppy`` (both MIT — no AGPL coupling).

Pipeline (see ``build_energy_model``):
  1. Footprint rationalization — simplify the GROUND floor, then snap the other
     floors to the ground-floor node set so vertical surfaces align across
     floors (needed for clean surface adjacency).
  2. Per-space subdivision — split each floor among the building spaces present
     on it (communal core carved central, parking/warehouse as an end block
     along the building's principal axis toward the nearest street, open-air
     pools using their real OtherConstruction geometry, the rest area-weighted).
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
import sys

import numpy as np
from shapely import wkb as _wkb
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, snap

from hypercadaster_ES import utils

# Use types whose zones are not occupied by people and are placed as a compact
# end block rather than wrapped around the floor.
_PARKING_USE_KEYS = ("parking", "warehouse", "storage", "garage")

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
    g["_LOAD_LIBRARY"] = {k: dict(v) for k, v in cfg.get("loads", {}).items()}
    g["_SCHEDULES"] = cfg.get("schedules", {}) or {}
    g["_RESIDENTIAL_BEHAVIOR"] = cfg.get("residential_behavior", {}) or {}
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
def synthesize_space_geometry(row, simplify_tol=0.3, cat_shrink=None,
                              buffer_distance=None):
    """Per-space region polygons per floor.

    Returns ``list[dict]`` of zone records:
    ``{key, reference, use_type, detailed, kind, floor_polys: {floor: Polygon}}``
    where one record = one building space (so a duplex spans several floors).

    When ``cat_shrink`` (default: config ``geometry.cat_area_shrink``) the zones
    are tiled over the SHRUNK footprint — the single volumetry from
    ``_walled_footprint_by_floor`` (footprint eroded to the CAT built area, free
    façades moved inward, party/adiabatic anchored, freestanding block shifted
    north). This is the SAME geometry the 3-D plot draws (via
    ``shrunk_wall_segments``) and the EnergyPlus model exports, so all consumers
    converge. Set ``cat_shrink=False`` to tile the gross footprint instead.
    """
    fp = _rationalize_floors(_footprints_by_floor(row), simplify_tol)
    if not fp:
        return []
    # Single volumetry: tile the SHRUNK footprint so the 3-D plot and the
    # EnergyPlus model share one geometry (see module docstring / shrunk_wall_segments).
    do_shrink = _CAT_AREA_SHRINK if cat_shrink is None else cat_shrink
    bd = _BUFFER_DISTANCE if buffer_distance is None else buffer_distance
    walled_by_floor = {}
    if do_shrink:
        try:
            walled_by_floor, _g, _r, _dy = _walled_footprint_by_floor(row, bd)
        except Exception as _e:
            sys.stderr.write(f"  cat-shrink: skipped ({_e})\n")
            walled_by_floor = {}
    # Zone identity comes from the authoritative ``br__thermal_zones`` when
    # present (so the EnergyPlus zones are the same set); fall back to the
    # independent CAT-based derivation only when it is absent.
    spaces = _zones_from_thermal_zones(row) or _building_spaces(row)
    if not spaces:
        return []
    pools = _pool_geoms(row)
    communal = _communal_area_by_floor(row)
    street_dir = _nearest_street_dir(row)
    # Prefer the parcel's main orientation (real data) over per-floor PCA for the
    # zone/parking axis, so carving follows the building's XY, not a noisy PCA.
    parcel_major = _parcel_main_angle(row)

    # Per space, accumulate its polygon on each floor of its span.
    for s in spaces:
        s["floor_polys"] = {}
    # Communal circulation core (carved central) as its own zone; its identity
    # mirrors the authoritative ``br__thermal_zones`` communal core.
    _core_id = _communal_zone_id(row)
    communal_zone = {"zone_id": _core_id, "reference": _core_id,
                     "use_type": "Communal", "detailed": None, "kind": "communal",
                     "floors": [], "floor_polys": {}}
    # Tag the pool space (Sports facilities w/ pool) so it uses real geometry.
    pool_union = unary_union(pools) if pools else None

    floors = sorted(fp)
    for fl in floors:
        gross_poly = _clean(fp[fl])
        if gross_poly is None:
            continue
        # Tile the shrunk (walled) footprint when available; gross otherwise.
        floor_poly = _clean(walled_by_floor.get(fl)) or gross_poly
        present = [s for s in spaces if fl in s["floors"]]
        if not present:
            continue
        major = parcel_major if parcel_major is not None else _major_axis_angle(floor_poly)
        remainder = floor_poly

        # 1) communal circulation core → a COMPACT block fixed at the building
        # CENTRE (inward erosion until it reaches the communal area), per floor
        # when communal exists there. Occupied spaces then tile the surrounding
        # ring (step 4).
        carea = communal.get(fl, 0.0)
        core_poly = None
        if carea > 0 and remainder is not None and not remainder.is_empty:
            _cx, _cy = remainder.centroid.x, remainder.centroid.y
            _core, _rem = utils._carve_central_rect(
                remainder, carea, angle=major, origin=(_cx, _cy))
            _core = _clean(_core)
            if _core is not None and not _core.is_empty:
                communal_zone["floor_polys"][fl] = _core
                communal_zone["floors"].append(fl)
                core_poly = _core
                remainder = _clean(_rem)

        # 2) pools — use their real geometry (intersected with the GROSS floor;
        # open-air pools sit in the setback, outside the shrunk outline)
        for s in present:
            if pool_union is not None and str(s["use_type"]).lower().startswith("sport"):
                try:
                    pg = _clean(pool_union.intersection(gross_poly)) or _clean(pool_union)
                except Exception:
                    pg = _clean(pool_union)
                if pg is not None and not pg.is_empty:
                    s["floor_polys"][fl] = pg
                    s["kind"] = "non_occupied"
                    s["is_pool"] = True
                    if remainder is not None:
                        remainder = _clean(remainder.difference(pg))

        # 3) parking / non-occupied → end block along principal axis
        for s in present:
            if s["kind"] == "non_occupied" and fl not in s["floor_polys"]:
                if remainder is None or remainder.is_empty:
                    break
                block, remainder = _carve_end_block(
                    remainder, s["area"] / max(1, len(s["floors"])), major, street_dir)
                if block is not None and not block.is_empty:
                    s["floor_polys"][fl] = block

        # 4) occupied spaces → share the remaining footprint by their
        # WITHOUT-COMMUNAL areas as RECTANGULAR blocks. When a central communal
        # core was carved the remainder is an ANNULUS: it is split into four
        # simply-connected perimeter bands around the core and each band is
        # guillotine-subdivided into rectangles (``_subdivide_ring_rectangular``)
        # — this keeps blocks rectangular WITHOUT the annular-cut fragility
        # (straight cuts across the hole would split a zone into a disconnected
        # lobe that gets dropped). With no central core, fall back to the
        # axis-aligned recursive bisection of the whole remainder.
        occ = [s for s in present
               if s["kind"] == "occupied" and fl not in s["floor_polys"]]
        if occ and remainder is not None and not remainder.is_empty:
            weights = [max(s["area"] / max(1, len(s["floors"])), 1e-6) for s in occ]
            cx, cy = floor_poly.centroid.x, floor_poly.centroid.y
            if core_poly is not None:
                regs = utils._subdivide_ring_rectangular(
                    remainder, core_poly, weights, angle=major, origin=(cx, cy))
            else:
                regs = utils._subdivide_polygon_weighted_oriented(
                    remainder, weights, angle=major, origin=(cx, cy))
            for s, reg in zip(occ, regs):
                reg = _clean(reg)
                if reg is not None and not reg.is_empty:
                    s["floor_polys"][fl] = reg

    out = [s for s in spaces if s["floor_polys"]]
    if communal_zone["floor_polys"]:
        out.append(communal_zone)
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
                        _add_surface(idf, na, "Wall", ps[i]["name"],
                                     [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)],
                                     "Surface", nb)
                        _add_surface(idf, nb, "Wall", ps[j]["name"],
                                     [(x1, y1, z0), (x0, y0, z0), (x0, y0, z1), (x1, y1, z1)],
                                     "Surface", na)
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
                                 [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)],
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
                       residential_period=None, behavior_seed=0):
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

    # A taller ground floor is only typical of larger buildings — apply
    # ``ground_floor_height`` solely when the building has more than 4
    # above-ground floors; otherwise the ground floor uses the regular
    # ``floor_height``.
    try:
        _n_floors_ag = int(row.get("br__floors_above_ground"))
    except (TypeError, ValueError):
        _n_floors_ag = len([f for f in _fp if f >= 0]) if _fp else 1
    eff_gfh = (ground_floor_height
               if _n_floors_ag >= _GROUND_FLOOR_MIN_FLOORS else floor_height)

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
    prisms = []
    zone_meta = {}  # zone_name -> space dict
    used_names = set()
    zi = 0
    for s in zones:
        # Open-air swimming pools are not enclosed conditioned spaces — they are
        # kept for the 3-D plot but must NOT become EnergyPlus thermal zones.
        if s.get("is_pool"):
            continue
        base = _safe(s.get("reference") or s.get("zone_id") or s.get("use_type"))
        for fl, poly in sorted(s["floor_polys"].items()):
            p = _clean(poly)
            if p is None or p.area < 1.0:
                continue
            p = affinity.translate(p, xoff=-ox, yoff=-oy)   # → local origin
            z0, z1 = _floor_z(fl, eff_gfh, floor_height)
            zname = f"{base}_F{fl}"
            while zname.upper() in used_names:               # guarantee uniqueness
                zname = f"{base}_F{fl}_{zi}"
            used_names.add(zname.upper())
            prisms.append({"name": zname, "poly": p, "z0": z0, "z1": z1,
                           "floor": fl, "floor_bc": "Ground" if fl <= 0 else "Outdoors"})
            zone_meta[zname.upper()] = s
            zi += 1

    # Build zones + surfaces with inter-zone adjacency wired by construction.
    # (Interior walls that adjacency detection misses are made Adiabatic there,
    # so they never surface as windowed exterior walls.)
    _build_surfaces(idf, prisms)
    # Refine exterior boundary conditions from the inferred wall typing. Party
    # walls become windowless Outdoors façades (their names are returned so
    # windows/doors skip them).
    party_walls = _refine_exterior_bcs(idf, row, zone_meta, floor_height, eff_gfh,
                                       origin=(ox, oy))
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
        _add_windows(idf, eff_wwr, zone_meta, skip_walls=party_walls)
    except Exception as _e:
        sys.stderr.write(f"  windows: could not place windows ({_e})\n")

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
        add_energy_systems(idf, zone_meta, elevation=_elev,
                           envelope=arch["envelope"], archetype=arch,
                           output_frequencies=output_frequencies, row=row,
                           residential_behavior=residential_behavior,
                           residential_period=residential_period,
                           behavior_seed=behavior_seed)

        # Attach the ERA5-Land EPW so idf.run() uses real local weather. The
        # EPW is cached per (ERA5-Land cell, year); coastal centroids resolve to
        # the nearest cell with data. Explicit epw_path overrides resolution.
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
    by_use = {}
    for s in zones:
        if s.get("kind") and s["kind"] != "occupied":
            continue
        ut = s.get("use_type")
        area = 0.0
        for poly in (s.get("floor_polys") or {}).values():
            try:
                area += float(poly.area)
            except Exception:
                pass
        by_use[ut] = by_use.get(ut, 0.0) + area
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
    the set of party-wall surface names."""
    from shapely import affinity
    from shapely.geometry import LineString
    party_names = set()
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
            continue                                  # stays Outdoors + windows
        if best_t == "party_wall":
            srf.Outside_Boundary_Condition = "Adiabatic"   # neighbour at similar T
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
            party_names.add(srf.Name)                 # no windows/doors
        elif best_t == "adiabatic":
            srf.Outside_Boundary_Condition = "Adiabatic"
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
        elif best_t == "terrain_contact":
            srf.Outside_Boundary_Condition = "Ground"
            srf.Outside_Boundary_Condition_Object = ""
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
    return party_names


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
    s = zone_meta.get(zname.upper())
    return bool(s) and _use_category(s.get("use_type")) == "parking"


def _wall_base(wall):
    """(P0, P1, z0, z_top) of a vertical wall's ground edge from its coords."""
    coords = [(float(x), float(y), float(z)) for x, y, z in wall.coords]
    z0 = min(c[2] for c in coords)
    ztop = max(c[2] for c in coords)
    base = [c for c in coords if abs(c[2] - z0) < 1e-6]
    if len(base) < 2:
        return None
    return (base[0][0], base[0][1]), (base[1][0], base[1][1]), z0, ztop


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


def _add_windows(idf, wwr, zone_meta, skip_walls=None):
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

        glaz = wwr * L * H
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


def _add_subsurface_reveals(idf, depth):
    """Add a recess/reveal shade around the perimeter of every window and door:
    Spanish glazing usually aligns to the INNER wall face, so the wall thickness
    forms a ~``depth`` reveal that shades the opening on all four sides. Each
    edge becomes a thin Shading:Building:Detailed fin projecting ``depth`` along
    the host wall's OUTWARD normal."""
    if depth <= 0:
        return 0
    wall_by_name = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    n = 0
    for sub in list(idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]):
        st = sub.Surface_Type.lower()
        if st not in ("window", "door", "glassdoor", "glazeddoor"):
            continue
        wall = wall_by_name.get(sub.Building_Surface_Name)
        if wall is None:
            continue
        wc = [np.array([float(x), float(y), float(z)]) for x, y, z in wall.coords]
        if len(wc) < 3:
            continue
        nrm = np.cross(wc[1] - wc[0], wc[2] - wc[0])   # wall outward normal
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        off = nrm / ln * depth
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
    the period's window ``infiltration`` (frame/glazing type + ACH)."""
    period = _construction_period(year)
    return {
        "period": period,
        "use_category": _use_category(use_type),
        "envelope": dict(_ARCHETYPE_ENVELOPE[period]),
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


def _construction_for(surface):
    st = surface.Surface_Type.lower()
    bc = surface.Outside_Boundary_Condition.lower()
    if st == "wall":
        return {"outdoors": "HC_ExtWall", "ground": "HC_GroundWall",
                "surface": "HC_IntWall", "adiabatic": "HC_PartyWall"}.get(bc, "HC_ExtWall")
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


def _add_loads(idf, zone_meta, window_infil_ach=None):
    """People / Lights / ElectricEquipment / Infiltration per zone, by use.
    Conditioned zones take their infiltration from the window technology of the
    construction period (``window_infil_ach``); non-conditioned zones
    (parking/communal/sports) keep their use-based ``infil_ach``."""
    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        cat = _use_category(s["use_type"]) if s else "other"
        p = _LOAD_LIBRARY.get(cat, _LOAD_LIBRARY["other"])
        occ = p["occ"]
        if p["ppl_per_m2"] > 0:
            o = idf.newidfobject(
                "PEOPLE", Name=f"{zone.Name}_People",
                Number_of_People_Schedule_Name=occ,
                Number_of_People_Calculation_Method="People/Area",
                Activity_Level_Schedule_Name=f"ACT_{cat.upper()}"
                if p.get("conditioned") else "ACT_OFFICE")
            _set_first(o, _F_ZONE, zone.Name)
            _set_first(o, _F_PPL_AREA, p["ppl_per_m2"])
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
        idf.newidfobject(
            "HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM",
            Zone_Name=zone.Name, Template_Thermostat_Name=tstat)


# --------------------------------------------------------------------------- #
# Residential occupant-behaviour archetypes (unaware / aware / always_on /
# passive) + occupancy periods — assigned per dwelling from section demographics
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
    """Resolve (behaviour, period) for one residential dwelling — fixed when
    given, else a demographic-weighted deterministic random draw (seeded by the
    dwelling reference so a rebuild is reproducible)."""
    import random
    rb = _RESIDENTIAL_BEHAVIOR
    behaviors = rb.get("behaviors", {})
    periods = rb.get("occupancy_periods", {})
    asg = rb.get("assignment", {})
    if fixed_behavior and fixed_behavior in behaviors:
        behavior = fixed_behavior
    else:
        ab = _bucket_name(mean_age, asg.get("age_buckets", {}), "max_mean_age")
        bw = (asg.get("behavior_weights", {}).get(ab)
              or asg.get("behavior_weights", {}).get("default", {}))
        behavior = (_weighted_choice(bw, random.Random(f"{seed}|{reference}|b"))
                    or next(iter(behaviors), "aware"))
    if fixed_period and fixed_period in periods:
        period = fixed_period
    else:
        hb = _bucket_name(household_size, asg.get("household_buckets", {}), "max_size")
        pw = (asg.get("period_weights", {}).get(hb)
              or asg.get("period_weights", {}).get("default", {}))
        period = (_weighted_choice(pw, random.Random(f"{seed}|{reference}|p"))
                  or next(iter(periods), "nights_workdays"))
    return behavior, period


def _period_map(profile, fn):
    """Apply ``fn(steps)`` to a period profile that is either a flat step list or
    a ``{weekdays, weekends}`` dict → same shape, for ``_steps_to_fields``."""
    if isinstance(profile, dict):
        return {k: fn(v) for k, v in profile.items() if k in ("weekdays", "weekends")}
    return fn(profile or [])


def _apply_residential_behavior(idf, zone_meta, row, fixed_behavior=None,
                                fixed_period=None, seed=0):
    """Assign each RESIDENTIAL zone an occupant behaviour + occupancy period and
    rewire its schedules accordingly: internal-gains occupancy, heating/cooling
    setpoints (constant, or occupied/away following the period), ideal-loads HVAC
    availability (always, or only when occupied), and natural ventilation
    (ZoneVentilation:DesignFlowRate scheduled by the behaviour). Returns a dict
    ``{zone_name: (behavior, period)}``. No-op when disabled or no residential
    zones."""
    rb = _RESIDENTIAL_BEHAVIOR
    if not rb or not rb.get("enabled", True):
        return {}
    behaviors = rb.get("behaviors", {})
    periods = rb.get("occupancy_periods", {})
    vent_scheds = rb.get("ventilation_schedules", {})
    if not behaviors or not periods:
        return {}
    hh, mean_age = _residential_demographics(row)
    m2pp = float(rb.get("m2_per_person", 35) or 35)

    # Index the per-zone load/HVAC objects by zone name for quick rewiring.
    def _by_zone(objkey, attr):
        out = {}
        for o in idf.idfobjects[objkey]:
            out[str(getattr(o, attr, "")).upper()] = o
        return out
    people = {o.Name: o for o in idf.idfobjects["PEOPLE"]}
    lights = {o.Name: o for o in idf.idfobjects["LIGHTS"]}
    equips = {o.Name: o for o in idf.idfobjects["ELECTRICEQUIPMENT"]}
    ideal = {}
    for o in idf.idfobjects["HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM"]:
        ideal[str(o.Zone_Name).upper()] = o

    made = set()          # schedule/thermostat names already created
    assigned = {}

    def _occ_sched(period):
        nm = f"OCCRES_{period.upper()}"
        if nm not in made:
            _sched(idf, nm, "Fraction",
                   ["Through: 12/31"] + _steps_to_fields(periods.get(period, {})))
            made.add(nm)
        return nm

    def _setpoint_sched(behavior, period, kind):
        spec = behaviors[behavior].get(kind)
        nm = f"{kind[:4].upper()}RES_{behavior.upper()}_{period.upper()}"
        if nm in made:
            return nm
        if isinstance(spec, dict):                     # occupied/away → follow period
            occ_v, away_v = spec.get("occupied"), spec.get("away", spec.get("occupied"))
            prof = _period_map(periods.get(period, {}),
                               lambda steps: [[u, (occ_v if float(f) >= 0.5 else away_v)]
                                              for u, f in steps])
            fields = _steps_to_fields(prof)
        else:                                          # constant all day
            fields = _steps_to_fields([["24:00", spec]])
        _sched(idf, nm, "Temperature", ["Through: 12/31"] + fields)
        made.add(nm)
        return nm

    def _avail_sched(period):
        nm = f"AVAILRES_{period.upper()}"
        if nm not in made:
            prof = _period_map(periods.get(period, {}),
                               lambda steps: [[u, (1 if float(f) > 0.1 else 0)]
                                              for u, f in steps])
            _sched(idf, nm, "Fraction", ["Through: 12/31"] + _steps_to_fields(prof))
            made.add(nm)
        return nm

    def _vent_sched(name):
        nm = f"VENTRES_{name.upper()}"
        if nm not in made:
            _sched(idf, nm, "Fraction",
                   ["Through: 12/31"] + _steps_to_fields(vent_scheds.get(name, [])))
            made.add(nm)
        return nm

    for zone in idf.idfobjects["ZONE"]:
        s = zone_meta.get(zone.Name.upper())
        if not s or _use_category(s.get("use_type")) != "residential":
            continue
        ref = s.get("reference") or zone.Name
        behavior, period = _assign_residential_user(
            ref, hh, mean_age, fixed_behavior, fixed_period, seed)
        assigned[zone.Name] = (behavior, period)
        bdef = behaviors[behavior]

        # 1) internal-gains occupancy schedule
        occ = _occ_sched(period)
        for nm, obj in ((f"{zone.Name}_People", people.get(f"{zone.Name}_People")),
                        (f"{zone.Name}_Lights", lights.get(f"{zone.Name}_Lights")),
                        (f"{zone.Name}_Equip", equips.get(f"{zone.Name}_Equip"))):
            if obj is None:
                continue
            if obj.key.upper() == "PEOPLE":
                obj.Number_of_People_Schedule_Name = occ
            else:
                obj.Schedule_Name = occ

        # 2) setpoints + 3) HVAC availability, via the zone's ideal-loads system
        heat = _setpoint_sched(behavior, period, "heating")
        cool = _setpoint_sched(behavior, period, "cooling")
        tstat = f"TSTATRES_{behavior.upper()}_{period.upper()}"
        if tstat not in made:
            idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name=tstat,
                             Heating_Setpoint_Schedule_Name=heat,
                             Cooling_Setpoint_Schedule_Name=cool)
            made.add(tstat)
        iz = ideal.get(zone.Name.upper())
        if iz is not None:
            iz.Template_Thermostat_Name = tstat
            if str(bdef.get("hvac_availability", "always")).lower() == "occupancy":
                iz.System_Availability_Schedule_Name = _avail_sched(period)
            else:
                iz.System_Availability_Schedule_Name = ""

        # 4) natural ventilation
        ach = float(bdef.get("ventilation_ach", 0.0) or 0.0)
        if ach > 0:
            vs = bdef.get("ventilation_schedule")
            o = idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE", Name=f"{zone.Name}_NatVent",
                Schedule_Name=_vent_sched(vs) if vs else "ALWAYS_ON",
                Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
                Air_Changes_per_Hour=ach, Ventilation_Type="Natural")
            _set_first(o, _F_ZONE, zone.Name)
    return assigned


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
                       behavior_seed=0):
    """Add constructions, schedules, loads, ideal-loads HVAC and simulation
    control to a geometry-only IDF, making it a runnable annual model.

    ``envelope`` is the archetype envelope dict (target U-values + glazing) from
    ``building_archetype(...)["envelope"]``; if omitted it defaults to the
    1980-2006 (NBE-CT-79) period. ``archetype`` carries the period, used here to
    set the conditioned-zone air infiltration from the window technology.
    ``output_frequencies`` selects the Output:Variable reporting frequencies
    (defaults to ``simulation.output_frequencies`` in the config). ``row`` (the
    building attributes) + ``residential_behavior`` / ``residential_period`` /
    ``behavior_seed`` drive the residential occupant-behaviour archetypes
    (``_apply_residential_behavior``); when the first two are None each dwelling
    draws a behaviour/period at random from section demographics."""
    if envelope is None:
        envelope = dict(_ARCHETYPE_ENVELOPE["1980_2006"])
    period = archetype["period"] if archetype else "1980_2006"
    win_infil = _window_infiltration(period)["air_changes_per_hour"]
    _add_constructions(idf, envelope)
    _assign_constructions(idf)
    _add_schedules(idf)
    _add_loads(idf, zone_meta, window_infil_ach=win_infil)
    _add_hvac(idf, zone_meta)
    # Residential occupant behaviour (setpoints / availability / ventilation /
    # occupancy) per dwelling — overrides the generic residential schedules.
    if row is not None:
        _apply_residential_behavior(idf, zone_meta, row,
                                    fixed_behavior=residential_behavior,
                                    fixed_period=residential_period,
                                    seed=behavior_seed)
    _add_interfloor_connections(idf, zone_meta)   # stair/lift coupling of duplexes
    _add_sim_control(idf, latitude, longitude, time_zone, elevation,
                     output_frequencies=output_frequencies)
    return idf


# Colours for the IDF viewer, keyed by the fine-grained surface CATEGORY
# (surface type + subtype from the boundary condition / construction). Unknown
# categories fall back to a generated colour.
_IDF_VIEW_COLORS = {
    "Wall: exterior":       "lightyellow",
    "Wall: party/adiabatic": "mediumpurple",
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
        return "Wall: " + {"outdoors": "exterior", "ground": "ground",
                           "adiabatic": "party/adiabatic",
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


def plot_idf_geometry(idf, output_path=None, show=False, opacity=0.5,
                      include_shading=True, title=None, **_ignored):
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
    opacity : surface fill opacity.
    title : figure title.

    Returns the plotly ``Figure``.
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
        traces.append(go.Mesh3d(
            x=xs, y=ys, z=zs, i=i, j=j, k=k,
            color=color, opacity=opacity, flatshading=True,
            name=cat, legendgroup=cat, showlegend=True,
            hoverinfo="name"))
        # ---- crisp edges: one Scatter3d per category, shares its legend group -
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="rgba(40,40,40,0.6)", width=2),
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
