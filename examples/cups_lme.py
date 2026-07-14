"""
Geolocate CUPS records by matching Excel addresses against a hypercadaster
reference dataset.

Improvements over the original script
--------------------------------------
1. Exact match first, then a fuzzy fallback (rapidfuzz) that matches the
   street *name* and then snaps to the nearest house number on that street.
2. A real density heatmap (log-scaled hexbin) instead of a low-alpha
   scatter plot.
3. Street normalization is generated programmatically and covers many more
   via types and common abbreviations.
4. Proper CRS handling so the basemap lines up.
5. A `match_type` / `match_score` audit trail and cleaner logging.

Requires: pandas, geopandas, matplotlib, shapely, rapidfuzz
          (optional: contextily for the basemap)
    pip install rapidfuzz contextily
"""

import re
import logging
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from shapely.geometry import Point
from rapidfuzz import process, fuzz

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH     = "/home/gmor/Downloads/LMETramitBeeData.xlsx"
PICKLE_FILE    = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES/08900_complete_inference.pkl"
OUTPUT_EXCEL   = "/home/gmor/Downloads/LMETramitBeeData_geolocated.xlsx"
OUTPUT_SUMMARY = "/home/gmor/Downloads/CUPS_by_neighborhood.csv"
OUTPUT_HEATMAP = "/home/gmor/Downloads/CUPS_heatmap_barcelona.png"
OUTPUT_BASEMAP = "/home/gmor/Downloads/CUPS_heatmap_basemap.png"

# CRS of the reference geometries if the pickle does not carry one.
# Catalonia / Barcelona cadastre is normally ETRS89 / UTM zone 31N.
FALLBACK_CRS = 25831

# Minimum rapidfuzz score (0-100) required to accept a fuzzy street match.
FUZZY_THRESHOLD = 88

# Heatmap hexagon grid. Lower = bigger hexagons / coarser grid.
HEX_GRIDSIZE = 45

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("geolocate")


# ============================================================
# NORMALIZATION
# ============================================================

# Common via types. Multi-word types must come before their prefixes.
VIA_TYPES = [
    "GRAN VIA", "TRAVESSERA", "PASSATGE", "AVINGUDA", "CARRETERA",
    "CARRER", "PASSEIG", "RAMBLA", "RONDA", "PLACA", "BAIXADA",
    "TORRENT", "JARDINS", "CAMI", "MOLL", "VIA", "PLA",
]

# Map abbreviations / language variants to a canonical via type.
VIA_ALIASES = {
    "C": "CARRER", "CL": "CARRER", "CALLE": "CARRER", "CARRE": "CARRER",
    "AV": "AVINGUDA", "AVD": "AVINGUDA", "AVDA": "AVINGUDA", "AVGDA": "AVINGUDA",
    "PG": "PASSEIG",
    "PL": "PLACA", "PCA": "PLACA", "PLAZA": "PLACA",
    "RB": "RAMBLA",
    "PTGE": "PASSATGE", "PSG": "PASSATGE",
    "RD": "RONDA",
    "TRAV": "TRAVESSERA", "TR": "TRAVESSERA",
    "GV": "GRAN VIA",
    "CTRA": "CARRETERA",
}

# Articles / connectors, longest first so "DE LES" is stripped before "DE".
_ARTICLES = sorted(
    ["DE LES", "DE LA", "DE L", "DELS", "DEL", "DE", "D"],
    key=len, reverse=True,
)

# Pre-compiled "<VIA> <ARTICLE> " -> "<VIA> " replacement patterns.
_VIA_PATTERNS = [
    (re.compile(rf"^{via} {art} "), f"{via} ")
    for via in VIA_TYPES
    for art in _ARTICLES
]

_NUMBER_RE = re.compile(r"(\d+)")


def normalize_text(text):
    """Uppercase, strip accents and symbols, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_street_name(text):
    """Normalize a street name: expand via aliases and drop articles.

    House numbers are intentionally NOT handled here (see split_address).
    """
    text = normalize_text(text)
    tokens = text.split()
    if not tokens:
        return ""

    # Expand a leading abbreviation, e.g. "AV" -> "AVINGUDA".
    if tokens[0] in VIA_ALIASES:
        tokens = VIA_ALIASES[tokens[0]].split() + tokens[1:]
        text = " ".join(tokens)

    for pattern, replacement in _VIA_PATTERNS:
        text = pattern.sub(replacement, text)

    return re.sub(r"\s+", " ", text).strip()


def split_address(text):
    """Split a raw address into (normalized_street_name, house_number).

    house_number is an int or None. The first standalone number is used,
    so 'CARRER DE BALMES 12 3 2' -> ('CARRER BALMES', 12).
    """
    text = normalize_text(text)
    m = _NUMBER_RE.search(text)
    number = int(m.group(1)) if m else None
    name_part = text[:m.start()] if m else text
    return normalize_street_name(name_part), number


def make_key(name, number):
    """Build the exact-match key from a normalized name + number."""
    if name and number is not None:
        return f"{name} {number}"
    return name


_SEQ_TYPES = (list, tuple, np.ndarray, pd.Series)


def seq_get(seq, i, default=None):
    """Get element i from a list/tuple/ndarray/Series; otherwise return seq.

    The reference dataset stores per-address attributes as numpy arrays,
    not plain lists, so a simple `isinstance(x, list)` check misses them.
    """
    if isinstance(seq, _SEQ_TYPES):
        return seq[i] if i < len(seq) else default
    return seq


def scalarize(value):
    """Reduce a possibly-nested sequence to a single hashable scalar.

    Without this, list/array values end up in cells and break groupby
    (`TypeError: unhashable type: 'list'`).
    """
    while isinstance(value, _SEQ_TYPES):
        value = value[0] if len(value) else None
    return value


# ============================================================
# LOAD + PREPARE EXCEL
# ============================================================

log.info("Reading Excel: %s", EXCEL_PATH)
df = pd.read_excel(EXCEL_PATH)
log.info("Excel rows: %d", len(df))

df.columns = (
    df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
)

required = ["Tipus via", "Nom carrer", "Núm carrer"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(
        f"Missing expected columns: {missing}. Found: {df.columns.tolist()}"
    )

# Normalized street name (via type + name), without the number.
df["street_name_clean"] = (
    (df["Tipus via"].astype(str) + " " + df["Nom carrer"].astype(str))
    .apply(normalize_street_name)
)

# House number as a nullable integer.
df["street_number"] = (
    df["Núm carrer"].astype(str)
    .str.extract(r"(\d+)")[0]
    .astype("Int64")
)

df["match_key"] = [
    make_key(n, (None if pd.isna(num) else int(num)))
    for n, num in zip(df["street_name_clean"], df["street_number"])
]

log.info("Example normalized addresses: %s",
         df["match_key"].head(5).tolist())


# ============================================================
# LOAD + EXPLODE REFERENCE DATASET
# ============================================================

log.info("Reading reference pickle: %s", PICKLE_FILE)
# The central DuckDB is now the canonical artifact — hc.merge() writes
# all merged data to building_results.duckdb.  We still read the pickle
# here; alternatively query via hc.utils.read_br_inferred_indicators().
gdf = pd.read_pickle(PICKLE_FILE, compression="gzip")

source_crs = getattr(gdf, "crs", None) or FALLBACK_CRS
log.info("Reference CRS: %s", source_crs)

records = []
for _, row in gdf.iterrows():
    addresses = row.get("postal_addresses_clean")
    locations = row.get("postal_addresses_location")
    neighborhoods = row.get("neighborhood_name")
    building_ref = scalarize(row.get("building_reference"))

    if not isinstance(addresses, _SEQ_TYPES):
        continue

    # neighborhood_name is a building-level attribute (usually a length-1
    # list). Index it per-address ONLY when it aligns 1:1 with the address
    # list; otherwise the single value applies to every address.
    nbh_aligned = (
        isinstance(neighborhoods, _SEQ_TYPES)
        and len(neighborhoods) == len(addresses)
    )
    nbh_building = scalarize(neighborhoods)

    for i, address in enumerate(addresses):
        geom = seq_get(locations, i, None)
        nbh = scalarize(neighborhoods[i]) if nbh_aligned else nbh_building
        name, number = split_address(address)
        if not name:
            continue
        records.append({
            "street_name_clean": name,
            "street_number": number,
            "match_key": make_key(name, number),
            "address_location": geom,
            "neighborhood_name": nbh,
            "building_reference": building_ref,
        })

ref = pd.DataFrame(records)
ref["street_number"] = ref["street_number"].astype("Int64")
ref_geo = ref.dropna(subset=["address_location"])
log.info("Exploded reference addresses: %d (with geometry: %d)",
         len(ref), len(ref_geo))

# One row per full address key for the exact join.
ref_exact = ref_geo.drop_duplicates("match_key")
log.info("Unique reference keys: %d", len(ref_exact))


# ============================================================
# STEP 1 - EXACT MATCH
# ============================================================

df_merged = df.merge(
    ref_exact[["match_key", "address_location", "neighborhood_name",
               "building_reference"]],
    on="match_key", how="left",
)

df_merged["match_type"] = np.where(
    df_merged["address_location"].notna(), "exact", "unmatched"
)
df_merged["match_score"] = np.where(
    df_merged["match_type"] == "exact", 100.0, np.nan
)

log.info("Exact matches: %d / %d",
         (df_merged["match_type"] == "exact").sum(), len(df_merged))


# ============================================================
# STEP 2 - FUZZY FALLBACK
# ============================================================

# Group reference rows by street name for nearest-number snapping.
ref_by_street = {name: grp for name, grp in ref_geo.groupby("street_name_clean")}
ref_street_names = list(ref_by_street.keys())
log.info("Fuzzy-matching against %d unique street names", len(ref_street_names))

_street_cache = {}


def fuzzy_street(name):
    """Return (best_reference_name, score) or (None, None)."""
    if name in _street_cache:
        return _street_cache[name]
    hit = process.extractOne(
        name, ref_street_names,
        scorer=fuzz.WRatio, score_cutoff=FUZZY_THRESHOLD,
    )
    result = (hit[0], hit[1]) if hit else (None, None)
    _street_cache[name] = result
    return result


def pick_by_number(group, number):
    """Pick the reference row whose house number is closest to `number`."""
    have_numbers = group["street_number"].notna()
    if number is not None and have_numbers.any():
        cand = group[have_numbers].copy()
        cand["_dist"] = (cand["street_number"].astype("int64") - int(number)).abs()
        return cand.sort_values("_dist").iloc[0]
    return group.iloc[0]


unmatched_idx = df_merged.index[df_merged["match_type"] == "unmatched"]
n_fuzzy = 0

for idx in unmatched_idx:
    name = df_merged.at[idx, "street_name_clean"]
    if not name:
        continue
    number = df_merged.at[idx, "street_number"]
    number = None if pd.isna(number) else int(number)

    best_name, score = fuzzy_street(name)
    if best_name is None:
        continue

    chosen = pick_by_number(ref_by_street[best_name], number)
    df_merged.at[idx, "address_location"] = chosen["address_location"]
    df_merged.at[idx, "neighborhood_name"] = chosen["neighborhood_name"]
    df_merged.at[idx, "building_reference"] = chosen["building_reference"]
    df_merged.at[idx, "match_type"] = "fuzzy"
    df_merged.at[idx, "match_score"] = float(score)
    n_fuzzy += 1

log.info("Fuzzy matches recovered: %d", n_fuzzy)


# ============================================================
# COORDINATES + SAVE ENRICHED EXCEL
# ============================================================

df_merged["x"] = df_merged["address_location"].apply(
    lambda g: g.x if isinstance(g, Point) else None
)
df_merged["y"] = df_merged["address_location"].apply(
    lambda g: g.y if isinstance(g, Point) else None
)

# Do not write shapely objects into the Excel file.
df_merged.drop(columns=["address_location"]).to_excel(OUTPUT_EXCEL, index=False)
log.info("Saved enriched Excel: %s", OUTPUT_EXCEL)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = (
    df_merged.dropna(subset=["neighborhood_name"])
    .groupby("neighborhood_name")
    .size()
    .reset_index(name="num_cups")
    .sort_values("num_cups", ascending=False)
)
summary.to_csv(OUTPUT_SUMMARY, index=False)
log.info("Saved neighborhood summary: %s", OUTPUT_SUMMARY)
print("\nCUPS per neighborhood (top 20):")
print(summary.head(20).to_string(index=False))


# ============================================================
# HEATMAP (log-scaled hexbin density)
# ============================================================

# Plot in the source CRS (UTM 31N) so the city keeps its true proportions.
# contextily reprojects the basemap tiles to match.
gdf_plot = gpd.GeoDataFrame(
    df_merged.dropna(subset=["address_location"]).copy(),
    geometry="address_location",
    crs=source_crs,
)

xs = gdf_plot.geometry.x.values
ys = gdf_plot.geometry.y.values

# Size the figure to the real data extent so there are no empty side
# margins. set_aspect("equal") below guarantees the shape is not deformed.
_dx = xs.max() - xs.min()
_dy = ys.max() - ys.min()
_pad = 0.04                       # 4% margin around the points
_BASE = 11                        # inches on the longer side
if _dy >= _dx:
    FIG_W, FIG_H = _BASE * _dx / _dy, _BASE
else:
    FIG_W, FIG_H = _BASE, _BASE * _dy / _dx

# Maximum distinct count values before the discrete legend gets unreadable.
HEX_MAX_LEVELS = 16


def draw_heatmap(path, basemap=False):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # A scalar gridsize makes hexbin assume a square extent, which stretches
    # the hexagons on a tall city. Derive the row count from the real
    # extent ratio so the hexagons render regular.
    nx = HEX_GRIDSIZE
    ny = max(1, int(round(nx * (_dy / _dx) / np.sqrt(3))))

    hb = ax.hexbin(
        xs, ys,
        gridsize=(nx, ny),         # bigger hexagons -> coarser grid
        cmap="viridis",
        mincnt=1,                  # hide empty cells
        linewidths=0.2,
        alpha=0.8 if basemap else 1.0,
        zorder=2,
    )

    # One discrete colour per unique integer count. Each cell is coloured by
    # its exact CUPS count rather than a binned interval.
    counts = hb.get_array().compressed()
    cmin, cmax = int(counts.min()), int(counts.max())
    levels = np.arange(cmin, cmax + 1)             # every integer in range
    cmap = plt.get_cmap("viridis", len(levels))    # discrete colour per level
    bounds = np.arange(cmin - 0.5, cmax + 1.5)     # band edges around each int
    hb.set_cmap(cmap)
    hb.set_norm(BoundaryNorm(bounds, ncolors=cmap.N))

    # If the range is very wide, label every Nth level to keep it readable.
    step = max(1, int(np.ceil(len(levels) / HEX_MAX_LEVELS)))
    ticks = levels[::step]

    cb = fig.colorbar(
        hb, ax=ax, boundaries=bounds, ticks=ticks,
        spacing="uniform", shrink=0.6, pad=0.02,
    )
    cb.set_label("CUPS per cell")
    cb.ax.set_yticklabels([str(int(v)) for v in ticks])

    ax.set_title("CUPS density in Barcelona", fontsize=16)

    # Lock the view to the data extent (+margin) and keep north up.
    ax.set_xlim(xs.min() - _pad * _dx, xs.max() + _pad * _dx)
    ax.set_ylim(ys.min() - _pad * _dy, ys.max() + _pad * _dy)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if basemap:
        import contextily as ctx
        ctx.add_basemap(
            ax, crs=gdf_plot.crs,
            source=ctx.providers.CartoDB.Positron, zorder=1,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


draw_heatmap(OUTPUT_HEATMAP, basemap=False)
log.info("Saved heatmap: %s", OUTPUT_HEATMAP)

try:
    draw_heatmap(OUTPUT_BASEMAP, basemap=True)
    log.info("Saved heatmap with basemap: %s", OUTPUT_BASEMAP)
except Exception as e:
    log.warning("Basemap step skipped: %s", e)


# ============================================================
# MATCHING REPORT
# ============================================================

counts = df_merged["match_type"].value_counts()
total = len(df_merged)
matched = total - counts.get("unmatched", 0)

print("\n" + "=" * 52)
print(f"  exact matches : {counts.get('exact', 0)}")
print(f"  fuzzy matches : {counts.get('fuzzy', 0)}")
print(f"  unmatched     : {counts.get('unmatched', 0)}")
print(f"  match rate    : {100 * matched / total:.2f}%  ({matched}/{total})")
print("=" * 52)

unmatched = df_merged[df_merged["match_type"] == "unmatched"]
if len(unmatched):
    print("\nStill unmatched (first 50):")
    print(
        unmatched[[
            "Tipus via", "Nom carrer", "Núm carrer",
            "street_name_clean", "street_number",
        ]].head(50).to_string(index=False)
    )