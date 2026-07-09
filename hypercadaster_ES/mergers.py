"""Data merging and joining operations for hypercadaster_ES.

This module provides functions for joining Spanish cadastral data with
various external geographic and administrative datasets.

Main functions:
    - join_cadaster_data(): Main orchestrator for joining all cadastral data
    - get_cadaster_address(): Extract and process address information
    - join_cadaster_building(): Join building geometry and attributes
    - join_DEM_raster(): Add elevation data from Digital Elevation Model
    - join_by_census_tracts(): Add census tract information
    - join_by_neighbourhoods(): Add neighborhood information (Barcelona)
    - join_by_postal_codes(): Add postal code information
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from ._pretty import PrettyDict
import polars as pl
import geopandas as gpd
import rasterio
import regex as re
from shapely import wkt
from shapely.geometry import Point, Polygon, box
from tqdm import tqdm

from hypercadaster_ES import utils
from hypercadaster_ES import building_inference
from hypercadaster_ES import downloaders


def _extract_street_type_and_name(street_full_name):
    """
    Extract street type and name from full street name.
    
    Args:
        street_full_name (str): Full street name (e.g., 'Carrer de Sant Pau')
        
    Returns:
        tuple: (street_type, street_name) or (None, None) if parsing fails
    """
    if pd.isna(street_full_name) or not street_full_name:
        return None, None
    
    try:
        # Use proper Unicode character classes for Spanish/Catalan
        capital_matches = list(re.finditer(r'\b[A-ZÁÉÍÓÚÑÜ][^\s]*', street_full_name))
        number_match = re.search(r'\b\d+', street_full_name)
        
        cutoff_indices = []
        if len(capital_matches) >= 2:
            cutoff_indices.append(capital_matches[1].start())
        if number_match:
            cutoff_indices.append(number_match.start())
            
        if cutoff_indices:
            street_type = street_full_name[:min(cutoff_indices)].strip()
            street_name = street_full_name[len(street_type):].strip()
            return street_type.upper(), street_name.upper()
        
        return None, None
    except Exception:
        return None, None

def _process_barcelona_open_data(
    open_data_layers_dir,
    parcels_gdf,
    target_crs
):

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------
    bcn_file_path = f"{open_data_layers_dir}/barcelona_carrerer.gpkg"

    try:
        bcn = gpd.read_file(bcn_file_path)
    except Exception as e:
        sys.stderr.write(f"Warning: Could not load Barcelona data: {e}\n")
        return gpd.GeoDataFrame()

    if bcn.empty:
        return gpd.GeoDataFrame()

    # ------------------------------------------------------------
    # CLEAN + STANDARDIZE
    # ------------------------------------------------------------
    if bcn.geometry.iloc[0].geom_type == "MultiPoint":
        bcn["geometry"] = bcn.geometry.apply(
            lambda g: g.geoms[0] if hasattr(g, "geoms") and len(g.geoms) else g
        )

    if bcn.crs != target_crs:
        bcn = bcn.to_crs(target_crs)

    # stable id
    bcn = bcn.reset_index(drop=True)
    bcn["bcn_id"] = bcn.index

    # street parsing
    street_split = bcn["NOM_CARRER"].str.upper().str.split(" ", n=1, expand=True)
    bcn["street_type"] = street_split[0]
    bcn["street_name"] = street_split[1]

    bcn["street_number_label"], bcn["street_number_clean"] = utils.normalize_street_number(
        bcn["NUMPOST"]
    )

    # ------------------------------------------------------------
    # PARCEL PREP
    # ------------------------------------------------------------
    parcels = parcels_gdf[["building_reference", "parcel_geometry"]].rename(
        columns={"building_reference": "parcel_ref"}
    ).copy()

    parcels = parcels.set_geometry("parcel_geometry")

    if parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)

    # classify parcels
    parcels["area"] = parcels.area

    # area already computed
    parcels["is_building"] = (parcels["area"] > 50) & (parcels["area"] < 10000)

    def count_holes(geom):
        if geom.geom_type == "Polygon":
            return len(geom.interiors)
        elif geom.geom_type == "MultiPolygon":
            return sum(len(p.interiors) for p in geom.geoms)
        return 0

    parcels["holes"] = parcels["parcel_geometry"].apply(count_holes)

    parcels["is_container"] = (parcels["area"] > 10000) & (parcels["holes"] > 5)

    building_parcels = parcels[parcels["is_building"]]
    other_parcels = parcels[~parcels["is_building"] & ~parcels["is_container"]]
    container_parcels = parcels[parcels["is_container"]]

    # ------------------------------------------------------------
    # HELPER: ASSIGN FUNCTION
    # ------------------------------------------------------------
    def assign_by_join(points, candidate_parcels, predicate, priority, buffer=None, max_distance=None):
        if points.empty or candidate_parcels.empty:
            return pd.DataFrame(columns=["bcn_id", "parcel_ref", "priority", "area", "distance"])

        parcels_ = candidate_parcels.copy()

        if buffer is not None:
            parcels_["parcel_geometry"] = parcels_.buffer(buffer)

        join = gpd.sjoin(
            points,
            parcels_.set_geometry("parcel_geometry"),
            how="inner",
            predicate=predicate
        )

        if join.empty:
            return pd.DataFrame(columns=["bcn_id", "parcel_ref", "priority", "area", "distance"])

        join["priority"] = priority
        join["distance"] = 0.0

        return join[["bcn_id", "parcel_ref", "priority", "area", "distance"]]

    def assign_nearest(points, candidate_parcels, priority, max_distance=20):
        if points.empty or candidate_parcels.empty:
            return pd.DataFrame(columns=["bcn_id", "parcel_ref", "priority", "area", "distance"])

        join = gpd.sjoin_nearest(
            points,
            candidate_parcels.set_geometry("parcel_geometry"),
            how="inner",
            distance_col="distance",
            max_distance=max_distance
        )

        if join.empty:
            return pd.DataFrame(columns=["bcn_id", "parcel_ref", "priority", "area", "distance"])

        join["priority"] = priority

        return join[["bcn_id", "parcel_ref", "priority", "area", "distance"]]

    # ------------------------------------------------------------
    # BUILD CANDIDATES
    # ------------------------------------------------------------
    candidates = []

    # 1. building parcels (strict)
    candidates.append(assign_by_join(bcn, building_parcels, "intersects", priority=1))

    # 2. other parcels
    candidates.append(assign_by_join(bcn, other_parcels, "intersects", priority=2))

    # 3. buffered buildings
    candidates.append(assign_by_join(bcn, building_parcels, "intersects", priority=3, buffer=1.5))

    # 4. containers
    candidates.append(assign_by_join(bcn, container_parcels, "intersects", priority=4))

    # 5. nearest fallback
    candidates.append(assign_nearest(bcn, building_parcels if not building_parcels.empty else parcels, priority=5))

    candidates = pd.concat(candidates, ignore_index=True)

    if candidates.empty:
        return gpd.GeoDataFrame()

    # ------------------------------------------------------------
    # BEST MATCH SELECTION (CRITICAL)
    # ------------------------------------------------------------
    candidates = candidates.sort_values(
        ["bcn_id", "priority", "area", "distance"],
        ascending=[True, True, True, True]
    )

    best = candidates.drop_duplicates(subset=["bcn_id"])

    # ------------------------------------------------------------
    # MERGE BACK SAFELY
    # ------------------------------------------------------------
    bcn = bcn.merge(best[["bcn_id", "parcel_ref"]], on="bcn_id", how="left")

    # ------------------------------------------------------------
    # FINAL FORMAT
    # ------------------------------------------------------------
    bcn["building_reference"] = bcn["parcel_ref"]
    bcn["location"] = bcn.geometry
    bcn["specification"] = "OpenDataBCN"
    bcn["cadaster_code"] = "08900"

    valid = (
        bcn["building_reference"].notna()
        & bcn["street_name"].notna()
        & bcn["street_type"].notna()
    )

    return gpd.GeoDataFrame(
        bcn.loc[valid, [
            "location",
            "specification",
            "cadaster_code",
            "street_type",
            "street_name",
            "street_number_label",
            "street_number_clean",
            "building_reference"
        ]],
        geometry="location",
        crs=target_crs
    )


def process_open_data_addresses(open_data_layers_dir, cadaster_codes, parcels_gdf, target_crs):
    """
    Process addresses from municipal open data sources using improved parcel-based assignment.
    
    This function currently supports Barcelona but is designed to be extensible
    for other municipalities with open data address databases.
    
    Uses different strategies based on parcel geometry:
    - Parcels with holes: Exact containment only (prevents wrong assignment to outer parcel)
    - Simple parcels: 1.5m buffered approach for better address matching
    
    Args:
        open_data_layers_dir (str): Directory containing open data files
        cadaster_codes (list): List of cadastral codes to process
        parcels_gdf (GeoDataFrame): Parcel geometries for spatial joining  
        target_crs: Target coordinate reference system
        
    Returns:
        GeoDataFrame: Combined addresses from all available open data sources
    """
    combined_addresses = []
    
    # Barcelona (cadaster code 08900)
    if "08900" in cadaster_codes:
        bcn_addresses = _process_barcelona_open_data(open_data_layers_dir, parcels_gdf, target_crs)
        if not bcn_addresses.empty:
            combined_addresses.append(bcn_addresses)
    
    # Future municipalities can be added here:
    # if "XXXXX" in cadaster_codes:
    #     other_addresses = _process_other_municipality_data(...)
    #     if not other_addresses.empty:
    #         combined_addresses.append(other_addresses)
    
    if combined_addresses:
        result = pd.concat(combined_addresses, ignore_index=True)
        return gpd.GeoDataFrame(result, geometry='location', crs=target_crs)
    else:
        return gpd.GeoDataFrame()

def get_cadaster_address(cadaster_dir, cadaster_codes, directions_from_CAT_files=True, CAT_files_dir="CAT_files",
                         directions_from_open_data=True, open_data_layers_dir="open_data"):
    """Read and assemble cadastral address points for the given municipalities.

    Loads the INSPIRE Address GML per municipality, parses street type/name/number
    and the geolocated address point, and optionally enriches street naming from
    CAT files and/or Barcelona open data.

    Parameters
    ----------
    cadaster_dir : str
        Root of the downloaded cadastral layers.
    cadaster_codes : list[str]
        Cadastral municipality codes to read.
    directions_from_CAT_files : bool, default True
        Also derive street names from CAT files in ``CAT_files_dir``.
    CAT_files_dir : str, default "CAT_files"
        CAT files directory (used when ``directions_from_CAT_files``).
    directions_from_open_data : bool, default True
        Also derive street names from Barcelona open data.
    open_data_layers_dir : str, default "open_data"
        Barcelona open-data directory (used when ``directions_from_open_data``).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per address with geometry ``address_location``.
    """
    sys.stderr.write(f"\nReading the cadaster addresses for {len(cadaster_codes)} municipalities\n")

    address_gdf = gpd.GeoDataFrame()
    address_street_names_df = gpd.GeoDataFrame()

    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes...") if len(cadaster_codes)>1 else cadaster_codes:
        # sys.stderr.write("\r" + " " * 60)
        # sys.stderr.flush()
        # sys.stderr.write(f"\r\tCadaster code: {code}")
        # sys.stderr.flush()

        address_gdf_ = gpd.read_file(f"{cadaster_dir}/address/unzip/A.ES.SDGC.AD.{code}.gml", layer="Address")
        address_gdf_['cadaster_code'] = code

        if not address_gdf.empty:
            if address_gdf_.crs != address_gdf.crs:
                address_gdf_ = address_gdf_.to_crs(address_gdf.crs)
            address_gdf = gpd.GeoDataFrame(pd.concat([address_gdf, address_gdf_], ignore_index=True))
        else:
            address_gdf = address_gdf_

        address_street_names_df_ = gpd.read_file(f"{cadaster_dir}/address/unzip/A.ES.SDGC.AD.{code}.gml",
                                                 layer="ThoroughfareName")

        if not address_street_names_df.empty:
            address_street_names_df = pd.concat([address_street_names_df, address_street_names_df_],
                                                ignore_index=True)
        else:
            address_street_names_df = address_street_names_df_

    sys.stderr.write("\r" + " " * 60)

    address_street_names_df = address_street_names_df[['gml_id', 'text']].copy()
    address_street_names_df["gml_id"] = address_street_names_df['gml_id'].apply(lambda x: x.split('ES.SDGC.TN.')[1])
    address_gdf["gml_id"] = address_gdf['gml_id'].apply(lambda x: '.'.join(x.split('ES.SDGC.AD.')[1].split('.')[:3]))

    gdf = pd.merge(address_gdf, address_street_names_df, left_on="gml_id", right_on="gml_id")

    gdf.rename(columns={'geometry': 'location', 'text': 'street_name', 'designator': 'street_number'}, inplace=True)
    gdf["street_type"] = gdf["street_name"].apply(lambda x: x.split(" ")[0])
    gdf["street_name"] = gdf["street_name"].apply(lambda x: ' '.join(x.split(" ")[1:]))
    gdf["street_number_label"], gdf["street_number_clean"] = utils.normalize_street_number(
        gdf["street_number"]
    )
    # gdf = gdf[gdf['specification'] == "Entrance"]
    gdf["building_reference"] = gdf['localId'].apply(lambda x: x.split('.')[-1])

    gdf.drop(
        ["gml_id", "namespace", "localId", "beginLifespanVersion", "validFrom", "level", "type", "method", "default"],
        inplace=True, axis=1)
    gdf = gdf.set_geometry("location")
    gdf = gdf.drop(columns=["street_number"], errors="ignore")
    _, parcels_gdf = join_cadaster_parcel(gdf, cadaster_dir, cadaster_codes, how="left")

    if directions_from_CAT_files:
        addresses_CAT = pd.DataFrame()

        for code in cadaster_codes:
            # Parse CAT file, if available
            buildings_CAT = building_inference.parse_horizontal_division_buildings_CAT_files(code, CAT_files_dir)
            if buildings_CAT is None:
                continue
            addresses_CAT_ = (pl.concat([
                buildings_CAT[["building_reference", "street_type", "street_name", "street_number1"]].rename(
                    {"street_number1": "street_number"}),
                buildings_CAT[["building_reference", "street_type", "street_name", "street_number2"]].rename(
                    {"street_number2": "street_number"}).filter(pl.col("street_number") != "")], how="vertical")
            ).to_pandas()
            addresses_CAT_["street_number_label"], addresses_CAT_["street_number_clean"] = utils.normalize_street_number(
                addresses_CAT_["street_number"]
            )
            # Tag THIS municipality's chunk before concatenating. Assigning to
            # the accumulated frame instead would overwrite cadaster_code for
            # every previously-added municipality with the current code, so all
            # CAT addresses province-wide would end up labelled with the last
            # processed code (mislabelling entire municipalities downstream).
            addresses_CAT_['cadaster_code'] = code
            addresses_CAT = pd.concat([addresses_CAT, addresses_CAT_], ignore_index=True)

        def calculate_centroid(group):
            valid_points = [pt for pt in group["location"] if isinstance(pt, Point)]

            if not valid_points:
                return pd.Series({"location": None})

            x_coords = [pt.x for pt in valid_points]
            y_coords = [pt.y for pt in valid_points]

            centroid = Point(np.mean(x_coords), np.mean(y_coords))

            return pd.Series({"location": centroid})

        if not addresses_CAT.empty:
            addresses_CAT = addresses_CAT.merge(
                    gdf[["building_reference", "location"]].
                    groupby(["building_reference"]).
                    apply(calculate_centroid,include_groups=False).reset_index(),
                on="building_reference", how="left")
            if sum(addresses_CAT['location'].isna())>0:
                for code in addresses_CAT.cadaster_code[addresses_CAT['location'].isna()].unique():
                    buildings = gpd.read_file(f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.building.gml", layer="Building")
                    for idx in addresses_CAT.index[(addresses_CAT['location'].isna()) & (addresses_CAT['cadaster_code'] == code)]:
                        try:
                            addresses_CAT.loc[idx, 'location'] = list(buildings[buildings['reference'] == addresses_CAT.loc[idx, 'building_reference']].geometry)[0].centroid
                        except:
                            pass
            addresses_CAT = gpd.GeoDataFrame(addresses_CAT)
            addresses_CAT = addresses_CAT.set_geometry("location")
            addresses_CAT = addresses_CAT.set_crs(gdf.crs)
            addresses_CAT["specification"] = "CATFile"
            gdf = pd.concat([gdf, addresses_CAT], ignore_index=True)
            gdf = gdf.drop_duplicates(subset=["street_name", "street_number", "street_type", "cadaster_code"], keep="first", ignore_index=True)

    if directions_from_open_data:
        open_data_addresses = process_open_data_addresses(
            open_data_layers_dir, cadaster_codes, parcels_gdf, gdf.crs
        )
        if not open_data_addresses.empty:
            # Deprecate the addresses from Cadaster when they are already available from a Municipal dataset
            # (normally more accurate).
            gdf = gdf[~gdf["cadaster_code"].isin(open_data_addresses["cadaster_code"].unique())]
            gdf = pd.concat([gdf, open_data_addresses], ignore_index=True)

    return gdf


def assign_building_zones(gdf, cadaster_dir, cadaster_codes):
    """
    Assign cadastral zones to buildings using building centroids for better accuracy.
    
    This replaces the old approach that used address points with a more accurate method
    based on building geometry centroids and simple spatial joins.
    
    Args:
        gdf (GeoDataFrame): Building data with building_geometry
        cadaster_dir (str): Directory containing cadastral data
        cadaster_codes (list): List of cadastral codes to process
        
    Returns:
        GeoDataFrame: Data with zone_reference and zone_type columns added
    """
    import time
    import os
    
    start_time = time.time()

    # Initialize zone columns
    gdf['zone_reference'] = "unassigned"
    gdf['zone_type'] = "urban"

    processed_buildings = 0

    codes = tqdm(cadaster_codes, desc="Processing zones by municipality") \
        if isinstance(cadaster_codes, list) else [cadaster_codes]

    # Tag every row with the current municipality code. The first 5 chars of a
    # Spanish cadastral reference are the parcel reference within a municipality
    # — NOT the cadaster/municipality code — so deriving cadaster_code from
    # building_reference[:5] silently drops the bulk of the data. When this
    # function is called per-municipality the GDF rows all belong to `code`.
    if 'cadaster_code' not in gdf.columns:
        if isinstance(cadaster_codes, str):
            gdf['cadaster_code'] = cadaster_codes
        elif isinstance(cadaster_codes, list) and len(cadaster_codes) == 1:
            gdf['cadaster_code'] = cadaster_codes[0]
        else:
            raise ValueError(
                "assign_building_zones: gdf has no 'cadaster_code' column and "
                "multiple cadaster_codes were passed; the caller must label rows "
                "per municipality before calling this function."
            )

    for code in codes:
        try:
            # Load zone file for this municipality
            zone_file = f"{cadaster_dir}/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralzoning.gml"
            if not os.path.exists(zone_file):
                continue
                
            zone_gdf = gpd.read_file(zone_file, layer="CadastralZoning")
            
            # Clean and prepare zone data
            zone_gdf.rename(columns={
                'LocalisedCharacterString': 'zone_type',
                'nationalCadastalZoningReference': 'zone_reference'
            }, inplace=True)
            
            # Avoid FutureWarning by checking for empty DataFrame before subsetting
            if len(zone_gdf) > 0:
                zone_gdf = zone_gdf[["zone_reference", "zone_type", "geometry"]].copy()
            else:
                continue  # Skip empty zone files
            
            # Get buildings for this municipality only
            building_mask = gdf['cadaster_code'] == code
            buildings_for_code = gdf[building_mask].copy()
            if len(buildings_for_code) == 0:
                continue
                
            processed_buildings += len(buildings_for_code)
            
            # Create building centroids for zone assignment.
            # Drop the pre-initialized zone columns so sjoin doesn't produce
            # zone_reference_left / zone_reference_right suffixes that the
            # downstream resolution can't disambiguate reliably.
            buildings_for_code = buildings_for_code.drop(
                columns=[c for c in ("zone_reference", "zone_type") if c in buildings_for_code.columns]
            )
            buildings_for_code['centroid'] = buildings_for_code['building_geometry'].centroid
            buildings_centroids = buildings_for_code.set_geometry('centroid')

            # Align CRS
            if buildings_centroids.crs != zone_gdf.crs:
                buildings_centroids = buildings_centroids.to_crs(zone_gdf.crs)

            # Spatial join using building centroids
            joined = gpd.sjoin(buildings_centroids, zone_gdf, how="left", predicate="within")

            if len(joined) == 0 or "zone_reference" not in joined.columns:
                continue

            # Vectorized assignment back to the original gdf (sjoin preserves left index)
            assigned_mask = joined["zone_reference"].notna()
            if not assigned_mask.any():
                continue

            zone_ref_series = joined.loc[assigned_mask, "zone_reference"]
            zone_type_series = joined.loc[assigned_mask, "zone_type"]

            # MANZANA → urban, POLIGONO → disseminated, anything else → urban
            zone_type_clean = np.where(
                zone_type_series.astype(str).str.strip() == "POLIGONO",
                "disseminated",
                "urban",
            )

            gdf.loc[zone_ref_series.index, "zone_reference"] = zone_ref_series.values
            gdf.loc[zone_ref_series.index, "zone_type"] = zone_type_clean

            sys.stderr.write(
                f"assign_building_zones[{code}]: {assigned_mask.sum()}/{len(joined)} "
                f"buildings assigned to {zone_ref_series.nunique()} unique zones\n"
            )

        except Exception as e:
            sys.stderr.write(f"Error processing zones for {code}: {e}\n")
            continue

    # Remove temporary columns (only if we added it)
    # Note: we always keep cadaster_code as it might be used elsewhere
    
    total_time = time.time() - start_time
    unassigned_count = (gdf["zone_reference"] == "unassigned").sum() if len(gdf) > 0 else 0

    return gdf


# Cadastral OtherConstruction `constructionNature` → thermal-zone use type.
# The Spanish Catastro only populates this layer with `openAirPool` (pools) in
# practice, but the INSPIRE code list is extensible, so this is a dict that's
# easy to grow. Any nature NOT listed here is still stored in
# `br__other_constructions` but is not routed into the thermal zoning (and is
# logged by `join_other_constructions` so new types surface for mapping).
_CONSTRUCTION_NATURE_USE_TYPE = {
    "openAirPool": "Sports facilities",
}
# Human-readable detailed-use label per nature (falls back to the raw nature).
_CONSTRUCTION_NATURE_LABEL = {
    "openAirPool": "Open-air pool",
}


def join_other_constructions(building_gdf_, cadaster_dir, code):
    """Join cadastral OtherConstruction geometries onto buildings and route
    mapped natures into the thermal zoning.

    The ``A.ES.SDGC.BU.<code>.otherconstruction.gml`` layer holds detached
    constructions (open-air swimming pools, etc.). Its ``localId`` is
    ``<building_reference>_PI.<n>``, so the prefix joins directly on
    ``building_reference``.

    Adds a ``br__other_constructions`` column — a list of
    ``{nature, area, geometry}`` per building. When ``br__thermal_zones`` is
    present, every construction whose ``constructionNature`` is in
    ``_CONSTRUCTION_NATURE_USE_TYPE`` has its geometry assigned to the building's
    zone of the mapped use type (e.g. a pool → "Sports facilities"), or a new
    such zone is appended, so downstream views place that zone at the
    construction's true footprint. Natures not in the map are kept in the
    column but left unrouted and logged as a warning.
    """
    path = f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.otherconstruction.gml"
    if not os.path.exists(path):
        return building_gdf_
    try:
        oc = gpd.read_file(path, layer="OtherConstruction")
    except Exception as _e:
        sys.stderr.write(f"Could not read OtherConstruction for {code}: {_e}\n")
        return building_gdf_
    if oc.empty or "localId" not in oc.columns:
        return building_gdf_

    oc = oc.copy()
    oc["building_reference"] = (
        oc["localId"].astype(str).str.replace(r"_PI\.\d+$", "", regex=True))
    if (oc.crs is not None and getattr(building_gdf_, "crs", None) is not None
            and oc.crs != building_gdf_.crs):
        oc = oc.to_crs(building_gdf_.crs)

    by_ref = {}
    for ref, sub in oc.groupby("building_reference"):
        items = []
        for _, r in sub.iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            items.append({"nature": r.get("constructionNature"),
                          "area": round(float(g.area), 2),
                          "geometry": g})
        if items:
            by_ref[ref] = items
    if not by_ref:
        return building_gdf_

    building_gdf_ = building_gdf_.copy(deep=False)
    building_gdf_["br__other_constructions"] = (
        building_gdf_["building_reference"].map(by_ref))

    # Self-reporting: tally every constructionNature seen, and flag any that
    # aren't mapped to a use type (so new municipalities surface new types).
    from collections import Counter
    nat_counts = Counter(o["nature"] for v in by_ref.values() for o in v)
    mapped = {n: c for n, c in nat_counts.items() if n in _CONSTRUCTION_NATURE_USE_TYPE}
    unmapped = {n: c for n, c in nat_counts.items() if n not in _CONSTRUCTION_NATURE_USE_TYPE}
    sys.stderr.write(
        f"join_other_constructions[{code}]: joined to "
        f"{building_gdf_['br__other_constructions'].notna().sum()} buildings; "
        f"natures={dict(nat_counts)}; mapped={mapped}\n"
    )
    if unmapped:
        sys.stderr.write(
            f"join_other_constructions[{code}]: WARNING unmapped "
            f"constructionNature {unmapped} kept in br__other_constructions but "
            f"NOT routed to a thermal zone — add to "
            f"mergers._CONSTRUCTION_NATURE_USE_TYPE to map.\n"
        )

    if "br__thermal_zones" in building_gdf_.columns:
        from shapely.ops import unary_union as _uu

        def _augment(zones, others):
            if not isinstance(zones, list) or not isinstance(others, list):
                return zones
            # Group mapped constructions by their target use type.
            by_use = {}
            for o in others:
                if not isinstance(o, dict):
                    continue
                use = _CONSTRUCTION_NATURE_USE_TYPE.get(o.get("nature"))
                g = o.get("geometry")
                if use is None or g is None or getattr(g, "is_empty", True):
                    continue
                slot = by_use.setdefault(use, {"geoms": [], "labels": set()})
                slot["geoms"].append(g)
                slot["labels"].add(
                    _CONSTRUCTION_NATURE_LABEL.get(o["nature"], o["nature"]))
            if not by_use:
                return zones
            zones = list(zones)
            for use, slot in by_use.items():
                geoms = slot["geoms"]
                geom = _uu(geoms) if len(geoms) > 1 else geoms[0]
                area = round(float(geom.area), 2)
                labels = sorted(slot["labels"])
                attached = False
                for i, z in enumerate(zones):
                    if isinstance(z, dict) and z.get("use_type") == use:
                        z = dict(z)
                        z["geometry"] = geom
                        z["detailed_use_types"] = sorted(set(
                            (z.get("detailed_use_types") or []) + labels))
                        z["source"] = "other_construction"
                        zones[i] = z
                        attached = True
                if not attached:
                    zones.append({
                        "zone_id": f"{use}|other_construction",
                        "kind": "occupied",
                        "use_type": use,
                        "detailed_use_types": labels,
                        "floor": 0, "floors": [0], "duplex": False,
                        "n_spaces": len(geoms),
                        "building_spaces_references": [],
                        "area_with_communals": area,
                        "area_without_communals": area,
                        "economic_value": 0.0,
                        "geometry": geom,
                        "source": "other_construction",
                    })
            return zones

        building_gdf_["br__thermal_zones"] = [
            _augment(z, o) for z, o in zip(
                building_gdf_["br__thermal_zones"],
                building_gdf_["br__other_constructions"])
        ]
    return building_gdf_


# Use-type keywords whose thermal zones are not occupied by people (parking,
# storage, …). Open-air pools (routed from OtherConstruction) are also marked
# non-occupied via their source/detailed-use markers below.
_NON_OCCUPIED_ZONE_USE_KEYS = ("parking", "warehouse", "storage", "garage")


def _thermal_zone_is_non_occupied(z):
    """True when a thermal-zone dict represents a space not occupied by people
    (warehouse/parking/storage, or an open-air pool)."""
    if not isinstance(z, dict) or z.get("kind") == "communal":
        return False
    ut = str(z.get("use_type", "")).lower()
    if any(k in ut for k in _NON_OCCUPIED_ZONE_USE_KEYS):
        return True
    det = " ".join(str(d) for d in (z.get("detailed_use_types") or [])).lower()
    if "pool" in det:
        return True
    return z.get("source") == "other_construction"


def classify_non_occupied_thermal_zones(building_gdf_):
    """Tag thermal zones that aren't occupied by people with
    ``kind='non_occupied'`` (warehouse/parking and open-air pools), leaving
    communal cores and people-occupied zones untouched. Runs at merge time so
    it applies without re-running the inference."""
    if "br__thermal_zones" not in building_gdf_.columns:
        return building_gdf_

    def _mark(zones):
        if not isinstance(zones, list):
            return zones
        out = []
        for z in zones:
            if _thermal_zone_is_non_occupied(z) and z.get("kind") != "non_occupied":
                z = dict(z)
                z["kind"] = "non_occupied"
            out.append(z)
        return out

    building_gdf_ = building_gdf_.copy(deep=False)
    building_gdf_["br__thermal_zones"] = building_gdf_["br__thermal_zones"].map(_mark)
    return building_gdf_


def join_cadaster_building(gdf, cadaster_dir, census_tracts_dir, DEM_raster_dir,
                           cadaster_codes, results_dir, open_street_dir, building_parts_plots=False,
                           plot_zones_ratio=0.01, building_parts_inference=False,
                           building_parts_inference_using_CAT_files=False, open_data_layers=False,
                           open_data_layers_dir=None, CAT_files_dir=None,
                           floor_height=3.0, ground_floor_height=5.0, num_workers=-1,
                           force_update=False, cleanup_shards=True):
    """Join building geometry, attributes and (optionally) inference to addresses.

    Reads the INSPIRE Building layer per municipality, attaches building
    footprint/use/status/year and space classification, and — when
    ``building_parts_inference`` is set — runs the geometric inference pipeline
    (footprints, walls, patios, orientation, shadows) writing ``br_results`` /
    ``sbr_results`` to the central DuckDB. The ``buildings`` table is upserted
    per municipality.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Address-level frame from :func:`get_cadaster_address`.
    cadaster_dir, census_tracts_dir, DEM_raster_dir, results_dir, open_street_dir : str
        Input/output directories.
    cadaster_codes : list[str]
        Municipalities to process.
    building_parts_inference : bool, default False
        Run the geometric inference engine (single municipality only).
    building_parts_inference_using_CAT_files : bool, default False
        Use CAT space data to guide inference.
    building_parts_plots, plot_zones_ratio : bool / float
        Emit per-zone diagnostic plots for a sampled fraction of zones.
    open_data_layers, open_data_layers_dir, CAT_files_dir : optional
        Barcelona open-data / CAT enrichment sources.
    floor_height, ground_floor_height : float
        Storey heights (m) used by the inference.
    num_workers : int, default -1
        Parallel workers (-1 = all cores).
    force_update, cleanup_shards : bool
        Rebuild cached tables / remove per-zone shard DBs after ingestion.

    Returns
    -------
    geopandas.GeoDataFrame
        Address×building frame with building attributes (and ``br__*`` inference
        columns when enabled).
    """
    sys.stderr.write(f"\nJoining the buildings description for {len(cadaster_codes)} municipalities\n")

    # Drop stale buildings table when force_update is requested so the first
    # write below uses 'replace' and the schema matches the current run.
    if force_update:
        try:
            from hypercadaster_ES import _db as _db_mod_force
            _fc = _db_mod_force.open_db(results_dir)
            _fc.execute("DROP TABLE IF EXISTS buildings")
            _fc.close()
        except Exception:
            pass

    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes...") if len(cadaster_codes)>1 else cadaster_codes:
        # sys.stderr.write("\r" + " " * 60)
        # sys.stderr.flush()
        # sys.stderr.write(f"\r\tCadaster code: {code}")
        # sys.stderr.flush()

        # Parse building harmonised to INSPIRE
        building_gdf_ = gpd.read_file(f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.building.gml", layer="Building")
        building_gdf_= building_gdf_.rename(columns={
            'geometry': 'building_geometry',
            'value': 'building_area',
            'conditionOfConstruction': 'building_status',
            'currentUse': 'building_use',
            'numberOfBuildingUnits': 'n_building_units',
            'numberOfDwellings': 'n_dwellings',
            'numberOfFloorsAboveGround': 'n_floors_above_ground',
            'numberOfFloorsBelowGround': 'n_floors_below_ground',
            'reference': 'building_reference',
            'beginning': 'year_of_construction'
        })
        building_gdf_['year_of_construction'] = building_gdf_['year_of_construction'].str[0:4]
        building_gdf_['year_of_construction'] = pd.to_numeric(
            building_gdf_['year_of_construction'], errors='coerce').astype('Int64')
        building_gdf_.drop(
            ["localId", "namespace", "officialAreaReference", "value_uom", "horizontalGeometryEstimatedAccuracy",
             "horizontalGeometryEstimatedAccuracy_uom", "horizontalGeometryReference", "referenceGeometry",
             "documentLink", "format", "sourceStatus", "beginLifespanVersion", "end", "endLifespanVersion",
             "informationSystem"],
            inplace=True, axis=1)
        building_gdf_ = building_gdf_.set_geometry("building_geometry")

        # Zone assignment
        if 'building_geometry' in building_gdf_.columns:
            building_gdf_ = assign_building_zones(building_gdf_, cadaster_dir, code)

        # Parse CAT file, if available
        if building_parts_inference_using_CAT_files:
            buildings_CAT = building_inference.parse_horizontal_division_buildings_CAT_files(code, CAT_files_dir)
        else:
            buildings_CAT = None

        if building_parts_inference:

            building_part_gdf_ = gpd.read_file(
                f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.buildingpart.gml",
                layer="BuildingPart"
            )

            nearby_buildings_buffer = 80

            # ------------------------------------------------------------
            # 1️⃣ Get neighboring municipalities (ONLY THESE)
            # ------------------------------------------------------------
            # Neighbours must be resolved for the *current* municipality only.
            # Passing the whole batch list here (a) recomputes an identical,
            # batch-wide union+intersect on every iteration — the dominant fixed
            # cost of this loop — and (b) treats every batch municipality as
            # "original", so adjacent municipalities that are *both* in the batch
            # get excluded from each other's neighbour set, dropping the border
            # building-parts this logic exists to load.
            nearby_cadaster_codes = utils.get_neighboring_cadaster_codes(
                [code],
                census_tracts_dir,
                buffer_distance=nearby_buildings_buffer
            )

            # ------------------------------------------------------------
            # 2️⃣ Prepare spatial filter area
            # ------------------------------------------------------------
            from shapely.geometry import box

            current_bounds = building_part_gdf_.total_bounds
            current_area = box(*current_bounds).buffer(nearby_buildings_buffer)

            # ------------------------------------------------------------
            # 3️⃣ Load ONLY neighboring building parts
            # ------------------------------------------------------------
            nearby_building_parts = []

            for neighbor_code in nearby_cadaster_codes:

                file_path = f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{neighbor_code}.buildingpart.gml"

                if not os.path.exists(file_path):
                    continue

                try:
                    neighbor_parts = gpd.read_file(file_path, layer="BuildingPart")

                    if neighbor_parts.empty:
                        continue

                    # Ensure CRS consistency
                    if neighbor_parts.crs != building_part_gdf_.crs:
                        neighbor_parts = neighbor_parts.to_crs(building_part_gdf_.crs)

                    # Spatial filter
                    neighbor_parts_intersect = neighbor_parts[
                        neighbor_parts.geometry.intersects(current_area)
                    ]

                    if not neighbor_parts_intersect.empty:
                        neighbor_parts_intersect = neighbor_parts_intersect.copy()
                        neighbor_parts_intersect["nearby_buildings"] = True
                        nearby_building_parts.append(neighbor_parts_intersect)

                except Exception as e:
                    print(f"[WARN] nearby_buildings error for {neighbor_code}: {e}")

            # ------------------------------------------------------------
            # 4️⃣ Merge with original building parts
            # ------------------------------------------------------------
            building_part_gdf_["nearby_buildings"] = False

            if nearby_building_parts:
                building_part_gdf_ = pd.concat(
                    [building_part_gdf_] + nearby_building_parts,
                    ignore_index=True
                )

            # ------------------------------------------------------------
            # 5️⃣ Continue original pipeline
            # ------------------------------------------------------------
            sys.stderr.write("\r" + " " * 60)

            building_part_gdf_.rename(columns={
                'geometry': 'building_part_geometry',
                'numberOfFloorsAboveGround': 'n_floors_above_ground',
                'numberOfFloorsBelowGround': 'n_floors_below_ground',
                'localId': 'building_reference'
            }, inplace=True)

            building_part_gdf_['building_reference'] = (
                building_part_gdf_['building_reference'].str.split("_").str[0]
            )

            building_part_gdf_.drop(
                ['gml_id', 'beginLifespanVersion', 'conditionOfConstruction',
                 'namespace', 'horizontalGeometryEstimatedAccuracy',
                 'horizontalGeometryEstimatedAccuracy_uom',
                 'horizontalGeometryReference', 'referenceGeometry',
                 'heightBelowGround', 'heightBelowGround_uom'],
                inplace=True, axis=1
            )

            if gdf is not None:
                gdf_unique = gdf.drop_duplicates(subset='building_reference')
                building_part_gdf_ = building_part_gdf_.join(
                    gdf_unique.set_index('building_reference'),
                    on="building_reference",
                    how="left"
                )

            # Zone info lives on building_gdf_ (from assign_building_zones), not on
            # the address gdf — pull it across so building parts can be grouped by zone.
            if {'building_reference', 'zone_reference', 'zone_type'}.issubset(building_gdf_.columns):
                zone_info = (building_gdf_[['building_reference', 'zone_reference', 'zone_type']]
                             .drop_duplicates('building_reference'))
                for col in ('zone_reference', 'zone_type'):
                    if col in building_part_gdf_.columns:
                        building_part_gdf_ = building_part_gdf_.drop(columns=[col])
                building_part_gdf_ = building_part_gdf_.merge(
                    zone_info, on='building_reference', how='left'
                )

            # Ensure zone columns exist
            building_part_gdf_["zone_type"] = building_part_gdf_.get("zone_type", "unknown")
            building_part_gdf_["zone_reference"] = building_part_gdf_.get("zone_reference", "unknown")

            building_part_gdf_.loc[
                building_part_gdf_["zone_type"].isna(), "zone_type"
            ] = "unknown"

            building_part_gdf_.loc[
                building_part_gdf_["zone_reference"].isna(), "zone_reference"
            ] = "unknown"

            # ------------------------------------------------------------
            # 6️⃣ Barcelona extra data
            # ------------------------------------------------------------
            if code == "08900" and open_data_layers:
                ground_premises = downloaders.load_and_transform_barcelona_ground_premises(
                    open_data_layers_dir
                )
                building_part_gdf_ = building_part_gdf_.join(
                    ground_premises.set_index("building_reference"),
                    on="building_reference",
                    how="left"
                )

            # ------------------------------------------------------------
            # 7️⃣ Join parcels
            # ------------------------------------------------------------
            building_part_gdf_, parcels_gdf = join_cadaster_parcel(
                building_part_gdf_, cadaster_dir, [code]
            )

            # ------------------------------------------------------------
            # 8️⃣ Elevation
            # ------------------------------------------------------------
            try:
                building_part_gdf_ = building_part_gdf_.set_geometry("building_part_geometry")
                dem_files = glob.glob(f"{DEM_raster_dir}/*.tif")
                if dem_files:
                    # join_DEM_raster computes centroids per chunk; no need
                    # to materialise a centroid_for_elevation column (which
                    # is 50M shapely Points / ~10 GB on Barcelona-sized input).
                    building_part_gdf_ = join_DEM_raster(
                        building_part_gdf_,
                        DEM_raster_dir,
                        geometry_column="building_part_geometry",
                    )
                else:
                    building_part_gdf_["elevation"] = 0.0
            except Exception:
                building_part_gdf_['elevation'] = 0.0

            # Process the building parts
            building_part_gdf_ = building_inference.process_building_parts(
                code=code, building_part_gdf_=building_part_gdf_, buildings_CAT=buildings_CAT,
                parcels_gdf=parcels_gdf, results_dir=results_dir, cadaster_dir=cadaster_dir,
                open_street_dir=open_street_dir, plots=building_parts_plots, plot_zones_ratio=plot_zones_ratio,
                floor_height=floor_height, ground_floor_height=ground_floor_height,
                DEM_raster_dir=DEM_raster_dir, nearby_buildings_buffer=nearby_buildings_buffer,
                num_workers=num_workers,
                force_update=force_update, cleanup_shards=cleanup_shards)

            # process_building_parts now returns a ResultsHandle pointing
            # to the central DuckDB.  The handle carries an open parent
            # connection from Phase A.  We use that connection for our
            # SQL queries, then close it when done.
            inference_handle = building_part_gdf_

            from hypercadaster_ES import _db as _db_mod

            code = cadaster_codes[0]  # single-municipality for inference

            # Quick check: does the handle have any rows?
            has_data = inference_handle.br_count() > 0

            if has_data:
                central_con = inference_handle._parent_con
                # Register the slim building_gdf_ as a temp relation so we
                # can LEFT JOIN it against br_results in DuckDB.
                slim_cols = ['gml_id', 'building_status', 'building_reference',
                             'building_use', 'building_geometry',
                             'year_of_construction', 'zone_reference',
                             'zone_type']
                _slim = building_gdf_[slim_cols].copy(deep=False)

                # Convert geometry column to WKB for DuckDB ingest
                if 'building_geometry' in _slim.columns:
                    _slim = pd.DataFrame(_slim)
                    _slim['building_geometry'] = (
                        building_gdf_['building_geometry'].to_wkb()
                    )

                central_con.register("_building_slim", _slim)
                # Exclude the heavy per-building BLOB columns (shadows /
                # terrain grids / wall segments) from the bulk frame. Each
                # is ~11 GB on disk for Barcelona and explodes into tens of
                # GB of Python dicts once read_gdf unpickles it across the
                # full building set — the cause of the post-BR-JOIN OOM.
                # They stay in br_results in the central DB and are fetched
                # per-building on demand via
                # utils.get_heavy_columns_for_building (see HEAVY_BLOB_COLS).
                _br_cols = {
                    r[0] for r in central_con.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'br_results'"
                    ).fetchall()
                }
                _br_exclude = ["building_reference", "cadaster_code", "zone_reference"]
                _br_exclude += sorted(_db_mod.HEAVY_BLOB_COLS & _br_cols)
                _br_exclude_sql = ", ".join(f'"{c}"' for c in _br_exclude)
                _merge_t = utils.stage_timer("merge: BR LEFT JOIN br_results (SQL)")
                _merge_t.__enter__()
                try:
                    # --- BR merge: one SQL statement ---
                    central_con.execute(f"""
                        CREATE OR REPLACE TABLE _br_merged AS
                        SELECT b."gml_id",
                               b."building_status",
                               b."building_reference",
                               b."building_use",
                               ST_GeomFromWKB(b."building_geometry") AS "building_geometry",
                               b."year_of_construction",
                               b."zone_reference",
                               b."zone_type",
                               r.* EXCLUDE ({_br_exclude_sql})
                        FROM _building_slim b
                        LEFT JOIN "main"."br_results" r
                                  ON r."building_reference" = b."building_reference"
                         AND r."cadaster_code" = ?
                    """, [code])
                finally:
                    central_con.unregister("_building_slim")
                    _merge_t.__exit__(None, None, None)

                # --- SBR aggregation: count + dict construction in SQL ---
                _sbr_t = utils.stage_timer("merge: SBR aggregation (json_group_object + UPDATEs)")
                _sbr_t.__enter__()
                sbr_cols_row = central_con.execute(r"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'sbr_results'
                      AND column_name LIKE 'sbr\__%'
                    ORDER BY column_name
                """).fetchall()
                # Drop the heavy per-SBR BLOB columns: json_group_object'ing
                # the pickled shadow/terrain grids would rebuild the same
                # tens-of-GB payload we just excluded from the BR side and
                # land it back in _br_merged for read_gdf to choke on.
                sbr_cols = [r[0] for r in sbr_cols_row
                            if r[0] not in _db_mod.HEAVY_BLOB_COLS]

                if sbr_cols:
                    utils.stage_log(f"  SBR: {len(sbr_cols)} sbr_* columns to aggregate")
                    # Count SBRs per building
                    central_con.execute("""
                        CREATE OR REPLACE TABLE _sbr_counts AS
                        SELECT "building_reference",
                               count(DISTINCT "single_building_reference") AS "n_sbr"
                        FROM "main"."sbr_results"
                        WHERE "cadaster_code" = ?
                        GROUP BY "building_reference"
                    """, [code])
                    utils.stage_log("  SBR: _sbr_counts built")

                    # Split into multi-SBR and single-SBR buildings.
                    # Multi-SBR: build {sbr_id: value} JSON objects per column
                    # via GROUP BY + json_group_object.
                    # Single-SBR: output NULL for all sbr columns.

                    # Multi-SBR aggregation
                    json_agg_parts = []
                    for col in sbr_cols:
                        json_agg_parts.append(f'''
                            json_group_object(
                                "single_building_reference", "{col}"
                            ) AS "{col}"
                        ''')

                    central_con.execute(f"""
                        CREATE OR REPLACE TABLE _sbr_multi AS
                        SELECT "building_reference",
                               {', '.join(json_agg_parts)}
                        FROM "main"."sbr_results"
                        WHERE "cadaster_code" = ?
                          AND "building_reference" IN (
                              SELECT "building_reference" FROM _sbr_counts WHERE "n_sbr" > 1
                          )
                        GROUP BY "building_reference"
                    """, [code])
                    utils.stage_log("  SBR: _sbr_multi (json_group_object) built")

                    # Single-SBR: NULL for all sbr columns
                    single_null_cols = ', '.join(
                        f'NULL AS "{col}"' for col in sbr_cols
                    )
                    central_con.execute(f"""
                        CREATE OR REPLACE TABLE _sbr_single AS
                        SELECT "building_reference",
                               {single_null_cols}
                        FROM _sbr_counts
                        WHERE "n_sbr" = 1
                    """)

                    # Union multi + single
                    all_cols = ', '.join(f'"{c}"' for c in sbr_cols)
                    central_con.execute(f"""
                        CREATE OR REPLACE TABLE _sbr_agg AS
                        SELECT "building_reference", {all_cols} FROM _sbr_multi
                        UNION ALL
                        SELECT "building_reference", {all_cols} FROM _sbr_single
                    """)
                    utils.stage_log("  SBR: _sbr_agg union built; starting UPDATE _br_merged")

                    # Merge SBR aggregation into br_merged
                    set_clauses = ', '.join(
                        f'"{c}" = s."{c}"' for c in sbr_cols
                    )
                    central_con.execute(f"""
                        UPDATE _br_merged
                        SET {set_clauses}
                        FROM _sbr_agg s
                        WHERE _br_merged."building_reference" = s."building_reference"
                    """)

                    # Also add n_sbr count
                    central_con.execute("""
                        UPDATE _br_merged
                        SET "n_sbr" = c."n_sbr"
                        FROM _sbr_counts c
                        WHERE _br_merged."building_reference" = c."building_reference"
                    """)

                _sbr_t.__exit__(None, None, None)

                # Read the merged result back as a GeoDataFrame.
                # read_gdf converts the designated geometry column and
                # unpickles any BLOB columns (pickled per-use-type dicts
                # like br__building_spaces_*); other GEOMETRY columns
                # stay as raw WKB bytes so write_gdf downstream doesn't
                # see unexpected geopandas 'geometry' dtype columns.
                with utils.stage_timer("merge: read _br_merged into GeoDataFrame"):
                    building_gdf_ = _db_mod.read_gdf(
                        central_con, "_br_merged", geom_col="building_geometry"
                    )

                # Also keep br_results for building_part_gdf downstream
                with utils.stage_timer("merge: br_results -> pandas (building_part_gdf)"):
                    building_part_gdf = inference_handle.br_to_pandas()

                # We're done with the parent connection — close it.
                if central_con is not None:
                    central_con.close()
            else:
                sys.stderr.write("Warning: Building parts inference returned no results\n")
                building_gdf_ = building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                              'building_geometry', 'year_of_construction', 'zone_reference', 'zone_type']]

        if buildings_CAT is not None:
            #['n_building_units', 'n_dwellings', 'n_floors_above_ground', 'building_area']
            if not building_parts_inference:
                building_gdf_ = building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                             'building_geometry', 'year_of_construction', 'zone_reference', 'zone_type']]
                # CAT yields the full br__* per-building aggregates (building_spaces_*,
                # area_with_communals, economic_value, ...) without any geometric
                # inference — fold them in so this path matches the inference path
                # for all CAT-derived columns.
                br_cat = building_inference.br_attributes_from_CAT(buildings_CAT)
                building_gdf_ = building_gdf_.merge(br_cat, on="building_reference", how="left")

            use_types = buildings_CAT["building_space_inferred_use_type"].unique().to_list()

            # Function to convert use type to snake_case with prefix
            def to_snake_case_prefix(use_type: str) -> str:
                if use_type is None or pd.isna(use_type):
                    return "building_area_unknown"
                return "building_area_" + re.sub(r'[^a-zA-Z0-9]+', '_', str(use_type).strip().lower()).strip('_')

            # Create the use type column mapping (filter out None values)
            valid_use_types = [use_type for use_type in use_types if use_type is not None and not pd.isna(use_type)]
            use_type_mapping = {use_type: to_snake_case_prefix(use_type) for use_type in valid_use_types}

            # First, replace None/NaN use types with a placeholder before pivoting
            buildings_CAT = buildings_CAT.with_columns(
                pl.col("building_space_inferred_use_type").fill_null("Unknown").alias("building_space_inferred_use_type")
            )

            # Add mapping for the placeholder
            use_type_mapping["Unknown"] = "building_area_unknown"

            # Summarize total area per building and use type
            area_per_use = (
                buildings_CAT
                .group_by(["building_reference", "building_space_inferred_use_type"])
                .agg(
                    pl.col("building_space_area_with_communal").sum().alias("area_by_use")
                )
            )

            # Pivot so each use type becomes a column
            area_pivot_raw = (
                area_per_use
                .pivot(
                    values="area_by_use",
                    index="building_reference",
                    on="building_space_inferred_use_type"
                )
            )
            
            # Filter mapping to only include columns that actually exist after pivot
            existing_columns = set(area_pivot_raw.columns)
            filtered_mapping = {k: v for k, v in use_type_mapping.items() if k in existing_columns}
            
            area_pivot = (
                area_pivot_raw
                .rename(filtered_mapping)  # rename only existing columns
                .fill_null(0.0)  # optional: replace nulls with 0 for missing use types
            )

            # Add additional metrics: total units, dwellings, floors, total area
            summary = (
                buildings_CAT
                .group_by("building_reference")
                .agg([
                    pl.len().alias("n_building_units"),
                    (pl.col("building_space_inferred_use_type") == "Residential")
                    .sum()
                    .alias("n_dwellings"),
                    pl.col("building_space_floor_name").unique().count().alias("n_floors_above_ground"),
                    pl.col("building_space_area_with_communal").sum().alias("building_area")
                ])
            )

            # Join both tables
            with utils.stage_timer("merge: CAT area/summary join + merge into buildings"):
                final_df = summary.join(area_pivot, on="building_reference", how="left").to_pandas()
                building_gdf_ = pd.merge(building_gdf_, final_df, on="building_reference", how="left")

            # Unit/dwelling counts from the inference's br__building_spaces, which
            # tallies spaces only on floors that exist in the real BuildingPart
            # geometry. CAT can list spaces on a floor the geometry lacks (so the
            # last floor gets synthesized); those must NOT inflate the counts.
            # br__building_spaces is {use_type: n_spaces}; n_building_units is its
            # sum and n_dwellings its 'Residential' entry. Falls back to the raw
            # CAT-summary value when br__building_spaces isn't available.
            if building_parts_inference and 'br__building_spaces' in building_gdf_.columns:
                def _units_from_spaces(d):
                    return sum(d.values()) if isinstance(d, dict) and d else None

                def _dwellings_from_spaces(d):
                    return d.get('Residential', 0) if isinstance(d, dict) and d else None

                _u = building_gdf_['br__building_spaces'].apply(_units_from_spaces)
                _w = building_gdf_['br__building_spaces'].apply(_dwellings_from_spaces)
                building_gdf_['n_building_units'] = _u.where(
                    _u.notna(), building_gdf_['n_building_units'])
                building_gdf_['n_dwellings'] = _w.where(
                    _w.notna(), building_gdf_['n_dwellings'])

            # Undeveloped parcels ("solares", CAT use type 'M') live only in CAT
            # dataset 15, so the dataset-14-based parse never emits them and they
            # have no BuildingPart geometry — they'd otherwise be missing from
            # building_gdf_ entirely and carry all-NaN CAT columns downstream.
            # Add a minimal row (undeveloped-land area + zeroed unit counts) so
            # they at least surface their parcel area.
            with utils.stage_timer("merge: undeveloped-parcel CAT supplement"):
                undev = building_inference.undeveloped_parcels_from_CAT(code, CAT_files_dir)
                if not undev.empty:
                    undev = undev[
                        ~undev["building_reference"].isin(building_gdf_["building_reference"])
                    ]
                if not undev.empty:
                    _geom_name = (
                        building_gdf_.geometry.name
                        if isinstance(building_gdf_, gpd.GeoDataFrame) else "building_geometry"
                    )
                    _crs = (
                        building_gdf_.crs
                        if isinstance(building_gdf_, gpd.GeoDataFrame) else None
                    )
                    building_gdf_ = gpd.GeoDataFrame(
                        pd.concat([building_gdf_, undev], ignore_index=True),
                        geometry=_geom_name, crs=_crs,
                    )

        elif not building_parts_inference:
            # Only strip to basic columns when building_parts_inference is explicitly False
            building_gdf_ = building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                         'building_geometry', 'year_of_construction', 'n_building_units', 'n_dwellings',
                                         'n_floors_above_ground', 'building_area', 'zone_reference', 'zone_type']]
        # When building_parts_inference=True but buildings_CAT=None, keep all columns (including br__ and sbr_data)

        # Join detached OtherConstruction geometries (pools, …) and route pools
        # into the thermal zoning as "Sports facilities".
        with utils.stage_timer("merge: join OtherConstruction (pools)"):
            building_gdf_ = join_other_constructions(building_gdf_, cadaster_dir, code)
        # Mark warehouse/parking and pool zones as not occupied by people.
        building_gdf_ = classify_non_occupied_thermal_zones(building_gdf_)

        # Persist the per-municipality result to the central DuckDB as a
        # `buildings` table so downstream consumers (read_addresses_indicators,
        # future MunicipalityHandle) can query on demand.
        from hypercadaster_ES import _db as _db_mod_build
        _build_con = _db_mod_build.open_db(results_dir)
        try:
            if "cadaster_code" not in building_gdf_.columns:
                building_gdf_ = building_gdf_.assign(cadaster_code=code)
            else:
                building_gdf_ = building_gdf_.copy(deep=False)
                building_gdf_["cadaster_code"] = code
            # Upsert keyed by building so a re-run refreshes the columns this
            # run produced WITHOUT duplicating rows and WITHOUT nulling columns
            # it didn't compute. In particular a building_parts_inference=False
            # re-run updates the simple/CAT features but preserves the complex
            # inference features (shadows, walls, structural floors, …) left by
            # an earlier inference run. write_gdf creates the table on first use.
            with utils.stage_timer(
                f"merge: upsert 'buildings' table to central DB "
                f"({len(building_gdf_)} rows)"
            ):
                _db_mod_build.write_gdf(
                    _build_con, "buildings", building_gdf_,
                    geom_cols=("building_geometry",),
                    if_exists="upsert",
                    key_cols=("cadaster_code", "building_reference"),
                )
        finally:
            _build_con.close()

        # Include it in the general building_gdf_
        if "building_gdf" in locals():
            if building_gdf_.crs != building_gdf.crs:
                building_gdf_ = building_gdf_.to_crs(building_gdf.crs)
            # Avoid FutureWarning by checking for empty DataFrames before concatenation
            if not building_gdf_.empty:
                if building_gdf.empty:
                    building_gdf = building_gdf_
                else:
                    building_gdf = gpd.GeoDataFrame(pd.concat([building_gdf, building_gdf_], ignore_index=True))
        else:
            building_gdf = building_gdf_

    if gdf is not None:
        # Avoid cadaster_code column collision: both gdf (from addresses) and
        # building_gdf carry it. Since they represent the same municipality,
        # drop it from the right side so pd.merge doesn't suffix the names.
        if "cadaster_code" in building_gdf.columns:
            building_gdf = building_gdf.drop(columns="cadaster_code")
        with utils.stage_timer(
            f"merge: address↔building pd.merge "
            f"({len(gdf)} addresses × {len(building_gdf)} buildings)"
        ):
            merged_gdf = pd.merge(gdf, building_gdf, left_on="building_reference", right_on="building_reference", how="left")
    else:
        merged_gdf = building_gdf

    return merged_gdf


def join_cadaster_parcel(gdf, cadaster_dir, cadaster_codes, how="left"):
    """Attach cadastral parcel geometry (and centroid) to buildings.

    Reads the INSPIRE CadastralParcel GML per municipality, keys it by
    ``building_reference`` and merges ``parcel_geometry`` onto ``gdf``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame | None
        Building frame to enrich. When None, only the parcel frame is returned.
    cadaster_dir : str
        Root of the downloaded cadastral layers.
    cadaster_codes : list[str]
        Municipalities to read.
    how : str, default "left"
        Merge strategy for joining parcels onto ``gdf``.

    Returns
    -------
    geopandas.GeoDataFrame | tuple
        ``gdf`` with ``parcel_geometry`` when ``gdf`` is given (returned as
        ``(gdf_joined, parcel_gdf)`` where ``parcel_gdf`` also carries
        ``parcel_centroid``); otherwise just the parcel frame.
    """
    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes...") if len(cadaster_codes)>1 else cadaster_codes:
        # sys.stderr.write("\r" + " " * 60)
        # sys.stderr.flush()
        # sys.stderr.write(f"\r\tJoining cadastral parcels for buildings in cadaster code: {code}")
        # sys.stderr.flush()

        parcel_gdf_ = gpd.read_file(f"{cadaster_dir}/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralparcel.gml",
                             layer="CadastralParcel")
        if "parcel_gdf" in locals():
            if parcel_gdf_.crs != parcel_gdf.crs:
                parcel_gdf_ = parcel_gdf_.to_crs(parcel_gdf.crs)
            parcel_gdf = gpd.GeoDataFrame(pd.concat([parcel_gdf, parcel_gdf_], ignore_index=True))
        else:
            parcel_gdf = parcel_gdf_

    parcel_gdf = parcel_gdf.rename({"geometry": "parcel_geometry", "localId": "building_reference"}, axis=1)
    parcel_gdf = parcel_gdf[["building_reference", "parcel_geometry"]]
    parcel_gdf = parcel_gdf.drop_duplicates(subset="building_reference", keep="first")
    if gdf is not None:
        gdf_joined = gdf.merge(parcel_gdf, on="building_reference", how=how)
    else:
        gdf_joined = parcel_gdf
    parcel_gdf = parcel_gdf.set_geometry("parcel_geometry")
    parcel_gdf["parcel_centroid"] = parcel_gdf.centroid

    return (gdf_joined,parcel_gdf) if gdf is not None else gdf_joined

def join_adm_div_naming(gdf, cadaster_dir, cadaster_codes):
    """Add administrative-division names (municipality/province/autonomous
    community) to ``gdf`` by joining on ``cadaster_code``.

    Parameters
    ----------
    gdf : (Geo)DataFrame
        Frame carrying a ``cadaster_code`` column.
    cadaster_dir : str
        Unused placeholder kept for signature symmetry with other joiners.
    cadaster_codes : list[str]
        Municipalities whose naming metadata to fetch.

    Returns
    -------
    (Geo)DataFrame
        ``gdf`` with municipality/province/autonomous-community name columns.
    """
    return pd.merge(gdf, utils.get_administrative_divisions_naming(cadaster_codes=cadaster_codes),
                    left_on="cadaster_code", right_on="cadaster_code", how="left")


def join_cadaster_data(cadaster_dir, census_tracts_dir, DEM_raster_dir, cadaster_codes, results_dir, open_street_dir,
                       building_parts_plots=False, building_parts_inference=False, plot_zones_ratio=0.01,
                       use_CAT_files=False, open_data_layers=False, open_data_layers_dir=None, CAT_files_dir = None,
                       floor_height=3.0, ground_floor_height=5.0, num_workers=-1,
                       force_update=False, cleanup_shards=True):
    """Orchestrate the full cadastral join: addresses + buildings + parcels + naming.

    This is the core of ``hc.merge()``. It reads addresses
    (:func:`get_cadaster_address`), joins building geometry and optional
    inference (:func:`join_cadaster_building`), attaches parcels
    (:func:`join_cadaster_parcel`) and administrative naming
    (:func:`join_adm_div_naming`), returning the unified building GeoDataFrame.

    Parameters
    ----------
    cadaster_dir, census_tracts_dir, DEM_raster_dir, results_dir, open_street_dir : str
        Input/output directories.
    cadaster_codes : list[str]
        Municipalities to process.
    building_parts_inference, use_CAT_files, open_data_layers : bool
        Feature toggles forwarded to the building join.
    building_parts_plots, plot_zones_ratio, open_data_layers_dir, CAT_files_dir,
    floor_height, ground_floor_height, num_workers, force_update, cleanup_shards :
        See :func:`join_cadaster_building`.

    Returns
    -------
    geopandas.GeoDataFrame
        The merged building-level frame (before the external layer joins in
        ``hc.merge``).
    """
    # Drop municipalities whose downloaded cadastral GMLs are missing (e.g. a
    # download that failed in a large province run), so one absent municipality
    # doesn't abort the whole merge. Each needs its address, building and parcel
    # GMLs to be processed.
    def _has_cadaster_files(code):
        return (
            os.path.exists(f"{cadaster_dir}/address/unzip/A.ES.SDGC.AD.{code}.gml") and
            os.path.exists(f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.building.gml") and
            os.path.exists(f"{cadaster_dir}/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralparcel.gml")
        )

    present = [c for c in cadaster_codes if _has_cadaster_files(c)]
    missing = [c for c in cadaster_codes if c not in present]
    if missing:
        sys.stderr.write(
            f"\nWarning: skipping {len(missing)} of {len(cadaster_codes)} municipalities "
            f"with missing cadastral GML files (re-run download() to fetch them): {missing}\n"
        )
    if not present:
        raise FileNotFoundError(
            f"None of the {len(cadaster_codes)} requested municipalities have their "
            f"cadastral GML files under '{cadaster_dir}'. Run download() first (and check "
            f"it completed)."
        )
    cadaster_codes = present

    # Address
    gdf = get_cadaster_address(
        cadaster_dir=cadaster_dir,
        cadaster_codes=cadaster_codes,
        directions_from_CAT_files=use_CAT_files,
        CAT_files_dir=CAT_files_dir,
        directions_from_open_data=open_data_layers,
        open_data_layers_dir=open_data_layers_dir
    )

    # Buildings
    # building_parts_inference_using_CAT_files = use_CAT_files
    # code = "08900"
    gdf = join_cadaster_building(gdf=gdf, cadaster_dir=cadaster_dir,
                                 census_tracts_dir=census_tracts_dir,
                                 DEM_raster_dir=DEM_raster_dir,
                                 cadaster_codes=cadaster_codes,
                                 results_dir=results_dir, open_street_dir=open_street_dir,
                                 building_parts_plots=building_parts_plots,
                                 plot_zones_ratio=plot_zones_ratio,
                                 building_parts_inference=building_parts_inference,
                                 building_parts_inference_using_CAT_files=use_CAT_files,
                                 open_data_layers=open_data_layers, open_data_layers_dir=open_data_layers_dir,
                                 CAT_files_dir = CAT_files_dir,
                                 floor_height=floor_height, ground_floor_height=ground_floor_height,
                                 num_workers=num_workers,
                                 force_update=force_update, cleanup_shards=cleanup_shards)
    # Administrative layers naming
    gdf = join_adm_div_naming(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes)

    gdf["building_centroid"] = gdf["building_geometry"].centroid
    gdf["building_centroid"] = np.where(gdf["building_geometry"] == None, gdf["location"], gdf["building_centroid"])
    if not "parcel_centroid" in gdf.columns:
        gdf = join_cadaster_parcel(gdf, cadaster_dir, cadaster_codes)[0]
        gdf["parcel_centroid"] = gdf["parcel_geometry"].centroid
    gdf["address_location"] = np.where(gdf["location"] == None, gdf["parcel_centroid"], gdf["location"])
    crs = gdf.crs
    gdf = gdf.drop(columns = ["location"])
    gdf = gdf.set_geometry("address_location")
    gdf = gdf.set_crs(crs)

    return gdf


def join_DEM_raster(gdf, raster_dir, geometry_column=None, chunk_size=100_000):
    """Sample DEM elevation at each row's geometry centroid.

    Chunked so peak RAM is O(chunk_size), not O(len(gdf)). Critically,
    centroids are computed *per chunk* and never materialised as a
    column: a 50M-row gdf would otherwise spawn 50M shapely Point
    objects (~10 GB) just to be sampled and thrown away.

    Caller does NOT need to add a centroid column or change the active
    geometry; pass `geometry_column` if the column to centroid is not
    the default `gdf.geometry`.

    Returns the same gdf with an added `elevation` column. The gdf's
    geometry column and CRS are untouched.
    """
    import numpy as np
    import pyproj

    sys.stderr.write("\nJoining the Digital Elevation Model information\n")

    geom = gdf[geometry_column] if geometry_column is not None else gdf.geometry
    # GeoSeries set on a non-active column inherits the gdf's CRS.
    src_crs = getattr(geom, "crs", None) or gdf.crs

    n = len(gdf)
    elev = np.full(n, np.nan, dtype=np.float64)

    transformer = pyproj.Transformer.from_crs(
        src_crs, "EPSG:4326", always_xy=True,
    )
    with rasterio.open(f"{raster_dir}/DEM.tif", "r") as rds:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_geom = geom.iloc[start:end]
            # Drop nulls within the chunk; positions are relative to the chunk.
            chunk_valid_pos = np.flatnonzero(~chunk_geom.isna().to_numpy())
            if chunk_valid_pos.size == 0:
                continue
            valid_geom = chunk_geom.iloc[chunk_valid_pos]
            # Centroids are computed for the chunk only and discarded
            # before the next iteration. shapely 2.x vectorises this.
            centroids = valid_geom.centroid
            xs = centroids.x.to_numpy()
            ys = centroids.y.to_numpy()
            del centroids, valid_geom, chunk_geom
            lons, lats = transformer.transform(xs, ys)
            del xs, ys
            # rds.sample is a generator; iterate to avoid a list of N samples.
            for j, sample in enumerate(rds.sample(zip(lons, lats))):
                elev[start + chunk_valid_pos[j]] = sample[0]

    gdf = gdf.copy(deep=False)
    gdf["elevation"] = elev
    return gdf

def join_by_census_tracts(gdf, census_tract_dir, columns=None, geometry_column = "census_geometry", year = 2022):
    """Spatially join INE census-tract codes onto buildings.

    Reads ``validated_census_{year}.gpkg`` and assigns each building the census
    section/district it falls within (``within`` predicate).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame to enrich.
    census_tract_dir : str
        Directory holding the validated census GPKG.
    columns : dict, optional
        Source→output column rename map (defaults to CUSEC→section_code,
        CUDIS→district_code, geometry→census_geometry).
    geometry_column : str, default "census_geometry"
        Name of the census geometry column after renaming.
    year : int, default 2022
        Census cartography year to load.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with ``section_code`` and ``district_code`` added.
    """
    if columns is None:
        columns = {
            "CUSEC": "section_code",
            "CUDIS": "district_code",
            "geometry": "census_geometry"
        }
    sys.stderr.write(f"\nJoining the census tracts\n")

    census_gdf = gpd.read_file(f"{census_tract_dir}/validated_census_{year}.gpkg")
    census_gdf.rename(columns = columns, inplace = True)
    census_gdf = census_gdf[columns.values()]
    census_gdf = census_gdf.set_geometry(geometry_column)
    census_gdf = census_gdf.to_crs(gdf.crs)
    census_gdf = gpd.sjoin(gdf, census_gdf, how="left", predicate="within").drop(["index_right"], axis=1)

    return census_gdf


def _ine_census2021_sections(wd):
    """Section-level INE 2021 Census indicators, loaded directly from the main
    ``C2021_Indicadores`` file and cached under ``{wd}/INE/Census2021_main``.

    Loaded here rather than via ``social_ES.INECensus2021`` because that helper
    additionally pulls supplementary INE files whose column layout INE has since
    changed (it currently raises ``KeyError: 'Municipio de residencia'``). The
    main indicators file already carries every field we need at section
    granularity. Returns a DataFrame keyed by ``section_code`` (CUSEC =
    province+municipality+district+section)."""
    path = os.path.join(wd, "INE", "Census2021_main")
    os.makedirs(path, exist_ok=True)
    fn = os.path.join(path, "sections.tsv")
    if os.path.exists(fn):
        return pd.read_csv(fn, sep="\t", dtype={"section_code": str})
    import requests
    from io import StringIO
    r = requests.get("https://www.ine.es/censos2021/C2021_Indicadores.csv",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    r.encoding = "utf-8"
    raw = pd.read_csv(StringIO(r.text), sep=",", dtype=str)
    rename = {
        "t3_1": "Average age",
        "t4_1": "Percentage of people under 16 years",
        "t4_3": "Percentage of people over 64 years of age",
        "t1_1": "Total People",
        "t21_1": "Total Households",
        "t22_1": "Households with 1 person",
        "t22_2": "Households with 2 people",
        "t22_3": "Households with 3 people",
        "t22_4": "Households of 4 people",
        "t22_5": "Households of 5 or more people",
    }
    for c in ("cpro", "cmun", "dist", "secc"):
        raw[c] = raw[c].astype(str).str.strip()
    raw["section_code"] = raw["cpro"] + raw["cmun"] + raw["dist"] + raw["secc"]
    keep = ["section_code"] + [c for c in rename if c in raw.columns]
    out = raw[keep].rename(columns=rename)
    out.to_csv(fn, sep="\t", index=False)
    return out


def join_sociodemographics(gdf, wd, year=None):
    """Join INE section-level sociodemographic/socioeconomic indicators onto each
    building by its census section (``section_code``, added by
    ``join_by_census_tracts``) → per-building columns

      * ``br__population_mean_age``            — mean age of the section (years)
      * ``br__population_share_over_64``       — % of the section aged >= 65
      * ``br__population_share_under_16``      — % of the section aged < 16
      * ``br__average_household_size``         — mean people per household
      * ``br__total_households``               — households in the section

    Source: the INE 2021 Population & Housing Census section indicators
    (``_ine_census2021_sections``). These feed the residential occupant-behaviour
    assignment in the energy model (``energy._assign_residential_user``). Degrades
    gracefully: on any missing column / failed download / absent ``section_code``
    the frame is returned unchanged (behaviour then falls back to per-dwelling
    area estimates + default demographic buckets)."""
    if "section_code" not in gdf.columns:
        sys.stderr.write("sociodemographics: no section_code column; skipping\n")
        return gdf
    try:
        census = _ine_census2021_sections(wd)
    except Exception as e:
        sys.stderr.write(f"sociodemographics: INE Census 2021 fetch failed ({e}); skipping\n")
        return gdf

    def _num(col):
        return (pd.to_numeric(census[col], errors="coerce")
                if col in census.columns else pd.Series(np.nan, index=census.index))

    h1, h2, h3 = _num("Households with 1 person"), _num("Households with 2 people"), _num("Households with 3 people")
    h4, h5 = _num("Households of 4 people"), _num("Households of 5 or more people")
    tot_hh = _num("Total Households")
    # Mean household size from the size distribution (5+ counted as 5.5); fall
    # back to Total People / Total Households when the distribution is absent.
    size_from_dist = (1 * h1 + 2 * h2 + 3 * h3 + 4 * h4 + 5.5 * h5) / tot_hh
    size_from_people = _num("Total People") / tot_hh
    avg_size = size_from_dist.where(size_from_dist.notna(), size_from_people)

    socio = pd.DataFrame({
        "_sd_key": census["section_code"].astype(str),
        "br__population_mean_age": _num("Average age"),
        "br__population_share_over_64": _num("Percentage of people over 64 years of age"),
        "br__population_share_under_16": _num("Percentage of people under 16 years"),
        "br__average_household_size": avg_size,
        "br__total_households": tot_hh,
    })
    socio = socio[socio["_sd_key"].str.len() >= 8].drop_duplicates("_sd_key")

    # ``section_code`` may be a scalar or (after address compaction) a 1-element
    # list; normalise to a scalar string key for the merge without mutating it.
    def _scalar(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            v = v[0] if len(v) else None
        return None if v is None else str(v)
    gdf = gdf.copy()
    gdf["_sd_key"] = gdf["section_code"].map(_scalar)
    n_match = gdf["_sd_key"].isin(socio["_sd_key"]).sum()
    gdf = gdf.merge(socio, on="_sd_key", how="left").drop(columns="_sd_key")
    sys.stderr.write(f"sociodemographics: matched {n_match}/{len(gdf)} buildings to section demographics\n")
    return gdf


def get_census_gdf(census_tract_dir, columns=None, geometry_column="geometry", year=2022, crs="EPSG:4326"):
    """Load census tracts as a GeoDataFrame with administrative naming columns.

    Parameters
    ----------
    census_tract_dir : str
        Directory holding the validated census GPKG.
    columns : dict, optional
        Source→output rename map (defaults to municipality/province/autonomous
        community names plus section/district codes and geometry).
    geometry_column : str, default "geometry"
        Active geometry column after renaming.
    year : int, default 2022
        Census cartography year.
    crs : str, default "EPSG:4326"
        Output CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Census tracts with naming/code columns in ``crs``.
    """
    if columns is None:
        columns = {
            "CUMUN": "ine_municipality_code",
            "NMUN": "municipality_name",
            "NPRO": "province_name",
            "NCA": "autonomous_community_name",
            "CUSEC": "section_code",
            "CUDIS": "district_code",
            "geometry": "geometry"
        }
    sys.stderr.write(f"\nReading census administrative divisions\n")

    census_gdf = gpd.read_file(f"{census_tract_dir}/validated_census_{year}.gpkg")
    census_gdf.rename(columns = columns, inplace = True)
    census_gdf = census_gdf[columns.values()]
    census_gdf = census_gdf.set_geometry(geometry_column)
    census_gdf = census_gdf.to_crs(crs)

    return census_gdf

def join_by_neighbourhoods(gdf, neighbourhoods_dir, columns=None, geometry_column="neighborhood_geometry"):
    """Spatially join Barcelona neighborhood/district names onto buildings.

    Barcelona-only: reads ``neighbourhoods.csv`` (WKT geometries) and assigns
    each building the neighborhood/district it falls within. Raises if the file
    is absent, so keep ``neighborhood_layer=False`` for other municipalities.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame to enrich.
    neighbourhoods_dir : str
        Directory holding ``neighbourhoods.csv``.
    columns : dict, optional
        Source→output rename map (defaults to codi_barri→neighborhood_code, …).
    geometry_column : str, default "neighborhood_geometry"
        Neighborhood geometry column after renaming.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with neighborhood/district columns added.
    """
    if columns is None:
        columns = {
            "codi_barri": "neighborhood_code",
            "nom_barri": "neighborhood_name",
            "nom_districte": "district_name",
            "geometria_etrs89": "neighborhood_geometry"
        }
    sys.stderr.write(f"\nJoining the neighborhoods description\n")

    neighbourhoods_gdf = gpd.read_file(f"{neighbourhoods_dir}/neighbourhoods.csv")
    neighbourhoods_gdf.rename(columns = columns, inplace = True)
    neighbourhoods_gdf = neighbourhoods_gdf[columns.values()]
    neighbourhoods_gdf[geometry_column] = neighbourhoods_gdf[geometry_column].apply(wkt.loads)
    neighbourhoods_gdf = gpd.GeoDataFrame(neighbourhoods_gdf, geometry=geometry_column, crs='EPSG:25831')
    neighbourhoods_gdf = neighbourhoods_gdf.to_crs(gdf.crs)
    neighbourhoods_gdf = gpd.sjoin(gdf, neighbourhoods_gdf, how="left",
              predicate="within").drop(["index_right"], axis=1)

    return neighbourhoods_gdf


def join_by_postal_codes(gdf, postal_codes_dir, columns=None, geometry_column="postal_code_geometry"):
    """Spatially join Spanish postal codes onto buildings.

    Reads ``postal_codes.geojson`` and assigns each building the postal code
    polygon it falls within.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame to enrich.
    postal_codes_dir : str
        Directory holding ``postal_codes.geojson``.
    columns : dict, optional
        Source→output rename map (defaults to CODPOS→postal_code, geometry→…).
    geometry_column : str, default "postal_code_geometry"
        Postal-code geometry column after renaming.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with a ``postal_code`` column added.
    """
    if columns is None:
        columns = {
            "CODPOS": "postal_code",
            "geometry": "postal_code_geometry"
        }
    sys.stderr.write(f"\nJoining the postal codes\n")

    postal_codes_gdf = gpd.read_file(f"{postal_codes_dir}/postal_codes.geojson")
    postal_codes_gdf.rename(columns = columns, inplace = True)
    postal_codes_gdf = postal_codes_gdf[columns.values()]
    postal_codes_gdf = gpd.GeoDataFrame(postal_codes_gdf, geometry=geometry_column, crs='EPSG:4326')
    postal_codes_gdf = postal_codes_gdf.to_crs(gdf.crs)
    postal_codes_gdf = gpd.sjoin(gdf, postal_codes_gdf,
              how="left", predicate="within").drop(["index_right"], axis=1)

    return postal_codes_gdf


def join_pois(gdf, open_street_dir, cadaster_dir, cadaster_codes):
    """Join POI indicators to buildings from OpenStreetMap data.

    For each POI group/subgroup (amenity, emergency, healthcare, leisure,
    office, shop, tourism) the following columns are added per building:

    - poi_{group}[_{subgroup}]_count  : number of POIs assigned to the building
    - poi_{group}[_{subgroup}]_ids    : list of POI identifiers on the building
    - poi_{group}[_{subgroup}]_100m   : count of POIs within 100 m
    - poi_{group}[_{subgroup}]_250m   : count of POIs within 250 m
    - poi_{group}[_{subgroup}]_500m   : count of POIs within 500 m
    - poi_{group}[_{subgroup}]_1000m  : count of POIs within 1000 m
    - poi_{group}[_{subgroup}]_1500m  : count of POIs within 1500 m
    """
    from collections import defaultdict

    GROUPS_OF_INTEREST = {"amenity", "emergency", "healthcare", "leisure", "office", "shop", "tourism"}
    DISTANCES = [100, 250, 500, 1000, 1500]
    METRIC_CRS = "EPSG:25831"

    sys.stderr.write(f"\nJoining POI indicators for {len(cadaster_codes)} municipalities\n")

    # Collect POIs across all cadaster codes, keyed by (group, subgroup)
    poi_collections = defaultdict(list)

    codes_iter = tqdm(cadaster_codes, desc="Fetching POIs by cadaster code") if len(cadaster_codes) > 1 else cadaster_codes
    for code in codes_iter:
        try:
            pois = utils.get_pois_open_street_maps(
                cadaster_code=code,
                cadaster_dir=cadaster_dir,
                open_street_dir=open_street_dir,
                # fetch a margin at least as wide as the largest proximity ring
                # so edge buildings' 1500 m counts aren't truncated
                buffer_m=max(DISTANCES),
            )
        except Exception as e:
            sys.stderr.write(f"Warning: Could not get POIs for {code}: {e}\n")
            continue

        for group, value in pois.items():
            if group not in GROUPS_OF_INTEREST:
                continue
            if isinstance(value, dict):
                for subgroup, poi_gdf in value.items():
                    if poi_gdf is not None and not poi_gdf.empty:
                        poi_collections[(group, subgroup)].append(poi_gdf)
            elif isinstance(value, gpd.GeoDataFrame) and not value.empty:
                poi_collections[(group, None)].append(value)

    if not poi_collections:
        return gdf

    # Merge GeoDataFrames of same group/subgroup across municipalities
    all_pois = {}
    for key, gdfs in poi_collections.items():
        valid = [g for g in gdfs if g is not None and not g.empty]
        if not valid:
            continue
        if len(valid) == 1:
            all_pois[key] = valid[0]
        else:
            merged = pd.concat(valid, ignore_index=True)
            all_pois[key] = gpd.GeoDataFrame(merged, geometry="geometry", crs=valid[0].crs)

    def _id_col(poi_gdf):
        for col in ("id", "osmid", "name"):
            if col in poi_gdf.columns:
                return col
        return None

    # ------------------------------------------------------------------
    # 1. Direct building_reference assignment
    # ------------------------------------------------------------------
    for (group, subgroup), poi_gdf in all_pois.items():
        col_prefix = f"poi_{group}_{subgroup}" if subgroup else f"poi_{group}"

        if "building_reference" not in poi_gdf.columns:
            continue

        poi_valid = poi_gdf[poi_gdf["building_reference"].notna()].copy()
        if poi_valid.empty:
            continue

        counts = (
            poi_valid.groupby("building_reference")
            .size()
            .reset_index(name=f"{col_prefix}_count")
        )
        gdf = gdf.merge(counts, on="building_reference", how="left")
        gdf[f"{col_prefix}_count"] = gdf[f"{col_prefix}_count"].fillna(0).astype(int)

        id_col = _id_col(poi_valid)
        if id_col:
            ids = (
                poi_valid.groupby("building_reference")[id_col]
                .apply(list)
                .reset_index(name=f"{col_prefix}_ids")
            )
            gdf = gdf.merge(ids, on="building_reference", how="left")

    # ------------------------------------------------------------------
    # 2. Distance-based counts (100 m, 250 m, 500 m, 1000 m, 1500 m)
    # ------------------------------------------------------------------
    if "building_centroid" in gdf.columns:
        buildings_pts = (
            gdf[["building_reference", "building_centroid"]]
            .drop_duplicates("building_reference")
            .copy()
        )
        buildings_pts = gpd.GeoDataFrame(buildings_pts, geometry="building_centroid", crs=gdf.crs)
    elif "building_geometry" in gdf.columns:
        buildings_pts = (
            gdf[["building_reference", "building_geometry"]]
            .drop_duplicates("building_reference")
            .copy()
        )
        buildings_pts = gpd.GeoDataFrame(buildings_pts, geometry="building_geometry", crs=gdf.crs)
        buildings_pts["building_centroid"] = buildings_pts.geometry.centroid
        buildings_pts = buildings_pts.set_geometry("building_centroid").drop(columns=["building_geometry"])
    else:
        sys.stderr.write("Warning: No building geometry found for POI distance calculation\n")
        return gdf

    buildings_pts = buildings_pts[buildings_pts.geometry.notna()].to_crs(METRIC_CRS)
    buildings_pts = buildings_pts.rename_geometry("geometry")

    for (group, subgroup), poi_gdf in all_pois.items():
        col_prefix = f"poi_{group}_{subgroup}" if subgroup else f"poi_{group}"

        poi_metric = poi_gdf.to_crs(METRIC_CRS)
        poi_metric = poi_metric[poi_metric.geometry.notna()][["geometry"]].copy()

        if poi_metric.empty:
            continue

        for dist in DISTANCES:
            col_name = f"{col_prefix}_{dist}m"

            buffered = buildings_pts[["building_reference", "geometry"]].copy()
            buffered = buffered.set_geometry("geometry")
            buffered["geometry"] = buffered.geometry.buffer(dist)

            joined = gpd.sjoin(buffered, poi_metric, how="left", predicate="contains")

            counts_dist = (
                joined[joined["index_right"].notna()]
                .groupby("building_reference")
                .size()
                .reset_index(name=col_name)
            )
            gdf = gdf.merge(counts_dist, on="building_reference", how="left")
            gdf[col_name] = gdf[col_name].fillna(0).astype(int)

    return gdf


# ==========================================
# SPATIAL AGGREGATION (grid / hexbins)
# ==========================================

# Default aggregation operation per known building-level column. Columns not
# listed here and not provided via the user `aggregations` kwarg are dropped.
DEFAULT_BUILDING_AGGS = {
    # Building stock counts and areas
    "n_dwellings": "sum",
    "n_building_units": "sum",
    "n_floors_above_ground": "mean",
    "n_floors_below_ground": "mean",
    "building_area": "sum",
    "year_of_construction": "mean",
    "elevation": "mean",
    # CAT-derived use type areas
    "building_area_residential": "sum",
    "building_area_commercial": "sum",
    "building_area_offices": "sum",
    "building_area_warehouse_parking": "sum",
    "building_area_industrial": "sum",
    "building_area_religious": "sum",
    "building_area_cultural": "sum",
    "building_area_healthcare_and_charity": "sum",
    "building_area_sports_facilities": "sum",
    "building_area_entertainment_venues": "sum",
    "building_area_leisure_and_hospitality": "sum",
    "building_area_singular_building": "sum",
    "building_area_urbanization_and_landscaping_works_undeveloped_land": "sum",
    "building_area_unknown": "sum",
    # Inference outputs (scalars)
    "br__floors_above_ground": "mean",
    "br__floors_below_ground": "mean",
    "br__building_footprint_area": "sum",
    "br__above_ground_built_area": "sum",
    "br__below_ground_built_area": "sum",
    "br__above_ground_roof_area": "sum",
    "br__adiabatic_wall": "sum",
    "br__walls_between_slabs": "sum",
    "br__patios_wall_total": "sum",
    "br__parcel_main_orientation": "mean",
    "br__street_width_main_orientation": "mean",
    "br__next_building_main_orientation": "mean",
    "br__exists_ground_commercial_premises": "sum",
    "br__detached": "sum",
}

DEFAULT_BUILDING_AGGS = PrettyDict(DEFAULT_BUILDING_AGGS)
"""Default aggregation operation per known building-level column, used by
:func:`aggregate_to_grid` / :func:`aggregate_to_h3`. Columns absent from both
this mapping and the caller's ``aggregations`` argument are dropped."""


def _coerce_centroid_gdf(gdf, geometry_col):
    """Return a GeoDataFrame using `geometry_col` (or its centroid) as geometry."""
    if geometry_col in gdf.columns:
        pts = gdf.set_geometry(geometry_col)
    else:
        pts = gdf.copy()
        if not isinstance(pts, gpd.GeoDataFrame) or pts.geometry.name is None:
            raise ValueError(
                f"geometry_col '{geometry_col}' not found and gdf has no active geometry"
            )
        if pts.geometry.geom_type.dropna().iloc[0] != "Point":
            pts["__centroid__"] = pts.geometry.centroid
            pts = pts.set_geometry("__centroid__")
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    return pts


def _resolve_aggregations(columns, aggregations):
    """Merge user overrides with defaults; return (agg_map, dropped, missing)."""
    user = dict(aggregations) if aggregations else {}
    agg_map = {}
    for col in columns:
        if col in user:
            agg_map[col] = user[col]
        elif col in DEFAULT_BUILDING_AGGS:
            agg_map[col] = DEFAULT_BUILDING_AGGS[col]
    dropped = [c for c in columns if c not in agg_map]
    missing = [c for c in user if c not in columns]
    return agg_map, dropped, missing


def aggregate_to_grid(gdf, cell_size_m=100, aggregations=None,
                      geometry_col="building_centroid", cell_id_col="cell_id"):
    """Aggregate a building-level GeoDataFrame onto a regular square grid.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of `join_cadaster_data` (or `hc.merge`). Must be in a metric CRS.
    cell_size_m : float, default 100
        Side length of each square cell in metres.
    aggregations : dict[str, str|callable], optional
        Override or extend `DEFAULT_BUILDING_AGGS`. Columns absent from both
        the override and the defaults are dropped (with a warning printed).
    geometry_col : str, default "building_centroid"
        Column to use as the building point for cell assignment.
    cell_id_col : str, default "cell_id"
        Name of the integer cell-id column on the output.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per non-empty cell, geometry = cell polygon, with `n_buildings`
        plus all aggregated columns.
    """
    if gdf.empty:
        return gpd.GeoDataFrame(columns=[cell_id_col, "n_buildings", "geometry"],
                                geometry="geometry", crs=gdf.crs)

    pts = _coerce_centroid_gdf(gdf, geometry_col)
    if pts.crs is None or not pts.crs.is_projected:
        raise ValueError(
            "aggregate_to_grid requires a projected (metric) CRS. "
            "Reproject the input GeoDataFrame first (e.g. to EPSG:25831)."
        )

    minx, miny, maxx, maxy = pts.total_bounds
    # Snap origin to the cell grid so cell IDs are stable across calls.
    x0 = np.floor(minx / cell_size_m) * cell_size_m
    y0 = np.floor(miny / cell_size_m) * cell_size_m

    ix = np.floor((pts.geometry.x.values - x0) / cell_size_m).astype(int)
    iy = np.floor((pts.geometry.y.values - y0) / cell_size_m).astype(int)
    pts = pts.assign(__ix=ix, __iy=iy)

    candidate_cols = [c for c in gdf.columns
                      if c not in ("geometry", geometry_col, "building_geometry",
                                   "parcel_geometry", "parcel_centroid",
                                   "address_location", "building_centroid")]
    agg_map, dropped, missing = _resolve_aggregations(candidate_cols, aggregations)
    if missing:
        sys.stderr.write(
            f"aggregate_to_grid: ignoring {len(missing)} aggregations for columns not in gdf: {missing}\n"
        )

    pts_subset = pts[["__ix", "__iy"] + list(agg_map.keys())]
    grouped = pts_subset.groupby(["__ix", "__iy"], as_index=False)
    counts = grouped.size().rename(columns={"size": "n_buildings"})
    if agg_map:
        agg = grouped.agg(agg_map)
        agg = agg.merge(counts, on=["__ix", "__iy"], how="left")
    else:
        agg = counts

    def _cell_geom(row):
        cx = x0 + row["__ix"] * cell_size_m
        cy = y0 + row["__iy"] * cell_size_m
        return box(cx, cy, cx + cell_size_m, cy + cell_size_m)

    agg["geometry"] = agg.apply(_cell_geom, axis=1)
    agg[cell_id_col] = agg["__ix"].astype(str) + "_" + agg["__iy"].astype(str)
    agg = agg.drop(columns=["__ix", "__iy"])
    return gpd.GeoDataFrame(agg, geometry="geometry", crs=pts.crs)


def aggregate_to_h3(gdf, resolution=10, aggregations=None,
                    geometry_col="building_centroid", cell_id_col="h3_id"):
    """Aggregate a building-level GeoDataFrame onto H3 hexagons.

    Requires the `h3` package (>= 4.x). Resolution 10 ≈ 15 000 m² hexagons,
    similar in order of magnitude to a 100 m grid cell; resolution 11 ≈
    2 100 m² for finer aggregation.
    """
    try:
        import h3
    except ImportError as exc:
        raise ImportError(
            "aggregate_to_h3 requires the h3 package. Install with `pip install h3`."
        ) from exc

    if gdf.empty:
        return gpd.GeoDataFrame(columns=[cell_id_col, "n_buildings", "geometry"],
                                geometry="geometry", crs="EPSG:4326")

    pts = _coerce_centroid_gdf(gdf, geometry_col)
    pts_wgs = pts.to_crs("EPSG:4326")

    lons = pts_wgs.geometry.x.values
    lats = pts_wgs.geometry.y.values
    cell_ids = [h3.latlng_to_cell(float(lat), float(lon), resolution)
                for lat, lon in zip(lats, lons)]
    pts = pts.assign(__h3=cell_ids)

    candidate_cols = [c for c in gdf.columns
                      if c not in ("geometry", geometry_col, "building_geometry",
                                   "parcel_geometry", "parcel_centroid",
                                   "address_location", "building_centroid")]
    agg_map, dropped, missing = _resolve_aggregations(candidate_cols, aggregations)
    if missing:
        sys.stderr.write(
            f"aggregate_to_h3: ignoring {len(missing)} aggregations for columns not in gdf: {missing}\n"
        )

    pts_subset = pts[["__h3"] + list(agg_map.keys())]
    grouped = pts_subset.groupby("__h3", as_index=False)
    counts = grouped.size().rename(columns={"size": "n_buildings"})
    if agg_map:
        agg = grouped.agg(agg_map)
        agg = agg.merge(counts, on="__h3", how="left")
    else:
        agg = counts

    def _hex_geom(hid):
        # h3 v4: cell_to_boundary returns (lat, lng) tuples
        boundary = h3.cell_to_boundary(hid)
        coords = [(lon, lat) for lat, lon in boundary]
        return Polygon(coords)

    agg["geometry"] = agg["__h3"].map(_hex_geom)
    agg = agg.rename(columns={"__h3": cell_id_col})
    return gpd.GeoDataFrame(agg, geometry="geometry", crs="EPSG:4326")
