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
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import rasterio
import regex as re
from shapely import wkt
from shapely.geometry import Point
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
    sys.stderr.write(f"\nReading the cadaster addresses for {len(cadaster_codes)} municipalities\n")

    address_gdf = gpd.GeoDataFrame()
    address_street_names_df = gpd.GeoDataFrame()

    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes..."):
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
            addresses_CAT_ = (pl.concat([
                buildings_CAT[["building_reference", "street_type", "street_name", "street_number1"]].rename(
                    {"street_number1": "street_number"}),
                buildings_CAT[["building_reference", "street_type", "street_name", "street_number2"]].rename(
                    {"street_number2": "street_number"}).filter(pl.col("street_number") != "")], how="vertical")
            ).to_pandas()
            addresses_CAT_["street_number_label"], addresses_CAT_["street_number_clean"] = utils.normalize_street_number(
                addresses_CAT_["street_number"]
            )
            addresses_CAT = pd.concat([addresses_CAT, addresses_CAT_], ignore_index=True)
            addresses_CAT['cadaster_code'] = code

        def calculate_centroid(group):
            valid_points = [pt for pt in group["location"] if isinstance(pt, Point)]

            if not valid_points:
                return pd.Series({"location": None})

            x_coords = [pt.x for pt in valid_points]
            y_coords = [pt.y for pt in valid_points]

            centroid = Point(np.mean(x_coords), np.mean(y_coords))

            return pd.Series({"location": centroid})

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

    # Add cadaster_code column if not present
    if 'cadaster_code' not in gdf.columns:
        gdf['cadaster_code'] = gdf['building_reference'].str[:5]
    
    # Initialize zone columns
    gdf['zone_reference'] = "unassigned"
    gdf['zone_type'] = "urban"
    
    processed_buildings = 0

    codes = tqdm(cadaster_codes, desc="Processing zones by municipality") \
        if isinstance(cadaster_codes, list) else [cadaster_codes]

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
            
            # Create building centroids for zone assignment
            buildings_for_code['centroid'] = buildings_for_code['building_geometry'].centroid
            buildings_centroids = buildings_for_code.set_geometry('centroid')
            
            # Align CRS
            if buildings_centroids.crs != zone_gdf.crs:
                buildings_centroids = buildings_centroids.to_crs(zone_gdf.crs)
            
            # Simple spatial join using building centroids - much more accurate than address points!
            joined = gpd.sjoin(buildings_centroids, zone_gdf, how="left", predicate="within")
            
            # Check what columns we got from the join
            right_zone_ref_col = None
            right_zone_type_col = None
            for col in joined.columns:
                if col.endswith('zone_reference'):
                    right_zone_ref_col = col
                elif col.endswith('zone_type'):
                    right_zone_type_col = col
            
            # Update the main dataframe with zone assignments
            if len(joined) > 0 and right_zone_ref_col and right_zone_type_col:
                # Get successful assignments
                assigned_mask = joined[right_zone_ref_col].notna()
                assigned_buildings = joined[assigned_mask]
                
                for idx in assigned_buildings.index:
                    zone_ref = assigned_buildings.loc[idx, right_zone_ref_col]
                    zone_type = assigned_buildings.loc[idx, right_zone_type_col]
                    
                    gdf.loc[idx, "zone_reference"] = zone_ref
                    # Clean up zone types
                    if zone_type == "MANZANA ":
                        gdf.loc[idx, "zone_type"] = "urban"
                    elif zone_type == "POLIGONO ":
                        gdf.loc[idx, "zone_type"] = "disseminated"
                    else:
                        gdf.loc[idx, "zone_type"] = "urban"  # default
            
        except Exception as e:
            sys.stderr.write(f"Error processing zones for {code}: {e}\n")
            continue

    # Remove temporary columns (only if we added it)
    # Note: we always keep cadaster_code as it might be used elsewhere
    
    total_time = time.time() - start_time
    unassigned_count = (gdf["zone_reference"] == "unassigned").sum() if len(gdf) > 0 else 0

    return gdf


def join_cadaster_building(gdf, cadaster_dir, cadaster_codes, results_dir, open_street_dir, building_parts_plots=False,
                           plot_zones_ratio=0.01, building_parts_inference=False,
                           building_parts_inference_using_CAT_files=False, open_data_layers=False,
                           open_data_layers_dir=None, CAT_files_dir=None):

    sys.stderr.write(f"\nJoining the buildings description for {len(cadaster_codes)} municipalities\n")

    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes..."):
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

            building_part_gdf_ = gpd.read_file(f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.buildingpart.gml",
                                          layer="BuildingPart")

            sys.stderr.write("\r" + " " * 60)
            building_part_gdf_.rename(columns={
                'geometry': 'building_part_geometry',
                'numberOfFloorsAboveGround': 'n_floors_above_ground',
                'numberOfFloorsBelowGround': 'n_floors_below_ground',
                'localId': 'building_reference'
            }, inplace=True)
            building_part_gdf_['building_reference'] = building_part_gdf_['building_reference'].str.split("_").str[0]
            building_part_gdf_.drop(
                ['gml_id', 'beginLifespanVersion', 'conditionOfConstruction',
                 'namespace', 'horizontalGeometryEstimatedAccuracy',
                 'horizontalGeometryEstimatedAccuracy_uom',
                 'horizontalGeometryReference', 'referenceGeometry', 'heightBelowGround',
                 'heightBelowGround_uom'],
                inplace=True, axis=1)

            if gdf is not None:
                gdf_unique = gdf.drop_duplicates(subset='building_reference')
                building_part_gdf_ = building_part_gdf_.join(gdf_unique.set_index('building_reference'),
                                                           on="building_reference", how="left")
            
            # Ensure zone columns exist
            if "zone_type" not in building_part_gdf_.columns:
                building_part_gdf_["zone_type"] = "unknown"
            if "zone_reference" not in building_part_gdf_.columns:
                building_part_gdf_["zone_reference"] = "unknown"
                
            # Fill missing values
            building_part_gdf_.loc[building_part_gdf_["zone_type"].isna(), "zone_type"] = "unknown"
            building_part_gdf_.loc[building_part_gdf_["zone_reference"].isna(), "zone_reference"] = "unknown"
            # building_part_gdf_.drop(columns=["building_part_geometry"]).set_geometry("location").to_file("test.gpkg")

            # building_part_gdf_ = building_part_gdf_.merge(building_gdf_[['building_reference','building_status']])

            # In case of Barcelona municipality analysis, use commercial establishments and ground premises datasets
            if code=="08900" and open_data_layers:
                # establishments = pd.read_csv(
                #     filepath_or_buffer=f"{open_data_layers_dir}/barcelona_establishments.csv",
                #     encoding=from_path(f"{open_data_layers_dir}/barcelona_establishments.csv").best().encoding,
                #     on_bad_lines='skip',
                #     sep=",")
                ground_premises = downloaders.load_and_transform_barcelona_ground_premises(open_data_layers_dir)
                building_part_gdf_ = building_part_gdf_.join(ground_premises.set_index("building_reference"),
                                                             on="building_reference", how="left")
            # building_part_gdf_.loc[
            #    ((building_part_gdf_.n_floors_above_ground == 0) &
            #     (building_part_gdf_.n_floors_below_ground == 1)),
            #    "n_floors_above_ground"] = 1

            # Join the parcel
            building_part_gdf_, parcels_gdf = join_cadaster_parcel(building_part_gdf_, cadaster_dir, [code])

            # Process the building parts
            building_part_gdf_ = building_inference.process_building_parts(
                code=code, building_part_gdf_=building_part_gdf_, buildings_CAT=buildings_CAT,
                parcels_gdf=parcels_gdf, results_dir=results_dir, cadaster_dir=cadaster_dir,
                open_street_dir=open_street_dir, plots=building_parts_plots, plot_zones_ratio=plot_zones_ratio)

            # Unpack the results: (sbr_results, br_results)
            sbr_results, br_results = building_part_gdf_

            # Check if results are valid
            if br_results is not None and len(br_results) > 0:
                # Join Building geodataframe with br_results
                building_gdf_ = (building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                              'building_geometry','year_of_construction', 'zone_reference', 'zone_type']].
                                merge(br_results, left_on="building_reference",
                                      right_on="building_reference", how="left"))

                # Also merge SBR results directly (they already have sbr__ column prefixes)
                if sbr_results is not None and len(sbr_results) > 0:

                    sbr_cols = [col for col in sbr_results.columns if col.startswith("sbr__")]

                    # ------------------------------------------------------------
                    # 1️⃣ Count SBR per building
                    # ------------------------------------------------------------
                    sbr_counts = (
                        sbr_results.groupby("building_reference")["single_building_reference"]
                        .nunique()
                        .rename("n_sbr")
                        .reset_index()
                    )

                    sbr_results = sbr_results.merge(sbr_counts, on="building_reference", how="left")

                    # ------------------------------------------------------------
                    # 2️⃣ MULTI SBR → dict
                    # ------------------------------------------------------------
                    multi_sbr = sbr_results[sbr_results["n_sbr"] > 1].copy()

                    if not multi_sbr.empty:

                        def aggregate_sbr(group):
                            result = {}

                            for col in sbr_cols:
                                result_col = {}

                                for _, row in group.iterrows():
                                    val = row[col]

                                    if utils.is_valid_value(val):
                                        sbr_id = row["single_building_reference"]
                                        result_col[sbr_id] = val

                                result[col] = result_col if result_col else None

                            return pd.Series(result)

                        sbr_multi = (
                            multi_sbr
                            .groupby("building_reference")
                            .apply(aggregate_sbr, include_groups=False)
                            .reset_index()
                        )

                    else:
                        sbr_multi = pd.DataFrame(columns=["building_reference"] + sbr_cols)

                    # ------------------------------------------------------------
                    # 3️⃣ SINGLE SBR → force None
                    # ------------------------------------------------------------
                    single_refs = sbr_results.loc[sbr_results["n_sbr"] == 1, "building_reference"].unique()

                    if len(single_refs) > 0:
                        sbr_single = pd.DataFrame({
                            "building_reference": single_refs
                        })

                        for col in sbr_cols:
                            sbr_single[col] = None
                    else:
                        sbr_single = pd.DataFrame(columns=["building_reference"] + sbr_cols)

                    # ------------------------------------------------------------
                    # 4️⃣ Combine
                    # ------------------------------------------------------------
                    sbr_final = pd.concat([sbr_multi, sbr_single], ignore_index=True)

                    # ------------------------------------------------------------
                    # 5️⃣ Merge into building_gdf_
                    # ------------------------------------------------------------
                    building_gdf_ = building_gdf_.merge(
                        sbr_final,
                        on="building_reference",
                        how="left"
                    )

                if "building_part_gdf" in locals():
                    building_part_gdf = pd.concat([building_part_gdf, br_results], ignore_index=True)
                else:
                    building_part_gdf = br_results
            else:
                sys.stderr.write("Warning: Building parts inference returned no results\n")

        if buildings_CAT is not None:
            #['n_building_units', 'n_dwellings', 'n_floors_above_ground', 'building_area']
            if not building_parts_inference:
                building_gdf_ = building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                             'building_geometry', 'year_of_construction', 'zone_reference', 'zone_type']]

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
            final_df = summary.join(area_pivot, on="building_reference", how="left").to_pandas()
            building_gdf_ = pd.merge(building_gdf_, final_df, on="building_reference", how="left")

        elif not building_parts_inference:
            # Only strip to basic columns when building_parts_inference is explicitly False
            building_gdf_ = building_gdf_[['gml_id', 'building_status', 'building_reference', 'building_use',
                                         'building_geometry', 'year_of_construction', 'n_building_units', 'n_dwellings',
                                         'n_floors_above_ground', 'building_area', 'zone_reference', 'zone_type']]
        # When building_parts_inference=True but buildings_CAT=None, keep all columns (including br__ and sbr_data)

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
        merged_gdf = pd.merge(gdf, building_gdf, left_on="building_reference", right_on="building_reference", how="left")
    else:
        merged_gdf = building_gdf

    return merged_gdf


def join_cadaster_parcel(gdf, cadaster_dir, cadaster_codes, how="left"):

    for code in tqdm(cadaster_codes, desc="Iterating by cadaster codes..."):
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

    return pd.merge(gdf, utils.get_administrative_divisions_naming(cadaster_codes=cadaster_codes),
                    left_on="cadaster_code", right_on="cadaster_code", how="left")


def join_cadaster_data(cadaster_dir, cadaster_codes, results_dir, open_street_dir, building_parts_plots=False,
                       building_parts_inference=False, plot_zones_ratio=0.01, use_CAT_files=False,
                       open_data_layers=False, open_data_layers_dir=None, CAT_files_dir = None):

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
    gdf = join_cadaster_building(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes,
                                 results_dir=results_dir, open_street_dir=open_street_dir,
                                 building_parts_plots=building_parts_plots,
                                 plot_zones_ratio=plot_zones_ratio,
                                 building_parts_inference=building_parts_inference,
                                 building_parts_inference_using_CAT_files=use_CAT_files,
                                 open_data_layers=open_data_layers, open_data_layers_dir=open_data_layers_dir,
                                 CAT_files_dir = CAT_files_dir)
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


def join_DEM_raster(gdf, raster_dir):

    sys.stderr.write(f"\nJoining the Digital Elevation Model information\n")

    with rasterio.open(f"{raster_dir}/DEM.tif", 'r+') as rds:
        ini_crs = gdf.crs
        gdf = gdf.to_crs(epsg=4326)
        gdf_ = gdf[~gdf.geometry.isna()]
        gdf_.loc[:,"elevation"] = [x[0] for x in rds.sample(
            [(x, y) for x, y in zip(gdf_.geometry.x, gdf_.geometry.y)])]
        gdf = pd.concat([gdf_,gdf[gdf.geometry.isna()]], axis=0)
        gdf = gdf.to_crs(ini_crs)

    return gdf

def join_by_census_tracts(gdf, census_tract_dir, columns=None, geometry_column = "census_geometry", year = 2022):

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


def get_census_gdf(census_tract_dir, columns=None, geometry_column="geometry", year=2022, crs="EPSG:4326"):

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
