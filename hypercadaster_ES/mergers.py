"""Data merging and joining operations for hypercadaster_ES.

This module provides functions for joining Spanish cadastral data with
various external geographic and administrative datasets.

Main functions:
    - join_cadaster_data(): Main orchestrator for joining all cadastral data
    - get_cadaster_address(): Extract and process address information
    - join_cadaster_building(): Join building geometry and attributes
    - join_cadaster_zone(): Join cadastral zoning information
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
    gdf['street_number_clean'] = gdf['street_number'].str.extract(r'(\d+(?=.*))').fillna(0).astype(int)
    # gdf = gdf[gdf['specification'] == "Entrance"]
    gdf["building_reference"] = gdf['localId'].apply(lambda x: x.split('.')[-1])

    gdf.drop(
        ["gml_id", "namespace", "localId", "beginLifespanVersion", "validFrom", "level", "type", "method", "default"],
        inplace=True, axis=1)
    gdf = gdf.set_geometry("location")
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
            addresses_CAT_['street_number_clean'] = addresses_CAT_['street_number'].str.extract(r'(\d+(?=.*))').fillna(0).astype(int)
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

    if directions_from_open_data and "08900" in cadaster_codes:
        bcn_open_data_streets = gpd.read_file(f"{open_data_layers_dir}/barcelona_carrerer.gpkg")
        def extract_up_to_second_capital_or_number(text):
            # Find all positions of capitalized words
            capital_matches = list(re.finditer(r'\b\p{Lu}[^\s]*', text))
            # Find all positions of numbers
            number_match = re.search(r'\b\d+', text)
            # Determine cut-off point: min of second capital or first number
            cutoff_indices = []
            if len(capital_matches) >= 2:
                cutoff_indices.append(capital_matches[1].start())
            if number_match:
                cutoff_indices.append(number_match.start())
            if cutoff_indices:
                return text[:min(cutoff_indices)]
            return None
        bcn_open_data_streets['street_type'] = bcn_open_data_streets['NOM_CARRER'].apply(
            extract_up_to_second_capital_or_number)
        bcn_open_data_streets['street_name'] = bcn_open_data_streets.apply(
            lambda row: row['NOM_CARRER'][len(row['street_type']):].strip() if pd.notnull(row['street_type']) else None,
            axis=1
        )
        bcn_open_data_streets['street_type'] = bcn_open_data_streets['street_type'].str.upper()
        bcn_open_data_streets['street_name'] = bcn_open_data_streets['street_name'].str.upper()

        # Step 1: Ensure CRS matches
        if bcn_open_data_streets.crs != parcels_gdf.crs:
            bcn_open_data_streets = bcn_open_data_streets.to_crs(parcels_gdf.crs)

        # Step 2: Spatial join - points inside parcels
        bcn_open_data_streets = gpd.sjoin(
            bcn_open_data_streets, parcels_gdf[['building_reference', 'parcel_geometry']],
            how="left", predicate="within"
        )

        # Step 3: Find unmatched (not inside any parcel)
        unmatched = bcn_open_data_streets[bcn_open_data_streets['building_reference'].isna()].copy()

        if not unmatched.empty:
            # Create a spatial index for parcels
            parcels_gdf = parcels_gdf.set_geometry('parcel_geometry')
            parcel_sindex = parcels_gdf.sindex

            # Function to find closest parcel geometry
            def find_closest_parcel(point):
                # Ensure input is a Shapely Point
                nearest_idx = list(parcel_sindex.nearest(point, 1))
                closest_geom = parcels_gdf.iloc[nearest_idx[0]]
                return closest_geom['building_reference']

            # Apply to unmatched geometries
            unmatched['building_reference'] = unmatched['geometry'].apply(find_closest_parcel)

            # Step 4: Merge results back
            bcn_open_data_streets.update(unmatched)

        bcn_open_data_streets["location"] = bcn_open_data_streets.geometry.centroid
        bcn_open_data_streets["specification"] = "OpenDataBCN"
        bcn_open_data_streets["cadaster_code"] = "08900"
        bcn_open_data_streets["street_number"] = pd.to_numeric(bcn_open_data_streets["NUMPOST"],
                                                                     errors='coerce').astype(str)
        bcn_open_data_streets["street_number_clean"] = pd.to_numeric(bcn_open_data_streets["NUMPOST"],
                                                                     errors='coerce').astype('Int64')

        aggregated = (
            bcn_open_data_streets.groupby('geometry')['street_number_clean']
            .agg(['min', 'max'])
            .reset_index()
        )

        aggregated['street_number_clean_label'] = aggregated.apply(
            lambda row: str(int(row['min'])) if row['min'] == row['max']
            else f"{int(row['min'])}-{int(row['max'])}",
            axis=1
        )

        bcn_open_data_streets = bcn_open_data_streets.merge(aggregated[['geometry', 'street_number_clean_label']],
                                                            on='geometry', how='left')
        bcn_open_data_streets["street_number_odbcn_label"] = bcn_open_data_streets["ETIQUETA"]

        bcn_open_data_streets = bcn_open_data_streets[
            ['location', 'specification', 'cadaster_code', 'street_type', 'street_name', 'street_number_clean_label',
             'street_number_odbcn_label', 'street_number', 'street_number_clean', 'building_reference']]

        gdf = gdf[gdf["cadaster_code"]!="08900"]
        gdf = pd.concat([gdf, bcn_open_data_streets])

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

        if "building_gdf" in locals():
            if building_gdf_.crs != building_gdf.crs:
                building_gdf_ = building_gdf_.to_crs(building_gdf.crs)
            building_gdf = gpd.GeoDataFrame(pd.concat([building_gdf, building_gdf_], ignore_index=True))
        else:
            building_gdf = building_gdf_

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

            gdf_unique = gdf.drop_duplicates(subset='building_reference')
            building_part_gdf_ = building_part_gdf_.join(gdf_unique.set_index('building_reference'),
                                                       on="building_reference", how="left")
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

            if "building_part_gdf" in locals():
                building_part_gdf = pd.concat([building_part_gdf, building_part_gdf_[1]], ignore_index=True)
            else:
                building_part_gdf = building_part_gdf_[1]

            # Join Building geodataframe with
            building_gdf = (building_gdf[['gml_id', 'building_status', 'building_reference', 'building_use',
                                          'building_geometry','year_of_construction']].
                            merge(building_part_gdf, left_on="building_reference",
                                  right_on="building_reference", how="left"))

        elif buildings_CAT is not None:
            #['n_building_units', 'n_dwellings', 'n_floors_above_ground', 'building_area']

            building_gdf = building_gdf[['gml_id', 'building_status', 'building_reference', 'building_use',
                                         'building_geometry', 'year_of_construction']]

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
            area_pivot = (
                area_per_use
                .pivot(
                    values="area_by_use",
                    index="building_reference",
                    on="building_space_inferred_use_type"
                )
                .rename(use_type_mapping)  # rename columns (now all keys are valid strings)
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
            building_gdf = pd.merge(building_gdf, final_df, on="building_reference", how="left")

        else:
            building_gdf = building_gdf[['gml_id', 'building_status', 'building_reference', 'building_use',
                                         'building_geometry', 'year_of_construction', 'n_building_units', 'n_dwellings',
                                         'n_floors_above_ground', 'building_area']]

    return pd.merge(gdf, building_gdf, left_on="building_reference", right_on="building_reference", how="left")


def join_cadaster_zone(gdf, cadaster_dir, cadaster_codes):
    import time
    from sklearn.neighbors import NearestNeighbors
    
    start_time = time.time()
    sys.stderr.write(f"\nJoining cadastral zones for {len(cadaster_codes)} municipalities ({len(gdf)} buildings)\n")
    
    # Add cadaster_code column to track building-to-municipality mapping
    if 'cadaster_code' not in gdf.columns:
        # Extract cadaster code from building_reference (first 5 characters)
        gdf['cadaster_code'] = gdf['building_reference'].str[:5]
    
    joined_results = []
    processed_buildings = 0
    
    # Process spatial joins by individual cadastral codes for much better performance
    for code in tqdm(cadaster_codes, desc="Processing zones by municipality"):
        try:
            # Load zone file for this municipality
            zone_gdf = gpd.read_file(f"{cadaster_dir}/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralzoning.gml",
                                   layer="CadastralZoning")
            
            # Clean and prepare zone data
            zone_gdf.rename(columns={
                'LocalisedCharacterString': 'zone_type',
                'nationalCadastalZoningReference': 'zone_reference'
            }, inplace=True)
            zone_gdf.drop(['gml_id', 'estimatedAccuracy', 'estimatedAccuracy_uom', 'localId',
                          'namespace', "label", "beginLifespanVersion", "pos", "endLifespanVersion",
                          "originalMapScaleDenominator"], inplace=True, axis=1, errors='ignore')
            
            # Get buildings for this municipality only
            buildings_for_code = gdf[gdf['cadaster_code'] == code].copy()
            if len(buildings_for_code) == 0:
                continue
                
            processed_buildings += len(buildings_for_code)
            
            # Align CRS
            if buildings_for_code.crs != zone_gdf.crs:
                buildings_for_code = buildings_for_code.to_crs(zone_gdf.crs)
            
            # Process urban zones first
            urban_zones = zone_gdf[zone_gdf["zone_type"] == "MANZANA "].copy()
            if len(urban_zones) > 0:
                joined = gpd.sjoin(buildings_for_code, urban_zones, how="left", predicate="within")
                joined = joined.drop(["index_right"], axis=1)
                joined["zone_type"] = joined["zone_type"].fillna("MANZANA ")
            else:
                joined = buildings_for_code.copy()
                joined["zone_reference"] = np.nan
                joined["zone_type"] = "MANZANA "
            
            # For unassigned buildings, try rural zones
            unassigned_mask = joined["zone_reference"].isna()
            if unassigned_mask.any():
                rural_zones = zone_gdf[zone_gdf["zone_type"] == "POLIGONO "].copy()
                if len(rural_zones) > 0:
                    unassigned_buildings = joined[unassigned_mask].drop(['zone_reference', 'zone_type'], axis=1)
                    rural_joined = gpd.sjoin(unassigned_buildings, rural_zones, how="left", predicate="within")
                    rural_joined = rural_joined.drop(["index_right"], axis=1)
                    
                    # Update the main joined result with rural assignments
                    for idx in rural_joined.index:
                        if pd.notna(rural_joined.loc[idx, "zone_reference"]):
                            joined.loc[idx, "zone_reference"] = rural_joined.loc[idx, "zone_reference"]
                            joined.loc[idx, "zone_type"] = "POLIGONO "
            
            # Quick nearest neighbor assignment for remaining unassigned (only if really needed)
            still_unassigned = joined["zone_reference"].isna()
            if still_unassigned.any() and len(zone_gdf) > 0:
                unassigned_count = still_unassigned.sum()
                if unassigned_count < 1000:  # Only for small numbers to keep it fast
                    unassigned_buildings = joined[still_unassigned]
                    valid_locations = unassigned_buildings["location"].notna()
                    
                    if valid_locations.any():
                        points = unassigned_buildings[valid_locations]
                        coords = np.array([[geom.x, geom.y] for geom in points["location"]])
                        zone_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in zone_gdf["geometry"]])
                        
                        if len(coords) > 0 and len(zone_coords) > 0:
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(zone_coords)
                            _, indices = nbrs.kneighbors(coords)
                            
                            for i, idx in enumerate(points.index):
                                zone_idx = indices[i][0]
                                joined.loc[idx, "zone_reference"] = zone_gdf.iloc[zone_idx]["zone_reference"]
                                joined.loc[idx, "zone_type"] = zone_gdf.iloc[zone_idx]["zone_type"]
                else:
                    # For large numbers, just assign default
                    joined.loc[still_unassigned, "zone_reference"] = "unassigned"
                    joined.loc[still_unassigned, "zone_type"] = "MANZANA "
            
            joined_results.append(joined)
            
        except Exception as e:
            # Skip failed municipalities but continue processing
            continue
    
    # Combine all results
    if joined_results:
        final_joined = pd.concat(joined_results, ignore_index=True)
    else:
        final_joined = gdf.copy()
        final_joined["zone_reference"] = "unassigned"
        final_joined["zone_type"] = "MANZANA "
    
    # Clean up zone types
    final_joined["zone_type"] = final_joined["zone_type"].replace({
        "MANZANA ": "urban", 
        "POLIGONO ": "disseminated"
    })
    
    # Remove temporary cadaster_code column if we added it
    if 'cadaster_code' in gdf.columns and 'cadaster_code' not in gdf.columns:
        final_joined = final_joined.drop('cadaster_code', axis=1)
    
    total_time = time.time() - start_time
    unassigned_count = (final_joined["zone_reference"] == "unassigned").sum()
    sys.stderr.write(f"Zone joining completed in {total_time:.1f}s. Processed {processed_buildings} buildings, {unassigned_count} unassigned\n")
    
    return final_joined

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
    gdf_joined = gdf.merge(parcel_gdf, on="building_reference", how=how)
    parcel_gdf = parcel_gdf.set_geometry("parcel_geometry")
    parcel_gdf["parcel_centroid"] = parcel_gdf.centroid

    return (gdf_joined,parcel_gdf)

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

    # Zones
    gdf = join_cadaster_zone(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes)
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
