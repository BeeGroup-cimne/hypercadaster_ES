import geopandas as gpd
import pandas as pd
import rasterio
from shapely import wkt
from hypercadaster_ES import utils
import sys


def make_valid(gdf):
    gdf.geometry = gdf.geometry.make_valid()
    return gdf


def get_cadaster_address(cadaster_dir, cadaster_codes):
    sys.stderr.write(f"\nReading the cadaster addresses for {len(cadaster_codes)} municipalities\n")

    address_gdf = gpd.GeoDataFrame()
    address_street_names_df = gpd.GeoDataFrame()

    for code in cadaster_codes:
        sys.stderr.write("\r" + " " * 60)
        sys.stderr.flush()
        sys.stderr.write(f"\r\tCadaster code: {code}")
        sys.stderr.flush()

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
            if address_street_names_df_.crs != address_street_names_df.crs:
                address_street_names_df_ = address_street_names_df_.to_crs(address_street_names_df.crs)
            address_street_names_df = gpd.GeoDataFrame(
                pd.concat([address_street_names_df, address_street_names_df_], ignore_index=True))
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

    return gdf


def join_cadaster_building(gdf, cadaster_dir, cadaster_codes, results_dir, building_parts_plots=False, building_parts_inference=False):

    sys.stderr.write(f"\nJoining the buildings description for {len(cadaster_codes)} municipalities\n")

    for code in cadaster_codes:

        sys.stderr.write("\r" + " " * 60)
        sys.stderr.flush()
        sys.stderr.write(f"\r\tCadaster code: {code}")
        sys.stderr.flush()

        building_gdf_ = gpd.read_file(f"{cadaster_dir}/buildings/unzip/A.ES.SDGC.BU.{code}.building.gml", layer="Building")
        if "building_gdf" in locals():
            if building_gdf_.crs != building_gdf.crs:
                building_gdf_ = building_gdf_.to_crs(building_gdf.crs)
            building_gdf = gpd.GeoDataFrame(pd.concat([building_gdf, building_gdf_], ignore_index=True))
        else:
            building_gdf = building_gdf_

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

            building_part_gdf_ = utils.process_building_parts(building_part_gdf_, results_dir=results_dir, plots=building_parts_plots)

            if "building_part_gdf" in locals():
                building_part_gdf = pd.concat([building_part_gdf, building_part_gdf_], ignore_index=True)
            else:
                building_part_gdf = building_part_gdf_

        building_gdf.rename(columns={
            'geometry': 'building_geometry',
            'value': 'building_area',
            'conditionOfConstruction': 'building_status',
            'currentUse': 'building_use',
            'numberOfBuildingUnits': 'n_building_units',
            'numberOfDwellings': 'n_dwellings',
            'numberOfFloorsAboveGround': 'n_floors_above_ground',
            'numberOfFloorsBelowGround': 'n_floors_below_ground',
            'reference': 'building_reference'
        }, inplace=True)
        building_gdf.drop(
            ["localId", "namespace", "officialAreaReference", "value_uom", "horizontalGeometryEstimatedAccuracy",
             "horizontalGeometryEstimatedAccuracy_uom", "horizontalGeometryReference", "referenceGeometry",
             "documentLink",
             "format", "sourceStatus", "beginLifespanVersion", "beginning", "end", "endLifespanVersion",
             "informationSystem"],
            inplace=True, axis=1)
        building_gdf = building_gdf.merge(building_part_gdf, left_on="building_reference",
                                          right_on="building_reference", how="left")

    return pd.merge(gdf, building_gdf, left_on="building_reference", right_on="building_reference", how="left")


def join_cadaster_zone(gdf, cadaster_dir, cadaster_codes):

    sys.stderr.write(f"\nJoining the cadaster zones for {len(cadaster_codes)} municipalities\n")

    for code in cadaster_codes:

        sys.stderr.write("\r" + " " * 60)
        sys.stderr.flush()
        sys.stderr.write(f"\r\tCadaster code: {code}")
        sys.stderr.flush()

        zone_gdf_ = gpd.read_file(f"{cadaster_dir}/parcels/unzip/A.ES.SDGC.CP.{code}.cadastralzoning.gml",
                             layer="CadastralZoning")
        if "zone_gdf" in locals():
            if zone_gdf_.crs != zone_gdf.crs:
                zone_gdf_ = zone_gdf_.to_crs(zone_gdf.crs)
            zone_gdf = gpd.GeoDataFrame(pd.concat([zone_gdf, zone_gdf_], ignore_index=True))
        else:
            zone_gdf = zone_gdf_

    sys.stderr.write("\r" + " " * 60)
    zone_gdf.rename(columns={
        'LocalisedCharacterString': 'zone_type',
        'nationalCadastalZoningReference': 'zone_reference'
    }, inplace=True)
    zone_gdf.drop(['gml_id', 'estimatedAccuracy', 'estimatedAccuracy_uom', 'localId',
                   'namespace', "label", "beginLifespanVersion", "pos", "endLifespanVersion",
                   "originalMapScaleDenominator"], inplace=True, axis=1)
    gdf = gdf.set_geometry("location")

    # Urban zones
    zone_gdf_urban = zone_gdf.loc[zone_gdf["zone_type"] == "MANZANA "].copy()
    zone_gdf_urban["geometry_buffered"] = zone_gdf_urban["geometry"].buffer(3)
    zone_gdf_urban = zone_gdf_urban.set_geometry("geometry_buffered").reset_index()

    joined_urban = gpd.sjoin(gdf, zone_gdf_urban, how="left", predicate="within").drop(["index_right", "index"], axis=1)
    joined_urban['dist'] = joined_urban['geometry'].distance(joined_urban['location'])
    joined_urban = joined_urban.sort_values("dist", ascending=True).groupby(["location"]).first().reset_index()
    joined_urban = joined_urban.drop(["geometry", "dist"], axis=1)

    # Disseminated zones
    zone_gdf_rural = zone_gdf.loc[zone_gdf["zone_type"] == "POLIGONO "].copy()
    zone_gdf_rural["geometry_buffered"] = zone_gdf_rural["geometry"].buffer(3)
    zone_gdf_rural = zone_gdf_rural.set_geometry("geometry_buffered").reset_index()

    joined_rural = gpd.sjoin(
        joined_urban.loc[
            joined_urban["zone_reference"].isna(), ~joined_urban.columns.isin(["zone_type", "zone_reference"])],
        zone_gdf_rural, how="left", predicate="within").drop(["index_right", "index"], axis=1
                                                             )
    joined_rural['dist'] = joined_rural['geometry'].distance(joined_rural['location'])
    joined_rural = joined_rural.sort_values("dist", ascending=True).groupby(["location"]).first().reset_index()
    joined_rural = joined_rural.drop(["geometry", "dist"], axis=1)

    joined_rural["zone_type"] = "disseminated"
    joined_rural["zone_reference"] = joined_rural["zone_reference"].fillna("disseminated")

    # Join urban and rural
    joined = pd.concat([joined_urban.loc[~joined_urban["zone_reference"].isna(), :],
                        joined_rural]).reset_index()
    joined["zone_type"] = joined["zone_type"].replace({"MANZANA ": "urban",  "POLIGONO ": "disseminated"})

    return joined


def join_adm_div_naming(gdf, cadaster_dir, cadaster_codes):

    return pd.merge(gdf, utils.get_administrative_divisions_naming(cadaster_dir, cadaster_codes=cadaster_codes),
                    left_on="cadaster_code", right_on="cadaster_code", how="left")


def join_cadaster_data(cadaster_dir, cadaster_codes, results_dir, building_parts_plots=False, building_parts_inference=False):

    # Address
    gdf = get_cadaster_address(cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes)
    # Zones
    gdf = join_cadaster_zone(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes)
    # Buildings
    gdf = join_cadaster_building(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes,
                                 results_dir=results_dir, building_parts_plots=building_parts_plots,
                                 building_parts_inference=building_parts_inference)
    # Administrative layers naming
    gdf = join_adm_div_naming(gdf=gdf, cadaster_dir=cadaster_dir, cadaster_codes=cadaster_codes)

    return gdf


def join_DEM_raster(gdf, raster_dir):

    sys.stderr.write(f"\nJoining the Digital Elevation Model information\n")

    gdf = gdf.set_geometry("location")

    with rasterio.open(f"{raster_dir}/DEM.tif", 'r+') as rds:
        ini_crs = gdf.crs
        gdf = gdf.to_crs(epsg=4326)
        gdf["elevation"] = [x[0] for x in rds.sample(
            [(x, y) for x, y in zip(gdf["location"].x, gdf["location"].y)])]
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
    gdf = gdf.set_geometry("location")
    census_gdf = census_gdf.to_crs(gdf.crs)
    census_gdf = gpd.sjoin(gdf, census_gdf, how="left", predicate="within").drop(["index_right"], axis=1)

    return census_gdf


def join_by_neighbourhoods(gdf, neighbourhoods_dir, columns=None, geometry_column="neighborhood_geometry"):

    if columns is None:
        columns = {
            "codi_barri": "neighborhood_code",
            "nom_barri": "neighborhood_name",
            "geometria_etrs89": "neighborhood_geometry"
        }
    sys.stderr.write(f"\nJoining the neighborhoods description\n")

    neighbourhoods_gdf = gpd.read_file(f"{neighbourhoods_dir}/neighbourhoods.csv")
    neighbourhoods_gdf.rename(columns = columns, inplace = True)
    neighbourhoods_gdf = neighbourhoods_gdf[columns.values()]
    neighbourhoods_gdf[geometry_column] = neighbourhoods_gdf[geometry_column].apply(wkt.loads)
    neighbourhoods_gdf = gpd.GeoDataFrame(neighbourhoods_gdf, geometry=geometry_column, crs='EPSG:25831')
    gdf = gdf.set_geometry("location")
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
    gdf = gdf.set_geometry("location")
    postal_codes_gdf = postal_codes_gdf.to_crs(gdf.crs)
    postal_codes_gdf = gpd.sjoin(gdf, postal_codes_gdf,
              how="left", predicate="within").drop(["index_right"], axis=1)

    return postal_codes_gdf
