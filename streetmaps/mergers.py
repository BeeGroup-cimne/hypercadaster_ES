import geopandas as gpd
import pandas as pd
import rasterio
from shapely.wkt import loads


def make_valid(gdf):
    gdf.geometry = gdf.geometry.make_valid()
    return gdf


def get_address_street_name(inspire_dir):
    address_gdf = gpd.read_file(f"{inspire_dir}/address/unzip/A.ES.SDGC.AD.08900.gml", layer="Address")
    address_street_names_df = gpd.read_file(f"{inspire_dir}/address/unzip/A.ES.SDGC.AD.08900.gml",
                                            layer="ThoroughfareName")
    address_street_names_df = address_street_names_df[['gml_id', 'text']]
    address_street_names_df["gml_id"] = address_street_names_df['gml_id'].apply(lambda x: x.split('ES.SDGC.TN.')[1])
    address_gdf["gml_id"] = address_gdf['gml_id'].apply(lambda x: '.'.join(x.split('ES.SDGC.AD.')[1].split('.')[:3]))
    merge_df = pd.merge(address_gdf, address_street_names_df, left_on="gml_id", right_on="gml_id")
    merge_df.drop(["beginLifespanVersion", "validFrom", "level", "type", "method", "default"], inplace=True, axis=1)
    merge_df.rename(columns={'text': 'street_name', 'designator': 'street_number'}, inplace=True)
    return merge_df


def merge_building(address_with_street_names_gdf, inspire_dir):
    building_gdf = gpd.read_file(f"{inspire_dir}/buildings/unzip/A.ES.SDGC.BU.08900.building.gml", layer="Building")
    address_with_street_names_gdf["cat_ref"] = address_with_street_names_gdf['localId'].apply(
        lambda x: x.split('.')[-1])
    building_gdf.drop(
        ["horizontalGeometryReference", "referenceGeometry", "documentLink", "format", "sourceStatus", "geometry",
         "beginLifespanVersion", "beginning", "end", "endLifespanVersion", "informationSystem"], inplace=True, axis=1)
    return pd.merge(address_with_street_names_gdf, building_gdf, left_on="cat_ref", right_on="reference", how="left")


# def merge_parcel(address_building_gdf, inspire_dir):
#     parcel_gdf = gpd.read_file(f"{inspire_dir}/parcels/unzip/A.ES.SDGC.CP.08900.cadastralparcel.gml", layer="CadastralParcel")
#     parcel_gdf.drop(["endLifespanVersion", "beginLifespanVersion", "pos", "geometry"], inplace=True, axis=1)
#     return pd.merge(address_building_gdf, parcel_gdf, left_on="cat_ref", right_on="nationalCadastralReference", how="left")

def merge_zone(address_building_gdf, inspire_dir):
    zone_gdf = gpd.read_file(f"{inspire_dir}/parcels/unzip/A.ES.SDGC.CP.08900.cadastralzoning.gml",
                             layer="CadastralZoning")
    zone_gdf.drop(["label", "beginLifespanVersion", "pos", "geometry", "endLifespanVersion"], inplace=True, axis=1)
    zone_gdf.loc[zone_gdf["LocalisedCharacterString"] == "MANZANA ", "gml_id"] = zone_gdf['gml_id'].apply(
        lambda x: x.split('ES.SDGC.CP.Z.')[1])
    address_building_gdf['cadastral_zonning_reference'] = address_building_gdf['localId_x'].apply(
        lambda x: x.split('.')[-1][:5] + x.split('.')[-1][7:])
    return pd.merge(address_building_gdf, zone_gdf, left_on="cadastral_zonning_reference",
                    right_on="nationalCadastalZoningReference", how="left")


def merge_inspire_data(inspire_dir="data/inspire"):
    # Address
    address_with_street_names_gdf = get_address_street_name(inspire_dir)
    # Buildings
    address_building_gdf = merge_building(address_with_street_names_gdf, inspire_dir)
    # Parcels
    # address_building_parcel_gdf = merge_parcel(address_building_gdf, inspire_dir)
    # Zones
    return merge_zone(address_building_gdf, inspire_dir)


def merge_raster(raster_path="data/PNOA_MDT200_ETRS89_HU30_Espana.tif", address_building_zone_gdf=None):
    src = rasterio.open(raster_path)
    address_building_zone_gdf = address_building_zone_gdf.to_crs(epsg=25830)
    address_building_zone_gdf["height_above_sea_level"] = [x[0] for x in src.sample(
        [(x, y) for x, y in zip(address_building_zone_gdf["geometry"].x, address_building_zone_gdf["geometry"].y)])]
    return address_building_zone_gdf


def join_by_census_tracts(address_building_zone_height_gdf=None, columns=["CUSEC", "CUMUN", "CUDIS", "geometry"],
                          census_tract_dir="data/census_tracts"):
    census_tracts_gdf = gpd.read_file(f"{census_tract_dir}/validated_census_2022.gpkg")
    return gpd.sjoin(address_building_zone_height_gdf, census_tracts_gdf[columns],
                     how="left", op="within").drop(["index_right"], axis=1)


def join_by_neighbourhoods(address_building_zone_height_census_tracts_gdf=None,
                           neighbourhoods_path="data/neighbourhoods/neighbourhoods.csv",
                           columns=["codi_districte", "nom_districte", "codi_barri", "nom_barri", "geometry"]):
    neighbourhoods_df = gpd.read_file(neighbourhoods_path)
    neighbourhoods_df.drop(["geometria_wgs84", "geometry"], inplace=True, axis=1)
    neighbourhoods_df.rename(columns={'geometria_etrs89': 'geometry'}, inplace=True)
    # Convierte la columna WKT en objetos de geometría
    neighbourhoods_df['geometry'] = neighbourhoods_df['geometry'].apply(loads)

    neighbourhoods_gdf = gpd.GeoDataFrame(neighbourhoods_df, geometry='geometry', crs='EPSG:25831')
    neighbourhoods_gdf = neighbourhoods_gdf.to_crs(address_building_zone_height_census_tracts_gdf.crs)
    return gpd.sjoin(address_building_zone_height_census_tracts_gdf, neighbourhoods_gdf[columns], how="left",
                     op="within").drop(["index_right"], axis=1)


def join_by_postal_codes(address_building_zone_height_census_tracts_neighbourhoods_gdf=None,
                         postal_codes_path="data/postal_codes/postal_codes.geojson",
                         columns=["PROV", "CODPOS", "geometry"]):
    postal_codes_df = gpd.read_file(postal_codes_path)

    postal_codes_gdf = gpd.GeoDataFrame(postal_codes_df, geometry='geometry', crs='EPSG:4326')
    postal_codes_gdf = postal_codes_gdf.to_crs(address_building_zone_height_census_tracts_neighbourhoods_gdf.crs)
    return gpd.sjoin(address_building_zone_height_census_tracts_neighbourhoods_gdf, postal_codes_gdf[columns],
                     how="left", op="within").drop(["index_right"], axis=1)
