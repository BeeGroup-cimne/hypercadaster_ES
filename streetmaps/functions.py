# from streetmaps.functions import downloaders, utils
import pandas as pd
from streetmaps import mergers
from streetmaps import utils
from streetmaps import downloaders

def download(wd, ine_codes):

    province_codes = list(set([code[:2] for code in ine_codes]))
    utils.create_dir(data_dir=wd)
    downloaders.download_file(wd=f"{wd}/districts", url="https://opendata-ajuntament.barcelona.cat/data/dataset/808daafa-d9ce-48c0-925a-fa5afdb1ed41/resource/576bc645-9481-4bc4-b8bf-f5972c20df3f/download", file="districts.csv")
    downloaders.download_file(wd=f"{wd}/neighbourhoods", url="https://opendata-ajuntament.barcelona.cat/data/dataset/808daafa-d9ce-48c0-925a-fa5afdb1ed41/resource/b21fa550-56ea-4f4c-9adc-b8009381896e/download", file="neighbourhoods.csv")
    downloaders.download_postal_code(wd=f"{wd}/postal_codes")
    downloaders.download_census(wd=f"{wd}/census_tracts", year="2022")
    downloaders.download_regions(wd=f"{wd}/regions")
    downloaders.download_file(wd=f"{wd}/inspire",
                              url="https://www.catastro.minhap.es/regularizacion/Regularizacion_municipios_finalizados.xlsx",
                              file="ine_inspire_codes.xlsx")
    downloaders.inspire_downloader(province_codes=province_codes, INE_codes=ine_codes, inspire_dir=f"{wd}/inspire")


def merge(wd, ine_codes):
    map_code_dict = pd.read_excel(f"{wd}/inspire/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl').set_index(
        'CÓDIGO INE').to_dict()['CÓDIGO CATASTRO']
    inspire_codes = [map_code_dict[key] for key in ine_codes]

    address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf = mergers.join_by_postal_codes(
        address_building_zone_height_census_tracts_neighbourhoods_gdf=mergers.join_by_neighbourhoods(
            address_building_zone_height_census_tracts_gdf=mergers.join_by_census_tracts(
                address_building_zone_height_gdf=mergers.merge_inspire_data(
                        inspire_dir=f"{wd}/inspire", code=inspire_codes[0]),
                columns=["CUSEC", "CUMUN", "CUDIS", "geometry"],
                census_tract_dir=f"{wd}/census_tracts"),
            neighbourhoods_path=f"{wd}/neighbourhoods/neighbourhoods.csv",
            columns=["codi_districte", "nom_districte", "codi_barri", "nom_barri", "geometry"]),
        postal_codes_path=f"{wd}/postal_codes/postal_codes.geojson",
        columns=["PROV", "CODPOS", "geometry"])

    address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf["street_type"] = address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf["street_name"].apply(lambda x: x.split(" ")[0])
    address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf["street_name"] = address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf["street_name"].apply(lambda x: ' '.join(x.split(" ")[1:]))
    return address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf.rename(columns={"CUSEC":"MUNDISSEC", "nationalCadastalZoningReference": "national_cadastal_zoning_reference"})

    '''address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf = mergers.join_by_postal_codes(
        address_building_zone_height_census_tracts_neighbourhoods_gdf=mergers.join_by_neighbourhoods(
            address_building_zone_height_census_tracts_gdf=mergers.join_by_census_tracts(
                address_building_zone_height_gdf=mergers.merge_raster(
                    raster_path=f"{wd}/PNOA_MDT200_ETRS89_HU30_Espana.tif",
                    address_building_zone_gdf=mergers.merge_inspire_data(
                        inspire_dir=f"{wd}/inspire")),
                columns=["CUSEC", "CUMUN", "CUDIS", "geometry"],
                census_tract_dir=f"{wd}/census_tracts"),
            neighbourhoods_path=f"{wd}/neighbourhoods/neighbourhoods.csv",
            columns=["codi_districte", "nom_districte", "codi_barri", "nom_barri", "geometry"]),
        postal_codes_path=f"{wd}/postal_codes/postal_codes.geojson",
        columns=["PROV", "CODPOS", "geometry"])
    return address_building_zone_height_census_tracts_neighbourhoods_postal_codes_gdf'''

