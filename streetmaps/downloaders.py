import sys
import zipfile
import geopandas as gpd
import json
import os
import requests
import pandas as pd
import fiona
from fastkml import kml
from utils import unzip_directory, list_municipalities


def download_file(wd, url, file):
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{wd}/{file}", 'wb') as archivo:
            archivo.write(response.content)


def kml_to_geojson(kml_file):
    features = []
    k = kml.KML()
    k.from_string(kml_file)
    for doc in k.features():
        for feature in doc.features():
            for placemark in feature.features():
                properties = {}
                if hasattr(placemark, 'extended_data') and placemark.extended_data:
                    for data in placemark.extended_data.elements:
                        if isinstance(data, kml.SchemaData):
                            for simple_data in data.data:
                                properties[simple_data['name']] = simple_data['value']
                coordinates = []
                if hasattr(placemark, 'geometry') and placemark.geometry:
                    if placemark.geometry._type in ['MultiPolygon']:

                        for polygon in placemark.geometry.geoms:
                            outer_boundary = polygon.exterior.coords
                            coordinates.append([[list(point)[:2] for point in outer_boundary]])

                    elif placemark.geometry._type in ['Polygon']:
                        outer_boundary = placemark.geometry.exterior.coords
                        coordinates.append([list(point)[:2] for point in outer_boundary])

                feature_json = {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": placemark.geometry.geom_type if coordinates else None,
                        "coordinates": coordinates
                    }
                }
                features.append(feature_json)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }
    return json.dumps(feature_collection)


def download_postal_code(wd):
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    sys.stdout.write('\nDownloading postal codes from Spain\n')
    for province_code in ["{:02d}".format(i) for i in range(1, 53)]:
        response = requests.get(f"https://www.codigospostales.com/kml/k{province_code}.kml")
        if response.status_code == 200:
            geojson_data = kml_to_geojson(response.content)
            with open(f"{wd}/raw/k{province_code}.geojson", 'w') as f:
                f.write(geojson_data)
        else:
            print(f"Error downloading postal code {province_code} file\n", response.status_code)

    # Filtrar la lista para obtener solo los archivos que terminan en ".geojson"
    archivos_geojson = [gpd.read_file(f"{wd}/raw/{file}") for file in
                        [archivo for archivo in os.listdir(f"{wd}/raw") if archivo.endswith(".geojson")]]
    concatenated_gdf = gpd.GeoDataFrame(pd.concat(archivos_geojson, ignore_index=True), crs=archivos_geojson[0].crs)
    concatenated_gdf.geometry = concatenated_gdf.geometry.make_valid()
    concatenated_gdf.to_file(f"{wd}/postal_codes.geojson", driver="GeoJSON")


def download_census(wd, year):
    if not f"España_Seccionado{year}_ETRS89H30" in os.listdir(wd):
        sys.stdout.write(f"\nDownloading census tract geometries for year: {year}\n")
        os.makedirs(f"{wd}/zip", exist_ok=True)
        response = requests.get(f"https://www.ine.es/prodyser/cartografia/seccionado_{year}.zip")
        if response.status_code == 200:
            with open(f"{wd}/zip/year.zip", 'wb') as archivo:
                archivo.write(response.content)

        with zipfile.ZipFile(f"{wd}/zip/year.zip", "r") as zip_ref:
            zip_ref.extractall(wd)

    shp = gpd.read_file([os.path.join(wd, f"España_Seccionado{year}_ETRS89H30", archivo) for archivo in
                         os.listdir(f"{wd}/España_Seccionado{year}_ETRS89H30") if archivo.endswith(".shp")][0])
    shp.geometry = shp.geometry.make_valid()
    shp.to_file(f"{wd}/validated_census_{year}.gpkg", driver="GPKG")


def inspire_downloader(province_codes=None, INE_codes=[], inspire_dir="data/inspire"):
    inspire_dict = {
        "parcels": "https://www.catastro.minhap.es/INSPIRE/CadastralParcels/ES.SDGC.CP.atom.xml",
        "address": "https://www.catastro.minhap.es/INSPIRE/Addresses/ES.SDGC.AD.atom.xml",
        "buildings": "https://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml"
    }
    map_code_dict = pd.read_excel(f"{inspire_dir}/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl').set_index(
        'CÓDIGO INE').to_dict()['CÓDIGO CATASTRO']
    inspire_codes = [map_code_dict[key] for key in INE_codes]

    for k, v in inspire_dict.items():

        municipalities_list = list_municipalities(province_codes=province_codes,
                                                  inspire_url=v)
        for municipality in municipalities_list:
            if municipality['name'].split("-")[0] in inspire_codes:
                if len(municipality['url'].split('.')) == 9 and municipality['url'].split('.')[8] == 'zip':
                    download(municipality['url'], f"{municipality['url'].split('.')[7]}.zip",
                             f"{inspire_dir}/{k}/zip/")
        unzip_directory(f"{inspire_dir}/{k}/zip/", f"{inspire_dir}/{k}/unzip/")


def download(url, name, save_path):
    get_response = requests.get(url, stream=True)
    if get_response:
        file_name = os.path.join(save_path, name)
        with open(file_name, 'wb') as f:
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print(get_response)

###
# Por cada dirección única calle-número, añadir el código postal, parcela cadastral, sección censal, barrio, distrito, municipio, número de viviendas, isla cadastral
###
