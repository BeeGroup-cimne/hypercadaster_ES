import sys
import zipfile
import geopandas as gpd
import json
import os
import requests
import pandas as pd
import fiona
from fastkml import kml
from hypercadaster_ES import utils

def download_file(dir, url, file):
    if not os.path.exists(f"{dir}/{file}"):
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{dir}/{file}", 'wb') as archivo:
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


def download_postal_codes(postal_codes_dir, province_codes=None):
    sys.stderr.write('\nDownloading postal codes from Spain\n')
    if province_codes is None:
        province_codes = ["{:02d}".format(i) for i in range(1, 53)]
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    for province_code in province_codes:
        if not os.path.exists(f"{postal_codes_dir}/raw/k{province_code}.geojson"):
            response = requests.get(f"https://www.codigospostales.com/kml/k{province_code}.kml")
            if response.status_code == 200:
                geojson_data = kml_to_geojson(response.content)
                with open(f"{postal_codes_dir}/raw/k{province_code}.geojson", 'w') as f:
                    f.write(geojson_data)
            else:
                print(f"Error downloading postal code {province_code} file\n", response.status_code)

    # Filtrar la lista para obtener solo los archivos que terminan en ".geojson"
    patterns = [f'k{pr}.geojson' for pr in province_codes]
    geojson_filenames = [gpd.read_file(f"{postal_codes_dir}/raw/{file}") for file in
                        [filename for filename in os.listdir(f"{postal_codes_dir}/raw") if
                         any(filename.endswith(pattern) for pattern in patterns)]]
    concatenated_gdf = gpd.GeoDataFrame(pd.concat(geojson_filenames, ignore_index=True), crs=geojson_filenames[0].crs)
    concatenated_gdf.geometry = concatenated_gdf.geometry.make_valid()
    concatenated_gdf.to_file(f"{postal_codes_dir}/postal_codes.geojson", driver="GeoJSON")


def download_census_tracts(census_tracts_dir, year):
    sys.stderr.write(f"\nDownloading census tract geometries for year: {year}\n")
    if not os.path.exists(f"{census_tracts_dir}/validated_census_{year}.gpkg"):
        if not f"España_Seccionado{year}_ETRS89H30" in os.listdir(census_tracts_dir):
            os.makedirs(f"{census_tracts_dir}/zip", exist_ok=True)
            response = requests.get(f"https://www.ine.es/prodyser/cartografia/seccionado_{year}.zip")
            if response.status_code == 200:
                with open(f"{census_tracts_dir}/zip/year.zip", 'wb') as archivo:
                    archivo.write(response.content)

            with zipfile.ZipFile(f"{census_tracts_dir}/zip/year.zip", "r") as zip_ref:
                zip_ref.extractall(census_tracts_dir)

        shp = gpd.read_file([os.path.join(census_tracts_dir, f"España_Seccionado{year}_ETRS89H30", archivo) for archivo in
                             os.listdir(f"{census_tracts_dir}/España_Seccionado{year}_ETRS89H30") if archivo.endswith(".shp")][0])
        shp.geometry = shp.geometry.make_valid()
        shp.to_file(f"{census_tracts_dir}/validated_census_{year}.gpkg", driver="GPKG")


def cadaster_downloader(cadaster_dir, cadaster_codes=None):

    inspire_dict = {
        "parcels": "https://www.catastro.hacienda.gob.es/INSPIRE/CadastralParcels/ES.SDGC.CP.atom.xml",
        "address": "https://www.catastro.hacienda.gob.es/INSPIRE/Addresses/ES.SDGC.AD.atom.xml",
        "buildings": "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml"
    }

    for k, v in inspire_dict.items():
        sys.stderr.write(f"\nDownloading INSPIRE-harmonised cadaster data: {k}\n")
        municipalities = utils.list_municipalities(
            province_codes=list(set([code[:2] for code in cadaster_codes])),
            inspire_url=v, echo=False)
        if cadaster_codes is not None:
            municipalities = [item for item in municipalities if item['name'].split("-")[0] in cadaster_codes]
        nf = False
        for municipality in municipalities:
            if (not os.path.exists(f"{cadaster_dir}/{k}/zip/{municipality['url'].split('.')[-2]}.zip") and
                    municipality['url'].split('.')[-1] == 'zip'):
                nf = True
                sys.stderr.write("\r" + " " * 60)
                sys.stderr.flush()
                sys.stderr.write(f"\r\t{municipality['name']}")
                sys.stderr.flush()
                download(municipality['url'], f"{municipality['url'].split('.')[-2]}.zip",
                         f"{cadaster_dir}/{k}/zip/")
        sys.stderr.write("\r" + " " * 60)
        if nf:
            utils.unzip_directory(f"{cadaster_dir}/{k}/zip/", f"{cadaster_dir}/{k}/unzip/")


def download_DEM_raster(raster_dir, bbox, year=2023):
    sys.stderr.write(f"\nDownloading Digital Elevation Models for year: {year}\n")
    os.makedirs(f"{raster_dir}/raw", exist_ok=True)
    os.makedirs(f"{raster_dir}/uncompressed", exist_ok=True)
    bbox = [int(i) for i in bbox]
    nf = False
    for latitude in range(bbox[1],bbox[3]+1):
        for longitude in range(bbox[0],bbox[2]+1):
            if not os.path.exists(f"{raster_dir}/raw/DEM_{latitude}_{longitude}_{year}.tar"):
                nf = True
                sys.stderr.write(f"\t--> Latitude {latitude}, longitude {longitude}\n")
                download_file(dir = f"{raster_dir}/raw",
                              url = f"https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload/"
                                    f"COP-DEM_GLO-30-DGED__{year}_1/"
                                    f"Copernicus_DSM_10_N{latitude:02}_00_E{longitude:03}_00.tar",
                              file = f"DEM_{latitude}_{longitude}_{year}.tar")
    if nf:
        utils.untar_directory(tar_directory = f"{raster_dir}/raw/",
                        untar_directory = f"{raster_dir}/uncompressed/",
                        files_to_extract= "*/DEM/*.tif")
        utils.concatenate_tiffs(input_dir=f"{raster_dir}/uncompressed/",
                                output_file=f"{raster_dir}/DEM.tif")


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

