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
import polars as pl
from charset_normalizer import from_path
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

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


def load_and_transform_barcelona_ground_premises(open_data_layers_dir):
    ground_premises = pd.read_csv(
        filepath_or_buffer=f"{open_data_layers_dir}/barcelona_ground_premises.csv",
        encoding=from_path(f"{open_data_layers_dir}/barcelona_ground_premises.csv").best().encoding,
        on_bad_lines='skip',
        sep=",", low_memory=False)
    ground_premises["Nom_Activitat"] = ground_premises["Nom_Activitat"].replace(
        {
            'Activitats emmagatzematge': "Activitats d'emmagatzematge@ca\tActividades de almacenamiento@es\tStorage activities@en",
            'Ensenyament': "Ensenyament@ca\tEnseñanza@es\tEducation@en",
            'Serveis a les empreses i oficines': "Serveis a les empreses i oficines@ca\tServicios a empresas y oficinas@es\tBusiness and office services@en",
            'Arts gràfiques': "Arts gràfiques@ca\tArtes gráficas@es\tGraphic arts@en",
            'Activitats de la construcció': "Activitats de la construcció@ca\tActividades de construcción@es\tConstruction activities@en",
            'Reparacions (Electrodomèstics i automòbils)': "Reparacions (Electrodomèstics i automòbils)@ca\tReparaciones (Electrodomésticos y automóviles)@es\tRepairs (Appliances and automobiles)@en",
            'Sanitat i assistència': "Sanitat i assistència@ca\tSanidad y asistencia@es\tHealthcare and assistance@en",
            'Maquinària': "Maquinària@ca\tMaquinaria@es\tMachinery@en",
            'Associacions': "Associacions@ca\tAsociaciones@es\tAssociations@en",
            'Locals buits en venda i lloguer – reanalisi': "Locals buits en venda i lloguer – reanalisi@ca\tLocales vacíos en venta y alquiler – reanálisis@es\tEmpty premises for sale and rent – reanalysis@en",
            'Vehicles': "Vehicles@ca\tVehículos@es\tVehicles@en",
            'Vestir': "Vestir@ca\tVestimenta@es\tClothing@en",
            'Restaurants': "Restaurants@ca\tRestaurantes@es\tRestaurants@en",
            'Pàrquings i garatges': "Pàrquings i garatges@ca\tAparcamientos y garajes@es\tParking and garages@en",
            'Locutoris': "Locutoris@ca\tLocutorios@es\tCall centers@en",
            'Autoservei / Supermercat': "Autoservei / Supermercat@ca\tAutoservicio / Supermercado@es\tSelf-service / Supermarket@en",
            'Altres': "Altres@ca\tOtros@es\tOthers@en",
            'Activitats industrials': "Activitats industrials@ca\tActividades industriales@es\tIndustrial activities@en",
            'Locals buits en venda': "Locals buits en venda@ca\tLocales vacíos en venta@es\tEmpty premises for sale@en",
            'Bars / CIBERCAFÈ': "Bars / CIBERCAFÈ@ca\tBares / CIBERCAFÉ@es\tBars / CYBERCAFÉ@en",
            'Farmàcies PARAFARMÀCIA': "Farmàcies PARAFARMÀCIA@ca\tFarmacias PARAFARMACIA@es\tPharmacies and para-pharmacies@en",
            'Arranjaments': "Arranjaments@ca\tArreglos@es\tAlterations@en",
            'Equipaments culturals i recreatius': "Equipaments culturals i recreatius@ca\tEquipamientos culturales y recreativos@es\tCultural and recreational facilities@en",
            "Centres d'estètica": "Centres d'estètica@ca\tCentros de estética@es\tAesthetic centers@en",
            'Serveis Socials': "Serveis Socials@ca\tServicios Sociales@es\tSocial services@en",
            'Fruites i verdures': "Fruites i verdures@ca\tFrutas y verduras@es\tFruits and vegetables@en",
            'Joieria, rellotgeria i bijuteria': "Joieria, rellotgeria i bijuteria@ca\tJoyería, relojería y bisutería@es\tJewelry, watches, and costume jewelry@en",
            'Perruqueries': "Perruqueries@ca\tPeluquerías@es\tHairdressing@en",
            'Drogueria i perfumeria': "Drogueria i perfumeria@ca\tDroguería y perfumería@es\tDrugstore and perfumery@en",
            'Material equipament llar': "Material equipament llar@ca\tMaterial de equipamiento del hogar@es\tHome equipment materials@en",
            'Basars': "Basars@ca\tBazares@es\tBazaars@en",
            'Pa, pastisseria i làctics': "Pa, pastisseria i làctics@ca\tPan, pastelería y lácteos@es\tBread, pastries, and dairy@en",
            'Activitats de transport': "Activitats de transport@ca\tActividades de transporte@es\tTransport activities@en",
            'Mobles i articles fusta i metall': "Mobles i articles fusta i metall@ca\tMuebles y artículos de madera y metal@es\tFurniture and wood/metal goods@en",
            'Serveis de telecomunicacions': "Serveis de telecomunicacions@ca\tServicios de telecomunicaciones@es\tTelecommunication services@en",
            'Plats preparats (no degustació)': "Plats preparats (no degustació)@ca\tPlatos preparados (sin degustación)@es\tPrepared dishes (no tasting)@en",
            'Bars especials amb actuació / Bars musicals / Discoteques /PUB': "Bars especials amb actuació / Bars musicals / Discoteques /PUB@ca\tBares especiales con actuación / Bares musicales / Discotecas / PUB@es\tSpecial bars with live performance / Music bars / Nightclubs / PUB@en",
            'Parament ferreteria': "Parament ferreteria@ca\tSuministros de ferretería@es\tHardware supplies@en",
            'Serveis de menjar take away MENJAR RÀPID': "Serveis de menjar take away MENJAR RÀPID@ca\tServicios de comida para llevar / Comida rápida@es\tTake-away / Fast food services@en",
            'Locals buits en lloguer': "Locals buits en lloguer@ca\tLocales vacíos en alquiler@es\tEmpty premises for rent@en",
            'Tintoreries': "Tintoreries@ca\tTintorerías@es\tDry cleaners@en",
            "serveis d'allotjament": "serveis d'allotjament@ca\tservicios de alojamiento@es\tAccommodation services@en",
            'Altres equipaments esportius': "Altres equipaments esportius@ca\tOtros equipamientos deportivos@es\tOther sports facilities@en",
            'Carn i Porc': "Carn i Porc@ca\tCarne y cerdo@es\tMeat and pork@en",
            'Begudes': "Begudes@ca\tBebidas@es\tBeverages@en",
            'Herbolaris, dietètica i NUTRICIÓ': "Herbolaris, dietètica i NUTRICIÓ@ca\tHerbolarios, dietética y NUTRICIÓN@es\tHerbalists, dietetics, and NUTRITION@en",
            'Informàtica': "Informàtica@ca\tInformática@es\tComputing@en",
            'Aparells domèstics': "Aparells domèstics@ca\tAparatos domésticos@es\tHousehold appliances@en",
            'Veterinaris / Mascotes': "Veterinaris / Mascotes@ca\tVeterinarios / Mascotas@es\tVeterinarians / Pets@en",
            'Música': "Música@ca\tMúsica@es\tMusic@en",
            'Finances i assegurances': "Finances i assegurances@ca\tFinanzas y seguros@es\tFinance and insurance@en",
            'Activitats immobiliàries': "Activitats immobiliàries@ca\tActividades inmobiliarias@es\tReal estate activities@en",
            'Equipaments religiosos': "Equipaments religiosos@ca\tEquipamientos religiosos@es\tReligious facilities@en",
            'Joguines i esports': "Joguines i esports@ca\tJuguetes y deportes@es\tToys and sports@en",
            'Manteniment, neteja i similars': "Manteniment, neteja i similars@ca\tMantenimiento, limpieza y similares@es\tMaintenance, cleaning, and similar@en",
            'Administració': "Administració@ca\tAdministración@es\tAdministration@en",
            'Fotografia': "Fotografia@ca\tFotografía@es\tPhotography@en",
            'Gimnàs /fitnes': "Gimnàs /fitnes@ca\tGimnasio / fitness@es\tGym / Fitness@en",
            'Locals buits en venda i lloguer': "Locals buits en venda i lloguer@ca\tLocales vacíos en venta y alquiler@es\tEmpty premises for sale and rent@en",
            'Combustibles i carburants': "Combustibles i carburants@ca\tCombustibles y carburantes@es\tFuels and combustibles@en",
            'Fabricació tèxtil': "Fabricació tèxtil@ca\tFabricación textil@es\tTextile manufacturing@en",
            'Tabac i articles fumadors': "Tabac i articles fumadors@ca\tTabaco y artículos de fumadores@es\tTobacco and smoking articles@en",
            'Merceria': "Merceria@ca\tMercería@es\tHaberdashery@en",
            'Floristeries': "Floristeries@ca\tFloristerías@es\tFlorists@en",
            'Llibres, diaris i revistes': "Llibres, diaris i revistes@ca\tLibros, diarios y revistas@es\tBooks, newspapers, and magazines@en",
            'Òptiques': "Òptiques@ca\tÓpticas@es\tOptics@en",
            'Ous i aus': "Ous i aus@ca\tHuevos y aves@es\tEggs and poultry@en",
            'Agències de viatge': "Agències de viatge@ca\tAgencias de viaje@es\tTravel agencies@en",
            'Souvenirs': "Souvenirs@ca\tSouvenirs@es\tSouvenirs@en",
            'Calçat i pell': "Calçat i pell@ca\tCalzado y piel@es\tFootwear and leather@en",
            'Xocolateries / Geladeries / Degustació': "Xocolateries / Geladeries / Degustació@ca\tChocolaterías / Heladerías / Degustación@es\tChocolate shops / Ice cream parlors / Tasting@en",
            'Segells, monedes i antiguitats': "Segells, monedes i antiguitats@ca\tSellos, monedas y antigüedades@es\tStamps, coins, and antiques@en",
            'Peix i marisc': "Peix i marisc@ca\tPescado y marisco@es\tFish and seafood@en",
            'serveis de menjar i begudes': "serveis de menjar i begudes@ca\tservicios de comida y bebidas@es\tFood and beverage services@en",
            'Altres ( per exemple VENDING)': "Altres ( per exemple VENDING)@ca\tOtros (por ejemplo, VENDING)@es\tOthers (e.g., VENDING)@en",
            'altres': "altres@ca\totros@es\tothers@en",
            'Resta alimentació': "Resta alimentació@ca\tResto alimentación@es\tOther food products@en",
            'Souvenirs i basars': "Souvenirs i basars@ca\tSouvenirs y bazares@es\tSouvenirs and bazaars@en",
            'Grans magatzems i hipermercats': "Grans magatzems i hipermercats@ca\tGrandes almacenes e hipermercados@es\tDepartment stores and hypermarkets@en"
        }
    )
    ground_premises = ground_premises.groupby("Referencia_Cadastral").agg(
        cases=('Referencia_Cadastral', 'size'),
        ground_premises_activities=('Nom_Activitat', lambda x: list(x)),
        ground_premises_names=('Nom_Local', lambda x: list(x)),
        ground_premises_last_revision=('Data_Revisio', 'max')
    ).reset_index().rename({"Referencia_Cadastral": "building_reference",
                            "cases": "number_of_ground_premises",
                            "last_revision": "last_revision_ground_premises"}, axis=1)
    ground_premises["building_reference"] = ground_premises["building_reference"].astype(str)

    return ground_premises

# Estructura de los ficheros .CAT
# Transposicion literal de la especificacion (salvo error u omision):
# https://www.catastro.hacienda.gob.es/documentos/formatos_intercambio/catastro_fin_cat_2006.pdf
catstruct = {}
catstruct[1] = [
    [3, 1, 'X', 'tipo_entidad_generadora',pl.Utf8],
    [4, 9, 'N', 'codigo_entidad_generadora',pl.Utf8],
    [13, 27, 'X', 'nombre_entidad_generadora',pl.Utf8],
    [40, 8, 'N', 'fecha_generacion_fichero',pl.Utf8],
    [48, 6, 'N', 'hora_generacion_fichero',pl.Utf8],
    [54, 4, 'X', 'tipo_fichero',pl.Utf8],
    [58, 39, 'X', 'descripcion_contenido_fichero',pl.Utf8],
    [97, 21, 'X', 'nombre_fichero',pl.Utf8],
    [118, 3, 'N', 'codigo_entidad_destinataria',pl.Utf8],
    [121, 8, 'N', 'fecha_inicio_periodo',pl.Utf8],
    [129, 8, 'N', 'fecha_finalizacion_periodo',pl.Utf8]
]

# 11 - Registro de Finca
catstruct[11] = [
    [24, 2, 'N', 'codigo_delegacion_meh',pl.Utf8],
    [26, 3, 'N', 'codigo_municipio_dgc',pl.Utf8],
    [31, 14, 'X', 'parcela_catastral',pl.Utf8],
    [51, 2, 'N', 'codigo_provincia_ine',pl.Utf8],
    [53, 25, 'X', 'nombre_provincia',pl.Utf8],
    [78, 3, 'N', 'codigo_municipio_dgc_2',pl.Utf8],
    [81, 3, 'N', 'codigo_municipio_ine',pl.Utf8],
    [84, 40, 'X', 'nombre_municipio',pl.Utf8],
    [124, 30, 'X', 'nombre_entidad_menor',pl.Utf8],
    [154, 5, 'N', 'codigo_via_publica_dgc',pl.Utf8],
    [159, 5, 'X', 'tipo_via',pl.Utf8],
    [164, 25, 'X', 'nombre_via',pl.Utf8],
    [189, 4, 'N', 'primer_numero_policia',pl.Utf8],
    [193, 1, 'X', 'primera_letra',pl.Utf8],
    [194, 4, 'N', 'segundo_numero_policia',pl.Utf8],
    [198, 1, 'X', 'segunda_letra',pl.Utf8],
    [199, 5, 'N', 'kilometro_por_cien',pl.Utf8],
    [204, 4, 'X', 'bloque',pl.Utf8],
    [216, 25, 'X', 'direccion_no_estructurada',pl.Utf8],
    [241, 5, 'N', 'codigo_postal',pl.Utf8],
    [246, 2, 'X', 'distrito_municipal',pl.Utf8],
    [248, 3, 'N', 'codigo_municipio_origen_caso_agregacion_dgc',pl.Utf8],
    [251, 2, 'N', 'codigo_zona_concentracion',pl.Utf8],
    [253, 3, 'N', 'codigo_poligono',pl.Utf8],
    [256, 5, 'N', 'codigo_parcela',pl.Utf8],
    [261, 5, 'X', 'codigo_paraje_dgc',pl.Utf8],
    [266, 30, 'X', 'nombre_paraje',pl.Utf8],
    [296, 10, 'N', 'superficie_finca_o_parcela_catastral_m2',pl.Float32],
    [306, 7, 'N', 'superficie_construida_total',pl.Float32],
    [313, 7, 'N', 'superficie_construida_sobre_rasante',pl.Float32],
    [320, 7, 'N', 'superficie_construida_bajo_rasante',pl.Float32],
    [327, 7, 'N', 'superficie_cubierta',pl.Float32],
    [334, 9, 'N', 'coordenada_x_por_cien',pl.Float32],
    [343, 10, 'N', 'coordenada_y_por_cien',pl.Float32],
    [582, 20, 'X', 'referencia_catastral_bice',pl.Utf8],
    [602, 65, 'X', 'denominacion_bice',pl.Utf8],
    [667, 10, 'X', 'codigo_epsg',pl.Utf8]
]

# 13 - Registro de Unidad Constructiva
catstruct[13] = [
    [24, 2, 'N', 'codigo_delegacion_meh',pl.Utf8],
    [26, 3, 'N', 'codigo_municipio_dgc',pl.Utf8],
    [29, 2, 'X', 'clase_unidad_constructiva',pl.Utf8],
    [31, 14, 'X', 'parcela_catastral',pl.Utf8],
    [45, 4, 'X', 'codigo_unidad_constructiva',pl.Utf8],
    [51, 2, 'N', 'codigo_provincia_ine',pl.Utf8],
    [53, 25, 'X', 'nombre_provincia',pl.Utf8],
    [78, 3, 'N', 'codigo_municipio_dgc_2',pl.Utf8],
    [81, 3, 'N', 'codigo_municipio_ine',pl.Utf8],
    [84, 40, 'X', 'nombre_municipio',pl.Utf8],
    [124, 30, 'X', 'nombre_entidad_menor',pl.Utf8],
    [154, 5, 'N', 'codigo_via_publica_dgc',pl.Utf8],
    [159, 5, 'X', 'tipo_via',pl.Utf8],
    [164, 25, 'X', 'nombre_via',pl.Utf8],
    [189, 4, 'N', 'primer_numero_policia',pl.Utf8],
    [193, 1, 'X', 'primera_letra',pl.Utf8],
    [194, 4, 'N', 'segundo_numero_policia',pl.Utf8],
    [198, 1, 'X', 'segunda_letra',pl.Utf8],
    [199, 5, 'N', 'kilometro_por_cien',pl.Utf8],
    [216, 25, 'X', 'direccion_no_estructurada',pl.Utf8],
    [296, 4, 'N', 'año_construccion',pl.Int16],
    [300, 1, 'X', 'exactitud_año_construccion',pl.Utf8],
    [301, 7, 'N', 'superficie_suelo_ocupado',pl.Float32],
    [308, 5, 'N', 'longitud_fachada_cm',pl.Float32],
    [410, 4, 'X', 'codigo_unidad_constructiva_matriz',pl.Utf8]
]

# 14 - Registro de Construccion
catstruct[14] = [
    [24, 2, 'N', 'delegation_meh_code', pl.Utf8],
    [26, 3, 'N', 'municipality_cadaster_code', pl.Utf8],
    [29, 2, 'X', 'real_estate_type', pl.Utf8],
    [31, 14, 'X', 'building_reference', pl.Utf8],
    [45, 4, 'N', 'element_reference',pl.Utf8],
    [51, 4, 'X', 'space1_reference',pl.Utf8],
    [59, 4, 'X', 'building_space_block_name',pl.Utf8],
    [63, 2, 'X', 'building_space_stair_name',pl.Utf8],
    [65, 3, 'X', 'building_space_floor_name',pl.Utf8],
    [68, 3, 'X', 'building_space_door_name',pl.Utf8],
    [71, 3, 'X', 'building_space_detailed_use_type', pl.Utf8],
    [74, 1, 'X', 'retrofitted', pl.Utf8],
    [75, 4, 'N', 'building_space_retroffiting_year', pl.Int16],
    [79, 4, 'N', 'building_space_effective_year', pl.Int16],
    [83, 1, 'X', 'local_interior_indicator', pl.Utf8],
    [84, 7, 'N', 'building_space_area_without_communal', pl.Float32],
    [91, 7, 'N', 'building_space_area_balconies_terraces', pl.Float32],
    [98, 7, 'N', 'building_space_area_imputable_to_other_floors', pl.Float32],
    [105, 5, 'X', 'building_space_typology', pl.Utf8],
    [112, 3, 'X', 'distribution_method_for_communal_areas', pl.Utf8]
]

building_space_detailed_use_types = {
  "A": "Storage",
  "AAL": "Warehouse",
  "AAP": "Parking",
  "AES": "Station",
  "AAV": "Parking in a household",
  "BCR": "Irrigation hut",
  "BCT": "Transformer hut",
  "BIG": "Livestock facilities",
  "C": "Commerce",
  "CAT": "Automobile commerce",
  "CBZ": "Bazaar commerce",
  "CCE": "Retail commerce",
  "CCL": "Shoe commerce",
  "CCR": "Butcher commerce",
  "CDM": "Personal/Home commerce",
  "CDR": "Drugstore commerce",
  "CFN": "Financial commerce",
  "CFR": "Pharmacy commerce",
  "CFT": "Plumbing commerce",
  "CGL": "Galleries commerce",
  "CIM": "Printing commerce",
  "CJY": "Jewelry commerce",
  "CLB": "Bookstore commerce",
  "CMB": "Furniture commerce",
  "CPA": "Wholesale commerce",
  "CPR": "Perfumery commerce",
  "CRL": "Watchmaking commerce",
  "CSP": "Supermarket commerce",
  "CTJ": "Fabric commerce",
  "E": "Education",
  "EBL": "Education (Library)",
  "EBS": "Basic education",
  "ECL": "Cultural house education",
  "EIN": "Institute education",
  "EMS": "Museum education",
  "EPR": "Professional education",
  "EUN": "University education",
  "IIM": "Chemical industry",
  "IMD": "Wood industry",
  "G": "Hotel",
  "GC1": "Hotel Cafe 1 Star",
  "GC2": "Hotel Cafe 2 Stars",
  "GC3": "Hotel Cafe 3 Stars",
  "GC4": "Hotel Cafe 4 Stars",
  "GC5": "Hotel Cafe 5 Stars",
  "GH1": "Hotel 1 Star",
  "GH2": "Hotel 2 Stars",
  "GH3": "Hotel 3 Stars",
  "GH4": "Hotel 4 Stars",
  "GH5": "Hotel 5 Stars",
  "GPL": "Luxury apartments",
  "GP1": "Luxury apartments 1 Star",
  "GP2": "Luxury apartments 2 Stars",
  "GP3": "Luxury apartments 3 Stars",
  "GR1": "Restaurant 1 Star",
  "GR2": "Restaurant 2 Stars",
  "GR3": "Restaurant 3 Stars",
  "GR4": "Restaurant 4 Stars",
  "GR5": "Restaurant 5 Stars",
  "GS1": "Hostel Standard 1",
  "GS2": "Hostel Standard 2",
  "GS3": "Hostel Standard 3",
  "GTL": "Luxury guesthouse",
  "GT1": "Luxury guesthouse 1 Star",
  "GT2": "Luxury guesthouse 2 Stars",
  "GT3": "Luxury guesthouse 3 Stars",
  "I": "Industry",
  "IAG": "Agricultural industry",
  "IAL": "Food industry",
  "IAR": "Farming industry",
  "IBB": "Beverage industry",
  "IBR": "Clay industry",
  "ICN": "Construction industry",
  "ICT": "Quarry/Mining industry",
  "IEL": "Electric industry",
  "O99": "Other office activities",
  "P": "Public",
  "IMN": "Manufacturing industry",
  "IMT": "Metal industry",
  "IMU": "Machinery industry",
  "IPL": "Plastics industry",
  "IPP": "Paper industry",
  "IPS": "Fishing industry",
  "IPT": "Petroleum industry",
  "ITB": "Tobacco industry",
  "ITX": "Textile industry",
  "IVD": "Glass industry",
  "JAM": "Oil mills",
  "JAS": "Sawmills",
  "JBD": "Wineries",
  "JCH": "Mushroom farms",
  "JGR": "Farms",
  "JIN": "Greenhouses",
  "K": "Sports",
  "KDP": "Sports facilities",
  "KES": "Stadium",
  "KPL": "Sports complex",
  "KPS": "Swimming pool",
  "M": "Undeveloped land",
  "O": "Office",
  "O02": "Superior office",
  "O03": "Medium office",
  "O06": "Medical/Law office",
  "O07": "Nursing office",
  "O11": "Teacher's office",
  "O13": "University professor office",
  "O15": "Writer's office",
  "O16": "Plastic arts office",
  "O17": "Musician's office",
  "O43": "Salesperson office",
  "O44": "Agent office",
  "O75": "Weaver's office",
  "O79": "Tailor's office",
  "O81": "Carpenter's office",
  "O88": "Jeweler's office",
  "YSC": "Other rescue facilities",
  "YSL": "Silos, solid storage",
  "YSN": "Other sanatorium",
  "YSO": "Other provincial union",
  "PAA": "Public town hall (<20,000)",
  "PAD": "Public courthouse",
  "PAE": "Public town hall (>20,000)",
  "PCB": "Public government hall",
  "PDL": "Public delegation",
  "PGB": "Government building",
  "PJA": "Regional court",
  "PJO": "Provincial court",
  "R": "Religious",
  "RBS": "Religious basilica",
  "RCP": "Religious chapel",
  "RCT": "Religious cathedral",
  "RER": "Religious hermitage",
  "RPR": "Religious parish",
  "RSN": "Religious sanctuary",
  "T": "Entertainment",
  "TAD": "Auditorium",
  "TCM": "Cinema",
  "TCN": "Cinema (undecorated)",
  "TSL": "Entertainment hall",
  "TTT": "Theater",
  "V": "Housing",
  "Y": "Other uses",
  "YAM": "Other outpatient clinic",
  "YCA": "Casino (<20,000)",
  "YCB": "Club",
  "YCE": "Casino (>20,000)",
  "YCL": "Clinic",
  "YDG": "Gas storage",
  "YDL": "Liquid storage tanks",
  "YDS": "Other dispensary",
  "YGR": "Daycare",
  "YHG": "Hygiene facilities",
  "YHS": "Hospital",
  "YJD": "Private garden (100%)",
  "YPO": "Porch (100%)",
  "YRS": "Residence",
  "YSA": "Local union",
  "YSP": "Colonnade (50%)",
  "YOU": "Urbanization works",
  "YTD": "Open terrace (100%)",
  "YTZ": "Covered terrace (100%)",
  "Z": "Other uses",
  "ZAM": "Outpatient clinic",
  "ZBE": "Ponds, tanks",
  "ZCA": "Casino (<20,000)",
  "ZCB": "Club",
  "ZCE": "Casino (>20,000)",
  "ZCL": "Clinic",
  "ZCT": "Quarries",
  "ZDE": "Water treatment plants",
  "ZDG": "Gas storage",
  "ZDL": "Liquid storage tanks",
  "ZDS": "Other dispensary",
  "ZGR": "Daycare",
  "ZGV": "Gravel pits",
  "ZHG": "Hygiene facilities",
  "ZHS": "Hospital",
  "ZMA": "Open-pit mines",
  "ZME": "Docks and piers",
  "ZPC": "Fish farms",
  "ZRS": "Residence",
  "ZSA": "Local union",
  "ZSC": "Other rescue facilities",
  "ZSL": "Silos, solid storage",
  "ZSN": "Other sanatorium",
  "ZSO": "Other provincial union",
  "ZVR": "Landfill"
}

building_space_typologies = { #https://www.boe.es/buscar/act.php?id=BOE-A-1993-19265
    "0111": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Collective Urban Housing",
        "UseClassModality": "Open Building",
        "ConstructionValue": [1.65, 1.40, 1.20, 1.05, 0.95, 0.85, 0.75, 0.65, 0.55]
    },
    "0112": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Collective Urban Housing",
        "UseClassModality": "Closed Block",
        "ConstructionValue": [1.60, 1.35, 1.15, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
    },
    "0113": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Collective Urban Housing",
        "UseClassModality": "Garages, Storage Rooms, and Premises in Structure",
        "ConstructionValue": [0.80, 0.70, 0.62, 0.53, 0.46, 0.40, 0.30, 0.26, 0.20]
    },
    "0121": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Single-Family Urban Housing",
        "UseClassModality": "Isolated or Semi-Detached Building",
        "ConstructionValue": [2.15, 1.80, 1.45, 1.25, 1.10, 1.00, 0.90, 0.80, 0.70]
    },
    "0122": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Single-Family Urban Housing",
        "UseClassModality": "Row or Closed Block",
        "ConstructionValue": [2.00, 1.65, 1.35, 1.15, 1.05, 0.95, 0.85, 0.75, 0.65]
    },
    "0123": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Single-Family Urban Housing",
        "UseClassModality": "Garages and Porches on Ground Floor",
        "ConstructionValue": [0.90, 0.85, 0.75, 0.65, 0.60, 0.55, 0.45, 0.40, 0.35]
    },
    "0131": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Rural Building",
        "UseClassModality": "Exclusive Housing Use",
        "ConstructionValue": [1.35, 1.20, 1.05, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    },
    "0132": {
        "Use": "Residential",
        "UseLevel": 1,
        "UseClass": "Rural Building",
        "UseClassModality": "Annexes",
        "ConstructionValue": [0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
    },
    "0211": {
        "Use": "Industrial",
        "UseLevel": 3,
        "UseClass": "Manufacturing and Storage Sheds",
        "UseClassModality": "Single-Story Manufacturing",
        "ConstructionValue": [1.05, 0.90, 0.75, 0.60, 0.50, 0.45, 0.40, 0.37, 0.35]
    },
    "0212": {
        "Use": "Industrial",
        "UseLevel": 3,
        "UseClass": "Manufacturing and Storage Sheds",
        "UseClassModality": "Multi-Story Manufacturing",
        "ConstructionValue": [1.15, 1.00, 0.85, 0.70, 0.60, 0.55, 0.52, 0.50, 0.40]
    },
    "0213": {
        "Use": "Industrial",
        "UseLevel": 2,
        "UseClass": "Manufacturing and Storage Sheds",
        "UseClassModality": "Storage",
        "ConstructionValue": [0.85, 0.70, 0.60, 0.50, 0.45, 0.35, 0.30, 0.25, 0.20]
    },
    "0221": {
        "Use": "Industrial",
        "UseLevel": 2,
        "UseClass": "Garages and Parking Lots",
        "UseClassModality": "Garages",
        "ConstructionValue": [1.15, 1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    },
    "0222": {
        "Use": "Industrial",
        "UseLevel": 2,
        "UseClass": "Garages and Parking Lots",
        "UseClassModality": "Parking Lots",
        "ConstructionValue": [0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.20, 0.10, 0.05]
    },
    "0231": {
        "Use": "Industrial",
        "UseLevel": 2,
        "UseClass": "Transport Services",
        "UseClassModality": "Service Stations",
        "ConstructionValue": [1.80, 1.60, 1.40, 1.25, 1.20, 1.10, 1.00, 0.90, 0.80]
    },
    "0232": {
        "Use": "Industrial",
        "UseLevel": 2,
        "UseClass": "Transport Services",
        "UseClassModality": "Stations",
        "ConstructionValue": [2.55, 2.25, 2.00, 1.80, 1.60, 1.40, 1.25, 1.10, 1.00]
    },
    "0311": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Exclusive Building",
        "UseClassModality": "Multiple Offices",
        "ConstructionValue": [2.35, 2.00, 1.70, 1.50, 1.30, 1.15, 1.00, 0.90, 0.80]
    },
    "0312": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Exclusive Building",
        "UseClassModality": "Single Offices",
        "ConstructionValue": [2.55, 2.20, 1.85, 1.60, 1.40, 1.25, 1.10, 1.00, 0.90]
    },
    "0321": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Mixed Building",
        "UseClassModality": "Attached to Housing",
        "ConstructionValue": [2.05, 1.80, 1.50, 1.30, 1.10, 1.00, 0.90, 0.80, 0.70]
    },
    "0322": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Mixed Building",
        "UseClassModality": "Attached to Industry",
        "ConstructionValue": [1.40, 1.25, 1.10, 1.00, 0.85, 0.65, 0.55, 0.45, 0.35]
    },
    "0331": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Banking and Insurance",
        "UseClassModality": "In Exclusive Building",
        "ConstructionValue": [2.95, 2.65, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20]
    },
    "0332": {
        "Use": "Offices",
        "UseLevel": 1,
        "UseClass": "Banking and Insurance",
        "UseClassModality": "In Mixed Building",
        "ConstructionValue": [2.65, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0411": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Commerce in Mixed Building",
        "UseClassModality": "Shops and Workshops",
        "ConstructionValue": [1.95, 1.60, 1.35, 1.20, 1.05, 0.95, 0.85, 0.75, 0.65]
    },
    "0412": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Commerce in Mixed Building",
        "UseClassModality": "Commercial Galleries",
        "ConstructionValue": [1.85, 1.65, 1.45, 1.30, 1.15, 1.00, 0.90, 0.80, 0.70]
    },
    "0421": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Commerce in Exclusive Building",
        "UseClassModality": "Single Floor",
        "ConstructionValue": [2.50, 2.15, 1.85, 1.60, 1.40, 1.25, 1.10, 1.00, 0.85]
    },
    "0422": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Commerce in Exclusive Building",
        "UseClassModality": "Multiple Floors",
        "ConstructionValue": [2.75, 2.35, 2.00, 1.75, 1.50, 1.35, 1.20, 1.05, 0.90]
    },
    "0431": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Markets and Supermarkets",
        "UseClassModality": "Markets",
        "ConstructionValue": [2.00, 1.80, 1.60, 1.45, 1.30, 1.15, 1.00, 0.90, 0.80]
    },
    "0432": {
        "Use": "Commercial",
        "UseLevel": 2,
        "UseClass": "Markets and Supermarkets",
        "UseClassModality": "Hypermarkets and Supermarkets",
        "ConstructionValue": [1.80, 1.60, 1.45, 1.30, 1.15, 1.00, 0.90, 0.80, 0.70]
    },
    "0511": {
        "Use": "Sports",
        "UseLevel": 2,
        "UseClass": "Covered",
        "UseClassModality": "Various Sports",
        "ConstructionValue": [2.10, 1.90, 1.70, 1.50, 1.30, 1.10, 0.90, 0.70, 0.50]
    },
    "0512": {
        "Use": "Sports",
        "UseLevel": 2,
        "UseClass": "Covered",
        "UseClassModality": "Pools",
        "ConstructionValue": [2.30, 2.05, 1.85, 1.65, 1.45, 1.30, 1.15, 1.00, 0.90]
    },
    "0521": {
        "Use": "Sports",
        "UseLevel": 2,
        "UseClass": "Uncovered",
        "UseClassModality": "Various Sports",
        "ConstructionValue": [0.70, 0.55, 0.50, 0.45, 0.35, 0.25, 0.20, 0.10, 0.05]
    },
    "0522": {
        "Use": "Sports",
        "UseLevel": 2,
        "UseClass": "Uncovered",
        "UseClassModality": "Pools",
        "ConstructionValue": [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25]
    },
    "0531": {
        "Use": "Sports",
        "UseLevel": 2,
        "UseClass": "Auxiliaries",
        "UseClassModality": "Locker Rooms, Water Treatment, Heating, etc.",
        "ConstructionValue": [1.50, 1.35, 1.20, 1.05, 0.90, 0.80, 0.70, 0.60, 0.50]
    },
    "0541": {
        "Use": "Sports",
        "UseLevel": 3,
        "UseClass": "Sports Shows",
        "UseClassModality": "Stadiums, Bullrings",
        "ConstructionValue": [2.40, 2.15, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05, 0.95]
    },
    "0542": {
        "Use": "Sports",
        "UseLevel": 3,
        "UseClass": "Sports Shows",
        "UseClassModality": "Racecourses, Dog Tracks, Velodromes, etc.",
        "ConstructionValue": [2.20, 1.95, 1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.90]
    },
    "0611": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Various",
        "UseClassModality": "Covered",
        "ConstructionValue": [1.90, 1.70, 1.50, 1.35, 1.20, 1.05, 0.95, 0.85, 0.75]
    },
    "0612": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Various",
        "UseClassModality": "Uncovered",
        "ConstructionValue": [0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    },
    "0621": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Musical Bars, Party Halls, and Discotheques",
        "UseClassModality": "In Exclusive Building",
        "ConstructionValue": [2.65, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0622": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Musical Bars, Party Halls, and Discotheques",
        "UseClassModality": "Attached to Other Uses",
        "ConstructionValue": [2.20, 1.95, 1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.90]
    },
    "0631": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Cinemas and Theaters",
        "UseClassModality": "Cinemas",
        "ConstructionValue": [2.55, 2.30, 2.05, 1.80, 1.60, 1.45, 1.30, 1.15, 1.00]
    },
    "0632": {
        "Use": "Shows",
        "UseLevel": 3,
        "UseClass": "Cinemas and Theaters",
        "UseClassModality": "Theaters",
        "ConstructionValue": [2.70, 2.40, 2.15, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0711": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "With Residence",
        "UseClassModality": "Hotels, Hostels, Motels",
        "ConstructionValue": [2.65, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0712": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "With Residence",
        "UseClassModality": "Aparthotels, Bungalows",
        "ConstructionValue": [2.85, 2.55, 2.30, 2.05, 1.85, 1.65, 1.45, 1.30, 1.15]
    },
    "0721": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "Without Residence",
        "UseClassModality": "Restaurants",
        "ConstructionValue": [2.60, 2.35, 2.00, 1.75, 1.50, 1.35, 1.20, 1.05, 0.95]
    },
    "0722": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "Without Residence",
        "UseClassModality": "Bars and Cafeterias",
        "ConstructionValue": [2.35, 2.00, 1.70, 1.50, 1.30, 1.15, 1.00, 0.90, 0.80]
    },
    "0731": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "Exhibitions and Meetings",
        "UseClassModality": "Casinos and Social Clubs",
        "ConstructionValue": [2.60, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0732": {
        "Use": "Leisure and Hospitality",
        "UseLevel": 2,
        "UseClass": "Exhibitions and Meetings",
        "UseClassModality": "Exhibitions and Congresses",
        "ConstructionValue": [2.50, 2.25, 2.00, 1.80, 1.60, 1.45, 1.25, 1.10, 1.00]
    },
    "0811": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Healthcare with Beds",
        "UseClassModality": "Sanatoriums and Clinics",
        "ConstructionValue": [3.15, 2.80, 2.50, 2.25, 2.00, 1.80, 1.60, 1.45, 1.30]
    },
    "0812": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Healthcare with Beds",
        "UseClassModality": "Hospitals",
        "ConstructionValue": [3.05, 2.70, 2.40, 2.15, 1.90, 1.70, 1.50, 1.35, 1.20]
    },
    "0821": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Various Healthcare",
        "UseClassModality": "Ambulatory Care and Clinics",
        "ConstructionValue": [2.40, 2.15, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05, 0.95]
    },
    "0822": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Various Healthcare",
        "UseClassModality": "Spas and Bathhouses",
        "ConstructionValue": [2.65, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0831": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Welfare and Assistance",
        "UseClassModality": "With Residence (Asylums, Residences, etc.)",
        "ConstructionValue": [2.45, 2.20, 2.00, 1.80, 1.60, 1.40, 1.25, 1.10, 1.00]
    },
    "0832": {
        "Use": "Health and Welfare",
        "UseLevel": 2,
        "UseClass": "Welfare and Assistance",
        "UseClassModality": "Without Residence (Dining Rooms, Clubs, Daycares, etc.)",
        "ConstructionValue": [1.95, 1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.90, 0.80]
    },
    "0911": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Cultural with Residence",
        "UseClassModality": "Boarding Schools",
        "ConstructionValue": [2.40, 2.15, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05, 0.95]
    },
    "0912": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Cultural with Residence",
        "UseClassModality": "University Halls of Residence",
        "ConstructionValue": [2.60, 2.35, 2.10, 1.90, 1.70, 1.50, 1.35, 1.20, 1.05]
    },
    "0921": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Cultural without Residence",
        "UseClassModality": "Faculties, Colleges, and Schools",
        "ConstructionValue": [1.95, 1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.90, 0.80]
    },
    "0922": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Cultural without Residence",
        "UseClassModality": "Libraries and Museums",
        "ConstructionValue": [2.30, 2.05, 1.85, 1.65, 1.45, 1.30, 1.15, 1.00, 0.90]
    },
    "0931": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Religious",
        "UseClassModality": "Convents and Parish Centers",
        "ConstructionValue": [1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.90, 0.80, 0.70]
    },
    "0932": {
        "Use": "Cultural and Religious",
        "UseLevel": 2,
        "UseClass": "Religious",
        "UseClassModality": "Churches and Chapels",
        "ConstructionValue": [2.90, 2.60, 2.30, 2.00, 1.80, 1.60, 1.40, 1.20, 1.05]
    },
    "1011": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Historic-Artistic",
        "UseClassModality": "Monumental",
        "ConstructionValue": [2.90, 2.60, 2.30, 2.00, 1.80, 1.60, 1.40, 1.20, 1.05]
    },
    "1012": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Historic-Artistic",
        "UseClassModality": "Environmental or Typical",
        "ConstructionValue": [2.30, 2.05, 1.85, 1.65, 1.45, 1.30, 1.15, 1.00, 0.90]
    },
    "1021": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Official",
        "UseClassModality": "Administrative",
        "ConstructionValue": [2.55, 2.20, 1.85, 1.60, 1.30, 1.15, 1.00, 0.90, 0.80]
    },
    "1022": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Official",
        "UseClassModality": "Representative",
        "ConstructionValue": [2.75, 2.35, 2.00, 1.75, 1.50, 1.35, 1.20, 1.05, 0.95]
    },
    "1031": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Penitentiary, Military, and Various",
        "ConstructionValue": [2.20, 1.95, 1.75, 1.55, 1.40, 1.25, 1.10, 1.00, 0.85]
    },
    "1032": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Interior Urbanization Works",
        "ConstructionValue": [0.26, 0.22, 0.18, 0.15, 0.11, 0.08, 0.06, 0.04, 0.03]
    },
    "1033": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Campgrounds",
        "ConstructionValue": [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
    },
    "1034": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Golf Courses",
        "ConstructionValue": [0.050, 0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005]
    },
    "1035": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Gardening",
        "ConstructionValue": [0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
    },
    "1036": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Silos and Solid Storage (m³)",
        "ConstructionValue": [0.35, 0.30, 0.25, 0.20, 0.17, 0.15, 0.14, 0.12, 0.10]
    },
    "1037": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Liquid Storage (m³)",
        "ConstructionValue": [0.37, 0.34, 0.31, 0.29, 0.25, 0.23, 0.20, 0.17, 0.15]
    },
    "1038": {
        "Use": "Unique Buildings",
        "UseLevel": 1,
        "UseClass": "Special",
        "UseClassModality": "Gas Storage (m³)",
        "ConstructionValue": [0.80, 0.65, 0.50, 0.40, 0.37, 0.35, 0.31, 0.27, 0.25]
    }
}

building_space_age_value = [
    {
        "Age": [0, 4],
        "1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "2": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "3": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    },
    {
        "Age": [5, 9],
        "1": [0.93, 0.93, 0.92, 0.92, 0.92, 0.92, 0.90, 0.90, 0.90],
        "2": [0.93, 0.93, 0.91, 0.91, 0.91, 0.91, 0.89, 0.89, 0.89],
        "3": [0.92, 0.92, 0.90, 0.90, 0.90, 0.90, 0.88, 0.88, 0.88]
    },
    {
        "Age": [10, 14],
        "1": [0.87, 0.87, 0.85, 0.85, 0.85, 0.85, 0.82, 0.82, 0.82],
        "2": [0.86, 0.86, 0.84, 0.84, 0.84, 0.84, 0.80, 0.80, 0.80],
        "3": [0.84, 0.84, 0.82, 0.82, 0.82, 0.82, 0.78, 0.78, 0.78]
    },
    {
        "Age": [15, 19],
        "1": [0.82, 0.82, 0.79, 0.79, 0.79, 0.79, 0.74, 0.74, 0.74],
        "2": [0.80, 0.80, 0.77, 0.77, 0.77, 0.77, 0.72, 0.72, 0.72],
        "3": [0.78, 0.78, 0.74, 0.74, 0.74, 0.74, 0.69, 0.69, 0.69]
    },
    {
        "Age": [20, 24],
        "1": [0.77, 0.77, 0.73, 0.73, 0.73, 0.73, 0.67, 0.67, 0.67],
        "2": [0.75, 0.75, 0.70, 0.70, 0.70, 0.70, 0.64, 0.64, 0.64],
        "3": [0.72, 0.72, 0.67, 0.67, 0.67, 0.67, 0.61, 0.61, 0.61]
    },
    {
        "Age": [25, 29],
        "1": [0.72, 0.72, 0.68, 0.68, 0.68, 0.68, 0.61, 0.61, 0.61],
        "2": [0.70, 0.70, 0.65, 0.65, 0.65, 0.65, 0.58, 0.58, 0.58],
        "3": [0.67, 0.67, 0.61, 0.61, 0.61, 0.61, 0.54, 0.54, 0.54]
    },
    {
        "Age": [30, 34],
        "1": [0.68, 0.68, 0.63, 0.63, 0.63, 0.63, 0.56, 0.56, 0.56],
        "2": [0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.53, 0.53, 0.53],
        "3": [0.62, 0.62, 0.56, 0.56, 0.56, 0.56, 0.49, 0.49, 0.49]
    },
    {
        "Age": [35, 39],
        "1": [0.64, 0.64, 0.59, 0.59, 0.59, 0.59, 0.51, 0.51, 0.51],
        "2": [0.61, 0.61, 0.56, 0.56, 0.56, 0.56, 0.48, 0.48, 0.48],
        "3": [0.58, 0.58, 0.51, 0.51, 0.51, 0.51, 0.44, 0.44, 0.44]
    },
    {
        "Age": [40, 44],
        "1": [0.61, 0.61, 0.55, 0.55, 0.55, 0.55, 0.47, 0.47, 0.47],
        "2": [0.57, 0.57, 0.52, 0.52, 0.52, 0.52, 0.44, 0.44, 0.44],
        "3": [0.54, 0.54, 0.47, 0.47, 0.47, 0.47, 0.39, 0.39, 0.39]
    },
    {
        "Age": [45, 49],
        "1": [0.58, 0.58, 0.52, 0.52, 0.52, 0.52, 0.43, 0.43, 0.43],
        "2": [0.54, 0.54, 0.48, 0.48, 0.48, 0.48, 0.40, 0.40, 0.40],
        "3": [0.50, 0.50, 0.43, 0.43, 0.43, 0.43, 0.35, 0.35, 0.35]
    },
    {
        "Age": [50, 54],
        "1": [0.55, 0.55, 0.49, 0.49, 0.49, 0.49, 0.40, 0.40, 0.40],
        "2": [0.51, 0.51, 0.45, 0.45, 0.45, 0.45, 0.37, 0.37, 0.37],
        "3": [0.47, 0.47, 0.40, 0.40, 0.40, 0.40, 0.32, 0.32, 0.32]
    },
    {
        "Age": [55, 59],
        "1": [0.52, 0.52, 0.46, 0.46, 0.46, 0.46, 0.37, 0.37, 0.37],
        "2": [0.48, 0.48, 0.42, 0.42, 0.42, 0.42, 0.34, 0.34, 0.34],
        "3": [0.44, 0.44, 0.37, 0.37, 0.37, 0.37, 0.29, 0.29, 0.29]
    },
    {
        "Age": [60, 64],
        "1": [0.49, 0.49, 0.43, 0.43, 0.43, 0.43, 0.34, 0.34, 0.34],
        "2": [0.45, 0.45, 0.39, 0.39, 0.39, 0.39, 0.31, 0.31, 0.31],
        "3": [0.41, 0.41, 0.34, 0.34, 0.34, 0.34, 0.26, 0.26, 0.26]
    },
    {
        "Age": [65, 69],
        "1": [0.47, 0.47, 0.41, 0.41, 0.41, 0.41, 0.32, 0.32, 0.32],
        "2": [0.43, 0.43, 0.37, 0.37, 0.37, 0.37, 0.29, 0.29, 0.29],
        "3": [0.39, 0.39, 0.32, 0.32, 0.32, 0.32, 0.24, 0.24, 0.24]
    },
    {
        "Age": [70, 74],
        "1": [0.45, 0.45, 0.39, 0.39, 0.39, 0.39, 0.30, 0.30, 0.30],
        "2": [0.41, 0.41, 0.35, 0.35, 0.35, 0.35, 0.27, 0.27, 0.27],
        "3": [0.37, 0.37, 0.30, 0.30, 0.30, 0.30, 0.22, 0.22, 0.22]
    },
    {
        "Age": [75, 79],
        "1": [0.43, 0.43, 0.37, 0.37, 0.37, 0.37, 0.28, 0.28, 0.28],
        "2": [0.39, 0.39, 0.33, 0.33, 0.33, 0.33, 0.25, 0.25, 0.25],
        "3": [0.35, 0.35, 0.28, 0.28, 0.28, 0.28, 0.20, 0.20, 0.20]
    },
    {
        "Age": [80, 84],
        "1": [0.41, 0.41, 0.35, 0.35, 0.35, 0.35, 0.26, 0.26, 0.26],
        "2": [0.37, 0.37, 0.31, 0.31, 0.31, 0.31, 0.23, 0.23, 0.23],
        "3": [0.33, 0.33, 0.26, 0.26, 0.26, 0.26, 0.19, 0.19, 0.19]
    },
    {
        "Age": [85, 89],
        "1": [0.40, 0.40, 0.33, 0.33, 0.33, 0.33, 0.25, 0.25, 0.25],
        "2": [0.36, 0.36, 0.29, 0.29, 0.29, 0.29, 0.21, 0.21, 0.21],
        "3": [0.31, 0.31, 0.25, 0.25, 0.25, 0.25, 0.18, 0.18, 0.18]
    },
    {
        "Age": [90, np.inf],
        "1": [0.39, 0.39, 0.32, 0.32, 0.32, 0.32, 0.24, 0.24, 0.24],
        "2": [0.35, 0.35, 0.28, 0.28, 0.28, 0.28, 0.20, 0.20, 0.20],
        "3": [0.30, 0.30, 0.24, 0.24, 0.24, 0.24, 0.17, 0.17, 0.17]
    }
]

# 15 - Registro de Inmueble
catstruct[15] = [
    [24, 2, 'N', 'delegation_meh_code', pl.Utf8],
    [26, 3, 'N', 'municipality_cadaster_code', pl.Utf8],
    [29, 2, 'X', 'real_estate_type', pl.Utf8],
    [31, 14, 'X', 'building_reference', pl.Utf8],
    [45, 4, 'N', 'space1_reference', pl.Utf8],
    [49, 1, 'X', 'space2_reference', pl.Utf8],
    [50, 1, 'X', 'space3_reference', pl.Utf8],
    [51, 8, 'N', 'real_estate_fix_number', pl.Utf8],
    [59, 15, 'X', 'real_estate_id_city_council', pl.Utf8],
    [74, 19, 'X', 'register_reference', pl.Utf8],
    [93, 2, 'N', 'province_code', pl.Utf8],
    [95, 25, 'X', 'province_name', pl.Utf8],
    [120, 3, 'N', 'municipality_cadaster_code_2', pl.Utf8],
    [123, 3, 'N', 'municipality_ine_code', pl.Utf8],
    [126, 40, 'X', 'municipality_name', pl.Utf8],
    [166, 30, 'X', 'minor_entity_name', pl.Utf8],
    [196, 5, 'N', 'street_cadaster_code', pl.Utf8],
    [201, 5, 'X', 'street_type', pl.Utf8],
    [206, 25, 'X', 'street_name', pl.Utf8],
    [231, 4, 'N', 'street_number1', pl.Utf8],
    [235, 1, 'X', 'street_letter1', pl.Utf8],
    [236, 4, 'N', 'street_number2', pl.Utf8],
    [240, 1, 'X', 'street_letter2', pl.Utf8],
    [241, 5, 'N', 'km', pl.Utf8],
    [246, 4, 'X', 'building_block_name', pl.Utf8],
    [250, 2, 'X', 'building_stair_name', pl.Utf8],
    [252, 3, 'X', 'building_floor_name', pl.Utf8],
    [255, 3, 'X', 'building_door_name', pl.Utf8],
    [258, 25, 'X', 'street_unstructured', pl.Utf8],
    [283, 5, 'N', 'postal_code', pl.Utf8],
    [288, 2, 'X', 'district_code', pl.Utf8],
    [290, 3, 'N', 'alternative_municipality_cadaster_code', pl.Utf8],
    [293, 2, 'N', 'concentration_zone_code', pl.Utf8],
    [295, 3, 'N', 'polygon_code', pl.Utf8],
    [298, 5, 'N', 'parcel_code', pl.Utf8],
    [303, 5, 'X', 'site_cadastral_code', pl.Utf8],
    [308, 30, 'X', 'site_name', pl.Utf8],
    [368, 4, 'X', 'real_estate_notarial_deed_order', pl.Utf8],
    [372, 4, 'N', 'building_space_year', pl.Int16],
    [428, 1, 'X', 'building_space_use_type', pl.Utf8],
    [442, 10, 'N', 'building_space_total_area', pl.Float32],
    [452, 10, 'N', 'building_space_related_area', pl.Float32],
    [462, 9, 'N', 'building_space_participation_rate', pl.Float32]
]

building_space_use_types = {
    "1": "Buildings intended for electricity and gas production, oil refining, and nuclear power plants",
    "2": "Dams, waterfalls, and reservoirs",
    "3": "Highways, roads, and toll tunnels",
    "4": "Airports and commercial ports",
    "A": "Warehouse - Parking",
    "V": "Residential",
    "I": "Industrial",
    "O": "Offices",
    "C": "Commercial",
    "K": "Sports facilities",
    "T": "Entertainment venues",
    "G": "Leisure and Hospitality",
    "Y": "Healthcare and Charity",
    "E": "Cultural",
    "R": "Religious",
    "M": "Urbanization and landscaping works, undeveloped land",
    "P": "Singular building",
    "B": "Agricultural warehouse",
    "J": "Agricultural industrial",
    "Z": "Agricultural"
}

# 16 - Registro de reparto de elementos comunes
catstruct[16] = [
   [24, 2, 'N', 'codigo_delegacion_meh', pl.Utf8],
   [26, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [31, 14, 'X', 'parcela_catastral', pl.Utf8],
   [45, 4, 'N', 'numero_elemento', pl.Utf8],
   [49, 2, 'X', 'calificacion_catastral_subparcela_abstracta', pl.Utf8],
   [51, 4, 'N', 'numero_orden_segmento', pl.Utf8],
   [55, 59, 'N', 'b1', pl.Utf8],
   [114, 59, 'N', 'b2', pl.Utf8],
   [173, 59, 'N', 'b3', pl.Utf8],
   [232, 59, 'N', 'b4', pl.Utf8],
   [291, 59, 'N', 'b5', pl.Utf8],
   [350, 59, 'N', 'b6', pl.Utf8],
   [409, 59, 'N', 'b7', pl.Utf8],
   [468, 59, 'N', 'b8', pl.Utf8],
   [527, 59, 'N', 'b9', pl.Utf8],
   [586, 59, 'N', 'b10', pl.Utf8],
   [645, 59, 'N', 'b11', pl.Utf8],
   [704, 59, 'N', 'b12', pl.Utf8],
   [763, 59, 'N', 'b13', pl.Utf8],
   [822, 59, 'N', 'b14', pl.Utf8],
   [881, 59, 'N', 'b15', pl.Utf8]
]

# 17 - Registro de cultivos
catstruct[17] = [
   [24, 2, 'N', 'codigo_delegacion_meh', pl.Utf8],
   [26, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [29, 2, 'X', 'naturaleza_suelo_ocupado', pl.Utf8], # 'UR' urbana, 'RU' rustica
   [31, 14, 'X', 'parcela_catastral', pl.Utf8],
   [45, 4, 'X', 'codigo_subparcela', pl.Utf8],
   [51, 4, 'N', 'numero_orden_fiscal_en_parcela', pl.Utf8],
   [55, 1, 'X', 'tipo_subparcela', pl.Utf8], # 'T' terreno, 'A' absracta, 'D' dominio publico
   [56, 10, 'N', 'superficie_subparcela_m2', pl.Float32],
   [66, 2, 'X', 'calificacion_catastral_o_clase_cultivo', pl.Utf8],
   [68, 40, 'X', 'denominacion_clase_cultivo', pl.Utf8],
   [108, 2, 'N', 'intensidad_productiva', pl.Utf8],
   [127, 3, 'X', 'codigo_modalidad_reparto', pl.Utf8] # [TA]C[1234]
]

# 46 - Registro de situaciones finales de titularidad
catstruct[46] = [
   [24, 2, 'N', 'codigo_delegacion_meh', pl.Utf8],
   [26, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [29, 2, 'X', 'naturaleza_suelo_ocupado', pl.Utf8], # 'UR' urbana, 'RU' rustica
   [31, 14, 'X', 'parcela_catastral', pl.Utf8],
   [45, 4, 'X', 'codigo_subparcela', pl.Utf8],
   [49, 1, 'X', 'primer_carac_control', pl.Utf8],
   [50, 1, 'X', 'segundo_carac_control', pl.Utf8],
   [51, 2, 'X', 'codigo_derecho', pl.Utf8],
   [53, 5, 'N', 'porcentaje_derecho', pl.Utf8],
   [58, 3, 'N', 'ordinal_derecho', pl.Utf8],
   [61, 9, 'X', 'nif_titular', pl.Utf8],
   [70, 60, 'X', 'nombre_titular', pl.Utf8], # Primer apellido, segundo y nombre o razón social
   [130, 1, 'X', 'motivo_no_nif', pl.Utf8], # 1 Extranjero, 2 menor de edad, 9 otras situaciones
   [131, 2, 'N', 'codigo_provincia_ine', pl.Utf8],
   [133, 25, 'X', 'nombre_provincia', pl.Utf8],
   [158, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [161, 3, 'N', 'codigo_municipio_ine', pl.Utf8],
   [164, 40, 'X', 'nombre_municipio', pl.Utf8],
   [204, 30, 'X', 'nombre_entidad_menor', pl.Utf8],
   [235, 5, 'N', 'codigo_via_publica_dgc', pl.Utf8],
   [239, 5, 'X', 'tipo_via', pl.Utf8],
   [244, 25, 'X', 'nombre_via', pl.Utf8],
   [269, 4, 'N', 'primer_numero_policia', pl.Utf8],
   [273, 1, 'X', 'primera_letra', pl.Utf8],
   [274, 4, 'N', 'segundo_numero_policia', pl.Utf8],
   [278, 1, 'X', 'segunda_letra', pl.Utf8],
   [279, 5, 'N', 'kilometro_por_cien', pl.Utf8],
   [284, 4, 'X', 'bloque', pl.Utf8],
   [288, 2, 'X', 'escalera', pl.Utf8],
   [290, 3, 'X', 'planta', pl.Utf8],
   [293, 3, 'X', 'puerta', pl.Utf8],
   [296, 25, 'X', 'direccion_no_estructurada', pl.Utf8],
   [321, 5, 'N', 'codigo_postal', pl.Utf8],
   [326, 5, 'N', 'apartado_correos', pl.Utf8],
   [331, 9, 'X', 'nif_conyuge', pl.Utf8],
   [340, 9, 'X', 'nif_cb', pl.Utf8],
   [349, 20, 'X', 'complemento_titularidad', pl.Utf8]
]

# 47 - Registro de comunidad de bienes formalmente constituida presente en una situación final
catstruct[47] = [
   [24, 2, 'N', 'codigo_delegacion_meh', pl.Utf8],
   [26, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [29, 2, 'X', 'naturaleza_suelo_ocupado', pl.Utf8], # 'UR' urbana, 'RU' rustica
   [31, 14, 'X', 'parcela_catastral', pl.Utf8],
   [45, 4, 'X', 'codigo_subparcela', pl.Utf8],
   [49, 1, 'X', 'primer_carac_control', pl.Utf8],
   [50, 1, 'X', 'segundo_carac_control', pl.Utf8],
   [51, 9, 'X', 'nif_comunidad_bienes', pl.Utf8],
   [60, 60, 'X', 'denominacion_razon_socil', pl.Utf8],
   [120, 2, 'N', 'codigo_provincia_ine', pl.Utf8],
   [122, 25, 'X', 'nombre_provincia', pl.Utf8],
   [147, 3, 'N', 'codigo_municipio_dgc', pl.Utf8],
   [150, 3, 'N', 'codigo_municipio_ine', pl.Utf8],
   [153, 40, 'X', 'nombre_municipio', pl.Utf8],
   [193, 30, 'X', 'nombre_entidad_menor', pl.Utf8],
   [223, 5, 'N', 'codigo_via_publica_dgc', pl.Utf8],
   [228, 5, 'X', 'tipo_via', pl.Utf8],
   [233, 25, 'X', 'nombre_via', pl.Utf8],
   [258, 4, 'N', 'primer_numero_policia', pl.Utf8],
   [262, 1, 'X', 'primera_letra', pl.Utf8],
   [263, 4, 'N', 'segundo_numero_policia', pl.Utf8],
   [267, 1, 'X', 'segunda_letra', pl.Utf8],
   [268, 5, 'N', 'kilometro_por_cien', pl.Utf8],
   [273, 4, 'X', 'bloque', pl.Utf8],
   [277, 2, 'X', 'escalera', pl.Utf8],
   [279, 3, 'X', 'planta', pl.Utf8],
   [282, 3, 'X', 'puerta', pl.Utf8],
   [285, 25, 'X', 'direccion_no_estructurada', pl.Utf8],
   [310, 5, 'N', 'codigo_postal', pl.Utf8],
   [315, 5, 'N', 'apartado_correos', pl.Utf8]
]

# 90 - Registro de cola
catstruct[90] = [
   [10, 7, 'N', 'numero_registros_tipo_11', pl.Utf8],
   [24, 7, 'N', 'numero_registros_tipo_13', pl.Utf8],
   [31, 7, 'N', 'numero_registros_tipo_14', pl.Utf8],
   [38, 7, 'N', 'numero_registros_tipo_15', pl.Utf8],
   [45, 7, 'N', 'numero_registros_tipo_16', pl.Utf8],
   [52, 7, 'N', 'numero_registros_tipo_17', pl.Utf8],
   [59, 7, 'N', 'numero_registros_tipo_46', pl.Utf8],
   [66, 7, 'N', 'numero_registros_tipo_47', pl.Utf8]
]


def parse_CAT_file(cadaster_code, CAT_files_dir, allowed_dataset_types = [14, 15]):

    # Function to parse a single line into a dictionary for each record type
    def process_line(line, allowed_dataset_types):
        parsed_row = {}

        line = line.encode('utf-8').decode('utf-8')
        line_type = int(line[0:2])  # Record type

        # Only process if the record type is below 15 and known in catstruct
        if line_type in allowed_dataset_types and line_type in catstruct:
            row = []
            for campos in catstruct[line_type]:
                ini = campos[0] - 1  # Offset
                fin = ini + campos[1]  # Length
                valor = line[ini:fin].strip()  # Extracted value
                row.append(valor)

            # Store parsed row with its type
            parsed_row[line_type] = row

        return parsed_row


    # Function to combine parsed rows into Polars DataFrames
    def combine_dataframes(parsed_rows):
        # Initialize an empty dictionary to accumulate rows for each record type
        row_data = {dataset_type: [] for dataset_type in catstruct}

        # Aggregate rows for each type from parsed rows
        for row_dict in parsed_rows:
            for dataset_type, row in row_dict.items():
                row_data[dataset_type].append(row)

        # Create DataFrames from aggregated rows and schema
        combined_dfs = {}
        for dataset_type, rows in row_data.items():
            schema = {i[3]: i[4] for i in catstruct[dataset_type]}
            combined_dfs[dataset_type] = pl.DataFrame(rows, schema=schema, orient="row")

        return combined_dfs


    # Main function to process the file in chunks and save as Parquet
    def process_file_in_chunks(inputfile, CAT_files_dir, cadaster_code, allowed_dataset_types):
        with open(inputfile, encoding='latin-1') as rf:
            lines = rf.readlines()  # Read all lines at once for chunk processing

        if isinstance(allowed_dataset_types, str) or isinstance(allowed_dataset_types,int):
            allowed_dataset_types = [int(allowed_dataset_types)]
        elif isinstance(allowed_dataset_types, list):
            allowed_dataset_types = [int(dt) for dt in allowed_dataset_types]

        if not all([os.path.exists(f"{CAT_files_dir}/parquet/{cadaster_code}_{dataset_type}.parquet") for
                    dataset_type in allowed_dataset_types]):
            # Process each line in parallel
            with utils.tqdm_joblib(tqdm(desc="Reading the CAT file...", total=len(lines))):
                parsed_rows = Parallel(n_jobs=-1)(
                    delayed(process_line)(line, allowed_dataset_types) for line in lines
                )
            get_reusable_executor().shutdown(wait=True)

            # Combine all parsed rows into DataFrames
            combined_dfs = combine_dataframes(parsed_rows)

            # Save each DataFrame to Parquet
            for dataset_type, df in combined_dfs.items():
                if len(df)>1:
                    output_path = f"{CAT_files_dir}/parquet/{cadaster_code}_{dataset_type}.parquet"
                    df.write_parquet(output_path)

        else:
            combined_dfs = {k:None for k in catstruct.keys()}
            for dataset_type in allowed_dataset_types:
                combined_dfs[dataset_type] = pl.read_parquet(f"{CAT_files_dir}/parquet/{cadaster_code}_{dataset_type}.parquet")

        return combined_dfs


    def get_CAT_file_path(CAT_files_dir, cadaster_code, timeout=3600):
        CAT_file = None
        message_displayed = False  # Track whether the message has been displayed
        task_time = 0
        start_time = time.time()
        while CAT_file is None or task_time < timeout:
            try:
                CAT_files = os.listdir(f"{CAT_files_dir}")
                CAT_files = sorted(
                    CAT_files,
                    key=lambda i: f"{i[:5]}_{i[-8:-4]}{i[-10:-8]}{i[-12:-10]}",
                    reverse=True
                )
                CAT_files = [file for file in CAT_files if file.startswith(cadaster_code)]
                if len(CAT_files) > 0:
                    CAT_file = CAT_files[0]
                    return os.path.join(CAT_files_dir, CAT_file)
                else:
                    if not message_displayed:
                        sys.stderr.write(
                            f"\nPlease, upload the CAT file in {CAT_files_dir} for municipality {cadaster_code} (cadaster code). "
                            "\nYou can download them in subsets of provinces clicking in 'Downloads of alphanumeric information by province (CAT format)'"
                            "\nof the following website: https://www.sedecatastro.gob.es/Accesos/SECAccDescargaDatos.aspx"
                        )
                        sys.stderr.flush()
                        message_displayed = True
                    # Check again after a short delay
                    time.sleep(3)
                    task_time = time.time() - start_time
            except KeyboardInterrupt:
                sys.stderr.write("\nProcess interrupted by user. Exiting gracefully...\n")
                return None

        return CAT_file

    # Ensure directories exist
    os.makedirs(f"{CAT_files_dir}", exist_ok=True)
    os.makedirs(f"{CAT_files_dir}/parquet", exist_ok=True)

    inputfile = get_CAT_file_path(CAT_files_dir, cadaster_code)

    if inputfile is not None:
        combined_dfs = process_file_in_chunks(inputfile, CAT_files_dir, cadaster_code, allowed_dataset_types)
        return combined_dfs
