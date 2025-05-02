import copy
import sys
import os
import shutil
import fnmatch
import networkx as nx
import numpy as np
import pandas as pd
from zipfile import ZipFile, BadZipFile
import tarfile
import requests
from bs4 import BeautifulSoup
import rasterio
import geopandas as gpd
from geopandas import sjoin
from rasterio.merge import merge
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.ops import unary_union, nearest_points, linemerge, polygonize
from shapely.geometry.polygon import orient
from shapely.geometry import (Polygon, Point, LineString, MultiPolygon, MultiPoint, MultiLineString, GeometryCollection,
                              LinearRing, JOIN_STYLE)
from shapely.errors import GEOSException
import math
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib.externals.loky import get_reusable_executor
import re
import osmnx as ox
from shapely.geometry import mapping
import matplotlib.patches as patches
from shapely.geometry.base import BaseGeometry

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def cadaster_dir_(wd):
    return f"{wd}/cadaster"

def districts_dir_(wd):
    return f"{wd}/districts"

def census_tracts_dir_(wd):
    return f"{wd}/census_tracts"

def results_dir_(wd):
    return f"{wd}/results"

def open_street_dir_(wd):
    return f"{wd}/os_maps"

def DEM_raster_dir_(wd):
    return f"{wd}/DEM_rasters"

def postal_codes_dir_(wd):
    return f"{wd}/postal_codes"

def neighborhoods_dir_(wd):
    return f"{wd}/neighbourhoods"

def open_data_dir_(wd):
    return f"{wd}/open_data"

def create_dirs(data_dir):
    os.makedirs(census_tracts_dir_(data_dir), exist_ok=True)
    os.makedirs(districts_dir_(data_dir), exist_ok=True)
    os.makedirs(cadaster_dir_(data_dir), exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/buildings", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/buildings/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/buildings/unzip", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/address", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/address/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/address/unzip", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/parcels", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/parcels/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir_(data_dir)}/parcels/unzip", exist_ok=True)
    os.makedirs(results_dir_(data_dir), exist_ok=True)
    os.makedirs(DEM_raster_dir_(data_dir), exist_ok=True)
    os.makedirs(f"{DEM_raster_dir_(data_dir)}/raw", exist_ok=True)
    os.makedirs(f"{DEM_raster_dir_(data_dir)}/uncompressed", exist_ok=True)
    os.makedirs(neighborhoods_dir_(data_dir), exist_ok=True)
    os.makedirs(postal_codes_dir_(data_dir), exist_ok=True)
    os.makedirs(f"{postal_codes_dir_(data_dir)}/raw", exist_ok=True)
    os.makedirs(open_data_dir_(data_dir), exist_ok=True)

# PyCatastro.ConsultaMunicipio("BARCELONA")['consulta_municipalero'][]
# def dwellings_per_building_reference(province_name, municipality_name, building_reference):
#     dnprc_result = PyCatastro.Consulta_DNPRC(province_name, municipality_name, building_reference)
#     dnprc_df = pd.DataFrame(
#         [item["dt"]["locs"]["lous"]["lourb"]["loint"] for item in
#          dnprc_result['consulta_dnp']['lrcdnp']['rcdnp']])
#     dnprc_df["pt"].sort_values()


def list_municipalities(province_codes=None,
                        inspire_url="https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
                        echo=True):
    response = requests.get(inspire_url)
    soup = BeautifulSoup(response.content, "xml")
    municipalities = soup.find_all("div", id='scrolltable')

    urls = [x.get('href') for x in soup.find_all("link", rel="enclosure")]
    list_municipalities = []
    for j in range(len(municipalities)):
        x = municipalities[j]
        url = urls[j]

        if province_codes is None or url.split('/')[5] in province_codes:
            if echo:
                sys.stdout.write('Downloading province: %s\n' % (url.split('/')[5]))
            # Obtain the municipality name
            x = copy.deepcopy(x)
            x = [line.strip() for line in x.get_text(separator='\n').strip().replace("\t", "")
                .replace("\r", "").replace(' ', '').replace('\n\n','\n')
                .split('\n') if line.strip()]
            x = copy.deepcopy(x)
            z = []
            for y in x:
                if y:
                    z.append(y)
            z.sort()
            # Obtain the URL's
            url_soup = BeautifulSoup(requests.get(url).content, "xml")
            municip_urls = [x.get('href') for x in url_soup.find_all("link", rel="enclosure")]
            municip_urls = [x for _, x in sorted(zip([y[50:56] for y in municip_urls], municip_urls))]
            # Extend the list of municipalities
            for i in range(len(z)):
                list_municipalities.append(
                    {
                        "name": z[i],
                        "url": municip_urls[i]
                    })
    return list_municipalities


def ine_to_cadaster_codes(cadaster_dir, ine_codes):
    if ine_codes is not None:
        map_code_dict = pd.read_excel(f"{cadaster_dir}/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl').set_index(
            'CÓDIGO INE').to_dict()['CÓDIGO CATASTRO']
        cadaster_codes = [map_code_dict[key] for key in ine_codes]
    else:
        cadaster_codes = None
    return cadaster_codes

def cadaster_to_ine_codes(cadaster_dir, cadaster_codes):
    if cadaster_codes is not None:
        map_code_dict = pd.read_excel(f"{cadaster_dir}/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl').set_index(
            'CÓDIGO CATASTRO').to_dict()['CÓDIGO INE']
        ine_codes = [map_code_dict[key] for key in cadaster_codes]
    else:
        ine_codes = None
    return ine_codes

def municipality_name(ine_code):

    df = pd.read_excel("https://www.ine.es/daco/daco42/codmun/diccionario24.xlsx", header=1)
    df['Municipality code'] = df['CPRO'].astype(int).apply(lambda x: f"{x:02d}") +\
                              df['CMUN'].astype(int).apply(lambda x: f"{x:03d}")
    df.rename(columns = {'NOMBRE':'Municipality name'}, inplace=True)
    df.drop(columns = ["CODAUTO","CPRO","CMUN","DC"],inplace=True)

    return list(df.loc[df["Municipality code"]==ine_code ,"Municipality name"])[0]

def cadaster_to_ine_codes(cadaster_dir, cadaster_codes):
    if cadaster_codes is not None:
        map_code_dict = pd.read_excel(f"{cadaster_dir}/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl').set_index(
            'CÓDIGO CATASTRO').to_dict()['CÓDIGO INE']
        ine_codes = [map_code_dict[key] for key in cadaster_codes]
    else:
        ine_codes = None
    return ine_codes

def get_administrative_divisions_naming(cadaster_dir, cadaster_codes):
    municipalities = pd.read_excel(f"{cadaster_dir}/ine_inspire_codes.xlsx", dtype=object, engine='openpyxl')
    municipalities.drop(['INMUEBLES TOTALES', 'INMUEBLES URBANOS', 'INMUEBLES RÚSTICOS',
                         'Regularizados', 'Regularizados\nurbanos', 'Regularizados\nrústicos',
                         'Nuevas Construcciones', 'Ampliaciones y Rehabilitaciones',
                         'Reformas y Cambios de Uso', 'Piscinas', 'FECHA FIN PROCESO'], inplace=True, axis=1)
    municipalities.rename(columns={
        'CÓDIGO INE': 'ine_code',
        'CÓDIGO CATASTRO': 'cadaster_code',
        'MUNICIPIO': 'municipality_name',
        'PROVINCIA': 'province_name',
        'COMUNIDAD AUTONOMA': 'autonomous_community_name'
        }, inplace=True)
    municipalities = municipalities[municipalities['cadaster_code'].apply(lambda x: x in cadaster_codes)]

    return municipalities


def unzip_directory(zip_directory, unzip_directory):
    for file in os.listdir(zip_directory):
        if file.endswith(".zip"):
            try:
                with ZipFile(f"{zip_directory}{file}", 'r') as zip:
                    zip.extractall(unzip_directory)
            except BadZipFile:
                os.remove(f"{zip_directory}{file}")


def untar_directory(tar_directory, untar_directory, files_to_extract):

    # Create the extraction directory if it doesn't exist, or remove all the files
    if not os.path.exists(untar_directory):
        os.makedirs(untar_directory)
    else:
        shutil.rmtree(untar_directory)
        os.makedirs(untar_directory)

    for file in os.listdir(tar_directory):

        try:
            # Determine the mode based on file extension
            if tar_directory.endswith(".tar.gz") or tar_directory.endswith(".tgz"):
                mode = 'r:gz'
            elif tar_directory.endswith(".tar.bz2") or tar_directory.endswith(".tbz"):
                mode = 'r:bz2'
            else:
                mode = 'r'

            # Open the tar file
            with tarfile.open(f"{tar_directory}{file}", mode) as tar:

                # Initialize a counter to create unique filenames
                counter = 0

                # Find and extract members that match the pattern
                for member in tar.getmembers():

                    newfile = file.replace(file.split(".")[-1], member.name.split(".")[-1])

                    if fnmatch.fnmatch(member.name, files_to_extract):

                        # Define the full path for the extracted file
                        extract_full_path = os.path.join(untar_directory, f"{'' if counter == 0 else str(counter)}{newfile}")

                        # Extract the file content to a temporary location
                        extracted_file = tar.extractfile(member)
                        with open(extract_full_path, 'wb') as out_file:
                            out_file.write(extracted_file.read())

                        counter += 1

        except Exception:
            os.remove(f"{tar_directory}{file}")


def get_bbox(gdf):
    bbox_gdf = gdf.bounds
    lat_min, lon_min, lat_max, lon_max = (
        min(bbox_gdf['minx']), min(bbox_gdf['miny']),
        max(bbox_gdf['maxx']), max(bbox_gdf['maxy']),
    )
    return [lat_min, lon_min, lat_max, lon_max]


def concatenate_tiffs(input_dir, output_file):
    # Open the files using rasterio
    src_files_to_mosaic = []
    for fp in os.listdir(input_dir):
        src = rasterio.open(f"{input_dir}{fp}")
        src_files_to_mosaic.append(src)

    # Merge the rasters
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Update the metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]
    })

    # Write the mosaic raster to a new file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all input files
    for src in src_files_to_mosaic:
        src.close()


def create_graph(gdf, geometry_name="building_part_geometry", buffer = 0.5):
    G = nx.Graph()
    G.add_nodes_from(gdf.index)

    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    gdf = gdf[[geometry_name]].copy()
    gdf['id'] = gdf.index
    gdf = gdf.set_geometry(geometry_name)
    buffered_gdf = gdf.copy()
    buffered_gdf[geometry_name] = gdf[geometry_name].buffer(buffer)
    # Perform a spatial join to find nearest neighbors
    joined = sjoin(buffered_gdf, gdf, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

    # Create edges between nearest neighbors
    edges = [(row['id_left'], row['id_right']) for _, row in joined.iterrows()]
    G.add_edges_from(edges)

    return G

def detect_close_buildings(gdf_building_parts, buffer_neighbours, neighbours_column_name,
                           neighbours_id_column_name = "single_building_reference",
                           geometry_variable="building_part_geometry"):
    G = create_graph(gdf_building_parts, geometry_name=geometry_variable, buffer = buffer_neighbours)
    clusters = list(nx.connected_components(G))
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
    if neighbours_id_column_name == "single_building_reference":
        gdf_building_parts['cluster_id'] = gdf_building_parts.index.map(cluster_map)
        gdf_building_parts['single_building_reference'] = gdf_building_parts['building_reference'].str.cat(gdf_building_parts['cluster_id'].astype(str), sep='_')

    # Update graph nodes to include the single_building_reference attribute
    for idx, row in gdf_building_parts.iterrows():
        G.nodes[idx][neighbours_id_column_name] = row[neighbours_id_column_name]

    # Create a dictionary to store related single_building_references
    related_buildings_map = {}
    for node in G.nodes:
        single_building_reference = G.nodes[node][neighbours_id_column_name]
        connected_references = {G.nodes[neighbor][neighbours_id_column_name] for neighbor in G.neighbors(node)}

        if single_building_reference not in related_buildings_map:
            related_buildings_map[single_building_reference] = connected_references
        else:
            related_buildings_map[single_building_reference].update(connected_references)

        # Ensure each single_building_reference includes itself in its related buildings
        related_buildings_map[single_building_reference].add(single_building_reference)

    # Convert sets to sorted comma-separated strings for display or further processing
    gdf_building_parts[neighbours_column_name] = gdf_building_parts[neighbours_id_column_name].map(lambda x: ','.join(sorted(related_buildings_map[x])))

    return gdf_building_parts


def detect_close_geometries_chunk(chunk, buffer_neighbours, neighbours_column_name, neighbours_id_column_name,
                                  geometry_variable):
    G = create_graph(chunk, geometry_name=geometry_variable, buffer=buffer_neighbours)
    clusters = list(nx.connected_components(G))
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}

    if neighbours_id_column_name == "single_building_reference":
        chunk['cluster_id'] = chunk.index.map(cluster_map)
        chunk['single_building_reference'] = chunk['building_reference'].str.cat(chunk['cluster_id'].astype(str),
                                                                                 sep='_')

    # Update graph nodes to include the single_building_reference attribute
    for idx, row in chunk.iterrows():
        G.nodes[idx][neighbours_id_column_name] = row[neighbours_id_column_name]

    # Create a dictionary to store related single_building_references
    related_buildings_map = {}
    for node in G.nodes:
        single_building_reference = G.nodes[node][neighbours_id_column_name]
        connected_references = {G.nodes[neighbor][neighbours_id_column_name] for neighbor in G.neighbors(node)}

        if single_building_reference not in related_buildings_map:
            related_buildings_map[single_building_reference] = connected_references
        else:
            related_buildings_map[single_building_reference].update(connected_references)

        # Ensure each single_building_reference includes itself in its related buildings
        related_buildings_map[single_building_reference].add(single_building_reference)

    # Convert sets to sorted comma-separated strings for display or further processing
    chunk = chunk[chunk["buffered"]].copy()
    chunk[neighbours_column_name] = chunk[neighbours_id_column_name].map(
        lambda x: ','.join(sorted(related_buildings_map[x])))

    return chunk


def detect_close_buildings_parallel(gdf_building_parts, buffer_neighbours, neighbours_column_name,
                          neighbours_id_column_name="single_building_reference", num_workers=4, column_name_to_split=None):
    # Split the data into chunks for parallel processing
    if column_name_to_split is not None:

        if isinstance(gdf_building_parts,gpd.GeoDataFrame):
            gdf_building_parts = gpd.GeoDataFrame(gdf_building_parts)

        gdf_building_parts = gdf_building_parts.set_geometry("building_part_geometry")
        gdf_building_parts["centroid"] = gdf_building_parts.centroid

        # Step 1: Group by the specified column
        groups = gdf_building_parts.groupby(column_name_to_split)

        # Step 2: Create a list to store the chunks with buffer information
        chunks = []

        for unique_value, group in tqdm(groups, desc="Creating the chunks to parallelise the detection of close buildings..."):
            # Step 3: Create a GeoSeries of buffered geometries around each point/geometry in the group
            group_buffer = MultiPoint(list(group["centroid"])).convex_hull.buffer(buffer_neighbours)

            # Step 4: Find all geometries in the original DataFrame that intersect with this buffered geometry
            all_within_buffer = gdf_building_parts[gdf_building_parts["centroid"].within(group_buffer)].copy()

            # Step 5: Add an indicator column
            # Mark as "False" if in the original subset, "True" if in the buffer zone
            all_within_buffer['buffered'] = all_within_buffer.index.isin(group.index)

            # Append this chunk to the chunks list
            chunks.append(all_within_buffer)
    else:
        chunks = np.array_split(gdf_building_parts, num_workers)

    with tqdm_joblib(tqdm(desc="Detect buildings related with others (Nearby, adjacent... depending on the buffer)",
                          total=len(chunks))):
        results = Parallel(n_jobs=num_workers)(
            delayed(detect_close_geometries_chunk)(chunk, buffer_neighbours, neighbours_column_name,
                                                  neighbours_id_column_name, geometry_variable = "building_part_geometry")
            for chunk in chunks
        )
    get_reusable_executor().shutdown(wait=True)

    # Concatenate the results into a single GeoDataFrame
    gdf_building_parts_final = pd.concat(results, ignore_index=True)

    return gdf_building_parts_final

def detect_close_parcels_parallel(gdf_parcels, buffer_neighbours, neighbours_column_name,
                          neighbours_id_column_name="single_building_reference", num_workers=4, column_name_to_split=None):
    # Split the data into chunks for parallel processing
    if column_name_to_split is not None:

        # Step 1: Group by the specified column
        groups = gdf_parcels.groupby(column_name_to_split)

        # Step 2: Create a list to store the chunks with buffer information
        chunks = []

        for unique_value, group in tqdm(groups, desc="Creating the chunks to parallelise the detection of close parcels..."):
            # Step 3: Create a GeoSeries of buffered geometries around each point/geometry in the group
            group_buffer = MultiPoint(list(group["parcel_centroid"])).convex_hull.buffer(buffer_neighbours)

            # Step 4: Find all geometries in the original DataFrame that intersect with this buffered geometry
            all_within_buffer = gdf_parcels[gdf_parcels["parcel_centroid"].within(group_buffer)].copy()

            # Step 5: Add an indicator column
            # Mark as "False" if in the original subset, "True" if in the buffer zone
            all_within_buffer['buffered'] = all_within_buffer.index.isin(group.index)

            # Append this chunk to the chunks list
            chunks.append(all_within_buffer)
    else:
        chunks = np.array_split(gdf_parcels, num_workers)

    with tqdm_joblib(tqdm(desc="Detect buildings related with others (Nearby, adjacent... depending on the buffer)",
                          total=len(chunks))):
        results = Parallel(n_jobs=num_workers)(
            delayed(detect_close_geometries_chunk)(
                chunk, buffer_neighbours, neighbours_column_name, neighbours_id_column_name,
                geometry_variable = "parcel_geometry")
            for chunk in chunks
        )
    get_reusable_executor().shutdown(wait=True)

    # Concatenate the results into a single GeoDataFrame
    gdf_parcels_final = pd.concat(results, ignore_index=True)

    return gdf_parcels_final

def union_geoseries_with_tolerance(geometries, gap_tolerance=1e-6, resolution=16):
    """
    Unions a GeoSeries with a specified tolerance to fill small gaps between geometries.

    Parameters:
    - geometries (GeoSeries): A GeoSeries containing the geometries to union.
    - gap_tolerance (float): The tolerance used to fill small gaps between geometries. Default is 1e-6.
    - resolution (int): The resolution of the buffer operation. Higher values result in more detailed buffering.

    Returns:
    - unioned_geometry (Geometry): A single unified geometry after applying the tolerance.
    """

    # Step 1: Perform a unary union on all geometries
    unioned_geometry = unary_union(geometries)

    # Step 2: Buffer by a small negative amount and then positive to fill gaps
    unioned_geometry = unioned_geometry.buffer(gap_tolerance, resolution=resolution,
                                               join_style=JOIN_STYLE.mitre).buffer(
        -gap_tolerance, resolution=resolution, join_style=JOIN_STYLE.mitre)

    # Step 3: Perform the union again if needed
    unioned_geometry = unary_union(unioned_geometry)

    return unioned_geometry

def calculate_floor_footprints_chunk(chunk, gdf, group_by, only_exterior_geometry, min_hole_area, gap_tolerance,
                                     by_floors):
    """
    Process a chunk of groups to calculate the 2D footprint geometry for each floor level.
    """
    floor_footprints = []

    for group in chunk:
        # Filter the grouped buildings
        gdf_ = gdf[gdf[group_by] == group] if group_by else gdf

        if by_floors==True:
            # Unique levels of floors
            max_floor = int(gdf_['n_floors_above_ground'].dropna().max())
            min_floor = int(gdf_['n_floors_below_ground'].dropna().max())
            unique_floors = list(range(max_floor))
            unique_underground_floors = list(range(1,min_floor+1))

            for floor in unique_floors:
                floor_geometries = gdf_[gdf_['n_floors_above_ground'] >= (floor + 1)].reset_index(drop=True)
                try:
                    unioned_geometry = union_geoseries_with_tolerance(floor_geometries.geometry,
                                                                      gap_tolerance=gap_tolerance, resolution=16)
                except GEOSException:
                    unioned_geometry = union_geoseries_with_tolerance(floor_geometries.geometry,
                                                                      gap_tolerance=0.1, resolution=16)
                # Handle geometry based on `only_exterior_geometry` and `min_hole_area`
                if only_exterior_geometry:
                    if isinstance(unioned_geometry, Polygon):
                        unioned_geometry = Polygon(unioned_geometry.exterior)
                    elif isinstance(unioned_geometry, MultiPolygon):
                        unioned_geometry = MultiPolygon([Polygon(poly.exterior) for poly in unioned_geometry.geoms])
                else:
                    if isinstance(unioned_geometry, Polygon):
                        cleaned_interiors = [interior for interior in unioned_geometry.interiors if
                                             Polygon(interior).area >= min_hole_area]
                        unioned_geometry = Polygon(unioned_geometry.exterior, cleaned_interiors)
                    elif isinstance(unioned_geometry, MultiPolygon):
                        cleaned_polygons = []
                        for poly in unioned_geometry.geoms:
                            cleaned_interiors = [interior for interior in poly.interiors if
                                                 Polygon(interior).area >= min_hole_area]
                            cleaned_polygons.append(Polygon(poly.exterior, cleaned_interiors))
                        unioned_geometry = MultiPolygon(cleaned_polygons)

                floor_footprints.append({
                    'group': group,
                    'floor': floor,
                    'geometry': unioned_geometry
                })

            if gdf_['n_floors_below_ground'].max() > 0:
                for floor in unique_underground_floors:
                    floor_geometries = gdf_[gdf_['n_floors_below_ground'] >= (floor)].reset_index(drop=True)
                    unioned_geometry = union_geoseries_with_tolerance(floor_geometries.geometry, gap_tolerance=gap_tolerance, resolution=16)

                    # Handle geometry based on `only_exterior_geometry` and `min_hole_area`
                    if only_exterior_geometry:
                        if isinstance(unioned_geometry, Polygon):
                            unioned_geometry = Polygon(unioned_geometry.exterior)
                        elif isinstance(unioned_geometry, MultiPolygon):
                            unioned_geometry = MultiPolygon([Polygon(poly.exterior) for poly in unioned_geometry.geoms])
                    else:
                        if isinstance(unioned_geometry, Polygon):
                            cleaned_interiors = [interior for interior in unioned_geometry.interiors if
                                                 Polygon(interior).area >= min_hole_area]
                            unioned_geometry = Polygon(unioned_geometry.exterior, cleaned_interiors)
                        elif isinstance(unioned_geometry, MultiPolygon):
                            cleaned_polygons = []
                            for poly in unioned_geometry.geoms:
                                cleaned_interiors = [interior for interior in poly.interiors if
                                                     Polygon(interior).area >= min_hole_area]
                                cleaned_polygons.append(Polygon(poly.exterior, cleaned_interiors))
                            unioned_geometry = MultiPolygon(cleaned_polygons)

                    floor_footprints.append({
                        'group': group,
                        'floor': -floor,
                        'geometry': unioned_geometry
                    })
        else:
            floor_geometries = gdf_[gdf_['n_floors_above_ground'] == 1].reset_index(drop=True)
            unioned_geometry = union_geoseries_with_tolerance(floor_geometries.geometry, gap_tolerance=gap_tolerance,
                                                              resolution=16)

            # Handle geometry based on `only_exterior_geometry` and `min_hole_area`
            if only_exterior_geometry:
                if isinstance(unioned_geometry, Polygon):
                    unioned_geometry = Polygon(unioned_geometry.exterior)
                elif isinstance(unioned_geometry, MultiPolygon):
                    unioned_geometry = MultiPolygon([Polygon(poly.exterior) for poly in unioned_geometry.geoms])
            else:
                if isinstance(unioned_geometry, Polygon):
                    cleaned_interiors = [interior for interior in unioned_geometry.interiors if
                                         Polygon(interior).area >= min_hole_area]
                    unioned_geometry = Polygon(unioned_geometry.exterior, cleaned_interiors)
                elif isinstance(unioned_geometry, MultiPolygon):
                    cleaned_polygons = []
                    for poly in unioned_geometry.geoms:
                        cleaned_interiors = [interior for interior in poly.interiors if
                                             Polygon(interior).area >= min_hole_area]
                        cleaned_polygons.append(Polygon(poly.exterior, cleaned_interiors))
                    unioned_geometry = MultiPolygon(cleaned_polygons)

            floor_footprints.append({
                'group': group,
                'geometry': unioned_geometry
            })

    sys.stderr.flush()

    return floor_footprints

def calculate_floor_footprints(gdf, group_by=None, geometry_name="geometry", only_exterior_geometry=False,
                               min_hole_area=1e-6, gap_tolerance=1e-6, chunk_size=200, num_workers=-1,
                               by_floors=True):
    """
    Generate the 2D footprint geometry for each floor level by merging overlapping geometries.
    """
    gdf = gdf.set_geometry(geometry_name)

    # Determine unique groups and create chunks
    unique_groups = gdf[group_by].unique() if group_by else ['all']
    chunks = np.array_split(unique_groups, len(unique_groups) // chunk_size + 1)

    if len(chunks)>2 and (num_workers==-1 or num_workers>1):
        # Parallel processing of each chunk
        with tqdm_joblib(tqdm(desc="Processing floor above/below ground footprints...",
                              total=len(chunks))):
            results = Parallel(n_jobs=num_workers)(
                delayed(calculate_floor_footprints_chunk)(chunk, gdf, group_by, only_exterior_geometry, min_hole_area,
                                                          gap_tolerance, by_floors)
                for chunk in chunks
            )
        get_reusable_executor().shutdown(wait=True)

        # Flatten the list of results and create the final GeoDataFrame
        floor_footprints = [item for sublist in results for item in sublist]
        floor_footprints_gdf = gpd.GeoDataFrame(floor_footprints, crs=gdf.crs)
    else:
        floor_footprints_gdf = gpd.GeoDataFrame()
        for i in range(len(chunks)):
            chunk = chunks[i]
            floor_footprints_gdf = pd.concat([
                floor_footprints_gdf,
                gpd.GeoDataFrame(
                    calculate_floor_footprints_chunk(chunk, gdf, group_by, only_exterior_geometry,
                                                     min_hole_area, gap_tolerance, by_floors),
                crs=gdf.crs)])

    return floor_footprints_gdf

def get_all_patios(geoseries):
    interiors = []

    for geom in geoseries:
        if geom is None:
            continue

        if isinstance(geom, Polygon):
            # If it's a Polygon, add its interiors directly
            interiors.extend(list(geom.interiors))

        elif isinstance(geom, MultiPolygon):
            # If it's a MultiPolygon, add the interiors of each Polygon
            for poly in geom.geoms:
                interiors.extend(list(poly.interiors))

    polygons = []

    # Step 1: Convert LinearRings to Polygons
    for ring in interiors:
        polygon = Polygon(ring)
        # Normalize the orientation (optional, but helps with consistency)
        polygon = orient(polygon, sign=1.0)  # Ensure all polygons are oriented clockwise
        # Normalize the starting point of the polygon
        polygon = normalize_polygon(polygon)
        polygons.append(polygon)

    return polygons



def patios_in_the_building(patios_geoms, building_geom, tolerance=0.5):
    """
    Check if patios are nearly totally inside a building.

    Parameters:
    patios_geoms (list): A list of Shapely Polygon objects representing small polygons.
    building_geom (Polygon): A Shapely Polygon object representing the main polygon.
    tolerance (float): The fraction of the small polygon's area that must intersect with the main polygon's area.

    Returns:
    list of bool: A list indicating whether each small polygon is nearly totally inside the main polygon.
    """
    results = []

    for patio_geom in patios_geoms:
        if not patio_geom.is_valid:
            patio_geom = patio_geom.buffer(0)
        if not building_geom.is_valid:
            building_geom = building_geom.buffer(0)
        intersection_area = patio_geom.buffer(0.5, cap_style=2, join_style=3).intersection(building_geom).area

        # Check if the intersection area is greater than or equal to the tolerance threshold
        if intersection_area / patio_geom.area >= tolerance:
            results.append(patio_geom)

    return results


def remove_duplicate_points(coords):
    """
    Remove consecutive duplicate points from a list of coordinates.

    Parameters:
    coords (list): A list of (x, y) tuples representing the coordinates.

    Returns:
    list: A list of coordinates with consecutive duplicates removed.
    """
    if not coords:
        return coords

    unique_coords = [coords[0]]
    for coord in coords[1:]:
        if coord != unique_coords[-1]:
            unique_coords.append(coord)

    # Ensure the polygon closes correctly by repeating the first point at the end
    if unique_coords[0] != unique_coords[-1]:
        unique_coords.append(unique_coords[0])

    return unique_coords


def normalize_polygon(polygon):
    """
    Normalize a polygon so that it starts from the lowest point (first in lexicographical order).
    Remove any duplicate consecutive points.
    """

    # Get the exterior coordinates of the polygon
    exterior_coords = remove_duplicate_points(list(polygon.exterior.coords))

    # Find the index of the lexicographically smallest point
    min_index = min(range(len(exterior_coords)), key=exterior_coords.__getitem__)

    # Rotate the exterior coordinates so the polygon starts from the smallest point
    exterior_coords = exterior_coords[min_index:] + exterior_coords[:min_index]

    # Get the exterior coordinates of the polygon
    exterior_coords = remove_duplicate_points(list(exterior_coords))

    # Process the interior rings (holes) to remove duplicates
    interiors_coords = [remove_duplicate_points(list(interior.coords)) for interior in polygon.interiors]

    # Recreate the polygon with the normalized exterior and cleaned interiors
    return Polygon(exterior_coords, holes=interiors_coords)


def unique_polygons(polygons, tolerance=0.1):
    """
    Return a list of unique or extremely similar polygons.

    Parameters:
    polygons (list): A list of Polygon objects.
    tolerance (float): Tolerance for comparing polygons to determine similarity.

    Returns:
    unique_polygons (list): A list of unique or extremely similar Polygon objects.
    """

    # Step 2: Deduplicate polygons based on a spatial measure like area
    unique_polygons = []
    seen_polygons = []

    for polygon in polygons:
        is_unique = True
        for seen in seen_polygons:
            # Check if the polygons are extremely similar
            if polygon.equals_exact(seen, tolerance):
                is_unique = False
                break
        if is_unique:
            unique_polygons.append(polygon)
            seen_polygons.append(polygon)

    return unique_polygons

#
# cardinal_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'SWW', 'W', 'WNW', 'NW', 'NNW']
# def calculate_orientation(angle):
#     """
#     Discretize an angle into one of the cardinal directions.
#
#     Parameters:
#     angle (float): Angle in degrees.
#
#     Returns:
#     str: The discretized direction (e.g., 'N', 'NE', 'E', etc.).
#     """
#
#     bin_size = 360 / len(cardinal_directions)
#     index = int((angle + bin_size / 2) % 360 / bin_size)
#     return cardinal_directions[index]


def calculate_wall_outdoor_normal_orientation(segment, orientation_interval=None):
    """
    Calculate the orientation of a wall.

    Parameters:
    segment (LineString): A Shapely LineString representing a segment of a polygon's exterior.

    Returns:
    float: Normal orientation of the wall (north is 0, east is 90, south is 180, and west is 270)
    """
    # Extract the coordinates
    x1, y1 = segment.coords[0]
    x2, y2 = segment.coords[1]

    # Calculate the direction vector (dx, dy)
    dx = x2 - x1
    dy = y1 - y2

    # Calculate the angle using atan2 (in radians)
    angle_radians = math.atan2(dy, dx)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    if orientation_interval is not None:
        index = int((angle_degrees + orientation_interval / 2) % 360 / orientation_interval)
        angle_degrees = list(range(0,360,orientation_interval))[index]

    return angle_degrees

def create_ray_from_centroid(centroid, angle, length=1000):
    angle_rad = math.radians(angle)
    x = centroid.x + length * math.sin(angle_rad)
    y = centroid.y + length * math.cos(angle_rad)
    return LineString([centroid, Point(x, y)])

# Function to convert MultiPolygon to a list of Polygons, or leave a Polygon unchanged
def convert_to_polygons(geometry):
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry]
    elif isinstance(geometry, Polygon):
        return [geometry]
    else:
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")

# Function to convert a geometry into one or more LinearRings
def convert_to_linearrings(geom):
    if isinstance(geom, LinearRing):
        return geom
    if isinstance(geom, LineString):
        return [geom]
    elif isinstance(geom, Polygon):
        return [LinearRing(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        return [LinearRing(p.exterior.coords) for p in geom.geoms]
    else:
        raise ValueError("Geometry must be a Polygon, MultiPolygon, LinearRing, or LineString")

# Function to convert a geometry into one or more LineStrings
def convert_to_linestrings(geom):
    if isinstance(geom, LineString):
        return [geom]

    elif isinstance(geom, LinearRing):
        return [LineString(geom)]

    elif isinstance(geom, Polygon):
        exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
        return [LineString(exterior_coords)]

    elif isinstance(geom, MultiPolygon):
        return [
            LineString(orient(p, sign=-1.0).exterior.coords)
            for p in geom.geoms
        ]

    else:
        raise ValueError("Geometry must be a LineString, LinearRing, Polygon, or MultiPolygon")

def segment_intersects_with_tolerance(segment, target_geom, buffer_distance=0.1, area_percentage_threshold=20.0):
    """
    Check if a segment intersects with a target geometry using a buffer and an area threshold.

    Parameters:
    segment (LineString): The line segment to check.
    target_geom (Polygon or MultiPolygon): The target geometry to check against.
    buffer_distance (float): The distance to buffer the segment.
    area_percentage_threshold (float): The minimum percentage of the buffered segment intersecting the target_geom.

    Returns:
    bool: True if the segment intersects the target geometry with an area above the threshold.
    """
    buffered_segment = segment.buffer(buffer_distance, cap_style="flat")
    intersection = buffered_segment.intersection(target_geom)

    # Check if the intersection area is above the threshold
    return intersection.area > (buffered_segment.area * area_percentage_threshold/100)


def get_furthest_point(multipoint, reference_point):
    """
    Get the furthest point from a MultiPoint object considering another reference point.

    Parameters:
    multipoint (MultiPoint): A shapely MultiPoint object.
    reference_point (Point): A shapely Point object to measure the distance from.

    Returns:
    Point: The furthest point from the reference point.
    """
    max_distance = 0
    furthest_point = None

    for geom in multipoint if isinstance(multipoint, list) or isinstance(multipoint, np.ndarray) else [multipoint]:
        if geom.is_empty:
            continue
        if isinstance(geom, Point):
            points = [geom]
        elif isinstance(geom, MultiPoint):
            points = list(geom.geoms)
        else:
            continue  # Skip non-point geometries like LineString

        for point in points:
            distance = point.distance(reference_point)
            if distance > max_distance:
                max_distance = distance
                furthest_point = point

    return furthest_point



def generate_random_interior_points_from_geometry(geometry, num_points, force_projection=True):
    """
    Generate random points inside or on the geometry for LinearRing, MultiLineString, Polygon, or MultiPolygon.

    Parameters:
    geometry (LinearRing, MultiLineString, Polygon, MultiPolygon): The geometry within which to generate points.
    num_points (int): The number of random points to generate.
    force_projection (bool): If True, random points are projected onto the nearest point on lines/rings.

    Returns:
    list: A list of Point objects inside or on the geometry.
    """
    points = []

    # Decompose geometry into individual geometries for iteration
    if isinstance(geometry, (MultiPolygon, MultiLineString)):
        geometries = list(geometry.geoms)
    else:
        geometries = [geometry]

    while len(points) < num_points:
        selected_geometry = np.random.choice(geometries)

        if isinstance(selected_geometry, Polygon):
            minx, miny, maxx, maxy = selected_geometry.bounds
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

            if selected_geometry.contains(random_point):
                points.append(random_point)

        elif isinstance(selected_geometry, LinearRing):
            minx, miny, maxx, maxy = selected_geometry.bounds
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

            if force_projection:
                random_point = selected_geometry.interpolate(selected_geometry.project(random_point))
            points.append(random_point)

        elif isinstance(selected_geometry, (LineString, MultiLineString)):
            # Handle MultiLineString by merging into a single LineString
            if isinstance(selected_geometry, MultiLineString):
                selected_geometry = linemerge(selected_geometry)

            distance_along_line = np.random.uniform(0, selected_geometry.length)
            random_point = selected_geometry.interpolate(distance_along_line)
            points.append(random_point)

    return points

def distance_from_points_to_polygons_by_orientation(geom, other_polygons, num_points_per_geom_element=10,
                                                    orientation_interval=5, orientations_to_analyse=None,
                                                    force_one_orientation_per_geom_element=False,
                                                    plots=False, pdf=None, max_distance=150):
    if not isinstance(geom, list):
        geom = [geom]
    if orientations_to_analyse is not None:
        orientations = [int(float(i)) for i in orientations_to_analyse]
    else:
        orientations = list(range(0, 360, orientation_interval))
    aggregated_distances_by_orientation = {str(i): [] for i in orientations}
    final_distances_by_orientation = {str(i): np.inf for i in orientations}
    representativity_by_orientation = {str(i): 0.0 for i in orientations}
    other_pol_intersection_points = {str(i): [] for i in orientations}
    further_point_geoms = {str(i): [] for i in orientations}
    points = {str(i): [] for i in orientations}
    elements = {str(i): 0 for i in orientations}

    for geom_element in geom:
        orientations_element = orientations
        if force_one_orientation_per_geom_element:
            orientations_element = [calculate_wall_outdoor_normal_orientation(geom_element, orientation_interval=orientation_interval)]
        random_points = generate_random_interior_points_from_geometry(geom_element, num_points_per_geom_element, force_projection=False)

        for angle in orientations_element:
            elements[str(orientations_element[0])] = elements[str(orientations_element[0])] + 1
            for point in random_points:

                distances_by_orientation = {str(i): np.inf for i in orientations_element}

                ray = create_ray_from_centroid(point, angle, length=max_distance)
                further_point_geom = ray.intersection(geom_element)
                further_point_geom = get_furthest_point(further_point_geom, point) if not (
                    isinstance(further_point_geom, Point)) else further_point_geom
                if further_point_geom is None:
                    further_point_geom = point
                point_to_linearring_distance = point.distance(further_point_geom)

                nearest_point_ = point
                for other_polygon in other_polygons:
                    polygons = other_polygon.geoms if isinstance(other_polygon, MultiPolygon) else [other_polygon]
                    for polygon in polygons:
                        intersection = ray.intersection(polygon)
                        if not intersection.is_empty:
                            nearest_point = nearest_points(point, intersection)[1]
                            distance = round(point.distance(nearest_point) - point_to_linearring_distance, 2)
                            if distance < 0.05:
                                distance = 0.0
                            if distance <= distances_by_orientation[str(angle)]:
                                nearest_point_ = nearest_point
                                distances_by_orientation[str(angle)] = distance
                other_pol_intersection_points[str(angle)].append(nearest_point_)
                further_point_geoms[str(angle)].append(further_point_geom)
                points[str(angle)].append(point)

                aggregated_distances_by_orientation[str(angle)].append(distances_by_orientation[str(angle)])

    for angle in orientations:

        # Aggregate the distances by orientation
        final_distances_by_orientation[str(angle)] = (
            round(np.mean([v for v in aggregated_distances_by_orientation[str(angle)] if v < np.inf]), 2)) if (
            any(v < np.inf for v in aggregated_distances_by_orientation[str(angle)])) else np.inf
        representativity_by_orientation[str(angle)] = (
            len([v for v in aggregated_distances_by_orientation[str(angle)] if v < np.inf]) / (
            elements[str(angle)]*num_points_per_geom_element if force_one_orientation_per_geom_element else
            len(geom)*num_points_per_geom_element)
        ) if (any(v < np.inf for v in aggregated_distances_by_orientation[str(angle)])) else 0.0

        mask = [not i.is_empty for i in other_pol_intersection_points[str(angle)]]
        further_point_geoms[str(angle)] = [
            geom for geom, keep in zip(further_point_geoms[str(angle)], mask) if keep
        ]
        other_pol_intersection_points[str(angle)] = [
            geom for geom, keep in zip(other_pol_intersection_points[str(angle)], mask) if keep
        ]
        if plots and len(other_pol_intersection_points[str(angle)])>0 and pdf is not None:
            fig, ax = plt.subplots()
            plot_shapely_geometries(
                geometries=list(other_polygons) + [
                    LineString([p1, p2]) for p1, p2 in zip(further_point_geoms[str(angle)],
                                                       other_pol_intersection_points[str(angle)])] + [
                    points[str(angle)]],
                contextual_geometry=Polygon(geom) if not isinstance(geom, list) else geom,
                title=f"Orientation: {angle}º,mean distance: "
                      f"{str(round(final_distances_by_orientation[str(angle)],2))}m\n, representativity: "
                      f"{str(round(representativity_by_orientation[str(angle)] * 100, 2))}%",
                ax=ax
            )
            pdf.savefig(fig)
            plt.close(fig)

    return final_distances_by_orientation, representativity_by_orientation

def distance_from_centroid_to_polygons_by_orientation(geom, other_polygons, centroid, orientation_interval=5,
                                                    plots=False, pdf=None, floor=None, max_distance=150):
    direction_angles = list(range(0, 360, orientation_interval))
    distances_by_orientation = {str(i): np.inf for i in direction_angles}
    contour_by_orientation = {str(i): np.inf for i in direction_angles}

    for angle in direction_angles:

        ray = create_ray_from_centroid(centroid, angle, length=max_distance)
        further_point_geom = ray.intersection(geom)
        further_point_geom = get_furthest_point(further_point_geom, centroid) if not (
            isinstance(further_point_geom, Point)) else further_point_geom
        point_to_linearring_distance = centroid.distance(further_point_geom)

        contour_by_orientation[str(angle)] = round(point_to_linearring_distance,2)
        other_pol_intersection_point = None

        for other_polygon in other_polygons:
            polygons = other_polygon.geoms if isinstance(other_polygon, MultiPolygon) else [other_polygon]
            for polygon in polygons:
                intersection = ray.intersection(polygon)
                if not intersection.is_empty:
                    nearest_point = nearest_points(centroid, intersection)[1]
                    distance = round(centroid.distance(nearest_point) - point_to_linearring_distance,2)
                    if distance < 0.2:
                        distance = 0.0
                    if distance <= distances_by_orientation[str(angle)]:
                        other_pol_intersection_point = nearest_point
                        distances_by_orientation[str(angle)] = distance

        if plots and other_pol_intersection_point is not None and pdf is not None and distance <= max_distance:
            fig, ax = plt.subplots()
            plot_shapely_geometries(
                geometries=list(other_polygons) + [further_point_geom] + [
                    LineString([centroid, other_pol_intersection_point])] + [centroid],
                contextual_geometry=Polygon(geom) if not isinstance(geom, list) else [Polygon(i) for i in geom],
                title=f"Building shadows, orientation: {angle}º, floor: {floor},"
                      f"\ncontour to shadow: {str(distances_by_orientation[str(angle)])}m, "
                      f"centroid to contour: {str(contour_by_orientation[str(angle)])}m",
                ax=ax
            )
            pdf.savefig(fig)
            plt.close(fig)

    return {"shadows": distances_by_orientation, "contour": contour_by_orientation}


def get_municipality_open_street_maps(open_street_dir, query_location, crs=None):

    fn = f"{open_street_dir}/{query_location}.gpkg"

    if not os.path.exists(fn):
        # Get geodataframe for the query location
        gdf_place = ox.geocode_to_gdf(query_location)

        # Create a single polygon from multiple geometries
        polygon = gdf_place.geometry.unary_union

        # Get streets within this polygon
        streets_gdf = ox.features_from_polygon(
            polygon,
            tags={"highway": True}
            # ["motorway", "trunk", "primary", "secondary", "tertiary", "road", "unclassified",
            #  "living_street", "service", "footway", "residential", "pedestrian", "motorway_link",
            #  "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
        )
        streets_gdf = streets_gdf.reset_index()
        streets_gdf = streets_gdf[streets_gdf['element']=="way"]
        streets_gdf = streets_gdf[~streets_gdf['geometry'].isna()]
        streets_gdf = streets_gdf.to_crs("EPSG:25831")
        streets_gdf = streets_gdf.loc[:,streets_gdf.columns.isin([
            "element", "id", "geometry"])]
        streets_gdf.to_file(fn, driver="GPKG")
    else:
        streets_gdf = gpd.read_file(fn)

    if crs is not None:
        streets_gdf.geometry = streets_gdf.geometry.to_crs(crs)

    return streets_gdf

def detect_number_of_orientations(orientation_lengths: dict, threshold_ratio: float = 0.1):
    total_length = sum(orientation_lengths.values())
    if total_length == 0:
        return 0  # avoid division by zero

    # Count orientations with a length greater than threshold_ratio * total_length
    significant_orientations = [
        deg for deg, length in orientation_lengths.items()
        if length / total_length >= threshold_ratio
    ]
    return len(significant_orientations)

def is_segment_in_contact_with_street(segment, streets, orientation_discrete_interval_in_degrees, max_distance=5, orientation_tolerance=10):
    """
    Determines if a wall segment is in contact with a street.

    Conditions:
    - The segment is within max_distance of a street.
    - The segment's orientation is within orientation_tolerance degrees of the street's orientation.
    """

    close_streets = streets[segment.buffer(max_distance).intersects(streets.geometry)]
    # plot_linestrings_and_polygons(linestrings=close_streets.geometry, polygons=[segment.buffer(8)],pdf_file="test3.pdf")

    if len(close_streets)==0:
        return False  # No nearby streets
    else:
        # Compute orientation of the wall and street
        segment_orientation = calculate_wall_outdoor_normal_orientation(
            segment, orientation_interval=orientation_discrete_interval_in_degrees)
        street_segments = []
        for street_linestring in close_streets.geometry:
            street_segments.extend(split_linestring_to_segments(street_linestring))
        street_orientation = [calculate_wall_outdoor_normal_orientation(
                street_segment, orientation_interval=orientation_discrete_interval_in_degrees
            ) for street_segment in street_segments]

        # Check if segment is aligned with street (within tolerance)
        angle_diff = [abs(segment_orientation - elem) % 180 for elem in street_orientation]  # Normalize to [0,180]
        return np.any(
            [elem <= orientation_tolerance for elem in angle_diff] or
            [elem >= (180 - orientation_tolerance) for elem in angle_diff])


def split_linestring_to_segments(line: LineString):
    if not isinstance(line, LineString) or len(line.coords) < 2:
        raise ValueError("Input must be a LineString with at least two points.")

    segments = []
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        segments.append(seg)
    return segments

def weighted_circular_mean(degrees, weights):
    radians = np.deg2rad(degrees)
    sin_sum = np.sum(np.sin(radians) * weights)
    cos_sum = np.sum(np.cos(radians) * weights)
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    return mean_angle_deg

def discrete_orientation(angle, bin_size=5):
    angle = angle % 360  # Normalize to [0, 360)
    centered_angle = (angle + bin_size / 2) % 360
    return int(centered_angle // bin_size) * bin_size

def plot_shapely_geometries(geometries, labels=None, clusters=None, ax=None, title=None, contextual_geometry=None,
    contextual_geometry_options=None):
    """
    Plots a list of Shapely geometries using matplotlib and labels them at a point inside the polygon.
    Fills the exterior of Polygon geometries with colors based on clusters, leaving holes unfilled.
    Optionally, a contextual geometry can be plotted with configurable style.

    Parameters:
        geometries (list): A list of Shapely geometries (Point, Polygon, LineString, etc.).
        labels (list, optional): A list of labels corresponding to the geometries.
        clusters (list, optional): A list of cluster identifiers (numeric or string) corresponding
                                   to the geometries for color coding.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib Axes object to plot on. If not provided,
                                                   one will be created.
        title (str, optional): Title for the plot.
        contextual_geometry (Polygon, MultiPolygon, LinearRing, or list of Polygons): A geometry
                                                                                    (or list)
                                                                                    to be plotted.
        contextual_geometry_options (dict, optional): Dictionary of style options for the contextual
                                                      geometry. Possible keys include "linestyle",
                                                      "linewidth", "color", "alpha", etc.

    Returns:
        matplotlib.axes._axes.Axes: The Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = [''] * len(geometries)

    if clusters is None:
        clusters = [0] * len(geometries)

    # Convert cluster identifiers to a numeric index
    unique_clusters = np.unique(clusters)
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

    # Create a colormap
    colormap = plt.get_cmap('viridis', len(unique_clusters))

    # Handle contextual geometry style options
    if contextual_geometry_options is None:
        contextual_geometry_options = {}
    line_style = contextual_geometry_options.get("linestyle", "--")
    line_color = contextual_geometry_options.get("color", "gray")
    line_width = contextual_geometry_options.get("linewidth", 1.0)
    line_alpha = contextual_geometry_options.get("alpha", 1.0)

    # Plot contextual geometry if provided
    if contextual_geometry is not None:
        # Check if it's a list, and process each one if necessary
        if isinstance(contextual_geometry, list):
            for geom in contextual_geometry:
                linearrings = convert_to_linearrings(geom)
                for item in linearrings:
                    x, y = item.xy
                    ax.plot(
                        x,
                        y,
                        linestyle=line_style,
                        color=line_color,
                        linewidth=line_width,
                        alpha=line_alpha
                    )
        else:
            # Convert to LinearRings and plot
            linearrings = convert_to_linearrings(contextual_geometry)
            for item in linearrings:
                x, y = item.xy
                ax.plot(
                    x,
                    y,
                    linestyle=line_style,
                    color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha
                )

    # Main geometries
    for geom, label, cluster in zip(geometries, labels, clusters):
        color = colormap(cluster_to_index[cluster])  # Get color for the cluster

        if isinstance(geom, Polygon):
            # Exterior
            path_data = [(Path.MOVETO, geom.exterior.coords[0])] + \
                        [(Path.LINETO, point) for point in geom.exterior.coords[1:]]
            path_data.append((Path.CLOSEPOLY, geom.exterior.coords[0]))

            # Interiors (holes)
            for interior in geom.interiors:
                path_data.append((Path.MOVETO, interior.coords[0]))
                path_data += [(Path.LINETO, point) for point in interior.coords[1:]]
                path_data.append((Path.CLOSEPOLY, interior.coords[0]))

            codes, verts = zip(*path_data)
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor=color, edgecolor='blue', alpha=0.5)
            ax.add_patch(patch)
            label_point = geom.representative_point()  # A point guaranteed inside

        elif isinstance(geom, MultiPolygon):
            # Plot each polygon in the MultiPolygon
            for poly in geom.geoms:
                path_data = [(Path.MOVETO, poly.exterior.coords[0])] + \
                            [(Path.LINETO, point) for point in poly.exterior.coords[1:]]
                path_data.append((Path.CLOSEPOLY, poly.exterior.coords[0]))

                for interior in poly.interiors:
                    path_data.append((Path.MOVETO, interior.coords[0]))
                    path_data += [(Path.LINETO, point) for point in interior.coords[1:]]
                    path_data.append((Path.CLOSEPOLY, interior.coords[0]))

                codes, verts = zip(*path_data)
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor=color, edgecolor='blue', alpha=0.5)
                ax.add_patch(patch)

            label_point = geom.representative_point()

        elif isinstance(geom, LineString):
            x, y = geom.xy
            ax.plot(x, y, color=color)  # Use cluster color
            label_point = geom.centroid

        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color)
            label_point = geom.centroid

        elif isinstance(geom, Point):
            ax.plot(geom.x, geom.y, 'o', color=color)
            label_point = geom

        elif isinstance(geom, MultiPoint):
            for point in geom.geoms:
                ax.plot(point.x, point.y, 'o', color=color)
            label_point = geom.centroid

        elif isinstance(geom, GeometryCollection):
            # Recursively call the function for each geometry in the GeometryCollection
            for sub_geom in geom.geoms:
                plot_shapely_geometries([sub_geom], ax=ax)
            label_point = geom.centroid

        # Label each geometry
        ax.text(
            label_point.x,
            label_point.y,
            label,
            ha='center',
            va='center',
            fontsize=10,
            color='black'
        )

    ax.set_aspect('equal', 'box')
    ax.relim()
    ax.autoscale_view()

    if title:
        ax.set_title(title)

    plt.show()
    return ax

def classify_above_ground_floor_names(floor):
    floor = floor.upper()  # Ensure uppercase for consistency

    # Common areas
    if floor in ['OM','-OM']:
        return np.nan

    # Penthouse
    elif floor in ['SAT','SA']:
        return 1500


    # Commercial floor or gardens
    elif floor in ['SM', 'LO', 'LC', 'LA', 'L1', 'L02', 'L01', 'JD', '']:
        return 0

    # Attic-related acronyms
    elif floor in ['A', 'ALT', 'AT', 'APT']:
        return 999

    # Ground floor (Bajos)
    elif floor in ['BJ', 'EPT', 'EP', 'BM', 'BX', 'ENT', 'EN', 'E1', 'B1', 'E', 'B', 'PRL', 'PBA', 'PBE', 'PR', 'PP']:
        return 0.5

    # Attics and uppermost levels
    elif floor.startswith('+'):
        try:
            return 999 + int(re.sub(r'[+T]', '', floor))
        except ValueError:
            return 999

    elif floor.startswith('A'):
        try:
            return 999 + int(floor.replace('A', ''))
        except ValueError:
            return 999

    # Floors starting with PR
    elif floor.startswith('PR'):
        try:
            return 0.5 + int(floor.replace('PR', '')) * 0.25
        except ValueError:
            return 0.5

    # Floors starting with P
    elif floor.startswith('P'):
        try:
            return 0 + int(floor.replace('P', ''))
        except ValueError:
            return 0

    # Numeric floors
    elif floor.isdigit() and int(floor)>=0:
        return int(floor)

    # Numeric floors with some letter before
    elif floor[0].isdigit():
        try:
            return int(re.sub(r"[a-zA-Z]","",floor))
        except ValueError:
            return 0

    # Default for unknown types
    else:
        return np.nan

def classify_below_ground_floor_names(floor):
    floor = floor.upper()  # Ensure uppercase for consistency

    # Parking
    if floor in ['PK','ST','T']:
        return -0.5

    # Sub-basements and underground floors
    elif '-' in floor:
        try:
            return int(re.sub(r'[-PCBA]', '', floor)) * -1
        except ValueError:
            return -1

    # Floors starting with PS
    elif floor.startswith('PS'):
        try:
            return 0.5 + int(floor.replace('PS', '')) * -1
        except ValueError:
            return 0.5

    # Floors starting with S
    elif floor.startswith('S'):
        try:
            return -0.5 + int(floor.replace('S', '')) * -1
        except ValueError:
            return -0.5

    # Numeric floors
    elif floor.isdigit():
        if int(floor)<0:
            return int(floor)

    # Default for unknown types
    else:
        return np.nan

def classify_cadaster_floor_names(floor):
    agf = classify_above_ground_floor_names(floor)
    if agf is not np.nan:
        return agf
    else:
        return classify_below_ground_floor_names(floor)

def agg_op(df, funcs, grouping, variable, multiplier=1, divider=1):
    """
    Enhanced aggregation function to support multiple operations and nested groupings.

    Args:
    - df: Input DataFrame (pandas.DataFrame).
    - funcs: List of functions to apply sequentially (e.g., [sum, lambda x: round(x, 0)]).
    - grouping: Column name or list of column names to group by.
    - variable: Target variable for the operation (str).
    - multiplier: Optional multiplier column or scalar (default: 1).
    - divider: Optional divider column or scalar (default: 1).

    Returns:
    - A (possibly nested) dictionary with results of applying the functions grouped by `grouping`.
    """
    import pandas as pd

    # Ensure grouping is a list
    if isinstance(grouping, str):
        grouping = [grouping]

    # Drop rows with NaN in relevant columns
    relevant_columns = grouping + [variable]
    if isinstance(multiplier, str):
        relevant_columns.append(multiplier)
    if isinstance(divider, str):
        relevant_columns.append(divider)
    df = df.dropna(subset=relevant_columns)

    # Prepare multiplier and divider
    if isinstance(multiplier, str):
        mult_values = df[multiplier]
    else:
        mult_values = multiplier

    if isinstance(divider, str):
        div_values = df[divider]
    else:
        div_values = divider

    # Apply multiplier/divider
    if df[variable].dtype != 'object':
        df['_adjusted_value'] = df[variable] * mult_values / div_values
    else:
        df['_adjusted_value'] = df[variable]

    # Apply functions to grouped data
    grouped = df.groupby(grouping)['_adjusted_value']

    def apply_funcs(series, funcs):
        for func in funcs:
            series = func(series)
        return series

    grouped_result = grouped.apply(lambda x: apply_funcs(x, funcs)).reset_index()

    # Build nested dictionary
    def build_nested_dict(df, group_cols, value_col):
        if len(group_cols) == 1:
            return dict(zip(df[group_cols[0]], df[value_col]))
        result = {}
        for key, group_df in df.groupby(group_cols[0]):
            result[key] = build_nested_dict(group_df.drop(columns=group_cols[0]), group_cols[1:], value_col)
        return result

    return build_nested_dict(grouped_result, grouping, '_adjusted_value')

def read_br_inferred_indicators(cadaster_code, wd):
    result_dir = results_dir_(wd)
    return pd.read_pickle(f"{result_dir}/{cadaster_code}_br_results.pkl", compression="gzip")

def read_sbr_inferred_indicators(cadaster_code, wd):
    result_dir = results_dir_(wd)
    return pd.read_pickle(f"{result_dir}/{cadaster_code}_sbr_results.pkl", compression="gzip")

def read_addresses_indicators(cadaster_code, wd):
    result_dir = results_dir_(wd)
    return pd.read_pickle(f"{result_dir}/{cadaster_code}_no_inference.pkl", compression="gzip")

def plot_points_with_indices(points, pdf_file):
    x, y = zip(*points)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Path')
    for i, (xi, yi) in enumerate(points):
        plt.text(xi, yi, str(i), fontsize=9, ha='right', va='bottom', color='red')
    plt.title(f"Plot of {len(points)} Points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal')  # Maintains aspect ratio
    plt.legend()
    plt.savefig(pdf_file)
    plt.show()


def plot_polygons_group(poly_list, poly2, pdf_file, color1='skyblue', color2='lightcoral'):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each polygon in the list
    for poly in poly_list:
        if not isinstance(poly, Polygon):
            raise ValueError("All items in the list must be Shapely Polygons")
        coords = list(mapping(poly)['coordinates'])[0]
        patch = patches.Polygon(coords, closed=True, edgecolor='black', facecolor=color1, alpha=0.5)
        ax.add_patch(patch)

    # Plot the second polygon in a different color
    if not isinstance(poly2, Polygon):
        raise ValueError("Second input must be a Shapely Polygon")
    coords2 = list(mapping(poly2)['coordinates'])[0]
    patch2 = patches.Polygon(coords2, closed=True, edgecolor='black', facecolor=color2, alpha=0.6)
    ax.add_patch(patch2)

    # Set axis limits based on all coordinates
    all_coords = [pt for poly in poly_list for pt in poly.exterior.coords] + list(poly2.exterior.coords)
    x_vals, y_vals = zip(*all_coords)
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)

    ax.set_aspect('equal')
    ax.set_title('Polygon Group vs Single Polygon')
    plt.grid(True)
    plt.savefig(pdf_file)
    plt.show()

# Required helper: cut function for slicing a LineString
def cut(line: LineString, start_dist: float, end_dist: float) -> LineString:
    """Cut a LineString between two distances from its start."""
    if start_dist >= end_dist:
        raise ValueError("Start distance must be less than end distance")

    coords = []
    current_dist = 0.0
    for (p1, p2) in zip(line.coords[:-1], line.coords[1:]):
        seg = LineString([p1, p2])
        seg_len = seg.length
        next_dist = current_dist + seg_len

        if next_dist <= start_dist:
            current_dist = next_dist
            continue
        if current_dist >= end_dist:
            break

        # Compute points within the segment
        part_start = max(start_dist, current_dist)
        part_end = min(end_dist, next_dist)
        if part_start < part_end:
            start_point = seg.interpolate(part_start - current_dist)
            end_point = seg.interpolate(part_end - current_dist)
            coords.append(start_point.coords[0])
            coords.append(end_point.coords[0])

        current_dist = next_dist

    if not coords:
        return LineString()

    # Deduplicate consecutive points
    final_coords = [coords[0]]
    for pt in coords[1:]:
        if pt != final_coords[-1]:
            final_coords.append(pt)

    return LineString(final_coords)

# Monkey patch the cut function to LineString (optional)
LineString.cut = cut

def shorten_linestring(line: LineString, percent: float) -> LineString:
    if not isinstance(line, LineString) or len(line.coords) < 2:
        raise ValueError("Input must be a LineString with at least two points.")

    total_length = line.length
    trim_start = total_length * percent
    trim_end = total_length * (1 - percent)

    # Use Shapely's line slicing via interpolate + linearly spaced points
    trimmed = line.segmentize(0.01)  # Ensure sufficient resolution
    subline = trimmed.interpolate(trim_start), trimmed.interpolate(trim_end)
    return line.cut(trim_start, trim_end)

def plot_linestrings_and_polygons(linestrings, polygons=None, pdf_file=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot LineStrings
    for ls in linestrings:
        if isinstance(ls, BaseGeometry) and not ls.is_empty:
            x, y = ls.xy
            ax.plot(x, y, color='blue', linewidth=2, label='LineString')

    # Plot Polygons
    if polygons:
        for poly in polygons:
            if isinstance(poly, BaseGeometry) and not poly.is_empty:
                x, y = poly.exterior.xy
                ax.fill(x, y, color='orange', alpha=0.5, edgecolor='black', linewidth=1, label='Polygon')

    ax.set_aspect('equal')
    ax.set_title("LineStrings and Polygons")
    ax.axis('off')
    plt.tight_layout()

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if pdf_file:
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {pdf_file}")
    else:
        plt.show()