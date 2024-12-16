import copy
import sys
import os
import shutil
import fnmatch
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
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
from matplotlib.backends.backend_pdf import PdfPages
from shapely.ops import unary_union, nearest_points
from shapely.geometry.polygon import orient
from shapely.geometry import (Polygon, Point, LineString, MultiPolygon, MultiPoint, MultiLineString, GeometryCollection,
                              LinearRing, JOIN_STYLE)
import itertools
import math
from tqdm import tqdm
import multiprocessing
from charset_normalizer import from_path
import joblib
from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib.externals.loky import get_reusable_executor
from datetime import date
import time
import re
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

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
                        inspire_url="https://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
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

def detect_close_buildings(gdf_building_parts, buffer_neighbours, neighbours_column_name, neighbours_id_column_name = "single_building_reference"):
    G = create_graph(gdf_building_parts, geometry_name="building_part_geometry",buffer = buffer_neighbours)
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


def detect_close_buildings_chunk(chunk, buffer_neighbours, neighbours_column_name, neighbours_id_column_name):
    G = create_graph(chunk, geometry_name="building_part_geometry", buffer=buffer_neighbours)
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

        gdf_building_parts["centroid"] = gdf_building_parts.building_part_geometry.centroid

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
            delayed(detect_close_buildings_chunk)(chunk, buffer_neighbours, neighbours_column_name,
                                                  neighbours_id_column_name)
            for chunk in chunks
        )
    get_reusable_executor().shutdown(wait=True)

    # Concatenate the results into a single GeoDataFrame
    gdf_building_parts_final = pd.concat(results, ignore_index=True)

    return gdf_building_parts_final

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
    unioned_geometry = unary_union(geometries.geometry)

    # Step 2: Buffer by a small negative amount and then positive to fill gaps
    unioned_geometry = unioned_geometry.buffer(gap_tolerance, resolution=resolution,
                                               join_style=JOIN_STYLE.mitre).buffer(
        -gap_tolerance, resolution=resolution, join_style=JOIN_STYLE.mitre)

    # Step 3: Perform the union again if needed
    unioned_geometry = unary_union(unioned_geometry)

    return unioned_geometry

def calculate_floor_footprints_chunk(chunk, gdf, group_by, only_exterior_geometry, min_hole_area, gap_tolerance):
    """
    Process a chunk of groups to calculate the 2D footprint geometry for each floor level.
    """
    floor_footprints = []

    for group in chunk:
        # Filter the grouped buildings
        gdf_ = gdf[gdf[group_by] == group] if group_by else gdf

        # Unique levels of floors
        unique_floors = list(range(gdf_['n_floors_above_ground'].max()))
        unique_underground_floors = list(range(1,gdf_['n_floors_below_ground'].max()))

        for floor in unique_floors:
            floor_geometries = gdf_[gdf_['n_floors_above_ground'] >= (floor + 1)].reset_index(drop=True)
            unioned_geometry = union_geoseries_with_tolerance(floor_geometries, gap_tolerance=gap_tolerance, resolution=16)

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
                unioned_geometry = union_geoseries_with_tolerance(floor_geometries, gap_tolerance=gap_tolerance, resolution=16)

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

    sys.stderr.flush()

    return floor_footprints

def calculate_floor_footprints(gdf, group_by=None, geometry_name="geometry", only_exterior_geometry=False,
                               min_hole_area=1e-6, gap_tolerance=1e-6, chunk_size=200, num_workers=-1):
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
                                                          gap_tolerance)
                for chunk in chunks
            )
        get_reusable_executor().shutdown(wait=True)

        # Flatten the list of results and create the final GeoDataFrame
        floor_footprints = [item for sublist in results for item in sublist]
        floor_footprints_gdf = gpd.GeoDataFrame(floor_footprints, crs=gdf.crs)
    else:
        floor_footprints_gdf = gpd.GeoDataFrame()
        for chunk in chunks:
            floor_footprints_gdf = pd.concat([
                floor_footprints_gdf,
                gpd.GeoDataFrame(
                    calculate_floor_footprints_chunk(chunk, gdf, group_by, only_exterior_geometry,
                                                     min_hole_area, gap_tolerance),
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

    for point in multipoint.geoms:
        distance = point.distance(reference_point)
        if distance > max_distance:
            max_distance = distance
            furthest_point = point

    return furthest_point


def generate_random_interior_points_from_geometry(geometry, num_points):
    """
    Generate random points inside a LinearRing, MultiLineString, Polygon, or MultiPolygon.

    Parameters:
    geometry (LinearRing, MultiLineString, Polygon, MultiPolygon): The geometry within which to generate points.
    num_points (int): The number of random points to generate.

    Returns:
    list: A list of Point objects inside the geometry.
    """
    points = []

    # Handle MultiPolygon and MultiLineString by iterating through each Polygon/LineString
    if isinstance(geometry, MultiPolygon) or isinstance(geometry, MultiLineString):
        geometries = list(geometry.geoms)
    else:
        geometries = [geometry]

    while len(points) < num_points:
        # Choose a geometry (Polygon or LineString) to generate the point in
        selected_geometry = np.random.choice(geometries)

        if isinstance(selected_geometry, Polygon):
            minx, miny, maxx, maxy = selected_geometry.bounds
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

            if selected_geometry.contains(random_point):
                points.append(random_point)

        elif isinstance(selected_geometry, LinearRing):
            # Generate random points inside the bounding box, then project them onto the LinearRing
            minx, miny, maxx, maxy = selected_geometry.bounds
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            projected_point = selected_geometry.interpolate(selected_geometry.project(random_point))
            points.append(projected_point)

        elif isinstance(selected_geometry, LineString):
            # For LineString, generate random points along the line
            point_on_line = selected_geometry.interpolate(np.random.uniform(0, selected_geometry.length))
            points.append(point_on_line)

    return points

def distance_from_points_to_polygons_by_orientation(linearring, other_polygons, num_points=20, orientation_interval=5,
                                                    plots=False, pdf=None, floor=None):
    random_points = generate_random_interior_points_from_geometry(linearring, num_points)
    direction_angles = list(range(0, 360, orientation_interval))
    aggregated_distances_by_orientation = {str(i): [] for i in direction_angles}
    final_distances_by_orientation = {str(i): np.inf for i in direction_angles}
    representativity_by_orientation = {str(i): 0.0 for i in direction_angles}

    for angle in direction_angles:
        for point in random_points:
            distances_by_orientation = {str(i): np.inf for i in direction_angles}

            ray = create_ray_from_centroid(point, angle, length=75)
            further_point_geom = ray.intersection(linearring)
            further_point_geom = get_furthest_point(further_point_geom, point) if isinstance(further_point_geom,
                                                                                             MultiPoint) else further_point_geom
            point_to_linearring_distance = point.distance(further_point_geom)
            other_pol_intersection_point = None

            for other_polygon in other_polygons:
                polygons = other_polygon.geoms if isinstance(other_polygon, MultiPolygon) else [other_polygon]
                for polygon in polygons:
                    intersection = ray.intersection(polygon)
                    if not intersection.is_empty:
                        nearest_point = nearest_points(point, intersection)[1]
                        distance = round(point.distance(nearest_point) - point_to_linearring_distance,2)
                        if distance < 0.05:
                            distance = 0.0
                        if distance <= distances_by_orientation[str(angle)]:
                            other_pol_intersection_point = nearest_point
                            distances_by_orientation[str(angle)] = distance

            aggregated_distances_by_orientation[str(angle)].append(distances_by_orientation[str(angle)])

            # Aggregate the distances by orientation
            final_distances_by_orientation[str(angle)] = (
                round(np.mean([v for v in aggregated_distances_by_orientation[str(angle)] if v < np.inf]), 2)) if (
                any(v < np.inf for v in aggregated_distances_by_orientation[str(angle)])) else np.inf
            representativity_by_orientation[str(angle)] = (
                len([v for v in aggregated_distances_by_orientation[str(angle)] if v < np.inf]) / len(random_points)
            ) if (any(v < np.inf for v in aggregated_distances_by_orientation[str(angle)])) else 0.0

            if plots and other_pol_intersection_point is not None and pdf is not None:
                fig, ax = plt.subplots()
                plot_shapely_geometries(
                    geometries=list(other_polygons) + [LineString([further_point_geom, other_pol_intersection_point])] +
                               [point],
                    contextual_geometry=Polygon(linearring),
                    title=f"Building shadows, orientation: {angle}º, floor: {floor},\ndistance case: "
                          f"{str(distances_by_orientation[str(angle)])}m, distance average: "
                          f"{str(final_distances_by_orientation[str(angle)])}m, representativity: "
                          f"{str(round(representativity_by_orientation[str(angle)] * 100, 2))}%",
                    ax=ax
                )
                pdf.savefig(fig)
                plt.close(fig)

    return final_distances_by_orientation, representativity_by_orientation

def distance_from_centroid_to_polygons_by_orientation(linearring, other_polygons, centroid, orientation_interval=5,
                                                    plots=False, pdf=None, floor=None):
    direction_angles = list(range(0, 360, orientation_interval))
    distances_by_orientation = {str(i): np.inf for i in direction_angles}
    contour_by_orientation = {str(i): np.inf for i in direction_angles}

    for angle in direction_angles:

        ray = create_ray_from_centroid(centroid, angle, length=75)
        further_point_geom = ray.intersection(linearring)
        further_point_geom = get_furthest_point(further_point_geom, centroid) if isinstance(further_point_geom,
                                                                                         MultiPoint) else further_point_geom
        point_to_linearring_distance = centroid.distance(further_point_geom)
        contour_by_orientation[str(angle)] = point_to_linearring_distance
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

        if plots and other_pol_intersection_point is not None and pdf is not None:
            fig, ax = plt.subplots()
            plot_shapely_geometries(
                geometries=list(other_polygons) + [further_point_geom] +
                           [LineString([centroid, other_pol_intersection_point])] + [centroid],
                contextual_geometry=Polygon(linearring),
                title=f"Building shadows, orientation: {angle}º, floor: {floor},"
                      f"\ncontour to shadow distance: {str(distances_by_orientation[str(angle)])}m, "
                      f"centroid to contour distance: {str(contour_by_orientation[str(angle)])}m",
                ax=ax
            )
            pdf.savefig(fig)
            plt.close(fig)

    return {"shadows": distances_by_orientation, "contour": contour_by_orientation}


def process_building_parts(building_part_gdf_, building_gdf_, buildings_CAT, results_dir = None, plots = False,
                           ratio_communal_areas=0.15, ratio_usable_areas=0.9,
                           orientation_discrete_interval_in_degrees = 5,
                           num_workers = max(1, math.ceil(multiprocessing.cpu_count()/2))):

    if plots:
        if results_dir is None:
            results_dir = "results"
        os.makedirs(f"{results_dir}/plots", exist_ok=True)

    gdf_global = detect_close_buildings_parallel(gdf_building_parts=building_part_gdf_,
                                                 buffer_neighbours=50,
                                                 neighbours_column_name="nearby_buildings",
                                                 neighbours_id_column_name="building_reference",
                                                 num_workers=-1,
                                                 column_name_to_split="zone_reference")
    gdf_footprints_global = calculate_floor_footprints(
        gdf=gdf_global,
        group_by="building_reference",
        geometry_name="building_part_geometry",
        only_exterior_geometry=True,
        min_hole_area=0.5,
        gap_tolerance=0.05,
        chunk_size=500,
        num_workers=-1)

    grouped = gdf_global.groupby("zone_reference")

    zone_references = list(grouped.groups.keys())

    # Define a wrapper for joblib's delayed processing
    def process_zone_delayed(zone_reference):
        # zone_reference="02086DF3800G"
        gdf_zone = grouped.get_group(zone_reference).reset_index(drop=True)
        return process_zone(
            gdf_zone,
            zone_reference,
            building_gdf_,
            gdf_footprints_global,
            buildings_CAT,
            results_dir,
            plots,
            ratio_communal_areas,
            ratio_usable_areas,
            orientation_discrete_interval_in_degrees
        )

    # Parallel processing for each zone
    with tqdm_joblib(tqdm(total=len(zone_references), desc="Inferring geometrical KPIs for each building...")):
        results_list = Parallel(n_jobs=num_workers)(
            delayed(process_zone_delayed)(zone_reference) for zone_reference in zone_references
        )

    # Concatenate all results into a single DataFrame
    results = pd.concat(results_list, ignore_index=True)

    return results

def process_zone(gdf_zone, zone_reference, building_gdf_, gdf_footprints_global, buildings_CAT, results_dir, plots, ratio_communal_areas, ratio_usable_areas, orientation_discrete_interval_in_degrees):

    try:
        # PDF setup for the zone if pdf_plots is True
        if plots:
            pdf = PdfPages(f"{results_dir}/plots/zone_{zone_reference}.pdf")

        # Calculation of the single building references and adjacent buildings
        gdf_zone = detect_close_buildings(gdf_building_parts = gdf_zone,
                               buffer_neighbours = 0.5,
                               neighbours_column_name = "adjacent_buildings",
                               neighbours_id_column_name = "single_building_reference")

        if plots:
            fig, ax = plt.subplots()
            plot_shapely_geometries(gdf_zone.building_part_geometry,
                                    labels=gdf_zone.n_floors_above_ground,
                                    clusters=gdf_zone.single_building_reference,
                                    ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

        gdf_aux_footprints = calculate_floor_footprints(
            gdf=gdf_zone,
            group_by="single_building_reference",
            geometry_name="building_part_geometry",
            min_hole_area=0.5,
            gap_tolerance=0.05,
            num_workers=1)

        gdf_aux_footprints_global = calculate_floor_footprints(
            gdf=gdf_zone,
            geometry_name="building_part_geometry",
            min_hole_area=0.5,
            gap_tolerance=0.05,
            num_workers=1)

        if plots:
            fig, ax = plt.subplots()
            plot_shapely_geometries(gdf_aux_footprints.geometry, clusters=gdf_aux_footprints.group, ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

        # Detect all the patios in a zone
        patios_detected = {}
        for floor in range(gdf_aux_footprints.floor.max() + 1):
            patios_detected[floor] = (
                get_all_patios(gdf_aux_footprints_global[gdf_aux_footprints_global['floor']==floor].geometry))

        unique_patios_detected = unique_polygons(list(itertools.chain(*patios_detected.values())),tolerance=0.1)

        if plots:
            fig, ax = plt.subplots()
            plot_shapely_geometries([i.exterior for i in gdf_zone.building_part_geometry] + unique_patios_detected, ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

        # Close the PDF file for the current zone
        if plots:
            pdf.close()

        results_ = []

        for single_building_reference, building_gdf_item in gdf_zone.groupby('single_building_reference'):
            # grouped2 = gdf_zone.groupby("single_building_reference")
            # single_building_reference = list(grouped2.groups.keys())[0]
            # building_gdf_item = grouped2.get_group(single_building_reference).reset_index().drop(['index'], axis=1).copy()

            # PDF setup for the building if plots is True
            if plots:
                pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}.pdf")

            # Obtain a general geometry of the building, considering all floors.
            building_geom = union_geoseries_with_tolerance(building_gdf_item['building_part_geometry'],
                                                           gap_tolerance=0.05, resolution=16)

            # Extract and clean the set of neighbouring buildings
            adjacent_buildings_set = {id for ids in building_gdf_item['adjacent_buildings'] for id in ids.split(",")}
            adjacent_buildings_set.discard(
                single_building_reference)  # Safely remove the single_building_reference itself if present
            adjacent_buildings = sorted(adjacent_buildings_set)

            # Extract and clean the set of nearby buildings
            nearby_buildings_set = {id for ids in building_gdf_item['nearby_buildings'] for id in ids.split(",")}
            nearby_buildings_set.discard(
                single_building_reference.split("_")[0])  # Safely remove the single_building_reference itself if present
            nearby_buildings = sorted(nearby_buildings_set)

            # Is there any premises in ground floor?
            premises = False
            premises_activity_typologies = []
            premises_names = []
            premises_last_revision = []
            if "number_of_ground_premises" in building_gdf_item.columns:
                if (~np.isfinite(building_gdf_item.iloc[0]["number_of_ground_premises"]) and
                        building_gdf_item.iloc[0]["number_of_ground_premises"] > 0):
                    premises = True
                    premises_activity_typologies = building_gdf_item.iloc[0]["ground_premises_activities"]
                    premises_names = building_gdf_item.iloc[0]["ground_premises_names"]
                    premises_last_revision = building_gdf_item.iloc[0]["ground_premises_last_revision"]

            # Filter building in study
            gdf_aux_footprint_building = gdf_aux_footprints[gdf_aux_footprints['group'] == single_building_reference]
            if plots:
                fig, ax = plt.subplots()
                plot_shapely_geometries(list(gdf_aux_footprint_building.geometry), ax=ax,
                                        title = "Building in analysis")
                pdf.savefig(fig)
                plt.close(fig)

            # Filter related buildings footprint
            gdf_aux_footprints_ = gdf_aux_footprints[gdf_aux_footprints['group'].isin(adjacent_buildings)]
            if plots:
                fig, ax = plt.subplots()
                plot_shapely_geometries(gdf_aux_footprints_.geometry, clusters=gdf_aux_footprints_.group, ax=ax,
                                        contextual_geometry = building_geom,
                                        title = "Adjacent buildings")
                pdf.savefig(fig)
                plt.close(fig)

            # Filter related buildings footprint
            gdf_aux_footprints_nearby = gdf_footprints_global[gdf_footprints_global['group'].isin(nearby_buildings)]
            if plots:
                fig, ax = plt.subplots()
                plot_shapely_geometries(gdf_aux_footprints_nearby.geometry, clusters=gdf_aux_footprints_nearby.group, ax=ax,
                                        contextual_geometry=building_geom,
                                        title="Nearby buildings")
                pdf.savefig(fig)
                plt.close(fig)

            floor_area = {}
            underground_floor_area = {}
            roof_area = {}
            patios_area = {}
            patios_n = {}
            perimeter = {}
            air_contact_wall = {}
            shadows_at_distance = {}
            adiabatic_wall = {}
            patios_wall = {}
            significant_orientations_by_floor = {}
            n_floors = gdf_aux_footprint_building.floor.max() if gdf_aux_footprint_building.floor.max() > 0 else np.nan
            n_underground_floors = -gdf_aux_footprint_building.floor.min() if gdf_aux_footprint_building.floor.min() < 0 else 0
            floor_area_with_possible_residential_use = []
            n_dwellings = building_gdf_[
                    building_gdf_["building_reference"] == building_gdf_item.iloc[0]["building_reference"]
                ].iloc[0]["n_dwellings"]
            n_items = building_gdf_[
                    building_gdf_["building_reference"] == building_gdf_item.iloc[0]["building_reference"]
                ].iloc[0]["n_building_units"] - n_dwellings
            n_buildings = len(
                gdf_zone[
                        gdf_zone["building_reference"] == building_gdf_item.iloc[0]["building_reference"]
                    ]["single_building_reference"].unique()
            )
            if n_dwellings == 0:
                building_type = "Non-residential"
                ratio_communal_areas_ = 0.0
                ratio_usable_private_areas_ = 0.0
            elif n_dwellings == 1:
                building_type = "Single-family"
                ratio_communal_areas_ = 0.0
                ratio_usable_private_areas_ = ratio_usable_areas
            elif n_dwellings > 1:
                building_type = "Multi-family"
                ratio_communal_areas_ = ratio_communal_areas
                ratio_usable_private_areas_ = ratio_usable_areas - ratio_communal_areas

            if n_underground_floors > 0:

                for underground_floor in range(1,(n_underground_floors+1)):
                    underground_floor_area[(underground_floor)] = round(
                        gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == -(underground_floor)]['geometry'].area.sum(),
                        2
                    )

            if n_floors is not np.nan:

                for floor in range(n_floors + 1):
                    floor_area[floor] = round(
                        gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                        2
                    )
                    if floor == n_floors:
                        roof_area[floor] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                            2
                        )
                    else:
                        roof_area[floor] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor][
                                'geometry'].area.sum() -
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == (floor + 1)][
                                'geometry'].area.sum(),
                            2
                        )
                    patios = patios_in_the_building(
                        patios_geoms=patios_detected[floor],
                        building_geom=building_geom,
                        tolerance=0.8
                    )
                    patios_n[floor] = len(patios)
                    patios_area[floor] = round(sum([patio.area for patio in patios]), 2)
                    walls = gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry']
                    if isinstance(list(walls)[0], MultiPolygon):
                        walls = [walls_i for walls_i in list(walls)[0].geoms]
                    elif isinstance(list(walls)[0], Polygon):
                        walls = list(walls)

                    # Initialize the lengths for each type of wall contact and for each orientation
                    air_contact_wall[floor] = {direction: 0.0 for direction in [str(i) for i in
                                                                                list(range(0, 360, orientation_discrete_interval_in_degrees))]}
                    adiabatic_wall[floor] = 0.0
                    patios_wall[floor] = 0.0
                    perimeter[floor] = round(sum([peri.exterior.length for peri in walls]), 2)

                    for geom in walls:

                        # Indoor patios
                        patios_wall[floor] += sum([item.length for item in list(geom.interiors)])

                        # Break down the exterior into segments
                        exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
                        for i in range(len(exterior_coords) - 1):
                            segment_assigned = False
                            segment = LineString([exterior_coords[i], exterior_coords[i + 1]])
                            if segment.length < 0.001:
                                continue

                            # Determine the orientation of this segment
                            segment_orientation = str(calculate_wall_outdoor_normal_orientation(
                                segment,
                                orientation_interval=orientation_discrete_interval_in_degrees))

                            # Check if the segment is in contact with patios
                            for patio in patios:
                                if segment_intersects_with_tolerance(
                                        segment, patio,
                                        buffer_distance=0.1,
                                        area_percentage_threshold=15
                                ):
                                    patios_wall[floor] += round(segment.length, 2)
                                    if plots:
                                        fig, ax = plt.subplots()
                                        plot_shapely_geometries(
                                            geometries = [geom] + [segment],
                                            title = f"Wall ID:{i}, floor: {floor},\n"
                                                    f"orientation: {segment_orientation},"
                                                    f"type: patio",
                                            ax = ax,
                                            contextual_geometry = building_geom)
                                        pdf.savefig(fig)
                                        plt.close(fig)
                                    segment_assigned = True
                                    break

                            # Check if the segment is in contact with nearby buildings
                            if not segment_assigned:
                                for aux_geom in gdf_aux_footprints_[gdf_aux_footprints_.floor == floor].geometry:
                                    if segment_intersects_with_tolerance(
                                            segment, aux_geom,
                                            buffer_distance=0.1,
                                            area_percentage_threshold=15
                                    ):
                                        adiabatic_wall[floor] += round(segment.length, 2)
                                        if plots:
                                            fig, ax = plt.subplots()
                                            plot_shapely_geometries(
                                                geometries = [geom] + [segment],
                                                title = f"Wall ID: {i}, floor: {floor},\n"
                                                        f"orientation: {segment_orientation}, "
                                                        f"type: adiabatic",
                                                ax = ax,
                                                contextual_geometry = building_geom)
                                            pdf.savefig(fig)
                                            plt.close(fig)
                                        segment_assigned = True
                                        break

                            # Check if the segment is in contact with outdoor (air contact)
                            if not segment_assigned:
                                air_contact_wall[floor][segment_orientation] += round(segment.length, 2)
                                if plots:
                                    fig, ax = plt.subplots()
                                    plot_shapely_geometries(
                                        geometries = [geom] + [segment],
                                        title = f"Wall ID: {i}, floor: {floor},\n"
                                                f"orientation: {segment_orientation}, "
                                                f"type: air contact",
                                        ax = ax,
                                        contextual_geometry = building_geom)
                                    pdf.savefig(fig)
                                    plt.close(fig)

                for floor, air_contact_walls in air_contact_wall.items():
                    significant_orientations = []
                    significant_threshold = 0.1 * perimeter[floor]

                    for orientation, wall_length in air_contact_walls.items():
                        if wall_length > significant_threshold:
                            significant_orientations.append(int(orientation))

                    # Sort the significant orientations in ascending order for better readability
                    significant_orientations.sort()
                    significant_orientations_by_floor[floor] = significant_orientations

                # Close the PDF file for the current building
                if plots:
                    pdf.close()

                if building_type == "Non-residential":
                    starting_residential_floor = np.nan
                elif premises:
                    starting_residential_floor = 1
                    if building_type == "Multi-family" and n_floors >= n_dwellings:
                        n_dwellings = n_dwellings - 1
                else:
                    if n_floors > 6 and building_type == "Multi-family":
                        starting_residential_floor = 1
                    else:
                        starting_residential_floor = 0
                floor_area_with_possible_residential_use = [floor_area[floor] if floor >= starting_residential_floor or starting_residential_floor is np.nan else 0.0 for floor in range(n_floors + 1)]

            if max(gdf_aux_footprint_building.floor.max(), gdf_aux_footprints_nearby.floor.max()) > 0:

                # Shadows depending orientation
                if plots:
                    pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_shadows.pdf")
                else:
                    pdf = None
                for floor in range(max(gdf_aux_footprint_building.floor.max(),
                                       gdf_aux_footprints_nearby.floor.max()) + 1):
                    # shadows_at_distance[floor] = distance_from_points_to_polygons_by_orientation(
                    #     linearring=building_geom.exterior if isinstance(building_geom, Polygon) else (
                    #         MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                    #     other_polygons=gdf_aux_footprints_nearby[gdf_aux_footprints_nearby.floor==floor].geometry,
                    #     num_points = 25,
                    #     orientation_interval = orientation_discrete_interval_in_degrees,
                    #     plots = plots,
                    #     pdf = pdf,
                    #     floor = floor
                    # )
                    shadows_at_distance[floor] = distance_from_centroid_to_polygons_by_orientation(
                        linearring=building_geom.exterior if isinstance(building_geom, Polygon) else (
                            MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                        other_polygons=gdf_aux_footprints_nearby[gdf_aux_footprints_nearby.floor == floor].geometry,
                        centroid=building_geom.centroid,
                        orientation_interval=orientation_discrete_interval_in_degrees,
                        plots=plots,
                        pdf=pdf,
                        floor=floor
                    )
                if plots:
                    pdf.close()

            # Store results
            results_.append({
                'building_reference': single_building_reference.split("_")[0],
                'single_building_reference': single_building_reference,
                'total_n_dwellings': n_dwellings,
                'total_n_other_building_items': n_items,
                'n_buildings': n_buildings,
                'type': building_type,
                'detached': (np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0) > 0.0,
                'exists_ground_commercial_premises': premises,
                'ground_commercial_premises_names': premises_names,
                'ground_commercial_premises_typology': premises_activity_typologies,
                'ground_commercial_premises_last_revision': premises_last_revision,
                'n_floors': n_floors+1,
                'n_underground_floors': n_underground_floors,
                'built_area_below_ground_by_floor': [underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)] if n_underground_floors > 0 else [],
                'built_area_below_ground_total': np.sum([underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)]) if n_underground_floors > 0 else [],
                'built_area_above_ground_by_floor': [floor_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                'built_area_above_ground_total': np.sum([floor_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                'roof_area_above_ground_by_floor': [roof_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                'roof_area_above_ground': np.sum([roof_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                'building_footprint_area': round(building_geom.area if isinstance(building_geom, Polygon)
                                                 else sum(polygon.area for polygon in building_geom.geoms) if isinstance(building_geom, MultiPolygon)
                                                 else 0.0, 2),
                'building_footprint': building_geom.exterior if isinstance(building_geom, Polygon) else (
                            MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                'building_footprint_by_floor': gdf_aux_footprint_building[["floor","geometry"]].to_dict(index=False, orient='split')['data'],
                'residential_area_by_floor': floor_area_with_possible_residential_use,
                'usable_residential_area': np.sum(floor_area_with_possible_residential_use) * ratio_usable_private_areas_ if floor_area_with_possible_residential_use != [] else 0.0,
                'communal_residential_area': np.sum(floor_area_with_possible_residential_use) * ratio_communal_areas_ if floor_area_with_possible_residential_use != [] else 0.0,
                'patios_area_by_floor': [patios_area[floor] for floor in range(n_floors + 1)] if patios_area != {} else [],
                'patios_n_by_floor': [patios_n[floor] for floor in range(n_floors + 1)] if patios_n != {} else [],
                'patios_area_common': np.max([patios_area[floor] for floor in range(n_floors + 1)]) if patios_area != {} else 0.0,
                'patios_n_common': np.max([patios_n[floor] for floor in range(n_floors + 1)]) if patios_n != {} else 0,
                'perimeter': [perimeter[floor] for floor in range(n_floors + 1)] if perimeter != {} else [],
                'air_contact_wall_by_floor': {key: [air_contact_wall[d][key] for d in air_contact_wall] for key in
                                     air_contact_wall[0]} if air_contact_wall != {} else {},
                'air_contact_wall_total':  {key: np.sum([air_contact_wall[d][key] for d in air_contact_wall]) for key in
                                     air_contact_wall[0]} if air_contact_wall != {} else {},
                'adiabatic_wall_by_floor': [adiabatic_wall[floor] for floor in range(n_floors + 1)] if adiabatic_wall != {} else [],
                'adiabatic_wall_total': np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0,
                'patios_wall_by_floor': [patios_wall[floor] for floor in range(n_floors + 1)] if patios_wall != {} else [],
                'patios_wall_total': np.sum([patios_wall[floor] for floor in range(n_floors + 1)]) if patios_wall != {} else 0.0,
                'shadows_at_distance': {key: [shadows_at_distance[d]['shadows'][key] for d in shadows_at_distance] for key in
                                        shadows_at_distance[0]['shadows']} if shadows_at_distance != {} else {},
                'building_contour_at_distance': {key: np.mean([shadows_at_distance[d]['contour'][key] for d in shadows_at_distance]) for key in
                                        shadows_at_distance[0]['contour']} if shadows_at_distance != {} else {}
            })

        # Consider the information from CAT files
        if buildings_CAT is not None and len(results_) > 0:
            results_ = pd.DataFrame(results_)

            # # Try to cluster the building by different parts to assign each space (in CAT files)
            # building_gdf_by_floor = gpd.GeoDataFrame(results_[results_.building_reference == '0208601DF3800G']['building_footprint_by_floor'][0])
            # building_gdf_by_floor.columns = ["floor", "geometry"]
            # building_gdf_by_floor = building_gdf_by_floor.set_geometry("geometry")
            # building_gdf_by_floor_exploded = building_gdf_by_floor.explode(index_parts=False).reset_index(drop=True)
            # building_gdf_by_floor_exploded['area'] = building_gdf_by_floor_exploded.area
            # building_gdf_by_floor_exploded = building_gdf_by_floor_exploded.sort_values("floor", ascending=False)
            # building_gdf_by_floor_exploded['centroid'] = building_gdf_by_floor_exploded.geometry.centroid
            # features = np.array([building_gdf_by_floor_exploded['centroid'].x,
            #                         building_gdf_by_floor_exploded['centroid'].y]).T
            # scaler = MinMaxScaler()
            # scaled_features = scaler.fit_transform(features)
            # db = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage="single")  # Adjust eps for your data's scale
            # building_gdf_by_floor_exploded['cluster'] = db.fit_predict(scaled_features)

            buildings_CAT_ = buildings_CAT.filter(
                pl.col("building_reference") == "0208601DF3800G") #single_building_reference.split("_")[0]
            buildings_CAT_ = buildings_CAT_.with_columns(
                (pl.when(pl.col("street_number1")=="0000").then(pl.lit("")).otherwise(pl.col("street_number1").cast(pl.Int16))).alias("street_number1"),
                (pl.when(pl.col("street_number2")=="0000").then(pl.lit("")).otherwise(pl.col("street_number2").cast(pl.Int16))).alias("street_number2"),
                (pl.when(pl.col("km")=="00000").then(pl.lit("")).otherwise(pl.col("km").cast(pl.Int16))).alias("km")
            )
            buildings_CAT_= buildings_CAT_.with_columns(
                pl.concat_str(["street_type", "street_name", "street_number1", "street_letter1", "street_number2",
                 "street_letter2", "km", "building_space_block_name", "building_space_stair_name"], separator=" ").
                str.strip_chars().str.replace_all(r"\s+", " ").alias("address")
            )
            buildings_CAT_grouped = buildings_CAT_.group_by("address").agg(
                (pl.col("building_space_floor_name").unique()).alias("unique_floors")
            ).to_pandas()
            buildings_CAT_ = buildings_CAT_.to_pandas()
            buildings_CAT_grouped["order_above_ground_floors"] = buildings_CAT_grouped.unique_floors.apply(
                lambda v: [classify_above_ground_floor_names(x) for x in v])
            buildings_CAT_grouped["order_below_ground_floors"] = buildings_CAT_grouped.unique_floors.apply(
                lambda v: [classify_below_ground_floor_names(x) for x in v])
            def sort_floors(unique_floors, order):
                # Filter out NaN or None in the order list
                valid_indices = [i for i, o in enumerate(order) if o is not None and not pd.isna(o)]
                # Create a list of valid floors and their corresponding order
                valid_floors = [(unique_floors[i], order[i]) for i in valid_indices]
                # Sort the valid floors by their order
                return [floor for floor, _ in sorted(valid_floors, key=lambda x: x[1])]
            buildings_CAT_grouped["unique_floors_above_ground_ordered"] = buildings_CAT_grouped.apply(
                lambda row: sort_floors(row["unique_floors"], row["order_above_ground_floors"]), axis=1
            )
            buildings_CAT_grouped["unique_floors_below_ground_ordered"] = buildings_CAT_grouped.apply(
                lambda row: sort_floors(row["unique_floors"], row["order_below_ground_floors"]), axis=1
            )

            buildings_CAT_ = buildings_CAT_.merge(buildings_CAT_grouped, on="address")
            buildings_CAT_ = buildings_CAT_["building_space_floor_name"]

            results_["total_usable_residential_area"] = results_.groupby("building_reference")[
                "usable_residential_area"].transform("sum")
            results_["ratio_usable_residential_area"] = (results_["usable_residential_area"] /
                                                         results_["total_usable_residential_area"])
            results_["ratio_usable_residential_area"] = np.where(
                np.isnan(results_["ratio_usable_residential_area"]), 0.0,
                results_["ratio_usable_residential_area"])
            results_["n_dwellings"] = (results_["total_n_dwellings"] *
                                       results_["ratio_usable_residential_area"]).astype(int)
            results_["n_dwellings_per_floor"] = results_["n_dwellings"] / np.sum(
                [item > 0 for item in floor_area_with_possible_residential_use])
            results_["n_dwellings_per_floor"] = np.where(
                np.isnan(results_["n_dwellings_per_floor"]), 0.0,
                np.ceil(results_["n_dwellings_per_floor"]))
            results_["usable_area_per_dwelling"] = (results_["usable_residential_area"] /
                                                    results_["n_dwellings"])
            results_["usable_area_per_dwelling"] = np.where(
                np.isnan(results_["usable_area_per_dwelling"]), 0.0,
                results_["usable_area_per_dwelling"])

        # If not CAT file:
        elif len(results_)>0:
            results_ = pd.DataFrame(results_)
            results_["total_usable_residential_area"] = results_.groupby("building_reference")[
                "usable_residential_area"].transform("sum")
            results_["ratio_usable_residential_area"] = (results_["usable_residential_area"] /
                                                               results_["total_usable_residential_area"])
            results_["ratio_usable_residential_area"] = np.where(
                np.isnan(results_["ratio_usable_residential_area"]), 0.0,
                results_["ratio_usable_residential_area"])
            results_["n_dwellings"] = (results_["total_n_dwellings"] *
                                               results_["ratio_usable_residential_area"]).astype(int)
            results_["n_dwellings_per_floor"] = results_["n_dwellings"] / np.sum([item>0 for item in floor_area_with_possible_residential_use])
            results_["n_dwellings_per_floor"] = np.where(
                np.isnan(results_["n_dwellings_per_floor"]), 0.0,
                np.ceil(results_["n_dwellings_per_floor"]))
            results_["usable_area_per_dwelling"] = (results_["usable_residential_area"] /
                                                            results_["n_dwellings"])
            results_["usable_area_per_dwelling"] = np.where(
                np.isnan(results_["usable_area_per_dwelling"]), 0.0,
                results_["usable_area_per_dwelling"])

        return results_

    except:
        print(zone_reference)

def plot_shapely_geometries(geometries, labels=None, clusters=None, ax=None, title=None, contextual_geometry=None):
    """
    Plots a list of Shapely geometries using matplotlib and labels them at a point inside the polygon.
    Fills the exterior of Polygon geometries with colors based on clusters, leaving holes unfilled.
    Optionally, a contextual geometry can be plotted as a dotted line in the background.

    Parameters:
        geometries (list): A list of Shapely geometries (Point, Polygon, LineString, etc.).
        labels (list, optional): A list of labels corresponding to the geometries.
        clusters (list, optional): A list of cluster identifiers (numeric or string) corresponding to the geometries for color coding.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib Axes object to plot on. If not provided, one will be created.
        title (str, optional): Title for the plot.
        contextual_geometry (Polygon, MultiPolygon, LinearRing, or list of Polygons): A geometry to be plotted as a dotted line.

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

    # Create a colormap using the updated method
    colormap = plt.get_cmap('viridis', len(unique_clusters))

    # Function to convert a geometry into one or more LinearRings
    def convert_to_linearrings(geom):
        if isinstance(geom, LinearRing):
            return geom
        elif isinstance(geom, Polygon):
            return [LinearRing(geom.exterior.coords)]
        elif isinstance(geom, MultiPolygon):
            return [LinearRing(p.exterior.coords) for p in geom.geoms]
        else:
            raise ValueError("contextual_geometry must be a Polygon, MultiPolygon, or LinearRing")

    # Plot contextual geometry if provided
    if contextual_geometry is not None:
        # Check if it's a list, and process each one if necessary
        if isinstance(contextual_geometry, list):
            for geom in contextual_geometry:
                linearrings = convert_to_linearrings(geom)
                for ring in linearrings:
                    x, y = ring.xy
                    ax.plot(x, y, linestyle='--', color='gray')  # Dotted line for contextual geometry
        else:
            # Convert to LinearRing and plot
            linearrings = convert_to_linearrings(contextual_geometry)
            for ring in linearrings:
                x, y = ring.xy
                ax.plot(x, y, linestyle='--', color='gray')  # Dotted line for contextual geometry

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

            label_point = geom.representative_point()  # Get a point guaranteed to be inside the polygon

        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:  # Correctly iterate over the polygons within the MultiPolygon
                # Exterior
                path_data = [(Path.MOVETO, poly.exterior.coords[0])] + \
                            [(Path.LINETO, point) for point in poly.exterior.coords[1:]]
                path_data.append((Path.CLOSEPOLY, poly.exterior.coords[0]))

                # Interiors (holes)
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
            ax.plot(x, y, color=color)  # Use the cluster color for the line
            label_point = geom.centroid

        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:  # Correctly iterate over the lines within the MultiLineString
                x, y = line.xy
                ax.plot(x, y, color=color)  # Use the cluster color for the lines
            label_point = geom.centroid

        elif isinstance(geom, Point):
            ax.plot(geom.x, geom.y, 'o', color=color)  # Use the cluster color for the point
            label_point = geom

        elif isinstance(geom, MultiPoint):
            for point in geom.geoms:  # Correctly iterate over the points within the MultiPoint
                ax.plot(point.x, point.y, 'o', color=color)  # Use the cluster color for the points
            label_point = geom.centroid

        elif isinstance(geom, GeometryCollection):
            for sub_geom in geom.geoms:  # Correctly iterate over the geometries within the GeometryCollection
                plot_shapely_geometries([sub_geom], ax=ax)
            label_point = geom.centroid

        # Add the label at the representative point (inside the polygon) in black
        ax.text(label_point.x, label_point.y, label, ha='center', va='center', fontsize=10, color='black')

    ax.set_aspect('equal', 'box')

    # Adjust the plot limits to ensure all geometries are visible
    ax.relim()
    ax.autoscale_view()

    # Set the title if provided
    if title:
        ax.set_title(title)

    plt.show()
    return ax


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
            with tqdm_joblib(tqdm(desc="Reading the CAT file...", total=len(lines))):
                parsed_rows = Parallel(n_jobs=-1)(
                    delayed(process_line)(line, allowed_dataset_types) for line in lines
                )

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
def parse_horizontal_division_buildings_CAT_files(cadaster_code, CAT_files_dir):

    combined_dfs = parse_CAT_file(cadaster_code, CAT_files_dir, allowed_dataset_types=[14, 15])
    building_spaces_detailed = combined_dfs[14]
    building_spaces = combined_dfs[15]

    # Filter
    building_spaces = building_spaces.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference"),
        pl.concat_str(['space2_reference', 'space3_reference'], separator=""
                      ).alias("building_space_reference_last_digits")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference")
    )
    building_spaces_detailed = building_spaces_detailed.filter(
        pl.col("distribution_method_for_communal_areas") == "")
    floor_names = sorted(building_spaces_detailed["building_space_floor_name"].unique().to_list())
    df = pd.DataFrame({'floor_name': floor_names})
    df['order'] = df['floor_name'].apply(classify_cadaster_floor_names)
    floor_names_sorted = list(df.sort_values(by='order').reset_index(drop=True).floor_name)

    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces.select(
            'building_space_reference', 'building_space_reference_last_digits', 'building_space_year',
            'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
            'building_space_total_area', 'building_space_participation_rate', 'building_space_use_type'),
        on = "building_space_reference").select(
        'building_reference', 'building_space_reference', 'building_space_reference_last_digits',
        'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
        'building_space_block_name', 'building_space_stair_name', 'building_space_floor_name', 'building_space_door_name',
        'building_space_year', 'retrofitted', 'building_space_retroffiting_year', 'building_space_effective_year',
        'building_space_total_area', 'building_space_area_without_communal', 'building_space_area_balconies_terraces',
        'building_space_area_imputable_to_other_floors', 'building_space_participation_rate',
        'building_space_use_type', 'building_space_detailed_use_type', 'building_space_typology'
    )
    order_mapping = {value: index for index, value in enumerate(floor_names_sorted)}
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_floor_name").map_elements(lambda x: order_mapping.get(x, -1),
                                                         return_dtype=pl.Int32).alias("custom_sort")
    )
    building_spaces_detailed = building_spaces_detailed.sort("custom_sort").drop("custom_sort").sort("building_space_reference")
    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces_detailed.group_by("building_space_reference").agg(
            pl.col("building_space_area_without_communal").sum().alias("building_space_total_area_without_communal"),
            pl.len().alias("building_spaces_considered"),
        ),
        on = "building_space_reference"
    )

    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_participation_rate") / 1000000).alias("building_space_participation_rate"),
        pl.concat_str(['building_space_reference', 'building_space_reference_last_digits'], separator=""
                      ).alias("building_space_reference"),
        ((pl.col("building_space_total_area") - pl.col("building_space_total_area_without_communal")) *
         pl.col("building_space_area_without_communal") / pl.col("building_space_total_area_without_communal")).alias("building_space_communal_area"),
        pl.col("building_space_typology").str.tail(1).alias("building_space_typology_category"),
        pl.col("building_space_typology").str.head(4).alias("building_space_typology_id"),
        (pl.lit(date.today().year) - pl.col("building_space_effective_year")).alias("building_space_age")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_communal_area") + pl.col("building_space_area_without_communal")).alias(
            "building_space_area_with_communal")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_typology_id").replace(
            {k: v.get("Use") for k, v in building_space_typologies.items() if "Use" in v}).
            alias("building_space_typology_use"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClass") for k, v in building_space_typologies.items() if "UseClass" in v}).
            alias("building_space_typology_use_class"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClassModality") for k, v in building_space_typologies.items() if "UseClassModality" in v}).
            alias("building_space_typology_use_class_modality"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseLevel") for k, v in building_space_typologies.items() if "UseLevel" in v}).
            alias("building_space_typology_use_level"),
        pl.when(pl.col("building_space_age") >= 90).
            then(pl.lit(90).cast(pl.Int16).cast(pl.Utf8)).
            otherwise(((pl.col("building_space_age") / 5).floor() * 5).cast(pl.Int16).cast(pl.Utf8)).
            alias("building_space_age_key")
    )
    building_space_age_value_dict = \
        {str(entry["Age"][0]): {k: v for k, v in entry.items() if k != "Age"} for entry in building_space_age_value}

    # Economical value coefficients of the constructions
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.struct(["building_space_typology_id", "building_space_typology_category"]).map_elements(
            lambda row:
                building_space_typologies[
                    row["building_space_typology_id"]]["ConstructionValue"][
                    int(row["building_space_typology_category"]) - 1], # Adjust index to match zero-based indexing
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("construction_relative_economic_value"),
        pl.struct(["building_space_age_key", "building_space_typology_use_level",
                   "building_space_typology_category"]).map_elements(
            lambda row:
            building_space_age_value_dict[
                row["building_space_age_key"]][
                row["building_space_typology_use_level"]][
                int(row["building_space_typology_category"]) - 1],  # Adjust index to match zero-based indexing
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("age_correction_relative_economic_value")
    )

    # Calculate relative economic value
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("construction_relative_economic_value") * pl.col("age_correction_relative_economic_value")).alias("relative_economic_value")
    )

    return building_spaces_detailed

    # rc = building_spaces_detailed.filter((building_spaces_detailed["building_floor_name"] == 'CUB'))["building_reference"][0]
    # result = (building_spaces.filter(building_spaces["building_reference"] == rc)[
    #     ["building_reference","building_stair_name","building_space_reference","building_floor_name","building_door_name","building_space_area"]].join(
    #     building_spaces_detailed.filter(building_spaces_detailed["building_reference"] == rc)[
    #         ["building_space_reference","building_stair_name","building_floor_name", "building_door_name", "building_space_area_without_communal","building_space_detailed_use_type"]],
    #     on="building_space_reference").sort("building_space_reference").join(
    #         pl.DataFrame({
    #             "building_reference": rc,
    #             "area": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["building_part_geometry"].area,
    #             "n_floors_above_ground": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["n_floors_above_ground"],
    #             "n_floors_below_ground": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["n_floors_below_ground"],
    #         }).select(
    #             pl.col("building_reference").first(),
    #             pl.col("n_floors_above_ground").max().alias("max_floors_above"),
    #             pl.col("n_floors_below_ground").max().alias("max_floors_below"),
    #         ),
    #         on="building_reference"))
    # result["building_floor_name_right"].to_list()
    # result.write_csv("joined_spaces.csv",quote_style='always')

