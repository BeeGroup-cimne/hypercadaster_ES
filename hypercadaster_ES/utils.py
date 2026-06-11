# Imports organized by category
# Core libraries
import copy
import sys
import os
import time
import shutil
import fnmatch
import math
import hashlib
import re
from contextlib import contextmanager
from zipfile import ZipFile, BadZipFile
import tarfile
import json
from importlib.resources import files

# Data processing
import numpy as np
import pandas as pd
import polars as pl

# Geospatial libraries
import geopandas as gpd
from geopandas import sjoin
from shapely.ops import unary_union, nearest_points, linemerge
from shapely.geometry.polygon import orient
from shapely.geometry import (box, Polygon, Point, LineString, MultiPolygon, MultiPoint,
                              MultiLineString, GeometryCollection, LinearRing, JOIN_STYLE)
from shapely.errors import GEOSException
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from fastkml import kml
import osmnx as ox
import fiona

# Network and graph libraries
import networkx as nx

# Raster processing
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform as warp_transform

# Parallel processing
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Web requests
import requests
from bs4 import BeautifulSoup

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patches as patches
import contextily as ctx

from shapely.geometry import box
from shapely.ops import unary_union

# ==========================================
# PARALLEL PROCESSING UTILITIES
# ==========================================

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


# ==========================================
# DIRECTORY MANAGEMENT
# ==========================================


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
    os.makedirs(f"{DEM_raster_dir_(data_dir)}/tiles", exist_ok=True)
    os.makedirs(neighborhoods_dir_(data_dir), exist_ok=True)
    os.makedirs(postal_codes_dir_(data_dir), exist_ok=True)
    os.makedirs(f"{postal_codes_dir_(data_dir)}/raw", exist_ok=True)
    os.makedirs(open_data_dir_(data_dir), exist_ok=True)
    os.makedirs(open_street_dir_(data_dir), exist_ok=True)

# PyCatastro.ConsultaMunicipio("BARCELONA")['consulta_municipalero'][]
# def dwellings_per_building_reference(province_name, municipality_name, building_reference):
#     dnprc_result = PyCatastro.Consulta_DNPRC(province_name, municipality_name, building_reference)
#     dnprc_df = pd.DataFrame(
#         [item["dt"]["locs"]["lous"]["lourb"]["loint"] for item in
#          dnprc_result['consulta_dnp']['lrcdnp']['rcdnp']])
#     dnprc_df["pt"].sort_values()


# ==========================================
# DATA CONVERSION & CODE MAPPING
# ==========================================

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
        p_code = re.search(r'(\d{2})\d{3}-', x.get_text()).group(1)
        url = [url for url in urls if url.split('/')[5]==p_code][0]

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


def ine_to_cadaster_codes(ine_codes):
    if ine_codes is not None:
        map_code_dict = pd.read_excel(files("hypercadaster_ES").joinpath("cached_files/ine_inspire_codes.xlsx"), dtype=object, engine='openpyxl').set_index(
            'CÓDIGO INE').to_dict()['CÓDIGO CATASTRO']
        cadaster_codes = [map_code_dict[key] for key in ine_codes]
    else:
        cadaster_codes = None
    return cadaster_codes

def cadaster_to_ine_codes(cadaster_codes):
    if cadaster_codes is not None:
        map_code_dict = pd.read_excel(files("hypercadaster_ES").joinpath("cached_files/ine_inspire_codes.xlsx"), dtype=object, engine='openpyxl').set_index(
            'CÓDIGO CATASTRO').to_dict()['CÓDIGO INE']
        ine_codes = [map_code_dict[key] for key in cadaster_codes]
    else:
        ine_codes = None
    return ine_codes

def municipality_name(ine_codes):

    ine_codes = [ine_codes] if not isinstance(ine_codes, list) else ine_codes
    df = pd.read_excel("https://www.ine.es/daco/daco42/codmun/diccionario24.xlsx", header=1)
    df['Municipality code'] = df['CPRO'].astype(int).apply(lambda x: f"{x:02d}") +\
                              df['CMUN'].astype(int).apply(lambda x: f"{x:03d}")
    df.rename(columns = {'NOMBRE':'Municipality name'}, inplace=True)
    df.drop(columns = ["CODAUTO","CPRO","CMUN","DC"],inplace=True)

    return list(df.loc[df["Municipality code"].isin(ine_codes) ,"Municipality name"])

def get_ine_codes_from_bounding_box(wd, bbox, year=2022):
    """
    Get INE municipality codes for municipalities that intersect with a bounding box.
    
    Parameters:
    -----------
    wd : str
        Working directory path containing the census_tracts data
    bbox : list or tuple
        Bounding box in longitude/latitude format [min_lon, min_lat, max_lon, max_lat]
    year : int, optional
        Census year (default: 2022)
    
    Returns:
    --------
    list
        List of INE municipality codes (strings) for municipalities intersecting the bbox
    
    Example:
    --------
    # Barcelona area bounding box
    bbox = [2.05, 41.32, 2.25, 41.47]
    ine_codes = get_ine_codes_from_bounding_box("/path/to/data", bbox)
    """
    
    # Load census data with municipalities
    from hypercadaster_ES.mergers import get_census_gdf
    census_dir = census_tracts_dir_(wd)
    census_gdf = get_census_gdf(census_dir, year=year, crs="EPSG:4326")
    
    # Aggregate by municipality to get unique municipality geometries
    municipalities_gdf = census_gdf.groupby('ine_municipality_code').agg({
        'municipality_name': 'first',
        'province_name': 'first', 
        'autonomous_community_name': 'first',
        'geometry': lambda x: unary_union(x.tolist())
    }).reset_index()
    
    # Convert back to GeoDataFrame
    municipalities_gdf = gpd.GeoDataFrame(municipalities_gdf, geometry='geometry', crs="EPSG:4326")
    
    # Create bounding box polygon
    bbox_polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:4326")
    
    # Find municipalities that intersect with the bounding box
    intersecting_municipalities = gpd.sjoin(municipalities_gdf, bbox_gdf, predicate='intersects')
    
    # Return list of INE codes
    return intersecting_municipalities['ine_municipality_code'].tolist()

def get_administrative_divisions_naming(cadaster_codes):
    municipalities = pd.read_excel(files("hypercadaster_ES").joinpath("cached_files/ine_inspire_codes.xlsx"), dtype=object, engine='openpyxl')
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

def kml_to_geojson(kml_url):
    r = requests.get(kml_url)
    r.raise_for_status()

    kml_bytes = r.content

    k = kml.KML()
    k = k.from_string(kml_bytes)

    features = []

    for doc in k.features:
        for feature in doc.features:
            for placemark in feature.features:

                properties = {}

                if placemark.extended_data:
                    for element in placemark.extended_data.elements:
                        if hasattr(element, "data"):
                            for sd in element.data:
                                name = getattr(sd, "name", None) or sd.get("name")
                                value = getattr(sd, "value", None) or sd.get("value")
                                if name is not None:
                                    properties[name] = value

                coordinates = []

                if hasattr(placemark, 'geometry') and placemark.geometry:

                    geom = placemark.geometry

                    if geom.geom_type == "MultiPolygon":
                        for polygon in geom.geoms:
                            outer = polygon.exterior.coords
                            coordinates.append([[list(pt)[:2] for pt in outer]])

                    elif geom.geom_type == "Polygon":
                        outer = geom.exterior.coords
                        coordinates.append([list(pt)[:2] for pt in outer])

                feature_json = {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": placemark.geometry.geom_type if coordinates else None,
                        "coordinates": coordinates if coordinates else None
                    }
                }

                features.append(feature_json)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    return json.dumps(feature_collection)

# ==========================================
# FILE OPERATIONS
# ==========================================

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


# ==========================================
# GEOMETRIC UTILITIES
# ==========================================

def make_valid(gdf):
    gdf.geometry = gdf.geometry.make_valid()
    return gdf

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


def create_graph(gdf, geometry_name="building_part_geometry", buffer=0.5):
    """Build a networkx Graph where nodes are gdf rows and edges connect
    geometries whose `buffer`-expanded version intersects the original of another.
    """
    G = nx.Graph()
    G.add_nodes_from(gdf.index)

    base = gdf[[geometry_name]].copy()
    base['id'] = base.index
    base = base.set_geometry(geometry_name)
    buffered = base.copy()
    buffered[geometry_name] = base[geometry_name].buffer(buffer)
    # Spatial self-join on buffered-vs-original
    joined = sjoin(buffered, base, how='inner', predicate='intersects',
                   lsuffix='left', rsuffix='right')

    # Vectorized edge construction (iterrows() was the previous bottleneck on
    # dense Barcelona zones where `joined` can have millions of rows).
    left = joined['id_left'].to_numpy()
    right = joined['id_right'].to_numpy()
    mask = left != right  # drop self-loops; sjoin includes them
    G.add_edges_from(zip(left[mask].tolist(), right[mask].tolist()))

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

    # Connected components are only needed when callers ask for the
    # "single_building_reference" clustering (adjacent-buildings case).
    needs_clusters = neighbours_id_column_name == "single_building_reference"
    if needs_clusters:
        clusters = list(nx.connected_components(G))
        cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
        chunk = chunk.copy()
        chunk['cluster_id'] = chunk.index.map(cluster_map)
        chunk['single_building_reference'] = chunk['building_reference'].str.cat(
            chunk['cluster_id'].astype(str), sep='_'
        )

    # Vectorized node-attribute assignment (replaces iterrows()).
    idx_to_ref = dict(zip(chunk.index.tolist(), chunk[neighbours_id_column_name].tolist()))

    # Build the related-references map directly from G.adjacency() — one pass,
    # set comprehensions, no Python-level per-row loops over the DataFrame.
    related_buildings_map = {}
    for node, neighbours in G.adjacency():
        ref = idx_to_ref.get(node)
        if ref is None:
            continue
        bucket = related_buildings_map.setdefault(ref, {ref})
        for nb in neighbours:
            nb_ref = idx_to_ref.get(nb)
            if nb_ref is not None:
                bucket.add(nb_ref)

    # Keep only the rows the caller asked for (the un-buffered originals).
    chunk = chunk[chunk["buffered"]].copy()
    chunk[neighbours_column_name] = chunk[neighbours_id_column_name].map(
        lambda x: ','.join(sorted(related_buildings_map.get(x, {x})))
    )

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

    def _safe_unary_union(geoms):
        # Some cadastral inputs trigger "TopologyException: unable to assign
        # free hole to a shell" inside GEOS. Repair invalid geometries with
        # make_valid() and retry; if that still fails, buffer(0) is a last
        # resort that almost always produces a polygonal result GEOS accepts.
        #
        # `geoms` here can be either a GeoSeries/list (step 1) or already a
        # single shapely Geometry (step 3) — handle both.
        def _repair(g):
            if g is None:
                return None
            if hasattr(g, "make_valid"):
                try:
                    return g.make_valid()
                except GEOSException:
                    pass
            try:
                return g.buffer(0)
            except GEOSException:
                return g

        try:
            return unary_union(geoms)
        except GEOSException:
            if isinstance(geoms, BaseGeometry):
                repaired = _repair(geoms)
            else:
                repaired = [_repair(g) for g in geoms]
            try:
                return unary_union(repaired)
            except GEOSException:
                # Last-ditch: buffer(0) on whatever came back from the
                # first repair pass.
                if isinstance(repaired, BaseGeometry):
                    return repaired.buffer(0)
                return unary_union([
                    (g.buffer(0) if g is not None else g) for g in repaired
                ])

    # Step 1: Perform a unary union on all geometries
    unioned_geometry = _safe_unary_union(geometries)

    # Step 2: Buffer by a small negative amount and then positive to fill gaps
    try:
        unioned_geometry = unioned_geometry.buffer(gap_tolerance, resolution=resolution,
                                                   join_style=JOIN_STYLE.mitre).buffer(
            -gap_tolerance, resolution=resolution, join_style=JOIN_STYLE.mitre)
    except GEOSException:
        # The buffer round-trip can itself blow up on degenerate output; fall
        # back to a single buffer(0) which is enough to produce a valid result.
        unioned_geometry = unioned_geometry.buffer(0)

    # Step 3: Perform the union again if needed
    unioned_geometry = _safe_unary_union(unioned_geometry)

    return unioned_geometry

def calculate_floor_footprints_chunk(chunk_groups, only_exterior_geometry, min_hole_area, gap_tolerance,
                                     by_floors):
    """
    Process a chunk of groups to calculate the 2D footprint geometry for each floor level.

    `chunk_groups` is a list of (group_label, group_gdf) tuples, already sliced
    from the global gdf by the caller. This avoids the previous per-group full
    scan `gdf[gdf[group_by] == group]` (which was O(N*G), the dominant cost on
    Barcelona) and avoids pickling the full gdf to every joblib worker.
    """
    floor_footprints = []

    for group, gdf_ in chunk_groups:

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
    crs = gdf.crs

    # Pre-group ONCE so per-group access is O(1) instead of O(N) per group.
    # Also keep only the columns the chunk function needs, which dramatically
    # reduces the per-worker memory footprint when joblib pickles the slices.
    needed_cols = [geometry_name]
    if by_floors:
        for c in ("n_floors_above_ground", "n_floors_below_ground"):
            if c in gdf.columns:
                needed_cols.append(c)
    else:
        if "n_floors_above_ground" in gdf.columns:
            needed_cols.append("n_floors_above_ground")
    if group_by and group_by in gdf.columns and group_by not in needed_cols:
        needed_cols.append(group_by)
    gdf_slim = gdf[needed_cols].copy()

    if group_by:
        groups_dict = {name: sub for name, sub in gdf_slim.groupby(group_by, sort=False)}
        unique_groups = list(groups_dict.keys())
    else:
        groups_dict = {"all": gdf_slim}
        unique_groups = ["all"]

    chunks_of_groups = np.array_split(unique_groups, len(unique_groups) // chunk_size + 1)
    # Pre-materialize per-chunk lists of (label, sub_gdf) so the worker doesn't
    # need a reference to the global gdf at all.
    chunk_payloads = [
        [(g, groups_dict[g]) for g in chunk_groups]
        for chunk_groups in chunks_of_groups
    ]

    # Accumulate raw row dicts and build the GeoDataFrame once at the end.
    # Avoids `gpd.GeoDataFrame([], crs=crs)` failing on empty input (no
    # geometry column inferable), and avoids the dtype-upcast trap of seeding
    # with an empty frame that promotes int `floor` values to float on concat
    # — which then breaks downstream `range(.floor.max() + 1)` calls.
    if len(chunk_payloads) > 2 and (num_workers == -1 or num_workers > 1):
        with tqdm_joblib(tqdm(desc="Processing floor above/below ground footprints...",
                              total=len(chunk_payloads))):
            results = Parallel(n_jobs=num_workers)(
                delayed(calculate_floor_footprints_chunk)(payload, only_exterior_geometry, min_hole_area,
                                                          gap_tolerance, by_floors)
                for payload in chunk_payloads
            )
        get_reusable_executor().shutdown(wait=True)

        floor_footprints = [item for sublist in results for item in sublist]
    else:
        floor_footprints = []
        for payload in chunk_payloads:
            floor_footprints.extend(
                calculate_floor_footprints_chunk(
                    payload, only_exterior_geometry, min_hole_area, gap_tolerance, by_floors
                )
            )

    if not floor_footprints:
        # "Ruins"-only group(s): no footprints produced. Return a typed
        # empty GeoDataFrame so callers can `.empty`-check and skip.
        empty_data = (
            {"group": pd.Series([], dtype="object"),
             "floor": pd.Series([], dtype="int64"),
             "geometry": gpd.GeoSeries([], crs=crs)}
            if by_floors else
            {"group": pd.Series([], dtype="object"),
             "geometry": gpd.GeoSeries([], crs=crs)}
        )
        return gpd.GeoDataFrame(empty_data, geometry="geometry", crs=crs)

    return gpd.GeoDataFrame(floor_footprints, crs=crs)

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


def _coerce_scalar_float(value, default=0.0):
    """Coerce `value` to a plain Python float, handling Series/array-like inputs.

    Common offender: a DataFrame slice with duplicate column names returns a
    Series rather than a scalar. Anywhere we need an unambiguous scalar (e.g.
    feeding into comparisons inside ray-shading), call this first.
    """
    try:
        if value is None:
            return default
        if hasattr(value, "iloc"):
            # Series/DataFrame slice — take first; recurse in case it's still array-like
            return _coerce_scalar_float(value.iloc[0], default)
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                return default
            return _coerce_scalar_float(value[0], default)
        v = float(value)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError, IndexError, KeyError):
        return default


def floor_base_elevation(ground_elevation, floor, ground_floor_height=5.0, floor_height=3.0):
    """Absolute elevation (m) of the slab at the bottom of `floor`.

    Floor 0 sits on the ground; floor 1's slab is above the (taller) ground
    floor; subsequent floors stack at the regular upper-floor height.
    Inputs are coerced to scalars defensively so that callers passing a Series
    (e.g. due to a duplicate 'elevation' column upstream) don't poison the
    downstream comparisons with Series-typed heights.
    """
    g = _coerce_scalar_float(ground_elevation, 0.0)
    f = int(floor) if not isinstance(floor, int) else floor
    if f <= 0:
        return g + f * floor_height
    return g + ground_floor_height + (f - 1) * floor_height


# Per-process cache: opening the DEM raster is expensive and, under a fork
# pool, opening it concurrently from many workers can deadlock on GDAL's
# global state. Cache one open handle per (dir, crs) per process.
_DEM_SAMPLER_CACHE = {}


def make_dem_sampler(dem_raster_dir, source_crs):
    """Return a callable that samples DEM elevations for points in `source_crs`.

    The sampler is `f(xs, ys) -> list[float]`. Returns None if no DEM tif is
    found, so callers can degrade gracefully. The opened rasterio dataset is
    cached at module level so subsequent calls within the same process reuse
    a single open handle (critical for multi-process pools where a leak per
    zone would exhaust file descriptors and serialize on GDAL locks).
    """
    dem_path = os.path.join(dem_raster_dir, "DEM.tif")
    if not os.path.exists(dem_path):
        return None
    cache_key = (os.path.abspath(dem_path), str(source_crs) if source_crs is not None else None)
    cached = _DEM_SAMPLER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rds = rasterio.open(dem_path)
    dem_crs = rds.crs
    same_crs = (source_crs is None) or (str(source_crs) == str(dem_crs))

    def sampler(xs, ys):
        if not same_crs:
            xs_d, ys_d = warp_transform(source_crs, dem_crs, list(xs), list(ys))
        else:
            xs_d, ys_d = list(xs), list(ys)
        return [v[0] for v in rds.sample(zip(xs_d, ys_d))]

    sampler.close = rds.close
    _DEM_SAMPLER_CACHE[cache_key] = sampler
    return sampler

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
                                                    plots=False, pdf=None, max_distance=150,
                                                    current_height=None, building_heights=None,
                                                    reference_centroid=None, floor=None,
                                                    aggregation="mean"):
    """Cast rays from many random interior points of `geom` and measure, per
    orientation, the distance from each point's exit through `geom` to the
    first `other_polygons` hit along the ray. Aggregate across rays per
    orientation.

    Two modes share the same code:

    Street-width mode (legacy, default — no `current_height`):
        Returns ``(final_distances_by_orientation, representativity_by_orientation)``
        — a 2-tuple, unchanged for existing callers (e.g. street widths).

    Shadow mode (when `current_height` + `building_heights` are supplied):
        Behaves like `distance_from_centroid_to_polygons_by_orientation` but
        uses many interior points instead of a single centroid, which is
        more representative for elongated / multi-wing buildings. The
        per-polygon gate compares neighbour roof elevation to the supplied
        eye level (`current_height`). Returns a dict:
            {
                "building_shadows":    {orient: aggregated_distance or inf},
                "contour":             {orient: centroid-relative ring distance},
                "vertical_distances":  {orient: aggregated vertical offset or inf},
                "elevation_at_shadow": {orient: neighbour-roof elevation or nan},
            }
        `reference_centroid` is used only to compute the canonical `contour`
        ring (so the downstream plot uses one consistent building outline);
        falls back to the unary union of `geom`'s centroid if not given.

    `aggregation` ∈ {"mean", "min"} controls how per-point distances collapse
    to a single per-orientation value. "mean" matches the legacy behaviour
    (typical visible distance); "min" gives the worst-case (closest blocker
    visible from any interior point).
    """
    shadow_mode = current_height is not None and building_heights is not None

    if not isinstance(geom, list):
        geom_list = [geom]
    else:
        geom_list = geom
    if orientations_to_analyse is not None:
        orientations = [int(float(i)) for i in orientations_to_analyse]
    else:
        orientations = list(range(0, 360, orientation_interval))
    aggregated_distances_by_orientation = {str(i): [] for i in orientations}
    aggregated_vertical_by_orientation = {str(i): [] for i in orientations}
    aggregated_elev_by_orientation = {str(i): [] for i in orientations}
    final_distances_by_orientation = {str(i): np.inf for i in orientations}
    representativity_by_orientation = {str(i): 0.0 for i in orientations}
    other_pol_intersection_points = {str(i): [] for i in orientations}
    further_point_geoms = {str(i): [] for i in orientations}
    points_log = {str(i): [] for i in orientations}
    elements = {str(i): 0 for i in orientations}

    for geom_element in geom_list:
        orientations_element = orientations
        if force_one_orientation_per_geom_element:
            orientations_element = [calculate_wall_outdoor_normal_orientation(geom_element, orientation_interval=orientation_interval)]
        random_points = generate_random_interior_points_from_geometry(
            geom_element, num_points_per_geom_element, force_projection=False)

        for angle in orientations_element:
            elements[str(orientations_element[0])] = elements[str(orientations_element[0])] + 1
            for point in random_points:

                distances_by_orientation = {str(i): np.inf for i in orientations_element}
                vertical_for_this_ray = np.inf
                elev_for_this_ray = np.nan

                ray = create_ray_from_centroid(point, angle, length=max_distance)
                further_point_geom = ray.intersection(geom_element)
                further_point_geom = get_furthest_point(further_point_geom, point) if not (
                    isinstance(further_point_geom, Point)) else further_point_geom
                if further_point_geom is None:
                    further_point_geom = point
                point_to_linearring_distance = point.distance(further_point_geom)

                nearest_point_ = point

                # Iterate preserving the original index labels when the caller
                # passes a pandas/GeoSeries — the building_heights dict in
                # shadow mode is keyed by those labels.
                if hasattr(other_polygons, "items"):
                    polygon_iter = other_polygons.items()
                else:
                    polygon_iter = enumerate(other_polygons)

                for idx, other_polygon in polygon_iter:
                    if shadow_mode:
                        info = building_heights.get(idx, {})
                        # Gate: neighbour roof must be above eye level.
                        if info.get('relative_height', 0) <= 0:
                            continue
                        if (info.get('is_nearby', False)
                                and info.get('relative_height', 0) < 3.0):
                            continue

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
                                if shadow_mode:
                                    info = building_heights.get(idx, {})
                                    vertical_for_this_ray = round(
                                        info.get('relative_height', 0), 2)
                                    elev_for_this_ray = round(
                                        info.get('height', 0), 2)
                other_pol_intersection_points[str(angle)].append(nearest_point_)
                further_point_geoms[str(angle)].append(further_point_geom)
                points_log[str(angle)].append(point)

                aggregated_distances_by_orientation[str(angle)].append(distances_by_orientation[str(angle)])
                aggregated_vertical_by_orientation[str(angle)].append(vertical_for_this_ray)
                aggregated_elev_by_orientation[str(angle)].append(elev_for_this_ray)

    def _agg(values, op):
        finite = [v for v in values if v != np.inf and not (isinstance(v, float) and np.isnan(v))]
        if not finite:
            return np.inf if op != "elev" else np.nan
        if op == "min":
            return round(float(np.min(finite)), 2)
        if op == "max":
            return round(float(np.max(finite)), 2)
        if op == "elev":
            # mean of finite elevations; nan if none observed
            return round(float(np.mean(finite)), 2)
        # default mean
        return round(float(np.mean(finite)), 2)

    for angle in orientations:
        vals = aggregated_distances_by_orientation[str(angle)]
        final_distances_by_orientation[str(angle)] = _agg(vals, aggregation)
        representativity_by_orientation[str(angle)] = (
            len([v for v in vals if v < np.inf]) / (
            elements[str(angle)] * num_points_per_geom_element if force_one_orientation_per_geom_element else
            len(geom_list) * num_points_per_geom_element)
        ) if any(v < np.inf for v in vals) else 0.0

        mask = [not i.is_empty for i in other_pol_intersection_points[str(angle)]]
        further_point_geoms[str(angle)] = [
            g for g, keep in zip(further_point_geoms[str(angle)], mask) if keep
        ]
        other_pol_intersection_points[str(angle)] = [
            g for g, keep in zip(other_pol_intersection_points[str(angle)], mask) if keep
        ]
        if plots and len(other_pol_intersection_points[str(angle)])>0 and pdf is not None:
            fig, ax = plt.subplots()
            plot_shapely_geometries(
                geometries=list(other_polygons) + [
                    LineString([p1, p2]) for p1, p2 in zip(further_point_geoms[str(angle)],
                                                       other_pol_intersection_points[str(angle)])] + [
                    points_log[str(angle)]],
                contextual_geometry=geom_list,
                title=(f"{'Building shadows' if shadow_mode else 'Distance'}, "
                       f"orientation: {angle}º"
                       + (f", floor: {floor}" if floor is not None else "")
                       + f"\nagg ({aggregation}) distance to blocker: "
                       f"{str(round(final_distances_by_orientation[str(angle)],2))}m"),
                ax=ax,
            )
            pdf.savefig(fig)
            plt.close(fig)

    if not shadow_mode:
        # Backwards-compatible return for street-width callers.
        return final_distances_by_orientation, representativity_by_orientation

    # Shadow mode: build the canonical contour ring from a single reference
    # centroid (so downstream plots draw one building outline, not a fuzzy
    # cloud of per-point contours). Falls back to the unary-union centroid.
    if reference_centroid is None:
        try:
            reference_centroid = unary_union(geom_list).centroid
        except Exception:
            reference_centroid = geom_list[0].centroid
    contour_by_orientation = {}
    for angle in orientations:
        ray = create_ray_from_centroid(reference_centroid, angle, length=max_distance)
        exit_pts = []
        for ge in geom_list:
            exit_geom = ray.intersection(ge)
            if isinstance(exit_geom, Point):
                exit_pts.append(exit_geom)
            elif not exit_geom.is_empty:
                fp = get_furthest_point(exit_geom, reference_centroid)
                if fp is not None:
                    exit_pts.append(fp)
        if exit_pts:
            furthest = max(exit_pts, key=lambda p: reference_centroid.distance(p))
            contour_by_orientation[str(angle)] = round(
                reference_centroid.distance(furthest), 2)
        else:
            contour_by_orientation[str(angle)] = 0.0

    vertical_distances = {
        str(a): _agg(aggregated_vertical_by_orientation[str(a)], aggregation)
        for a in orientations
    }
    elevation_at_shadow = {
        str(a): _agg(aggregated_elev_by_orientation[str(a)], "elev")
        for a in orientations
    }

    return {
        "building_shadows": final_distances_by_orientation,
        "contour": contour_by_orientation,
        "vertical_distances": vertical_distances,
        "elevation_at_shadow": elevation_at_shadow,
        "representativity": representativity_by_orientation,
    }

def distance_from_centroid_to_polygons_by_orientation(geom, other_polygons, centroid, orientation_interval=5,
                                                    plots=False, pdf=None, floor=None, max_distance=150,
                                                    current_height=None, building_heights=None):
    """Cast rays from `centroid` outward at `orientation_interval` steps and
    measure, for each orientation, the distance from the building contour to
    the nearest *building* obstruction.

    Terrain (DEM) shading is intentionally NOT computed here: it is
    observer-height-dependent only via a single threshold check, and we only
    need it once per building (at the top floor's eye level), not per floor.
    See `sample_terrain_profile_around_centroid` and the post-loop derivation
    in `building_inference.process_zone` for that.

    Returns a dict with keys:
        "building_shadows"     {orientation_str: distance_to_first_taller_building or inf}
        "contour"              {orientation_str: distance_from_centroid_to_building_edge}
        "vertical_distances"   {orientation_str: roof_height_above_current_floor or inf}
        "elevation_at_shadow"  {orientation_str: absolute_roof_elevation or nan}
    """
    direction_angles = list(range(0, 360, orientation_interval))
    distances_by_orientation = {str(i): np.inf for i in direction_angles}
    contour_by_orientation = {str(i): np.inf for i in direction_angles}
    vertical_distances_by_orientation = {str(i): np.inf for i in direction_angles}  # vertical distance when hitting building
    elevation_at_shadow = {str(i): np.nan for i in direction_angles}  # elevation at closest neighbor building

    for angle in direction_angles:

        ray = create_ray_from_centroid(centroid, angle, length=max_distance)
        further_point_geom = ray.intersection(geom)
        further_point_geom = get_furthest_point(further_point_geom, centroid) if not (
            isinstance(further_point_geom, Point)) else further_point_geom
        point_to_linearring_distance = centroid.distance(further_point_geom)

        contour_by_orientation[str(angle)] = round(point_to_linearring_distance,2)
        other_pol_intersection_point = None

        # Iterate preserving the original index labels (e.g. GeoSeries labels),
        # since `building_heights` is keyed by those labels in the caller.
        if hasattr(other_polygons, "items"):
            polygon_iter = other_polygons.items()
        else:
            polygon_iter = enumerate(other_polygons)
        for idx, other_polygon in polygon_iter:
            # Check if height information is available for shadow casting
            can_cast_shadow = True
            if building_heights is not None and current_height is not None:
                # Get building height info for this polygon
                building_info = building_heights.get(idx, {})
                relative_height = building_info.get('relative_height', 0)
                is_nearby = building_info.get('is_nearby', False)
                
                # Only buildings that are taller than current building can cast meaningful shadows
                # For nearby buildings (outside municipality), be more conservative 
                if relative_height <= 0:  # Same height or lower
                    can_cast_shadow = False
                elif is_nearby and relative_height < 3.0:  # Nearby buildings need significant height difference
                    can_cast_shadow = False
            
            if can_cast_shadow:
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
                            
                            # Calculate vertical distance when hitting this building
                            if building_heights is not None and current_height is not None:
                                building_info = building_heights.get(idx, {})
                                neighbor_height = building_info.get('relative_height', 0)
                                neighbor_absolute_height = building_info.get('height', 0)
                                # Vertical distance = difference in heights when collision occurs
                                vertical_distance = neighbor_height - current_height
                                vertical_distances_by_orientation[str(angle)] = round(vertical_distance, 2)
                                
                                # Store elevation at the closest neighbor building
                                elevation_at_shadow[str(angle)] = round(neighbor_absolute_height, 2)

        if plots and other_pol_intersection_point is not None and pdf is not None:
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

    return {
        "building_shadows": distances_by_orientation,
        "contour": contour_by_orientation,
        "vertical_distances": vertical_distances_by_orientation,
        "elevation_at_shadow": elevation_at_shadow,
    }


def sample_terrain_profile_around_centroid(centroid, dem_sampler, orientation_interval=5,
                                           max_distance=150.0, sample_step=5.0):
    """Sample the DEM along outgoing rays from `centroid`, returning the
    observer-independent terrain elevation profile.

    Returns (profile, sample_distances) where:
        profile = {orientation_str: [elev_at_d1, elev_at_d2, ...]}
        sample_distances = [d1, d2, ...]  (offsets from centroid, metres)

    If `dem_sampler` is None or all samples fail, returns ({}, []).
    """
    if dem_sampler is None:
        return {}, []
    sample_distances = [float(d) for d in
                        np.arange(sample_step, max_distance + 1e-6, sample_step)]
    if not sample_distances:
        return {}, []
    direction_angles = list(range(0, 360, orientation_interval))
    profile = {}
    for angle in direction_angles:
        angle_rad = math.radians(angle)
        sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)
        xs = [centroid.x + d * sin_a for d in sample_distances]
        ys = [centroid.y + d * cos_a for d in sample_distances]
        try:
            elevs = dem_sampler(xs, ys)
        except Exception:
            elevs = [None] * len(sample_distances)
        coerced = []
        for e in elevs:
            try:
                f = float(e)
                coerced.append(round(f, 2) if np.isfinite(f) else np.nan)
            except (TypeError, ValueError):
                coerced.append(np.nan)
        profile[str(angle)] = coerced
    return profile, sample_distances


def derive_terrain_shadow_from_profile(profile, sample_distances, contour_by_orientation,
                                       observer_height, clearance=0.5):
    """From a terrain elevation profile, compute per-orientation distance and
    elevation of the first point where the ground rises above
    `observer_height + clearance`.

    Returns ({orientation: distance_from_contour}, {orientation: terrain_elev}).
    Distances are inf and elevations NaN where the terrain never blocks.
    """
    shadow_dist = {}
    shadow_elev = {}
    if not profile or not sample_distances or observer_height is None:
        for k in profile.keys():
            shadow_dist[k] = np.inf
            shadow_elev[k] = np.nan
        return shadow_dist, shadow_elev
    threshold = float(observer_height) + float(clearance)
    for orientation_key, elevs in profile.items():
        blocked = False
        contour_d = contour_by_orientation.get(orientation_key, 0.0) if contour_by_orientation else 0.0
        if contour_d is None or not np.isfinite(contour_d):
            contour_d = 0.0
        for d, e in zip(sample_distances, elevs):
            if e is None or (isinstance(e, float) and np.isnan(e)):
                continue
            if e >= threshold:
                block_dist = round(d - contour_d, 2)
                if block_dist < 0:
                    block_dist = 0.0
                shadow_dist[orientation_key] = block_dist
                shadow_elev[orientation_key] = round(float(e), 2)
                blocked = True
                break
        if not blocked:
            shadow_dist[orientation_key] = np.inf
            shadow_elev[orientation_key] = np.nan
    return shadow_dist, shadow_elev


def sample_dem_grid_around_geometry(geometry, dem_sampler, cell_size=30.0, margin=150.0):
    """Sample the DEM on a regular grid covering `geometry`'s bbox + `margin`.

    Returns a dict ready to be pickled/serialised:

        {
            "x_min": float, "y_min": float,
            "cell_size": float,
            "nx": int, "ny": int,
            "elev": [[...], ...]   # row-major, rows in increasing y, cols in increasing x
        }

    The plot layer reads the dict back and reconstructs the meshgrid for a
    `contour` / `contourf` call. We deliberately store the grid in metric
    offsets rather than coordinates so a downstream consumer doesn't need to
    know the CRS to use it; absolute coordinates can be recovered by adding
    `x_min` / `y_min`.

    Returns an empty dict if `dem_sampler` is None or every sample fails.
    """
    if dem_sampler is None or geometry is None or geometry.is_empty:
        return {}
    minx, miny, maxx, maxy = geometry.bounds
    x_min = float(minx) - float(margin)
    y_min = float(miny) - float(margin)
    x_max = float(maxx) + float(margin)
    y_max = float(maxy) + float(margin)
    nx = max(2, int(math.ceil((x_max - x_min) / cell_size)) + 1)
    ny = max(2, int(math.ceil((y_max - y_min) / cell_size)) + 1)
    xs = [x_min + i * cell_size for i in range(nx)]
    ys = [y_min + j * cell_size for j in range(ny)]

    # Sample the DEM row by row so a single sampler call processes one full
    # row of points; cheaper than per-point calls.
    elev = []
    any_finite = False
    for y in ys:
        try:
            row_vals = dem_sampler(xs, [y] * nx)
        except Exception:
            row_vals = [None] * nx
        row = []
        for v in row_vals:
            try:
                f = float(v)
                if not np.isfinite(f):
                    f = float("nan")
                else:
                    any_finite = True
                row.append(round(f, 2))
            except (TypeError, ValueError):
                row.append(float("nan"))
        elev.append(row)
    if not any_finite:
        return {}
    return {
        "x_min": round(x_min, 3),
        "y_min": round(y_min, 3),
        "cell_size": float(cell_size),
        "nx": nx,
        "ny": ny,
        "elev": elev,
    }


def compute_floor_shadow_region(building_geom, nearby_buildings,
                                dem_sampler, eye_level,
                                ground_floor_height=5.0, floor_height=3.0,
                                max_distance=150.0, clearance=0.5,
                                orientation_interval=5, sample_step=2.5,
                                terrain_profile=None,
                                terrain_profile_distances=None,
                                centroid=None, buffer_disk=None):
    """Return a single (Multi)Polygon representing the area blocked from the
    floor's eye level, in plan view, within `max_distance` of the building.

    The returned region is the union of:
      - the terrain-shadow region: everything inside the `max_distance`
        buffer of the building that is *behind* a terrain rise above
        `eye_level + clearance` along the line of sight from the centroid;
      - the building-shadow region: nearby building footprints whose
        portions ABOVE the eye level intersect the visibility polygon
        (so buildings *behind* a closer terrain rise are skipped — they
        can't be seen anyway). For each nearby building, only the floors
        whose top exceeds the eye level contribute; their per-floor
        footprints are unioned and used as the blocker shape, so stepped
        buildings (wider podium + narrow tower) correctly project only
        the tower at high analyzed floors.

    The analyzed building's own footprint is subtracted so the result reads
    as "what's blocked outside this building".

    Parameters
    ----------
    building_geom : shapely Polygon or MultiPolygon
        The analyzed building's footprint.
    nearby_buildings : list[dict]
        One dict per nearby building, with keys:
            'elev'   : float — ground elevation (m AMSL).
            'floors' : dict[int, shapely geom] — per-floor footprint of the
                       neighbour. Floor 0 is the ground floor; floor k's
                       slab top elevation is `elev + g + k*h`.
        We pick the lowest floor `k_min` whose top elevation > eye_level
        and union the footprints of floors ≥ k_min as the blocker shape.
    dem_sampler : callable
        Function (xs, ys) → list of elevations (m). If None, no terrain
        shadow is computed and the return is purely building-driven.
    eye_level : float
        Mid-floor eye-level elevation (m AMSL).
    ground_floor_height : float
        Ground-floor height (m) used to compute per-floor neighbour-top
        elevations. Must match the convention used in `floor_base_elevation`.
    floor_height : float
        Upper-floor height (m), same convention.
    max_distance : float
        Radius of the disk we consider around the building (m).
    clearance : float
        Terrain must exceed `eye_level + clearance` to count as a blocker.
    orientation_interval : int
        Ray sweep step (degrees) for the terrain visibility polygon.
    sample_step : float
        DEM sampling step along each ray (m). Smaller = smoother visibility
        polygon at higher cost.

    Returns
    -------
    shapely Polygon / MultiPolygon (possibly empty).
    """
    centroid = building_geom.centroid
    # Bound the analysis to a disk of radius `max_distance` around the
    # CENTROID (not the building edge). Building.buffer(max_distance) would
    # extend `max_distance` beyond every edge, giving a region that reaches
    # ~`max_distance + building_radius` from the centroid in some
    # directions — and since the visibility polygon is built from rays of
    # length `max_distance` from the centroid, the difference of the two
    # produces a spurious "shadow" annulus at the outer rim.
    buffer_disk = centroid.buffer(max_distance, resolution=64)

    # --- Visibility polygon: star-shaped, bounded by per-direction terrain
    # blocker distance.
    if dem_sampler is not None:
        n_angles = max(8, int(round(360.0 / max(1, orientation_interval))))
        angles = [2.0 * math.pi * i / n_angles for i in range(n_angles)]
        sample_distances = [float(d) for d in
                            np.arange(sample_step, max_distance + 1e-6, sample_step)]
        boundary_pts = []
        for theta in angles:
            sin_a, cos_a = math.sin(theta), math.cos(theta)
            # Batch DEM sampling along the ray.
            xs = [centroid.x + d * sin_a for d in sample_distances]
            ys = [centroid.y + d * cos_a for d in sample_distances]
            try:
                elevs = dem_sampler(xs, ys)
            except Exception:
                elevs = [None] * len(sample_distances)
            block_d = max_distance
            threshold = eye_level + clearance
            for d, e in zip(sample_distances, elevs):
                try:
                    fe = float(e)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fe) and fe > threshold:
                    block_d = d
                    break
            boundary_pts.append((centroid.x + block_d * sin_a,
                                 centroid.y + block_d * cos_a))
        try:
            visibility_polygon = Polygon(boundary_pts).buffer(0)
            if visibility_polygon.is_empty:
                visibility_polygon = buffer_disk
        except Exception:
            visibility_polygon = buffer_disk
    else:
        visibility_polygon = buffer_disk

    # --- Terrain shadow region: buffer disk minus visibility polygon.
    try:
        terrain_shadow = buffer_disk.difference(visibility_polygon)
    except Exception:
        terrain_shadow = Polygon()

    # --- Building shadow region: per-neighbour, find the lowest floor
    # `k_min` whose top elevation exceeds the eye level, then union the
    # neighbour's footprints at floors ≥ k_min as the blocker shape (so
    # stepped buildings only project their above-eye-level portion). Clip
    # the result to the visibility polygon (buildings behind a closer
    # terrain rise drop out).
    building_pieces = []
    for nb in nearby_buildings:
        if not isinstance(nb, dict):
            continue
        floors = nb.get('floors') or {}
        if not floors:
            continue
        try:
            nb_elev = float(nb.get('elev', 0.0))
        except (TypeError, ValueError):
            nb_elev = 0.0

        # Top of floor k is at nb_elev + g + k*h. We want the smallest k
        # with that > eye_level. Solve for k > (eye_level − nb_elev − g)/h.
        # The smallest such integer is ceil(threshold), then clamp to >= 0.
        threshold = (eye_level - nb_elev - ground_floor_height) / floor_height
        # Tiny epsilon to avoid floating-point edge cases at exact equality.
        k_min = max(0, int(math.ceil(threshold + 1e-9)))

        relevant_floors = [k for k in sorted(floors.keys()) if k >= k_min]
        if not relevant_floors:
            continue

        try:
            blocker = unary_union([
                floors[k] for k in relevant_floors
                if floors[k] is not None and not floors[k].is_empty
            ])
        except Exception:
            continue
        if blocker is None or blocker.is_empty:
            continue
        try:
            clipped = blocker.intersection(visibility_polygon)
        except Exception:
            continue
        if clipped is None or clipped.is_empty:
            continue
        building_pieces.append(clipped)

    if building_pieces:
        try:
            building_shadow = unary_union(building_pieces)
        except Exception:
            building_shadow = Polygon()
    else:
        building_shadow = Polygon()

    # --- Combine and subtract the analyzed building's own footprint.
    try:
        combined = unary_union([terrain_shadow, building_shadow])
    except Exception:
        combined = terrain_shadow
    try:
        combined = combined.difference(building_geom)
    except Exception:
        pass

    if combined is None or combined.is_empty:
        return Polygon()
    if isinstance(combined, Polygon):
        return combined
    if isinstance(combined, MultiPolygon):
        return combined
    # Fallback: keep only polygonal pieces.
    polys = [g for g in getattr(combined, 'geoms', []) if isinstance(g, (Polygon, MultiPolygon))]
    if not polys:
        return Polygon()
    return unary_union(polys)


def get_streets_open_street_maps(open_street_dir, ine_code=None, cadaster_code=None, query_location=None,
                                      crs=None):

    if not query_location:
        if not ine_code:
            if cadaster_code:
                ine_code = cadaster_to_ine_codes([cadaster_code])[0]
            else:
                raise("One of these should be defined: ine_code, cadaster_code or query_location.")
        mun = municipality_name(ine_code)
        if isinstance(mun, list):
            if not mun:
                raise ValueError(f"No municipality name found for INE code {ine_code}")
            mun = mun[0]
        query_location = f"{mun}, Spain"

    os.makedirs(open_street_dir, exist_ok=True)
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
        # Clean layer name for GPKG compatibility (remove special characters and spaces)
        import re
        clean_layer_name = re.sub(r'[^a-zA-Z0-9_]', '_', query_location.replace(' ', '_'))
        streets_gdf.to_file(fn, driver="GPKG", layer=clean_layer_name)
    else:
        # Try to read existing file - check available layers first
        import re
        clean_layer_name = re.sub(r'[^a-zA-Z0-9_]', '_', query_location.replace(' ', '_'))
        
        try:
            # Check what layers exist in the file
            available_layers = fiona.listlayers(fn)
            
            # Try to use the clean layer name first
            if clean_layer_name in available_layers:
                streets_gdf = gpd.read_file(fn, layer=clean_layer_name)
            # Fall back to the original query_location if it exists as a layer
            elif query_location in available_layers:
                streets_gdf = gpd.read_file(fn, layer=query_location)
            # Use the first available layer as fallback
            elif available_layers:
                streets_gdf = gpd.read_file(fn, layer=available_layers[0])
            else:
                # If no layers, read without specifying layer (use default)
                streets_gdf = gpd.read_file(fn)
                
        except Exception:
            # If all else fails, try reading without specifying layer
            streets_gdf = gpd.read_file(fn)

    if crs is not None:
        streets_gdf.geometry = streets_gdf.geometry.to_crs(crs)

    return streets_gdf

def build_osmnx_tags(osm_tags_structured):
    tags = {}

    for main_key, subgroups in osm_tags_structured.items():
        values = set()
        for subgroup_values in subgroups.values():
            values.update(subgroup_values)
        tags[main_key] = list(values)

    return tags


def build_poi_columns(osm_tags_structured):
    col_map = {}

    for main_key, subgroups in osm_tags_structured.items():
        col_map.setdefault(main_key, {})

        for sub_key, values in subgroups.items():
            col_name = f"{main_key}.{sub_key}" if sub_key != "all" else main_key

            for v in values:
                col_map[main_key][v] = col_name

    return col_map

OSM_TAGS_STRUCTURED = {
        "amenity": {
            "food": ["restaurant", "cafe", "bar", "pub", "fast_food", "food_court"],
            "education": ["school", "college", "university", "kindergarten", "library"],
            "transport": ["bus_station", "ferry_terminal", "taxi", "parking",
                          "bicycle_parking", "charging_station"],
            "financial": ["bank", "atm", "bureau_de_change"],
            "health": ["hospital", "clinic", "doctors", "dentist", "pharmacy"],
            "culture": ["cinema", "theatre", "arts_centre", "community_centre"],
            "public": ["townhall", "courthouse", "police", "fire_station", "post_office"],
            "facilities": ["toilets", "drinking_water", "shower", "bench", "shelter"],
            "waste": ["waste_basket", "waste_disposal", "recycling"],
            "other": ["place_of_worship", "crematorium", "funeral_hall"]
        },
        "building": {
            "residential": ["apartments", "house", "residential", "detached", "terrace"],
            "commercial": ["commercial", "retail", "office", "industrial"],
            "religious": ["church", "mosque", "temple", "cathedral", "chapel"],
            "civic": ["school", "hospital", "government", "public"],
            "agriculture": ["farm", "barn", "greenhouse"],
            "sports": ["stadium", "sports_hall", "pavilion"],
            "storage": ["warehouse", "storage_tank"],
            "vehicle": ["garage", "carport"],
            "technical": ["transformer_tower", "service", "power"],
            "other": ["yes", "construction", "roof"]
        },
        "emergency": {"all": ["ambulance_station", "defibrillator", "fire_hydrant",
                              "lifeguard", "assembly_point", "phone"]},
        "healthcare": {"all": ["hospital", "clinic", "doctor", "dentist",
                               "physiotherapist", "laboratory"]},
        "historic": {"all": ["monument", "memorial", "archaeological_site",
                             "castle", "ruins"]},
        "leisure": {"all": ["park", "garden", "playground", "sports_centre",
                            "fitness_centre", "stadium", "swimming_pool", "pitch"]},
        "office": {"all": ["company", "government", "insurance",
                           "architect", "engineer", "lawyer"]},
        "shop": {
            "food": ["supermarket", "convenience", "bakery", "butcher",
                     "greengrocer", "beverages"],
            "general": ["department_store", "mall", "general"],
            "clothing": ["clothes", "shoes", "fashion", "jewelry"],
            "health": ["pharmacy", "cosmetics", "optician"],
            "diy": ["hardware", "doityourself", "garden_centre"],
            "electronics": ["electronics", "mobile_phone", "computer"],
            "sports": ["sports", "bicycle", "outdoor"],
            "culture": ["books", "music", "art", "hobby"],
            "other": ["pet", "gift", "florist"]
        },
        "tourism": {
            "accommodation": ["hotel","hostel","guest_house","apartment","motel","resort","chalet","alpine_hut",
                              "wilderness_hut"],
            "camping": ["camp_site","caravan_site"],
            "culture": ["museum","gallery","exhibition","artwork"],
            "attraction": ["attraction","viewpoint","picnic_site"],
            "leisure": ["theme_park","zoo","aquarium","spa","beach"],
            "information": ["information"],
            "rental": ["bicycle_rental","boat_rental","car_rental"]
        }
    }

POIS_COLS = {
    "general": [
        "element",
        "id",
        "geometry",
        "name",
        "alt_name",
        "short_name",
        "brand",
        "operator",
        "operator:type",
        "addr:housenumber",
        "addr:street",
        "addr:postcode",
        "addr:city",
        "access",
        "wheelchair",
        "phone",
        "website",
        "email",
        "wikidata",
        "wikipedia",
        "opening_hours",
        "ref",
        "source",
        "check_date",
        "level",
        "layer"
    ],
    "amenity": {
        "general": [
            "amenity",
            "toilets"
            "outdoor_seating",
            "indoor_seating",
            "internet_access"
        ],
        "subgroups": {
            "food": [
                "cuisine",
                "takeaway",
                "delivery",
                "diet:vegetarian",
                "diet:vegan",
                "drive_through"
            ],
            "education": [
                "isced:level",
                "min_age",
                "max_age"
            ],
            "transport": [
                "public_transport",
                "capacity",
                "parking",
                "fee"
            ],
            "financial": [
                "cash_in",
                "cash_out"
            ],
            "health": [
                "dispensing"
            ],
            "culture": [
                "fee",
                "capacity",
                "theatre:genre"
            ],
            "public": [],
            "facilities": [
                "fee"
            ],
            "waste": [
                "recycling_type",
            ],
            "other": [
                "religion",
                "denomination"
            ]
        }
    },
    "building": {
        "general": [
            "building",
            "building:use",
            "building:levels",
            "building:levels:underground",
            "height"
        ],
        "subgroups": {
            "residential": [
                "building:flats",
                "rooms",
                "building:material"
            ],
            "commercial": [
                "retail"
                "industrial",
                "commercial",
                "office"
            ],
            "religious": [
                "religion",
                "denomination",
                "service_times"
            ],
            "civic": [
                "public_transport",
                "community_centre",
                "social_facility",
                "capacity"
            ],
            "agriculture": [
                "landuse",
                "garden:type"
            ],
            "sports": [
                "capacity"
            ],
            "storage": [
                "industrial",
                "storage",
                "content"
            ],
            "vehicle": [
                "parking",
                "motorcycle_parking"
                "capacity",
                "fee"
            ],
            "technical": [
                "power",
                "substation",
                "utility"
            ],
            "other": []
        }
    },
    "emergency": {
        "general": [
            "emergency",
            "emergency:location",
            "defibrillator:location",
            "fire_hydrant:type",
            "fire_hydrant:diameter"
        ],
        "subgroups": {}
    },
    "healthcare": {
        "general": [
            "healthcare",
            "healthcare:speciality",
            "health_facility:type",
            "medical_system:western"
        ],
        "subgroups": {}
    },
    "historic": {
        "general": [
            "historic",
            "memorial",
            "artwork_type",
            "statue",
            "monument"
        ],
        "subgroups": {}
    },
    "leisure": {
        "general": [
            "leisure",
            "sport"
        ],
        "subgroups": {}
    },
    "office": {
        "general": [
            "office",
            "government",
            "designation",
            "branch",
            "member_of",
            "company"
        ],
        "subgroups": {}
    },
    "shop": {
        "general": [
            "shop",
            "payment:cash",
            "payment:credit_cards",
            "payment:debit_cards",
            "payment:contactless"
        ],
        "subgroups": {
             "food": [
                "cuisine",
                "organic",
                "diet:vegetarian",
                "diet:vegan",
                "diet:gluten_free",
                "bulk_purchase"
            ],
            "general": [
                "toilets",
                "toilets:wheelchair"
            ],
            "clothing": [
                "clothes",
                "shoes",
                "fashion_accessories",
                "second_hand"
            ],
            "health": [
                "healthcare:speciality"
            ],
            "diy": [
                "craft",
                "bulk_purchase"
            ],
            "electronics": [
                "repair",
            ],
            "sports": [
                "service:bicycle:repair",
                "service:bicycle:rental",
                "service:bicycle:retail",
                "rental",
                "repair"
            ],
            "culture": [
                "books",
                "second_hand",
                "internet_access"
            ],
            "other": []
        }
    },
    "tourism": {
        "general": [
            "tourism",
            "stars",
            "rooms",
            "check_in",
            "check_out",
            "reservation",
            "fee"
        ],
        "subgroups": {
            "accommodation": [
                "capacity",
                "air_conditioning",
                "parking",
                "elevator"
            ],
            "attraction": [
                "historic",
                "heritage",
                "description"
            ],
            "culture": [
                "museum",
                "museum_type",
                "artwork_type",
                "artist_name",
                "historic",
                "heritage",
            ],
            "camping": [
                "capacity",
                "internet_access",
                "shower",
                "toilets"
            ],
            "leisure": [
                "capacity"
            ],
            "information": [
                "information"
            ]
        }
    }
}

# ------------------------------------------------------------
# 6️⃣ MAIN PIPELINE
# ------------------------------------------------------------
def get_pois_open_street_maps(
        ine_code=None,
        cadaster_code=None,
        query_location=None,
        geolocate_to_cadaster=True,
        cadaster_dir=None,
        open_street_dir="./osm_cache",
        buffer_m=200,
        crs_out="EPSG:25831"
):

    if not query_location:
        if not ine_code:
            if cadaster_code:
                ine_code = cadaster_to_ine_codes([cadaster_code])[0]
            else:
                raise("One of these should be defined: ine_code, cadaster_code or query_location.")
            mun = municipality_name(ine_code)
            query_location=f"{mun}, Spain"

    os.makedirs(f"{open_street_dir}/POIS_{query_location}", exist_ok=True)

    ox.settings.timeout = 300
    ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"

    # ------------------------------------------------------------
    # 1️⃣ GEOMETRY
    # ------------------------------------------------------------
    gdf_place = ox.geocode_to_gdf(query_location)

    gdf_place = gdf_place[gdf_place.geometry.notna()]
    gdf_place.set_geometry("geometry")
    gdf_place = gdf_place.to_crs(3857)

    polygon = gdf_place.geometry.union_all()
    polygon = polygon.buffer(buffer_m)

    if polygon.is_empty:
        raise ValueError("Polygon is empty")

    polygon_wgs = gpd.GeoSeries([polygon], crs=3857).to_crs(4326).iloc[0]

    # ------------------------------------------------------------
    # 2️⃣ ITERATE SUBGROUPS
    # ------------------------------------------------------------
    results = {}
    total_subgroups = sum(len(v) for v in OSM_TAGS_STRUCTURED.values())
    bar = tqdm(total=total_subgroups, desc="Loading POI subgroups",
               mininterval=1.0, leave=True, ncols=80)

    for main_key, subgroups in OSM_TAGS_STRUCTURED.items():

        for sub_key, values in subgroups.items():

            subgroup_name = f"{main_key}.{sub_key}" if sub_key != "all" else main_key
            fn = f"{open_street_dir}/POIS_{query_location}/{subgroup_name}.gpkg"

            # ----------------------------------------------------
            # LOAD CACHE
            # ----------------------------------------------------
            if os.path.exists(fn):
                try:
                    gdf = gpd.read_file(fn)
                    results[subgroup_name] = gdf
                    bar.update(1)
                    continue
                except Exception:
                    pass

            # ----------------------------------------------------
            # QUERY
            # ----------------------------------------------------
            tags = {main_key: values}

            try:
                gdf = ox.features_from_polygon(polygon_wgs, tags=tags)

                if gdf.empty:
                    bar.update(1)
                    continue

                gdf = gdf.reset_index()
                gdf = gdf[~gdf.geometry.isna()]

                # Normalize geometry - transform to projected CRS before centroid calculation
                gdf = gdf.to_crs(crs_out)
                gdf["geometry"] = gdf.geometry.centroid

                # Add metadata column
                gdf["poi_subgroup"] = subgroup_name

                # Save
                gdf.to_file(fn, driver="GPKG")

                results[subgroup_name] = gdf

            except Exception as e:
                # Surface genuine failures to stderr — they're rare and signal
                # a real problem (network, tag config, etc.).
                sys.stderr.write(f"[POI FAIL] {subgroup_name}: {e}\n")
            finally:
                bar.update(1)

    bar.close()


    results = clean_pois(results)

    # -------------------
    # GEOLOCATE TO CADASTRAL REFERENCES
    # -------------------
    if geolocate_to_cadaster:
        if not cadaster_code or not ine_code:
            if ine_code and not cadaster_code:
                cadaster_code = ine_to_cadaster_codes([ine_code])[0]

        if not cadaster_dir:
            raise ValueError("Cadaster directory (cadaster_dir) must be defined.")

        # Load parcels data
        from hypercadaster_ES.mergers import join_cadaster_parcel

        parcels_gdf = join_cadaster_parcel(None, cadaster_dir, [cadaster_code], how="left")
        parcels_gdf = parcels_gdf.set_geometry("parcel_geometry")
        parcels_gdf = parcels_gdf.to_crs(crs_out)

        def geolocate_leaf_gdf(poi_gdf, subgroup_name):
            """
            Apply cadastral geolocation to a single GeoDataFrame leaf.
            """
            if poi_gdf is None or poi_gdf.empty:
                return poi_gdf

            poi_gdf = poi_gdf.to_crs(crs_out)

            # Spatial join: POIs inside parcels
            poi_with_parcels = gpd.sjoin(
                poi_gdf,
                parcels_gdf[["building_reference", "parcel_geometry"]],
                how="left",
                predicate="within"
            )

            # Fallback: nearest parcel for unmatched POIs
            no_parcel_mask = poi_with_parcels["building_reference"].isna()

            if no_parcel_mask.any():
                pois_no_parcel = poi_with_parcels[no_parcel_mask].copy()

                parcel_centroids = parcels_gdf.copy()
                parcel_centroids["geometry"] = parcel_centroids["parcel_geometry"].centroid
                parcel_centroids = parcel_centroids.set_geometry("geometry")

                cols = [
                    col for col in pois_no_parcel.columns
                    if col not in ["building_reference", "index_right"]
                ]

                nearest_parcels = gpd.sjoin_nearest(
                    pois_no_parcel[cols],
                    parcel_centroids[["building_reference", "geometry"]],
                    how="left"
                )

                poi_with_parcels.loc[
                    no_parcel_mask, "building_reference"
                ] = nearest_parcels["building_reference"].values

            poi_with_parcels = poi_with_parcels.drop(
                columns=["index_right"], errors="ignore"
            )

            return poi_with_parcels

        def recursive_process(obj, path="root"):
            """
            Traverse nested dictionaries until leaf GeoDataFrames are found.
            """
            if isinstance(obj, dict):
                return {
                    key: recursive_process(value, f"{path}.{key}")
                    for key, value in obj.items()
                }

            elif isinstance(obj, gpd.GeoDataFrame):
                return geolocate_leaf_gdf(obj, path)

            else:
                # leave any non-dict / non-gdf objects untouched
                return obj

        # Process nested results structure
        results = recursive_process(results)

    return results

def clean_pois(pois_gdf: dict) -> dict:
    """
    Returns cleaned dataframes per POIS group
    with consistent schema.
    """
    outputs = {}

    for group, df in pois_gdf.items():

        # Get schema
        try:
            group, subgroup = group.split(".")
        except ValueError:
            subgroup = None

        # Merge shared + specific
        target_cols = (POIS_COLS["general"] +
                       POIS_COLS[group]["general"] +
                       POIS_COLS[group]["subgroups"][subgroup] if subgroup else [])

        # Keep only existing columns
        existing_cols = [c for c in target_cols if c in df.columns]

        # Select + copy
        out_df = df[existing_cols].copy()
        if subgroup:
            if not group in outputs.keys():
                outputs[group] = {}
            outputs[group][subgroup] = out_df
        else:
            outputs[group] = out_df

    return outputs


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

# ==========================================
# ADDRESS COMPACTION UTILITIES
# ==========================================

def _create_full_street_names(gdf):
    """
    Helper function to create full street name combinations.
    
    Creates the following columns if the required components exist:
    - street_full_label: "{street_type} {street_name}, {street_number_clean_label}"
    - street_full_clean: "{street_type} {street_name}, {street_number_clean}"
    """
    gdf_result = gdf.copy()
    
    # Define the combinations to create
    street_formats = {
        'street_full_labels': ['street_type', 'street_name', 'street_number_label'],
        'street_full_clean': ['street_type', 'street_name', 'street_number_clean']
    }
    
    created_cols = []
    
    for col_name, required_cols in street_formats.items():
        # Check if all required columns exist
        if all(col in gdf_result.columns for col in required_cols):
            try:
                # Create the full street name
                gdf_result[col_name] = (
                    gdf_result[required_cols[0]].fillna('').astype(str) + ' ' +
                    gdf_result[required_cols[1]].fillna('').astype(str) + ', ' + 
                    gdf_result[required_cols[2]].fillna('').astype(str)
                )
                # Clean up extra spaces and commas
                gdf_result[col_name] = gdf_result[col_name].str.replace(r'\s+', ' ', regex=True)
                gdf_result[col_name] = gdf_result[col_name].str.replace(r',\s*$', '', regex=True)
                gdf_result[col_name] = gdf_result[col_name].str.strip()
                
                created_cols.append(col_name)
            except Exception:
                pass  # Silently skip failed column creation
    
    return gdf_result

def normalize_street_number(series):
    """
    Returns:
    - label: original string cleaned
    - clean: ordered numeric representation (e.g. "1-3")
    """

    def extract_numbers(s):
        if pd.isna(s):
            return None, None

        s = str(s).upper().strip()

        # extract all numbers
        nums = re.findall(r'\d+', s)
        if not nums:
            return s, None

        nums = sorted(set(int(n) for n in nums))

        if len(nums) == 1:
            clean = str(nums[0])
        else:
            clean = f"{nums[0]}-{nums[-1]}"

        return s, clean

    labels = []
    cleans = []

    for val in series:
        label, clean = extract_numbers(val)
        labels.append(label)
        cleans.append(clean)

    return pd.Series(labels), pd.Series(cleans)

def is_valid_value(val):
    if val is None:
        return False

    if isinstance(val, float) and pd.isna(val):
        return False

    if isinstance(val, (list, np.ndarray)):
        return len(val) > 0

    return True

def compact_addresses(
    gdf,
    building_reference_col='building_reference',
    address_cols=[
        'street_type', 'street_name', 'street_number_clean', 'street_number_label'
    ],
    geolocation_cols=['address_location'],
    admin_cols=[
        'district_code', 'district_name', 'postal_code',
        'neighborhood_code', 'neighborhood_name', 'section_code'
    ],
    create_full_street_names=True,
    use_unique_for_admin=True):

    sys.stderr.write(f"\nCompacting the addresses (Polars optimized)\n")

    if gdf.empty:
        return gdf

    gdf_work = gdf.copy()

    # --------------------------------------------------
    # 1️⃣ Street names
    # --------------------------------------------------
    if create_full_street_names:
        gdf_work = _create_full_street_names(gdf_work)

    # --------------------------------------------------
    # 2️⃣ Geometry split
    # --------------------------------------------------
    geometry_cols = [c for c in gdf_work.columns if str(gdf_work[c].dtype) == "geometry"]
    df_no_geom = pd.DataFrame(gdf_work.drop(columns=geometry_cols))
    street_full_cols = [c for c in df_no_geom.columns if c.startswith("street_full")]

    # --------------------------------------------------
    # 3️⃣ Polars columns
    # --------------------------------------------------
    cols_to_compact = list(set(address_cols + admin_cols + street_full_cols))
    cols_to_compact = [c for c in cols_to_compact if c in df_no_geom.columns]
    pl_df = pl.from_pandas(df_no_geom[cols_to_compact + [building_reference_col]])

    # --------------------------------------------------
    # 4️⃣ Aggregations
    # --------------------------------------------------
    agg_exprs = []

    for col in pl_df.columns:

        if col == building_reference_col:
            continue

        elif col in address_cols:
            agg_exprs.append(pl.col(col).drop_nulls().implode().alias(col))

        elif col in admin_cols:
            if use_unique_for_admin:
                agg_exprs.append(
                    pl.col(col)
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .sort()
                    .alias(col)
                )
            else:
                agg_exprs.append(pl.col(col).drop_nulls().implode().alias(col))
        elif col.startswith("street_full"):
            agg_exprs.append(
                pl.col(col)
                .drop_nulls()
                .unique(maintain_order=True)
                .alias(col)
            )

    # --------------------------------------------------
    # 5️⃣ Polars groupby
    # --------------------------------------------------
    pl_grouped = pl_df.group_by(building_reference_col).agg(agg_exprs)
    df_compacted = pl_grouped.to_pandas()

    # Fix numpy arrays → lists
    for col in df_compacted.columns:
        df_compacted[col] = df_compacted[col].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

    # --------------------------------------------------
    # 6️⃣ Geolocation (list)
    # --------------------------------------------------
    if geolocation_cols:
        geo_df = (
            gdf_work[[building_reference_col] + geolocation_cols]
            .groupby(building_reference_col, as_index=False)
            .agg({col: lambda x: list(x.dropna()) for col in geolocation_cols})
        )

        df_compacted = df_compacted.drop(columns=geolocation_cols, errors="ignore")

        df_compacted = df_compacted.merge(geo_df, on=building_reference_col, how="left")

    # --------------------------------------------------
    # 7️⃣ Non-compacted columns (KEEP THEM)
    # --------------------------------------------------
    non_compact_cols = [
        col for col in gdf_work.columns
        if col not in address_cols
           and col not in admin_cols
           and col not in street_full_cols
           and col not in geolocation_cols
           and col != building_reference_col
    ]

    if non_compact_cols:
        gdf_work = gdf_work.copy()
        non_compact_df = (
            gdf_work
            .sort_values(building_reference_col)
            [[building_reference_col] + non_compact_cols]
            .drop_duplicates(subset=[building_reference_col], keep="first")
        )

        df_compacted = df_compacted.merge(non_compact_df, on=building_reference_col, how="left")

    # --------------------------------------------------
    # 9️⃣ Drop redundant street cols
    # --------------------------------------------------
    if create_full_street_names:
        df_compacted = df_compacted.drop(columns=[c for c in address_cols if c in df_compacted.columns])

    # --------------------------------------------------
    # 🔟 GeoDataFrame
    # --------------------------------------------------
    df_compacted = gpd.GeoDataFrame(
        df_compacted,
        geometry="building_geometry",
        crs=gdf_work.crs
    )

    return df_compacted

# ==========================================
# SHADOW VISUALIZATION
# ==========================================

def plot_building_shadows_analysis(gdf, building_reference, output_path=None, interactive=True,
                                   include_elevation=True, floor_height=None, wd=None):
    """
    Create comprehensive shadow analysis visualization for a building.

    This function creates multiple visualizations showing:
    - Polar plot of shadow distances by orientation
    - Building contour vs shadow distances
    - Elevation profile effects on shadows
    - Multi-floor shadow comparison
    - Interactive 3D shadow casting visualization

    Parameters:
    -----------
    gdf : pandas.DataFrame or geopandas.GeoDataFrame
        Results DataFrame containing building shadow analysis results
    building_reference : str
        Building reference ID to analyze
    output_path : str, optional
        Path to save the plot (PDF or HTML). If None, displays interactively
    interactive : bool, default True
        Whether to create interactive plots (using plotly) or static plots (matplotlib)
    include_elevation : bool, default True
        Whether to include elevation data in the analysis
    floor_height : float, optional
        Height per floor in meters for multi-floor analysis. If None, automatically
        detects from shadow data or defaults to 3.0m
    wd : str, optional
        Working directory of the central DuckDB. When provided, missing
        heavy columns (`br__shadows`, `br__shadows_at_distance`,
        `br__terrain_grid` — excluded from the bulk `hc.merge` result
        for memory reasons) are auto-fetched for this single building
        before plotting. Without `wd`, the plot degrades gracefully but
        the per-floor shadow + DEM panels stay empty.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        If interactive=True and output_path=None, returns the plotly figure
    """
    import numpy as np
    import pandas as pd

    # Filter data for the specific building
    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None

    # Get building data
    building_row = building_data.iloc[0]

    # Auto-hydrate heavy BLOB columns from the central DB if missing.
    # `hc.merge` deliberately excludes them so the bulk frame stays small;
    # for a single-building plot they fit easily.
    if wd is not None:
        from hypercadaster_ES import _db
        cadaster_code = building_row.get("cadaster_code")
        if cadaster_code is None and "cadaster_code" in building_row.index:
            cadaster_code = building_row["cadaster_code"]
        if cadaster_code is not None:
            missing = [
                c for c in _db.HEAVY_BLOB_COLS
                if c.startswith("br__")
                and (c not in building_row.index or building_row[c] is None)
            ]
            if missing:
                try:
                    fetched = get_heavy_columns_for_building(
                        wd, str(cadaster_code), building_reference,
                        table="br_results", columns=missing,
                    )
                    # `building_row` is a Series view; mutate a copy so
                    # downstream uses see the hydrated values without
                    # touching the caller's gdf.
                    building_row = building_row.copy()
                    for col, val in fetched.items():
                        building_row[col] = val
                except Exception as _hydrate_err:
                    print(f"Warning: could not auto-hydrate heavy cols "
                          f"for {building_reference}: {_hydrate_err}")
    
    # Auto-detect floor height if not provided
    if floor_height is None:
        floor_height = _detect_floor_height_from_data(building_row)
    
    # If the user asked for a raster/vector still (.png/.pdf/.svg/.jpg/.jpeg),
    # always go through matplotlib — plotly's fig.write_image() depends on
    # kaleido, which SIGABRTs on Linux when the bundled Chromium can't start.
    # Plotly is reserved for .html output or interactive display (output_path=None).
    static_exts = ('.png', '.pdf', '.svg', '.jpg', '.jpeg')
    if isinstance(output_path, str) and output_path.lower().endswith(static_exts):
        return _create_static_shadow_plot(building_row, building_reference, output_path,
                                          include_elevation, floor_height)

    if interactive:
        return _create_interactive_shadow_plot(building_row, building_reference, output_path,
                                               include_elevation, floor_height)
    return _create_static_shadow_plot(building_row, building_reference, output_path,
                                      include_elevation, floor_height)




def _floor_eye_levels(shadow_data, floor_height, ground_floor_height=5.0):
    """Absolute eye-level elevations (m) for each floor of the analyzed building.

    floor 0 sits at ground+ground_floor_height/2; subsequent floors at
    ground+ground_floor_height+(i-1)*floor_height + floor_height/2. This
    matches `floor_base_elevation` semantics used by the inference side.
    """
    ground = float(shadow_data.get('elevation', 0.0) or 0.0)
    n = int(shadow_data.get('max_neighbor_floors', shadow_data.get('building_n_floors', 1)) or 1)
    levels = []
    for i in range(n):
        if i == 0:
            levels.append(ground + ground_floor_height / 2.0)
        else:
            levels.append(ground + ground_floor_height + (i - 1) * floor_height + floor_height / 2.0)
    return levels


def _pick_worst_orientation(shadow_data):
    """Return the orientation key whose top-floor shadow distance is smallest
    (finite). Falls back to '180' (south) if no finite shadow exists.
    """
    bs = shadow_data.get('building_shadows')
    if not bs:
        return '180'
    best_key = None
    best_d = float('inf')
    for k, lst in bs.items():
        if not isinstance(lst, list) or not lst:
            continue
        v = lst[-1]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v < best_d:
            best_d = v
            best_key = k
    return best_key or '180'


def _create_interactive_shadow_plot(building_row, building_reference, output_path,
                                    include_elevation, floor_height):
    """Interactive 3-panel shadow + terrain visualization using plotly.

    Panels:
        (1) polar  — per-floor combined shadow distance (building roof OR
                     rising terrain), measured from the building contour.
        (2) xy     — same per-floor shadow distances vs orientation, linear axis.
        (3) heatmap — DEM contour map from the 5 m grid sample around the
                     building.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return _create_static_shadow_plot(building_row, building_reference, output_path,
                                          include_elevation, floor_height)

    shadow_data = _extract_shadow_data(building_row)
    if not shadow_data:
        print(f"No shadow data found for building {building_reference}")
        return None

    orientations = shadow_data['orientations']
    orient_keys = [str(int(o)) for o in orientations]
    contour = shadow_data.get('contour_distances') or [0.0] * len(orientations)
    sh = shadow_data.get('shadows', {})
    max_floors = int(shadow_data.get('max_neighbor_floors', 1) or 1)
    bldg_floors = int(shadow_data.get('building_n_floors', max_floors) or max_floors)
    cmap = _floor_colors(max_floors)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Per-floor shadow distance (polar)',
            'Per-floor shadow distance vs orientation',
            'DEM contour map (5 m grid)',
            None,
        ),
        specs=[[{"type": "polar"}, {"type": "xy"}],
               [{"type": "heatmap"}, None]],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    # ----- Panel 1: building contour + per-floor shadow distances (polar) -----
    fig.add_trace(go.Scatterpolar(
        r=contour + [contour[0]],
        theta=orient_keys + [orient_keys[0]],
        mode='lines', name='Building contour',
        line=dict(color='black', width=2),
        fill='toself', fillcolor='rgba(0,0,0,0.05)',
        hovertemplate='Orientation %{theta}°<br>Contour radius %{r:.1f} m<extra></extra>',
    ), row=1, col=1)

    for floor_idx in range(max_floors):
        r_total = []
        for o_idx, key in enumerate(orient_keys):
            shadow_list = sh.get(key, [])
            if floor_idx < len(shadow_list) and np.isfinite(shadow_list[floor_idx]):
                r_total.append(float(contour[o_idx]) + float(shadow_list[floor_idx]))
            else:
                r_total.append(None)
        is_neighbor_extra = floor_idx >= bldg_floors
        fig.add_trace(go.Scatterpolar(
            r=r_total + [r_total[0]],
            theta=orient_keys + [orient_keys[0]],
            mode='lines+markers',
            name=f'Floor {floor_idx}' + (' (neighbor only)' if is_neighbor_extra else ''),
            line=dict(color=cmap[floor_idx], width=1.5,
                      dash='dot' if is_neighbor_extra else 'solid'),
            marker=dict(size=4),
            hovertemplate=(f'Floor {floor_idx}<br>Orientation %{{theta}}°'
                           '<br>Blocker at %{r:.1f} m<extra></extra>'),
        ), row=1, col=1)

    # ----- Panel 2: same on a linear axis ------------------------------------
    for floor_idx in range(max_floors):
        ys = []
        for key in orient_keys:
            shadow_list = sh.get(key, [])
            if floor_idx < len(shadow_list) and np.isfinite(shadow_list[floor_idx]):
                ys.append(float(shadow_list[floor_idx]))
            else:
                ys.append(None)
        is_neighbor_extra = floor_idx >= bldg_floors
        fig.add_trace(go.Scatter(
            x=orientations, y=ys, mode='lines+markers',
            name=f'Floor {floor_idx}'
                 + (' (neighbor only)' if is_neighbor_extra else ''),
            line=dict(color=cmap[floor_idx], width=1.5,
                      dash='dot' if is_neighbor_extra else 'solid'),
            showlegend=False,
            hovertemplate=(f'Floor {floor_idx}<br>Orientation %{{x}}°'
                           '<br>Shadow distance %{y:.1f} m<extra></extra>'),
        ), row=1, col=2)

    # ----- Panel 3: DEM contour map from the 5 m grid -------------------------
    grid = shadow_data.get('terrain_grid')
    if grid and grid.get('elev'):
        cell = float(grid['cell_size'])
        nx = int(grid['nx']); ny = int(grid['ny'])
        elev = np.array(grid['elev'], dtype=float)
        cx_off = (nx - 1) * cell / 2.0
        cy_off = (ny - 1) * cell / 2.0
        gx = np.arange(nx) * cell - cx_off
        gy = np.arange(ny) * cell - cy_off
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=elev,
            colorscale='Earth',
            contours=dict(showlines=True, coloring='heatmap'),
            colorbar=dict(title='Elev (m)', x=0.46, len=0.4, y=0.22),
            hovertemplate='x=%{x:.0f} m<br>y=%{y:.0f} m<br>elev=%{z:.1f} m<extra></extra>',
        ), row=2, col=1)
    fig.update_xaxes(title_text='East offset (m)', row=2, col=1, scaleanchor='y2', scaleratio=1)
    fig.update_yaxes(title_text='North offset (m)', row=2, col=1)

    fig.update_layout(
        title=(f'Shadow & terrain analysis for {building_reference}'
               f'  (building floors: {bldg_floors}, max floor seen: {max_floors})'),
        height=900, width=1200, showlegend=True,
        legend=dict(orientation='v', x=1.02, y=1.0),
    )
    fig.update_xaxes(title_text='Orientation (°)', row=1, col=2)
    fig.update_yaxes(title_text='Shadow distance (m)', row=1, col=2)

    if output_path:
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            try:
                fig.write_image(output_path)
            except Exception as e:
                print(f"Could not write {output_path} ({e}); falling back to HTML")
                fig.write_html(output_path.rsplit('.', 1)[0] + '.html')
        print(f"Shadow analysis saved to {output_path}")
        return fig
    fig.show()
    return fig


def _floor_colors(n):
    """Return n distinct colors interpolated through a viridis-like ramp."""
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        cmap = cm.get_cmap('viridis')
        if n <= 1:
            return [mcolors.to_hex(cmap(0.5))]
        return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
    except Exception:
        # Reasonable fallback gradient if matplotlib not available
        base = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
        if n <= len(base):
            return base[:n]
        return [base[i * (len(base) - 1) // max(1, n - 1)] for i in range(n)]


def _draw_polygon_cartesian(ax, geom, cx, cy, *, facecolor, edgecolor,
                             alpha, linewidth=0.5, hatch=None):
    """Render a (Multi)Polygon onto a cartesian axis, recentred on (cx, cy).
    Uses PathPatch so holes render correctly. No-op for empty geometry.
    """
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def _poly_to_path(p):
        ex = [(x - cx, y - cy) for x, y in p.exterior.coords]
        ints = [[(x - cx, y - cy) for x, y in r.coords] for r in p.interiors]
        verts = list(ex)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(ex) - 2) + [Path.CLOSEPOLY]
        for ring in ints:
            verts.extend(ring)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY])
        return Path(verts, codes)

    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in polys:
        if not hasattr(p, "exterior") or len(p.exterior.coords) < 3:
            continue
        path = _poly_to_path(p)
        patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, alpha=alpha, hatch=hatch)
        ax.add_patch(patch)


def _draw_polygon_polar(ax, geom, centroid_xy, *, facecolor, edgecolor,
                         alpha, linewidth=0.5, hatch=None):
    """Render a (Multi)Polygon onto a polar axis. Each vertex (x, y) is
    converted to (theta, r) relative to `centroid_xy`. theta uses the
    set_theta_zero_location('N') + set_theta_direction(-1) convention
    already on the axis (0 = north, increasing east) → theta = atan2(dx, dy).
    """
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    cx, cy = centroid_xy

    def _poly_to_path(p):
        def _ring_to_polar(coords):
            pts = []
            for x, y in coords:
                dx = x - cx
                dy = y - cy
                pts.append((math.atan2(dx, dy), math.hypot(dx, dy)))
            return pts

        ex = _ring_to_polar(p.exterior.coords)
        verts = list(ex)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(ex) - 2) + [Path.CLOSEPOLY]
        for r in p.interiors:
            ring = _ring_to_polar(r.coords)
            verts.extend(ring)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY])
        return Path(verts, codes)

    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in polys:
        if not hasattr(p, "exterior") or len(p.exterior.coords) < 3:
            continue
        path = _poly_to_path(p)
        patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, alpha=alpha, hatch=hatch)
        ax.add_patch(patch)


def _render_dem_panel(ax, terrain_grid, building_footprint_general, centroid_xy):
    """Render the DEM contour map (bilinear-upsampled 5 m grid) on `ax`.
    The general building footprint is overlaid (recentred on the centroid).
    Returns True if anything was drawn.
    """
    import matplotlib.pyplot as plt
    if not (terrain_grid and terrain_grid.get('elev')):
        return False
    try:
        cell = float(terrain_grid['cell_size'])
        nx = int(terrain_grid['nx']); ny = int(terrain_grid['ny'])
        elev = np.array(terrain_grid['elev'], dtype=float)
        cx_off = (nx - 1) * cell / 2.0
        cy_off = (ny - 1) * cell / 2.0
        # gx/gy are the CENTROID coordinates of each DEM pixel.
        gx = np.arange(nx) * cell - cx_off
        gy = np.arange(ny) * cell - cy_off

        # Interpolate a smooth surface treating each pixel value as a sample
        # located at that pixel's centroid (gx[i], gy[j]); evaluate a cubic
        # spline on a fine grid spanning the centroid hull.
        grid_x_fine, grid_y_fine, elev_fine = gx, gy, elev
        try:
            from scipy.interpolate import RectBivariateSpline
            up = 5
            grid_x_fine = np.linspace(gx[0], gx[-1], nx * up)
            grid_y_fine = np.linspace(gy[0], gy[-1], ny * up)
            kx = min(3, nx - 1); ky = min(3, ny - 1)
            if kx >= 1 and ky >= 1:
                spline = RectBivariateSpline(gy, gx, elev, kx=ky, ky=kx)
                elev_fine = spline(grid_y_fine, grid_x_fine)
        except ImportError:
            pass

        # Smooth continuous surface (Gouraud shading on the upsampled grid)
        # matches the appearance of the 3D go.Surface plot.
        pm = ax.pcolormesh(grid_x_fine, grid_y_fine, elev_fine,
                           cmap='terrain', shading='gouraud', alpha=0.88)
        ax.contour(grid_x_fine, grid_y_fine, elev_fine, levels=10,
                   colors='saddlebrown', linewidths=0.4, alpha=0.6)
        plt.colorbar(pm, ax=ax, fraction=0.04, pad=0.02, label='Elev (m)')

        # General building footprint (recentred) — blue so it reads
        # clearly against the terrain colormap.
        _draw_polygon_cartesian(
            ax, building_footprint_general,
            centroid_xy[0], centroid_xy[1],
            facecolor='none', edgecolor='#1565C0', alpha=1.0, linewidth=2.0,
        )

        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])
        ax.set_aspect('equal')
        ax.set_xlabel('East offset (m)')
        ax.set_ylabel('North offset (m)')
        ax.set_title(
            f'DEM  ({cell:.0f} m grid, ±{nx * cell / 2:.0f} m, '
            f'centroid spline ×5)'
        )
        return True
    except Exception:
        return False


def _build_polar_shadow_page(shadow_data, building_reference, floor_height):
    """One figure: per-floor polar plots with blocked-horizon shading
    (driven by the distance-per-orientation legacy schema) + DEM panel.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    orientations = shadow_data['orientations']
    orient_keys = [str(int(o)) for o in orientations]
    contour = shadow_data.get('contour_distances') or [0.0] * len(orientations)
    sh = shadow_data.get('shadows', {})
    max_floors = int(shadow_data.get('max_neighbor_floors', 1) or 1)
    bldg_floors = int(shadow_data.get('building_n_floors', max_floors) or max_floors)
    footprint_by_floor = shadow_data.get('footprint_by_floor') or {}
    general_footprint = shadow_data.get('building_footprint')

    n_floors_plot = max(1, max_floors)
    theta_rad = np.radians(orientations)
    theta_closed = np.append(theta_rad, theta_rad[0])

    # Common outer radius for comparability.
    finite_radii = []
    for vals in sh.values():
        for v in vals[:n_floors_plot]:
            if isinstance(v, (int, float)) and np.isfinite(v):
                finite_radii.append(float(v))
    max_contour = max((float(c) for c in contour if np.isfinite(c)), default=10.0)
    outer_r = max(max_contour * 1.5, max(finite_radii, default=20.0) + max_contour)
    outer_r = float(min(150.0, max(outer_r, 30.0)))

    # Centroid used by polar-coord conversion for per-floor footprints.
    cx, cy = (general_footprint.centroid.x, general_footprint.centroid.y) \
        if general_footprint is not None else (0.0, 0.0)

    n_cols = 4
    n_rows_polars = math.ceil(n_floors_plot / n_cols)
    fig = plt.figure(figsize=(4.0 * n_cols, 3.6 * n_rows_polars + 1.0))
    gs = GridSpec(n_rows_polars, n_cols, hspace=0.5, wspace=0.45)

    for f in range(n_floors_plot):
        r_idx, c_idx = divmod(f, n_cols)
        ax = fig.add_subplot(gs[r_idx, c_idx], projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, outer_r)

        # Per-orientation blocker radius from centroid.
        blocker_r = []
        for o_idx, key in enumerate(orient_keys):
            vals = sh.get(key, [])
            if f < len(vals) and isinstance(vals[f], (int, float)) and np.isfinite(vals[f]):
                blocker_r.append(float(contour[o_idx]) + float(vals[f]))
            else:
                blocker_r.append(outer_r)
        blocker_arr = np.array(blocker_r + [blocker_r[0]], dtype=float)
        inner = np.minimum(blocker_arr, outer_r)
        ax.fill_between(theta_closed, inner, outer_r,
                        color='#202020', alpha=0.55, linewidth=0)

        # Per-floor building footprint (disappears if this floor doesn't exist).
        floor_geom = footprint_by_floor.get(f)
        is_neighbor_only = f >= bldg_floors
        if floor_geom is not None:
            if is_neighbor_only:
                _draw_polygon_polar(ax, floor_geom, (cx, cy),
                                    facecolor='none', edgecolor='black',
                                    alpha=0.9, linewidth=1.0, hatch='//')
            else:
                _draw_polygon_polar(ax, floor_geom, (cx, cy),
                                    facecolor='#bbbbbb', edgecolor='black',
                                    alpha=0.7, linewidth=1.0)

        suffix = ' (nbr-only)' if is_neighbor_only else ''
        ax.set_title(f'Floor {f}{suffix}', fontsize=10, pad=8)
        ax.tick_params(labelsize=7)
        rticks = [outer_r * q for q in (0.25, 0.5, 0.75, 1.0)]
        ax.set_rticks(rticks)
        ax.set_yticklabels([f'{int(t)}' for t in rticks], fontsize=7)
        ax.grid(True, alpha=0.3)

    for f in range(n_floors_plot, n_rows_polars * n_cols):
        r_idx, c_idx = divmod(f, n_cols)
        ax_blank = fig.add_subplot(gs[r_idx, c_idx])
        ax_blank.set_visible(False)

    fig.suptitle(
        f'Per-floor shadow horizon (polar / by orientation) — {building_reference}'
        f'  (building floors: {bldg_floors}, max floor seen: {max_floors})',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _build_multipoly_shadow_page(shadow_data, building_reference, floor_height):
    """One figure: per-floor cartesian plots of the shadow multipolygon
    (terrain shadow + nearby building shadow) + DEM panel.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle

    max_floors = int(shadow_data.get('max_neighbor_floors', 1) or 1)
    bldg_floors = int(shadow_data.get('building_n_floors', max_floors) or max_floors)
    shadows_geom = shadow_data.get('shadows_geom') or {}
    footprint_by_floor = shadow_data.get('footprint_by_floor') or {}
    general_footprint = shadow_data.get('building_footprint')

    floor_keys_sorted = sorted(shadows_geom.keys()) if shadows_geom else list(range(max_floors))
    n_floors_plot = max(1, len(floor_keys_sorted))

    cx, cy = (general_footprint.centroid.x, general_footprint.centroid.y) \
        if general_footprint is not None else (0.0, 0.0)
    extent_r = 150.0

    n_cols = 4
    n_rows_polars = math.ceil(n_floors_plot / n_cols)
    fig = plt.figure(figsize=(4.0 * n_cols, 3.6 * n_rows_polars + 1.0))
    gs = GridSpec(n_rows_polars, n_cols,
                  height_ratios=[1.0] * n_rows_polars,
                  hspace=0.4, wspace=0.35)

    for i, f in enumerate(floor_keys_sorted):
        r_idx, c_idx = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.set_xlim(-extent_r, extent_r)
        ax.set_ylim(-extent_r, extent_r)
        ax.set_aspect('equal')
        ax.add_patch(Circle((0, 0), extent_r, fill=False,
                            edgecolor='#cccccc', linewidth=0.8, linestyle=':'))

        _draw_polygon_cartesian(ax, shadows_geom.get(f), cx, cy,
                                facecolor='#1a1a1a', edgecolor='#000000',
                                alpha=0.55, linewidth=0.3)

        floor_geom = footprint_by_floor.get(f)
        is_neighbor_only = f >= bldg_floors
        if floor_geom is not None:
            if is_neighbor_only:
                _draw_polygon_cartesian(ax, floor_geom, cx, cy,
                                        facecolor='none', edgecolor='black',
                                        alpha=0.9, linewidth=1.0, hatch='//')
            else:
                _draw_polygon_cartesian(ax, floor_geom, cx, cy,
                                        facecolor='#bbbbbb', edgecolor='black',
                                        alpha=0.8, linewidth=1.0)

        suffix = ' (nbr-only)' if is_neighbor_only else ''
        ax.set_title(f'Floor {f}{suffix}', fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle=':')

    for i in range(n_floors_plot, n_rows_polars * n_cols):
        r_idx, c_idx = divmod(i, n_cols)
        ax_blank = fig.add_subplot(gs[r_idx, c_idx])
        ax_blank.set_visible(False)

    fig.suptitle(
        f'Per-floor blocked-region multipolygon — {building_reference}'
        f'  (building floors: {bldg_floors}, max floor seen: {max_floors})',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _create_static_shadow_plot(building_row, building_reference, output_path,
                               include_elevation, floor_height):
    """Static per-floor shadow + DEM visualization using matplotlib.

    Emits TWO pages:
        Page 1 — polar plots driven by `br__shadows_at_distance` /
                 `br__building_contour_at_distance` (legacy per-orientation
                 schema), with the blocked horizon shaded.
        Page 2 — cartesian plots driven by `br__shadows` (per-floor
                 blocked-region multipolygon).

    Both pages include a DEM contour panel at the bottom and overlay the
    per-floor building footprint (which disappears at floors above the
    building's own roof).

    Output handling:
        - `.pdf` → both pages in one PDF.
        - `.png`/`.jpg`/`.svg`/etc. → two files written next to the
          requested path with `_polar` and `_multipoly` suffixes.
    """
    try:
        import os
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib not available")
        return None

    shadow_data = _extract_shadow_data(building_row)
    if not shadow_data:
        print(f"No shadow data found for building {building_reference}")
        return None

    fig_polar = _build_polar_shadow_page(shadow_data, building_reference, floor_height)
    fig_multi = _build_multipoly_shadow_page(shadow_data, building_reference, floor_height)

    if output_path:
        if output_path.lower().endswith('.pdf'):
            with PdfPages(output_path) as pdf:
                pdf.savefig(fig_polar, bbox_inches='tight')
                pdf.savefig(fig_multi, bbox_inches='tight')
            print(f"Shadow analysis saved to {output_path} (2 pages)")
        else:
            base, ext = os.path.splitext(output_path)
            path_polar = f"{base}_polar{ext}"
            path_multi = f"{base}_multipoly{ext}"
            fig_polar.savefig(path_polar, bbox_inches='tight', dpi=300)
            fig_multi.savefig(path_multi, bbox_inches='tight', dpi=300)
            print(f"Shadow analysis saved to {path_polar} and {path_multi}")
        plt.close(fig_polar); plt.close(fig_multi)
    else:
        plt.show()
    return [fig_polar, fig_multi]


def _detect_floor_height_from_data(building_row):
    """Best-effort floor height (m). The previous heuristic read per-floor
    vertical-distance spacing from `br__vertical_distances_at_collision`,
    which no longer exists in the schema. Fall back to a sensible default;
    callers can override via the `floor_height` argument of the plot.
    """
    return 3.0


def _extract_shadow_data(building_row):
    """Extract shadow / terrain data from a building_row in the current schema.
    Returns None if no usable shadow data is present.

    Output dict keys (all optional except `orientations`):

        orientations              [float]                — sorted 0..359 in degrees
        contour_distances         [float]                — centroid → building edge per orientation
        shadows                   dict orientation→list  — combined per-floor blocker distance
                                                           (building roof OR terrain rise above eye level)
        max_neighbor_floors       int                    — longest shadows list across orientations
        building_n_floors         int                    — analyzed building's own floor count
        terrain_grid              dict — 5 m DEM grid for contour rendering
                                       {x_min, y_min, cell_size, nx, ny, elev[ny][nx]}
        elevation                 float                  — building's ground elevation
    """
    shadow_data = {}

    def _to_dict(col):
        v = building_row[col] if col in building_row.index else None
        return v if isinstance(v, dict) else None

    def _sorted_orientations(d):
        return sorted(float(k) for k in d.keys() if str(k).replace('.', '').isdigit())

    shadows = _to_dict('br__shadows_at_distance')
    contours = _to_dict('br__building_contour_at_distance')

    orientations = None
    for d in (shadows, contours):
        if d:
            orientations = _sorted_orientations(d)
            if orientations:
                break
    if not orientations:
        return None
    shadow_data['orientations'] = orientations

    if contours:
        shadow_data['contour_distances'] = [
            float(contours.get(str(int(o)), 0.0) or 0.0) for o in orientations
        ]

    if shadows:
        max_floors = 0
        normalized = {}
        for o in orientations:
            key = str(int(o))
            vals = shadows.get(key)
            if isinstance(vals, list):
                normalized[key] = list(vals)
                max_floors = max(max_floors, len(vals))
            elif isinstance(vals, (int, float)):
                normalized[key] = [float(vals)]
                max_floors = max(max_floors, 1)
            else:
                normalized[key] = []
        shadow_data['shadows'] = normalized
        shadow_data['max_neighbor_floors'] = max_floors

    if 'br__terrain_grid' in building_row.index:
        grid = building_row['br__terrain_grid']
        if isinstance(grid, dict) and grid.get("elev"):
            shadow_data['terrain_grid'] = grid

    # New: per-floor blocked-region multipolygon.
    # {floor_int: Polygon/MultiPolygon}
    if 'br__shadows' in building_row.index:
        sh_geom = building_row['br__shadows']
        if isinstance(sh_geom, dict) and sh_geom:
            shadow_data['shadows_geom'] = sh_geom

    if 'br__building_footprint_geometry' in building_row.index:
        bfg = building_row['br__building_footprint_geometry']
        if isinstance(bfg, (bytes, bytearray, memoryview)):
            try:
                from shapely import wkb as _shapely_wkb_utils
                bfg = _shapely_wkb_utils.loads(bytes(bfg))
            except Exception:
                bfg = None
        if bfg is not None and not (hasattr(bfg, 'is_empty') and bfg.is_empty):
            shadow_data['building_footprint'] = bfg

    # Per-floor footprint: list of [sbr_ref, floor_index, geometry] triplets,
    # one row per (sbr, floor). Reduce to {floor: unioned_geometry} across
    # SBRs of this building so the plot can show the actual outline at each
    # floor (which shrinks as the building steps back at upper floors).
    if 'br__building_footprint_by_floor' in building_row.index:
        bfbf = building_row['br__building_footprint_by_floor']
        if isinstance(bfbf, list) and bfbf:
            per_floor = {}
            for triplet in bfbf:
                if not (isinstance(triplet, (list, tuple)) and len(triplet) >= 3):
                    continue
                _, fnum, geom = triplet[0], triplet[1], triplet[2]
                if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
                    continue
                try:
                    fi = int(fnum)
                except (TypeError, ValueError):
                    continue
                per_floor.setdefault(fi, []).append(geom)
            if per_floor:
                from shapely.ops import unary_union as _uu
                shadow_data['footprint_by_floor'] = {
                    fi: (_uu(geoms) if len(geoms) > 1 else geoms[0])
                    for fi, geoms in per_floor.items()
                }

    if 'elevation' in building_row.index:
        try:
            shadow_data['elevation'] = float(building_row['elevation'])
        except (TypeError, ValueError):
            pass

    for col in ('br__floors_above_ground', 'br__number_of_floors', 'numberOfFloors'):
        if col in building_row.index:
            try:
                shadow_data['building_n_floors'] = int(building_row[col])
                break
            except (TypeError, ValueError):
                pass

    return shadow_data if shadow_data else None


# ==========================================
# NEIGHBORING MUNICIPALITIES
# ==========================================
def get_neighboring_cadaster_codes(
    cadaster_codes,
    census_tracts_dir,
    buffer_distance=1000,
    include_original=False,
    year=2022
):
    """
    Get cadaster codes of neighboring municipalities based on census tract adjacency.

    Parameters
    ----------
    cadaster_codes : str or list
        Cadaster code(s)
    census_tracts_dir : str
        Directory containing census_tracts.geojson
    buffer_distance : int, default 1000
        Buffer distance in meters
    include_original : bool, default False
        Whether to include original municipalities in the output

    Returns
    -------
    list
        List of cadaster codes (neighbors or neighbors + original)
    """

    # ------------------------------------------------------------
    # 1️⃣ Normalize input
    # ------------------------------------------------------------
    if not cadaster_codes:
        return []

    if isinstance(cadaster_codes, str):
        cadaster_codes = [cadaster_codes]

    # ------------------------------------------------------------
    # 2️⃣ Load census tracts
    # ------------------------------------------------------------
    census_file = f"{census_tracts_dir}//validated_census_{year}.gpkg"

    if not os.path.exists(census_file):
        print("[WARN] Census file not found")
        return cadaster_codes if include_original else []

    census_gdf = gpd.read_file(census_file)

    if census_gdf.empty:
        print("[WARN] Census GeoDataFrame is empty")
        return cadaster_codes if include_original else []

    # ------------------------------------------------------------
    # 3️⃣ Validate schema
    # ------------------------------------------------------------
    if "CUMUN" not in census_gdf.columns:
        raise ValueError("CUMUN column not found in census data")

    census_gdf["CUMUN"] = census_gdf["CUMUN"].astype(str)

    # ------------------------------------------------------------
    # 4️⃣ CRS fix (buffer requires projected CRS)
    # ------------------------------------------------------------
    if census_gdf.crs is None:
        raise ValueError("Census CRS is undefined")

    if census_gdf.crs.is_geographic:
        census_gdf = census_gdf.to_crs(3857)

    # ------------------------------------------------------------
    # 5️⃣ Convert cadaster → INE
    # ------------------------------------------------------------
    original_ine_codes = cadaster_to_ine_codes(cadaster_codes)

    if not original_ine_codes:
        print("[WARN] No INE codes derived")
        return cadaster_codes if include_original else []

    original_ine_codes = [str(code) for code in original_ine_codes]

    # ------------------------------------------------------------
    # 6️⃣ Filter original municipalities
    # ------------------------------------------------------------
    original_tracts = census_gdf[
        census_gdf["CUMUN"].isin(original_ine_codes)
    ]

    if original_tracts.empty:
        print("[WARN] No matching census tracts")
        return cadaster_codes if include_original else []

    # ------------------------------------------------------------
    # 7️⃣ Buffer
    # ------------------------------------------------------------
    original_union = original_tracts.unary_union
    buffered_area = original_union.buffer(buffer_distance)

    # ------------------------------------------------------------
    # 8️⃣ Spatial query
    # ------------------------------------------------------------
    neighboring_tracts = census_gdf[
        census_gdf.geometry.intersects(buffered_area)
    ]

    neighbor_ine_codes = neighboring_tracts["CUMUN"].unique()

    # ------------------------------------------------------------
    # 9️⃣ Remove originals (KEY STEP)
    # ------------------------------------------------------------
    neighbor_only_ine = set(neighbor_ine_codes) - set(original_ine_codes)

    if include_original:
        final_ine = set(original_ine_codes) | neighbor_only_ine
    else:
        final_ine = neighbor_only_ine

    if not final_ine:
        return cadaster_codes if include_original else []

    # ------------------------------------------------------------
    # 🔟 Convert back to cadaster
    # ------------------------------------------------------------
    all_cadaster_codes = ine_to_cadaster_codes(list(final_ine))

    if not all_cadaster_codes:
        print("[WARN] ine_to_cadaster_codes returned empty")

    return all_cadaster_codes

# ==========================================
# DATA ANALYSIS & PROCESSING
# ==========================================

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

def _read_inferred_indicators_from_db(cadaster_code, wd, table):
    """Read a per-municipality slice of an inferred-indicators table from
    the central DuckDB. Returns the same pandas frame shape that the
    pickled file used to expose. Drops the central-DB-only bookkeeping
    columns so the result is what downstream consumers expect."""
    from hypercadaster_ES import _db
    result_dir = results_dir_(wd)
    db_path = _db.db_path(result_dir)
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Central DuckDB not found at {db_path}; run hc.merge(...) "
            f"with building_parts_inference=True first."
        )
    con = _db.open_shard_read(db_path)
    try:
        return _db.read_df_safe(
            con, table,
            where="cadaster_code = ?", params=[cadaster_code],
        ).drop(columns=["cadaster_code", "zone_reference"], errors="ignore")
    finally:
        con.close()


def read_br_inferred_indicators(cadaster_code, wd):
    return _read_inferred_indicators_from_db(cadaster_code, wd, "br_results")


def read_sbr_inferred_indicators(cadaster_code, wd):
    return _read_inferred_indicators_from_db(cadaster_code, wd, "sbr_results")


def get_heavy_columns_for_building(wd, cadaster_code, building_reference,
                                   *, table="br_results", columns=None):
    """Fetch the heavy BLOB columns that `hc.merge` excludes by default.

    `hc.merge` keeps memory bounded on large municipalities by NOT
    materialising a handful of huge per-building dict columns
    (shadows, terrain grids — see `hypercadaster_ES._db.HEAVY_BLOB_COLS`).
    Each of those columns is ~11 GB on disk for Barcelona and explodes
    into many tens of GB of Python dicts when unpickled into a 70 k-row
    frame. Use this helper to retrieve them one building at a time.

    Parameters
    ----------
    wd : str
        Working directory (same one passed to `hc.merge`).
    cadaster_code : str
        Municipality cadaster code (e.g. '08900').
    building_reference : str
        Building reference id (e.g. '8341105DF2884A').
    table : str, default 'br_results'
        Source table — 'br_results' (per-building) or 'sbr_results'
        (per single-building-reference).
    columns : iterable[str], optional
        Specific column names to fetch. If None, fetches every
        HEAVY_BLOB_COL belonging to the chosen `table`.

    Returns
    -------
    dict[str, object]
        {column_name: unpickled_value}. Missing rows yield None for
        every requested column.

    Example
    -------
    >>> import hypercadaster_ES as hc
    >>> wd = '/path/to/working_dir'
    >>> heavy = hc.utils.get_heavy_columns_for_building(
    ...     wd, '08900', '8341105DF2884A'
    ... )
    >>> set(heavy)
    {'br__shadows', 'br__shadows_at_distance', 'br__terrain_grid'}
    >>> heavy['br__shadows']         # dict keyed by floor number
    {0: <shapely Polygon ...>, 1: <shapely Polygon ...>, ...}

    To plot shadows for one building without manually merging, pass
    `wd=` to `plot_building_shadows_analysis` — it will auto-hydrate
    the missing columns for the requested building.
    """
    from hypercadaster_ES import _db
    result_dir = results_dir_(wd)
    return _db.get_blob_for_building(
        result_dir, cadaster_code, building_reference,
        table=table, columns=columns,
    )


def read_addresses_indicators(cadaster_code, wd):
    """Read basic building indicators for a municipality.

    When ``hc.merge(...)`` was run with ``building_parts_inference=False``,
    the central DuckDB still carries the 12 basic columns (gml_id,
    building_status, building_reference, ..., zone_type).  Query that
    table directly instead of a separate pickle artifact.

    Falls back to the legacy pickle path if the DuckDB table is missing
    (e.g. old data or external producer).
    """
    from hypercadaster_ES import _db
    result_dir = results_dir_(wd)
    db_path = _db.db_path(result_dir)
    if os.path.exists(db_path):
        try:
            con = _db.open_shard_read(db_path)
            try:
                df = _db.read_df_safe(
                    con, "buildings",
                    where="cadaster_code = ?", params=[cadaster_code],
                )
                if not df.empty and set(_db.BASIC_BUILDING_COLUMNS).issubset(df.columns):
                    return df[_db.BASIC_BUILDING_COLUMNS]
            finally:
                con.close()
        except Exception:
            pass  # fall through to legacy path

    # Legacy pickle fallback — `_no_inference.pkl` written by user scripts
    return pd.read_pickle(f"{result_dir}/{cadaster_code}_no_inference.pkl", compression="gzip")

# ==========================================
# VISUALIZATION UTILITIES
# ==========================================

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

def plot_building_reference(
    gdf,
    building_reference="",
    address_col="address_location",
    building_centroid_col="building_centroid",
    parcel_centroid_col="parcel_centroid",
    parcel_geom_col="parcel_geometry",
    building_geom_col="building_geometry",
    buffer_m=100,
    output_path=None,
    dpi=200
):
    """
    Plot parcel, building footprint, centroids and addresses over OSM
    and save to file.

    Parameters
    ----------
    gdf : GeoDataFrame
    idx : int
    output_path : str or None
        If None → auto-generate filename
    dpi : int
        Image resolution
    """

    row = gdf[gdf['building_reference'] == building_reference].iloc[0]

    # --------------------------------------------------
    # 1️⃣ Collect geometries
    # --------------------------------------------------
    geoms = []

    if parcel_geom_col in row and row[parcel_geom_col] is not None:
        geoms.append(("parcel", row[parcel_geom_col]))

    if building_geom_col in row and row[building_geom_col] is not None:
        geoms.append(("building", row[building_geom_col]))

    if building_centroid_col in row and row[building_centroid_col] is not None:
        geoms.append(("building_centroid", row[building_centroid_col]))

    if parcel_centroid_col in row and row[parcel_centroid_col] is not None:
        geoms.append(("parcel_centroid", row[parcel_centroid_col]))

    if address_col in row and row[address_col] is not None:
        addr = row[address_col]

        if isinstance(addr, list):
            for p in addr:
                if isinstance(p, Point):
                    geoms.append(("address", p))
        elif isinstance(addr, Point):
            geoms.append(("address", addr))

    # --------------------------------------------------
    # 2️⃣ Build GeoDataFrame
    # --------------------------------------------------
    plot_gdf = gpd.GeoDataFrame(
        [{"type": t, "geometry": g} for t, g in geoms],
        geometry="geometry",
        crs=gdf.crs
    )

    # --------------------------------------------------
    # 3️⃣ Reproject to Web Mercator
    # --------------------------------------------------
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    # --------------------------------------------------
    # 4️⃣ Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_gdf[plot_gdf["type"] == "parcel"].plot(ax=ax, alpha=0.3)
    plot_gdf[plot_gdf["type"] == "building"].plot(ax=ax, alpha=0.6)

    plot_gdf[plot_gdf["type"] == "building_centroid"].plot(ax=ax, markersize=80)
    plot_gdf[plot_gdf["type"] == "parcel_centroid"].plot(ax=ax, markersize=80)
    plot_gdf[plot_gdf["type"] == "address"].plot(ax=ax, markersize=40)

    # --------------------------------------------------
    # 5️⃣ Extent
    # --------------------------------------------------
    bounds = plot_gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds

    ax.set_xlim(x_min - buffer_m, x_max + buffer_m)
    ax.set_ylim(y_min - buffer_m, y_max + buffer_m)

    # --------------------------------------------------
    # 6️⃣ Basemap
    # --------------------------------------------------
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=19
    )

    ax.set_axis_off()

    title = f"Building: {row["building_reference"]}" if "building_reference" in row else "Building"
    plt.title(title)

    # --------------------------------------------------
    # 7️⃣ Output path
    # --------------------------------------------------
    if output_path is None:
        building_ref = row.get("building_reference", f"idx_{idx}")
        output_path = f"building_{building_ref}.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    # --------------------------------------------------
    # 8️⃣ Robust extent (FIXED)
    # --------------------------------------------------

    # Priority: parcel → building → centroid
    main_geom = None

    if parcel_geom_col in row and row[parcel_geom_col] is not None:
        main_geom = row[parcel_geom_col]
    elif building_geom_col in row and row[building_geom_col] is not None:
        main_geom = row[building_geom_col]
    elif building_centroid_col in row and row[building_centroid_col] is not None:
        main_geom = row[building_centroid_col]

    # Fallback
    if main_geom is None:
        main_geom = plot_gdf.unary_union

    # Convert to GeoSeries to project
    main_geom = gpd.GeoSeries([main_geom], crs=gdf.crs).to_crs(epsg=3857).iloc[0]

    # Buffer in meters (correct in EPSG:3857)
    buffered = main_geom.buffer(buffer_m)

    x_min, y_min, x_max, y_max = buffered.bounds

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------
    # 9⃣ Save
    # --------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


# ==========================================
# DEM / GEOMETRY VISUALISATIONS
# ==========================================

_WALL_TYPE_COLOURS = {
    'main_facade':    '#E86B2E',
    'party_wall':     '#4C72B0',
    'adiabatic':      '#9B59B6',
    'patios':         '#55A868',
    'terrain_contact':'#8B5A2B',
}
_WALL_TYPE_DEFAULT = '#AAAAAA'

# Horizontal structural slab/roof surfaces (br__structural_floors). Kept
# visually distinct from the (vertical) wall palette above.
_STRUCTURAL_FLOOR_COLOURS = {
    'adiabatic': '#884EA0',   # slab between two floors (purple)
    'roof':      '#C0392B',   # top exposed to air (red)
    'floor':     '#16A085',   # underside exposed to air / overhang (teal)
    'terrain':   '#7E5109',   # on/under ground (dark brown)
}
_STRUCTURAL_FLOOR_DEFAULT = '#AAAAAA'


def _triangulate_polygon_xy(poly):
    """Triangulate a 2-D (multi)polygon for a flat Mesh3d surface.

    Returns (verts, tris) where `verts` is a list of (x, y) tuples and
    `tris` a list of (i, j, k) index triples into `verts`. Concavities and
    holes (patios) are respected by discarding Delaunay triangles whose
    centroid falls outside the polygon.
    """
    from shapely.ops import triangulate as _shp_triangulate
    verts, tris = [], []
    vidx = {}

    def _vi(pt):
        key = (round(pt[0], 4), round(pt[1], 4))
        idx = vidx.get(key)
        if idx is None:
            idx = len(verts)
            vidx[key] = idx
            verts.append(key)
        return idx

    polys = list(poly.geoms) if hasattr(poly, 'geoms') else [poly]
    for p in polys:
        if p is None or p.is_empty or not hasattr(p, 'exterior'):
            continue
        try:
            for tri in _shp_triangulate(p):
                if not p.contains(tri.centroid):
                    continue
                cc = list(tri.exterior.coords)[:3]
                tris.append((_vi(cc[0]), _vi(cc[1]), _vi(cc[2])))
        except Exception:
            continue
    return verts, tris


def plot_building_dem(gdf, building_reference, output_path=None, wd=None, dpi=150):
    """Standalone DEM contour map for a single building.

    Uses the ``br__terrain_grid`` heavy column (auto-fetched from DuckDB
    when *wd* is supplied).  The building's general footprint is overlaid
    in blue.

    Parameters
    ----------
    gdf : DataFrame / GeoDataFrame
        Output of ``hc.merge``.
    building_reference : str
        Building reference id.
    output_path : str, optional
        Save path (.png / .pdf / …).  Displays interactively when None.
    wd : str, optional
        Working directory for the central DuckDB — required to fetch the
        terrain grid when it is absent from *gdf*.
    dpi : int, default 150
    """
    import matplotlib.pyplot as plt

    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None
    row = building_data.iloc[0].copy()

    if wd is not None and (
        'br__terrain_grid' not in row.index or row.get('br__terrain_grid') is None
    ):
        from hypercadaster_ES import _db
        cadaster_code = row.get('cadaster_code')
        if cadaster_code is not None:
            try:
                fetched = get_heavy_columns_for_building(
                    wd, str(cadaster_code), building_reference,
                    table='br_results', columns=['br__terrain_grid'],
                )
                for col, val in fetched.items():
                    row[col] = val
            except Exception as _e:
                print(f"Warning: could not fetch terrain grid: {_e}")

    terrain_grid = row.get('br__terrain_grid')
    if not (isinstance(terrain_grid, dict) and terrain_grid.get('elev')):
        print(f"No terrain grid available for building {building_reference}")
        return None

    bfg = row.get('br__building_footprint_geometry')
    if isinstance(bfg, (bytes, bytearray, memoryview)):
        try:
            from shapely import wkb as _swkb
            bfg = _swkb.loads(bytes(bfg))
        except Exception:
            bfg = None
    cx = bfg.centroid.x if bfg is not None else 0.0
    cy = bfg.centroid.y if bfg is not None else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    drawn = _render_dem_panel(ax, terrain_grid, bfg, (cx, cy))
    if not drawn:
        print(f"Could not render DEM for building {building_reference}")
        plt.close(fig)
        return None

    fig.suptitle(f'Terrain DEM — {building_reference}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"DEM plot saved to {output_path}")
    else:
        plt.show()
    return fig


def plot_building_geometry(gdf, building_reference, output_path=None, wd=None, dpi=150):
    """2-D floor-plan grid showing wall segments coloured by wall type.

    Wall type colour legend:
        main_facade     #E86B2E  (orange)
        party_wall      #4C72B0  (blue)
        patios          #55A868  (green)
        terrain_contact #8B5A2B  (brown)
        unknown/None    #AAAAAA  (grey)

    Parameters
    ----------
    gdf : DataFrame / GeoDataFrame
    building_reference : str
    output_path : str, optional
    wd : str, optional
        Working directory — used to auto-fetch ``br__wall_segments``.
    dpi : int, default 150
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None
    row = building_data.iloc[0].copy()

    if wd is not None and (
        'br__wall_segments' not in row.index or row.get('br__wall_segments') is None
    ):
        from hypercadaster_ES import _db
        cadaster_code = row.get('cadaster_code')
        if cadaster_code is not None:
            try:
                fetched = get_heavy_columns_for_building(
                    wd, str(cadaster_code), building_reference,
                    table='br_results', columns=['br__wall_segments'],
                )
                for col, val in fetched.items():
                    row[col] = val
            except Exception as _e:
                print(f"Warning: could not fetch wall segments: {_e}")

    wall_segments = row.get('br__wall_segments')
    if not isinstance(wall_segments, list) or not wall_segments:
        print(f"No wall segment data available for building {building_reference}")
        return None

    # Decode any remaining bytearray / bytes values in the row so that
    # subsequent code never has to deal with raw binary objects.
    def _robust_decode(v):
        """Try WKB (geometry) then pickle on any bytes-like value."""
        if not isinstance(v, (bytes, bytearray, memoryview)):
            return v
        b = bytes(v)
        # shapely WKB (plain or EWKB with SRID)
        try:
            from shapely import wkb as _swkb
            return _swkb.loads(b, include_srid=True)
        except Exception:
            pass
        # pickle-encoded Python object
        try:
            import pickle as _pkl
            return _pkl.loads(b)
        except Exception:
            pass
        return v   # return as-is if both fail

    for _col in list(row.index):
        _v = row[_col]
        if isinstance(_v, (bytes, bytearray, memoryview)):
            row[_col] = _robust_decode(_v)

    bfg = row.get('br__building_footprint_geometry')
    if not hasattr(bfg, 'exterior') and not hasattr(bfg, 'geoms'):
        bfg = row.get('building_geometry')

    _footprint_rings = []   # list of (xs, ys) tuples — one per ring to draw
    if bfg is not None:
        geoms = list(bfg.geoms) if hasattr(bfg, 'geoms') else [bfg]
        for g in geoms:
            if hasattr(g, 'exterior'):
                coords = list(g.exterior.coords)
                _footprint_rings.append(([p[0] for p in coords],
                                         [p[1] for p in coords]))

    # Group segments by floor
    floors_map = {}
    for seg in wall_segments:
        if not isinstance(seg, dict):
            continue
        f = seg.get('floor', 0)
        try:
            f = int(f)
        except (TypeError, ValueError):
            f = 0
        floors_map.setdefault(f, []).append(seg)

    # Pre-compute shared axis extent from ALL segments + footprint so every
    # floor subplot uses the same scale.
    all_xs, all_ys = [], []
    for seg in wall_segments:
        geom = seg.get('geometry') if isinstance(seg, dict) else None
        if geom is None:
            continue
        try:
            xs, ys = geom.xy
            all_xs.extend(xs); all_ys.extend(ys)
        except Exception:
            pass
    for ring_xs, ring_ys in _footprint_rings:
        all_xs.extend(ring_xs); all_ys.extend(ring_ys)

    if all_xs:
        cx = (min(all_xs) + max(all_xs)) / 2
        cy = (min(all_ys) + max(all_ys)) / 2
        half = max(max(all_xs) - min(all_xs),
                   max(all_ys) - min(all_ys)) / 2 * 1.15 + 1.0
        x_lim = (cx - half, cx + half)
        y_lim = (cy - half, cy + half)
        # Offset all coords to centroid-relative for readable axis labels
        ox, oy = cx, cy
    else:
        x_lim = y_lim = (-10, 10)
        ox, oy = 0.0, 0.0

    floor_keys = sorted(floors_map)
    n_floors = max(1, len(floor_keys))
    n_cols = min(4, n_floors)
    n_rows = math.ceil(n_floors / n_cols)

    fig = plt.figure(figsize=(4.5 * n_cols, 4.0 * n_rows + 1.0))
    gs = GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    for idx, floor in enumerate(floor_keys):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_aspect('equal')
        ax.set_xlim(x_lim[0] - ox, x_lim[1] - ox)
        ax.set_ylim(y_lim[0] - oy, y_lim[1] - oy)

        for ring_xs, ring_ys in _footprint_rings:
            ax.plot([x - ox for x in ring_xs],
                    [y - oy for y in ring_ys],
                    color='#BBBBBB', linewidth=0.8, linestyle=':', zorder=1)

        for seg in floors_map[floor]:
            geom = seg.get('geometry')
            wtype = seg.get('type')
            colour = _WALL_TYPE_COLOURS.get(wtype, _WALL_TYPE_DEFAULT)
            if geom is None:
                continue
            try:
                xs, ys = geom.xy
                ax.plot([x - ox for x in xs],
                        [y - oy for y in ys],
                        color=colour, linewidth=1.5,
                        solid_capstyle='round', zorder=2)
            except Exception:
                pass

        ax.set_title(f'Floor {floor}', fontsize=10, pad=5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_xlabel('ΔX (m)', fontsize=7)
        ax.set_ylabel('ΔY (m)', fontsize=7)

    for idx in range(n_floors, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    legend_patches = [
        mpatches.Patch(color=v, label=k.replace('_', ' ').title())
        for k, v in _WALL_TYPE_COLOURS.items()
    ] + [mpatches.Patch(color=_WALL_TYPE_DEFAULT, label='Unknown')]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=len(legend_patches), fontsize=9,
               bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.suptitle(f'Wall geometry by floor — {building_reference}', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"2-D geometry plot saved to {output_path}")
    else:
        plt.show()
    return fig


def plot_building_3d_geometry(gdf, building_reference, output_path=None, wd=None,
                               floor_height=3.0):
    """Interactive 3-D view of a building's geometry with terrain and neighbour shadows.

    Renders three layers in a single plotly scene:
      1. **Terrain**  — ``br__terrain_grid`` as a coloured heightmap surface
         (recentred on the building centroid).
      2. **Building shades** — ``br__shadows`` per-floor blocked-region polygons
         from neighbouring buildings, shown as semi-transparent horizontal planes
         at the corresponding floor height  (floor × floor_height).
      3. **Wall segments** — ``br__wall_segments`` extruded into vertical rectangles
         spanning *floor* × floor_height to (floor+1) × floor_height, coloured by
         wall type.

    All three axes use the same metre scale (1:1:1 aspect ratio).

    Output
    ------
    Always saves as an interactive HTML file.  If *output_path* has a
    non-``.html`` extension it is replaced automatically.
    Pass ``output_path=None`` to open directly in the default browser.

    Parameters
    ----------
    gdf : DataFrame / GeoDataFrame
    building_reference : str
    output_path : str, optional
    wd : str, optional
        Working directory — used to auto-fetch heavy BLOB columns
        (``br__wall_segments``, ``br__shadows``, ``br__terrain_grid``).
    floor_height : float, default 3.0
        Height of each storey in metres.
    """
    import plotly.graph_objects as go

    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None
    row = building_data.iloc[0].copy()

    # ---- Auto-fetch heavy columns from DuckDB ---------------------------------
    heavy_cols = ['br__wall_segments', 'br__shadows', 'br__terrain_grid',
                  'br__building_footprint_by_floor']
    missing = [c for c in heavy_cols
               if c not in row.index or row.get(c) is None]
    if wd is not None and missing:
        cadaster_code = row.get('cadaster_code')
        if cadaster_code is not None:
            try:
                fetched = get_heavy_columns_for_building(
                    wd, str(cadaster_code), building_reference,
                    table='br_results', columns=missing,
                )
                for col, val in fetched.items():
                    row[col] = val
            except Exception as _e:
                print(f"Warning: could not fetch heavy columns: {_e}")

    wall_segments = row.get('br__wall_segments')
    if not isinstance(wall_segments, list) or not wall_segments:
        print(f"No wall segment data available for building {building_reference}")
        return None

    # ---- Ground elevation (absolute metres) — offsets all building z-coords ---
    ground_z = None
    for _ecol in ('elevation', 'br__elevation', 'br__mean_elevation'):
        try:
            _ev = row.get(_ecol)
            if _ev is not None:
                ground_z = float(_ev)
                break
        except (TypeError, ValueError):
            pass
    # Fallback: centre cell of the terrain grid (computed below if needed)

    # ---- Determine building centroid (for terrain grid alignment) -------------
    centroid_x, centroid_y = None, None
    bfg = row.get('br__building_footprint_geometry')
    if bfg is not None and isinstance(bfg, (bytes, bytearray, memoryview)):
        try:
            from shapely import wkb as _swkb
            bfg = _swkb.loads(bytes(bfg))
        except Exception:
            bfg = None
    if bfg is not None and not (hasattr(bfg, 'is_empty') and bfg.is_empty):
        centroid_x, centroid_y = bfg.centroid.x, bfg.centroid.y
    # Fallback: centre of wall-segment bounding box
    if centroid_x is None:
        seg_xs, seg_ys = [], []
        for seg in wall_segments:
            geom = seg.get('geometry')
            if geom is None:
                continue
            try:
                for x, y, *_ in geom.coords:
                    seg_xs.append(x); seg_ys.append(y)
            except Exception:
                pass
        if seg_xs:
            centroid_x = (min(seg_xs) + max(seg_xs)) / 2
            centroid_y = (min(seg_ys) + max(seg_ys)) / 2

    import numpy as np

    # ---- Coordinate offsets: all scene coords are centroid-relative (X, Y)
    # and ground-relative (Z) so the building fills the view regardless of
    # absolute UTM values or absolute elevation.
    ox = centroid_x if centroid_x is not None else 0.0
    oy = centroid_y if centroid_y is not None else 0.0

    traces = []
    all_x, all_y, all_z = [], [], []   # relative coords for range calc

    # ---- Layer 1: Terrain DEM surface (bicubic-upsampled, no pixels) ---------
    terrain_grid = row.get('br__terrain_grid')
    if isinstance(terrain_grid, dict) and terrain_grid.get('elev'):
        try:
            cell = float(terrain_grid['cell_size'])
            tnx = int(terrain_grid['nx'])
            tny = int(terrain_grid['ny'])
            elev = np.array(terrain_grid['elev'], dtype=float)

            cx_off = (tnx - 1) * cell / 2.0
            cy_off = (tny - 1) * cell / 2.0
            # lx/ly are each pixel's CENTROID coordinate (centroid-relative).
            lx = np.arange(tnx) * cell - cx_off
            ly = np.arange(tny) * cell - cy_off

            # Smooth surface by interpolating each pixel value as a sample at
            # its centroid (lx[i], ly[j]) with a cubic spline on a fine grid.
            try:
                from scipy.interpolate import RectBivariateSpline
                up = 4
                lx_f = np.linspace(lx[0], lx[-1], tnx * up)
                ly_f = np.linspace(ly[0], ly[-1], tny * up)
                kx = min(3, tnx - 1); ky = min(3, tny - 1)
                if kx >= 1 and ky >= 1:
                    _sp = RectBivariateSpline(ly, lx, elev, kx=ky, ky=kx)
                    elev_f = _sp(ly_f, lx_f)
                else:
                    elev_f, lx_f, ly_f = elev, lx, ly
            except ImportError:
                elev_f, lx_f, ly_f = elev, lx, ly

            # Use terrain centre elevation as ground_z fallback
            if ground_z is None:
                ground_z = float(elev[tny // 2, tnx // 2])

            oz = ground_z
            elev_rel = elev_f - oz           # height above building ground

            GX, GY = np.meshgrid(lx_f, ly_f)
            traces.append(go.Surface(
                x=GX, y=GY, z=elev_rel,
                colorscale='earth', showscale=True,
                colorbar=dict(title='Height above ground (m)', thickness=12, len=0.5, x=1.02),
                opacity=0.85,
                name='Terrain',
                hovertemplate='ΔX=%{x:.1f} m<br>ΔY=%{y:.1f} m<br>h=%{z:.1f} m<extra></extra>',
            ))
            all_x.extend([float(lx_f[0]), float(lx_f[-1])])
            all_y.extend([float(ly_f[0]), float(ly_f[-1])])
            all_z.extend([float(elev_rel.min()), float(elev_rel.max())])
        except Exception:
            pass  # terrain is nice-to-have

    if ground_z is None:
        ground_z = 0.0
    oz = ground_z

    # ---- Layer 2: Shadow boundary panels (buffer circle edges excluded) -------
    # Each floor's shadow polygon boundary is extruded ±floor_height/2 around
    # the floor mid-height.  Segments that lie on the analysis buffer circle
    # (artefacts of the search radius, not real buildings) are filtered out.
    shadows_geom = row.get('br__shadows')
    if isinstance(shadows_geom, dict) and shadows_geom:
        try:
            # Determine buffer radius: max distance of any shadow vertex from origin
            shadow_radius = 0.0
            for poly in shadows_geom.values():
                if poly is None or (hasattr(poly, 'is_empty') and poly.is_empty):
                    continue
                for g in (list(poly.geoms) if hasattr(poly, 'geoms') else [poly]):
                    if hasattr(g, 'exterior'):
                        for xp, yp, *_ in g.exterior.coords:
                            d = math.sqrt((xp - ox) ** 2 + (yp - oy) ** 2)
                            shadow_radius = max(shadow_radius, d)
            # Segments are on the circle if both endpoints are within 2 % of radius
            circle_tol = shadow_radius * 0.02 if shadow_radius > 0 else 0.0

            def _on_circle(x0, y0, x1, y1):
                d0 = math.sqrt((x0 - ox) ** 2 + (y0 - oy) ** 2)
                d1 = math.sqrt((x1 - ox) ** 2 + (y1 - oy) ** 2)
                return (d0 >= shadow_radius - circle_tol and
                        d1 >= shadow_radius - circle_tol)

            shade_colours = ['#7F8C8D', '#95A5A6', '#BDC3C7', '#909090',
                             '#808B96', '#778899', '#A9A9A9', '#696969']
            for floor_idx, poly in sorted(shadows_geom.items()):
                if poly is None or (hasattr(poly, 'is_empty') and poly.is_empty):
                    continue
                z_bot_rel = floor_idx * floor_height
                z_top_rel = floor_idx * floor_height + floor_height
                z_mid_rel = (z_bot_rel + z_top_rel) / 2
                colour = shade_colours[floor_idx % len(shade_colours)]

                geoms_list = list(poly.geoms) if hasattr(poly, 'geoms') else [poly]
                sx, sy, sz = [], [], []
                si, sj, sk = [], [], []
                for g in geoms_list:
                    if not hasattr(g, 'exterior'):
                        continue
                    ring = list(g.exterior.coords)
                    for ci in range(len(ring) - 1):
                        x0, y0 = ring[ci][:2]
                        x1, y1 = ring[ci + 1][:2]
                        if _on_circle(x0, y0, x1, y1):
                            continue
                        rx0, ry0 = x0 - ox, y0 - oy
                        rx1, ry1 = x1 - ox, y1 - oy
                        base = len(sx)
                        sx.extend([rx0, rx1, rx1, rx0])
                        sy.extend([ry0, ry1, ry1, ry0])
                        sz.extend([z_bot_rel, z_bot_rel, z_top_rel, z_top_rel])
                        si.extend([base,     base])
                        sj.extend([base + 1, base + 2])
                        sk.extend([base + 2, base + 3])
                        all_x.extend([rx0, rx1])
                        all_y.extend([ry0, ry1])
                        all_z.extend([z_bot_rel, z_top_rel])

                if sx:
                    traces.append(go.Mesh3d(
                        x=sx, y=sy, z=sz,
                        i=si, j=sj, k=sk,
                        color=colour, opacity=0.35,
                        name=f'Shade floor {floor_idx}',
                        showlegend=True, flatshading=True,
                        hovertemplate=f'Floor {floor_idx}<br>h={z_mid_rel:.1f} m<extra></extra>',
                    ))
        except Exception:
            pass  # shades are nice-to-have

    # ---- Layer 3: Wall segments (centroid-relative X/Y, ground-relative Z) ---
    type_verts = {}
    _wall_x, _wall_y, _wall_z = [], [], []   # wall-only coords for scene span
    for seg in wall_segments:
        if not isinstance(seg, dict):
            continue
        geom = seg.get('geometry')
        wtype = seg.get('type')
        floor = seg.get('floor', 0)
        try:
            floor = int(floor)
        except (TypeError, ValueError):
            floor = 0
        if geom is None:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            continue

        z_bot_rel = floor * floor_height
        z_top_rel = (floor + 1) * floor_height

        bucket = type_verts.setdefault(wtype, {'x': [], 'y': [], 'z': [],
                                                'i': [], 'j': [], 'k': []})
        for ci in range(len(coords) - 1):
            x0, y0 = coords[ci][:2]
            x1, y1 = coords[ci + 1][:2]
            rx0, ry0 = x0 - ox, y0 - oy
            rx1, ry1 = x1 - ox, y1 - oy
            base = len(bucket['x'])
            bucket['x'].extend([rx0, rx1, rx1, rx0])
            bucket['y'].extend([ry0, ry1, ry1, ry0])
            bucket['z'].extend([z_bot_rel, z_bot_rel, z_top_rel, z_top_rel])
            bucket['i'].extend([base,     base])
            bucket['j'].extend([base + 1, base + 2])
            bucket['k'].extend([base + 2, base + 3])
            all_x.extend([rx0, rx1])
            all_y.extend([ry0, ry1])
            all_z.extend([z_bot_rel, z_top_rel])
            _wall_x.extend([rx0, rx1])
            _wall_y.extend([ry0, ry1])
            _wall_z.extend([z_bot_rel, z_top_rel])

    for wtype, b in type_verts.items():
        colour = _WALL_TYPE_COLOURS.get(wtype, _WALL_TYPE_DEFAULT)
        label = (wtype or 'unknown').replace('_', ' ').title()
        traces.append(go.Mesh3d(
            x=b['x'], y=b['y'], z=b['z'],
            i=b['i'], j=b['j'], k=b['k'],
            color=colour, opacity=1.0,
            name=label, showlegend=True, flatshading=True,
        ))

    # ---- Layer 4: Structural floor/roof slabs (br__structural_floors) --------
    # Reconstruct the horizontal slab surfaces from the per-floor footprints
    # and classify each region with the same interface logic used during
    # inference:  terrain (floor-0 base), adiabatic (slab between two floors),
    # roof (top exposed to air), floor (overhang underside exposed to air).
    fp_by_floor = row.get('br__building_footprint_by_floor')
    if isinstance(fp_by_floor, list) and fp_by_floor:
        try:
            from shapely.ops import unary_union as _uu_floor
            from shapely import wkb as _swkb_floor

            def _decode_floor_geom(g):
                if isinstance(g, (bytes, bytearray, memoryview)):
                    try:
                        return _swkb_floor.loads(bytes(g))
                    except Exception:
                        return None
                return g

            # Group footprints by floor index → union geometry per floor.
            floor_union = {}
            for rec in fp_by_floor:
                try:
                    # records are [group, floor, geometry]
                    _fl = int(rec[1]); _g = _decode_floor_geom(rec[2])
                except Exception:
                    continue
                if _g is None or getattr(_g, 'is_empty', True):
                    continue
                floor_union.setdefault(_fl, []).append(_g)
            for _fl in list(floor_union):
                try:
                    floor_union[_fl] = _uu_floor(floor_union[_fl])
                except Exception:
                    floor_union[_fl] = None

            above_floors = [f for f in floor_union if f >= 0]
            max_floor = max(above_floors) if above_floors else -1

            # region (polygon), type, z-elevation
            struct_regions = []

            def _safe_diff(a, b):
                try:
                    return a.difference(b)
                except Exception:
                    return None

            def _safe_inter(a, b):
                try:
                    return a.intersection(b)
                except Exception:
                    return None

            # Terrain: base of floor 0 at ground level (z = 0).
            base0 = floor_union.get(0)
            if base0 is not None and not base0.is_empty:
                struct_regions.append((base0, 'terrain', 0.0))

            for _f in range(0, max_floor + 1):
                a = floor_union.get(_f)
                if a is None or a.is_empty:
                    continue
                z_top = (_f + 1) * floor_height        # top-of-floor slab level
                above = floor_union.get(_f + 1)
                if above is None or above.is_empty:
                    struct_regions.append((a, 'roof', z_top))
                    continue
                inter = _safe_inter(a, above)
                roof_reg = _safe_diff(a, above)        # not covered above → roof
                over_reg = _safe_diff(above, a)        # overhang underside → floor
                if inter is not None and not inter.is_empty:
                    struct_regions.append((inter, 'adiabatic', z_top))
                if roof_reg is not None and not roof_reg.is_empty:
                    struct_regions.append((roof_reg, 'roof', z_top))
                if over_reg is not None and not over_reg.is_empty:
                    struct_regions.append((over_reg, 'floor', z_top))

            # Accumulate triangulated meshes per type so each type is a single
            # toggleable legend trace.
            struct_buckets = {}
            for region, stype, z_val in struct_regions:
                verts, tris = _triangulate_polygon_xy(region)
                if not verts or not tris:
                    continue
                bkt = struct_buckets.setdefault(
                    stype, {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': []})
                base = len(bkt['x'])
                for (vx, vy) in verts:
                    rx, ry = vx - ox, vy - oy
                    bkt['x'].append(rx); bkt['y'].append(ry); bkt['z'].append(z_val)
                    all_x.append(rx); all_y.append(ry); all_z.append(z_val)
                for (ti, tj, tk) in tris:
                    bkt['i'].append(base + ti)
                    bkt['j'].append(base + tj)
                    bkt['k'].append(base + tk)

            for stype, b in struct_buckets.items():
                colour = _STRUCTURAL_FLOOR_COLOURS.get(stype, _STRUCTURAL_FLOOR_DEFAULT)
                label = f'Slab: {stype}'
                traces.append(go.Mesh3d(
                    x=b['x'], y=b['y'], z=b['z'],
                    i=b['i'], j=b['j'], k=b['k'],
                    color=colour, opacity=0.85,
                    name=label, showlegend=True, flatshading=True,
                    hovertemplate=f'{label}<extra></extra>',
                ))
        except Exception:
            pass  # structural slabs are nice-to-have

    # ---- 1:1:1 aspect — XY up to 4× building span (or available data) --------
    # Z is sized to the building; XY expands to show terrain/shadow context
    # up to 4× the building footprint or the terrain extent, whichever is less.
    if _wall_x and _wall_z:
        wx_span = max(_wall_x) - min(_wall_x)
        wy_span = max(_wall_y) - min(_wall_y)
        wz_span = max(_wall_z) - min(_wall_z)
        bldg_xy = max(wx_span, wy_span, 1.0)
        wcx     = (min(_wall_x) + max(_wall_x)) / 2
        wcy     = (min(_wall_y) + max(_wall_y)) / 2
        wcz     = (min(_wall_z) + max(_wall_z)) / 2
        # XY: expand to 4× building footprint, capped by available all_x/y data
        target_xy = bldg_xy * 4.0
        if all_x:
            data_xy = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
            xy_span = min(target_xy, data_xy)
        else:
            xy_span = target_xy
        # Z: building height with small padding
        z_span  = max(wz_span, 1.0) * 1.3
        # Use the larger of xy/z so the cube satisfies 1:1:1
        span    = max(xy_span, z_span)
        x_range = [wcx - span / 2, wcx + span / 2]
        y_range = [wcy - span / 2, wcy + span / 2]
        z_range = [wcz - span / 2, wcz + span / 2]
    else:
        x_range = y_range = z_range = [-1, 1]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'3-D building geometry — {building_reference}',
        scene=dict(
            xaxis=dict(title='ΔX (m)', range=x_range),
            yaxis=dict(title='ΔY (m)', range=y_range),
            zaxis=dict(title='Height above ground (m)', range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        legend=dict(title='Layer', x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if output_path:
        ext = output_path.rsplit('.', 1)[-1].lower() if '.' in output_path else ''
        if ext != 'html':
            output_path = (output_path.rsplit('.', 1)[0] + '.html'
                           if '.' in output_path else output_path + '.html')
        fig.write_html(output_path)
        print(f"3-D geometry plot saved to {output_path}")
    else:
        fig.show()
    return fig