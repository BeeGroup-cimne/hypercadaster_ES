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
from geopandas import sjoin, sjoin_nearest
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
import concurrent.futures
import multiprocessing

def cadaster_dir(wd):
    return f"{wd}/cadaster"

def districts_dir(wd):
    return f"{wd}/districts"

def census_tracts_dir(wd):
    return f"{wd}/census_tracts"

def results_dir(wd):
    return f"{wd}/results"

def DEM_raster_dir(wd):
    return f"{wd}/DEM_rasters"

def postal_codes_dir(wd):
    return f"{wd}/postal_codes"

def neighborhoods_dir(wd):
    return f"{wd}/neighbourhoods"

def open_data_dir(wd):
    return f"{wd}/open_data"

def create_dirs(data_dir):
    os.makedirs(census_tracts_dir(data_dir), exist_ok=True)
    os.makedirs(districts_dir(data_dir), exist_ok=True)
    os.makedirs(cadaster_dir(data_dir), exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/buildings", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/buildings/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/buildings/unzip", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/address", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/address/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/address/unzip", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/parcels", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/parcels/zip", exist_ok=True)
    os.makedirs(f"{cadaster_dir(data_dir)}/parcels/unzip", exist_ok=True)
    os.makedirs(results_dir(data_dir), exist_ok=True)
    os.makedirs(DEM_raster_dir(data_dir), exist_ok=True)
    os.makedirs(f"{DEM_raster_dir(data_dir)}/raw", exist_ok=True)
    os.makedirs(f"{DEM_raster_dir(data_dir)}/uncompressed", exist_ok=True)
    os.makedirs(neighborhoods_dir(data_dir), exist_ok=True)
    os.makedirs(postal_codes_dir(data_dir), exist_ok=True)
    os.makedirs(f"{postal_codes_dir(data_dir)}/raw", exist_ok=True)
    os.makedirs(open_data_dir(data_dir), exist_ok=True)


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
                        extract_full_path = os.path.join(untar_directory, f"{"" if counter == 0 else str(counter)}{newfile}")

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

def process_zone(gdf_building_parts, buffer_neighbours, neighbours_column_name, neighbours_id_column_name = "single_building_reference"):
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


def process_zone_chunk(chunk, buffer_neighbours, neighbours_column_name, neighbours_id_column_name):
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
    chunk[neighbours_column_name] = chunk[neighbours_id_column_name].map(
        lambda x: ','.join(sorted(related_buildings_map[x])))

    return chunk


def process_zone_parallel(gdf_building_parts, buffer_neighbours, neighbours_column_name,
                          neighbours_id_column_name="single_building_reference", num_workers=4):
    # Split the data into chunks for parallel processing
    chunks = np.array_split(gdf_building_parts, num_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Parallel processing of each chunk
        futures = [
            executor.submit(process_zone_chunk, chunk, buffer_neighbours, neighbours_column_name,
                            neighbours_id_column_name)
            for chunk in chunks
        ]

        # Collect the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

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

def calculate_floor_footprints(gdf, group_by = None, geometry_name = "geometry", only_exterior_geometry = False,
                               min_hole_area = 1e-6, gap_tolerance=1e-6):
    """
    Generate the 2D footprint geometry for each floor level by merging overlapping geometries.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing building parts with geometries and associated floor levels.
    group_by (str, optional): A column name to group the buildings. Default is None.
    geometry_name (str): The name of the geometry column. Default is "geometry".

    Returns:
    floor_footprints_gdf (GeoDataFrame): A GeoDataFrame with the 2D footprint for each floor.
    """
    # Group building parts by their floor levels
    floor_footprints = []

    gdf = gdf.set_geometry(geometry_name)

    if group_by is not None:
        unique_groups = gdf[group_by].unique()
    else:
        unique_groups = ['all']
    if len(unique_groups) > 1000:
        unique_groups = tqdm(unique_groups, desc="Calculating footprints by building floor...")

    for group in unique_groups:
        # Select the grouped buildings
        if group_by is None:
            gdf_ = gdf
        else:
            gdf_ = gdf[gdf[group_by] == group]

        # Unique levels of floors (could be actual heights or floor numbers)
        unique_floors = list(range(gdf_['n_floors_above_ground'].max()))  # Include all floors starting from 0
        for floor in unique_floors:
            # Filter geometries that belong to the current floor level
            floor_geometries = gdf_[gdf_['n_floors_above_ground'] >= (floor + 1)]

            # Reset the index and prepare geometries for union
            floor_geometries = floor_geometries.reset_index(drop=True)

            unioned_geometry = union_geoseries_with_tolerance(floor_geometries, gap_tolerance=gap_tolerance, resolution=16)

            # Step 2: Subtract any holes (interiors) from the resultant geometry
            if only_exterior_geometry:
                if isinstance(unioned_geometry, Polygon):
                    # Create a new polygon without the holes
                    unioned_geometry = Polygon(unioned_geometry.exterior)
                elif isinstance(unioned_geometry, MultiPolygon):
                    # Process each polygon in the MultiPolygon
                    unioned_geometry = MultiPolygon([Polygon(poly.exterior) for poly in unioned_geometry.geoms])
            else:
                # Step 2: Process the unioned geometry to remove small holes
                if isinstance(unioned_geometry, Polygon):
                    # Remove small holes (interiors)
                    cleaned_interiors = [interior for interior in unioned_geometry.interiors if
                                         Polygon(interior).area >= min_hole_area]
                    unioned_geometry = Polygon(unioned_geometry.exterior, cleaned_interiors)

                elif isinstance(unioned_geometry, MultiPolygon):
                    cleaned_polygons = []
                    for poly in unioned_geometry.geoms:
                        # Remove small holes (interiors)
                        cleaned_interiors = [interior for interior in poly.interiors if
                                             Polygon(interior).area >= min_hole_area]
                        cleaned_polygons.append(Polygon(poly.exterior, cleaned_interiors))
                    unioned_geometry = MultiPolygon(cleaned_polygons)

            # Append the result to the list with the associated floor level
            floor_footprints.append({
                'group': group,
                'floor': floor,
                'geometry': unioned_geometry
            })

    # Create a new GeoDataFrame from the list of floor footprints
    floor_footprints_gdf = gpd.GeoDataFrame(floor_footprints, crs=gdf.crs)

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



def patios_in_the_building(patios_geoms, building_geom, tolerance=0.95):
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
        intersection_area = patio_geom.intersection(building_geom).area

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

    for angle in direction_angles:

        ray = create_ray_from_centroid(centroid, angle, length=75)
        further_point_geom = ray.intersection(linearring)
        further_point_geom = get_furthest_point(further_point_geom, centroid) if isinstance(further_point_geom,
                                                                                         MultiPoint) else further_point_geom
        point_to_linearring_distance = centroid.distance(further_point_geom)
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
                geometries=list(other_polygons) + [LineString([further_point_geom, other_pol_intersection_point])] +
                           [centroid],
                contextual_geometry=Polygon(linearring),
                title=f"Building shadows, orientation: {angle}º, floor: {floor},\ndistance: "
                      f"{str(distances_by_orientation[str(angle)])}m",
                ax=ax
            )
            pdf.savefig(fig)
            plt.close(fig)

    return distances_by_orientation


def process_building_parts(building_part_gdf_, results_dir = None, plots = False):
    results = []
    if plots:
        if results_dir is None:
            results_dir = "results"
        os.makedirs(f"{results_dir}/plots", exist_ok=True)

    gdf_global = process_zone_parallel(gdf_building_parts = building_part_gdf_,
                                       buffer_neighbours = 50,
                                       neighbours_column_name = "nearby_buildings",
                                       neighbours_id_column_name = "building_reference",
                                       num_workers=max(1, math.ceil(multiprocessing.cpu_count()/3)) )
    gdf_footprints_global = calculate_floor_footprints(
        gdf=gdf_global,
        group_by="building_reference",
        geometry_name="building_part_geometry",
        only_exterior_geometry=True,
        min_hole_area=0.5,
        gap_tolerance=0.05)

    grouped = gdf_global.groupby("zone_reference")
    for i_zone in tqdm(range(len(grouped)), desc="Inferring attributes of building parts..."):
        zone_reference = list(grouped.groups.keys())[i_zone]
        gdf_zone = grouped.get_group(zone_reference).reset_index().drop(['index'], axis=1).copy()

        # PDF setup for the zone if pdf_plots is True
        if plots:
            pdf = PdfPages(f"{results_dir}/plots/zone_{zone_reference}.pdf")

        gdf_zone = process_zone(gdf_building_parts = gdf_zone,
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
            gap_tolerance=0.05)

        gdf_aux_footprints_global = calculate_floor_footprints(
            gdf=gdf_zone,
            geometry_name="building_part_geometry",
            min_hole_area=0.5,
            gap_tolerance=0.05)

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

        for single_building_reference, gdf_building in gdf_zone.groupby('single_building_reference'):
            # grouped2 = gdf_zone.groupby("single_building_reference")
            # single_building_reference = list(grouped2.groups.keys())[8]
            # gdf_building = grouped2.get_group(single_building_reference).reset_index().drop(['index'], axis=1).copy()

            # PDF setup for the building if plots is True
            if plots:
                pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}.pdf")

            # Obtain a general geometry of the building, considering all floors.
            building_geom = union_geoseries_with_tolerance(gdf_building['building_part_geometry'], gap_tolerance=0.05,
                                                           resolution=16)

            # Extract and clean the set of neighbouring buildings
            adjacent_buildings_set = {id for ids in gdf_building['adjacent_buildings'] for id in ids.split(",")}
            adjacent_buildings_set.discard(
                single_building_reference)  # Safely remove the single_building_reference itself if present
            adjacent_buildings = sorted(adjacent_buildings_set)

            # Extract and clean the set of nearby buildings
            nearby_buildings_set = {id for ids in gdf_building['nearby_buildings'] for id in ids.split(",")}
            nearby_buildings_set.discard(
                single_building_reference.split("_")[0])  # Safely remove the single_building_reference itself if present
            nearby_buildings = sorted(nearby_buildings_set)

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
            roof_area = {}
            patios_area = {}
            patios_n = {}
            perimeter = {}
            air_contact_wall = {}
            shadows_at_distance = {}
            adiabatic_wall = {}
            patios_wall = {}
            n_floors = gdf_aux_footprint_building.floor.max()
            orientation_interval = 5

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
                        building_geom=building_geom
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
                                                                                list(range(0, 360, orientation_interval))]}
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
                                orientation_interval=orientation_interval))

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

                # Close the PDF file for the current building
                if plots:
                    pdf.close()

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
                    #     orientation_interval = orientation_interval,
                    #     plots = plots,
                    #     pdf = pdf,
                    #     floor = floor
                    # )
                    shadows_at_distance[floor] = distance_from_centroid_to_polygons_by_orientation(
                        linearring=building_geom.exterior if isinstance(building_geom, Polygon) else (
                            MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                        other_polygons=gdf_aux_footprints_nearby[gdf_aux_footprints_nearby.floor == floor].geometry,
                        centroid=building_geom.centroid,
                        orientation_interval=orientation_interval,
                        plots=plots,
                        pdf=pdf,
                        floor=floor
                    )
                if plots:
                    pdf.close()

                # Store results
                results.append({
                    'building_reference': single_building_reference.split("_")[0],
                    'single_building_reference': single_building_reference,
                    'n_floors': n_floors,
                    'n_underground_floors': gdf_building.n_floors_below_ground.max(),
                    'floor_area': [floor_area[floor] for floor in range(n_floors + 1)],
                    'roof_area': [roof_area[floor] for floor in range(n_floors + 1)],
                    'patios_area': [patios_area[floor] for floor in range(n_floors + 1)],
                    'patios_n': [patios_n[floor] for floor in range(n_floors + 1)],
                    'perimeter': [perimeter[floor] for floor in range(n_floors + 1)],
                    'air_contact_wall': {key: [air_contact_wall[d][key] for d in air_contact_wall] for key in
                                         air_contact_wall[0]},
                    'adiabatic_wall': [adiabatic_wall[floor] for floor in range(n_floors + 1)],
                    'patios_wall': [patios_wall[floor] for floor in range(n_floors + 1)],
                    'shadows_at_distance': {key: [shadows_at_distance[d][key] for d in shadows_at_distance] for key in
                                            shadows_at_distance[0]}
                    # 'shadows_at_distance': {key: [shadows_at_distance[d][0][key] for d in shadows_at_distance] for key in
                    #                         shadows_at_distance[0][0]},
                    # 'shadows_representativity': {key: [shadows_at_distance[d][1][key] for d in shadows_at_distance] for key in
                    #                         shadows_at_distance[0][1]},
                })


    return pd.DataFrame(results)

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