# Imports organized by category
# Core libraries
import copy
import sys
import os
import time
import shutil
from ._pretty import PrettyDict
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
# COLUMN NAMING UTILITIES
# ==========================================

def to_snake_case(name):
    """Normalise an identifier to ``snake_case``.

    Handles CamelCase (``airTemperatureMean`` -> ``air_temperature_mean``),
    acronym runs (``GHI`` -> ``ghi``, ``IncomeP80P20`` -> ``income_p80_p20``)
    and free text with spaces/punctuation (``Gini Index`` -> ``gini_index``).
    Already-snake lowercase names are returned unchanged. Output column names in
    hypercadaster_ES are always snake_case (never CamelCase)."""
    if name is None:
        return name
    s = str(name).strip()
    # ABCDef -> ABC_Def (split an acronym run from a following CamelCase word)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # abcDef / 8Def -> abc_Def / 8_Def (split lower/digit from a following Upper)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    # collapse any run of non-alphanumerics into a single underscore
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    return s.strip('_').lower()


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
# STAGE TIMING / INSTRUMENTATION
# ==========================================
# Lightweight, always-flushed stderr markers used to locate the
# post-KPI-inference bottleneck on Barcelona-scale runs. The pipeline
# can sit silently for a long time between progress bars (e.g. the gap
# between "Inferring building KPIs" finishing and "Aggregating zone
# shards" starting); these markers make every stage boundary visible
# with wall-clock timing so a single run pinpoints where the time goes.

# Process start, captured at import time, so each marker also shows how
# far into the overall run we are.
_STAGE_T0 = time.time()

# Verbosity: when False (verbose="simple", the default), the per-stage
# ``[stage +… | RSS …]`` markers and per-municipality progress prints are
# suppressed; ``verbose="full"`` turns them on. Set via :func:`set_verbose`
# (called by ``hc.merge``).
_VERBOSE_FULL = False


def set_verbose(level):
    """Set global logging verbosity from a ``"simple"``/``"full"`` level.

    ``"full"`` enables the ``[stage …]`` markers and per-municipality progress
    logs; anything else (``"simple"``, the default) suppresses them.
    """
    global _VERBOSE_FULL
    _VERBOSE_FULL = str(level).lower() == "full"


def is_verbose():
    """Whether full (verbose="full") logging is currently enabled."""
    return _VERBOSE_FULL


def verbose_log(message):
    """Write ``message`` to stderr only when full verbosity is enabled."""
    if _VERBOSE_FULL:
        sys.stderr.write(message)
        sys.stderr.flush()


def _rss_mb():
    """Current process resident-set size in MiB, or None if unavailable.

    Reads /proc/self/status (Linux, no dependency). Used to surface the
    memory footprint at each stage boundary so OOM-prone stages are
    obvious in the log without a profiler.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MiB
    except Exception:
        pass
    return None


def stage_log(message):
    """Emit an immediately-flushed, timestamped stage marker to stderr.

    Format: ``[stage +H:MM:SS | RSS NNNN MB] message`` where the offset
    is wall-clock time since this module was imported (≈ process start)
    and RSS is the current resident memory. Cheap enough to call
    anywhere in the hot path's outer loop. Suppressed unless full verbosity
    is enabled (``hc.merge(verbose="full")``).
    """
    if not _VERBOSE_FULL:
        return
    elapsed = int(time.time() - _STAGE_T0)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    rss = _rss_mb()
    rss_str = f" | RSS {rss:,.0f} MB" if rss is not None else ""
    sys.stderr.write(f"[stage +{h}:{m:02d}:{s:02d}{rss_str}] {message}\n")
    sys.stderr.flush()


@contextmanager
def stage_timer(name):
    """Context manager that brackets a stage with start/end stderr markers.

    Logs ``▶ <name>`` on entry and ``✔ <name> (took Xs)`` on exit (or
    ``�’ <name> FAILED`` if the body raises, re-raising afterwards). The
    "took" duration is the stage's own wall time, independent of the
    process-relative offset in the marker prefix.
    """
    stage_log(f"▶ {name}")
    t = time.time()
    try:
        yield
    except BaseException:
        stage_log(f"✗ {name} FAILED after {time.time() - t:.1f}s")
        raise
    else:
        stage_log(f"✔ {name} (took {time.time() - t:.1f}s)")


# ==========================================
# DIRECTORY MANAGEMENT
# ==========================================


def cadaster_dir_(wd):
    """Path to the cadastral data subdirectory of working dir ``wd``."""
    return f"{wd}/cadaster"

def districts_dir_(wd):
    """Path to the districts subdirectory of working dir ``wd``."""
    return f"{wd}/districts"

def census_tracts_dir_(wd):
    """Path to the census-tracts subdirectory of working dir ``wd``."""
    return f"{wd}/census_tracts"

def results_dir_(wd):
    """Path to the results subdirectory (holds the DuckDB) of working dir ``wd``."""
    return f"{wd}/results"

def open_street_dir_(wd):
    """Path to the OpenStreetMap cache subdirectory of working dir ``wd``."""
    return f"{wd}/os_maps"

def DEM_raster_dir_(wd):
    """Path to the DEM raster subdirectory of working dir ``wd``."""
    return f"{wd}/DEM_rasters"

def postal_codes_dir_(wd):
    """Path to the postal-codes subdirectory of working dir ``wd``."""
    return f"{wd}/postal_codes"

def neighborhoods_dir_(wd):
    """Path to the neighbourhoods subdirectory of working dir ``wd``."""
    return f"{wd}/neighbourhoods"

def open_data_dir_(wd):
    """Path to the (Barcelona) open-data subdirectory of working dir ``wd``."""
    return f"{wd}/open_data"

def cartociudad_dir_(wd):
    """Path to the CartoCiudad (per-province address gpkg) subdirectory of ``wd``."""
    return f"{wd}/cartociudad"

def weather_era5_dir_(wd):
    """Path to the ERA5-Land raw/parquet weather cache subdirectory of ``wd``."""
    return f"{wd}/weather/era5land"

def create_dirs(data_dir):
    """Create the full working-directory tree used by :func:`hypercadaster_ES.download`.

    Idempotently makes the cadaster/address/parcel/buildings, census, districts,
    DEM, neighbourhoods, postal-code, open-data and results subfolders under
    ``data_dir``.
    """
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
    os.makedirs(cartociudad_dir_(data_dir), exist_ok=True)
    os.makedirs(open_street_dir_(data_dir), exist_ok=True)
    os.makedirs(weather_era5_dir_(data_dir), exist_ok=True)

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
    """List Catastro municipalities (with their INSPIRE download URLs).

    Scrapes the INSPIRE ATOM feed and, for each municipality, returns its
    ``name`` (``{cadaster_code}-{MUNICIPALITY}``) and the ``url`` of its data
    archive.

    Parameters
    ----------
    province_codes : list[str], optional
        Restrict to these 2-digit province codes; None returns all provinces.
    inspire_url : str
        ATOM feed to scrape (buildings feed by default; pass the parcels or
        addresses feed to enumerate those instead).
    echo : bool, default True
        Print progress per province.

    Returns
    -------
    list[dict]
        ``[{"name": ..., "url": ...}, ...]``.
    """
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
    """Map INE municipality codes to cadastral codes (via the bundled xlsx).

    Parameters
    ----------
    ine_codes : list[str] | None
        INE municipality codes; None returns None.

    Returns
    -------
    list[str] | None
        Cadastral codes in the same order.
    """
    if ine_codes is not None:
        map_code_dict = pd.read_excel(files("hypercadaster_ES").joinpath("cached_files/ine_inspire_codes.xlsx"), dtype=object, engine='openpyxl').set_index(
            'CÓDIGO INE').to_dict()['CÓDIGO CATASTRO']
        cadaster_codes = [map_code_dict[key] for key in ine_codes]
    else:
        cadaster_codes = None
    return cadaster_codes

def cadaster_to_ine_codes(cadaster_codes):
    """Map cadastral codes to INE municipality codes (via the bundled xlsx).

    Parameters
    ----------
    cadaster_codes : list[str] | None
        Cadastral codes; None returns None.

    Returns
    -------
    list[str] | None
        INE codes in the same order.
    """
    if cadaster_codes is not None:
        map_code_dict = pd.read_excel(files("hypercadaster_ES").joinpath("cached_files/ine_inspire_codes.xlsx"), dtype=object, engine='openpyxl').set_index(
            'CÓDIGO CATASTRO').to_dict()['CÓDIGO INE']
        ine_codes = [map_code_dict[key] for key in cadaster_codes]
    else:
        ine_codes = None
    return ine_codes

def municipality_name(ine_codes):
    """Return the official municipality name(s) for the given INE code(s).

    Looks the codes up in the current INE municipality dictionary (fetched from
    ine.es). Accepts a single code or a list.

    Parameters
    ----------
    ine_codes : str | list[str]
        INE municipality code(s).

    Returns
    -------
    list[str]
        Matching municipality names.
    """
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
    """Return municipality/province/autonomous-community naming for the codes.

    Reads the bundled INE↔INSPIRE table and returns the ``ine_code``,
    ``cadaster_code``, ``municipality_name``, ``province_name`` and
    ``autonomous_community_name`` rows for the requested cadastral codes.

    Parameters
    ----------
    cadaster_codes : list[str]
        Cadastral municipality codes to look up.

    Returns
    -------
    pandas.DataFrame
        One row per matched municipality with the naming columns above.
    """
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
    """Download a KML file and convert its placemarks to a GeoJSON string.

    Handles Polygon and MultiPolygon geometries and copies each placemark's
    extended-data attributes into the feature ``properties``.

    Parameters
    ----------
    kml_url : str
        URL of the KML resource.

    Returns
    -------
    str
        A serialised GeoJSON ``FeatureCollection``.
    """
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
    """Extract every ``.zip`` in ``zip_directory`` into ``unzip_directory``.

    Corrupt archives (``BadZipFile``) are deleted rather than raising.
    """
    for file in os.listdir(zip_directory):
        if file.endswith(".zip"):
            try:
                with ZipFile(f"{zip_directory}{file}", 'r') as zip:
                    zip.extractall(unzip_directory)
            except BadZipFile:
                os.remove(f"{zip_directory}{file}")


def untar_directory(tar_directory, untar_directory, files_to_extract):
    """Extract tar members matching a glob pattern into a fresh directory.

    ``untar_directory`` is recreated empty first. Members whose names match
    ``files_to_extract`` (``fnmatch`` pattern) are written out with unique names;
    unreadable archives are removed.

    Parameters
    ----------
    tar_directory : str
        Directory containing the tar archives.
    untar_directory : str
        Output directory (cleared before extraction).
    files_to_extract : str
        ``fnmatch`` glob selecting which members to extract.

    Returns
    -------
    None
    """
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
    """Repair invalid geometries in ``gdf`` in place and return it."""
    gdf.geometry = gdf.geometry.make_valid()
    return gdf

def get_bbox(gdf):
    """Return the overall bounding box of ``gdf`` as ``[minx, miny, maxx, maxy]``."""
    bbox_gdf = gdf.bounds
    lat_min, lon_min, lat_max, lon_max = (
        min(bbox_gdf['minx']), min(bbox_gdf['miny']),
        max(bbox_gdf['maxx']), max(bbox_gdf['maxy']),
    )
    return [lat_min, lon_min, lat_max, lon_max]


def concatenate_tiffs(input_dir, output_file):
    """Mosaic all GeoTIFFs in ``input_dir`` into a single ``output_file`` raster.

    Parameters
    ----------
    input_dir : str
        Directory of input ``.tif`` tiles.
    output_file : str
        Path for the merged GeoTIFF.

    Returns
    -------
    None
    """
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
    """Group touching/near building parts and record each one's neighbours.

    Builds an adjacency graph of parts whose geometries lie within
    ``buffer_neighbours`` of each other. When keyed by
    ``single_building_reference`` the connected components define physically
    distinct buildings (a ``single_building_reference`` is assigned). Each row's
    ``neighbours_column_name`` is set to the comma-separated sorted set of
    related references.

    Parameters
    ----------
    gdf_building_parts : geopandas.GeoDataFrame
        Building parts with geometry in ``geometry_variable``.
    buffer_neighbours : float
        Proximity buffer (metres) defining adjacency.
    neighbours_column_name : str
        Output column receiving the neighbour reference list.
    neighbours_id_column_name : str, default "single_building_reference"
        Identity column used as graph node id (and clustered when this default).
    geometry_variable : str, default "building_part_geometry"
        Geometry column to use.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf_building_parts`` with the cluster id / neighbour columns added.
    """
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
    """Worker for :func:`detect_close_buildings_parallel`: run the adjacency /
    neighbour detection on a single chunk of geometries and return it enriched
    with the neighbour column (and cluster id when clustering is requested)."""
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
    """Parallel version of :func:`detect_close_buildings`.

    Splits ``gdf_building_parts`` into overlapping chunks (grouped by
    ``column_name_to_split``, each padded by ``buffer_neighbours`` so cross-chunk
    adjacencies aren't missed), runs :func:`detect_close_geometries_chunk` across
    ``num_workers`` processes and recombines the results.

    Parameters
    ----------
    gdf_building_parts : geopandas.GeoDataFrame
        Building parts to analyse.
    buffer_neighbours : float
        Proximity buffer (metres).
    neighbours_column_name : str
        Output neighbour-list column.
    neighbours_id_column_name : str, default "single_building_reference"
        Identity column / clustering key.
    num_workers : int, default 4
        Number of parallel workers.
    column_name_to_split : str, optional
        Column used to partition the data into chunks (e.g. cadastral zone).

    Returns
    -------
    geopandas.GeoDataFrame
        Parts enriched with the neighbour / cluster columns.
    """
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
    """Parcel-level counterpart of :func:`detect_close_buildings_parallel`.

    Detects, in parallel, which cadastral parcels lie within
    ``buffer_neighbours`` of each other and records each parcel's neighbours in
    ``neighbours_column_name``.

    Parameters
    ----------
    gdf_parcels : geopandas.GeoDataFrame
        Parcels to analyse.
    buffer_neighbours : float
        Proximity buffer (metres).
    neighbours_column_name : str
        Output neighbour-list column.
    neighbours_id_column_name : str, default "single_building_reference"
        Identity column used as node id.
    num_workers : int, default 4
        Number of parallel workers.
    column_name_to_split : str, optional
        Column used to partition the data into chunks.

    Returns
    -------
    geopandas.GeoDataFrame
        Parcels enriched with the neighbour column.
    """
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
            n_above = gdf_['n_floors_above_ground']
            max_floor = int(n_above.dropna().max())
            min_floor = int(gdf_['n_floors_below_ground'].dropna().max())

            # A building whose every part declares no floors at all — neither
            # above nor below ground — still has a real BuildingPart footprint.
            # Taken literally it yields range(0) == no floors, so no footprints,
            # and the building drops out of the whole per-floor analysis (and,
            # downstream, out of the CAT integration). Read it as a single
            # storey instead; the CAT floor count then rebuilds the rest via
            # ``_augment_building_floors_from_cat``.
            # Deliberately narrow: a 0-floor part *inside* a building that has
            # floors is a canopy/porch and must stay excluded from the storeys,
            # and an underground-only structure must not gain a ground floor.
            if max_floor == 0 and min_floor == 0:
                n_above = n_above.fillna(0).where(lambda s: s > 0, 1)
                max_floor = 1

            unique_floors = list(range(max_floor))
            unique_underground_floors = list(range(1,min_floor+1))

            for floor in unique_floors:
                floor_geometries = gdf_[n_above >= (floor + 1)].reset_index(drop=True)
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
    """Extract all interior rings (patios/courtyards) as polygons.

    Collects the holes of every Polygon/MultiPolygon in ``geoseries`` and returns
    them as standalone Polygon geometries.

    Parameters
    ----------
    geoseries : iterable of shapely geometries
        Building footprints to inspect.

    Returns
    -------
    list[shapely.geometry.Polygon]
        One polygon per interior ring found.
    """
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
    """Return a ``LineString`` ray from ``centroid`` at a compass ``angle``.

    ``angle`` is measured in degrees clockwise from north (0° = +y). Used to cast
    orientation/shadow rays.

    Parameters
    ----------
    centroid : shapely.geometry.Point
        Ray origin.
    angle : float
        Direction in degrees clockwise from north.
    length : float, default 1000
        Ray length in the coordinate units.

    Returns
    -------
    shapely.geometry.LineString
        The two-point ray.
    """
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


def is_tall_ground_floor_use(building_use):
    """Whether a building's use forces the TALLER ground-floor storey regardless of
    how many floors it has. Industrial and agricultural buildings (warehouses, sheds,
    workshops) need the machinery/vehicle clearance of the taller ground floor even
    when they are single-storey. ``building_use`` is the cadastral use string
    (e.g. ``"3_industrial"``, ``"2_agriculture"``)."""
    u = str(building_use or "").lower()
    return "industrial" in u or "agricultur" in u


def effective_ground_floor_height(n_floors_above_ground, ground_floor_height,
                                  floor_height, building_use=None):
    """The ground storey height to use: the taller ``ground_floor_height`` when the
    building has ≥ 5 floors above ground OR is an industrial/agricultural building
    (``is_tall_ground_floor_use``); otherwise the regular ``floor_height``."""
    try:
        n = int(n_floors_above_ground)
    except (TypeError, ValueError):
        n = 0
    if n >= 5 or is_tall_ground_floor_use(building_use):
        return ground_floor_height
    return floor_height


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

    # Read the band once and bilinearly interpolate, so sampling a grid finer
    # than (or comparable to) the DEM resolution yields a smooth surface rather
    # than nearest-neighbour steps. Falls back to nearest sampling if the band
    # cannot be held in memory.
    _band = None
    _inv = None
    try:
        from scipy.ndimage import map_coordinates as _map_coordinates
        _band = rds.read(1, masked=True).astype('float64').filled(np.nan)
        nodata = rds.nodata
        if nodata is not None:
            _band[_band == nodata] = np.nan
        _inv = ~rds.transform   # (x, y) -> fractional (col, row)
    except Exception:
        _band = None

    def sampler(xs, ys):
        if not same_crs:
            xs_d, ys_d = warp_transform(source_crs, dem_crs, list(xs), list(ys))
        else:
            xs_d, ys_d = list(xs), list(ys)
        if _band is not None:
            xa = np.asarray(xs_d, dtype='float64')
            ya = np.asarray(ys_d, dtype='float64')
            # Inverse affine → fractional pixel coords; -0.5 shifts pixel-corner
            # convention to pixel-centre (array-index) convention.
            cols = _inv.a * xa + _inv.b * ya + _inv.c - 0.5
            rows = _inv.d * xa + _inv.e * ya + _inv.f - 0.5
            vals = _map_coordinates(
                _band, [rows, cols], order=1, mode='nearest', cval=np.nan)
            return list(vals)
        # Fallback: nearest-neighbour sampling.
        return [v[0] for v in rds.sample(zip(xs_d, ys_d))]

    sampler.close = rds.close
    _DEM_SAMPLER_CACHE[cache_key] = sampler
    return sampler

# Function to convert MultiPolygon to a list of Polygons, or leave a Polygon unchanged
def convert_to_polygons(geometry):
    """Return a list of ``Polygon`` parts from a Polygon or MultiPolygon.

    Raises ``TypeError`` for any other geometry type.
    """
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry]
    elif isinstance(geometry, Polygon):
        return [geometry]
    else:
        raise TypeError("Input geometry must be a Polygon or MultiPolygon.")

# Function to convert a geometry into one or more LinearRings
def convert_to_linearrings(geom):
    """Convert a Polygon/MultiPolygon/LineString/LinearRing into LinearRing(s).

    Returns the exterior ring(s); raises ``ValueError`` for other types.
    """
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
    """Convert a Polygon/MultiPolygon/LinearRing/LineString into LineString(s).

    Polygon exteriors are oriented clockwise before conversion; raises for other
    geometry types.
    """
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
                                centroid=None, buffer_disk=None,
                                terrain_coverage_threshold=0.75):
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
        blocked_dirs = 0                 # directions where terrain rises above eye
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
            if block_d < max_distance - 1e-9:
                blocked_dirs += 1
            boundary_pts.append((centroid.x + block_d * sin_a,
                                 centroid.y + block_d * cos_a))
        # Only treat the terrain as a shadow caster for this floor once it
        # substantially ENCLOSES the building — the terrain rises above the eye
        # level in at least ``terrain_coverage_threshold`` of the swept
        # directions. Below that, a few local rises are not a terrain contour
        # shadow, so the visibility polygon is the full disk (no terrain shadow).
        if blocked_dirs < terrain_coverage_threshold * n_angles:
            visibility_polygon = buffer_disk    # not terrain-enclosed → no terrain shadow
        else:
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
    """Fetch (and cache) the OpenStreetMap street network for a municipality.

    The municipality is resolved from ``ine_code``/``cadaster_code`` (or a free
    ``query_location``); results are cached under ``open_street_dir`` so repeat
    calls skip the network.

    Parameters
    ----------
    open_street_dir : str
        Cache directory for OSM downloads.
    ine_code, cadaster_code : str, optional
        Municipality selector (one of these or ``query_location`` is required).
    query_location : str, optional
        Free-text place query (e.g. ``"Barcelona, Spain"``).
    crs : optional
        Target CRS for the returned geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        The street-network edges for the municipality.
    """
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
    """Flatten a structured POI tag tree into an osmnx ``tags`` dict.

    Collapses the per-subgroup value lists of ``osm_tags_structured`` (e.g.
    :data:`OSM_TAGS_STRUCTURED`) into ``{main_key: [all values]}`` suitable for
    ``osmnx.features_from_*``.
    """
    tags = {}

    for main_key, subgroups in osm_tags_structured.items():
        values = set()
        for subgroup_values in subgroups.values():
            values.update(subgroup_values)
        tags[main_key] = list(values)

    return tags


def build_poi_columns(osm_tags_structured):
    """Build a tag-value → POI column-name lookup from the structured tag tree.

    Returns ``{main_key: {tag_value: "main_key.sub_key"}}`` (or just ``main_key``
    when the subgroup is ``"all"``), used to bucket POIs into group/subgroup
    columns.
    """
    col_map = {}

    for main_key, subgroups in osm_tags_structured.items():
        col_map.setdefault(main_key, {})

        for sub_key, values in subgroups.items():
            col_name = f"{main_key}.{sub_key}" if sub_key != "all" else main_key

            for v in values:
                col_map[main_key][v] = col_name

    return col_map

# POI groups actually consumed downstream (see mergers.join_pois GROUPS_OF_INTEREST).
# The heavy ``building``/``historic`` groups are intentionally not parsed: they
# pull hundreds of thousands of polygons and are dropped downstream anyway.
POI_SOURCE_KEYS = ["amenity", "emergency", "healthcare", "leisure", "office", "shop", "tourism"]

# In-memory cache of parsed subregion POIs, keyed by (subregion_slug, crs_out),
# so a whole province (231 municipalities in one extract) parses the PBF once.
_SUBREGION_POI_CACHE = {}


def _normalize_ccaa_name(name):
    """Normalize an autonomous-community name for lookup (upper, collapse spaces)."""
    return re.sub(r"\s+", " ", str(name).strip().upper())


def cadaster_code_to_geofabrik_subregion(cadaster_code):
    """Return the Geofabrik subregion slug covering a cadastral municipality code.

    Resolves the municipality's autonomous community offline (from the bundled
    INE↔INSPIRE table) and maps it to the matching Geofabrik OSM extract.
    """
    from hypercadaster_ES import downloaders
    naming = get_administrative_divisions_naming([cadaster_code])
    if naming.empty:
        raise ValueError(f"Could not resolve autonomous community for cadaster code {cadaster_code}.")
    ccaa = naming["autonomous_community_name"].iloc[0]
    slug = downloaders.AUTONOMOUS_COMMUNITY_TO_GEOFABRIK.get(_normalize_ccaa_name(ccaa))
    if slug is None:
        raise ValueError(f"No Geofabrik OSM subregion mapped for autonomous community '{ccaa}'.")
    return slug


def _wanted_poi_attribute_columns():
    """Set of OSM tag columns to retain for the parsed POI groups (from POIS_COLS)."""
    cols = set(POIS_COLS["general"])
    for group in POI_SOURCE_KEYS:
        spec = POIS_COLS.get(group)
        if not spec:
            continue
        cols.update(spec.get("general", []))
        for sub_vals in spec.get("subgroups", {}).values():
            cols.update(sub_vals)
    return cols


def load_subregion_pois(subregion_slug, pbf_path, open_street_dir, crs_out="EPSG:25831"):
    """Parse (and cache) all POIs of interest for a whole Geofabrik subregion.

    Reads the local ``.osm.pbf`` with pyrosm in a single pass over
    :data:`POI_SOURCE_KEYS`, expands the leftover ``tags`` JSON into the columns
    named in :data:`POIS_COLS`, and reduces each POI geometry to its centroid in
    ``crs_out``. The result is memoised in-memory and persisted to a GeoParquet
    file so repeated municipalities (and re-runs) skip the parse.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per POI, geometry = centroid in ``crs_out``, with the group key
        columns (``amenity``, ``shop`` …) plus retained attribute columns.
    """
    cache_key = (subregion_slug, crs_out)
    if cache_key in _SUBREGION_POI_CACHE:
        return _SUBREGION_POI_CACHE[cache_key]

    parquet_dir = os.path.join(open_street_dir, "pbf_pois")
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, f"{subregion_slug.split('/')[-1]}.parquet")

    if os.path.exists(parquet_path):
        try:
            gdf = gpd.read_parquet(parquet_path)
            if gdf.crs is None or str(gdf.crs) != str(crs_out):
                gdf = gdf.to_crs(crs_out)
            _SUBREGION_POI_CACHE[cache_key] = gdf
            return gdf
        except Exception:
            pass

    from pyrosm import OSM

    sys.stderr.write(f"Parsing POIs from OSM extract {subregion_slug} (one-time)...\n")
    osm = OSM(pbf_path)
    pois = osm.get_pois(custom_filter={k: True for k in POI_SOURCE_KEYS})

    if pois is None or len(pois) == 0:
        gdf = gpd.GeoDataFrame(
            {k: [] for k in POI_SOURCE_KEYS}, geometry=[], crs=crs_out
        )
        _SUBREGION_POI_CACHE[cache_key] = gdf
        return gdf

    pois = pois[pois.geometry.notna()].copy()

    # pyrosm exposes common tags as columns and leaves the rest in a ``tags``
    # JSON string; expand the columns our schema needs but that pyrosm folded in.
    wanted = _wanted_poi_attribute_columns() | set(POI_SOURCE_KEYS) | {"id", "name"}
    if "tags" in pois.columns:
        needed = [c for c in wanted if c not in pois.columns]
        if needed:
            parsed = pois["tags"].apply(
                lambda s: json.loads(s) if isinstance(s, str) and s else {}
            )
            tags_df = pd.DataFrame(parsed.tolist(), index=pois.index)
            add_cols = [c for c in needed if c in tags_df.columns]
            if add_cols:
                pois = pd.concat([pois, tags_df[add_cols]], axis=1)

    if "osm_type" in pois.columns and "element" not in pois.columns:
        pois["element"] = pois["osm_type"]

    # Centroids in the output CRS (POIs are reduced to points downstream).
    pois = pois.to_crs(crs_out)
    pois["geometry"] = pois.geometry.centroid

    keep = [c for c in (wanted | {"element", "geometry"}) if c in pois.columns]
    gdf = gpd.GeoDataFrame(pois[keep], geometry="geometry", crs=crs_out)

    try:
        gdf.to_parquet(parquet_path)
    except Exception as e:
        sys.stderr.write(f"Warning: could not cache parsed POIs to {parquet_path}: {e}\n")

    _SUBREGION_POI_CACHE[cache_key] = gdf
    return gdf

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

OSM_TAGS_STRUCTURED = PrettyDict(OSM_TAGS_STRUCTURED)
"""OpenStreetMap POI taxonomy consumed by the POI layer:
``group → subgroup → [OSM tag values]``. Drives both the OSM query tags and the
generated ``poi__{group}[_{subgroup}]`` output columns."""

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

POIS_COLS = PrettyDict(POIS_COLS)
"""OSM attribute columns retained per category when parsing fetched POIs into
GeoDataFrames."""

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
    """Read (and cache) OpenStreetMap POIs for a municipality, grouped by tag.

    POIs are parsed locally with pyrosm from the Geofabrik ``.osm.pbf`` extract
    covering the municipality's autonomous community (downloaded and parsed once
    per subregion, then memoised) instead of being queried tag-by-tag from
    Overpass. The parsed subregion POIs are clipped to the municipality's
    cadastral extent (buffered by ``buffer_m``), bucketed into the group/subgroup
    structure of :data:`OSM_TAGS_STRUCTURED`, and returned organised by
    group/subgroup.

    Parameters
    ----------
    ine_code, cadaster_code, query_location : str, optional
        Municipality selector (one is required).
    geolocate_to_cadaster : bool, default True
        Snap fetched POIs to the cadastral extent using ``cadaster_dir``.
    cadaster_dir : str, optional
        Cadastral data directory (used when ``geolocate_to_cadaster``).
    open_street_dir : str, default "./osm_cache"
        Cache directory for the downloaded ``.osm.pbf`` extracts (``pbf/``) and
        the parsed per-subregion POIs (``pbf_pois/``).
    buffer_m : int, default 200
        Margin (metres) added around the municipality's cadastral extent when
        clipping POIs, so edge buildings' proximity rings still capture them.
    crs_out : str, default "EPSG:25831"
        CRS of the returned geometries.

    Returns
    -------
    dict
        ``{group: GeoDataFrame}`` or ``{group: {subgroup: GeoDataFrame}}``.
    """

    # Resolve the cadastral code (needed both to pick the OSM extract and to
    # derive the cadastral extent used to clip the POIs).
    if not cadaster_code:
        if ine_code:
            cadaster_code = ine_to_cadaster_codes([ine_code])[0]
        else:
            raise ValueError("One of these must be defined: ine_code or cadaster_code.")

    if not cadaster_dir:
        raise ValueError("Cadaster directory (cadaster_dir) must be defined.")

    from hypercadaster_ES import downloaders
    from hypercadaster_ES.mergers import join_cadaster_parcel

    # ------------------------------------------------------------
    # 1) LOCAL OSM EXTRACT -> subregion POIs (downloaded & parsed once)
    # ------------------------------------------------------------
    subregion_slug = cadaster_code_to_geofabrik_subregion(cadaster_code)
    pbf_path = downloaders.download_osm_pbf(open_street_dir, subregion_slug)
    if pbf_path is None:
        raise RuntimeError(f"Could not obtain OSM extract for subregion '{subregion_slug}'.")
    sub_pois = load_subregion_pois(subregion_slug, pbf_path, open_street_dir, crs_out=crs_out)

    # ------------------------------------------------------------
    # 2) MUNICIPALITY EXTENT (from cadastral parcels) + CLIP
    # ------------------------------------------------------------
    parcels_gdf = join_cadaster_parcel(None, cadaster_dir, [cadaster_code], how="left")
    parcels_gdf = parcels_gdf.set_geometry("parcel_geometry")
    parcels_gdf = parcels_gdf.to_crs(crs_out)

    if sub_pois.empty or parcels_gdf.empty:
        clipped = sub_pois.iloc[0:0]
    else:
        # Clip to the municipality bounding box widened by ``buffer_m`` so edge
        # buildings' proximity rings still see nearby POIs.
        minx, miny, maxx, maxy = parcels_gdf.total_bounds
        clipped = sub_pois.cx[minx - buffer_m: maxx + buffer_m,
                              miny - buffer_m: maxy + buffer_m].copy()

    # ------------------------------------------------------------
    # 3) BUCKET INTO GROUP/SUBGROUP (local, vectorised)
    # ------------------------------------------------------------
    # A POI is assigned to every group whose tag it carries (e.g. an amenity that
    # is also a shop lands in both), matching the previous per-tag Overpass runs.
    col_map = build_poi_columns({k: OSM_TAGS_STRUCTURED[k] for k in POI_SOURCE_KEYS})
    results = {}
    for main_key, value_to_subgroup in col_map.items():
        if main_key not in clipped.columns:
            continue
        mapped = clipped[main_key].map(value_to_subgroup)
        valid = mapped.notna()
        if not valid.any():
            continue
        for subgroup_name, grp in clipped[valid].groupby(mapped[valid], sort=False):
            grp = grp.copy()
            grp["poi_subgroup"] = subgroup_name
            results[subgroup_name] = grp

    results = clean_pois(results)

    # -------------------
    # GEOLOCATE TO CADASTRAL REFERENCES
    # -------------------
    if geolocate_to_cadaster:

        def geolocate_leaf_gdf(poi_gdf, subgroup_name):
            """
            Apply cadastral geolocation to a single GeoDataFrame leaf.
            """
            if poi_gdf is None or poi_gdf.empty:
                return poi_gdf

            # Unique index so the row-per-POI bookkeeping below is unambiguous.
            poi_gdf = poi_gdf.to_crs(crs_out).reset_index(drop=True)

            # Spatial join: POIs inside parcels
            poi_with_parcels = gpd.sjoin(
                poi_gdf,
                parcels_gdf[["building_reference", "parcel_geometry"]],
                how="left",
                predicate="within"
            )
            # A POI that falls within several (overlapping) parcels yields one
            # row per parcel; keep the first so there is exactly one row per POI.
            poi_with_parcels = poi_with_parcels[
                ~poi_with_parcels.index.duplicated(keep="first")
            ]

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
                # sjoin_nearest emits one row per equidistant tie; collapse to the
                # first match per POI so the assignment aligns one-to-one with the
                # masked rows (positional ``.values`` would otherwise raise
                # "Must have equal len keys and value ..." when ties inflate it).
                nearest_parcels = nearest_parcels[
                    ~nearest_parcels.index.duplicated(keep="first")
                ]

                poi_with_parcels.loc[
                    no_parcel_mask, "building_reference"
                ] = nearest_parcels["building_reference"]

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

        # Merge shared + specific. ``.get`` guards subgroups present in the tag
        # taxonomy but absent from POIS_COLS (e.g. tourism.rental), which would
        # otherwise raise and drop the whole municipality.
        target_cols = (POIS_COLS["general"] +
                       POIS_COLS[group]["general"] +
                       POIS_COLS[group]["subgroups"].get(subgroup, []) if subgroup else [])

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
    """Count the orientations that carry a significant share of façade length.

    Parameters
    ----------
    orientation_lengths : dict
        ``{orientation_deg: facade_length}``.
    threshold_ratio : float, default 0.1
        Minimum fraction of the total length for an orientation to count.

    Returns
    -------
    int
        Number of orientations at or above the threshold.
    """
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
    """Split a ``LineString`` into its consecutive two-point segments.

    Raises ``ValueError`` if ``line`` is not a LineString with ≥2 points.
    """
    if not isinstance(line, LineString) or len(line.coords) < 2:
        raise ValueError("Input must be a LineString with at least two points.")

    segments = []
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        segments.append(seg)
    return segments

def weighted_circular_mean(degrees, weights):
    """Weighted circular mean of angles (degrees), returned in ``[0, 360)``.

    Averages via the mean sine/cosine so the wrap-around at 360° is handled
    correctly (e.g. the mean of 350° and 10° is 0°, not 180°).
    """
    radians = np.deg2rad(degrees)
    sin_sum = np.sum(np.sin(radians) * weights)
    cos_sum = np.sum(np.cos(radians) * weights)
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    return mean_angle_deg

def discrete_orientation(angle, bin_size=5):
    """Snap an angle (degrees) to the nearest ``bin_size``-degree bin centre."""
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
    """Map an above-ground cadastral floor code to a numeric level.

    Interprets Spanish cadastral abbreviations (e.g. ``BJ``/``EN`` → ground,
    ``PR`` → mezzanine fractions, ``AT``/``+n`` → attic levels, numeric floors as
    themselves) and returns a float level (attics encoded as ``999+n``), or
    ``NaN`` for unrecognised/common-area codes.
    """
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
    """Map a below-ground cadastral floor code to a negative numeric level.

    Interprets basement/parking abbreviations (e.g. ``PK``/``ST`` → -0.5,
    ``-n``/``PS n`` → -n). Returns a float level, or ``NaN`` when the code is not
    a below-ground one.
    """
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
    """Map any cadastral floor code to a numeric level.

    Tries :func:`classify_above_ground_floor_names` first and falls back to
    :func:`classify_below_ground_floor_names` for basement codes.
    """
    agf = classify_above_ground_floor_names(floor)
    if agf is not np.nan:
        return agf
    else:
        return classify_below_ground_floor_names(floor)

# ==========================================
# ADDRESS COMPACTION UTILITIES
# ==========================================

def _reorder_inverted_street_name(name):
    """Un-invert a cadastral street name for human-readable output.

    Spanish cadastre stores street names with the article/preposition moved to
    the end after a comma (e.g. ``"MANRESA, DE"``, ``"TINTS, DELS"``). This
    restores the natural reading order by reversing the comma-separated
    segments: ``"MANRESA, DE" -> "DE MANRESA"``, ``"TINTS, DELS" -> "DELS TINTS"``.
    Names without a comma are returned unchanged; non-strings pass through.
    """
    if not isinstance(name, str):
        return name
    parts = [p.strip() for p in name.split(',') if p.strip()]
    if len(parts) <= 1:
        return name.strip()
    return ' '.join(reversed(parts))


# In Catalan the preposition "de" elides to "d'" only before a vowel or h
# ("d'Aragó", "d'Homs"); before a consonant it stays "de". Some source street
# names carry a wrong/attached elision (e.g. "D'GUITART"), so expand "D'" back to
# "DE " when it is immediately followed by a consonant. A lookahead keeps the
# consonant; the apostrophe may be straight (') or typographic (’).
_CATALAN_ELISION_RE = re.compile(r"\bD['’]\s*(?=[BCÇDFGJKLMNÑPQRSTVWXYZ])", re.IGNORECASE)


def _expand_catalan_elision(name):
    """Expand an incorrectly-elided Catalan ``D'`` to ``DE`` before a consonant.

    ``"D'GUITART" -> "DE GUITART"``, ``"D' GUITART" -> "DE GUITART"``. Vowel/h
    cases (``"D'URGELL"``, ``"D'HOMS"``) and non-strings are left untouched.
    """
    if not isinstance(name, str):
        return name
    return _CATALAN_ELISION_RE.sub("DE ", name)


# Cadastral thoroughfare texts embed the aggregated municipality (pedanía) as an
# ``AG.<NAME>`` marker somewhere inside the street name, delimited from the rest
# by ``;``, ``-`` or whitespace, and the marker itself can sit at the start,
# middle or end (e.g. ``CATRAN;AG.TREDOS``, ``NOU-AG.PALOUET``, ``AG.ESCAS MAJOR``,
# ``DES;AG.BAGERGUE MARIES``). The municipality is taken as the single token that
# follows ``AG.`` — best-effort, so a handful of multi-word pedanía names are
# truncated to their first token.
_AGGREGATED_MUNICIPALITY_RE = re.compile(r"[;\-\s]*AG\.\s*([^\s;,\-]+)", re.IGNORECASE)


def _split_aggregated_municipality(name):
    """Split a cadastral ``AG.<municipality>`` marker off a street name.

    ``"CATRAN;AG.TREDOS" -> ("CATRAN", "TREDOS")``,
    ``"AG.ESCAS MAJOR" -> ("MAJOR", "ESCAS")``. Returns ``(clean_name, None)``
    when no marker is present; non-strings pass through as ``(name, None)``.
    """
    if not isinstance(name, str):
        return name, None
    m = _AGGREGATED_MUNICIPALITY_RE.search(name)
    if not m:
        return name.strip(), None
    municipality = m.group(1).strip()
    clean = re.sub(r"\s+", " ", (name[:m.start()] + " " + name[m.end():])).strip()
    return clean, municipality


def _aggregated_municipality_suffix(municipality):
    """Trailing bracketed note for an aggregated municipality, or ``""``.

    ``"TREDOS" -> " [TREDOS]"`` so ``"CALLE CATRAN;AG.TREDOS, 23"`` renders as
    ``"CALLE CATRAN, 23 [TREDOS]"``.
    """
    return f" [{municipality}]" if municipality else ""


def _create_full_street_names(gdf):
    """
    Helper function to create full street name combinations.

    Creates the following columns if the required components exist:
    - street_full_label: "{street_type} {street_name}, {street_number_clean_label}"
    - street_full_clean: "{street_type} {street_name}, {street_number_clean}"

    The ``street_type`` acronym is expanded to its full literal ("CR" ->
    "CARRETERA") and the ``street_name`` component is un-inverted via
    :func:`_reorder_inverted_street_name`, so ``"CR MANRESA, DE, 50"`` renders as
    ``"CARRETERA DE MANRESA, 50"`` (matching ``br__building_spaces_postal_address``).
    An aggregated-municipality (pedanía) ``AG.<NAME>`` marker is lifted out of the
    name and appended as a bracketed trailing note, so ``"CALLE CATRAN;AG.TREDOS, 23"``
    renders as ``"CALLE CATRAN, 23 [TREDOS]"``.
    """
    from hypercadaster_ES import downloaders  # local import: downloaders imports utils
    _type_map = downloaders.street_type_names

    def _expand_street_type(t):
        return _type_map.get(t, t) if isinstance(t, str) else t

    def _clean_name(n):
        # strip the AG.<municipality> marker, un-invert the remainder, then fix a
        # wrong Catalan "D'" elision that only surfaces once the name is un-inverted
        # (e.g. "GUITART, D'" -> "D' GUITART" -> "DE GUITART").
        clean, _ = _split_aggregated_municipality(n)
        return _expand_catalan_elision(_reorder_inverted_street_name(clean))

    def _agg_suffix(n):
        return _aggregated_municipality_suffix(_split_aggregated_municipality(n)[1])

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
                # Create the full street name: expand the street_type acronym
                # ("CR" -> "CARRETERA") and un-invert the street_name so the
                # article/preposition leads ("MANRESA, DE" -> "DE MANRESA").
                gdf_result[col_name] = (
                    gdf_result[required_cols[0]].map(_expand_street_type).fillna('').astype(str) + ' ' +
                    gdf_result[required_cols[1]].map(_clean_name).fillna('').astype(str) + ', ' +
                    gdf_result[required_cols[2]].fillna('').astype(str)
                )
                # Clean up extra spaces and commas
                gdf_result[col_name] = gdf_result[col_name].str.replace(r'\s+', ' ', regex=True)
                gdf_result[col_name] = gdf_result[col_name].str.replace(r',\s*$', '', regex=True)
                gdf_result[col_name] = gdf_result[col_name].str.strip()
                # Append the aggregated-municipality note AFTER the number
                gdf_result[col_name] = (
                    gdf_result[col_name] + gdf_result[required_cols[1]].map(_agg_suffix).fillna('')
                )

                created_cols.append(col_name)
            except Exception:
                pass  # Silently skip failed column creation

    return gdf_result

def flat_gdf_to_building_spaces_resolution(gdf, building_reference_col="building_reference", chunk_size=50_000):
    """Explode a building-level (Geo)DataFrame to one row per building space.

    Each ``br__building_spaces_<name>`` column holds a ``{use_type: [values]}``
    dict, with the lists aligned across all such columns (they are built
    together, per building, in the CAT aggregation). This flattens them into a
    long DataFrame with one row per individual building space, carrying:

    - ``use_type`` — the space's inferred primary use,
    - one column per ``<name>`` with the ``br__building_spaces_`` prefix rewritten
      to ``building_space_`` (``building_space_reference``,
      ``building_space_postal_address``, ``building_space_inner_address``,
      ``building_space_floor_number``, ``building_space_floors``,
      ``building_space_category``, ``building_space_effective_year``,
      ``building_space_area_without_communal``, ``building_space_area_with_communal``,
      ``building_space_economic_value``, ``building_space_detailed_use_type``, ...),
    - **every other column of** ``gdf`` (building-level context: geometry, areas,
      elevation, POIs, the scalar ``br__*`` aggregates, administrative names,
      ...), replicated on each space row.

    ``br__building_spaces_by_floor`` (a per-floor summary, not a per-space list)
    is excluded from the exploded columns but, being building-level, is retained
    as a context column. Buildings with no per-space data (e.g. undeveloped
    parcels) contribute no rows.

    Parameters
    ----------
    gdf : (Geo)DataFrame
        Building-level frame produced by :func:`hypercadaster_ES.merge`.
    building_reference_col : str, default "building_reference"
        Building id column; kept as context and used to order the output.

    Returns
    -------
    pandas.DataFrame
        One row per building space.
    """

    import gc

    prefix = "br__building_spaces_"
    out_prefix = "building_space_"
    excluded_per_space = {"br__building_spaces_by_floor"}

    def _is_per_space(v):
        # per-space columns are {use_type: [values]} dicts; excludes scalars,
        # NaN, and the {..: count} by-floor summary
        return (isinstance(v, dict) and len(v) > 0
                and all(isinstance(x, list) for x in v.values()))

    # Find per-space columns
    candidate_cols = [
        c for c in gdf.columns
        if c.startswith(prefix) and c not in excluded_per_space
    ]

    per_space_cols = []

    for col in candidate_cols:
        for value in gdf[col]:
            if _is_per_space(value):
                per_space_cols.append(col)
                break

    if not per_space_cols:
        sys.stderr.write("[FLAT] No per-space columns found")
        return pd.DataFrame()

    renamed_space_cols = {
        c: out_prefix + c[len(prefix):]
        for c in per_space_cols
    }

    context_cols = [
        c for c in gdf.columns
        if c not in per_space_cols
    ]

    sys.stderr.write(f"[FLAT] per-space columns: {len(per_space_cols)}")

    sys.stderr.write(f"[FLAT] context columns: {len(context_cols)}")

    sys.stderr.write("[FLAT] Counting building spaces...")

    # Determine how many output rows we need.
    total_records = 0

    n_buildings = len(gdf)

    for start in range(0, n_buildings, chunk_size):
        stop = min(start + chunk_size, n_buildings)
        sys.stderr.write(f"[FLAT] counting buildings {start:,} -> {stop:,}")
        part = gdf.iloc[start:stop]

        for _, row in part.iterrows():
            ref = None
            for col in per_space_cols:
                value = row[col]
                if _is_per_space(value):
                    ref = value
                    break

            if ref is None:
                continue

            for values in ref.values():
                if isinstance(values, list):
                    total_records += len(values)

        del part

    gc.collect()

    sys.stderr.write(f"[FLAT] Total records: {total_records:,}")

    if total_records == 0:
        return pd.DataFrame()

    sys.stderr.write("[FLAT] Allocating output columns...")

    output = {}

    for col in context_cols:
        output[col] = [None] * total_records

    output["use_type"] = [None] * total_records

    for col in per_space_cols:
        output[renamed_space_cols[col]] = [None] * total_records

    pos = 0
    # Fill the preallocated columns.
    for start in range(0, n_buildings, chunk_size):
        stop = min(start + chunk_size, n_buildings)

        sys.stderr.write(f"\n[FLAT] filling buildings {start:,} -> {stop:,}")

        part = gdf.iloc[start:stop]

        for _, row in part.iterrows():
            values_by_col = {}
            for col in per_space_cols:
                value = row[col]
                if _is_per_space(value):
                    values_by_col[col] = value

            if not values_by_col:
                continue

            # First valid per-space column is reference.
            ref = next(iter(values_by_col.values()))

            for use_type, values in ref.items():
                if not isinstance(values, list):
                    continue

                n_spaces = len(values)

                for i in range(n_spaces):
                    for col in context_cols:
                        output[col][pos] = row[col]

                    output["use_type"][pos] = use_type

                    for col in per_space_cols:
                        by_use = values_by_col.get(col)
                        if by_use is None:
                            continue

                        seq = by_use.get(use_type)

                        if isinstance(seq, list) and i < len(seq):
                            output[renamed_space_cols[col]][pos] = seq[i]

                    pos += 1

        del part

        gc.collect()

        sys.stderr.write(f"[FLAT] filled {pos:,} / {total_records:,}")

    if pos != total_records:
        raise RuntimeError(
            f"Expected {total_records:,} records but generated {pos:,}"
        )

    sys.stderr.write("\n[FLAT] Creating final DataFrame...")

    result = pd.DataFrame(output)

    del output
    gc.collect()

    lead = [c for c in (building_reference_col,"use_type") if c in result.columns]
    rest = [c for c in result.columns if c not in lead]
    result = result[lead + rest]

    gc.collect()

    sys.stderr.write(f"[FLAT] Final DataFrame: {len(result):,} rows")

    return result


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
    """Return whether ``val`` is a meaningful (non-empty, non-NaN) value.

    False for None, NaN floats and empty lists/arrays; True otherwise.
    """
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
        'district_code', 'district_name', 'postal_code', 'population_entity',
        'neighborhood_code', 'neighborhood_name', 'census_tract_code'
    ],
    create_full_street_names=True,
    use_unique_for_admin=True):
    """Collapse multiple address rows per building into one row of lists.

    Buildings with several addresses are reduced to a single row where the
    address/geolocation columns become lists; administrative columns are kept as
    single values when uniform (``use_unique_for_admin``). Optionally builds
    combined ``street_full*`` labels.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Address-level frame (one row per address).
    building_reference_col : str, default "building_reference"
        Column identifying the building to group by.
    address_cols, geolocation_cols, admin_cols : list[str]
        Columns treated as per-address, geolocation, and administrative.
    create_full_street_names : bool, default True
        Also emit combined ``street_full*`` label columns.
    use_unique_for_admin : bool, default True
        Keep a single value for administrative columns when uniform per building.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per building with list-valued address columns.
    """
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


def _best_address_match_index(address, ref_labels):
    """Index of the ``ref_labels`` entry most similar to ``address`` (or None).

    The street TYPE prefix is stripped from both sides before comparison
    (``downloaders.strip_street_type``) so a CAT ``CALLE URBANIZACION DEVESA PARK``
    still harmonises to a CartoCiudad ``URBANIZACION DEVESA PARK`` — the street
    NAME (and number) is what should drive the match, not the type token."""
    import difflib
    from hypercadaster_ES import downloaders
    if not isinstance(address, str) or not ref_labels:
        return None
    a = downloaders.strip_street_type(address)
    best_i, best_r = None, -1.0
    for i, lab in enumerate(ref_labels):
        if not isinstance(lab, str):
            continue
        r = difflib.SequenceMatcher(None, a, downloaders.strip_street_type(lab)).ratio()
        if r > best_r:
            best_r, best_i = r, i
    return best_i


def _match_space_addresses(space_addresses, ref_labels, ref_locations):
    """For each space address, return ``(matched_label, matched_location)`` taken
    from the closest reference parcel address (both None when there is no match)."""
    if not isinstance(space_addresses, list):
        return space_addresses, space_addresses
    labels = ref_labels if isinstance(ref_labels, list) else []
    locs = ref_locations if isinstance(ref_locations, list) else []
    out_labels, out_locs = [], []
    for a in space_addresses:
        idx = _best_address_match_index(a, labels)
        out_labels.append(labels[idx] if idx is not None and idx < len(labels) else None)
        out_locs.append(locs[idx] if idx is not None and idx < len(locs) else None)
    return out_labels, out_locs


def match_building_space_postal_addresses(
    gdf,
    spaces_col="br__building_spaces_postal_address",
    labels_col="street_full_labels",
    locations_col="address_location",
    from_cat_col="br__building_spaces_postal_address_from_CAT_file",
    location_col="br__building_spaces_postal_address_location",
):
    """Match every building space's CAT postal address to the closest parcel address.

    For each ``br__building_spaces_postal_address`` entry (a CAT per-space postal
    address) the closest compacted parcel address in ``labels_col``
    (``street_full_labels`` / ``postal_addresses_with_extensions``) is found, and:

    - the original CAT address is preserved in ``from_cat_col``
      (``br__building_spaces_postal_address_from_CAT_file``);
    - ``spaces_col`` (``br__building_spaces_postal_address``) is **replaced** with the
      matched parcel-address string (None when the parcel has no address);
    - ``location_col`` (``br__building_spaces_postal_address_location``) gets the
      matched address point.

    All three mirror the ``{use_type: [...]}`` dict (or plain list) shape of the
    space-address column, so :func:`flat_gdf_to_building_spaces_resolution` explodes
    them to per-space ``building_space_postal_address`` /
    ``building_space_postal_address_from_CAT_file`` /
    ``building_space_postal_address_location`` automatically.

    No-op (returns ``gdf`` unchanged) when any required column is absent — e.g. when
    building-parts inference is disabled and there are no building spaces.
    """
    if not all(c in gdf.columns for c in (spaces_col, labels_col, locations_col)):
        return gdf

    def _row(row):
        spaces = row[spaces_col]
        labels = row[labels_col]
        locs = row[locations_col]
        if isinstance(spaces, dict):
            lab_d, loc_d = {}, {}
            for ut, v in spaces.items():
                lab, loc = _match_space_addresses(v, labels, locs)
                lab_d[ut], loc_d[ut] = lab, loc
            return spaces, lab_d, loc_d
        if isinstance(spaces, list):
            lab, loc = _match_space_addresses(spaces, labels, locs)
            return spaces, lab, loc
        return spaces, None, None

    gdf = gdf.copy()
    res = gdf.apply(_row, axis=1, result_type="expand")
    gdf[from_cat_col] = res[0]     # original CAT address preserved
    gdf[spaces_col] = res[1]       # replaced with the matched parcel address
    gdf[location_col] = res[2]     # matched address point
    return gdf

# ==========================================
# SHADOW VISUALIZATION
# ==========================================

# Half-width (m) of the square XY window used by the shadow-analysis spatial
# panels. `plot_building_3d_geometry` reuses it so both views frame the same
# ground area around the building.
SHADOW_PLOT_EXTENT_R = 150.0


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

    floor 0 sits at ground+g/2; subsequent floors at ground+g+(i-1)*floor_height
    + floor_height/2, where ``g`` is the (taller) ground-floor height — applied
    only to buildings with >= 5 floors above ground (else ``floor_height``),
    matching ``building_inference`` and ``floor_base_elevation``.
    """
    ground = float(shadow_data.get('elevation', 0.0) or 0.0)
    n_bld = int(shadow_data.get('building_n_floors', 1) or 1)
    n = int(shadow_data.get('max_neighbor_floors', n_bld) or 1)
    # >=5-floors rule, OR industrial/agricultural (always the taller ground floor).
    g = effective_ground_floor_height(n_bld, ground_floor_height, floor_height,
                                      shadow_data.get('building_use'))
    levels = []
    for i in range(n):
        if i == 0:
            levels.append(ground + g / 2.0)
        else:
            levels.append(ground + g + (i - 1) * floor_height + floor_height / 2.0)
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


def _terrain_point_cloud(elev, gx, gy):
    """Flatten a DEM grid into an unstructured (x, y, z) point cloud of pixel
    samples, dropping non-finite cells. The grid structure is discarded — the
    samples are treated as scattered points."""
    GX, GY = np.meshgrid(gx, gy)
    xs = GX.ravel(); ys = GY.ravel(); zs = elev.ravel()
    finite = np.isfinite(zs)
    return xs[finite], ys[finite], zs[finite]


def _build_terrain_tin(elev, gx, gy, adaptive=False, max_points=300, tol=0.20):
    """Triangulated Irregular Network from the DEM treated as a **point cloud**.

    The grid is flattened to scattered ``(x, y, z)`` samples and Delaunay-
    triangulated in the XY plane (a terrain height field). Every vertex is a
    real sample — planar facets, no fabricated undulation.

    - ``adaptive=False`` (default): triangulate the *whole* point cloud.
    - ``adaptive=True``: greedy point insertion keeps only the samples needed
      to approximate the surface within `tol` metres (up to `max_points`),
      giving a lighter, irregular mesh — dense where the terrain bends, sparse
      where it is flat.

    Returns ``(x, y, z, triangles)`` with ``triangles`` an (M, 3) int array of
    vertex indices, or ``triangles=None`` on failure.
    """
    xs, ys, zs = _terrain_point_cloud(elev, gx, gy)
    pts = np.column_stack([xs, ys])
    n = len(zs)
    if n < 3:
        return xs, ys, zs, None

    try:
        from scipy.spatial import Delaunay
        if not adaptive:
            # Direct triangulation of the full point cloud.
            return xs, ys, zs, Delaunay(pts).simplices

        from scipy.interpolate import LinearNDInterpolator
        # Seed with the four extreme (corner-most) samples.
        seeds = []
        for tx, ty in ((xs.min(), ys.min()), (xs.min(), ys.max()),
                       (xs.max(), ys.min()), (xs.max(), ys.max())):
            seeds.append(int(np.argmin((xs - tx) ** 2 + (ys - ty) ** 2)))
        chosen = list(dict.fromkeys(seeds))
        cap = min(max_points, n)
        while len(chosen) < cap and len(chosen) >= 3:
            interp = LinearNDInterpolator(pts[chosen], zs[chosen])
            zi = interp(pts)
            err = np.abs(zs - zi)
            err[~np.isfinite(err)] = 0.0       # outside current hull
            err[chosen] = 0.0
            w = int(np.argmax(err))
            if err[w] < tol:
                break
            chosen.append(w)
        if len(chosen) < 3:
            return xs, ys, zs, Delaunay(pts).simplices
        return xs[chosen], ys[chosen], zs[chosen], Delaunay(pts[chosen]).simplices
    except Exception:
        try:
            from scipy.spatial import Delaunay
            return xs, ys, zs, Delaunay(pts).simplices
        except Exception:
            return xs, ys, zs, None


def _smooth_terrain_surface(elev, gx, gy, upsample=4, smoothing=0.0):
    """Interpolate a DEM pixel grid into a smooth fine surface.

    Each pixel value is treated as a sample located at the pixel centroid
    ``(gx[i], gy[j])`` and a **thin-plate-spline RBF** is fitted through them.
    A TPS minimises bending energy, so it produces a smooth, generalistic
    surface without the grid-aligned planes of linear interpolation or the
    oscillation waves of tensor-product cubic splines. For large grids the
    fit is done locally (k nearest centroids per query) to stay fast.

    Returns ``(gx_fine, gy_fine, elev_fine)``. Falls back to a cubic
    ``RectBivariateSpline`` and finally the raw grid if SciPy is unavailable.
    """
    ny, nx = elev.shape
    # Cap the fine resolution so the RBF query count (and the browser mesh)
    # stay responsive on large grids while remaining visually smooth.
    _MAX_FINE = 140
    nxf = min(max(nx * upsample, 2), _MAX_FINE)
    nyf = min(max(ny * upsample, 2), _MAX_FINE)
    gx_fine = np.linspace(gx[0], gx[-1], nxf)
    gy_fine = np.linspace(gy[0], gy[-1], nyf)

    # Preferred: thin-plate-spline radial basis function over pixel centroids.
    try:
        from scipy.interpolate import RBFInterpolator
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([GX.ravel(), GY.ravel()])
        vals = elev.ravel()
        finite = np.isfinite(vals)
        pts, vals = pts[finite], vals[finite]
        n = pts.shape[0]
        if n >= 4:
            neighbors = None if n <= 500 else min(n, 64)
            rbf = RBFInterpolator(
                pts, vals, kernel='thin_plate_spline',
                neighbors=neighbors, smoothing=smoothing)
            GXf, GYf = np.meshgrid(gx_fine, gy_fine)
            elev_fine = rbf(np.column_stack([GXf.ravel(), GYf.ravel()])
                            ).reshape(GXf.shape)
            return gx_fine, gy_fine, elev_fine
    except Exception:
        pass

    # Fallback: cubic tensor-product spline.
    try:
        from scipy.interpolate import RectBivariateSpline
        kx = min(3, nx - 1); ky = min(3, ny - 1)
        if kx >= 1 and ky >= 1:
            sp = RectBivariateSpline(gy, gx, elev, kx=ky, ky=kx)
            return gx_fine, gy_fine, sp(gy_fine, gx_fine)
    except Exception:
        pass

    return gx, gy, elev


def _render_dem_panel(ax, terrain_grid, building_footprint_general, centroid_xy,
                      interp='tin'):
    """Render the DEM map on `ax`, the building footprint overlaid (recentred).

    `interp` selects the surface model:
      - ``'tin'``      : triangulate the DEM as a point cloud (default) —
                         every sample a vertex, faithful planar facets.
      - ``'adaptive'`` : decimated irregular network (vertices where the
                         terrain bends; lighter).
      - ``'tps'``      : smooth thin-plate-spline through the pixel centroids.
    (``'linear'`` is accepted as an alias of ``'tin'``.)
    Returns True if anything was drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
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

        if interp in ('tin', 'linear', 'adaptive'):
            # Treat the DEM as a point cloud and Delaunay-triangulate it.
            tx, ty, tz, tris = _build_terrain_tin(
                elev, gx, gy, adaptive=(interp == 'adaptive'))
            mtri = (Triangulation(tx, ty, tris) if tris is not None
                    else Triangulation(tx, ty))
            title_mode = (f'adaptive TIN ({len(tx)} pts)'
                          if interp == 'adaptive' else f'point-cloud TIN ({len(tx)} pts)')
            pm = ax.tripcolor(mtri, tz, cmap='terrain',
                              shading='gouraud', alpha=0.88)
            ax.tricontour(mtri, tz, levels=10,
                          colors='saddlebrown', linewidths=0.4, alpha=0.6)
        else:
            # Smooth thin-plate-spline surface through the pixel centroids
            # (no planes, no waves).
            grid_x_fine, grid_y_fine, elev_fine = _smooth_terrain_surface(
                elev, gx, gy, upsample=5)
            pm = ax.pcolormesh(grid_x_fine, grid_y_fine, elev_fine,
                               cmap='terrain', shading='gouraud', alpha=0.88)
            ax.contour(grid_x_fine, grid_y_fine, elev_fine, levels=10,
                       colors='saddlebrown', linewidths=0.4, alpha=0.6)
            title_mode = 'thin-plate spline ×5'
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
            f'DEM  ({cell:.0f} m grid, ±{nx * cell / 2:.0f} m, {title_mode})'
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
    extent_r = SHADOW_PLOT_EXTENT_R

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

    # Building use drives the taller-ground-floor rule for industrial/agricultural
    # buildings (see ``effective_ground_floor_height``).
    if 'building_use' in building_row.index:
        shadow_data['building_use'] = building_row['building_use']

    return shadow_data if shadow_data else None


# ==========================================
# NEIGHBORING MUNICIPALITIES
# ==========================================
_PREPARED_CENSUS_CACHE = {}


def _load_prepared_census(census_file):
    """Load ``validated_census`` once and cache the prepared frame.

    Returns a GeoDataFrame with ``CUMUN`` as ``str`` and geometry in a metric
    CRS (EPSG:3857 when the source is geographic), or ``None`` if the schema is
    invalid. The cache is keyed by (path, mtime) so an updated file is picked up.
    """
    try:
        key = (census_file, os.path.getmtime(census_file))
    except OSError:
        key = (census_file, None)
    cached = _PREPARED_CENSUS_CACHE.get(key)
    if cached is not None:
        return cached

    census_gdf = gpd.read_file(census_file)
    if census_gdf.empty:
        _PREPARED_CENSUS_CACHE[key] = census_gdf
        return census_gdf
    if "CUMUN" not in census_gdf.columns:
        raise ValueError("CUMUN column not found in census data")
    census_gdf["CUMUN"] = census_gdf["CUMUN"].astype(str)
    if census_gdf.crs is None:
        raise ValueError("Census CRS is undefined")
    if census_gdf.crs.is_geographic:
        census_gdf = census_gdf.to_crs(3857)

    _PREPARED_CENSUS_CACHE[key] = census_gdf
    return census_gdf


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
    # 2️⃣ Load census tracts (cached, reprojected)
    # ------------------------------------------------------------
    census_file = f"{census_tracts_dir}//validated_census_{year}.gpkg"

    if not os.path.exists(census_file):
        print("[WARN] Census file not found")
        return cadaster_codes if include_original else []

    # This function is called once per municipality inside the batch loop, so
    # re-reading/re-projecting the (national) census file every call dominates
    # the loop's fixed overhead. Cache the prepared frame — CUMUN normalised to
    # str and reprojected to a metric CRS — keyed by path. Callers below only
    # filter/union/intersect it, never mutate, so sharing the object is safe.
    census_gdf = _load_prepared_census(census_file)

    if census_gdf is None or census_gdf.empty:
        print("[WARN] Census GeoDataFrame is empty")
        return cadaster_codes if include_original else []

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

    # Build the (possibly nested) dictionary. ``grouped_result`` is already
    # sorted by the grouping columns (pandas groupby sorts by default), so a
    # single insertion pass reproduces the same sorted key order the old
    # recursive groupby-per-level builder produced — but without the
    # per-group DataFrame.drop() that dominated the runtime on large inputs.
    def build_nested_dict(df, group_cols, value_col):
        if len(group_cols) == 1:
            return dict(zip(df[group_cols[0]], df[value_col]))
        # Index via numpy arrays (not .tolist()) so key/value scalar types match
        # the old recursive builder exactly (e.g. np.int32 floor keys).
        key_cols = [df[c].to_numpy() for c in group_cols]
        values = df[value_col].to_numpy()
        root = {}
        last = len(group_cols) - 1
        for i in range(len(values)):
            node = root
            for lvl in range(last):
                node = node.setdefault(key_cols[lvl][i], {})
            node[key_cols[last][i]] = values[i]
        return root

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
    """Read the building-reference inference table (``br_results``) from the DB.

    Excludes the heavy BLOB columns (fetch those with
    :func:`get_heavy_columns_for_building`).

    Parameters
    ----------
    cadaster_code : str
        Municipality cadastral code.
    wd : str
        Working directory passed to ``hc.merge``.

    Returns
    -------
    pandas.DataFrame
        One row per building with the ``br__*`` indicators.
    """
    return _read_inferred_indicators_from_db(cadaster_code, wd, "br_results")


def read_sbr_inferred_indicators(cadaster_code, wd):
    """Read the space/single-building inference table (``sbr_results``) from the DB.

    Excludes the heavy BLOB columns (fetch those with
    :func:`get_heavy_columns_for_building`, ``table="sbr_results"``).

    Parameters
    ----------
    cadaster_code : str
        Municipality cadastral code.
    wd : str
        Working directory passed to ``hc.merge``.

    Returns
    -------
    pandas.DataFrame
        One row per single-building-reference with the ``sbr__*`` indicators.
    """
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


def clear_cat_cache(wd, cadaster_code=None):
    """Invalidate the cached CAT data so the next ``merge`` recomputes it.

    Clears both central-DuckDB CAT caches: the parsed per-space table
    (``cat_building_spaces``) and the per-building ``br__*`` aggregates
    (``cat_building_attributes``). Both are pure functions of the static CAT
    files, so they are reused on every run regardless of ``force_update`` — call
    this after replacing the CAT files to force a re-parse. Pass
    ``cadaster_code`` to clear a single municipality, or leave it ``None`` to
    clear all. Returns the total number of cached rows removed.
    """
    from hypercadaster_ES import _db
    return _db.cat_cache_clear(results_dir_(wd), cadaster_code)


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
    """Plot an ordered sequence of ``(x, y)`` points, labelling each by index,
    and save the figure to ``pdf_file`` (debugging aid)."""
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
    """Plot a group of polygons plus one highlighted polygon and save to PDF.

    ``poly_list`` is drawn in ``color1`` and the reference ``poly2`` in
    ``color2`` (debugging/visualisation aid). All inputs must be shapely
    ``Polygon``s.
    """
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
    """Trim a ``LineString`` by ``percent`` of its length at each end.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Line to trim (must have ≥2 points).
    percent : float
        Fraction (0–0.5) of the total length to remove from each end.

    Returns
    -------
    shapely.geometry.LineString
        The trimmed line.
    """
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
    """Plot a mix of LineStrings and (optional) polygons on one axis.

    Debugging/visualisation aid; saves to ``pdf_file`` when given, otherwise
    shows the figure.
    """
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
    address_col="postal_addresses_location",
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
    'interior':  '#884EA0',   # slab between two stacked zones (purple)
    'adiabatic': '#884EA0',   # legacy alias for 'interior'
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


def open_patio_voids_by_floor(footprint_by_floor, patio_voids,
                              coverage_threshold=0.5):
    """Per floor, the courtyard voids that are still OPEN (not yet roofed over).

    A patio is an interior courtyard open to the sky. Per-floor geometry
    analysis can miss it on an upper floor whose footprint failed to close the
    interior ring, so this works from the building-wide set of detected
    courtyards (`patio_voids`) instead of any single floor. A courtyard is
    "open" at a floor while that floor's footprint does not cover it, and
    "capped" (the first roof) at the first floor that builds over it — above
    which it is no longer a patio.

    Parameters
    ----------
    footprint_by_floor : dict[int, shapely geometry]
        Union of the building's footprint per floor index.
    patio_voids : iterable[shapely Polygon]
        All courtyard void polygons detected anywhere in the building.
    coverage_threshold : float, default 0.5
        A courtyard counts as covered (roofed) at a floor once this fraction of
        its area is overlapped by that floor's footprint.

    Returns
    -------
    dict[int, list[shapely Polygon]]
        floor index -> list of courtyard voids open at that floor. Floors with
        no open courtyard are omitted.
    """
    voids = [p for p in (patio_voids or [])
             if p is not None and not getattr(p, 'is_empty', True)
             and getattr(p, 'area', 0.0) > 0.0]
    if not voids:
        return {}
    out = {}
    for fl, fp in footprint_by_floor.items():
        if fp is None or getattr(fp, 'is_empty', True):
            continue
        open_here = []
        for v in voids:
            try:
                covered = v.intersection(fp).area
            except Exception:
                covered = 0.0
            if covered < coverage_threshold * v.area:
                open_here.append(v)
        if open_here:
            out[fl] = open_here
    return out


def _split_polygon_by_area_fraction(poly, frac):
    """Cut `poly` into two pieces (a, b) where `a` carries ~`frac` of the area,
    by a straight line perpendicular to the longest bounding-box axis (binary
    search on the cut position). Returns (a, b) or (None, None) on failure."""
    from shapely.geometry import box
    try:
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if w <= 0 or h <= 0:
            return None, None
        target = poly.area * frac
        if w >= h:
            lo, hi = minx, maxx
            for _ in range(30):
                c = (lo + hi) / 2.0
                if poly.intersection(box(minx, miny, c, maxy)).area < target:
                    lo = c
                else:
                    hi = c
            c = (lo + hi) / 2.0
            return (poly.intersection(box(minx, miny, c, maxy)),
                    poly.intersection(box(c, miny, maxx, maxy)))
        lo, hi = miny, maxy
        for _ in range(30):
            c = (lo + hi) / 2.0
            if poly.intersection(box(minx, miny, maxx, c)).area < target:
                lo = c
            else:
                hi = c
        c = (lo + hi) / 2.0
        return (poly.intersection(box(minx, miny, maxx, c)),
                poly.intersection(box(minx, c, maxx, maxy)))
    except Exception:
        return None, None


def _subdivide_polygon_weighted(poly, weights):
    """Partition `poly` into ``len(weights)`` sub-polygons whose areas are
    proportional to `weights`, by recursive area-balanced bisection. Returns a
    list aligned with `weights` (entries may be None if a split fails)."""
    n = len(weights)
    if n == 0:
        return []
    if n == 1 or poly is None or getattr(poly, 'is_empty', True):
        return [poly]
    total = float(sum(weights)) or 1.0
    target = total / 2.0
    cum, m = 0.0, 1
    for idx in range(n):
        cum += weights[idx]
        if cum >= target:
            m = idx + 1
            break
    m = max(1, min(n - 1, m))
    frac = sum(weights[:m]) / total
    a, b = _split_polygon_by_area_fraction(poly, frac)
    if a is None:
        return [poly] + [None] * (n - 1)
    return (_subdivide_polygon_weighted(a, weights[:m])
            + _subdivide_polygon_weighted(b, weights[m:]))


def _principal_axis_angle(poly):
    """Angle (degrees) of the building's principal axis — the long edge of its
    minimum rotated rectangle. Used to orient zone partition cuts along the
    building's own directions rather than world X/Y."""
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        best_len, best_ang = -1.0, 0.0
        for i in range(len(coords) - 1):
            x0, y0 = coords[i][:2]
            x1, y1 = coords[i + 1][:2]
            length = math.hypot(x1 - x0, y1 - y0)
            if length > best_len:
                best_len, best_ang = length, math.degrees(math.atan2(y1 - y0, x1 - x0))
        return best_ang
    except Exception:
        return 0.0


def _subdivide_polygon_weighted_oriented(poly, weights, angle=0.0, origin=(0.0, 0.0)):
    """Like `_subdivide_polygon_weighted` but with the bisection cuts aligned to
    the building's principal axes: the polygon is rotated by ``-angle`` about
    ``origin`` (so the principal axis lands on X/Y), subdivided, and the pieces
    are rotated back."""
    if abs(angle) < 1e-9:
        return _subdivide_polygon_weighted(poly, weights)
    from shapely.affinity import rotate as _rot
    rp = _rot(poly, -angle, origin=origin)
    pieces = _subdivide_polygon_weighted(rp, weights)
    out = []
    for p in pieces:
        if p is None or getattr(p, 'is_empty', True):
            out.append(p)
        else:
            out.append(_rot(p, angle, origin=origin))
    return out


def _subdivide_ring_angular(ring, center, weights, start_angle=0.0):
    """Partition an annular ``ring`` (a region surrounding ``center``) into
    contiguous angular WEDGES whose areas are proportional to ``weights``, by
    sweeping the polar angle around ``center``. Returns a list aligned with
    ``weights``.

    Used to tile the floor area around a central communal core: unlike straight
    recursive bisection — which cuts an annulus into pieces that become
    DISCONNECTED across the central hole (their minor lobe then gets dropped) —
    each angular wedge stays connected for a ring that is star-shaped about
    ``center`` (the typical building floor around a central stair/lift core).
    Sweep starts at ``start_angle`` (radians, CCW from +x) so the ordering can
    be aligned to the building's main axis.
    """
    n = len(weights)
    if n == 0:
        return []
    if n == 1 or ring is None or getattr(ring, "is_empty", True):
        return [ring] + [None] * (n - 1)
    total_area = ring.area
    if total_area <= 0:
        return [ring] + [None] * (n - 1)
    cx, cy = center
    minx, miny, maxx, maxy = ring.bounds
    R = 2.0 * max(maxx - cx, cx - minx, maxy - cy, cy - miny, 1.0) + 10.0
    total_w = float(sum(weights)) or 1.0

    def _sector(a0, a1):
        """Fan polygon from ``center`` covering angles [a0, a1] (radius R)."""
        steps = max(2, int(math.degrees(a1 - a0) / 5.0) + 1)
        pts = [(cx, cy)]
        for k in range(steps + 1):
            a = a0 + (a1 - a0) * k / steps
            pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
        return Polygon(pts)

    out = []
    cursor = start_angle
    for i in range(n):
        if i == n - 1:
            a_end = start_angle + 2.0 * math.pi
        else:
            target = total_area * (sum(weights[: i + 1]) / total_w)
            lo, hi = cursor, start_angle + 2.0 * math.pi
            for _ in range(30):
                mid = (lo + hi) / 2.0
                try:
                    a = ring.intersection(_sector(start_angle, mid)).area
                except Exception:
                    a = 0.0
                if a < target:
                    lo = mid
                else:
                    hi = mid
            a_end = (lo + hi) / 2.0
        try:
            piece = ring.intersection(_sector(cursor, a_end))
        except Exception:
            piece = None
        out.append(piece)
        cursor = a_end
    return out


def _carve_central_rect(poly, target_area, angle=0.0, origin=(0.0, 0.0)):
    """Carve a CENTRAL, axis-aligned (in the frame rotated by ``-angle``)
    RECTANGLE of area ≈ ``target_area`` from ``poly``, centred on ``poly``'s
    centroid and matching the footprint's aspect ratio. Returns (core,
    remainder). Unlike ``_carve_central_eroded`` (which yields a ROUNDED core and
    hence rounded-corner slivers in the surrounding ring), this gives a clean
    rectangular core so the perimeter bands of ``_subdivide_ring_rectangular``
    tile the ring exactly. Size is binary-searched so the clipped area matches."""
    from shapely.affinity import rotate as _rot
    from shapely.geometry import box as _box
    if poly is None or getattr(poly, "is_empty", True) or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    rp = _rot(poly, -angle, origin=origin)
    c = rp.centroid
    cx, cy = c.x, c.y
    minx, miny, maxx, maxy = rp.bounds
    W, H = maxx - minx, maxy - miny
    aspect = (W / H) if H > 1e-9 else 1.0
    rh0 = (target_area / aspect) ** 0.5 if aspect > 0 else target_area ** 0.5
    rw0 = aspect * rh0
    lo, hi = 0.0, 2.0
    for _ in range(30):
        s = (lo + hi) / 2.0
        rect = _box(cx - s * rw0 / 2, cy - s * rh0 / 2,
                    cx + s * rw0 / 2, cy + s * rh0 / 2)
        try:
            a = rp.intersection(rect).area
        except Exception:
            a = 0.0
        if a < target_area:
            lo = s
        else:
            hi = s
    s = (lo + hi) / 2.0
    rect = _box(cx - s * rw0 / 2, cy - s * rh0 / 2,
                cx + s * rw0 / 2, cy + s * rh0 / 2)
    try:
        core_r = rp.intersection(rect)
        rem_r = rp.difference(rect)
    except Exception:
        return None, poly
    core = _rot(core_r, angle, origin=origin)
    rem = _rot(rem_r, angle, origin=origin)
    return core, rem


def _subdivide_ring_rectangular(ring, core, weights, angle=0.0, origin=(0.0, 0.0)):
    """Partition an annular ``ring`` (floor minus a central ``core``) into
    RECTANGULAR blocks whose areas are proportional to ``weights``.

    Robust against the annular-cut fragility: rather than cutting straight across
    the ring (which splits a zone into disconnected lobes across the central
    hole), the ring is first divided into FOUR simply-connected perimeter bands
    around the core's bounding box (left / top / right / bottom, in the frame
    rotated by ``-angle`` so the core is axis-aligned). Each band is a hole-free
    rectangle-like region, so guillotine subdivision (``_subdivide_polygon_
    weighted``) of it yields rectangular rooms with no disconnected pieces. Bands
    tile the ring exactly; a band with no assigned zone is merged into the next
    (cyclically) so no area is left ungapped. Returns a list aligned to
    ``weights``.
    """
    from shapely.affinity import rotate as _rot
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _uu
    n = len(weights)
    if n == 0:
        return []
    if ring is None or getattr(ring, "is_empty", True):
        return [None] * n
    if n == 1:
        return [ring]
    rp = _rot(ring, -angle, origin=origin)
    rc = _rot(core, -angle, origin=origin)
    cxmin, cymin, cxmax, cymax = rc.bounds
    fxmin, fymin, fxmax, fymax = rp.bounds
    cymid = (cymin + cymax) / 2.0
    # Four perimeter bands that EXACTLY tile the ring (union = ring): left/right
    # full-height columns beside the core, top/bottom caps split at the core's
    # mid-height. Each is `ring ∩ box` → hole-free.
    try:
        bands = [
            rp.intersection(_box(fxmin, fymin, cxmin, fymax)),   # left
            rp.intersection(_box(cxmin, cymid, cxmax, fymax)),   # top
            rp.intersection(_box(cxmax, fymin, fxmax, fymax)),   # right
            rp.intersection(_box(cxmin, fymin, cxmax, cymid)),   # bottom
        ]
    except Exception:
        return [None] * n
    band_area = [b.area if (b is not None and not b.is_empty) else 0.0 for b in bands]
    tot_band = sum(band_area) or 1.0
    tot_w = float(sum(weights)) or 1.0
    # Cumulative band-area fraction boundaries; assign each zone (in order) to the
    # band containing its cumulative-weight midpoint.
    cum, acc = [], 0.0
    for a in band_area:
        acc += a / tot_band
        cum.append(acc)
    groups = [[] for _ in bands]
    cw = 0.0
    for i, w in enumerate(weights):
        mid = (cw + w / 2.0) / tot_w
        cw += w
        b = 0
        while b < len(cum) - 1 and mid > cum[b]:
            b += 1
        groups[b].append((i, w))
    # Merge any band that received no zone into the next band that did (cyclic),
    # so its area is not left as a gap.
    band_polys = list(bands)
    for b in range(len(band_polys)):
        if not groups[b] and band_polys[b] is not None and not band_polys[b].is_empty:
            for d in range(1, len(band_polys)):
                t = (b + d) % len(band_polys)
                if groups[t]:
                    try:
                        band_polys[t] = _uu([band_polys[t], band_polys[b]])
                    except Exception:
                        pass
                    break
    result = [None] * n
    for band, grp in zip(band_polys, groups):
        if not grp or band is None or getattr(band, "is_empty", True):
            continue
        gw = [w for _i, w in grp]
        pieces = _subdivide_polygon_weighted(band, gw)
        for (idx, _w), piece in zip(grp, pieces):
            if piece is not None and not getattr(piece, "is_empty", True):
                result[idx] = _rot(piece, angle, origin=origin)
    return result


def _split_vertical(poly, frac):
    """Split `poly` with a single vertical line so the LEFT piece carries ~`frac`
    of the area. Returns (left, right) or (None, None)."""
    from shapely.geometry import box
    try:
        minx, miny, maxx, maxy = poly.bounds
        if maxx - minx <= 0:
            return None, None
        target = poly.area * frac
        lo, hi = minx, maxx
        for _ in range(30):
            c = (lo + hi) / 2.0
            if poly.intersection(box(minx, miny, c, maxy)).area < target:
                lo = c
            else:
                hi = c
        c = (lo + hi) / 2.0
        return (poly.intersection(box(minx, miny, c, maxy)),
                poly.intersection(box(c, miny, maxx, maxy)))
    except Exception:
        return None, None


def _subdivide_vertical(poly, weights):
    """Subdivide `poly` into pieces with areas ∝ `weights`, always cutting
    vertically, returned LEFT→RIGHT (so weights[0] is the leftmost piece)."""
    n = len(weights)
    if n == 0:
        return []
    if n == 1 or poly is None or getattr(poly, 'is_empty', True):
        return [poly]
    total = float(sum(weights)) or 1.0
    target = total / 2.0
    cum, m = 0.0, 1
    for idx in range(n):
        cum += weights[idx]
        if cum >= target:
            m = idx + 1
            break
    m = max(1, min(n - 1, m))
    a, b = _split_vertical(poly, sum(weights[:m]) / total)
    if a is None:
        return [poly] + [None] * (n - 1)
    return _subdivide_vertical(a, weights[:m]) + _subdivide_vertical(b, weights[m:])


def _subdivide_polygon_weighted_directional(poly, weights, stack_angle, origin=(0.0, 0.0)):
    """Subdivide `poly` into pieces with areas ∝ `weights`, **stacked along the
    `stack_angle` direction** (degrees, math convention: 0°=E, 90°=N). Cuts are
    perpendicular to that direction; pieces are returned ordered from the
    negative end of the direction to the positive end — so with
    ``stack_angle`` pointing North, ``weights[0]`` lands on the SOUTH side."""
    from shapely.affinity import rotate as _rot
    n = len(weights)
    if n <= 1 or poly is None or getattr(poly, 'is_empty', True):
        return [poly]
    # Rotate so the stacking direction maps to +X; cut vertically; rotate back.
    rp = _rot(poly, -stack_angle, origin=origin)
    pieces = _subdivide_vertical(rp, weights)
    return [(_rot(p, stack_angle, origin=origin)
             if p is not None and not getattr(p, 'is_empty', True) else p)
            for p in pieces]


def _stack_angle_towards_north(principal_angle):
    """Of the building's two principal axes (`principal_angle`, +90°), return
    the one most aligned with North–South, oriented to point North (+Y). Used
    to stack thermal zones from south (high priority) to north."""
    best_a, best_v = principal_angle, -1.0
    for a in (principal_angle, principal_angle + 90.0):
        v = abs(math.sin(math.radians(a)))   # how N–S the axis is
        if v > best_v:
            best_v, best_a = v, a
    a = best_a % 360.0
    if math.sin(math.radians(a)) < 0:         # flip to point North
        a = (a + 180.0) % 360.0
    return a


# Thermal-zone south-placement priority (lower rank → more southern). The rest
# (people living/working) sit between tertiary and the parking/storage group.
_ZONE_USE_PRIORITY = (
    (('residential', 'housing', 'dwelling'), 0),
    (('commercial', 'retail', 'shop'), 1),
    (('office', 'tertiary'), 2),
    (('parking', 'warehouse', 'storage', 'garage'), 9),
)


def _zone_south_priority(zone):
    """Rank a thermal zone for south placement (0 = most southern). Communal
    cores are interior and not ranked here."""
    txt = ' '.join(str(zone.get(k, '')) for k in ('use_type',)).lower()
    det = zone.get('detailed_use_types') or []
    txt += ' ' + ' '.join(str(d) for d in det).lower()
    for keys, rank in _ZONE_USE_PRIORITY:
        if any(k in txt for k in keys):
            return rank
    return 5   # the rest (people living/working) — before parking/storage


def _carve_central_eroded(poly, target_area):
    """Carve the **interior** of `poly` whose area ≈ `target_area` by inward
    erosion (negative buffer). Returns (core_region, remainder).

    The core is the floor footprint shrunk inward until it reaches the target
    (communal) area, so it always sits in the inner part of the building and
    follows the building's own shape, with the occupied zones forming the
    surrounding ring (``poly`` minus the core). The erosion distance is found
    by binary search; area decreases monotonically with erosion, so ``lo``
    always holds area > target and ``hi`` area < target.
    """
    if poly is None or poly.is_empty or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    minx, miny, maxx, maxy = poly.bounds
    dmax = max(maxx - minx, maxy - miny) / 2.0   # enough to fully erode
    lo, hi = 0.0, dmax
    for _ in range(30):
        d = (lo + hi) / 2.0
        try:
            a = poly.buffer(-d).area
        except Exception:
            a = 0.0
        if a > target_area:
            lo = d        # still too big → erode more
        else:
            hi = d        # too small → erode less
    d = (lo + hi) / 2.0
    try:
        core = poly.buffer(-d)
        if core.is_empty:
            core = poly.buffer(-lo)
    except Exception:
        return None, poly
    try:
        remainder = poly.difference(core)
    except Exception:
        remainder = None
    return core, remainder


def _carve_square_block(poly, target_area, direction=None):
    """Carve a compact, ~square block of area ``target_area`` out of ``poly``,
    hugging the side indicated by ``direction`` (a unit (dx, dy), e.g. toward a
    street). Returns (block, remainder).

    Used for non-occupied zones (parking/warehouse) that read better as a
    square block against the street-facing edge than as a thin strip. The
    square side is grown by binary search until its clipped area matches; if it
    can't stay square (footprint too narrow on that side) it degrades to a
    strip of the available width, which is still compact.
    """
    from shapely.geometry import box
    if poly is None or getattr(poly, 'is_empty', True) or target_area <= 0:
        return None, poly
    if target_area >= poly.area:
        return poly, None
    minx, miny, maxx, maxy = poly.bounds
    try:
        c = poly.centroid
        cx, cy = c.x, c.y
    except Exception:
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0

    side = 'NE'
    if direction is not None:
        dx, dy = direction
        if abs(dx) >= abs(dy):
            side = 'E' if dx >= 0 else 'W'
        else:
            side = 'N' if dy >= 0 else 'S'

    def _sq(s):
        if side == 'E':
            return box(maxx - s, cy - s / 2, maxx, cy + s / 2)
        if side == 'W':
            return box(minx, cy - s / 2, minx + s, cy + s / 2)
        if side == 'N':
            return box(cx - s / 2, maxy - s, cx + s / 2, maxy)
        if side == 'S':
            return box(cx - s / 2, miny, cx + s / 2, miny + s)
        return box(maxx - s, maxy - s, maxx, maxy)  # NE corner default

    diag = math.hypot(maxx - minx, maxy - miny)
    lo, hi = 0.0, diag
    for _ in range(30):
        s = (lo + hi) / 2.0
        try:
            a = poly.intersection(_sq(s)).area
        except Exception:
            a = 0.0
        if a < target_area:
            lo = s
        else:
            hi = s
    sq = _sq((lo + hi) / 2.0)
    try:
        block = poly.intersection(sq)
        remainder = poly.difference(sq)
    except Exception:
        return None, poly
    return block, remainder


def _thermal_zone_render_kind(z):
    """Classify a thermal-zone dict for rendering: 'communal', 'non_occupied'
    or 'occupied'. Trusts an explicit ``kind`` when present, else falls back to
    use-type/detailed heuristics so older data (without the kind) still renders
    parking/warehouse and pools as non-occupied."""
    if not isinstance(z, dict):
        return 'occupied'
    k = z.get('kind')
    if k in ('communal', 'non_occupied', 'occupied'):
        return k
    ut = str(z.get('use_type', '')).lower()
    if any(t in ut for t in ('parking', 'warehouse', 'storage', 'garage', 'agricultur')):
        return 'non_occupied'
    det = ' '.join(str(d) for d in (z.get('detailed_use_types') or [])).lower()
    if 'pool' in det or z.get('source') == 'other_construction':
        return 'non_occupied'
    return 'occupied'


def _zone_legend_category(z):
    """Coarse category for grouping thermal zones in the 3-D legend / toggle menu
    (one entry per category instead of one per zone). Occupied zones group by use
    type; the rest by role."""
    kind = _thermal_zone_render_kind(z)
    if kind == 'communal':
        return 'Communal core'
    if kind == 'non_occupied':
        if z.get('is_pool'):
            return 'Pool'
        ut = str(z.get('use_type', '')).lower()
        if any(t in ut for t in ('parking', 'warehouse', 'garage', 'storage')):
            return 'Parking / warehouse'
        return 'Non-occupied'
    return str(z.get('use_type') or 'Occupied')


def _append_prism_mesh(bucket, region, z_bot, z_top, ox, oy):
    """Append a closed extruded prism of `region` (between z_bot and z_top) to
    a Mesh3d vertex/face `bucket` (dict of x/y/z/i/j/k lists), offset by ox/oy.
    Builds top + bottom faces (triangulated) and vertical side walls."""
    if region is None or getattr(region, 'is_empty', True):
        return
    verts, tris = _triangulate_polygon_xy(region)
    if verts and tris:
        for zlevel in (z_bot, z_top):
            base = len(bucket['x'])
            for (vx, vy) in verts:
                bucket['x'].append(vx - ox); bucket['y'].append(vy - oy)
                bucket['z'].append(zlevel)
            for (a, b, c) in tris:
                bucket['i'].append(base + a)
                bucket['j'].append(base + b)
                bucket['k'].append(base + c)
    polys = list(region.geoms) if hasattr(region, 'geoms') else [region]
    for p in polys:
        if not hasattr(p, 'exterior'):
            continue
        for ring in [p.exterior] + list(p.interiors):
            cc = list(ring.coords)
            for ci in range(len(cc) - 1):
                x0, y0 = cc[ci][:2]; x1, y1 = cc[ci + 1][:2]
                rx0, ry0 = x0 - ox, y0 - oy
                rx1, ry1 = x1 - ox, y1 - oy
                b0 = len(bucket['x'])
                bucket['x'].extend([rx0, rx1, rx1, rx0])
                bucket['y'].extend([ry0, ry1, ry1, ry0])
                bucket['z'].extend([z_bot, z_bot, z_top, z_top])
                bucket['i'].extend([b0,     b0])
                bucket['j'].extend([b0 + 1, b0 + 2])
                bucket['k'].extend([b0 + 2, b0 + 3])


_THERMAL_ZONE_PALETTE = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#17BECF', '#BCBD22', '#E377C2', '#8C564B', '#7F7F7F',
]
_THERMAL_ZONE_COMMUNAL_COLOUR = '#8B5A2B'
_THERMAL_ZONE_NON_OCCUPIED_COLOUR = '#7F7F7F'  # parking / warehouse / pools


def plot_building_dem(gdf, building_reference, output_path=None, wd=None, dpi=150,
                      terrain_interp='tin'):
    """Standalone DEM contour map for a single building.

    ``terrain_interp``: ``'tin'`` (default — triangulate the DEM as a point
    cloud), ``'adaptive'`` (decimated irregular network), or ``'tps'`` (smooth
    thin-plate spline).

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
    drawn = _render_dem_panel(ax, terrain_grid, bfg, (cx, cy), interp=terrain_interp)
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


def _zone_use_colour(use_type, mapping):
    """Stable colour for an occupied zone's use type — assigns the next palette
    colour the first time a use type is seen, so the legend stays consistent."""
    key = str(use_type or "Other").strip() or "Other"
    if key not in mapping:
        mapping[key] = _THERMAL_ZONE_PALETTE[len(mapping) % len(_THERMAL_ZONE_PALETTE)]
    return mapping[key]


def _thermal_zone_label(zone, building_reference):
    """Short label identifying the BUILDING SPACE behind a thermal zone. Every
    space of a parcel shares the 14-char cadastral parcel prefix (``3444707BG9034S``)
    — useless to display — so show the DISTINGUISHING building-space suffix
    (``0003IL``); the communal core is labelled plainly."""
    if zone.get('kind') == 'communal' or zone.get('zone_id') == 'communal_core':
        return 'communal'
    ref = str(zone.get('reference') or zone.get('zone_id') or '').split('|')[-1]
    if not ref:
        return ''
    parcel = str(building_reference or '')
    if parcel and ref.startswith(parcel) and len(ref) > len(parcel):
        return ref[len(parcel):]           # the space discriminator (e.g. 0003IL)
    return ref[-6:] if len(ref) > 6 else ref


def _add_polygon_patch(ax, poly, ox, oy, facecolor, edgecolor, alpha=0.75,
                       hatch=None, lw=1.0, zorder=2):
    """Fill a shapely (Multi)Polygon on ``ax`` (offset by ox/oy), honouring holes
    via a compound matplotlib Path."""
    from matplotlib.path import Path as _Path
    from matplotlib.patches import PathPatch as _PathPatch
    if poly is None or getattr(poly, "is_empty", True):
        return
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    for p in polys:
        if not hasattr(p, "exterior"):
            continue
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            cc = list(ring.coords)
            if len(cc) < 3:
                continue
            verts.extend([(x - ox, y - oy) for x, y, *_ in cc])
            codes.extend([_Path.MOVETO] + [_Path.LINETO] * (len(cc) - 2) + [_Path.CLOSEPOLY])
        if not verts:
            continue
        patch = _PathPatch(_Path(verts, codes), facecolor=facecolor,
                           edgecolor=edgecolor, alpha=alpha, hatch=hatch,
                           linewidth=lw, zorder=zorder)
        ax.add_patch(patch)


def plot_building_thermal_zones(gdf, building_reference, output_path=None, wd=None,
                                dpi=150):
    """2-D per-floor plan of the estimated thermal-zone distribution.

    One subplot per floor; every thermal zone (dwelling / premise / parking /
    communal core) is drawn as a filled polygon from the on-demand geometry of
    ``energy.synthesize_space_geometry`` — the SAME partition the 3-D view and the
    EnergyPlus model use. All zones use SOLID fills (no hatching, which would be
    confused with the zone-division lines): occupied zones are coloured by use
    type, the communal circulation core is brown, non-occupied zones (parking /
    warehouse / pools) are grey, and any Housing zone that could not secure the
    minimum main-façade frontage is outlined in red (``facade_deficit``). Each
    zone is labelled with its reference and floor area.

    Parameters
    ----------
    gdf : DataFrame / GeoDataFrame
        Output of ``hc.merge``.
    building_reference : str
    output_path : str, optional
        Save path (.png / .pdf / …). Displays interactively when None.
    wd : str, optional
        Working directory — used to auto-fetch the heavy BLOB columns
        (``br__thermal_zones``, ``br__building_footprint_by_floor``,
        ``br__wall_segments``) from the central DuckDB.
    dpi : int, default 150
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from hypercadaster_ES import energy as _energy

    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None
    row = building_data.iloc[0].copy()

    # ---- Auto-fetch heavy columns from DuckDB --------------------------------
    heavy_cols = ['br__thermal_zones', 'br__building_footprint_by_floor',
                  'br__wall_segments']
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

    # Decode any bytearray/bytes values (WKB geometry or pickled objects).
    def _robust_decode(v):
        if not isinstance(v, (bytes, bytearray, memoryview)):
            return v
        b = bytes(v)
        try:
            from shapely import wkb as _swkb
            return _swkb.loads(b, include_srid=True)
        except Exception:
            pass
        try:
            import pickle as _pkl
            return _pkl.loads(b)
        except Exception:
            pass
        return v

    for _col in list(row.index):
        if isinstance(row[_col], (bytes, bytearray, memoryview)):
            row[_col] = _robust_decode(row[_col])

    try:
        zones = _energy.synthesize_space_geometry(row)
    except Exception as _e:
        print(f"Could not synthesize zone geometry: {_e}")
        return None
    if not zones:
        print(f"No thermal-zone geometry available for building {building_reference}")
        return None

    # Footprint outline per floor (context) from the same source synthesis uses.
    fp_by_floor = _energy._footprints_by_floor(row)

    # Gather floors and shared extent across all zone polygons + footprints.
    floors_set, all_xs, all_ys = set(), [], []
    for z in zones:
        for fl, poly in (z.get('floor_polys') or {}).items():
            if poly is None or getattr(poly, 'is_empty', True):
                continue
            floors_set.add(int(fl))
            minx, miny, maxx, maxy = poly.bounds
            all_xs.extend([minx, maxx]); all_ys.extend([miny, maxy])
    for fl, poly in fp_by_floor.items():
        floors_set.add(int(fl))
        minx, miny, maxx, maxy = poly.bounds
        all_xs.extend([minx, maxx]); all_ys.extend([miny, maxy])
    if not floors_set or not all_xs:
        print(f"No thermal-zone geometry available for building {building_reference}")
        return None

    cx = (min(all_xs) + max(all_xs)) / 2
    cy = (min(all_ys) + max(all_ys)) / 2
    half = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) / 2 * 1.15 + 1.0
    ox, oy = cx, cy
    x_lim = (cx - half - ox, cx + half - ox)
    y_lim = (cy - half - oy, cy + half - oy)

    floor_keys = sorted(floors_set)
    n_floors = max(1, len(floor_keys))
    n_cols = min(4, n_floors)
    n_rows = math.ceil(n_floors / n_cols)

    fig = plt.figure(figsize=(4.5 * n_cols, 4.0 * n_rows + 1.2))
    gs = GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    use_colours = {}          # use_type -> colour (filled lazily, drives legend)
    saw_communal = saw_nonocc = saw_deficit = False

    for idx, floor in enumerate(floor_keys):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_aspect('equal')
        ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)

        fpoly = fp_by_floor.get(floor)
        if fpoly is not None:
            for g in (list(fpoly.geoms) if hasattr(fpoly, 'geoms') else [fpoly]):
                if hasattr(g, 'exterior'):
                    xs = [p[0] - ox for p in g.exterior.coords]
                    ys = [p[1] - oy for p in g.exterior.coords]
                    ax.plot(xs, ys, color='#BBBBBB', lw=0.8, ls=':', zorder=1)

        # Dense floors (many small dwellings, e.g. a row of attached houses) turn
        # per-zone labels into stacked noise — drop labels there and let the fills
        # carry the distribution.
        n_labelable = sum(1 for z in zones
                          if (z.get('floor_polys') or {}).get(floor) is not None
                          and _thermal_zone_render_kind(z) != 'non_occupied')
        label_floor = n_labelable <= 20

        for z in zones:
            poly = (z.get('floor_polys') or {}).get(floor)
            if poly is None or getattr(poly, 'is_empty', True):
                continue
            kind = _thermal_zone_render_kind(z)
            deficit = bool(z.get('facade_deficit'))
            if kind == 'communal':
                fc = _THERMAL_ZONE_COMMUNAL_COLOUR
                saw_communal = True
            elif kind == 'non_occupied':
                fc = _THERMAL_ZONE_NON_OCCUPIED_COLOUR
                saw_nonocc = True
            else:
                fc = _zone_use_colour(z.get('use_type'), use_colours)
            edge = '#D62728' if deficit else '#333333'
            lw = 2.0 if deficit else 0.8
            saw_deficit = saw_deficit or deficit
            # SOLID fills only — no hatching, which would be confused with the
            # thermal-zone division lines. Categories are told apart by colour.
            _add_polygon_patch(ax, poly, ox, oy, fc, edge, alpha=0.72, lw=lw)
            # Label reference + area at the zone centroid. Skip tiny/non-occupied
            # zones and whole dense floors so the plot doesn't turn into stacked
            # text — the fills + legend already identify them.
            if not label_floor or poly.area < 6.0 or kind == 'non_occupied':
                continue
            try:
                cpt = poly.representative_point()
                ax.annotate(f"{_thermal_zone_label(z, building_reference)}"
                            f"\n{poly.area:.0f} m²",
                            (cpt.x - ox, cpt.y - oy), ha='center', va='center',
                            fontsize=6, color='#111111', zorder=3)
            except Exception:
                pass

        ax.set_title(f'Floor {floor}', fontsize=10, pad=5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_xlabel('ΔX (m)', fontsize=7); ax.set_ylabel('ΔY (m)', fontsize=7)

    for idx in range(n_floors, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    legend_patches = [mpatches.Patch(facecolor=col, edgecolor='#333333', label=use)
                      for use, col in sorted(use_colours.items())]
    if saw_communal:
        legend_patches.append(mpatches.Patch(
            facecolor=_THERMAL_ZONE_COMMUNAL_COLOUR, edgecolor='#333333',
            label='Communal core'))
    if saw_nonocc:
        legend_patches.append(mpatches.Patch(
            facecolor=_THERMAL_ZONE_NON_OCCUPIED_COLOUR, edgecolor='#333333',
            label='Non-occupied (parking/pool)'))
    if saw_deficit:
        legend_patches.append(mpatches.Patch(
            facecolor='none', edgecolor='#D62728', lw=2.0,
            label='Façade frontage < 3 m'))
    if legend_patches:
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=min(5, len(legend_patches)), fontsize=8,
                   bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.suptitle(f'Thermal-zone floor plan — {building_reference}', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Thermal-zone floor plan saved to {output_path}")
    else:
        plt.show()
    return fig


def plot_building_3d_geometry(gdf, building_reference, output_path=None, wd=None,
                               floor_height=3.0, ground_floor_height=5.0,
                               show_thermal_zones=True, terrain_interp='tin'):
    """Interactive 3-D view of a building's geometry with terrain and neighbour shadows.

    ``ground_floor_height`` (default 5.0 m) is the TALLER ground storey; upper
    storeys use ``floor_height`` (default 3.0 m). All heights come from
    ``floor_base_elevation`` so the shades/zones/walls sit at the SAME elevations
    the shadow analysis used (which assumes a 5 m ground floor) — otherwise the
    neighbour-shadow contours render too low.

    Renders three layers in a single plotly scene:
      1. **Terrain**  — ``br__terrain_grid`` as a coloured heightmap surface
         (recentred on the building centroid).
      2. **Building shades** — ``br__shadows`` per-floor blocked-region polygons
         from neighbouring buildings, shown as semi-transparent planes at each
         floor's slab elevation (``floor_base_elevation``, taller ground floor).
      3. **Wall segments** — ``br__wall_segments`` extruded into vertical rectangles
         spanning each floor's slab to the one above (``floor_base_elevation``),
         coloured by wall type.

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

    Notes
    -----
    Patio walls and the "no slab over an open patio" rule are resolved during
    building inference (see ``building_inference`` patio-void handling), so this
    function simply renders ``br__wall_segments`` / ``br__structural_floors`` as
    produced. Re-run inference to pick up changes to that logic.
    """
    import plotly.graph_objects as go

    building_data = gdf[gdf['building_reference'] == building_reference]
    if building_data.empty:
        print(f"No data found for building {building_reference}")
        return None
    row = building_data.iloc[0].copy()

    # ---- Auto-fetch heavy columns from DuckDB ---------------------------------
    heavy_cols = ['br__wall_segments', 'br__shadows', 'br__terrain_grid',
                  'br__building_footprint_by_floor', 'br__thermal_zones']
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
    tz_cat_traces = {}                  # category -> [trace_index] for the zone menu
    tz_cat_shown = set()                # categories already given a legend entry
    _shade_legend_shown = set()         # collapse per-floor shades to one entry
    terrain_trace_idx = None            # index of the terrain surface trace

    # ---- Layer 1: Terrain DEM surface (smooth thin-plate spline, no pixels) --
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

            # Use terrain centre elevation as ground_z fallback
            if ground_z is None:
                ground_z = float(elev[tny // 2, tnx // 2])
            oz = ground_z

            # DEM height legend along the BOTTOM of the plot.
            _terrain_cbar = dict(title='Height above ground (m)',
                                 orientation='h', thickness=12, len=0.6,
                                 x=0.5, xanchor='center', y=0.0, yanchor='top')

            if terrain_interp in ('tin', 'linear', 'adaptive'):
                # Treat the DEM as a point cloud and Delaunay-triangulate it
                # into a TIN — faithful planar facets, every vertex a real
                # sample, no fabricated undulation.
                tx, ty, tz, tris = _build_terrain_tin(
                    elev, lx, ly, adaptive=(terrain_interp == 'adaptive'))
                zs = tz - oz
                mesh_kw = (dict(i=tris[:, 0], j=tris[:, 1], k=tris[:, 2])
                           if tris is not None else dict(delaunayaxis='z'))
                traces.append(go.Mesh3d(
                    x=tx, y=ty, z=zs,
                    intensity=zs, colorscale='earth', showscale=True,
                    colorbar=_terrain_cbar,
                    opacity=0.85, name='Terrain', flatshading=True,
                    hovertemplate='ΔX=%{x:.1f} m<br>ΔY=%{y:.1f} m<br>h=%{z:.1f} m<extra></extra>',
                    **mesh_kw,
                ))
                terrain_trace_idx = len(traces) - 1
                all_x.extend([float(tx.min()), float(tx.max())])
                all_y.extend([float(ty.min()), float(ty.max())])
                all_z.extend([float(np.nanmin(zs)), float(np.nanmax(zs))])
            else:
                # Smooth thin-plate-spline surface through the pixel centroids.
                lx_f, ly_f, elev_f = _smooth_terrain_surface(elev, lx, ly, upsample=4)
                elev_rel = elev_f - oz       # height above building ground
                GX, GY = np.meshgrid(lx_f, ly_f)
                traces.append(go.Surface(
                    x=GX, y=GY, z=elev_rel,
                    colorscale='earth', showscale=True,
                    colorbar=_terrain_cbar,
                    opacity=0.85,
                    name='Terrain',
                    hovertemplate='ΔX=%{x:.1f} m<br>ΔY=%{y:.1f} m<br>h=%{z:.1f} m<extra></extra>',
                ))
                terrain_trace_idx = len(traces) - 1
                all_x.extend([float(lx_f[0]), float(lx_f[-1])])
                all_y.extend([float(ly_f[0]), float(ly_f[-1])])
                all_z.extend([float(np.nanmin(elev_rel)), float(np.nanmax(elev_rel))])
        except Exception:
            pass  # terrain is nice-to-have

    if ground_z is None:
        ground_z = 0.0
    oz = ground_z

    # Floor slab elevations (ground-relative): the ground floor is taller, matching
    # the shadow analysis and the EnergyPlus ``_floor_z`` — a flat ``fl*floor_height``
    # would place every upper floor (and its shadow contour) too low. The taller
    # ground floor applies to buildings with >= 5 floors above ground OR to
    # industrial/agricultural buildings (the same rule the EnergyPlus model uses);
    # other smaller buildings use ``floor_height`` for every storey.
    try:
        _n_ag = int(row.get('br__floors_above_ground'))
    except (TypeError, ValueError):
        _fp_floors = row.get('br__building_footprint_by_floor')
        _n_ag = (1 + max((int(r[1]) for r in _fp_floors if int(r[1]) >= 0), default=0)
                 if isinstance(_fp_floors, (list, tuple)) and _fp_floors else 1)
    _eff_gfh = effective_ground_floor_height(_n_ag, ground_floor_height, floor_height,
                                             row.get('building_use'))

    def _fb(fl):
        return floor_base_elevation(0.0, fl, _eff_gfh, floor_height)

    # ---- Layer 2: Shadow boundary panels (buffer circle edges excluded) -------
    # Each floor's shadow polygon boundary is extruded between the floor's slab and
    # the one above.  Segments that lie on the analysis buffer circle (artefacts of
    # the search radius, not real buildings) are filtered out.
    shadows_geom = row.get('br__shadows')
    if isinstance(shadows_geom, dict) and shadows_geom:
        try:
            # Distances of every shadow vertex from the centroid.
            _radii = []
            for poly in shadows_geom.values():
                if poly is None or (hasattr(poly, 'is_empty') and poly.is_empty):
                    continue
                for g in (list(poly.geoms) if hasattr(poly, 'geoms') else [poly]):
                    if hasattr(g, 'exterior'):
                        for xp, yp, *_ in g.exterior.coords:
                            _radii.append(math.hypot(xp - ox, yp - oy))
            shadow_radius = max(_radii) if _radii else 0.0

            # In open-sky directions the shadow polygon runs all the way out to
            # the analysis buffer, forming a dense outer band near the search
            # radius that is separated from the real, blocker-cast shadows by a
            # radial gap. Find the largest gap in the outer region (> 0.85 R)
            # and cut there, so the buffer-circle band AND the spokes reaching
            # it are dropped entirely; fall back to trimming the outer 2 %.
            cut_radius = shadow_radius * 0.98 if shadow_radius > 0 else 0.0
            if shadow_radius > 0:
                _outer = sorted(r for r in _radii if r > 0.85 * shadow_radius)
                _best_gap, _best_at = 0.0, None
                for _i in range(len(_outer) - 1):
                    _g = _outer[_i + 1] - _outer[_i]
                    if _g > _best_gap:
                        _best_gap, _best_at = _g, _outer[_i]
                if _best_at is not None and _best_gap >= 3.0:
                    cut_radius = _best_at + 1e-6

            def _on_circle(x0, y0, x1, y1):
                # A segment is a buffer-circle artefact if EITHER endpoint lies
                # beyond the cut radius (drops both the arc and its spokes).
                return (math.hypot(x0 - ox, y0 - oy) > cut_radius or
                        math.hypot(x1 - ox, y1 - oy) > cut_radius)

            shade_colours = ['#7F8C8D', '#95A5A6', '#BDC3C7', '#909090',
                             '#808B96', '#778899', '#A9A9A9', '#696969']
            for floor_idx, poly in sorted(shadows_geom.items()):
                if poly is None or (hasattr(poly, 'is_empty') and poly.is_empty):
                    continue
                z_bot_rel = _fb(floor_idx)
                z_top_rel = _fb(floor_idx + 1)
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
                        name='Neighbour shades', legendgroup='shades',
                        showlegend=('shades' not in _shade_legend_shown),
                        flatshading=True,
                        hovertemplate=f'Floor {floor_idx}<br>h={z_mid_rel:.1f} m<extra></extra>',
                    ))
                    _shade_legend_shown.add('shades')
        except Exception:
            pass  # shades are nice-to-have

    # ---- Layer 3: Wall segments (centroid-relative X/Y, ground-relative Z) ---
    # Draw the SHRUNK wall segments (free façades moved inward to the CAT built
    # area, party/adiabatic anchored) so the plotted envelope coincides with the
    # thermal zones (Layer 5) and the EnergyPlus model — one single volumetry.
    try:
        from hypercadaster_ES import energy as _energy
        wall_segments_draw = _energy.shrunk_wall_segments(row) or wall_segments
    except Exception:
        wall_segments_draw = wall_segments
    type_verts = {}
    _wall_x, _wall_y, _wall_z = [], [], []   # wall-only coords for scene span
    for seg in wall_segments_draw:
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

        z_bot_rel = _fb(floor)
        z_top_rel = _fb(floor + 1)

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

    # ---- Layer 4: Structural floor/roof slabs -------------------------------
    # Classify each horizontal slab region by the SHRUNK thermal-zone volumetry
    # (the single geometry shared with the EnergyPlus model), NOT the gross
    # footprint: terrain (lowest base), interior (slab between two stacked zones),
    # roof (top with no zone above → exposed to air), floor (underside with no
    # zone below → exposed to air). Using the zones is what makes air-contact
    # slabs show correctly instead of being lumped into the interior ("adiabatic")
    # category. The zones are computed ONCE here and reused by Layer 5.
    try:
        from hypercadaster_ES import energy as _energy
        _space_zones = _energy.synthesize_space_geometry(row)
    except Exception:
        _space_zones = []
    _zone_floor_union = {}
    for _z in _space_zones:
        if _z.get('is_pool'):
            continue                                   # pools are not enclosed slabs
        for _fl, _poly in (_z.get('floor_polys') or {}).items():
            if _poly is not None and not getattr(_poly, 'is_empty', True):
                _zone_floor_union.setdefault(int(_fl), []).append(_poly)

    fp_by_floor = row.get('br__building_footprint_by_floor')
    if _zone_floor_union or (isinstance(fp_by_floor, list) and fp_by_floor):
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

            # Per-floor union from the thermal zones (single volumetry); fall back
            # to the gross footprint only if no zones were synthesized.
            floor_union = {}
            if _zone_floor_union:
                for _fl, _polys in _zone_floor_union.items():
                    try:
                        floor_union[_fl] = _uu_floor(_polys)
                    except Exception:
                        floor_union[_fl] = _polys[0] if _polys else None
            else:
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
            min_floor = min(floor_union) if floor_union else 0

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

            # Terrain: base of the lowest floor (slab on/under ground). For a
            # building with basements this is the deepest underground floor;
            # otherwise floor 0 at ground level (z = 0).
            base_lowest = floor_union.get(min_floor)
            if base_lowest is not None and not base_lowest.is_empty:
                struct_regions.append((base_lowest, 'terrain', _fb(min_floor)))

            # Iterate every interface from the lowest floor up to the roof,
            # below-ground floors included.
            for _f in range(min_floor, max_floor + 1):
                a = floor_union.get(_f)
                if a is None or a.is_empty:
                    continue
                z_top = _fb(_f + 1)                    # top-of-floor slab level
                above = floor_union.get(_f + 1)
                if above is None or above.is_empty:
                    # Topmost present floor — its top is roof (above ground) or
                    # the ground slab to floor 0 (below ground).
                    stype = 'terrain' if _f < 0 else 'roof'
                    struct_regions.append((a, stype, z_top))
                    continue
                inter = _safe_inter(a, above)
                roof_reg = _safe_diff(a, above)        # not covered above → roof
                over_reg = _safe_diff(above, a)        # overhang underside → floor
                if inter is not None and not inter.is_empty:
                    struct_regions.append((inter, 'interior', z_top))
                if roof_reg is not None and not roof_reg.is_empty:
                    # Below ground a "not covered above" slab is still earth-
                    # covered, so it is terrain rather than sky-facing roof.
                    struct_regions.append((roof_reg, 'terrain' if _f < 0 else 'roof', z_top))
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

            # ---- Layer 5: Thermal zones (one per building space) ------------
            # Render the exact per-space 3-D zones that feed the EnergyPlus model
            # (energy.synthesize_space_geometry, computed once above): one prism
            # per (space, floor). Pools → shallow basin from their real footprint;
            # parking/warehouse → grey end block; communal core → brown; occupied
            # → palette. Zones start hidden and are toggled via the checkbox menu.
            if show_thermal_zones:
                for _zi, _zone in enumerate(_space_zones):
                    _kind = _zone.get('kind', 'occupied')
                    if _kind == 'communal':
                        colour = _THERMAL_ZONE_COMMUNAL_COLOUR
                    elif _kind == 'non_occupied':
                        colour = _THERMAL_ZONE_NON_OCCUPIED_COLOUR
                    else:
                        colour = _THERMAL_ZONE_PALETTE[_zi % len(_THERMAL_ZONE_PALETTE)]
                    bkt = {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': []}
                    is_pool = bool(_zone.get('is_pool'))
                    for _fl, _poly in sorted(_zone.get('floor_polys', {}).items()):
                        if is_pool:                       # water surface at floor level
                            _zt = _fb(_fl)            # water surface at floor level
                            _append_prism_mesh(bkt, _poly, _zt - 1.2, _zt, ox, oy)
                        else:
                            _append_prism_mesh(bkt, _poly, _fb(_fl), _fb(_fl + 1), ox, oy)
                    if not bkt['x']:
                        continue
                    # Group zones by CATEGORY in the legend/menu (one entry per use
                    # type / role) — a per-zone legend of hundreds of items would
                    # squeeze the 3-D scene to nothing. Only the first zone of each
                    # category carries the legend entry; the whole group toggles
                    # together (legend ``groupclick`` + the per-category menu).
                    _cat = _zone_legend_category(_zone)
                    _hover = f"{_zone.get('use_type')}|{_zone.get('reference', '')}"
                    traces.append(go.Mesh3d(
                        x=bkt['x'], y=bkt['y'], z=bkt['z'],
                        i=bkt['i'], j=bkt['j'], k=bkt['k'],
                        color=colour, opacity=0.45,
                        name=_cat, legendgroup=_cat,
                        showlegend=(_cat not in tz_cat_shown), flatshading=True,
                        visible='legendonly',   # start hidden; toggle via menu/legend
                        hovertemplate=f'{_hover}<extra></extra>',
                    ))
                    all_x.extend(bkt['x']); all_y.extend(bkt['y']); all_z.extend(bkt['z'])
                    tz_cat_shown.add(_cat)
                    tz_cat_traces.setdefault(_cat, []).append(len(traces) - 1)
        except Exception:
            pass  # structural slabs are nice-to-have

    # ---- XY window matched to plot_building_shadows_analysis -----------------
    # Use the SAME fixed ±extent window the shadow-analysis spatial panels draw
    # (SHADOW_PLOT_EXTENT_R), centred on the building footprint centroid. Scene
    # coords are already centroid-relative, so the centre is (0, 0). Z spans the
    # building height; aspectmode 'data' (set below) keeps a TRUE 1:1:1 metre
    # scale across X/Y/Z regardless of how the ranges differ.
    if _wall_z:
        half = SHADOW_PLOT_EXTENT_R
        x_range = [-half, half]
        y_range = [-half, half]
        # Z must span EVERYTHING drawn (terrain dips well below the building's
        # base across the 150 m window, plus shadow panels and the building
        # itself), otherwise the terrain surface is clipped at low altitudes.
        _zs = all_z if all_z else _wall_z
        zmin, zmax = min(_zs), max(_zs)
        zpad = max((zmax - zmin) * 0.05, 0.5)
        z_range = [zmin - zpad, zmax + zpad]
    else:
        x_range = y_range = z_range = [-SHADOW_PLOT_EXTENT_R, SHADOW_PLOT_EXTENT_R]

    # ---- Title: building ref + first 3 (unique) full addresses --------------
    # Street labels live per-address; after address compaction the cell holds a
    # list (e.g. ['CL LLOBERA, 47', 'CL NOU, 40', ...]). Keep the COMPLETE
    # address (street + house number), de-duplicate, take the first 3.
    _street_names = []
    for _scol in ('postal_addresses_with_extensions', 'postal_addresses_clean'):
        if _scol in building_data.columns:
            for _v in building_data[_scol].dropna():
                for _it in (_v if isinstance(_v, (list, tuple)) else [_v]):
                    if _it is None:
                        continue
                    _s = str(_it).strip()
                    if _s and _s not in _street_names:
                        _street_names.append(_s)
            if _street_names:
                break
    _title = f'3-D building geometry — {building_reference}'
    if _street_names:
        _title += '<br>' + ' · '.join(_street_names[:3])

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=_title,
        scene=dict(
            xaxis=dict(title='ΔX (m)', range=x_range),
            yaxis=dict(title='ΔY (m)', range=y_range),
            zaxis=dict(title='Height above ground (m)', range=z_range),
            # 'data' keeps a true 1:1:1 metre scale derived from the (possibly
            # unequal) X/Y/Z ranges, so the XY frame matches the 2-D plot while
            # the Z axis still spans the full building height.
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        legend=dict(title='Layer', x=0.01, y=0.99, groupclick='togglegroup',
                    itemsizing='constant'),
        margin=dict(l=0, r=0, b=60, t=40),
    )

    updatemenus = []
    annotations = []

    # ---- Terrain visibility toggle ------------------------------------------
    # Single checkbox-style button: first click hides the terrain surface,
    # next click shows it again (args / args2 alternate).
    if terrain_trace_idx is not None:
        updatemenus.append(dict(
            type='buttons', direction='down', showactive=True,
            x=1.02, xanchor='left', y=0.98, yanchor='top',
            pad={'r': 4, 't': 4, 'b': 4, 'l': 4},
            bgcolor='#FFFFFF', bordercolor='#CCCCCC', borderwidth=1,
            font=dict(size=10),
            buttons=[dict(
                label='Terrain on/off', method='restyle',
                args=[{'visible': False}, [terrain_trace_idx]],
                args2=[{'visible': True}, [terrain_trace_idx]],
            )],
        ))

    # ---- Thermal-zone checkbox menu -----------------------------------------
    # Each zone gets a toggle button: `args` shows it, `args2` hides it again
    # (Plotly applies args/args2 on alternating clicks → checkbox behaviour).
    if tz_cat_traces:
        # One toggle per CATEGORY (use type / role), not per zone — otherwise the
        # menu is a scroll-list of hundreds of buttons that hides the plot.
        all_idx = [i for idxs in tz_cat_traces.values() for i in idxs]
        zone_buttons = [
            dict(label='✓ Show all zones', method='restyle',
                 args=[{'visible': True}, all_idx]),
            dict(label='✗ Hide all zones', method='restyle',
                 args=[{'visible': 'legendonly'}, all_idx]),
        ] + [dict(label=cat, method='restyle',
                  args=[{'visible': True}, idxs],
                  args2=[{'visible': 'legendonly'}, idxs])
             for cat, idxs in tz_cat_traces.items()]
        updatemenus.append(dict(
            type='buttons', direction='down', showactive=False,
            x=1.02, xanchor='left', y=0.85, yanchor='top',
            pad={'r': 4, 't': 4, 'b': 4, 'l': 4},
            bgcolor='#FFFFFF', bordercolor='#CCCCCC', borderwidth=1,
            font=dict(size=10),
            buttons=zone_buttons,
        ))
        annotations.append(dict(
            x=1.02, y=0.87, xref='paper', yref='paper',
            xanchor='left', yanchor='bottom', showarrow=False,
            text='<b>Thermal zones</b>', font=dict(size=11),
        ))

    if updatemenus:
        fig.update_layout(updatemenus=updatemenus)
    if annotations:
        fig.update_layout(annotations=annotations)

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
