"""
hypercadaster_ES: A Python library for Spanish cadastral data processing.

This library provides comprehensive tools for downloading, processing, and merging
Spanish cadastral data with external geographic datasets including census tracts,
postal codes, elevation models, and Barcelona open data.

Main functionality:
- Download cadastral data for specified provinces/municipalities
- Merge with external geographic and administrative datasets
- Building inference and analysis capabilities
- Export capabilities for external simulation tools

Examples:
    Basic usage:
    >>> import hypercadaster_ES as hc
    >>> hc.download("./data", province_codes=["08"])
    >>> gdf = hc.merge("./data", province_codes=["08"])
"""

from .functions import download, merge
from .utils import get_ine_codes_from_bounding_box, ine_to_cadaster_codes, municipality_name
from .mergers import aggregate_to_grid, aggregate_to_h3
from ._db import ResultsHandle
from . import mergers, utils, building_inference, downloaders, interoperability

__version__ = "1.1.0"
__author__ = "Jose Manuel Broto Vispe"
__email__ = "jmbrotovispe@gmail.com"

__all__ = [
    "download", "merge", "get_ine_codes_from_bounding_box",
    "ine_to_cadaster_codes", "municipality_name",
    "aggregate_to_grid", "aggregate_to_h3",
    "ResultsHandle",
]