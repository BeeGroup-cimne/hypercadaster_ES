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

# Minimal top-level API. Everything else is reached through the submodules
# (e.g. hc.utils.ine_to_cadaster_codes, hc.mergers.aggregate_to_grid,
# hc.energy.build_energy_model, hc.weather.municipality_epw).
from .functions import download, merge
from ._db import ResultsHandle
from . import (mergers, utils, building_inference, downloaders, interoperability,
               energy, weather)

__version__ = "1.1.0"
__author__ = "Jose Manuel Broto Vispe"
__email__ = "jmbrotovispe@gmail.com"

# Docstrings across the package are written in NumPy style; tell pdoc so it
# renders the Parameters/Returns sections as proper lists rather than plain text.
__docformat__ = "numpy"

__all__ = [
    # Top-level API
    "download", "merge", "ResultsHandle",
    # Submodules (this order drives the API-docs index)
    "downloaders", "energy", "interoperability", "mergers", "utils", "weather",
]