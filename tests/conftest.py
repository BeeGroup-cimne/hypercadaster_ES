"""Shared fixtures for the hypercadaster_ES test-suite.

All fixtures here are fully synthetic and offline. Network/end-to-end tests
live in ``test_live_smoke.py`` and are gated behind the ``live`` marker.
"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon


# --------------------------------------------------------------------------- #
# Network / live-test gating
# --------------------------------------------------------------------------- #
def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.live`` tests unless explicitly requested.

    Enable them with ``pytest -m live`` or by setting ``HC_RUN_LIVE=1``.
    """
    if config.getoption("-m") == "live" or os.environ.get("HC_RUN_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live network test; set HC_RUN_LIVE=1 or run -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# --------------------------------------------------------------------------- #
# Building-level GeoDataFrame (metric CRS) for aggregation tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def building_gdf():
    """A tiny building-level GeoDataFrame in EPSG:25831 (metres).

    Ten buildings placed so that they fall into exactly two 100 m grid cells:
    six around (430050, 4580050) and four around (430150, 4580050).
    """
    xs = [430010, 430020, 430030, 430040, 430050, 430060,   # cell A (ix=0)
          430110, 430120, 430130, 430140]                    # cell B (ix=1)
    ys = [4580010, 4580020, 4580030, 4580040, 4580050, 4580060,
          4580010, 4580020, 4580030, 4580040]
    n = len(xs)
    centroids = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:25831")
    # A small square footprint per building (so the default geometry is a polygon).
    footprints = [Point(x, y).buffer(4.0).envelope for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(
        {
            "building_reference": [f"B{i:03d}" for i in range(n)],
            "n_dwellings": [2, 1, 3, 4, 2, 1, 5, 2, 1, 3],
            "building_area": [100.0 * (i + 1) for i in range(n)],
            "year_of_construction": [1950 + 5 * i for i in range(n)],
            "building_centroid": centroids,
            "geometry": footprints,
        },
        geometry="geometry",
        crs="EPSG:25831",
    )
    return gdf


# --------------------------------------------------------------------------- #
# Synthetic ERA5-Land-style hourly weather (one calendar year)
# --------------------------------------------------------------------------- #
@pytest.fixture
def weather_year_df():
    """One year of hourly weather for a single cell as a polars DataFrame.

    Air temperature is a constant 10 °C so that degree-days are deterministic:
    HDD(base 18) = 8 K * 365 days = 2920, CDD(base 21) = 0.
    """
    start = dt.datetime(2020, 1, 1, 0, 0, 0)
    times = [start + dt.timedelta(hours=h) for h in range(24 * 366)]
    # keep only the 2020 hours the callers expect (Feb 29 is fine, it is dropped
    # by write_epw itself)
    times = [t for t in times if t.year == 2020]
    n = len(times)
    df = pl.DataFrame(
        {
            "time": times,
            "airTemperature": [10.0] * n,
            "dewAirTemperature": [5.0] * n,
            "relativeHumidity": [70.0] * n,
            "windSpeed": [3.0] * n,
            "windDirection": [180.0] * n,
            "pressure": [101325.0] * n,
            "GHI": [100.0] * n,
            "DNI": [0.0] * n,
            "DHI": [100.0] * n,
            "totalPrecipitation": [0.001] * n,
        }
    )
    return df
