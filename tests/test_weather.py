"""Climate statistics and EPW export from a synthetic ERA5-Land year."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

import hypercadaster_ES as hc
from hypercadaster_ES import weather


# --------------------------------------------------------------------------- #
# Helper: a canonical single-cell ERA5-Land hourly frame (post transformation)
# --------------------------------------------------------------------------- #
def _fake_cell_frame(lat, lon, year=2023, days=3, null=False):
    """A few days of hourly canonical weather for one cell, matching the columns
    join_weather / weather_statistics consume. ``null`` blanks airTemperature to
    emulate an over-sea (no-data) cell."""
    start = dt.datetime(year, 1, 1)
    times = pl.datetime_range(
        start, start + dt.timedelta(hours=24 * days - 1),
        interval="1h", eager=True, time_zone="UTC")
    n = len(times)
    at = [None] * n if null else list(np.linspace(2.0, 12.0, n))
    return pl.DataFrame({
        "latitude": [lat] * n,
        "longitude": [lon] * n,
        "time": times,
        "airTemperature": pl.Series(at, dtype=pl.Float64),
        "relativeHumidity": [65.0] * n,
        "windSpeed": [3.0] * n,
        "GHI": [120.0] * n,
        "totalPrecipitation": [0.2] * n,
    })


# --------------------------------------------------------------------------- #
# Pure helpers: grid bounds, year resolution, ARD config
# --------------------------------------------------------------------------- #
def test_grid_bounds_snaps_outward_to_whole_degrees():
    b = weather._grid_bounds([41.32, 41.47], [1.98, 2.23])
    assert (b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"]) == (41.0, 42.0, 1.0, 3.0)
    # works for a western (negative-lon) province too, and is clamped to valid range
    b2 = weather._grid_bounds([40.0, 41.2], [-4.6, -3.05])
    assert (b2["lat_min"], b2["lat_max"], b2["lon_min"], b2["lon_max"]) == (40.0, 42.0, -5.0, -3.0)
    assert weather._bounds_str(b) == "42.0_1.0_41.0_3.0"


def test_resolve_weather_years():
    assert weather._resolve_weather_years([2022, 2020, 2020]) == [2020, 2022]
    assert weather._resolve_weather_years(0) == []
    last = dt.date.today().year - 1
    assert weather._resolve_weather_years(3) == [last - 2, last - 1, last]


def test_ard_helpers(monkeypatch):
    url = weather._ard_store_url("sfc-2m-temperature", "007", "geo")
    assert url == ("https://arco.datastores.ecmwf.int/cadl-arco-geo-007/"
                   "arco/reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr")
    # env override wins over key presence
    monkeypatch.setenv("HC_ERA5_ARD", "0")
    assert weather._ard_enabled() is False
    monkeypatch.setenv("HC_ERA5_ARD", "1")
    assert weather._ard_enabled() is True
    # without the override, it follows key availability
    monkeypatch.delenv("HC_ERA5_ARD", raising=False)
    monkeypatch.setattr(weather, "_ard_cds_key", lambda: None)
    assert weather._ard_enabled() is False


# --------------------------------------------------------------------------- #
# join_weather (network mocked)
# --------------------------------------------------------------------------- #
def test_join_weather_adds_indicator_columns(monkeypatch, building_gdf):
    monkeypatch.setattr(weather, "load_ERA5Land",
                        lambda **kw: _fake_cell_frame(41.6, 0.6))
    out = weather.join_weather(building_gdf, "ignored", years=[2023], verbose=False)
    expected = {f"{weather.WEATHER_PREFIX}{stem}_2023"
                for stem in weather.WEATHER_INDICATORS.values()}
    assert expected.issubset(set(out.columns))
    assert len(expected) == 9
    # every building resolves to the single cell -> all non-null and identical
    assert out["we__hdd_2023"].notna().all()
    assert out["we__hdd_2023"].nunique() == 1
    # original rows/columns preserved
    assert len(out) == len(building_gdf)


def test_join_weather_indicator_subset(monkeypatch, building_gdf):
    monkeypatch.setattr(weather, "load_ERA5Land",
                        lambda **kw: _fake_cell_frame(41.6, 0.6))
    out = weather.join_weather(building_gdf, "ignored", years=[2023],
                               indicators=["HDD", "CDD"], verbose=False)
    assert "we__hdd_2023" in out.columns and "we__cdd_2023" in out.columns
    assert "we__precipitation_2023" not in out.columns
    assert "we__ghi_2023" not in out.columns


def test_join_weather_bbox_forces_grid(monkeypatch, building_gdf):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return _fake_cell_frame(41.6, 0.6)

    monkeypatch.setattr(weather, "load_ERA5Land", fake)
    # bbox given + point flag on: must use the grid and the bbox extent
    weather.join_weather(building_gdf, "ignored", years=[2023],
                         bbox=[1.9, 41.2, 2.6, 41.6], use_point_location=True,
                         search_deg=0.6, verbose=False)
    assert len(calls) == 1
    assert calls[0]["use_point_location"] is False
    assert calls[0]["lat_range"] == pytest.approx([41.2 - 0.6, 41.6 + 0.6])
    assert calls[0]["lon_range"] == pytest.approx([1.9 - 0.6, 2.6 + 0.6])


def test_join_weather_point_mode_sea_fallback(monkeypatch, building_gdf):
    seen = []

    def fake(**kw):
        seen.append(kw["use_point_location"])
        # point request lands on an all-null (sea) cell; grid request has data
        return _fake_cell_frame(41.6, 0.6, null=kw["use_point_location"])

    monkeypatch.setattr(weather, "load_ERA5Land", fake)
    out = weather.join_weather(building_gdf, "ignored", years=[2023],
                               use_point_location=True, verbose=False)
    assert seen == [True, False]           # tried point, fell back to grid
    assert out["we__hdd_2023"].notna().all()


def test_weather_statistics_degree_days(weather_year_df):
    stats = hc.weather.weather_statistics(weather_year_df, hdd_base=18.0, cdd_base=21.0)
    # constant 10 C -> HDD = (18-10) * 366 days, CDD = 0
    assert stats["CDD"] == 0.0
    assert stats["HDD"] == round(8.0 * stats["n_days"], 1)
    assert stats["t_mean"] == 10.0
    assert stats["n_days"] == 366  # 2020 is a leap year


def test_weather_statistics_monthly(weather_year_df):
    stats = hc.weather.weather_statistics(weather_year_df, by_month=True)
    monthly = stats["monthly"]
    assert monthly.height == 12
    assert {"month", "HDD", "CDD"}.issubset(set(monthly.columns))


def test_write_epw_is_8760(weather_year_df, tmp_path):
    path = tmp_path / "test.epw"
    weather.write_epw(
        weather_year_df, str(path),
        year=2020, latitude=41.4, longitude=2.2, elevation=12.0,
    )
    assert path.exists()
    lines = path.read_text().splitlines()
    # 8 header rows + 8760 hourly data rows (Feb 29 dropped)
    assert len(lines) == 8 + 8760
    assert lines[0].startswith("LOCATION,")
    # data rows begin with the year
    assert lines[8].split(",")[0] == "2020"
