# -*- coding: utf-8 -*-
import polars as pl
import sys
import pandas as pd
import numpy as np
import warnings
import pvlib
import os
from tqdm import tqdm
import random
import functools as ft
import cdsapi
import xarray as xr
import datetime as dt
import zipfile
import shutil


def cleaning_pipe(df, lat=None, lon=None):
    """Clean a raw ERA5-Land frame: drop nulls, optionally filter to one cell,
    and rename CDS/GRIB variable codes to the library's canonical names.

    Parameters
    ----------
    df : polars.DataFrame
        Raw ERA5-Land records (CDS or GRIB column names).
    lat, lon : float, optional
        When given, keep only the cell whose rounded (0.1°) coordinates match.

    Returns
    -------
    polars.DataFrame
        Cleaned frame with ``valid_time`` renamed to ``time`` and variables
        renamed per :func:`ERA5Land_variables`.
    """
    df = df.drop_nulls()

    if lat is not None or lon is not None:
        df = df.filter(
            (pl.col("latitude").round(1) == round(lat, 1)) &
            (pl.col("longitude").round(1) == round(lon, 1))
        )
    elif lon is not None:
        df = df.filter(pl.col("longitude").round(1) == round(lon, 1))
    elif lat is not None:
        df = df.filter(pl.col("latitude").round(1) == round(lat, 1))

    df = df.drop(["time", "step", "number", "surface"], strict=False)  # Ignore errors

    renamed_columns = {**{'valid_time': 'time'},
                       **{x['CDSShortName']: x['name'] for x in ERA5Land_variables()},
                       **{x['CDSGribShortName']: x['name'] for x in ERA5Land_variables()}}
    renamed_columns_filtered = {k: v for k, v in renamed_columns.items() if k in df.columns}

    df = df.rename(renamed_columns_filtered)

    return df


def transformation_pipe(df, mode="averaged"):
    """Convert cleaned ERA5-Land variables to physical/engineering units.

    Derives wind speed/direction from the u/v components, converts temperatures
    from kelvin to Celsius, computes relative humidity from air and dew-point
    temperature, localises ``time`` to UTC, and (when ``mode == "averaged"``)
    applies temporal averaging to the instantaneous variables. Backfills
    ``surfacePressure`` with the ICAO standard sea-level value for older caches
    that predate that download field.

    Parameters
    ----------
    df : polars.DataFrame
        Output of :func:`cleaning_pipe`.
    mode : {"averaged", "instantaneous"}, default "averaged"
        Whether to temporally average the instantaneous variables.

    Returns
    -------
    polars.DataFrame
        Transformed frame with ``windSpeed``, ``windDirection``,
        ``relativeHumidity`` and Celsius temperatures.
    """
    df = df.sort(["latitude", "longitude", "time"])

    # Surface pressure was added to the download list after some early caches
    # were created; if it is absent fall back to the ICAO standard-atmosphere
    # sea-level pressure so the solar-position/irradiance step still runs.
    if "surfacePressure" not in df.columns:
        df = df.with_columns(pl.lit(101325.0).alias("surfacePressure"))

    # Transform time
    df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))  # Localize to UTC

    # Transform temperature, wind variables and humidity
    df = df.with_columns([
        np.sqrt(pl.col("windSpeedEast") ** 2 + pl.col("windSpeedNorth") ** 2)
        .alias("windSpeed"),
        ((np.degrees(np.arctan2(pl.col("windSpeedEast"), pl.col("windSpeedNorth")))) % 360)
        .alias("windDirection"),
        (pl.col("soilTemperature") - 273.15).alias("soilTemperature"),
        (pl.col("dewAirTemperature") - 273.15).alias("dewAirTemperature"),
        (pl.col("airTemperature") - 273.15).alias("airTemperature")
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.with_columns(
            pl.when(pl.col("airTemperature").is_null() | pl.col("dewAirTemperature").is_null())
            .then(None)
            .otherwise(
                (100 * (np.exp(17.67 * pl.col("dewAirTemperature") / (pl.col("dewAirTemperature") + 243.5)) /
                        np.exp(17.67 * pl.col("airTemperature") / (pl.col("airTemperature") + 243.5))))
            ).alias("relativeHumidity")
        )

    # Apply temporal averaging only if mode is "averaged"
    for var in ["windSpeed", "soilTemperature", "dewAirTemperature", "airTemperature", "relativeHumidity"]:
        if var in df.columns:
            if mode == "averaged":
                df = df.with_columns(
                    pl.when(pl.col("time") < df.select(pl.col("time").max()).item())
                    .then((pl.col(var).shift(-1) + pl.col(var)) / 2)
                    .otherwise(pl.col(var))
                    .alias(f"{var}_avg")
                )
            else:
                df = df.with_columns(
                    pl.col(f"{var}").alias(f"{var}_avg")
                )

    # Transform GHI and total precipitation based on mode
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(0)
        .otherwise(pl.col("GHI") / 3600)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(pl.col("totalPrecipitation"))
        .otherwise(pl.col("totalPrecipitation") * 1000)
        .alias("totalPrecipitation_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("GHI_avg") > 0.0)
        .then(pl.col("GHI_avg"))
        .otherwise(0.0)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("totalPrecipitation_avg") > 0.0)
        .then(pl.col("totalPrecipitation_avg"))
        .otherwise(0.0)
        .alias("totalPrecipitation_avg")
    )

    # Add the weatherStation column
    df = df.with_columns(
        pl.format("{},{}", pl.col("latitude").round(1), pl.col("longitude").round(1)).alias("weatherStation")
    )

    # Drop and rename columns
    df = df.drop(["windSpeed", "soilTemperature", "dewAirTemperature", "airTemperature", "relativeHumidity", "GHI",
                  "totalPrecipitation"])
    rename_map = {col: col.replace("_avg", "") for col in df.columns if col.endswith("_avg")}
    df = df.rename(rename_map)

    # Add solar position and solar radiation components
    def join_solar_data(df):

        dfp = df.to_pandas()
        location = pvlib.location.Location(
            latitude=df.select("latitude").unique().item(),
            longitude=df.select("longitude").unique().item())
        solar_df = location.get_solarposition(
            dfp['time'] + pd.Timedelta(minutes=30) if mode == "averaged" else dfp['time'],
            pressure=dfp['surfacePressure'],
            temperature=dfp['airTemperature']).reset_index()
        dni = pvlib.irradiance.disc(
            ghi=dfp["GHI"],
            solar_zenith=solar_df['apparent_zenith'],
            datetime_or_doy=solar_df['time'].dt.dayofyear,
            pressure=dfp["surfacePressure"])
        rad_df = pvlib.irradiance.complete_irradiance(
            solar_zenith=solar_df['apparent_zenith'],
            ghi=dfp["GHI"],
            dni=dni["dni"],
            dhi=None).rename(columns={'ghi': 'GHI', 'dni': 'DNI', 'dhi': 'DHI'})
        solar_df = solar_df.drop(['apparent_zenith', 'zenith', 'apparent_elevation', 'equation_of_time'], axis=1)
        solar_df = solar_df.rename(columns={'elevation': 'sunElevation', 'azimuth': 'sunAzimuth'})
        if mode == "averaged":
            solar_df['time'] = solar_df['time'] - pd.Timedelta(minutes=30)
        solar_df = pl.from_pandas(pd.concat([solar_df, rad_df], axis=1))
        # Ensure time column has consistent microsecond precision
        solar_df = solar_df.with_columns(
            pl.col("time").dt.cast_time_unit("us")
        )

        return solar_df.join(df.drop("GHI"), on="time", how="inner")

    # Loop through each group and apply the join_solar_data function
    result_list = []
    for group_name, group_df in df.group_by("weatherStation"):
        result = join_solar_data(group_df)
        result_list.append(result)
    df = pl.concat(result_list)

    return df


def load_ERA5Land(data_dir, lat_range, lon_range, ym_range, tz_local, update=True, use_point_location=True,
                  mode="averaged"):
    """Download (if needed), clean, transform and cache ERA5-Land weather.

    Ensures the raw ERA5-Land data covering ``ym_range`` is present (via
    :func:`download_ERA5Land`), runs it through :func:`cleaning_pipe` and
    :func:`transformation_pipe`, and caches the result as a Parquet keyed by the
    Catalan region bounds and time range. Returns the cached frame on repeat
    calls unless ``update`` is set.

    Parameters
    ----------
    data_dir : str
        Directory for the raw ERA5-Land files and the derived Parquet cache.
    lat_range, lon_range : sequence[float]
        Bounding coordinates of the area of interest.
    ym_range : sequence
        Year-month range to cover.
    tz_local : str
        Local timezone applied during transformation.
    update : bool, default True
        Force a re-download/rebuild even if the cache exists.
    use_point_location : bool, default True
        Use closest-point (CSV) extraction rather than the full GRIB grid.
    mode : {"averaged", "instantaneous"}, default "averaged"
        Temporal-averaging mode passed to :func:`transformation_pipe`.

    Returns
    -------
    polars.DataFrame
        Cleaned, transformed ERA5-Land weather for the area/period.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Always use Catalan region for parquet file (same as download)
    CATALAN_BOUNDS = {
        'lat_min': 40.5,
        'lat_max': 42.9,
        'lon_min': 0.1,
        'lon_max': 3.4
    }

    # Create parquet file name based on Catalan region bounds, not requested bounds
    catalan_bounds_str = f"{CATALAN_BOUNDS['lat_max']}_{CATALAN_BOUNDS['lon_min']}_{CATALAN_BOUNDS['lat_min']}_{CATALAN_BOUNDS['lon_max']}"
    file = f"{data_dir}/weather_{'~'.join([str(i) for i in ym_range])}_{catalan_bounds_str}_{"ts" if use_point_location else "grid"}.parquet"

    if (not os.path.exists(file)) or update:

        download_ERA5Land(
            data_dir=data_dir,
            ym_range=ym_range,
            lat_range=lat_range,
            lon_range=lon_range,
            use_point_location=use_point_location)

        # Filter GRIB files using Catalan bounds and time range
        # Convert datetime objects to YYYYMM integers for comparison
        min_ym = min(ym_range).year * 100 + min(ym_range).month
        max_ym = max(ym_range).year * 100 + max(ym_range).month

        # Filter GRIB, or CSV files
        list_data_files = []
        for fn in os.listdir(data_dir):
            if (fn.endswith(".csv") and use_point_location):
                # For NetCDF/CSV files from point location, check if date range overlaps
                try:
                    date_part = '_'.join(fn.split('_')[:2])  # Get YYYY-MM-DD_YYYY-MM-DD part
                    file_start, file_end = date_part.split('_')
                    file_start_dt = dt.datetime.strptime(file_start, '%Y-%m-%d')
                    file_end_dt = dt.datetime.strptime(file_end, '%Y-%m-%d')

                    # Check if file date range overlaps with requested range
                    if (file_start_dt <= max(ym_range) and file_end_dt >= min(ym_range)):
                        list_data_files.append(f'{data_dir}/{fn}')
                except:
                    continue
            elif (fn.endswith(".grib") and not use_point_location):
                # For GRIB files, use the original logic (fallback when no point location files exist)
                try:
                    file_ym = int(os.path.splitext(fn)[0].split('_')[0])
                    if file_ym >= min_ym and file_ym <= max_ym and catalan_bounds_str in fn:
                        list_data_files.append(f'{data_dir}/{fn}')
                except:
                    continue

        list_data_files = sorted(list_data_files)
        df = None
        l = len(list_data_files)

        if l == 0:
            raise FileNotFoundError(f"No data files found for the requested time range {ym_range} and area. "
                                    f"Available files: {[fn for fn in os.listdir(data_dir) if fn.endswith(('.grib', '.csv'))]}")

        for i in tqdm(range(l), desc="Loading..."):

            fn = list_data_files[i]

            hash = random.getrandbits(128)

            # Step 1: Convert data to Polars DataFrame
            if fn.endswith('.csv'):
                # Process CSV files (from timeseries API)
                try:
                    df_csv = pl.read_csv(fn)  # Skip header row
                    df_csv = df_csv.pipe(cleaning_pipe)
                    df_csv = df_csv.with_columns([
                        pl.col('time').str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S").dt.replace_time_zone("UTC")
                        # tz_local,non_existent='null')
                    ])
                    df_csv = df_csv.pipe(transformation_pipe, mode=mode)

                    # Add weather station ID based on location
                    lat = df_csv.select("latitude").unique().item()
                    lon = df_csv.select("longitude").unique().item()
                    weather_station_id = f"{lat:.1f}_{lon:.1f}"
                    df_csv = df_csv.with_columns([
                        pl.lit(weather_station_id).alias("weatherStation")
                    ])

                    df_ = df_csv

                except Exception as e:
                    sys.stderr.write(f'\t\tWarning: Error processing CSV {fn}: {e}\n')
                    continue

            else:
                # Process GRIB files
                try:
                    # Most robust approach: open each variable separately
                    datasets = []
                    for var_info in ERA5Land_variables():
                        try:
                            # Try to open specific variable
                            ds = xr.open_dataset(
                                fn,
                                engine='cfgrib',
                                backend_kwargs={'filter_by_keys': {'shortName': var_info['CDSGribShortName']}}
                            )
                            if ds:
                                datasets.append(ds)
                        except Exception:
                            # Skip this variable if it fails
                            continue

                    if not datasets:
                        # Fallback: try opening entire file
                        ds = xr.open_dataset(fn, engine='cfgrib')
                        datasets = [ds]

                    # Convert each dataset to DataFrame and combine
                    df_list = []
                    for ds in datasets:
                        try:
                            df_pandas = ds.to_dataframe()
                            if not df_pandas.empty:
                                df_pandas = df_pandas.reset_index()
                                df_list.append(pl.from_pandas(df_pandas))
                            ds.close()
                        except Exception:
                            ds.close()
                            continue

                    if df_list:
                        df_ = df_list
                    else:
                        continue

                except Exception as e:
                    sys.stderr.write(f'\t\tWarning: Error processing GRIB {fn}: {e}\n')
                    continue

                # Step 2: Apply cleaning and transformation pipelines
                df_ = [dfi.pipe(cleaning_pipe) for dfi in df_]
                df_ = ft.reduce(lambda x, y: x.join(y, on=["latitude", "longitude", "time"], how="left"), df_)
                df_ = df_.with_columns([
                    pl.col('time').dt.cast_time_unit("us").dt.replace_time_zone("UTC")
                    # "#tz_local,non_existent='null')
                ])
                # GRIB accumulated fields (ssrd→GHI, tp→precip) accumulate since
                # 00 UTC and reset each day; de-accumulate to per-hour values
                # BEFORE the transformation (so pvlib gets the real hourly GHI).
                # The timeseries-CSV path already returns per-hour values.
                df_ = _deaccumulate_grib(df_)
                df_ = df_.pipe(transformation_pipe, mode=mode)

            if df is not None:
                df = pl.concat([df, df_])
            else:
                df = df_

        # Only write parquet if we have data
        if df is not None:
            df = df.unique(subset=["latitude", "longitude", "time"]).sort(["latitude", "longitude", "time"])
            df.write_parquet(file)
        else:
            raise RuntimeError("No data was loaded from GRIB/CSV files")

    else:
        df = pl.read_parquet(file)

    # Filter for requested geographical area and time range
    requested_lat_min = round(min(lat_range), 1)
    requested_lat_max = round(max(lat_range), 1)
    requested_lon_min = round(min(lon_range), 1)
    requested_lon_max = round(max(lon_range), 1)

    df = df.filter(
        (pl.col("latitude").round(1) >= requested_lat_min) &
        (pl.col("latitude").round(1) <= requested_lat_max) &
        (pl.col("longitude").round(1) >= requested_lon_min) &
        (pl.col("longitude").round(1) <= requested_lon_max) &
        (pl.col("time") >= pl.lit(min(ym_range)).dt.replace_time_zone("UTC")) &
        (pl.col("time") <= pl.lit(max(ym_range)).dt.replace_time_zone("UTC").dt.offset_by("23h"))
    )

    # Discretise sun position
    df = df.with_columns(
        ((((pl.col("sunAzimuth") + 2) / 4).floor() * 4).cast(pl.Int64).cast(str)).alias("sunAzimuthDiscrete"),
        ((((pl.col("sunElevation") + 1) / 2).floor() * 2).cast(pl.Int64).cast(str)).alias("sunElevationDiscrete")
    )
    df = df.with_columns(
        pl.when(pl.col("sunElevation") > 1)
        .then(
            pl.format("{},{}",
                      pl.col("sunAzimuthDiscrete").str.zfill(3),
                      pl.col("sunElevationDiscrete").str.zfill(2))
        )
        .otherwise(None)
        .alias("sunPosition")
    )

    return df


# Define the variables that should be obtained from the ERA5-Land dataset.
# Check the units and description: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
# Check the long and short names used in CDS API: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation (table 2)
def ERA5Land_variables():
    """Return the ERA5-Land variable catalogue used by the download/clean steps.

    Returns
    -------
    list[dict]
        One entry per variable with its CDS long name, CDS short name, GRIB
        short name and the library's canonical ``name`` (e.g. ``windSpeedEast``,
        ``airTemperature``, ``GHI``, ``totalPrecipitation``).
    """
    return [
        {
            'CDSLongName': '10m_u_component_of_wind',
            'CDSShortName': 'u10',
            'CDSGribShortName': '10u',
            'name': 'windSpeedEast'
        },
        {
            'CDSLongName': '10m_v_component_of_wind',
            'CDSShortName': 'v10',
            'CDSGribShortName': '10v',
            'name': 'windSpeedNorth'
        },
        {
            'CDSLongName': '2m_dewpoint_temperature',
            'CDSShortName': 'd2m',
            'CDSGribShortName': '2d',
            'name': 'dewAirTemperature'
        },
        {
            'CDSLongName': '2m_temperature',
            'CDSShortName': 't2m',
            'CDSGribShortName': '2t',
            'name': 'airTemperature'
        },
        {
            'CDSLongName': 'leaf_area_index_high_vegetation',
            'CDSShortName': 'lai_hv',
            'CDSGribShortName': 'lai_hv',
            'name': 'highVegetationRatio'
        },
        {
            'CDSLongName': 'leaf_area_index_low_vegetation',
            'CDSShortName': 'lai_lv',
            'CDSGribShortName': 'lai_lv',
            'name': 'lowVegetationRatio'
        },
        {
            'CDSLongName': 'total_precipitation',
            'CDSShortName': 'tp',
            'CDSGribShortName': 'tp',
            'name': 'totalPrecipitation'
        },
        {
            'CDSLongName': 'surface_solar_radiation_downwards',
            'CDSShortName': 'ssrd',
            'CDSGribShortName': 'ssrd',
            'name': 'GHI'
        },
        {
            'CDSLongName': 'forecast_albedo',
            'CDSShortName': 'fal',
            'CDSGribShortName': 'fal',
            'name': 'albedo'
        },
        {
            'CDSLongName': 'soil_temperature_level_4',
            'CDSShortName': 'stl4',
            'CDSGribShortName': 'stl4',
            'name': 'soilTemperature'
        },
        {
            'CDSLongName': 'volumetric_soil_water_layer_4',
            'CDSShortName': 'swvl4',
            'CDSGribShortName': 'swvl4',
            'name': 'soilWaterRatio'
        },
        {
            'CDSLongName': 'surface_pressure',
            'CDSShortName': 'sp',
            'CDSGribShortName': 'sp',
            'name': 'surfacePressure'
        }
    ]


def _parse_grib_filename(filename):
    """Parse GRIB filename to extract bounding box info."""
    basename = os.path.basename(filename)
    if not basename.endswith('.grib'):
        return None

    try:
        # Format: YYYYMM_lat_max_lon_min_lat_min_lon_max.grib
        parts = basename.replace('.grib', '').split('_')
        if len(parts) == 5:
            ym, lat_max, lon_min, lat_min, lon_max = parts
            return {
                'ym': int(ym),
                'lat_range': [float(lat_min), float(lat_max)],
                'lon_range': [float(lon_min), float(lon_max)]
            }
    except (ValueError, IndexError):
        pass

    return None


def _parse_csv_filename(filename):
    """Parse CSV filename to extract date range and location info."""
    basename = os.path.basename(filename)
    if not basename.endswith('.csv'):
        return None

    try:
        # Format: YYYY-MM-DD_YYYY-MM-DD_lat_lon.csv
        parts = basename.replace('.csv', '').split('_')
        if len(parts) == 4:
            start_date, end_date, lat, lon = parts
            return {
                'start_date': dt.datetime.strptime(start_date, '%Y-%m-%d').date(),
                'end_date': dt.datetime.strptime(end_date, '%Y-%m-%d').date(),
                'lat': float(lat),
                'lon': float(lon)
            }
    except (ValueError, IndexError):
        pass

    return None


def _check_existing_coverage(data_dir, ym_range, requested_lat_range, requested_lon_range, use_point_location):
    """Check if existing GRIB files cover the requested area and time range."""
    if not os.path.exists(data_dir):
        return False

    # Round requested ranges to 1 decimal for consistency
    req_lat_min = round(min(requested_lat_range), 1)
    req_lat_max = round(max(requested_lat_range), 1)
    req_lon_min = round(min(requested_lon_range), 1)
    req_lon_max = round(max(requested_lon_range), 1)

    if use_point_location == True:
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        # Convert requested date range
        req_start_date = min(ym_range).date() if hasattr(min(ym_range), 'date') else min(ym_range)
        req_end_date = max(ym_range).date() if hasattr(max(ym_range), 'date') else max(ym_range)

        # Parse existing CSV files
        for csv_file in csv_files:
            parsed = _parse_csv_filename(csv_file)
            if parsed:
                # Check if existing file covers requested date range and location
                existing_lat = round(parsed['lat'], 1)
                existing_lon = round(parsed['lon'], 1)

                # Check location coverage (point location should match center of requested area)
                center_lat = round((req_lat_min + req_lat_max) / 2, 1)
                center_lon = round((req_lon_min + req_lon_max) / 2, 1)

                if (existing_lat == center_lat and existing_lon == center_lon and
                        parsed['start_date'] <= req_start_date and parsed['end_date'] >= req_end_date):
                    return True

        return False

    else:
        # Get all GRIB files
        grib_files = [f for f in os.listdir(data_dir) if f.endswith('.grib')]

        # Parse existing files
        existing_files = {}
        for grib_file in grib_files:
            parsed = _parse_grib_filename(grib_file)
            if parsed:
                existing_files[parsed['ym']] = parsed

        # Check coverage for each required year-month
        # Convert datetime objects to YYYYMM integers
        min_ym = min(ym_range).year * 100 + min(ym_range).month
        max_ym = max(ym_range).year * 100 + max(ym_range).month

        ym = min_ym
        while ym <= max_ym:
            if ym not in existing_files:
                return False

            existing = existing_files[ym]
            existing_lat_min = round(min(existing['lat_range']), 1)
            existing_lat_max = round(max(existing['lat_range']), 1)
            existing_lon_min = round(min(existing['lon_range']), 1)
            existing_lon_max = round(max(existing['lon_range']), 1)

            # Check if existing file covers requested area
            if not (existing_lat_min <= req_lat_min and
                    existing_lat_max >= req_lat_max and
                    existing_lon_min <= req_lon_min and
                    existing_lon_max >= req_lon_max):
                return False

            # Move to next month
            year = int(str(ym)[0:4])
            month = int(str(ym)[4:6]) + 1
            if month > 12:
                year += 1
                month = 1
            ym = int(f'{year}{month:02}')

            return True


def _deaccumulate_grib(df_, accumulated=("GHI", "totalPrecipitation")):
    """De-accumulate ERA5-Land GRIB flux fields to per-hour values.

    ERA5-Land accumulated variables (``ssrd``→GHI, ``tp``→precipitation) build
    up since 00 UTC and reset each day. For each cell sorted by time the hourly
    value is the forward difference; a negative difference marks the daily reset
    (use the raw value), and the leading hour of each cell has no predecessor
    (set 0). Leaves non-accumulated columns untouched."""
    present = [c for c in accumulated if c in df_.columns]
    if not present:
        return df_
    df_ = df_.sort(["latitude", "longitude", "time"])
    exprs = []
    for c in present:
        diff = pl.col(c) - pl.col(c).shift(1).over(["latitude", "longitude"])
        exprs.append(
            pl.when(diff.is_null()).then(0.0)
              .when(diff < 0).then(pl.col(c))
              .otherwise(diff)
              .alias(c)
        )
    return df_.with_columns(exprs)


# Function to gather weather data from ERA5-Land
def download_ERA5Land(data_dir, ym_range, lat_range=[90, -90], lon_range=[-180, 180], use_point_location=True):
    """Download ERA5-Land reanalysis from the Copernicus Climate Data Store.

    Skips the download when existing files already cover the requested area and
    period (see ``_check_existing_coverage``). Requires a configured CDS API key
    (``~/.cdsapirc``). With ``use_point_location`` the closest cell is retrieved
    as CSV; otherwise the full GRIB grid for the bounding box is fetched.

    Parameters
    ----------
    data_dir : str
        Destination directory for the downloaded files.
    ym_range : sequence
        Year-month range to download.
    lat_range, lon_range : sequence[float]
        Bounding coordinates (rounded to 0.1° for cell coverage).
    use_point_location : bool, default True
        Retrieve the closest point (CSV) instead of the full GRIB grid.

    Returns
    -------
    None
    """
    # Round lat/lon ranges to 1 decimal to ensure weather station coverage
    lat_range_rounded = [round(min(lat_range), 1), round(max(lat_range), 1)]
    lon_range_rounded = [round(min(lon_range), 1), round(max(lon_range), 1)]

    # Check if existing files already cover the requested area and time
    if _check_existing_coverage(data_dir, ym_range, lat_range_rounded, lon_range_rounded, use_point_location):
        sys.stderr.write(f"\tExisting {"CSV" if use_point_location else "GRIB"} "
                         f"files cover requested area and time range. Skipping download.\n")
        return

    sys.stderr.write("\tDownloading the ERA5Land from Climate Data Store\n")

    if use_point_location:
        # Use the center point of the requested area for more efficient download
        center_lat = round((lat_range_rounded[0] + lat_range_rounded[1]) / 2, 1)
        center_lon = round((lon_range_rounded[0] + lon_range_rounded[1]) / 2, 1)
        download_lat_range = [center_lat, center_lat]
        download_lon_range = [center_lon, center_lon]
        sys.stderr.write(f"\tUsing point location: lat {center_lat}, lon {center_lon}\n")
    else:
        # Force Catalan region bounding box for downloads
        CATALAN_BOUNDS = {
            'lat_min': 40.5,
            'lat_max': 42.9,
            'lon_min': 0.1,
            'lon_max': 3.4
        }

        # Use Catalan bounds instead of requested bounds for download
        download_lat_range = [CATALAN_BOUNDS['lat_min'], CATALAN_BOUNDS['lat_max']]
        download_lon_range = [CATALAN_BOUNDS['lon_min'], CATALAN_BOUNDS['lon_max']]

        sys.stderr.write(f"\tUsing Catalan region bounds: lat {download_lat_range[0]} to {download_lat_range[1]}, "
                         f"lon {download_lon_range[0]} to {download_lon_range[1]}\n")

    # Connect to the Copernicus Climate Date Store
    c = None

    def load_copernicus():
        try:
            c = cdsapi.Client()  # You'll need a CDS credentials file in ~/.cdsapirc
        except:
            raise sys.stderr.write(
                "\tObtain credentials to Climate Data Store and set them in ~/.cdsapirc file. " \
                "\n\tPlease, follow instructions in: " \
                "\n\thttps://cds.climate.copernicus.eu/how-to-api")
        return c

    os.makedirs(data_dir, exist_ok=True)

    if use_point_location:
        # For point locations, download entire date range in one request
        start_date = '2020-01-01' if (min(ym_range).strftime("%Y-%m-%d")) >= '2020-01-01' else \
            (min(ym_range).strftime("%Y-%m-%d"))
        end_date = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y-%m-%d")

        # Create filename based on full date range
        filename_csv = (f'{data_dir}/'
                        f'{start_date}_'
                        f'{end_date}_'
                        f'{download_lat_range[0]}_{download_lon_range[0]}.csv')
        filename_zip = filename_csv.replace('.csv', '.zip')

        sys.stderr.write(f'\t\t--> Checking point location data for {start_date} to {end_date}\n')

        if not os.path.exists(filename_csv):
            sys.stderr.write(f'\t\t--> Downloading point data for {start_date} to {end_date}...\n')
            if c is None:
                c = load_copernicus()

            dataset = "reanalysis-era5-land-timeseries"
            request = {
                'variable': [x['CDSLongName'] for x in ERA5Land_variables()],
                'location': {
                    'longitude': download_lon_range[0],
                    'latitude': download_lat_range[0]
                },
                'date': [f'{start_date}/{end_date}'],
                'data_format': 'csv'
            }
            c.retrieve(dataset, request).download(filename_zip)
            sys.stderr.write(f'\t\t--> Downloaded {filename_zip}\n')

            # Extract the ZIP file
            sys.stderr.write(f'\t\t--> Extracting {filename_zip}\n')
            tmp_dir = f"{data_dir}/tmp"
            os.makedirs(tmp_dir, exist_ok=True)

            with zipfile.ZipFile(filename_zip, 'r') as zip_ref:
                extracted_files = zip_ref.namelist()
                zip_ref.extractall(tmp_dir)

                # Find the CSV files inside the ZIP
                csv_files = [f for f in extracted_files if f.endswith('.csv')]
                if csv_files:
                    # Join all CSV files by columns into one file
                    combined_df = None

                    for csv_file in csv_files:
                        csv_path = os.path.join(tmp_dir, csv_file)
                        if os.path.exists(csv_path):
                            df_temp = pl.read_csv(csv_path)

                            if combined_df is None:
                                combined_df = df_temp
                            else:
                                # Join by common columns (time, lat, lon)
                                common_cols = ['valid_time'] if 'valid_time' in df_temp.columns else []
                                if 'latitude' in df_temp.columns: common_cols.append('latitude')
                                if 'longitude' in df_temp.columns: common_cols.append('longitude')

                                if common_cols:
                                    combined_df = combined_df.join(df_temp, on=common_cols, how='left')

                    # Save the combined CSV
                    if combined_df is not None:
                        combined_df.write_csv(filename_csv)
                        sys.stderr.write(f'\t\t--> Combined {len(csv_files)} CSV files into {filename_csv}\n')

            # Clean up temporary directory
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            # Clean up the ZIP file
            if os.path.exists(filename_zip):
                os.remove(filename_zip)

        else:
            sys.stderr.write(f'\t\t--> File already exists: {filename_csv}\n')
    else:
        # For area downloads, keep month-by-month approach (GRIB format)
        # Convert datetime objects to YYYYMM integers
        min_ym = min(ym_range).year * 100 + min(ym_range).month
        max_ym = max(ym_range).year * 100 + max(ym_range).month

        ym = min_ym
        while ym <= max_ym:
            year = int(str(ym)[0:4])
            month = int(str(ym)[4:6])

            # Use Catalan bounds for filename
            filename = f'{data_dir}/{year}{month:02}_{download_lat_range[1]}_{download_lon_range[0]}_{download_lat_range[0]}_{download_lon_range[1]}.grib'

            sys.stderr.write(f'\t\t--> Checking {year}{month:02} in Catalan region bounds\n')

            if not os.path.exists(filename):
                sys.stderr.write(f'\t\t--> Downloading {year}{month:02}...\n')
                if c is None:
                    c = load_copernicus()
                # Use regular grid API for area downloads
                dataset = "reanalysis-era5-land"
                request = {
                    'variable': [x['CDSLongName'] for x in ERA5Land_variables()],
                    'year': f'{year}',
                    'month': f'{month:02}',
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [  # Catalan region bounding box
                        download_lat_range[1], download_lon_range[0],  # Upper left point (Lat-lon)
                        download_lat_range[0], download_lon_range[1]  # Lower right point (Lat-lon)
                    ],
                    'data_format': 'grib',
                    'download_format': "unarchived"
                }
                c.retrieve(dataset, request).download(filename)
                sys.stderr.write(f'\t\t--> Downloaded {filename}\n')
            else:
                sys.stderr.write(f'\t\t--> File already exists: {filename}\n')

            month += 1
            if month > 12:
                year += 1
                month = 1
            ym = int(f'{year}{month:02}')

    sys.stderr.write('\t\tSuccess! All data has been downloaded.\n')


# =========================================================================== #
# Municipality weather: closest ERA5-Land cell, aggregated statistics and EPW
# =========================================================================== #
# ERA5-Land has NO data over the sea, so the grid cell nearest a coastal
# municipality centroid (e.g. Barcelona) is often empty. The helpers below load
# the surrounding grid and pick the CLOSEST cell that actually carries data,
# compute heating/cooling degree days and other annual statistics, and export
# an EnergyPlus EPW cached per natural year + cell (so neighbouring
# municipalities that resolve to the same cell reuse the same weather file).

def _haversine_km(lat0, lon0, lat, lon):
    """Great-circle distance (km) from a scalar point to array(s) of points."""
    R = 6371.0088
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    p0, p = np.radians(lat0), np.radians(lat)
    dphi = np.radians(lat - lat0)
    dl = np.radians(lon - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def gdf_centroid_lonlat(gdf, geom_col=None):
    """Return the (lon, lat) WGS84 centroid of a buildings (Geo)DataFrame.

    Uses the bounding-box centre (cheap, and accurate enough to pick a 0.1°
    ERA5-Land cell even for a city the size of Barcelona)."""
    import geopandas as gpd
    if geom_col is None:
        geom_col = gdf.geometry.name if hasattr(gdf, "geometry") else "geometry"
    geom = gdf[geom_col]
    crs = getattr(gdf, "crs", None) or getattr(geom, "crs", None)
    minx, miny, maxx, maxy = geom.total_bounds
    from shapely.geometry import Point
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    pt = gpd.GeoSeries([Point(cx, cy)], crs=crs).to_crs(4326).iloc[0]
    return float(pt.x), float(pt.y)


def closest_station_with_data(df, lat, lon, value_col="airTemperature"):
    """Pick the ERA5-Land cell nearest (lat, lon) that carries valid data.

    Returns ``(station_lat, station_lon, distance_km)``. Cells over the sea
    (all-null ``value_col``) are skipped, so coastal municipalities resolve to
    the closest inland/coastline cell with data."""
    stations = (
        df.filter(pl.col(value_col).is_not_null())
          .group_by(["latitude", "longitude"])
          .agg(pl.len().alias("n"))
          .filter(pl.col("n") > 0)
    )
    if stations.is_empty():
        raise ValueError("no ERA5-Land cell with valid data in the loaded area; "
                         "widen `search_deg` or check the download")
    lats = stations["latitude"].to_numpy()
    lons = stations["longitude"].to_numpy()
    d = _haversine_km(lat, lon, lats, lons)
    i = int(np.argmin(d))
    return float(lats[i]), float(lons[i]), float(d[i])


def station_timeseries(df, station_lat, station_lon):
    """Filter a loaded ERA5-Land frame to a single cell, sorted by time."""
    return (
        df.filter(
            (pl.col("latitude").round(1) == round(station_lat, 1)) &
            (pl.col("longitude").round(1) == round(station_lon, 1))
        ).sort("time")
    )


def weather_statistics(df_station, hdd_base=18.0, cdd_base=21.0, by_month=False):
    """Aggregated annual (and optionally monthly) climate statistics for one
    cell: heating/cooling degree days plus temperature, humidity, wind, solar
    and precipitation summaries.

    Degree days use the daily-mean-temperature method:
    ``HDD = Σ max(0, hdd_base − T_day)``, ``CDD = Σ max(0, T_day − cdd_base)``.
    Radiation is summed as kWh/m² (hourly W/m² ≈ Wh/m²); precipitation as mm."""
    d = df_station.sort("time")

    daily = (
        d.group_by(pl.col("time").dt.date().alias("date"))
         .agg(pl.col("airTemperature").mean().alias("tmean"))
         .sort("date")
    )

    def _hdd(col):
        return pl.when(hdd_base - col > 0).then(hdd_base - col).otherwise(0.0)

    def _cdd(col):
        return pl.when(col - cdd_base > 0).then(col - cdd_base).otherwise(0.0)

    dd = daily.select(
        _hdd(pl.col("tmean")).sum().alias("HDD"),
        _cdd(pl.col("tmean")).sum().alias("CDD"),
    ).row(0, named=True)

    agg = d.select(
        pl.len().alias("n_hours"),
        pl.col("airTemperature").mean().alias("t_mean"),
        pl.col("airTemperature").min().alias("t_min"),
        pl.col("airTemperature").max().alias("t_max"),
        pl.col("relativeHumidity").mean().alias("rh_mean"),
        pl.col("windSpeed").mean().alias("wind_mean"),
        (pl.col("GHI").sum() / 1000.0).alias("ghi_total_kwh_m2"),
        pl.col("totalPrecipitation").sum().alias("precip_total_mm"),
    ).row(0, named=True)

    stats = {
        "hdd_base": float(hdd_base),
        "cdd_base": float(cdd_base),
        "HDD": round(float(dd["HDD"]), 1),
        "CDD": round(float(dd["CDD"]), 1),
        "n_days": int(daily.height),
        "n_hours": int(agg["n_hours"]),
        "t_mean": round(float(agg["t_mean"]), 2),
        "t_min": round(float(agg["t_min"]), 2),
        "t_max": round(float(agg["t_max"]), 2),
        "rh_mean": round(float(agg["rh_mean"]), 1),
        "wind_mean": round(float(agg["wind_mean"]), 2),
        "ghi_total_kwh_m2": round(float(agg["ghi_total_kwh_m2"]), 1),
        "precip_total_mm": round(float(agg["precip_total_mm"]), 1),
    }

    if by_month:
        monthly = (
            d.group_by(pl.col("time").dt.month().alias("month"))
             .agg(
                 pl.col("airTemperature").mean().alias("t_mean"),
                 pl.col("relativeHumidity").mean().alias("rh_mean"),
                 (pl.col("GHI").sum() / 1000.0).alias("ghi_kwh_m2"),
                 pl.col("totalPrecipitation").sum().alias("precip_mm"),
             ).sort("month")
        )
        dmonth = (
            daily.with_columns(pl.col("date").dt.month().alias("month"))
                 .group_by("month")
                 .agg(_hdd(pl.col("tmean")).sum().alias("HDD"),
                      _cdd(pl.col("tmean")).sum().alias("CDD"))
        )
        stats["monthly"] = monthly.join(dmonth, on="month", how="left").sort("month")

    return stats


# --------------------------------------------------------------------------- #
# EPW export
# --------------------------------------------------------------------------- #
_EPW_FLAGS = "?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9"
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday"]


def _f(x, default, nd=1):
    """Format a value for an EPW field, substituting a missing-value default."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return default
    return f"{float(x):.{nd}f}"


def write_epw(df_station, path, *, year, latitude, longitude, elevation=0.0,
              tz_offset=1.0, city="ERA5Land", region="-", country="ESP",
              wmo="999999", source="ERA5-Land", comments=""):
    """Write an EnergyPlus EPW (actual-meteorological-year) for one ERA5-Land
    cell and natural ``year``.

    ERA5-Land timestamps are UTC; they are shifted by ``tz_offset`` hours to
    local standard time. Hour *h* (local, 0-23) maps to EPW hour *h+1* (1-24)
    so each calendar day holds its 24 records. Feb 29 is dropped so the file is
    always 8760 records. Dry-bulb/dew-point/RH/pressure/GHI/DNI/DHI/wind and
    liquid precipitation come from the data; other fields are EPW missing
    codes (EnergyPlus derives horizontal IR / sky from these)."""
    d = df_station.sort("time").with_columns(
        pl.col("time").dt.offset_by(f"{int(tz_offset)}h").alias("ltime")
    )
    d = d.with_columns(
        pl.col("ltime").dt.year().alias("ly"),
        pl.col("ltime").dt.month().alias("lm"),
        pl.col("ltime").dt.day().alias("ld"),
        pl.col("ltime").dt.hour().alias("lh"),
    ).filter(pl.col("ly") == int(year))

    # Lookup keyed by (month, day, epw_hour 1..24).
    lut = {}
    first = None
    for r in d.iter_rows(named=True):
        if first is None:
            first = r
        lut[(r["lm"], r["ld"], r["lh"] + 1)] = r

    lines = []
    start_dow = _DAY_NAMES[dt.date(int(year), 1, 1).weekday()]
    lines.append(
        f"LOCATION,{city},{region},{country},{source},{wmo},"
        f"{latitude:.3f},{longitude:.3f},{tz_offset:.1f},{elevation:.1f}")
    lines.append("DESIGN CONDITIONS,0")
    lines.append("TYPICAL/EXTREME PERIODS,0")
    lines.append("GROUND TEMPERATURES,0")
    lines.append("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0")
    lines.append(f"COMMENTS 1,{comments or ('Generated by hypercadaster_ES from ERA5-Land')}")
    lines.append("COMMENTS 2,Actual meteorological year; EnergyPlus EPW")
    lines.append(f"DATA PERIODS,1,1,Data,{start_dow},1/1,12/31")

    last = first   # back-fill any leading gap (e.g. local 00:00 Jan 1)
    for m in range(1, 13):
        for day in range(1, 32):
            try:
                dt.date(int(year), m, day)
            except ValueError:
                continue
            if m == 2 and day == 29:      # always emit a 8760 calendar
                continue
            for h in range(1, 25):
                r = lut.get((m, day, h), last)
                last = r if r is not None else last
                rr = r or {}
                fields = [
                    str(year), str(m), str(day), str(h), "0", _EPW_FLAGS,
                    _f(rr.get("airTemperature"), "99.9", 1),       # 7 dry bulb
                    _f(rr.get("dewAirTemperature"), "99.9", 1),    # 8 dew point
                    _f(rr.get("relativeHumidity"), "999", 0),      # 9 RH
                    _f(rr.get("surfacePressure"), "999999", 0),    # 10 pressure Pa
                    "9999", "9999", "9999",                        # 11-13 ext/IR
                    _f(rr.get("GHI"), "9999", 0),                  # 14 GHI
                    _f(rr.get("DNI"), "9999", 0),                  # 15 DNI
                    _f(rr.get("DHI"), "9999", 0),                  # 16 DHI
                    "999999", "999999", "999999", "9999",          # 17-20 illum
                    _f(rr.get("windDirection"), "999", 0),         # 21 wind dir
                    _f(rr.get("windSpeed"), "999", 1),             # 22 wind spd
                    "99", "99", "9999", "99999", "9", "999999999", # 23-28
                    "999", "0.999", "999", "99", "999",            # 29-33
                    _f(rr.get("totalPrecipitation"), "0", 1),      # 34 liq precip mm
                    "99",                                          # 35 liq precip qty
                ]
                lines.append(",".join(fields))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# High-level orchestrators
# --------------------------------------------------------------------------- #
def get_municipality_weather(era5_dir, lat, lon, year, *, search_deg=0.6,
                             tz_local="Europe/Madrid", update=False,
                             mode="averaged", hdd_base=18.0, cdd_base=21.0,
                             by_month=False):
    """Load ERA5-Land around (lat, lon), resolve the closest cell with data and
    return ``(df_station, stats, (station_lat, station_lon, distance_km))``."""
    ym_range = [dt.datetime(int(year), 1, 1), dt.datetime(int(year), 12, 31)]
    df = load_ERA5Land(
        data_dir=era5_dir,
        lat_range=[lat - search_deg, lat + search_deg],
        lon_range=[lon - search_deg, lon + search_deg],
        ym_range=ym_range, tz_local=tz_local, update=update,
        use_point_location=False, mode=mode)
    slat, slon, dist = closest_station_with_data(df, lat, lon)
    dst = station_timeseries(df, slat, slon)
    stats = weather_statistics(dst, hdd_base=hdd_base, cdd_base=cdd_base,
                               by_month=by_month)
    stats["station_latitude"] = round(slat, 3)
    stats["station_longitude"] = round(slon, 3)
    stats["distance_km"] = round(dist, 2)
    stats["year"] = int(year)
    return dst, stats, (slat, slon, dist)


def municipality_epw(era5_dir, epw_dir, lat, lon, year, *, elevation=0.0,
                     tz_offset=1.0, city="ERA5Land", search_deg=0.6,
                     tz_local="Europe/Madrid", update=False, mode="averaged",
                     return_stats=False):
    """Return the path to an EPW for the natural ``year`` at the ERA5-Land cell
    closest to (lat, lon), building and caching it on first use.

    The EPW is cached as ``ERA5Land_{cellLat}_{cellLon}_{year}.epw`` keyed by
    the RESOLVED cell, so different municipalities sharing the nearest cell
    reuse the same file. Returns ``path`` (or ``(path, stats)``)."""
    os.makedirs(epw_dir, exist_ok=True)
    dst, stats, (slat, slon, _dist) = get_municipality_weather(
        era5_dir, lat, lon, year, search_deg=search_deg, tz_local=tz_local,
        update=update, mode=mode)
    epw_path = os.path.join(epw_dir, f"ERA5Land_{slat:.1f}_{slon:.1f}_{int(year)}.epw")
    if (not os.path.exists(epw_path)) or update:
        write_epw(dst, epw_path, year=int(year), latitude=slat, longitude=slon,
                  elevation=elevation, tz_offset=tz_offset, city=city)
    return (epw_path, stats) if return_stats else epw_path
