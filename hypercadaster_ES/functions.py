"""Main functions module for hypercadaster_ES.

This module provides the primary entry points for downloading and merging
Spanish cadastral data with external geographic datasets.

Main functions:
    - download(): Downloads cadastral and related geographic data
    - merge(): Merges all downloaded data into a unified GeoDataFrame
"""

import os
import sys
import geopandas as gpd
from zipfile import ZipFile, BadZipFile

from hypercadaster_ES import mergers
from hypercadaster_ES import utils
from hypercadaster_ES import downloaders
from hypercadaster_ES import social


def _as_code_list(value, width):
    """Normalise a code selector to a list of zero-padded string codes.

    Accepts a single value or a list, of ints or strings. Every purely numeric
    code is left-zero-padded to ``width`` (province codes are 2-digit, INE and
    cadaster codes 5-digit), so ``25``/``"25"`` → ``"25"``, ``8``/``"8"`` →
    ``"08"`` and ``8019``/``"8019"`` → ``"08019"`` — the leading-zero string form
    the matching logic expects. Non-numeric codes and codes already at/over
    ``width`` are kept verbatim (``zfill`` never truncates). Returns ``None``
    unchanged.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        value = [value]
    return [str(v).zfill(width) if str(v).isdigit() else str(v) for v in value]


def download(wd, province_codes=None, ine_codes=None, cadaster_codes=None,
             neighborhood_layer=True, postal_code_layer=True, census_layer=True,
             elevation_layer=True, open_data_layers=True, force=False,
             building_parts_inference=False):
    """Download every requested data layer for a geographic area into ``wd``.

    The area is selected with any combination of ``province_codes`` (2-digit),
    ``ine_codes`` (5-digit INE municipality) or ``cadaster_codes`` (cadastral
    municipality); the three are resolved to a de-duplicated set of cadastral
    codes. At least one selector must be provided. Each argument accepts a single
    string or a list.

    Parameters
    ----------
    wd : str
        Working directory. The directory tree is created if missing.
    province_codes, ine_codes, cadaster_codes : str | list[str], optional
        Area selectors (see above).
    neighborhood_layer, open_data_layers : bool, default True
        Barcelona-only layers (neighborhood/district boundaries and the
        commercial/street open-data resources); ignored for other municipalities.
    postal_code_layer : bool, default True
        Download Spanish postal-code boundaries (also required by the DEM layer).
    census_layer : bool, default True
        Download INE census-tract boundaries.
    elevation_layer : bool, default True
        Download the Digital Elevation Model raster covering the area.
    force : bool, default False
        Remove ``wd`` before downloading to force a clean re-download.
    building_parts_inference : bool, default False
        When True, also download neighbouring municipalities (200 m buffer) so
        edge buildings have complete context for later inference in ``merge()``.

    Returns
    -------
    None
        Files are written under ``wd``; nothing is returned.
    """
    if force:
        try:
            os.removedirs(wd)
        except FileNotFoundError:
            pass
    utils.create_dirs(data_dir=wd)

    # Convert inputs to a list of zero-padded string codes (accepts ints too)
    province_codes = _as_code_list(province_codes, 2)
    ine_codes = _as_code_list(ine_codes, 5)
    cadaster_codes = _as_code_list(cadaster_codes, 5)

    # Filter which geographical area to download
    all_cadaster_codes = []

    if province_codes is None and ine_codes is None and cadaster_codes is None:
        raise ValueError("At least one of the arguments must be provided: province codes (province_codes), "
                         "municipality INE codes (ine_codes), or cadaster codes (cadaster_codes)")

    # Add cadaster codes from province_codes
    if province_codes is not None:
        municipalities = utils.list_municipalities(province_codes=province_codes, echo=False)
        province_cadaster_codes = [item['name'].split("-")[0] for item in municipalities]
        all_cadaster_codes.extend(province_cadaster_codes)

    # Add cadaster codes from ine_codes
    if ine_codes is not None:
        ine_cadaster_codes = utils.ine_to_cadaster_codes(ine_codes)
        all_cadaster_codes.extend(ine_cadaster_codes)

    # Add directly provided cadaster codes
    if cadaster_codes is not None:
        all_cadaster_codes.extend(cadaster_codes)

    # Remove duplicates and set final cadaster_codes
    cadaster_codes = list(set(all_cadaster_codes))

    if not cadaster_codes:
        raise ValueError(
            "No municipalities resolved from the given selectors "
            f"(province_codes={province_codes}, ine_codes={ine_codes}). "
            "Province codes must be 2-digit (e.g. '25' or 25) and INE/cadaster "
            "codes 5-digit; check the values."
        )

    # Generate derived codes
    ine_codes = utils.cadaster_to_ine_codes(cadaster_codes)
    province_codes = list(set([code[:2] for code in ine_codes]))

    # Census tracts (download early if needed for neighboring municipalities)
    if census_layer:
        downloaders.download_census_tracts(census_tracts_dir=utils.census_tracts_dir_(wd), year=2022)

    # If building parts inference is enabled, include neighboring municipalities
    if building_parts_inference:
        extended_cadaster_codes = utils.get_neighboring_cadaster_codes(
            cadaster_codes, 
            utils.census_tracts_dir_(wd),
            buffer_distance=200,  # 200m buffer to find neighbors
            include_original=True
        )
        
        # Download cadaster data for extended area
        downloaders.cadaster_downloader(cadaster_dir=utils.cadaster_dir_(wd), cadaster_codes=extended_cadaster_codes)

        # Update province codes for other downloads
        ine_codes = utils.cadaster_to_ine_codes(extended_cadaster_codes)
        province_codes = list(set([code[:2] for code in ine_codes]))
    else:
        # Download the cadaster datasets of that area
        downloaders.cadaster_downloader(cadaster_dir=utils.cadaster_dir_(wd), cadaster_codes=cadaster_codes)

    # CartoCiudad addresses (one GeoPackage per province), the address source that
    # replaces the deprecated INSPIRE Address feed and the Barcelona carrerer.
    downloaders.cartociudad_downloader(cartociudad_dir=utils.cartociudad_dir_(wd),
                                       province_codes=province_codes)

    # Districts and neighborhoods definition are only available for the city of Barcelona
    if '08019' in ine_codes or '08900' in cadaster_codes and neighborhood_layer:
        downloaders.download_file(dir=utils.districts_dir_(wd),
                                  url="https://opendata-ajuntament.barcelona.cat/data/dataset/808daafa-d9ce-48c0-"
                                      "925a-fa5afdb1ed41/resource/576bc645-9481-4bc4-b8bf-f5972c20df3f/download",
                                  file="districts.csv")
        downloaders.download_file(dir=utils.neighborhoods_dir_(wd),
                                  url="https://opendata-ajuntament.barcelona.cat/data/dataset/808daafa-d9ce-48c0-"
                                      "925a-fa5afdb1ed41/resource/b21fa550-56ea-4f4c-9adc-b8009381896e/download",
                                  file="neighbourhoods.csv")

    # Postal codes
    if postal_code_layer or elevation_layer:
        downloaders.download_postal_codes(postal_codes_dir=utils.postal_codes_dir_(wd), province_codes=province_codes)

    # Digital Elevation Model layer
    if elevation_layer:
        postal_codes_gdf = gpd.read_file(f"{utils.postal_codes_dir_(wd)}/postal_codes.geojson")
        postal_codes_gdf = gpd.GeoDataFrame(postal_codes_gdf, geometry='geometry', crs='EPSG:4326')
        bbox_postal_code = utils.get_bbox(gdf = postal_codes_gdf)
        downloaders.download_DEM_raster(raster_dir=utils.DEM_raster_dir_(wd), bbox=bbox_postal_code)

    # Census tracts and districts (if not already downloaded)
    if census_layer and not building_parts_inference:
        downloaders.download_census_tracts(census_tracts_dir=utils.census_tracts_dir_(wd), year=2022)

    # OpenDataBarcelona
    if '08019' in ine_codes or '08900' in cadaster_codes and open_data_layers:
        downloaders.download_file(dir=utils.open_data_dir_(wd),
                                  url="https://opendata-ajuntament.barcelona.cat/data/dataset/3dc277bf-ff89-4b49-8f"
                                      "29-48a1122bb813/resource/2e123ea9-1819-46cf-a545-be61151fa97d/download",
                                  file="barcelona_establishments.csv")
        downloaders.download_file(dir=utils.open_data_dir_(wd),
                                  url="https://opendata-ajuntament.barcelona.cat/data/dataset/fe177673-0f83-42e7-b3"
                                      "5a-ddea901be8bc/resource/99764d55-b1be-4281-b822-4277442cc721/download/22093"
                                      "0_censcomercialbcn_opendata_2022_v10_mod.csv",
                                  file="barcelona_ground_premises.csv")
        # DEPRECATED: the Barcelona carrerer (street/address register) download is
        # superseded by CartoCiudad (cartociudad_downloader above), which provides
        # nationwide address points. Kept commented for reference.
        # _od = utils.open_data_dir_(wd)
        # _carrerer_gpkg = f"{_od}/barcelona_carrerer.gpkg"
        # if not os.path.exists(_carrerer_gpkg):
        #     _carrerer_dl = f"{_od}/barcelona_carrerer.zip"
        #     downloaders.download_file(
        #         dir=_od,
        #         url="https://opendata-ajuntament.barcelona.cat/data/dataset/6b5cfa7b-1d8d-45f0-990a-"
        #             "d1844d43ffd1/resource/26c6be33-44f5-4596-8a29-7ac152546ca7/download",
        #         file="barcelona_carrerer.zip")
        #     if os.path.exists(_carrerer_dl):
        #         try:
        #             with ZipFile(_carrerer_dl, 'r') as zip:
        #                 zip.extractall(_od)
        #             os.rename(f"{_od}/Adreces_elementals.gpkg", _carrerer_gpkg)
        #         except BadZipFile:
        #             # Not a zip — the download is the GeoPackage itself.
        #             os.rename(_carrerer_dl, _carrerer_gpkg)
        #         finally:
        #             if os.path.exists(_carrerer_dl):
        #                 os.remove(_carrerer_dl)
        #     else:
        #         sys.stderr.write(
        #             "Warning: could not download barcelona_carrerer; "
        #             "open-data street register will be unavailable\n")

def merge(wd, province_codes=None, ine_codes=None, cadaster_codes=None,
          neighborhood_layer=True, postal_code_layer=True, census_layer=True, elevations_layer=True,
          open_data_layers=True, building_parts_inference=False, building_parts_plots=False,
          plot_zones_ratio=0.01, use_CAT_files=False, CAT_files_rel_dir="CAT_files",
          pois_layer=False, sociodemographic_layer=False,
          social_ES_layer=False, social_ES_years=5, social_ES_datasets=None,
          weather_layer=False, weather_years=5, weather_indicators=None,
          weather_hdd_base=18.0, weather_cdd_base=21.0, weather_bbox=None,
          floor_height=3.0, ground_floor_height=5.0, num_workers=-1,
          force_update=False, cleanup_shards=True, verbose="simple"):
    """Merge the downloaded cadastral and contextual layers into one GeoDataFrame.

    This is the primary entry point. It reads the data that :func:`download`
    placed under ``wd``, joins the cadastral buildings/addresses/parcels with the
    requested contextual layers (census, postal codes, neighborhoods, elevation,
    sociodemographics, POIs), optionally runs the geometric **building-parts
    inference**, persists everything to the central DuckDB
    (``{wd}/results/building_results.duckdb``) and returns the assembled frame.

    The layer/selector arguments should mirror the ones passed to
    :func:`download` — ``merge`` reads local files and does not download; a layer
    enabled here but not downloaded will fail or be skipped.

    Geographic selection (``province_codes`` / ``ine_codes`` / ``cadaster_codes``)
    works exactly as in :func:`download`: any combination is accepted, each takes
    a string or list, and at least one is required.

    Parameters
    ----------
    wd : str
        Working directory previously populated by :func:`download`. The results
        DuckDB is written under ``{wd}/results``.
    province_codes, ine_codes, cadaster_codes : str | list[str], optional
        Area selectors, resolved and de-duplicated to cadastral codes. At least
        one must be given.
    neighborhood_layer : bool, default True
        Join Barcelona neighborhood/district boundaries (``neighborhood_name``,
        ``district_name``). **Barcelona-only** — set ``False`` for any other
        municipality or the join raises (no neighbourhoods file).
    postal_code_layer : bool, default True
        Join the Spanish postal code (``postal_code``) each building falls in.
    census_layer : bool, default True
        Join INE census-tract identifiers (``section_code``, ``district_code``).
        Required for ``sociodemographic_layer``.
    elevations_layer : bool, default True
        Sample the DEM to add ground ``elevation`` (m). Note the trailing "s":
        this is ``elevations_layer`` here, whereas :func:`download` uses
        ``elevation_layer``.
    open_data_layers : bool, default True
        Enrich addresses and add ground-floor commercial-premises indicators from
        Barcelona open data. **Barcelona-only**; set ``False`` elsewhere.
    building_parts_inference : bool, default False
        Run the geometric inference engine (floor footprints, adiabatic/party/
        main-façade walls, patios, orientation, street relationships, shadows),
        adding the ``br__*`` columns and populating ``br_results`` / ``sbr_results``.
        Computationally and memory intensive, and **limited to a single
        municipality** (raises if more than one is selected). Heavy BLOB columns
        (shadows/terrain/wall segments) are excluded from the returned frame —
        fetch them with :func:`hypercadaster_ES.utils.get_heavy_columns_for_building`.
    building_parts_plots : bool, default False
        Emit per-zone diagnostic plots during inference. Single municipality only.
    plot_zones_ratio : float, default 0.01
        Fraction (0–1) of cadastral zones to plot when ``building_parts_plots``
        is enabled (0.01 = 1 % of zones).
    use_CAT_files : bool, default False
        Use the official cadastral **CAT** files for detailed per-space use-type
        classification and areas. The CAT files must be present under
        ``{wd}/{CAT_files_rel_dir}`` (downloaded manually). Adds the
        ``building_area_*`` and space-breakdown columns.
    CAT_files_rel_dir : str, default "CAT_files"
        Directory (relative to ``wd``) holding the CAT files.
    pois_layer : bool, default False
        Attach OpenStreetMap POI indicators per building for the groups
        ``amenity``, ``emergency``, ``healthcare``, ``leisure``, ``office``,
        ``shop`` and ``tourism``. For each group (and subgroup) it adds, with
        ``{p}`` = ``poi__{group}[_{subgroup}]`` (``poi__`` namespace prefix,
        snake_case group/subgroup):

        - ``{p}_count`` — POIs located on the building footprint
        - ``{p}_ids``   — their OSM identifiers
        - ``{p}_100m`` / ``{p}_250m`` / ``{p}_500m`` / ``{p}_1000m`` / ``{p}_1500m``
          — POI counts within that radius of the building centroid

        Raw OSM query results are cached under the ``os_maps`` directory so
        re-runs skip the network.
    sociodemographic_layer : bool, default False
        Join INE 2021 census-section sociodemographic indicators (mean age,
        share over 64 / under 16, average household size, total households) as
        ``br__population_*`` / ``br__*_household*`` columns. Requires
        ``census_layer=True`` (needs ``section_code``); degrades gracefully to a
        no-op if the source is unavailable.
    social_ES_layer : bool, default False
        Join INE socioeconomic indicators from the ``social_ES`` package onto
        every building, each dataset at its finest geographic level (census
        section → district → municipality → province, using the codes added by
        ``census_layer``). Adds ``so__{indicator}_{year}`` columns (``so__``
        namespace prefix, snake_case indicator, year as suffix)
        for the most recent ``social_ES_years`` years of the income atlas,
        population census, education/employment census, rental & housing price
        indices, and district electricity consumption. Runs **before** the POI
        join. Requires ``census_layer=True``; ``social_ES`` caches the whole-Spain
        source files under ``{wd}/INE`` on first use (a slow one-off download).
        See :mod:`hypercadaster_ES.social` for the curated indicator set.
    social_ES_years : int, default 5
        Number of most-recent years to keep per ``social_ES`` dataset.
    social_ES_datasets : sequence[str], optional
        Subset of the ``social_ES`` datasets to include; default is all of
        :data:`hypercadaster_ES.social.ALL_DATASETS` (``"atlas"``,
        ``"population"``, ``"education"``, ``"rental"``, ``"price"``,
        ``"electricity"``), plus the per-building **Essential Characteristics**
        downscaling (``"essential_characteristics"`` →
        :data:`hypercadaster_ES.social.ESSENTIAL_CHARACTERISTICS`, adding
        ``ec__*`` proportion columns) which is additionally gated on
        ``use_CAT_files=True``. That dataset is included when
        ``social_ES_datasets`` is ``None`` (and ``use_CAT_files=True``) or when it
        explicitly lists ``"essential_characteristics"``; requesting it without
        ``use_CAT_files`` skips it with a warning.
    weather_layer : bool, default False
        Join annually-aggregated ERA5-Land climate indicators onto every
        building. Each building is snapped to the nearest 0.1° ERA5-Land cell
        with data (coastal buildings fall back to the closest inland cell), and
        the per-cell annual statistics are added as ``we__{indicator}_{year}``
        columns (``we__`` namespace prefix, snake_case indicator) for each
        requested year: ``hdd`` / ``cdd`` (heating/cooling
        degree days), ``precipitation`` (mm), ``ghi`` (kWh/m²),
        ``air_temperature_mean`` / ``_min`` / ``_max`` (°C),
        ``relative_humidity_mean`` (%) and ``wind_speed_mean`` (m/s). Uses
        :mod:`hypercadaster_ES.weather`; the ERA5-Land data is cached under
        ``{wd}/weather/era5land`` on first use (a one-off download requiring a
        configured ``~/.cdsapirc``). For a **single municipality** only the one
        ERA5-Land point nearest the municipality centroid is downloaded (the
        timeseries endpoint); for several municipalities the province-wide grid
        is used. Runs **before** the POI join.
    weather_years : int | sequence[int], default 5
        Number of most-recent **complete** calendar years to include (the last
        complete year is the previous one relative to today, since ERA5-Land
        trails real time), or an explicit list of years (e.g. ``[2021, 2022]``).
    weather_indicators : sequence[str], optional
        Subset of the weather indicator stems to keep (see
        :data:`hypercadaster_ES.weather.WEATHER_INDICATORS`); default is all.
    weather_hdd_base : float, default 18.0
        Base temperature (°C) for the heating-degree-day sum.
    weather_cdd_base : float, default 21.0
        Base temperature (°C) for the cooling-degree-day sum.
    weather_bbox : sequence[float], optional
        Predefined WGS84 bounding box ``[min_lon, min_lat, max_lon, max_lat]``
        (same order as :func:`hypercadaster_ES.utils.get_ine_codes_from_bounding_box`)
        over which to download the ERA5-Land weather grid, instead of deriving
        the extent from the buildings. When given, the weather layer always uses
        the grid (the single-municipality point download is skipped).
    floor_height : float, default 3.0
        Assumed above-ground storey height (m) used by the inference geometry.
    ground_floor_height : float, default 5.0
        Assumed ground-floor height (m) used by the inference geometry.
    num_workers : int, default -1
        Parallel worker processes for the inference (``-1`` = all cores).
    force_update : bool, default False
        Recompute and overwrite cached DuckDB tables instead of reusing existing
        (already-processed zones are otherwise skipped on re-runs). Note this does
        **not** re-read/re-aggregate the CAT files: with ``use_CAT_files=True`` the
        parsed per-space table (``cat_building_spaces``) and the expensive
        per-building ``br__*`` aggregates (``cat_building_attributes``, the
        dominant cost of a CAT merge) are cached in the central DuckDB keyed by
        ``cadaster_code`` and reused on every run. Like the raw ``.CAT``→parquet
        cache, they are pure functions of the static CAT files. To force a CAT
        re-parse/re-aggregate (e.g. after replacing the CAT files), clear them
        with :func:`hypercadaster_ES.utils.clear_cat_cache`.
    cleanup_shards : bool, default True
        Delete the per-zone shard DuckDB files after ingesting them into the
        central database.
    verbose : str, default "simple"
        Logging verbosity. ``"simple"`` suppresses the per-stage
        ``[stage +… | RSS …]`` markers and per-municipality progress logs
        (``assign_building_zones``/``join_other_constructions``); ``"full"``
        shows them.

    Returns
    -------
    geopandas.GeoDataFrame
        The merged building-level data (CRS ``EPSG:25831`` for the mainland).
        Contents depend on the flags above: contextual columns for the enabled
        layers, ``br__*`` inference columns when ``building_parts_inference`` is
        set (minus the heavy BLOB columns). Addresses are always compacted to one
        row per parcel with list-valued address columns (e.g. ``street_name`` is a
        list of the parcel's addresses). The same data is also persisted to the
        ``buildings`` table of the central DuckDB.

    Raises
    ------
    ValueError
        If no geographic selector is given, or if ``building_parts_inference`` /
        ``building_parts_plots`` is requested for more than one municipality.

    See Also
    --------
    download : Fetch the layers this function consumes.
    hypercadaster_ES.utils.get_heavy_columns_for_building :
        Retrieve the heavy shadow/terrain columns excluded from the output.

    Examples
    --------
    >>> import hypercadaster_ES as hc
    >>> hc.download("./data", cadaster_codes=["08900"])
    >>> gdf = hc.merge("./data", cadaster_codes=["08900"],
    ...                 building_parts_inference=True)
    """
    # Logging verbosity: "simple" (default) hides the per-stage [stage | RSS]
    # markers and per-municipality progress logs; "full" shows them.
    utils.set_verbose(verbose)

    # Convert inputs to a list of zero-padded string codes (accepts ints too)
    province_codes = _as_code_list(province_codes, 2)
    ine_codes = _as_code_list(ine_codes, 5)
    cadaster_codes = _as_code_list(cadaster_codes, 5)

    # Filter which geographical area to download
    all_cadaster_codes = []

    if province_codes is None and ine_codes is None and cadaster_codes is None:
        raise ValueError("At least one of the arguments must be provided: province codes (province_codes), "
                         "municipality INE codes (ine_codes), or cadaster codes (cadaster_codes)")

    # Add cadaster codes from province_codes
    if province_codes is not None:
        municipalities = utils.list_municipalities(province_codes=province_codes, echo=False)
        province_cadaster_codes = [item['name'].split("-")[0] for item in municipalities]
        all_cadaster_codes.extend(province_cadaster_codes)

    # Add cadaster codes from ine_codes
    if ine_codes is not None:
        ine_cadaster_codes = utils.ine_to_cadaster_codes(ine_codes)
        all_cadaster_codes.extend(ine_cadaster_codes)

    # Add directly provided cadaster codes
    if cadaster_codes is not None:
        all_cadaster_codes.extend(cadaster_codes)

    # Remove duplicates and set final cadaster_codes
    cadaster_codes = list(set(all_cadaster_codes))

    if not cadaster_codes:
        raise ValueError(
            "No municipalities resolved from the given selectors "
            f"(province_codes={province_codes}, ine_codes={ine_codes}). "
            "Province codes must be 2-digit (e.g. '25' or 25) and INE/cadaster "
            "codes 5-digit; check the values."
        )

    # Validate building analysis constraints
    if (building_parts_inference or building_parts_plots) and len(cadaster_codes) > 1:
        raise ValueError(
            f"Building parts inference and plots are computationally intensive and can only be enabled "
            f"for single municipality processing. Found {len(cadaster_codes)} municipalities: {cadaster_codes}. "
            f"Please process municipalities individually when using building_parts_inference=True or "
            f"building_parts_plots=True."
        )

    utils.stage_log("merge(): starting join_cadaster_data (addresses + building inference)")
    gdf = mergers.join_cadaster_data(
        cadaster_dir=utils.cadaster_dir_(wd),
        census_tracts_dir=utils.census_tracts_dir_(wd),
        DEM_raster_dir=utils.DEM_raster_dir_(wd),
        cadaster_codes=cadaster_codes,
        results_dir=utils.results_dir_(wd),
        open_street_dir=utils.open_street_dir_(wd),
        cartociudad_dir=utils.cartociudad_dir_(wd),
        building_parts_inference=building_parts_inference,
        use_CAT_files=use_CAT_files,
        building_parts_plots=building_parts_plots,
        plot_zones_ratio=plot_zones_ratio,
        open_data_layers=open_data_layers,
        open_data_layers_dir=utils.open_data_dir_(wd),
        CAT_files_dir=f"{wd}/{CAT_files_rel_dir}",
        floor_height=floor_height,
        ground_floor_height=ground_floor_height,
        num_workers=num_workers,
        force_update=force_update,
        cleanup_shards=cleanup_shards,
    )

    utils.stage_log(f"merge(): join_cadaster_data done -> {len(gdf)} rows")
    if census_layer:
        with utils.stage_timer("merge(): join_by_census_tracts"):
            gdf = mergers.join_by_census_tracts(
                gdf = gdf,
                census_tract_dir=utils.census_tracts_dir_(wd))
    if sociodemographic_layer:
        with utils.stage_timer("merge(): join_sociodemographics"):
            gdf = mergers.join_sociodemographics(gdf=gdf, wd=wd)
    if neighborhood_layer:
        with utils.stage_timer("merge(): join_by_neighbourhoods"):
            gdf = mergers.join_by_neighbourhoods(
                gdf = gdf,
                neighbourhoods_dir=utils.neighborhoods_dir_(wd))
    if postal_code_layer:
        with utils.stage_timer("merge(): join_by_postal_codes"):
            gdf = mergers.join_by_postal_codes(
                gdf = gdf,
                postal_codes_dir=utils.postal_codes_dir_(wd))
    if elevations_layer:
        with utils.stage_timer("merge(): join_DEM_raster"):
            gdf = mergers.join_DEM_raster(
                gdf = gdf,
                raster_dir = utils.DEM_raster_dir_(wd))
    if "index" in gdf.columns:
        gdf.drop("index", axis=1, inplace=True)

    # columns that have dicts, lists, or bytearrays (BLOB geometry data)
    with utils.stage_timer("merge(): detect unhashable columns + drop_duplicates"):
        cols_with_unhashable = [c for c in gdf.columns
                           if gdf[c].apply(lambda v: isinstance(v, (dict, list, bytearray))).any()]

        subset = [c for c in gdf.columns if c not in cols_with_unhashable + ['geometry']]
        gdf = gdf.drop_duplicates(subset=subset, keep='first')

    # Addresses are always compacted to one row per parcel (building_reference),
    # with the per-address columns collapsed into lists of possible addresses.
    if 'building_reference' in gdf.columns:
        building_counts = gdf['building_reference'].value_counts()
        buildings_with_multiple_addresses = building_counts[building_counts > 1]
        with utils.stage_timer(
            f"merge(): compact_addresses "
            f"({len(buildings_with_multiple_addresses)} multi-address buildings)"
        ):
            gdf = utils.compact_addresses(
                gdf,
                building_reference_col='building_reference'
            )

        # Match each building space's CAT postal address to the closest of the
        # parcel's compacted addresses (no-op when inference is disabled): keeps the
        # CAT value as br__building_spaces_postal_address_from_CAT_file, replaces
        # br__building_spaces_postal_address with the matched parcel address, and
        # sets br__building_spaces_postal_address_location to its point.
        gdf = utils.match_building_space_postal_addresses(gdf)

        # Public-facing names for the compacted postal-address columns.
        gdf = gdf.rename(columns={
            'street_full_labels': 'postal_addresses_with_extensions',
            'street_full_clean': 'postal_addresses_clean',
            'address_location': 'postal_addresses_location',
        })

    if social_ES_layer:
        with utils.stage_timer("merge(): join_social_ES"):
            gdf = social.join_social_ES(
                gdf=gdf,
                wd=wd,
                n_years=social_ES_years,
                datasets=social_ES_datasets,
                use_CAT_files=use_CAT_files,
                verbose=(verbose == "full"),
            )

    if weather_layer:
        with utils.stage_timer("merge(): join_weather"):
            from hypercadaster_ES import weather
            gdf = weather.join_weather(
                gdf=gdf,
                era5_dir=utils.weather_era5_dir_(wd),
                years=weather_years,
                indicators=weather_indicators,
                hdd_base=weather_hdd_base,
                cdd_base=weather_cdd_base,
                update=force_update,
                bbox=weather_bbox,
                # Single municipality: fetch only the nearest ERA5-Land point
                # (timeseries endpoint) instead of the whole province grid. Ignored
                # when weather_bbox is set (a box always uses the grid).
                use_point_location=(len(cadaster_codes) == 1),
                verbose=(verbose == "full"),
            )

    if pois_layer:
        with utils.stage_timer("merge(): join_pois"):
            gdf = mergers.join_pois(
                gdf=gdf,
                open_street_dir=utils.open_street_dir_(wd),
                cadaster_dir=utils.cadaster_dir_(wd),
                cadaster_codes=cadaster_codes,
            )

    utils.stage_log(f"merge(): complete -> returning {len(gdf)} rows")
    return gdf

