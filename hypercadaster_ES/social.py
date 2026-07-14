"""Socioeconomic enrichment from the ``social_ES`` library (INE datasets).

This module attaches INE socioeconomic / demographic indicators to every
building, joining each source dataset **at its own finest geographic level**
(census section → district → municipality → province) using the codes that
:func:`hypercadaster_ES.mergers.join_by_census_tracts` already added to the
frame (``section_code`` = CUSEC, ``district_code`` = CUDIS). Municipality
(``section_code[:5]``) and province (``section_code[:2]``) codes are derived.

The public entry point is :func:`join_social_ES`, wired into
:func:`hypercadaster_ES.merge` through the ``social_ES_layer`` flag and executed
**before the POI join**. Output columns follow the ``so__{indicator}_{year}``
convention (``so__`` namespace prefix, snake_case indicator, year as a suffix),
e.g. ``so__net_income_per_household_2020``. Only the most recent ``n_years``
available per dataset are emitted.

Data comes from the ``social_ES`` package (EUPL-1.2), which downloads and caches
the whole-Spain source files under ``{wd}/INE/...`` on first use; subsequent runs
read from that cache.
"""

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from hypercadaster_ES import utils


# --- Curated indicator maps: {source column in social_ES -> clean output name} ---

# HouseholdIncomeDistributionAtlas (census-section level, years 2015-2023).
# The headline ~20 indicators (no sex/age/nationality breakdowns).
ATLAS_INDICATORS = {
    "Average household net income": "net_income_per_household",
    "Average household income": "gross_income_per_household",
    "Average net income per person": "net_income_per_person",
    "Average per person gross income": "gross_income_per_person",
    "Average income by unit of consumption": "net_income_per_consumption_unit",
    "Median income by unit of consumption": "median_income_per_consumption_unit",
    "Source of income: wages": "salary_income_per_person",
    "Source of income: pensions": "pension_income_per_person",
    "Source of income: unemployment benefits": "unemployment_income_per_person",
    "Source of income: other benefits": "other_benefits_income_per_person",
    "Source of income: other income": "other_income_per_person",
    "Gini Index": "gini_index",
    "Income distribution P80/P20": "income_p80_p20",
    "Average age of the population": "average_age",
    "Average size of the household": "average_household_size",
    "Percentage of Spanish population": "pct_spanish",
    "Percentage of single-person households": "pct_single_person_households",
    "Percentage of the population over the age of 65": "pct_over_65",
    "Percentage of the population under the age of 18": "pct_under_18",
    "Population": "population",
}

# EducationAndEmploymentCensus, mode="relative" (already percentages), section
# level, years 2021-2024.
EDU_INDICATORS = {
    "Population ~ Educational level:Primary education or below": "pct_edu_primary_or_below",
    "Population ~ Educational level:Lower secondary education or equivalent": "pct_edu_lower_secondary",
    "Population ~ Educational level:Upper secondary and post-secondary non-tertiary education": "pct_edu_upper_secondary",
    "Population ~ Educational level:Tertiary education": "pct_edu_tertiary",
    "Population ~ Labour force status:Employed": "pct_employed",
    "Population ~ Labour force status:Unemployed": "pct_unemployed",
    "Population ~ Labour force status:Student": "pct_student",
    "Population ~ Labour force status:Recipient of disability, retirement, or early retirement pension": "pct_pensioner",
    "Population ~ Labour force status:Other inactive situation": "pct_other_inactive",
}

# HouseholdsRentalPriceIndex, municipality level, years 2011-2024.
RENTAL_INDICATORS = {"Household rental index": "rental_price_index"}

# HouseholdsPriceIndex, province level, quarterly -> averaged to annual.
PRICE_INDICATORS = {
    "Whole housing market": "housing_price_index_total",
    "First-hand housing market": "housing_price_index_new",
    "Second-hand housing market": "housing_price_index_second_hand",
}

# AggregatedElectricityConsumption, district level, single year (2021).
ELECTRICITY_INDICATORS = {
    "Percentile 10 of electricity consumption in kwh": "electricity_consumption_p10",
    "Percentile 25 of electricity consumption in kwh": "electricity_consumption_p25",
    "Percentile 50 of electricity consumption in kwh": "electricity_consumption_p50",
    "Percentile 75 of electricity consumption in kwh": "electricity_consumption_p75",
    "Percentile 90 of electricity consumption in kwh": "electricity_consumption_p90",
}

# PopulationCensus age bands used to build broad age-structure shares.
_AGE_UNDER_15 = ["Population ~ Age:0-4", "Population ~ Age:5-9", "Population ~ Age:10-14"]
_AGE_WORKING = [
    "Population ~ Age:15-19", "Population ~ Age:20-24", "Population ~ Age:25-29",
    "Population ~ Age:30-34", "Population ~ Age:35-39", "Population ~ Age:40-44",
    "Population ~ Age:45-49", "Population ~ Age:50-54", "Population ~ Age:55-59",
    "Population ~ Age:60-64",
]
_AGE_65_PLUS = [
    "Population ~ Age:65-69", "Population ~ Age:70-74", "Population ~ Age:75-79",
    "Population ~ Age:80-84", "Population ~ Age:85-89", "Population ~ Age:90-94",
    "Population ~ Age:95-99", "Population ~ Age:>99",
]

# Section/district/municipality/province indicator datasets (always joinable
# from the census codes alone).
ALL_DATASETS = ("atlas", "population", "education", "rental", "price", "electricity")

# The per-building Essential Characteristics downscaling is a special dataset:
# it needs the CAT-based ``br__*`` columns, so it is only considered when
# ``use_CAT_files=True``. Selectable by name in ``social_ES_datasets`` alongside
# :data:`ALL_DATASETS`.
ESSENTIAL_CHARACTERISTICS = "essential_characteristics"

# Namespace prefix for the section/district/municipality/province social_ES
# indicator columns (output columns are ``so__{indicator}_{year}``).
SOCIAL_PREFIX = "so__"


def _scalar(v):
    """Normalise a code that may be a scalar or a 1-element list/array (the latter
    happens after address compaction) to a plain string, or ``None``."""
    if isinstance(v, (list, tuple, np.ndarray)):
        v = v[0] if len(v) else None
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _last_years(df, n_years):
    """Restrict a long dataframe to its ``n_years`` most recent ``Year`` values."""
    df = df.copy()
    df["Year"] = _num(df["Year"]).astype("Int64")
    years = sorted(df["Year"].dropna().unique())[-n_years:]
    return df[df["Year"].isin(years)]


def _pivot_wide(long_df, key_col, value_map, n_years):
    """Long → wide table keyed by ``key_col`` with ``so__{clean}_{year}`` columns.

    ``value_map`` maps source column names to clean snake_case output names. The
    ``so__`` namespace prefix is added here. Years are kept to the most recent
    ``n_years``. Duplicate (key, year) rows are averaged.
    """
    cols = [c for c in value_map if c in long_df.columns]
    if not cols:
        return None
    df = long_df[[key_col, "Year"] + cols].rename(columns=value_map)
    names = [value_map[c] for c in cols]
    df = _last_years(df, n_years)
    df = df.dropna(subset=[key_col])
    df = df[df[key_col].astype(str).str.len() > 0]
    for name in names:
        df[name] = _num(df[name])
    if df.empty:
        return None
    wide = df.pivot_table(index=key_col, columns="Year", values=names, aggfunc="mean")
    wide.columns = [f"{SOCIAL_PREFIX}{name}_{int(year)}" for name, year in wide.columns]
    return wide.reset_index()


def _section_key(df):
    """Build a 10-digit CUSEC key from Municipality/District/Section codes."""
    muni = df["Municipality code"].astype(str).str.zfill(5)
    dist = df["District code"].astype(str).str.zfill(2)
    sect = df["Section code"].astype(str).str.zfill(3)
    return muni + dist + sect


def _population_shares(sections):
    """Compute broad demographic shares from the PopulationCensus section table."""
    pop = _num(sections["Population"]).replace(0, np.nan)

    def share(cols):
        present = [c for c in cols if c in sections.columns]
        if not present:
            return pd.Series(np.nan, index=sections.index)
        return sections[present].apply(_num).sum(axis=1) / pop * 100

    out = pd.DataFrame({
        "Year": sections["Year"],
        "_key": _section_key(sections),
        "pct_foreign_nationality": share(["Population ~ Nationality:Foreign"]),
        "pct_foreign_born": share(["Population ~ Birth origin:Extranjero"]),
        "pct_male": share(["Population ~ Sex:Males"]),
        "pct_under_15": share(_AGE_UNDER_15),
        "pct_working_age": share(_AGE_WORKING),
        "pct_65_plus": share(_AGE_65_PLUS),
    })
    return out


def join_social_ES(gdf, wd, n_years=5, datasets=None, use_CAT_files=False,
                   verbose=True):
    """Join ``social_ES`` (INE) socioeconomic indicators onto every building.

    Each dataset is joined at its finest geographic level using the codes already
    present on ``gdf`` (``section_code``/``district_code``, plus municipality and
    province derived from them). Output columns are ``so__{indicator}_{year}`` for
    the most recent ``n_years`` available per dataset (see the ``*_INDICATORS``
    maps in this module for the curated indicator set; indicators are snake_case).

    The per-building **Essential Characteristics** downscaling
    (:data:`ESSENTIAL_CHARACTERISTICS`, → ``ec__*`` columns) is included only when
    ``use_CAT_files=True`` **and** it is selected (``datasets`` is ``None`` or
    lists ``"essential_characteristics"``); it needs the CAT-based ``br__*``
    columns. If it is explicitly requested without ``use_CAT_files`` it is skipped
    with a warning.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame; must carry ``section_code`` (from
        :func:`join_by_census_tracts`). ``district_code`` is used when present,
        otherwise derived from ``section_code``.
    wd : str
        hypercadaster working directory. ``social_ES`` caches its whole-Spain
        source files under ``{wd}/INE/...``.
    n_years : int, default 5
        Number of most-recent years to keep per dataset.
    datasets : sequence[str], optional
        Subset of :data:`ALL_DATASETS` plus :data:`ESSENTIAL_CHARACTERISTICS` to
        include (default: all of ``ALL_DATASETS``, and Essential Characteristics
        when ``use_CAT_files=True``).
    use_CAT_files : bool, default False
        Whether the CAT-based building-space aggregation ran (gates the Essential
        Characteristics dataset).
    verbose : bool, default True
        Emit per-dataset progress/matched-row counts to stderr.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with the socioeconomic columns added. Degrades gracefully: if
        ``social_ES`` is missing, ``section_code`` is absent, or an individual
        dataset fails to download/parse, that part is skipped with a warning and
        the frame is returned with whatever joined successfully.
    """
    # Resolve which section/district datasets to run, and whether the Essential
    # Characteristics downscaling is in scope.
    if datasets is None:
        run_simple = set(ALL_DATASETS)
        include_ec = bool(use_CAT_files)
    else:
        run_simple = {d for d in datasets if d in ALL_DATASETS}
        include_ec = ESSENTIAL_CHARACTERISTICS in datasets
        if include_ec and not use_CAT_files:
            sys.stderr.write(
                "social_ES: essential_characteristics requested but "
                "use_CAT_files=False; skipping (it needs the CAT-based br__ "
                "columns)\n")
            include_ec = False

    if "section_code" not in gdf.columns:
        sys.stderr.write("social_ES: no section_code column; skipping social layer\n")
        return gdf

    try:
        from social_ES import INE
    except ImportError:
        sys.stderr.write(
            "social_ES: package not installed; skipping social layer "
            "(`pip install social_ES`)\n")
        return gdf

    def log(msg):
        if verbose:
            sys.stderr.write(f"social_ES: {msg}\n")

    gdf = gdf.copy()
    # Normalised join keys (scalars, even after address compaction).
    sec = gdf["section_code"].map(_scalar)
    if "district_code" in gdf.columns:
        dist = gdf["district_code"].map(_scalar)
        dist = dist.where(dist.notna(), sec.str[:7])
    else:
        dist = sec.str[:7]
    muni = sec.str[:5]
    prov = sec.str[:2]
    gdf["_sec_key"] = sec
    gdf["_dist_key"] = dist
    gdf["_muni_key"] = muni
    gdf["_prov_key"] = prov

    muni_list = sorted(m for m in muni.dropna().unique() if m)

    def _require(df, cols, label):
        """Return True if ``df`` is usable (non-empty, has ``cols``). Many INE
        price/rental series only cover larger municipalities, so an empty return
        for the selected area is normal, not an error."""
        if df is None or df.empty:
            log(f"{label}: no data for the selected area; skipping")
            return False
        missing = [c for c in cols if c not in df.columns]
        if missing:
            log(f"{label}: no data for the selected area; skipping")
            return False
        return True

    def _merge(wide, key_col):
        nonlocal gdf
        if wide is None or wide.empty:
            return 0
        new_cols = [c for c in wide.columns if c != wide.columns[0]]
        wide = wide.rename(columns={wide.columns[0]: key_col})
        wide[key_col] = wide[key_col].astype(str)
        # Drop pre-existing same-named columns on re-runs to avoid _x/_y suffixes.
        gdf = gdf.drop(columns=[c for c in new_cols if c in gdf.columns], errors="ignore")
        n_match = gdf[key_col].isin(wide[key_col]).sum()
        gdf = gdf.merge(wide, on=key_col, how="left")
        return int(n_match)

    # --- Atlas (census section) ---
    if "atlas" in run_simple:
        try:
            log("HouseholdIncomeDistributionAtlas ...")
            atlas = INE.HouseholdIncomeDistributionAtlas(wd, municipality_code=muni_list)["Sections"]
            atlas["_key"] = _section_key(atlas)
            wide = _pivot_wide(atlas, "_key", ATLAS_INDICATORS, n_years)
            log(f"atlas: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: atlas failed ({e}); skipping\n")

    # --- Population census (census section) ---
    if "population" in run_simple:
        try:
            log("PopulationCensus ...")
            pop = INE.PopulationCensus(wd, municipality_code=muni_list)["Sections"]
            shares = _population_shares(pop)
            share_map = {c: c for c in shares.columns if c not in ("Year", "_key")}
            wide = _pivot_wide(shares, "_key", share_map, n_years)
            log(f"population: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: population census failed ({e}); skipping\n")

    # --- Education & employment census (census section, relative %) ---
    if "education" in run_simple:
        try:
            log("EducationAndEmploymentCensus ...")
            edu = INE.EducationAndEmploymentCensus(
                wd, municipality_code=muni_list, mode="relative")["Sections"]
            edu["_key"] = _section_key(edu)
            wide = _pivot_wide(edu, "_key", EDU_INDICATORS, n_years)
            log(f"education: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: education census failed ({e}); skipping\n")

    # --- Household rental price index (municipality) ---
    if "rental" in run_simple:
        try:
            log("HouseholdsRentalPriceIndex ...")
            rental = INE.HouseholdsRentalPriceIndex(wd, municipality_code=muni_list)["Municipality"]
            if _require(rental, ["Municipality code"], "rental"):
                rental["_key"] = rental["Municipality code"].astype(str).str.zfill(5)
                wide = _pivot_wide(rental, "_key", RENTAL_INDICATORS, n_years)
                log(f"rental: matched {_merge(wide, '_muni_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: rental price index failed ({e}); skipping\n")

    # --- Housing price index (province, quarterly -> annual mean) ---
    if "price" in run_simple:
        try:
            log("HouseholdsPriceIndex ...")
            price = INE.HouseholdsPriceIndex(wd, municipality_code=muni_list)["Province"]
            if _require(price, ["Province code"], "price"):
                price = price.dropna(subset=["Province code"]).copy()
                price["_key"] = price["Province code"].astype(str).str.zfill(2)
                wide = _pivot_wide(price, "_key", PRICE_INDICATORS, n_years)
                log(f"price: matched {_merge(wide, '_prov_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: housing price index failed ({e}); skipping\n")

    # --- Aggregated electricity consumption (district) ---
    if "electricity" in run_simple:
        try:
            log("AggregatedElectricityConsumption ...")
            elec = INE.AggregatedElectricityConsumption(wd, municipality_code=muni_list)["Districts"]
            if _require(elec, ["Municipality code", "District code"], "electricity"):
                elec["_key"] = (elec["Municipality code"].astype(str).str.zfill(5)
                                + elec["District code"].astype(str).str.zfill(2))
                wide = _pivot_wide(elec, "_key", ELECTRICITY_INDICATORS, n_years)
                log(f"electricity: matched {_merge(wide, '_dist_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: electricity consumption failed ({e}); skipping\n")

    gdf = gdf.drop(columns=["_sec_key", "_dist_key", "_muni_key", "_prov_key"],
                   errors="ignore")

    # --- Essential Characteristics per-building downscaling (needs CAT br__*) ---
    if include_ec:
        gdf = join_essential_characteristics(gdf, wd, verbose=verbose)

    return gdf


# Columns the social_ES EssentialCharacteristics downscaling needs. The ``br__*``
# come from the CAT-based building-space aggregation (``use_CAT_files=True``).
_EC_REQUIRED_COLS = (
    "section_code", "building_reference", "ine_code", "year_of_construction",
    "br__area_without_communals", "br__building_spaces",
)

# Namespace prefix for the per-building essential-characteristics proportions.
EC_PREFIX = "ec__"


def join_essential_characteristics(gdf, wd, prefix=EC_PREFIX, verbose=True):
    """Downscale the INE *Essential Characteristics of Population and Households*
    survey to the building level and join it onto ``gdf``.

    Wraps ``social_ES.INE.EssentialCharacteristicsOfPopulationAndHouseholds``,
    which allocates section-level survey proportions (age, income band, household
    nationality, dwelling size, building type/age, heating, vehicles, internet
    access, …) to individual buildings using the CAT-based building-space
    aggregation (``br__area_without_communals`` / ``br__building_spaces`` per use
    type, ``year_of_construction``). Each downscaled variable becomes a
    ``{prefix}{…}`` column (proportions in %, one group summing to 100). Spanish
    INE labels are translated to English snake_case keys at source (via
    :func:`ec_column_rename_map`), e.g. ``ec__main_dwellings_heating_type_individual``.

    Only meaningful when the CAT-based building-space aggregation has run
    (``use_CAT_files=True``) — the required ``br__*`` columns are otherwise
    absent, in which case the frame is returned unchanged with a warning. It is
    invoked from :func:`join_social_ES` when ``use_CAT_files=True`` and the
    ``"essential_characteristics"`` dataset is selected.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame carrying the inference outputs and ``section_code``.
    wd : str
        hypercadaster working directory (``social_ES`` reads its cached whole-Spain
        atlas/population files from ``{wd}/INE/...``).
    prefix : str, default ``"ec__"``
        Namespace prefix for the added proportion columns.
    verbose : bool, default True
        Emit progress to stderr.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with the ``{prefix}…`` proportion columns added (left join on
        ``building_reference``; non-residential/unmatched buildings get NaN).
        Degrades gracefully to a no-op on missing ``social_ES``, missing required
        columns, or an empty downscaling result.
    """
    def log(msg):
        if verbose:
            sys.stderr.write(f"social_ES: {msg}\n")

    missing = [c for c in _EC_REQUIRED_COLS if c not in gdf.columns]
    if missing:
        sys.stderr.write(
            "social_ES: essential characteristics needs CAT-based columns "
            f"{missing}; skipping (run with use_CAT_files=True)\n")
        return gdf

    try:
        from social_ES import INE
    except ImportError:
        sys.stderr.write(
            "social_ES: package not installed; skipping essential characteristics\n")
        return gdf

    # The social_ES downscaling merges on a *scalar* section_code; after address
    # compaction it can be a 1-element list. Pass a copy with a scalar key.
    work = gdf.copy()
    work["section_code"] = work["section_code"].map(_scalar)
    work = work[work["section_code"].notna()]
    if work.empty:
        log("essential characteristics: no buildings with a section_code; skipping")
        return gdf

    try:
        log("EssentialCharacteristicsOfPopulationAndHouseholds ...")
        proportions = INE.EssentialCharacteristicsOfPopulationAndHouseholds(
            wd, hypercadaster_ES_input_gdf=work)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"social_ES: essential characteristics failed ({e}); skipping\n")
        return gdf

    # Returns a dict {variable -> DataFrame indexed by building_reference}. Stitch
    # them into one wide frame keyed by building_reference.
    frames = [df for df in proportions.values()
              if df is not None and not df.empty]
    if not frames:
        log("essential characteristics: empty result; skipping")
        return gdf
    ec = pd.concat(frames, axis=1)
    ec = ec.loc[:, ~ec.columns.duplicated()]
    ec.columns = [f"{prefix}{c}" for c in ec.columns]
    # Emit the columns with English snake_case keys directly (Spanish INE labels
    # -> e.g. ec__main_dwellings_heating_type_individual).
    ec = ec.rename(columns=ec_column_rename_map(ec.columns))
    ec = ec.reset_index().rename(columns={ec.index.name or "index": "building_reference"})
    if "building_reference" not in ec.columns:
        ec = ec.rename(columns={ec.columns[0]: "building_reference"})
    ec["building_reference"] = ec["building_reference"].astype(str)

    new_cols = [c for c in ec.columns if c != "building_reference"]
    # The downscaler returns rows for every residential building, but the INE
    # *Características esenciales* survey is only published for larger
    # municipalities (~147 nationwide) — buildings elsewhere come back all-NaN.
    # Drop them so only buildings with real data are joined (and the reported
    # count is meaningful).
    ec = ec.dropna(how="all", subset=new_cols)
    if ec.empty:
        log("essential characteristics: no municipalities with survey data in the "
            "selected area; skipping (survey is published only for larger "
            "municipalities)")
        return gdf
    gdf = gdf.drop(columns=[c for c in new_cols if c in gdf.columns], errors="ignore")
    gdf = gdf.copy()
    gdf["building_reference"] = gdf["building_reference"].astype(str)
    n_match = gdf["building_reference"].isin(ec["building_reference"]).sum()
    gdf = gdf.merge(ec, on="building_reference", how="left")
    log(f"essential characteristics: matched {int(n_match)} buildings "
        f"(survey covers larger municipalities only), added {len(new_cols)} columns")
    return gdf


# =============================================================================
# Essential Characteristics column renaming: Spanish -> English snake_case
# =============================================================================
#
# The ``ec__*`` columns follow the structure ``ec__{group} - {variable} ~
# {subclass}``. :func:`rename_ec_columns` translates each of the three parts via
# the curated dictionaries below and re-joins them as snake_case, underscore-
# separated, e.g.
#   ``ec__Viviendas principales - Tipo de calefacción ~ Sí, individual``
#   -> ``ec__main_dwellings_heating_type_individual``
# The curated maps hold clean PascalCase words; the assembled body is lowered to
# snake_case in :func:`translate_ec_column`. Anything not in the dictionaries
# falls back to an accent-stripped CamelCase of the Spanish text (also snaked;
# deterministic, never raises), so new INE categories still get a stable key;
# pass ``warn_unmapped=True`` to list them so the maps can be extended.

_EC_GROUPS = {
    "Personas": "People",
    "Personas de 16 años o más": "People16OrOver",
    "Personas de 16 años o más que conviven con otras personas": "People16OrOverLivingWithOthers",
    "Personas de 16 años o más que conviven con otras personas y participan en cuidados a personas dependientes dentro del hogar": "People16OrOverCaringForDependentsInHousehold",
    "Personas de 16 años o más que participan en cuidados a personas dependientes fuera del hogar": "People16OrOverCaringForDependentsOutsideHousehold",
    "Personas de 16 años o más que viven solas": "People16OrOverLivingAlone",
    "Personas nacidas en España": "PeopleBornInSpain",
    "Personas nacidas fuera de España": "PeopleBornOutsideSpain",
    "Hogares": "Households",
    "Hogares/personas": "PeopleInHouseholds",
    "Hogares de una sola persona (unipersonales)": "SinglePersonHouseholds",
    "Parejas convivientes": "CohabitingCouples",
    "Hogares/personas en viviendas alquiladas": "PeopleLivingInRentedHomes",
    "Hogares/personas en viviendas propias": "PeopleLivingInTheirOwnHomes",
    "Hogares/personas con segunda residencia": "HouseholdsPersonsWithSecondHome",
    "Hogares/personas con vehículo": "HouseholdsPersonsWithVehicle",
    "Hogares con vehículo": "HouseholdsWithVehicle",
    "Hogares/personas con servicio doméstico remunerado": "HouseholdsPersonsWithPaidDomesticService",
    "Hogares con servicio doméstico remunerado": "HouseholdsWithPaidDomesticService",
    "Hogares/personas con ayuda externa": "HouseholdsPersonsWithExternalHelp",
    "Hogares con ayuda externa": "HouseholdsWithExternalHelp",
    "Viviendas principales": "MainDwellings",
    "Viviendas principales con calefacción": "MainDwellingsWithHeating",
}

_EC_VARIABLES = {
    "Lugar de trabajo/estudio": "WorkStudyLocation",
    "Número de desplazamaientos": "NumberOfTrips",
    "Medio de transporte": "TransportMode",
    "Tipo de vehículo": "VehicleType",
    "Tiempo diario": "DailyCommuteTime",
    "Grado de satisfacción": "SatisfactionLevel",
    "Grado de participación en las tareas domésticas": "HouseworkParticipation",
    "Grado de participación en cuidados a menores o dependientes dentro del hogar": "CareParticipationInHousehold",
    "Grado de participación en cuidados a menores o dependientes fuera del hogar": "CareParticipationOutsideHousehold",
    "Tipo de dependiente": "DependentType",
    "Horas diarias dedicadas al cuidado": "DailyCareHours",
    "Tiene apoyo social": "HasSocialSupport",
    "Tipo de relación": "RelationshipType",
    "Lugar de residencia": "ResidenceLocation",
    "Lugar de nacimiento progenitores": "ParentsBirthplace",
    "Residencia progenitores": "ParentsResidence",
    "Nacionalidad progenitores": "ParentsNationality",
    "Nivel estudios progenitores": "ParentsEducationLevel",
    "Accede habitualmente a internet": "RegularInternetAccess",
    "Dispone de perfil en alguna red social": "HasSocialMediaProfile",
    "Dispone de smartphone": "HasSmartphone",
    "Realiza compras por internet": "OnlineShopping",
    "Realiza ventas por internet": "OnlineSelling",
    "Edad": "Age",
    "Nacionalidad": "Nationality",
    "Estado civil": "MaritalStatus",
    "Nivel educativo": "EducationLevel",
    "Situación laboral": "EmploymentStatus",
    "Nivel de ingresos mensuales netos del hogar": "MonthlyNetHouseholdIncomeLevel",
    "Sexo de la pareja": "CoupleSex",
    "Nacionalidad de la pareja": "CoupleNationality",
    "Número de hijos": "NumberOfChildren",
    "Número de habitaciones de la vivienda": "NumberOfRoomsInDwelling",
    "Superficie útil de la vivienda": "UsefulFloorArea",
    "Tipo de edificio": "BuildingType",
    "Régimen de tenencia de la vivienda": "TenureRegime",
    "Cuota mensual del alquiler": "MonthlyRent",
    "Cuota mensual de la hipoteca": "MonthlyMortgage",
    "Disponen de segunda residencia": "HasSecondHome",
    "Lugar de la segunda residencia": "SecondHomeLocation",
    "Días de uso de la segunda residencia al año": "SecondHomeDaysPerYear",
    "Número de vehículos": "NumberOfVehicles",
    "Vehículo ecólogico": "EcoFriendlyVehicle",
    "Separan algún tipo de residuo": "SeparatesWaste",
    "Tipo de residuos separados": "SeparatedWasteType",
    "Servicio doméstico remunerado": "PaidDomesticService",
    "Ayuda externa": "ExternalHelp",
    "Adaptada a necesidades propias del envejecimiento": "AdaptedForAging",
    "Problema de aislamiento": "IsolationProblem",
    "Tipo de calefacción": "HeatingType",
    "Tipo de combustible": "FuelType",
    "Sistema de suministro de agua": "WaterSupplySystem",
    "Tiene sistema de refrigeración": "HasCooling",
    "Tipo de conexión a internet": "InternetConnectionType",
    "Tipo de electrodoméstico": "ApplianceType",
    "Tipo de bombillas": "BulbType",
    "Aseo con inodoro / Bañera o ducha": "ToiletBathOrShower",
    "Número de cuartos de baño o aseos": "NumberOfBathrooms",
    "Cocina independiente de 4 m2 o más": "SeparateKitchen4m2OrMore",
    "Número de habitaciones": "NumberOfRooms",
    "Tipo de problemática en la zona": "AreaProblemType",
    "Tipo de infraestructura o servicio": "InfrastructureServiceType",
    "Estado de conservación": "ConservationStatus",
    "Accesibilidad": "Accessibility",
    "Tipo de instalación": "InstallationType",
    "Número de plazas de garaje": "NumberOfGarageSpaces",
    "Tipo de dispositivo de energía renovable": "RenewableEnergyDeviceType",
}

_EC_SUBCLASSES = {
    # generic yes/no and counts
    "Sí": "Yes",
    "No": "No",
    "0": "0", "1": "1", "2": "2", "3": "3",
    "4 o más": "4OrMore", "3 o más": "3OrMore", "2 o más": "2OrMore",
    # work / study location
    "En el municipio en el que resido": "InMunicipalityOfResidence",
    "En el propio domicilio": "AtOwnHome",
    "En otro municipio de la misma provincia": "AnotherMunicipalitySameProvince",
    "En varios municipios (soy comercial, repartidor, taxista...)": "InSeveralMunicipalities",
    "Otros lugares": "OtherPlaces",
    "Más de 2": "MoreThan2",
    # transport
    "Andando": "OnFoot",
    "De empresa u otro medio": "CompanyOrOther",
    "Particular": "Private",
    "Público": "Public",
    "Autobús, tren, metro o tranvía": "BusTrainMetroTram",
    "Coche": "Car",
    "Moto, bicicleta y otros tipos": "MotorbikeBicycleOther",
    # daily commute time
    "90 minutos o más": "90MinOrMore",
    "Entre 20 y 39 minutos": "20To39Min",
    "Entre 40 y 59 minutos": "40To59Min",
    "Entre 60 y 89 minutos": "60To89Min",
    "Menos de 20 minutos": "LessThan20Min",
    # satisfaction
    "Insatisfecho": "Dissatisfied",
    "Muy satisfecho": "VerySatisfied",
    "Satisfecho": "Satisfied",
    # housework / care participation
    "Me encargo de la mayor parte de las tareas domésticas de las tareas domésticas": "MostOfTasks",
    "Me encargo de una parte importante de las tareas, compartiéndolas con otra/s personas": "SignificantPartShared",
    "Me encargo de una pequeña parte de las tareas domésticas": "SmallPart",
    "No participo habitualmente en las tareas domésticas": "DoesNotParticipate",
    "Me encargo de la mayor parte de los cuidados": "MostOfCare",
    "Me encargo de una parte importante de los cuidados, compartiéndolos con otra/s personas": "SignificantPartShared",
    "Me encargo de una pequeña parte de los cuidados": "SmallPart",
    "No hay personas menores ni personas dependientes en el hogar": "NoMinorsOrDependents",
    "No participio habitualmente en los cuidados": "DoesNotParticipate",
    "No tengo ninguna persona dependiente a mi cargo": "NoDependentsInCharge",
    # dependent type
    "Discapacitado/a": "Disabled",
    "Enfermo/a crónico/a": "ChronicallyIll",
    "Mayor de 70 años u otro tipo": "Over70OrOther",
    "Menor de edad": "Minor",
    # daily care hours
    "6 horas o más": "6HOrMore",
    "Entre 2 y 6 horas": "2To6H",
    "Menos de 2 horas": "LessThan2H",
    "3 horas o más": "3HOrMore",
    "Entre 1 y 3 horas": "1To3H",
    "Menos de 1 hora": "LessThan1H",
    # relationship type (living alone)
    "Amigos/as o vecinos/as y otros parientes (que no son hijos/as)": "FriendsNeighborsAndOtherRelatives",
    "Hijos/as y amigos/as o vecinos/as": "ChildrenAndFriendsNeighbors",
    "Hijos/as y otras personas (parientes o no)": "ChildrenAndOthers",
    "Otros parientes (que no son hijos/as)": "OtherRelatives",
    "Sólo amigos/as o vecinos/as": "OnlyFriendsNeighbors",
    "Sólo hijos/as": "OnlyChildren",
    # residence location
    "En el mismo municipio": "SameMunicipality",
    "Fuera de la provincia": "OutsideProvince",
    # parents birthplace / residence
    "España pero distinta comunidad autónoma": "SpainDifferentRegion",
    "Misma comunidad autónoma pero distinta provincia": "SameRegionDifferentProvince",
    "Misma provincia pero distinto municipio": "SameProvinceDifferentMunicipality",
    "Mismo municipio": "SameMunicipality",
    "Ningún progenitor ha nacido en España": "NoParentBornInSpain",
    "Distinto país": "DifferentCountry",
    "Mismo país": "SameCountry",
    "Mismo hogar": "SameHousehold",
    "Mismo municipio pero distinto hogar": "SameMunicipalityDifferentHousehold",
    "Ningún progenitor reside en España": "NoParentResidesInSpain",
    "Diferente nacionalidad": "DifferentNationality",
    "Misma nacionalidad": "SameNationality",
    # parents education level
    "Mayor nivel de estudios que sus progenitores": "HigherThanParents",
    "Menor nivel de estudios que sus progenitores": "LowerThanParents",
    "Mismo nivel de estudios que sus progenitores": "SameAsParents",
    # age (single-person households)
    "De 30 a 39 años": "30To39",
    "De 40 a 49 años": "40To49",
    "De 50 a 59 años": "50To59",
    "De 60 a 69 años": "60To69",
    "De 70 a 79 años": "70To79",
    "De 80 y más años": "80OrOver",
    "Menos de 30 años": "LessThan30",
    # nationality / marital status
    "Española": "Spanish",
    "Extranjera": "Foreign",
    "Casado/a": "Married",
    "Separado/a o divorciado/a": "SeparatedOrDivorced",
    "Soltero/a": "Single",
    "Viudo/a": "Widowed",
    # education level
    "Educación primaria e inferior": "PrimaryOrBelow",
    "Educación superior": "Tertiary",
    "Primera etapa de educación secundaria y similar": "LowerSecondary",
    "Segunda etapa de educación secundaria y educación postsecundaria no superior": "UpperSecondary",
    # employment status
    "Jubilado/a, prejubilado/a o incapacitado/a permamentemente para trabajar": "RetiredOrDisabled",
    "No trabaja": "NotWorking",
    "Ocupado/a": "Employed",
    # income levels (single-person households)
    "3.000 euros o más": "3000EurOrMore",
    "De 1.000 euros a menos de 1.500 euros": "1000To1500Eur",
    "De 1.500 euros a menos de 2.000 euros": "1500To2000Eur",
    "De 2.000 euros a menos de 3.000 euros": "2000To3000Eur",
    "Menos de 1.000 euros": "LessThan1000Eur",
    # couple sex / nationality
    "Pareja de distinto sexo": "DifferentSex",
    "Pareja del mismo sexo": "SameSex",
    "Ambos españoles": "BothSpanish",
    "Ambos extranjeros": "BothForeign",
    "Español/a y extranjero/a": "SpanishAndForeign",
    # number of children
    "0 hijos conviviendo": "0Cohabiting",
    "0 hijos conviviendo menores de 25 años": "0CohabitingUnder25",
    "Algún hijo conviviendo": "SomeCohabiting",
    "Algún hijo conviviendo menor de 25 años": "SomeCohabitingUnder25",
    "Ningún hijo conviviendo menor de 25 años": "NoneCohabitingUnder25",
    # useful floor area
    "Hasta 45 m2": "UpTo45m2",
    "Entre 46 y 60 m2": "46To60m2",
    "Menos de 60 m2": "LessThan60m2",
    "Entre 61 y 75 m2": "61To75m2",
    "Entre 76 y 90 m2": "76To90m2",
    "Entre 91 y 105 m2": "91To105m2",
    "Entre 106 y 120 m2": "106To120m2",
    "Entre 121 y 150 m2": "121To150m2",
    "Más de 150 m2": "MoreThan150m2",
    # building type
    "Edificio de 2 o más viviendas": "MultiFamily",
    "Vivienda unifamiliar (chalet, adosado, pareado...)": "SingleFamily",
    # tenure regime
    "Alquilada": "Rented",
    "Cedida gratis o a bajo precio (por otro hogar, pagada por la empresa...) u otra forma": "CededOrOther",
    "Propia por herencia o donación": "OwnedInheritedOrGifted",
    "Propia, por compra, con pagos pendientes (hipotecas)": "OwnedWithMortgage",
    "Propia, por compra, totalmente pagada": "OwnedFullyPaid",
    # rent / mortgage brackets
    "Menos de 200 euros": "LessThan200Eur",
    "De 200 euros a menos de 500 euros": "200To500Eur",
    "De 500 euros a menos de 700 euros": "500To700Eur",
    "De 700 euros a menos de 1.000 euros": "700To1000Eur",
    "De 1.000 euros o más": "1000EurOrMore",
    # second home
    "En el mismo municipio que la primera residencia": "SameMunicipalityAsPrimary",
    "En otra CCAA": "AnotherRegion",
    "En otra provincia, pero en la misma CCAA": "AnotherProvinceSameRegion",
    "60 o más días": "60OrMoreDays",
    "Entre 15 y 29 días": "15To29Days",
    "Entre 30 y 59 días": "30To59Days",
    "Menos de 15 días": "LessThan15Days",
    # eco vehicle
    "No tiene vehículo ecológico (híbrido o eléctrico)": "No",
    "Tiene algún vehículo ecológico (híbrido o eléctrico)": "Yes",
    # separated waste type
    "Envases": "Packaging",
    "Orgánico": "Organic",
    "Papel": "Paper",
    "Vidrio": "Glass",
    # paid domestic service / external help
    "No dispone de servicio doméstico": "No",
    "Sí, dispone de servicio doméstico": "Yes",
    "No dispone de ayudas externas no remuneradas": "None",
    "Sí, dispone de ayudas de familiares, parientes, amigos, vecinos": "FromFamilyOrFriends",
    "Sí, dispone de ayudas de otros, como servicios sociales o una ONG": "FromSocialServicesOrNGO",
    # heating type
    "No tiene calefacción": "None",
    "No tiene instalación de calefacción pero sí algún aparato que permite calentar alguna habitación (por ejemplo radiadores eléctricos)": "NoInstallationButAppliance",
    "Sí, colectiva": "CollectiveCentral",
    "Sí, individual": "Individual",
    # fuel type
    "Electricidad": "Electricity",
    "Gas natural": "NaturalGas",
    "Otros": "Other",
    "Petróleo o derivados (gasoil, fueloil, gasolina...)": "OilDerivatives",
    # water supply
    "Agua corriente por abastecimiento privado o particular del edificio": "PrivateSupply",
    "Agua corriente por abastecimiento público": "PublicSupply",
    # internet connection
    "Conexión fija": "Fixed",
    "Conexión fija y móvil": "FixedAndMobile",
    "Conexión móvil": "Mobile",
    "Sin conexión": "NoConnection",
    # appliances
    "Horno": "Oven",
    "Lavadora": "WashingMachine",
    "Lavavajillas": "Dishwasher",
    "Microondas": "Microwave",
    "Secadora": "Dryer",
    "Vitocerámica/Inducción": "CeramicInductionHob",
    # bulbs
    "Bajo consumo": "LowConsumption",
    "Halógena": "Halogen",
    "Led": "Led",
    # toilet / bath
    "Aseo con inodoro": "ToiletWithWC",
    "Bañera o ducha": "BathOrShower",
    # area problems
    "Contaminación o malos olores": "PollutionOrOdors",
    "Delincuencia": "Crime",
    "Malas comunicaciones": "PoorTransportLinks",
    "Molestias relacionadas con actividades turísticas o locales de hostelería": "TourismOrHospitalityNuisance",
    "Poca limpieza en las calles": "DirtyStreets",
    "Pocas zonas verdes": "FewGreenAreas",
    "Ruidos exteriores": "OutdoorNoise",
    # nearby infrastructure / services
    "Colegio": "School",
    "Farmacia": "Pharmacy",
    "Hospital, centros de salud o ambulatorio": "HospitalOrHealthCenter",
    "Servicios de restauración (bares, restaurantes...)": "FoodServices",
    "Supermercado": "Supermarket",
    # conservation status
    "Excelente": "Excellent",
    "Muy bueno": "VeryGood",
    "Bueno": "Good",
    "Normal": "Normal",
    "Regular": "Fair",
    "Malo": "Bad",
    # accessibility
    "Es accesible": "Accessible",
    "No es accesible": "NotAccessible",
    # installation type
    "Agua caliente central": "CentralHotWater",
    "Ascensor": "Elevator",
    "Dispositivo de energía renovable": "RenewableEnergyDevice",
    "Garaje": "Garage",
    "Gas por tubería": "PipedGas",
    # garage spaces
    "De 3 a 20": "3To20",
    "De 21 a 50": "21To50",
    "Más de 50": "MoreThan50",
    # renewable energy device type
    "No dispone de dispositivo de energía renovable": "None",
    "Un único tipo de dispositivo: Energía solar fotovoltaica, eólica, biomasa, etc": "SingleDeviceSolarPvWindBiomass",
    "Un único tipo de dispositivo: Energía solar térmica": "SingleDeviceSolarThermal",
}


def _camelize(text):
    """Accent-stripped PascalCase fallback for text absent from the maps."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^0-9A-Za-z]+", " ", t).strip()
    return "".join(w[:1].upper() + w[1:] for w in t.split()) if t else ""


def _clean_ec_part(text):
    """Strip whitespace, a trailing ``(%)`` marker and trailing punctuation."""
    t = re.sub(r"\s*\(%\)\s*$", "", str(text)).strip()
    return t.rstrip(",.").strip()


def _translate_ec_part(text, mapping, unmapped=None):
    cleaned = _clean_ec_part(text)
    if cleaned in mapping:
        return mapping[cleaned]
    if unmapped is not None and cleaned:
        unmapped.add(cleaned)
    return _camelize(cleaned)


def translate_ec_column(col, unmapped=None):
    """Translate one ``ec__…`` Spanish column name to English snake_case.

    Parses ``ec__{group} - {variable} ~ {subclass}`` and joins the translated
    parts as ``ec__{group}_{variable}_{subclass}`` (snake_case, ``ec__``
    namespace prefix), e.g. ``ec__main_dwellings_heating_type_individual``.
    Non-``ec__`` names are returned unchanged. Unknown parts fall back to
    :func:`_camelize` (and are collected in ``unmapped`` when a set is passed).
    """
    if not isinstance(col, str) or not col.startswith("ec__"):
        return col
    body = col[len("ec__"):]
    # Idempotency: already-translated names carry neither the " - " nor " ~ "
    # separators of the Spanish INE labels, so leave them untouched (avoids
    # mangling English columns on a second pass).
    if " - " not in body and " ~ " not in body:
        return col
    left, sep, subclass = body.partition(" ~ ")
    group, gsep, variable = left.partition(" - ")
    parts = [
        _translate_ec_part(group, _EC_GROUPS, unmapped),
        _translate_ec_part(variable, _EC_VARIABLES, unmapped) if gsep else "",
        _translate_ec_part(subclass, _EC_SUBCLASSES, unmapped) if sep else "",
    ]
    # Curated maps hold clean PascalCase words; snake_case the joined body and
    # re-prepend the ``ec__`` namespace (kept literal, not snaked).
    return "ec__" + utils.to_snake_case("_".join(p for p in parts if p))


def ec_column_rename_map(columns, warn_unmapped=False):
    """Build a ``{spanish_ec_column -> english_ec_column}`` map for ``columns``.

    Only ``ec__…`` columns are included. Collisions (two source columns mapping
    to the same target) are disambiguated with a ``_2``, ``_3``… suffix so no
    column is silently dropped. With ``warn_unmapped=True``, Spanish group/
    variable/subclass fragments missing from the dictionaries are reported to
    stderr (they still get an accent-stripped CamelCase fallback).
    """
    unmapped = set() if warn_unmapped else None
    mapping, seen = {}, {}
    for col in columns:
        if not (isinstance(col, str) and col.startswith("ec__")):
            continue
        new = translate_ec_column(col, unmapped)
        n = seen.get(new, 0) + 1
        seen[new] = n
        mapping[col] = new if n == 1 else f"{new}_{n}"
    if warn_unmapped and unmapped:
        sys.stderr.write(
            "social_ES: unmapped ec__ fragments (accent-stripped fallback used): "
            + "; ".join(sorted(unmapped)) + "\n")
    return mapping


def rename_ec_columns(df, warn_unmapped=False):
    """Return ``df`` with its Spanish ``ec__…`` columns renamed to English
    snake_case (see :func:`translate_ec_column`). Non-``ec__`` columns and the
    data are untouched; the original frame is not mutated.

    Examples
    --------
    >>> gdf = hc.social.rename_ec_columns(gdf)
    # 'ec__Viviendas principales - Tipo de calefacción ~ Sí, individual'
    #   -> 'ec__main_dwellings_heating_type_individual'
    """
    mapping = ec_column_rename_map(df.columns, warn_unmapped=warn_unmapped)
    return df.rename(columns=mapping)
