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
import warnings

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
    # Income by source: these are the % share of household income coming from
    # each source (not € amounts).
    "Source of income: wages": "pct_income_source_salary",
    "Source of income: pensions": "pct_income_source_pension",
    "Source of income: unemployment benefits": "pct_income_source_unemployment",
    "Source of income: other benefits": "pct_income_source_other_benefits",
    "Source of income: other income": "pct_income_source_capital_investments_secondary",
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

    # Announce the join (always, regardless of verbose): the ordered set of
    # datasets and the number of most-recent years being joined.
    ds_names = [d for d in ALL_DATASETS if d in run_simple]
    if include_ec:
        ds_names.append(ESSENTIAL_CHARACTERISTICS)
    sys.stderr.write(
        f"Joining the socioeconomic and sociodemographic data of last "
        f"{n_years} years (datasets: {', '.join(ds_names)})\n")

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
    :func:`ec_column_rename_map`), e.g. ``ec__main_dwellings_heating_individual``.

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
        # social_ES divides each variable by its group row-sum to get a
        # proportion; groups with a zero row-sum (no dwellings/people of that
        # kind in a section) yield 0/0 -> NaN and numpy floods stderr with
        # "invalid value encountered in divide" RuntimeWarnings. The NaN is the
        # intended result, so silence just that noise around the call.
        with np.errstate(divide="ignore", invalid="ignore"), \
                warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="invalid value encountered in divide")
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
    # -> e.g. ec__main_dwellings_heating_individual).
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
# the curated dictionaries below and re-joins them with underscores, e.g.
#   ``ec__Viviendas principales - Tipo de calefacción ~ Sí, individual``
#   -> ``ec__main_dwellings_heating_individual``
# The curated maps hold the final **short snake_case** values directly (used
# verbatim), so digit spacing and abbreviations are controlled at the source.
# Anything not in the dictionaries falls back to an accent-stripped, snake_cased
# form of the Spanish text (deterministic, never raises), so new INE categories
# still get a stable key; pass ``warn_unmapped=True`` to list them so the maps
# can be extended.

_EC_GROUPS = {
    "Personas": "people",
    "Personas de 16 años o más": "people_16plus",
    "Personas de 16 años o más que conviven con otras personas": "people_16plus_cohabiting",
    "Personas de 16 años o más que conviven con otras personas y participan en cuidados a personas dependientes dentro del hogar": "people_16plus_caring_in_hh",
    "Personas de 16 años o más que participan en cuidados a personas dependientes fuera del hogar": "people_16plus_caring_out_hh",
    "Personas de 16 años o más que viven solas": "people_16plus_alone",
    "Personas nacidas en España": "people_born_es",
    "Personas nacidas fuera de España": "people_born_foreign",
    "Hogares": "households",
    "Hogares/personas": "hh_people",
    "Hogares de una sola persona (unipersonales)": "single_person_hh",
    "Parejas convivientes": "cohabiting_couples",
    "Hogares/personas en viviendas alquiladas": "rented_homes",
    "Hogares/personas en viviendas propias": "owned_homes",
    "Hogares/personas con segunda residencia": "hh_second_home",
    "Hogares/personas con vehículo": "hh_vehicle",
    "Hogares con vehículo": "households_vehicle",
    "Hogares/personas con servicio doméstico remunerado": "hh_paid_domestic",
    "Hogares con servicio doméstico remunerado": "households_paid_domestic",
    "Hogares/personas con ayuda externa": "hh_external_help",
    "Hogares con ayuda externa": "households_external_help",
    "Viviendas principales": "main_dwellings",
    "Viviendas principales con calefacción": "main_dwellings_heated",
}

_EC_VARIABLES = {
    "Lugar de trabajo/estudio": "work_study_place",
    "Número de desplazamaientos": "n_trips",
    "Medio de transporte": "transport_mode",
    "Tipo de vehículo": "vehicle_type",
    "Tiempo diario": "daily_time",
    "Grado de satisfacción": "satisfaction",
    "Grado de participación en las tareas domésticas": "housework_participation",
    "Grado de participación en cuidados a menores o dependientes dentro del hogar": "participation",
    "Grado de participación en cuidados a menores o dependientes fuera del hogar": "participation",
    "Tipo de dependiente": "dependent_type",
    "Horas diarias dedicadas al cuidado": "daily_care_hours",
    "Tiene apoyo social": "has_social_support",
    "Tipo de relación": "relationship_type",
    "Lugar de residencia": "residence_place",
    "Lugar de nacimiento progenitores": "parents_birthplace",
    "Residencia progenitores": "parents_residence",
    "Nacionalidad progenitores": "parents_nationality",
    "Nivel estudios progenitores": "parents_education",
    "Accede habitualmente a internet": "internet_access",
    "Dispone de perfil en alguna red social": "social_media",
    "Dispone de smartphone": "smartphone",
    "Realiza compras por internet": "online_shopping",
    "Realiza ventas por internet": "online_selling",
    "Edad": "age",
    "Nacionalidad": "nationality",
    "Estado civil": "marital_status",
    "Nivel educativo": "education",
    "Situación laboral": "employment",
    "Nivel de ingresos mensuales netos del hogar": "net_income",
    "Sexo de la pareja": "couple_sex",
    "Nacionalidad de la pareja": "couple_nationality",
    "Número de hijos": "n_children",
    "Número de habitaciones de la vivienda": "n_rooms",
    "Superficie útil de la vivienda": "floor_area",
    "Tipo de edificio": "building_type",
    "Régimen de tenencia de la vivienda": "tenure",
    "Cuota mensual del alquiler": "monthly_rent",
    "Cuota mensual de la hipoteca": "monthly_mortgage",
    "Disponen de segunda residencia": "has_second_home",
    "Lugar de la segunda residencia": "second_home_place",
    "Días de uso de la segunda residencia al año": "second_home_days_year",
    "Número de vehículos": "n_vehicles",
    "Vehículo ecólogico": "eco_vehicle",
    "Separan algún tipo de residuo": "separates_waste",
    "Tipo de residuos separados": "waste_type",
    "Servicio doméstico remunerado": "paid_domestic",
    "Ayuda externa": "external_help",
    "Adaptada a necesidades propias del envejecimiento": "aging_adapted",
    "Problema de aislamiento": "isolation_problem",
    "Tipo de calefacción": "heating",
    "Tipo de combustible": "fuel",
    "Sistema de suministro de agua": "water_supply",
    "Tiene sistema de refrigeración": "has_cooling",
    "Tipo de conexión a internet": "internet_connection",
    "Tipo de electrodoméstico": "appliance",
    "Tipo de bombillas": "bulb",
    "Aseo con inodoro / Bañera o ducha": "toilet_bath",
    "Número de cuartos de baño o aseos": "n_bathrooms",
    "Cocina independiente de 4 m2 o más": "separate_kitchen_4m2plus",
    "Número de habitaciones": "n_rooms",
    "Tipo de problemática en la zona": "area_problem",
    "Tipo de infraestructura o servicio": "infrastructure_service",
    "Estado de conservación": "conservation",
    "Accesibilidad": "accessibility",
    "Tipo de instalación": "installation",
    "Número de plazas de garaje": "n_garage_spaces",
    "Tipo de dispositivo de energía renovable": "renewable_device",
}

_EC_SUBCLASSES = {
    # generic yes/no and counts
    "Sí": "yes",
    "No": "no",
    "0": "0", "1": "1", "2": "2", "3": "3",
    "4 o más": "4plus", "3 o más": "3plus", "2 o más": "2plus",
    # work / study location
    "En el municipio en el que resido": "same_muni",
    "En el propio domicilio": "at_home",
    "En otro municipio de la misma provincia": "other_muni_same_prov",
    "En varios municipios (soy comercial, repartidor, taxista...)": "several_munis",
    "Otros lugares": "other_places",
    "Más de 2": "over_2",
    # transport
    "Andando": "on_foot",
    "De empresa u otro medio": "company_or_other",
    "Particular": "private",
    "Público": "public",
    "Autobús, tren, metro o tranvía": "bus_train_metro_tram",
    "Coche": "car",
    "Moto, bicicleta y otros tipos": "motorbike_bike_other",
    # daily commute time
    "90 minutos o más": "90min_plus",
    "Entre 20 y 39 minutos": "20_39min",
    "Entre 40 y 59 minutos": "40_59min",
    "Entre 60 y 89 minutos": "60_89min",
    "Menos de 20 minutos": "under_20min",
    # satisfaction
    "Insatisfecho": "dissatisfied",
    "Muy satisfecho": "very_satisfied",
    "Satisfecho": "satisfied",
    # housework / care participation
    "Me encargo de la mayor parte de las tareas domésticas de las tareas domésticas": "most",
    "Me encargo de una parte importante de las tareas, compartiéndolas con otra/s personas": "significant_shared",
    "Me encargo de una pequeña parte de las tareas domésticas": "small_part",
    "No participo habitualmente en las tareas domésticas": "none",
    "Me encargo de la mayor parte de los cuidados": "most",
    "Me encargo de una parte importante de los cuidados, compartiéndolos con otra/s personas": "significant_shared",
    "Me encargo de una pequeña parte de los cuidados": "small_part",
    "No hay personas menores ni personas dependientes en el hogar": "no_minors_or_dependents",
    "No participio habitualmente en los cuidados": "none",
    "No tengo ninguna persona dependiente a mi cargo": "no_dependents",
    # dependent type
    "Discapacitado/a": "disabled",
    "Enfermo/a crónico/a": "chronically_ill",
    "Mayor de 70 años u otro tipo": "over_70_or_other",
    "Menor de edad": "minor",
    # daily care hours
    "6 horas o más": "6h_plus",
    "Entre 2 y 6 horas": "2_6h",
    "Menos de 2 horas": "under_2h",
    "3 horas o más": "3h_plus",
    "Entre 1 y 3 horas": "1_3h",
    "Menos de 1 hora": "under_1h",
    # relationship type (living alone)
    "Amigos/as o vecinos/as y otros parientes (que no son hijos/as)": "friends_neighbors_relatives",
    "Hijos/as y amigos/as o vecinos/as": "children_friends_neighbors",
    "Hijos/as y otras personas (parientes o no)": "children_others",
    "Otros parientes (que no son hijos/as)": "other_relatives",
    "Sólo amigos/as o vecinos/as": "only_friends_neighbors",
    "Sólo hijos/as": "only_children",
    # residence location
    "En el mismo municipio": "same_muni",
    "Fuera de la provincia": "outside_prov",
    # parents birthplace / residence
    "España pero distinta comunidad autónoma": "es_other_region",
    "Misma comunidad autónoma pero distinta provincia": "same_region_other_prov",
    "Misma provincia pero distinto municipio": "same_prov_other_muni",
    "Mismo municipio": "same_muni",
    "Ningún progenitor ha nacido en España": "no_parent_born_es",
    "Distinto país": "other_country",
    "Mismo país": "same_country",
    "Mismo hogar": "same_hh",
    "Mismo municipio pero distinto hogar": "same_muni_other_hh",
    "Ningún progenitor reside en España": "no_parent_resides_es",
    "Diferente nacionalidad": "different_nationality",
    "Misma nacionalidad": "same_nationality",
    # parents education level
    "Mayor nivel de estudios que sus progenitores": "higher_than_parents",
    "Menor nivel de estudios que sus progenitores": "lower_than_parents",
    "Mismo nivel de estudios que sus progenitores": "same_as_parents",
    # age (single-person households)
    "De 30 a 39 años": "30_39",
    "De 40 a 49 años": "40_49",
    "De 50 a 59 años": "50_59",
    "De 60 a 69 años": "60_69",
    "De 70 a 79 años": "70_79",
    "De 80 y más años": "80plus",
    "Menos de 30 años": "under_30",
    # nationality / marital status
    "Española": "spanish",
    "Extranjera": "foreign",
    "Casado/a": "married",
    "Separado/a o divorciado/a": "separated_divorced",
    "Soltero/a": "single",
    "Viudo/a": "widowed",
    # education level
    "Educación primaria e inferior": "primary_or_below",
    "Educación superior": "tertiary",
    "Primera etapa de educación secundaria y similar": "lower_secondary",
    "Segunda etapa de educación secundaria y educación postsecundaria no superior": "upper_secondary",
    # employment status
    "Jubilado/a, prejubilado/a o incapacitado/a permamentemente para trabajar": "retired_or_disabled",
    "No trabaja": "not_working",
    "Ocupado/a": "employed",
    # income levels (single-person households)
    "3.000 euros o más": "ge_3000eur",
    "De 1.000 euros a menos de 1.500 euros": "1000_1500eur",
    "De 1.500 euros a menos de 2.000 euros": "1500_2000eur",
    "De 2.000 euros a menos de 3.000 euros": "2000_3000eur",
    "Menos de 1.000 euros": "under_1000eur",
    # couple sex / nationality
    "Pareja de distinto sexo": "different_sex",
    "Pareja del mismo sexo": "same_sex",
    "Ambos españoles": "both_spanish",
    "Ambos extranjeros": "both_foreign",
    "Español/a y extranjero/a": "spanish_and_foreign",
    # number of children
    "0 hijos conviviendo": "0_cohabiting",
    "0 hijos conviviendo menores de 25 años": "0_cohabiting_under25",
    "Algún hijo conviviendo": "some_cohabiting",
    "Algún hijo conviviendo menor de 25 años": "some_cohabiting_under25",
    "Ningún hijo conviviendo menor de 25 años": "none_cohabiting_under25",
    # useful floor area
    "Hasta 45 m2": "le_45m2",
    "Entre 46 y 60 m2": "46_60m2",
    "Menos de 60 m2": "under_60m2",
    "Entre 61 y 75 m2": "61_75m2",
    "Entre 76 y 90 m2": "76_90m2",
    "Entre 91 y 105 m2": "91_105m2",
    "Entre 106 y 120 m2": "106_120m2",
    "Entre 121 y 150 m2": "121_150m2",
    "Más de 150 m2": "over_150m2",
    # building type
    "Edificio de 2 o más viviendas": "multi_family",
    "Vivienda unifamiliar (chalet, adosado, pareado...)": "single_family",
    # tenure regime
    "Alquilada": "rented",
    "Cedida gratis o a bajo precio (por otro hogar, pagada por la empresa...) u otra forma": "ceded_or_other",
    "Propia por herencia o donación": "owned_inherited",
    "Propia, por compra, con pagos pendientes (hipotecas)": "owned_with_mortgage",
    "Propia, por compra, totalmente pagada": "owned_fully_paid",
    # rent / mortgage brackets
    "Menos de 200 euros": "under_200eur",
    "De 200 euros a menos de 500 euros": "200_500eur",
    "De 500 euros a menos de 700 euros": "500_700eur",
    "De 700 euros a menos de 1.000 euros": "700_1000eur",
    "De 1.000 euros o más": "ge_1000eur",
    # second home
    "En el mismo municipio que la primera residencia": "same_muni_as_primary",
    "En otra CCAA": "other_region",
    "En otra provincia, pero en la misma CCAA": "other_prov_same_region",
    "60 o más días": "60plus_days",
    "Entre 15 y 29 días": "15_29days",
    "Entre 30 y 59 días": "30_59days",
    "Menos de 15 días": "under_15days",
    # eco vehicle
    "No tiene vehículo ecológico (híbrido o eléctrico)": "no",
    "Tiene algún vehículo ecológico (híbrido o eléctrico)": "yes",
    # separated waste type
    "Envases": "packaging",
    "Orgánico": "organic",
    "Papel": "paper",
    "Vidrio": "glass",
    # paid domestic service / external help
    "No dispone de servicio doméstico": "no",
    "Sí, dispone de servicio doméstico": "yes",
    "No dispone de ayudas externas no remuneradas": "none",
    "Sí, dispone de ayudas de familiares, parientes, amigos, vecinos": "family_or_friends",
    "Sí, dispone de ayudas de otros, como servicios sociales o una ONG": "social_services_or_ngo",
    # heating type
    "No tiene calefacción": "none",
    "No tiene instalación de calefacción pero sí algún aparato que permite calentar alguna habitación (por ejemplo radiadores eléctricos)": "no_install_but_appliance",
    "Sí, colectiva": "collective",
    "Sí, individual": "individual",
    # fuel type
    "Electricidad": "electricity",
    "Gas natural": "natural_gas",
    "Otros": "other",
    "Petróleo o derivados (gasoil, fueloil, gasolina...)": "oil_derivatives",
    # water supply
    "Agua corriente por abastecimiento privado o particular del edificio": "private_supply",
    "Agua corriente por abastecimiento público": "public_supply",
    # internet connection
    "Conexión fija": "fixed",
    "Conexión fija y móvil": "fixed_and_mobile",
    "Conexión móvil": "mobile",
    "Sin conexión": "none",
    # appliances
    "Horno": "oven",
    "Lavadora": "washing_machine",
    "Lavavajillas": "dishwasher",
    "Microondas": "microwave",
    "Secadora": "dryer",
    "Vitocerámica/Inducción": "ceramic_induction_hob",
    # bulbs
    "Bajo consumo": "low_consumption",
    "Halógena": "halogen",
    "Led": "led",
    # toilet / bath
    "Aseo con inodoro": "toilet",
    "Bañera o ducha": "bath_or_shower",
    # area problems
    "Contaminación o malos olores": "pollution_odors",
    "Delincuencia": "crime",
    "Malas comunicaciones": "poor_transport",
    "Molestias relacionadas con actividades turísticas o locales de hostelería": "tourism_nuisance",
    "Poca limpieza en las calles": "dirty_streets",
    "Pocas zonas verdes": "few_green_areas",
    "Ruidos exteriores": "outdoor_noise",
    # nearby infrastructure / services
    "Colegio": "school",
    "Farmacia": "pharmacy",
    "Hospital, centros de salud o ambulatorio": "hospital_health_center",
    "Servicios de restauración (bares, restaurantes...)": "food_services",
    "Supermercado": "supermarket",
    # conservation status
    "Excelente": "excellent",
    "Muy bueno": "very_good",
    "Bueno": "good",
    "Normal": "normal",
    "Regular": "fair",
    "Malo": "bad",
    # accessibility
    "Es accesible": "accessible",
    "No es accesible": "not_accessible",
    # installation type
    "Agua caliente central": "central_hot_water",
    "Ascensor": "elevator",
    "Dispositivo de energía renovable": "renewable_device",
    "Garaje": "garage",
    "Gas por tubería": "piped_gas",
    # garage spaces
    "De 3 a 20": "3_20",
    "De 21 a 50": "21_50",
    "Más de 50": "over_50",
    # renewable energy device type
    "No dispone de dispositivo de energía renovable": "none",
    "Un único tipo de dispositivo: Energía solar fotovoltaica, eólica, biomasa, etc": "solar_pv_wind_biomass",
    "Un único tipo de dispositivo: Energía solar térmica": "solar_thermal",
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
        # Curated values are authored directly in the final short snake_case.
        return mapping[cleaned]
    if unmapped is not None and cleaned:
        unmapped.add(cleaned)
    # Unmapped fragments fall back to an accent-stripped, snake_cased form.
    return utils.to_snake_case(_camelize(cleaned))


def translate_ec_column(col, unmapped=None):
    """Translate one ``ec__…`` Spanish column name to English snake_case.

    Parses ``ec__{group} - {variable} ~ {subclass}`` and joins the translated
    parts as ``ec__{group}_{variable}_{subclass}`` (short snake_case, ``ec__``
    namespace prefix), e.g. ``ec__main_dwellings_heating_individual``.
    Non-``ec__`` names are returned unchanged. Unknown parts fall back to a
    snake_cased :func:`_camelize` (and are collected in ``unmapped`` when a set
    is passed).
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
    # Parts are already snake_case (mapped verbatim, or snaked in the fallback);
    # join them and re-prepend the ``ec__`` namespace (kept literal).
    return "ec__" + "_".join(p for p in parts if p)


def ec_column_rename_map(columns, warn_unmapped=False):
    """Build a ``{spanish_ec_column -> english_ec_column}`` map for ``columns``.

    Only ``ec__…`` columns are included. Collisions (two source columns mapping
    to the same target) are disambiguated with a ``_2``, ``_3``… suffix so no
    column is silently dropped. With ``warn_unmapped=True``, Spanish group/
    variable/subclass fragments missing from the dictionaries are reported to
    stderr (they still get an accent-stripped, snake_cased fallback).
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
    #   -> 'ec__main_dwellings_heating_individual'
    """
    mapping = ec_column_rename_map(df.columns, warn_unmapped=warn_unmapped)
    return df.rename(columns=mapping)
