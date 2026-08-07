"""Socioeconomic enrichment from the ``social_ES`` library (INE datasets).

This module attaches INE socioeconomic / demographic indicators to every
building, joining each source dataset **at its own finest geographic level**
(census tract → district → municipality → province) using the codes that
:func:`hypercadaster_ES.mergers.join_by_census_tracts` already added to the
frame (``census_tract_code`` = CUSEC, ``district_code`` = CUDIS). Municipality
(``census_tract_code[:5]``) and province (``census_tract_code[:2]``) codes are derived.

The public entry point is :func:`join_social_ES`, wired into
:func:`hypercadaster_ES.merge` through the ``social_ES_layer`` flag and executed
**before the POI join**. Output columns follow the ``so__{indicator}_{year}``
convention (``so__`` namespace prefix, snake_case indicator, year as a suffix),
e.g. ``so__net_income_per_household_2020``. Only the most recent ``n_years``
available per dataset are emitted.

Each indicator is taken from **one** dataset only — the one that publishes it
at the finest geography and the longest run of years — so the curated maps
below are deliberately non-overlapping: income and inequality come from the
income atlas, population and its demographic shares from the annual population
census, attainment and labour-force status from the education census, the
dwelling stock and its tenure from the 2021 dwellings census, and the
empty/secondary split from the three-census dwelling-use series.

Data comes from the ``social_ES`` package (EUPL-1.2, ≥ 1.0.3), which downloads
and caches the whole-Spain source files under ``{wd}/INE/...`` on first use;
subsequent runs read from that cache.
"""

import contextlib
import gc
import sys
import warnings

import numpy as np
import pandas as pd


# --- Curated indicator maps: {source column in social_ES -> clean output name} ---

# HouseholdIncomeDistributionAtlas (census-tract level, years 2015-2023).
# Income, its sources and its dispersion — the atlas is the only census-tract
# source for those, and it is built from tax records rather than a survey. Its
# demographic side (population, age structure, share of Spanish nationals) is
# NOT taken from here: the annual population census below publishes the same
# figures as head counts, per census tract, and three years further forward.
ATLAS_INDICATORS = {
    "Average household net income": "net_income_per_household",
    "Average household income": "gross_income_per_household",
    "Average net income per person": "net_income_per_person",
    "Average per person gross income": "gross_income_per_person",
    "Average income by unit of consumption": "net_income_per_consumption_unit",
    "Median income by unit of consumption": "median_income_per_consumption_unit",
    # Deflated to 2010 euros by social_ES, so income is comparable across years
    # (every other income column here is nominal, in euros of its own year).
    "Average household net income (2010 EUR)": "net_income_per_household_2010eur",
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
    "Percentage of single-person households": "pct_single_person_households",
    # The tract's monthly net household income, deflated by social_ES to 2010
    # euros, bucketed into the four bands the INE Time Use Survey respondents
    # answered in (1 = ≤1,200 €, 2 = 1,201-2,000 €, 3 = 2,001-3,000 €, 4 = >3,000 €).
    # It is the join key into the survey's schedules — see
    # :func:`time_use_household_occupancy`, which the energy model reads to build
    # its residential occupancy agenda.
    "Household income group": "household_income_group",
}

# EducationAndEmploymentCensus, mode="relative" (already percentages), census
# tract level, years 2021-2024. Attainment is NOT taken from here directly: the
# census publishes the four levels as disjoint shares and
# :func:`_education_shares` accumulates them into the ``pct_edu_lte_*`` KPIs
# below. Only the labour-force block is a straight pass-through.
EDU_INDICATORS = {
    "Population ~ Labour force status:Employed": "pct_employed",
    "Population ~ Labour force status:Unemployed": "pct_unemployed",
    "Population ~ Labour force status:Student": "pct_student",
    "Population ~ Labour force status:Recipient of disability, retirement, or early retirement pension": "pct_pensioner",
    "Population ~ Labour force status:Other inactive situation": "pct_other_inactive",
}

# The four attainment levels INE publishes, ordered from lowest to highest. Each
# source column is the share of the population aged 16+ whose *highest* completed
# level is that one, so the levels are disjoint and sum to 100.
_EDU_LEVELS = (
    ("Population ~ Educational level:Primary education or below", "primary_or_below"),
    ("Population ~ Educational level:Lower secondary education or equivalent", "lower_secondary"),
    ("Population ~ Educational level:Upper secondary and post-secondary non-tertiary education", "upper_secondary"),
    ("Population ~ Educational level:Tertiary education", "tertiary"),
)

# DwellingsAndPopulationCensus (census-tract level, 2021 only). Only the
# dwelling-stock / tenure / households block is taken: its population,
# education and labour-force indicators repeat, for a single year, what the
# population and education censuses above publish annually.
#
# Head counts read from the source (INE publishes this block as counts only);
# :func:`_dwelling_shares` turns them into the emitted indicators, which are the
# three magnitudes below plus the tenure composition. ``Non-main`` is read to
# check nothing is missing but not emitted — it is 100 % minus the main share.
_DWELLING_SOURCE_COLUMNS = {
    "Dwellings": "dwellings",
    "Dwellings ~ Dwelling type:Main": "dwellings_main",
    "Dwellings ~ Dwelling type:Non-main": "dwellings_non_main",
    "Main dwellings ~ Tenure:Owned": "main_dwellings_owned",
    "Main dwellings ~ Tenure:Rented": "main_dwellings_rented",
    "Main dwellings ~ Tenure:Other tenure": "main_dwellings_other_tenure",
    "Households": "households",
}

# The dwelling-stock indicators actually emitted, as ``so__{name}_2021``.
DWELLING_INDICATORS = (
    "dwellings", "dwellings_main", "households",
    "pct_dwellings_main", "pct_main_dwellings_owned", "pct_main_dwellings_rented",
    "pct_main_dwellings_other_tenure",
)

# EmptyAndSecondaryDwellingsCensus, municipality level, censuses 2001/2011/2021.
# The ``Comparable use`` classification is the one social_ES harmonises across
# the three censuses (2021 replaced the census agent's judgement with an
# electricity-consumption classification), so it is the only one read here; the
# raw ``Dwelling type`` / ``Electricity use`` splits it is built from would be
# the same dwellings counted twice. ``Predicted`` flags the rows whose
# secondary/empty split social_ES modelled because INE withholds it for small
# municipalities (see :data:`EMPTY_DWELLINGS_PREDICT`).
EMPTY_DWELLING_INDICATORS = {
    "Dwellings": "municipality_dwellings",
    "Percentage of dwellings ~ Comparable use:Main": "pct_dwellings_use_main",
    "Percentage of dwellings ~ Comparable use:Secondary": "pct_dwellings_use_secondary",
    "Percentage of dwellings ~ Comparable use:Empty": "pct_dwellings_use_empty",
    "Predicted": "dwelling_use_predicted",
}

# Ask social_ES to model the municipal figures INE withholds (the 2011
# secondary/empty split below 2,000 inhabitants, the 2021 electricity
# classification below 1,000) instead of leaving them NaN. The modelled shares
# are calibrated so each province reproduces the figure INE publishes for it;
# the ``so__dwelling_use_predicted_{year}`` column marks the affected rows so
# they can be excluded. The fit is cached by social_ES after the first run.
EMPTY_DWELLINGS_PREDICT = True

# HouseholdsRentalPriceIndex, district level with a municipality fallback,
# years 2011-2024. INE publishes it for ~672 municipalities only.
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

# PopulationCensus age bands used to build broad age-structure shares. The
# census publishes head counts by five-year band; the shares below are derived
# here (see :func:`_population_shares`).
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

# Census-tract/district/municipality/province indicator datasets (always joinable
# from the census codes alone).
ALL_DATASETS = ("atlas", "population", "education", "dwellings", "empty_dwellings",
                "rental", "price", "electricity")

# The per-building Essential Characteristics downscaling is a special dataset:
# it needs the CAT-based ``br__*`` columns, so it is only considered when
# ``use_CAT_files=True``. Selectable by name in ``social_ES_datasets`` alongside
# :data:`ALL_DATASETS`.
ESSENTIAL_CHARACTERISTICS = "essential_characteristics"

# Namespace prefix for the census-tract/district/municipality/province social_ES
# indicator columns (output columns are ``so__{indicator}_{year}``).
SOCIAL_PREFIX = "so__"


def release(*frames):
    """Drop a dataset's source tables as soon as its wide indicator table exists.

    ``social_ES`` hands back every geographic level of a dataset (census tract /
    district / municipality / province), of which only one is ever joined, and
    each of the eight otherwise stays alive until :func:`join_social_ES` returns.
    Emptying them in place frees the block manager's buffers rather than just this
    reference, so the release does not wait on the caller's own local going out of
    scope.

    Worth being clear about the size of this: the tables come back already
    filtered to the municipalities being enriched, so the total held is tens of
    megabytes (~30 MB for Barcelona across the eight datasets), not gigabytes. The
    peak of the layer is inside ``social_ES`` itself, while it reads the national
    source files, and nothing here can reach that. This is housekeeping, not a
    fix for a memory ceiling.

    The disk cache under ``{wd}/INE`` is untouched: a later dataset that needs the
    same file re-reads it instead of re-downloading it.
    """
    for f in frames:
        try:
            if isinstance(f, dict):
                f.clear()
            elif isinstance(f, pd.DataFrame):
                f.drop(f.index, inplace=True)
        except Exception:  # noqa: BLE001 - freeing memory must never fail a join
            pass
    del frames
    gc.collect()


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


def _code(series, width):
    """Normalise an INE geographic code to a fixed-width, zero-padded string.

    Whitespace is stripped first: several INE tables pad their codes on the
    right (the rental index publishes districts as ``"01 "``), and zero-padding
    alone would carry that through into a key that matches nothing.
    """
    return series.astype(str).str.strip().str.zfill(width)


def _census_tract_key(df):
    """Build a 10-digit CUSEC key from Municipality/District/Census tract codes."""
    return (_code(df["Municipality code"], 5)
            + _code(df["District code"], 2)
            + _code(df["Census tract code"], 3))


def _column(df, name):
    """``df[name]`` as a numeric Series, or an all-NaN Series when absent."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _num(df[name])


def _population_shares(census_tracts):
    """Population count and broad demographic shares from the PopulationCensus
    census-tract table.

    The head count is emitted as ``population`` (this is the source of
    ``so__population_{year}``, which the energy model reads as its occupancy
    driver); everything else is a percentage of it.
    """
    pop = _num(census_tracts["Population"]).replace(0, np.nan)

    def share(cols):
        present = [c for c in cols if c in census_tracts.columns]
        if not present:
            return pd.Series(np.nan, index=census_tracts.index)
        return census_tracts[present].apply(_num).sum(axis=1) / pop * 100

    out = pd.DataFrame({
        "Year": census_tracts["Year"],
        "_key": _census_tract_key(census_tracts),
        "population": _num(census_tracts["Population"]),
        "pct_foreign_nationality": share(["Population ~ Nationality:Foreign"]),
        "pct_foreign_born": share(["Population ~ Birth origin:Extranjero"]),
        "pct_born_same_municipality": share(
            ["Population ~ Birth origin in Spain:Born in the same municipality"]),
        # Women only: the census splits the population in two, so the male share
        # is 100 minus this one and carrying both would be redundant.
        "pct_woman": share(["Population ~ Sex:Females"]),
        "pct_under_15": share(_AGE_UNDER_15),
        "pct_working_age": share(_AGE_WORKING),
        "pct_65_plus": share(_AGE_65_PLUS),
    })
    return out


def _education_shares(census_tracts):
    """Cumulative educational-attainment shares from the EducationAndEmploymentCensus
    census-tract table, plus the labour-force block unchanged.

    INE publishes attainment as four *disjoint* shares of the population aged 16
    and over — the share whose highest completed level is each one. Those answer
    "how many stopped exactly here", which is awkward to read across tracts
    because a tract can look better or worse depending on where the mass sits.
    The emitted ``pct_edu_lte_{level}`` KPIs are their running sum instead: the
    share of the population that reached **at most** that level, so each one is a
    monotonically increasing ladder ending at ``pct_edu_lte_tertiary`` = 100 %
    wherever the four source shares are complete.

    A level missing from the source leaves every KPI from it upwards NaN rather
    than silently under-counting the ones above it.
    """
    out = {"Year": census_tracts["Year"], "_key": _census_tract_key(census_tracts)}
    cumulative = None
    for src, level in _EDU_LEVELS:
        share = _column(census_tracts, src)
        cumulative = share if cumulative is None else cumulative + share
        out[f"pct_edu_lte_{level}"] = cumulative
    for src, clean in EDU_INDICATORS.items():
        out[clean] = _column(census_tracts, src)
    return pd.DataFrame(out)


def _dwelling_shares(census_tracts):
    """Dwelling-stock counts plus occupancy and tenure shares from the 2021
    DwellingsAndPopulationCensus census-tract table.

    INE publishes this block as head counts only. The shares are derived here:
    the main-dwelling share against all dwellings of the tract, tenure against
    the main dwellings it actually describes (the three tenure classes
    partition them). Only the columns listed in :data:`DWELLING_INDICATORS` are
    emitted; the rest are read to build those.
    """
    counts = {clean: _column(census_tracts, src)
              for src, clean in _DWELLING_SOURCE_COLUMNS.items()}
    dwellings = counts["dwellings"].replace(0, np.nan)
    tenure = ["main_dwellings_owned", "main_dwellings_rented",
              "main_dwellings_other_tenure"]
    tenure_total = sum(counts[t] for t in tenure).replace(0, np.nan)

    derived = {"pct_dwellings_main": counts["dwellings_main"] / dwellings * 100}
    for name in tenure:
        derived[f"pct_{name}"] = counts[name] / tenure_total * 100

    out = pd.DataFrame({
        "Year": census_tracts["Year"],
        "_key": _census_tract_key(census_tracts),
        **{k: v for k, v in {**counts, **derived}.items() if k in DWELLING_INDICATORS},
    })
    return out


# Columns the social_ES EssentialCharacteristics downscaling needs. The ``br__*``
# come from the CAT-based building-space aggregation (``use_CAT_files=True``).
_EC_REQUIRED_COLS = (
    "census_tract_code", "building_reference", "ine_code", "year_of_construction",
    "br__area_without_communals", "br__building_spaces",
)

# Namespace prefix for the per-building essential-characteristics proportions.
EC_PREFIX = "ec__"

# The ECEPOV variables kept by default, out of the ~70 ``social_ES`` publishes
# (256 ``ec__{variable}~{class}`` columns in total). The criterion is that the
# variable says something the ``so__`` layer above and the cadaster itself do
# NOT already say — the survey is a modelled downscaling, so where an indicator
# is available as a census head count (``so__``) or as a measured building
# attribute (``br__``), that source wins and the survey variable is dropped.
#
# Set ``ec_variables=None`` on :func:`join_essential_characteristics` to keep
# all of them instead, or pass an explicit sequence to select your own.
EC_SELECTED_VARIABLES = (
    # -- Dwelling energy systems: the survey is the ONLY source for these; the
    # cadaster records geometry and use, never the installed systems, and the
    # census publishes no heating/cooling question at tract level. They drive
    # the energy model directly (system type, fuel, whether cooling exists).
    "main_dwellings_heating_type",
    "main_dwellings_heating_fuel",
    "main_dwellings_has_cooling",
    "main_dwellings_insulation_issues",
    "main_dwellings_renewable_device",
    "main_dwellings_installations",          # elevator / garage / piped gas / DHW / renewables
    "main_dwellings_lighting_type",          # halogen vs low-consumption vs LED
    "main_dwellings_domestic_appliances_type",
    "main_dwellings_water_supply",
    # -- Building condition and accessibility. ``conservation_status`` already
    # feeds the model's infiltration factor (``energy._building_conservation``);
    # the other two qualify the retrofit/ageing-in-place potential of the stock.
    "main_dwellings_conservation_status",
    "main_dwellings_accessibility_status",
    "main_dwellings_adapted_to_elderly",
    # -- Dwelling interior. The cadaster gives floor area but not what is inside
    # it, and bathrooms/kitchen size drive the DHW demand and the internal gains.
    "main_dwellings_n_bathrooms",
    "main_dwellings_kitchen_sized_4m2plus",
    "main_dwellings_n_rooms",
    "main_dwellings_internet_connection",
    # -- Presence and absence. ``so__pct_dwellings_use_secondary/_empty`` is a
    # municipal figure about the dwelling being enriched; these say how often the
    # households living in it are somewhere else, which is an occupancy input.
    "households_second_home_ownership",
    "households_secondary_second_home_occupancy",
    "people16plus_work_or_study_place",
    "people16plus_transport_daily_time",
    # -- Mobility and vehicles: the electrification demand of the building's
    # parking, absent from every other layer.
    "households_vehicles_number",
    "households_vehicles_eco",
    "people16plus_transport_mode",
    # -- Housing cost burden. ``so__net_income_per_household`` gives income and
    # ``so__rental_price_index`` a district-level index; neither gives what the
    # households in THIS building actually pay, which is what makes energy
    # poverty computable alongside the income columns.
    "households_rented_monthly_rent",
    "households_owned_monthly_mortgage",
    # -- Household composition beyond ``so__average_household_size`` /
    # ``so__pct_single_person_households``: children set the occupancy pattern.
    "households_n_children",
    # -- Neighbourhood quality (noise, pollution, green areas, crime …): the only
    # perceived-environment variable in the whole stack.
    "main_dwellings_surroundings_issues",
)

# Dropped on purpose, and why — each is already answered better elsewhere:
#   households_tenure, households_building_type, main_dwellings_floor_area,
#   main_dwellings_n_garages_places       -> so__pct_main_dwellings_* / the cadaster
#   households_single_person_*            -> so__pct_single_person_households plus
#                                            so__ income / education / labour force
#   cohabiting_couples_*, people_parents_*, people_born_*
#                                         -> so__pct_foreign_* / so__pct_born_*
#   people16plus_has_smartphone / has_social_networks / internet_access /
#   online_shopping / online_selling      -> main_dwellings_internet_connection
#   *_daycare_*, housework_involvement, has_social_support, with_paid_domestic_service,
#   with_unpaid_external_help, waste_*, transport_satisfaction / _type / _daily_trips,
#   second_home_place, toilet_bath, infrastructure_service, parents_education
#                                         -> outside what this library models


def join_social_ES(gdf, wd, n_years=5, datasets=None, use_CAT_files=False,
                   ec_variables=EC_SELECTED_VARIABLES, verbose=True):
    """Join ``social_ES`` (INE) socioeconomic indicators onto every building.

    Each dataset is joined at its finest geographic level using the codes already
    present on ``gdf`` (``census_tract_code``/``district_code``, plus municipality and
    province derived from them). Output columns are ``so__{indicator}_{year}`` for
    the most recent ``n_years`` available per dataset (see the ``*_INDICATORS``
    maps in this module for the curated indicator set; indicators are snake_case).

    The datasets and what each one contributes — no indicator is taken from two
    of them:

    ``atlas``
        Income atlas, census tract, 2015-2023: average/median income per
        household, person and consumption unit (nominal, plus a 2010-euro
        series), the share of household income by source, Gini and P80/P20,
        mean age, household size, and the Time Use Survey household income band
        (``so__household_income_group_{year}``).
    ``population``
        Annual population census, census tract, 2021-2025: the head count
        (``so__population_{year}``) and the shares by nationality, birth
        origin, sex (``so__pct_woman_{year}``) and broad age band.
    ``education``
        Education & employment census, census tract, 2021-2024: educational
        attainment as the **cumulative** ``so__pct_edu_lte_{level}_{year}``
        ladder (share of the population aged 16+ that reached at most that
        level; see :func:`_education_shares`) and labour-force status, as shares
        of the population each one describes.
    ``dwellings``
        2021 dwellings census, census tract: the dwelling stock, its
        main-dwelling share and the tenure of those main dwellings.
    ``empty_dwellings``
        Empty and secondary dwellings, municipality, censuses 2001/2011/2021:
        the main/secondary/empty split harmonised across the three censuses.
        The split INE withholds for small municipalities is modelled by
        ``social_ES`` (:data:`EMPTY_DWELLINGS_PREDICT`); the affected rows are
        flagged in ``so__dwelling_use_predicted_{year}``.
    ``rental``
        Rental price index, by district where INE publishes it and by
        municipality elsewhere.
    ``price``
        Housing price index, province, quarterly averaged to annual.
    ``electricity``
        Household electricity consumption percentiles, district, 2021.

    The per-building **Essential Characteristics** downscaling
    (:data:`ESSENTIAL_CHARACTERISTICS`, → ``ec__*`` columns) is included only when
    ``use_CAT_files=True`` **and** it is selected (``datasets`` is ``None`` or
    lists ``"essential_characteristics"``); it needs the CAT-based ``br__*``
    columns. If it is explicitly requested without ``use_CAT_files`` it is skipped
    with a warning.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame; must carry ``census_tract_code`` (from
        :func:`join_by_census_tracts`). ``district_code`` is used when present,
        otherwise derived from ``census_tract_code``.
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
    ec_variables : sequence[str] or None, default :data:`EC_SELECTED_VARIABLES`
        Essential-Characteristics variables to keep; ``None`` keeps all of them.
    verbose : bool, default True
        Emit per-dataset progress/matched-row counts to stderr.

    Returns
    -------
    geopandas.GeoDataFrame
        ``gdf`` with the socioeconomic columns added. Degrades gracefully: if
        ``social_ES`` is missing, ``census_tract_code`` is absent, or an individual
        dataset fails to download/parse, that part is skipped with a warning and
        the frame is returned with whatever joined successfully.
    """
    # Resolve which census-tract/district datasets to run, and whether the Essential
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

    if "census_tract_code" not in gdf.columns:
        sys.stderr.write("social_ES: no census_tract_code column; skipping social layer\n")
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
    # Normalised join keys (scalars, even after address compaction), stripped so
    # they compare equal to the ``_code``-built keys of the source tables.
    sec = gdf["census_tract_code"].map(_scalar).str.strip()
    if "district_code" in gdf.columns:
        dist = gdf["district_code"].map(_scalar).str.strip()
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

    def _merge(wide, key_col, fill=False):
        """Join a wide indicator table (key in its first column) onto ``gdf``.

        ``fill=True`` marks the table as a coarser-level fallback for one already
        joined: columns it shares with ``gdf`` are only used where the finer join
        left a NaN. Otherwise a shared column is replaced outright, so re-running
        the layer overwrites instead of producing ``_x``/``_y`` suffixes.
        """
        nonlocal gdf
        if wide is None or wide.empty:
            return 0
        wide = wide.rename(columns={wide.columns[0]: key_col})
        wide[key_col] = wide[key_col].astype(str)
        wide = wide.drop_duplicates(subset=[key_col])
        new_cols = [c for c in wide.columns if c != key_col]
        overlap = [c for c in new_cols if c in gdf.columns]
        n_match = int(gdf[key_col].isin(wide[key_col]).sum())
        if overlap:
            # Positional alignment: the left merge on a de-duplicated key keeps
            # gdf's row order and length, whatever its index looks like.
            aligned = gdf[[key_col]].merge(wide[[key_col] + overlap], on=key_col,
                                           how="left")
            for c in overlap:
                values = pd.Series(aligned[c].to_numpy(), index=gdf.index)
                gdf[c] = gdf[c].fillna(values) if fill else values
            wide = wide.drop(columns=overlap)
            new_cols = [c for c in new_cols if c not in overlap]
        if new_cols:
            gdf = gdf.merge(wide, on=key_col, how="left")
        return n_match

    # --- Atlas (census tract) ---
    if "atlas" in run_simple:
        try:
            log("HouseholdIncomeDistributionAtlas ...")
            source = INE.HouseholdIncomeDistributionAtlas(wd, municipality_code=muni_list)
            atlas = source["Census tracts"]
            atlas["_key"] = _census_tract_key(atlas)
            wide = _pivot_wide(atlas, "_key", ATLAS_INDICATORS, n_years)
            release(source, atlas)
            log(f"atlas: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: atlas failed ({e}); skipping\n")

    # --- Population census (census tract) ---
    if "population" in run_simple:
        try:
            log("PopulationCensus ...")
            source = INE.PopulationCensus(wd, municipality_code=muni_list)
            pop = source["Census tracts"]
            shares = _population_shares(pop)
            release(source, pop)
            share_map = {c: c for c in shares.columns if c not in ("Year", "_key")}
            wide = _pivot_wide(shares, "_key", share_map, n_years)
            release(shares)
            log(f"population: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: population census failed ({e}); skipping\n")

    # --- Education & employment census (census tract, relative %) ---
    if "education" in run_simple:
        try:
            log("EducationAndEmploymentCensus ...")
            source = INE.EducationAndEmploymentCensus(
                wd, municipality_code=muni_list, mode="relative")
            edu = source["Census tracts"]
            shares = _education_shares(edu)
            release(source, edu)
            share_map = {c: c for c in shares.columns if c not in ("Year", "_key")}
            wide = _pivot_wide(shares, "_key", share_map, n_years)
            release(shares)
            log(f"education: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: education census failed ({e}); skipping\n")

    # --- Dwelling stock & tenure (census tract, 2021 census) ---
    if "dwellings" in run_simple:
        try:
            log("DwellingsAndPopulationCensus ...")
            source = INE.DwellingsAndPopulationCensus(wd, municipality_code=muni_list)
            dwe = source["Census tracts"]
            shares = _dwelling_shares(dwe)
            release(source, dwe)
            share_map = {c: c for c in shares.columns if c not in ("Year", "_key")}
            wide = _pivot_wide(shares, "_key", share_map, n_years)
            release(shares)
            log(f"dwellings: matched {_merge(wide, '_sec_key')} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: dwellings census failed ({e}); skipping\n")

    # --- Empty & secondary dwellings (municipality, censuses 2001/2011/2021) ---
    if "empty_dwellings" in run_simple:
        try:
            log("EmptyAndSecondaryDwellingsCensus ...")
            # The whole-Spain table is needed before filtering: social_ES fits the
            # models on every municipality that carries a split and calibrates them
            # against whole provinces, so municipality_code is applied afterwards.
            source = INE.EmptyAndSecondaryDwellingsCensus(
                wd, municipality_code=muni_list,
                predict=EMPTY_DWELLINGS_PREDICT)
            empty = source["Municipality"]
            if _require(empty, ["Municipality code"], "empty_dwellings"):
                empty = empty.copy()
                empty["_key"] = _code(empty["Municipality code"], 5)
                wide = _pivot_wide(empty, "_key", EMPTY_DWELLING_INDICATORS, n_years)
                release(source, empty)
                log(f"empty_dwellings: matched {_merge(wide, '_muni_key')} buildings")
            else:
                release(source)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"social_ES: empty & secondary dwellings failed ({e}); skipping\n")

    # --- Household rental price index (district, municipality as fallback) ---
    if "rental" in run_simple:
        try:
            log("HouseholdsRentalPriceIndex ...")
            rental = INE.HouseholdsRentalPriceIndex(wd, municipality_code=muni_list)
            n_rental = 0
            districts = rental.get("Districts")
            if _require(districts, ["Municipality code", "District code"], "rental"):
                districts = districts.copy()
                districts["_key"] = (_code(districts["Municipality code"], 5)
                                     + _code(districts["District code"], 2))
                wide = _pivot_wide(districts, "_key", RENTAL_INDICATORS, n_years)
                n_rental = _merge(wide, "_dist_key")
            # INE publishes the index by district only for the larger cities; the
            # municipal series covers the rest, so it fills what the district join
            # left empty rather than overwriting it.
            municipal = rental.get("Municipality")
            if _require(municipal, ["Municipality code"], "rental"):
                municipal = municipal.copy()
                municipal["_key"] = _code(municipal["Municipality code"], 5)
                wide = _pivot_wide(municipal, "_key", RENTAL_INDICATORS, n_years)
                n_rental = max(n_rental, _merge(wide, "_muni_key", fill=True))
            release(rental)
            log(f"rental: matched {n_rental} buildings")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: rental price index failed ({e}); skipping\n")

    # --- Housing price index (province, quarterly -> annual mean) ---
    if "price" in run_simple:
        try:
            log("HouseholdsPriceIndex ...")
            source = INE.HouseholdsPriceIndex(wd, municipality_code=muni_list)
            price = source["Province"]
            if _require(price, ["Province code"], "price"):
                price = price.dropna(subset=["Province code"]).copy()
                price["_key"] = _code(price["Province code"], 2)
                wide = _pivot_wide(price, "_key", PRICE_INDICATORS, n_years)
                release(source, price)
                log(f"price: matched {_merge(wide, '_prov_key')} buildings")
            else:
                release(source)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: housing price index failed ({e}); skipping\n")

    # --- Aggregated electricity consumption (district) ---
    if "electricity" in run_simple:
        try:
            log("AggregatedElectricityConsumption ...")
            source = INE.AggregatedElectricityConsumption(wd, municipality_code=muni_list)
            elec = source["Districts"]
            if _require(elec, ["Municipality code", "District code"], "electricity"):
                elec["_key"] = (_code(elec["Municipality code"], 5)
                                + _code(elec["District code"], 2))
                wide = _pivot_wide(elec, "_key", ELECTRICITY_INDICATORS, n_years)
                release(source, elec)
                log(f"electricity: matched {_merge(wide, '_dist_key')} buildings")
            else:
                release(source)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"social_ES: electricity consumption failed ({e}); skipping\n")

    gdf = gdf.drop(columns=["_sec_key", "_dist_key", "_muni_key", "_prov_key"],
                   errors="ignore")

    # --- Essential Characteristics per-building downscaling (needs CAT br__*) ---
    if include_ec:
        gdf = join_essential_characteristics(gdf, wd, ec_variables=ec_variables,
                                             verbose=verbose)

    return gdf


def join_essential_characteristics(gdf, wd, prefix=EC_PREFIX,
                                   ec_variables=EC_SELECTED_VARIABLES, verbose=True):
    """Downscale the INE *Essential Characteristics of Population and Households*
    survey to the building level and join it onto ``gdf``.

    Wraps ``social_ES.INE.EssentialCharacteristicsOfPopulationAndHouseholds``,
    which allocates census-tract-level survey proportions (age, income band, household
    nationality, dwelling size, building type/age, heating, vehicles, internet
    access, …) to individual buildings using the CAT-based building-space
    aggregation (``br__area_without_communals`` / ``br__building_spaces`` per use
    type, ``year_of_construction``). Each downscaled variable becomes a
    ``{prefix}{…}`` column (proportions in %, one group summing to 100).

    Since ``social_ES`` 1.0.3 the survey's Spanish INE labels are already
    translated to short English snake_case at source, and the variables it
    publishes twice (once counting households, once counting main dwellings)
    are already dropped there. **The names it returns are kept exactly as given**
    — the ``{variable}~{class}`` separator included — so the only thing added
    here is the namespace prefix: ``main_dwellings_heating_type~individual`` →
    ``ec__main_dwellings_heating_type~individual``. Reading a column name
    therefore tells you, unambiguously, which variable it belongs to (everything
    left of the ``~``) and which class of it the proportion measures.

    By default only the variables in :data:`EC_SELECTED_VARIABLES` are kept: the
    ones that add something the ``so__`` census layer and the cadaster do not
    already carry. Pass ``ec_variables=None`` for all ~70 of them.

    Only meaningful when the CAT-based building-space aggregation has run
    (``use_CAT_files=True``) — the required ``br__*`` columns are otherwise
    absent, in which case the frame is returned unchanged with a warning. It is
    invoked from :func:`join_social_ES` when ``use_CAT_files=True`` and the
    ``"essential_characteristics"`` dataset is selected.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building frame carrying the inference outputs and ``census_tract_code``.
    wd : str
        hypercadaster working directory (``social_ES`` reads its cached whole-Spain
        atlas/population files from ``{wd}/INE/...``).
    prefix : str, default ``"ec__"``
        Namespace prefix for the added proportion columns.
    ec_variables : sequence[str] or None, default :data:`EC_SELECTED_VARIABLES`
        Survey variables to keep, named as ``social_ES`` names them (the part of
        a column left of the ``~``). ``None`` keeps every variable.
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

    # The social_ES downscaling merges on a *scalar* census_tract_code; after address
    # compaction it can be a 1-element list. Pass a copy with a scalar key.
    work = gdf.copy()
    work["census_tract_code"] = work["census_tract_code"].map(_scalar)
    work = work[work["census_tract_code"].notna()]
    if work.empty:
        log("essential characteristics: no buildings with a census_tract_code; skipping")
        return gdf

    try:
        log("EssentialCharacteristicsOfPopulationAndHouseholds ...")
        # social_ES divides each variable by its group row-sum to get a
        # proportion; groups with a zero row-sum (no dwellings/people of that
        # kind in a census tract) yield 0/0 -> NaN and numpy floods stderr with
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

    # Returns a dict {variable -> DataFrame indexed by building_reference}, keyed
    # by the variable name social_ES gives it. Keep the selected variables (all of
    # them when ``ec_variables`` is None) and stitch them into one wide frame.
    if ec_variables is None:
        keep = list(proportions)
    else:
        wanted = set(ec_variables)
        keep = [v for v in proportions if v in wanted]
        unknown = sorted(wanted - set(proportions))
        if unknown:
            log(f"essential characteristics: requested variable(s) not published "
                f"by social_ES, ignored: {', '.join(unknown)}")
    frames = [proportions[v] for v in keep
              if proportions[v] is not None and not proportions[v].empty]
    release(proportions)
    if not frames:
        log("essential characteristics: empty result; skipping")
        return gdf
    ec = pd.concat(frames, axis=1)
    ec = ec.loc[:, ~ec.columns.duplicated()]
    # social_ES already names these in English snake_case, joining the class a
    # column measures to its variable with "~". Those names are kept verbatim;
    # only the namespace prefix is ours.
    ec.columns = [f"{prefix}{c}" for c in ec.columns]
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


# --------------------------------------------------------------------------- #
# INE Time Use Survey (EET 2009-2010) — weekly household occupancy agenda
# --------------------------------------------------------------------------- #
#
# ``social_ES.INE.TimeUseSurvey`` turns the survey microdata into a weekly hourly
# activity schedule per (autonomous community, household income band) cell, and
# publishes the ``WeeklyScheduleHouseholdOccupancy`` table: for each cell, the
# dominant household occupancy STATE at every one of the 168 hours of the week
# (hour 0 = Monday 00:00), one of:
#
#   ``Sleeping``      — at home, asleep (personal care at night)
#   ``WithActivity``  — at home, awake and doing something
#   ``NonOccupied``   — out (paid work, study, social life, sport, travel …)
#
# A census tract is matched to a cell by its autonomous community and by the
# income band of its Atlas household income, deflated to 2010 euros — the same
# ``Household income group`` this module already joins as
# ``so__household_income_group_{year}`` (see :data:`ATLAS_INDICATORS`).
#
# :func:`time_use_household_occupancy` is what the energy model calls to give its
# residential dwellings an occupancy agenda measured on real households rather
# than a stylised profile.

# The three occupancy states of the weekly agenda, in "most present" order.
TIME_USE_OCCUPANCY_STATES = ("Sleeping", "WithActivity", "NonOccupied")

# One week at hourly resolution.
_WEEK_HOURS = 168

# Cache of the per-municipality survey tables, keyed by (wd, municipality code):
# the tables are small (one row per (community, income band) cell) but building
# them re-reads the whole-Spain income atlas, which is not something to repeat
# once per building of a run.
_TIME_USE_CACHE = {}


@contextlib.contextmanager
def _social_es_numeric_round():
    """Work around a ``social_ES`` 1.0.3 bug that makes ``TimeUseSurvey`` raise.

    The function seeds its ``Sleeping_share`` / ``WithActivity_share`` /
    ``NonOccupied_share`` columns with ``None`` and fills them cell by cell, so
    they keep pandas' ``object`` dtype, and the ``.round(2)`` it then calls raises
    ``TypeError: Expected numeric dtype, got object instead``. The whole function
    is therefore unusable in that release, with no argument that avoids it.

    Rather than reimplement it, this makes ``Series.round`` coerce an object
    column to numeric before rounding, for the duration of the call only. Once
    ``social_ES`` fixes the dtype the shim simply never fires — it only touches
    columns that are already the wrong dtype.
    """
    original = pd.Series.round

    def round_coercing_objects(self, decimals=0, *args, **kwargs):
        if self.dtype == object:
            try:
                self = pd.to_numeric(self)
            except (TypeError, ValueError):
                return self          # genuinely non-numeric: leave it alone
        return original(self, decimals, *args, **kwargs)

    pd.Series.round = round_coercing_objects
    try:
        yield
    finally:
        pd.Series.round = original


def _time_use_tables(wd, municipality_code):
    """``(tracts, occupancy, shares)`` from ``social_ES.INE.TimeUseSurvey`` for one
    municipality, or ``None`` when the survey is unavailable.

    ``tracts`` maps each census tract to its (autonomous community, household
    income group) cell, ``occupancy`` holds the 168 hourly state labels per cell
    and ``shares`` the ``{state: DataFrame}`` percentage of households in each
    state at each hour. Cached per (wd, municipality).
    """
    key = (str(wd), str(municipality_code))
    if key in _TIME_USE_CACHE:
        return _TIME_USE_CACHE[key]

    result = None
    try:
        from social_ES import INE
        with _social_es_numeric_round():
            tus = INE.TimeUseSurvey(wd, municipality_code=[str(municipality_code)])
        tracts = tus["Census tracts"]
        occupancy = tus["WeeklyScheduleHouseholdOccupancy"]
        shares = tus.get("WeeklyScheduleHouseholdOccupancyValue") or {}
        if tracts is not None and not tracts.empty and occupancy is not None:
            result = (tracts, occupancy, shares)
        # The weekly/holiday schedules are not used here and are the bulk of it.
        release(tus.get("WeeklySchedule"), tus.get("HolidaySchedule"))
    except ImportError:
        sys.stderr.write("social_ES: package not installed; no time-use occupancy\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"social_ES: time use survey failed ({e}); "
                         "no time-use occupancy\n")

    _TIME_USE_CACHE[key] = result
    return result


def _cell_row(table, community_code, income_group):
    """The ``table`` row for one (autonomous community, income group) cell, or
    ``None``. Codes are compared as strings and groups as integers, because the
    survey publishes the community zero-padded ("09") and the band as a number."""
    if table is None or table.empty:
        return None
    match = table[
        (table["Autonomous community code"].astype(str).str.zfill(2)
         == str(community_code).zfill(2))
        & (_num(table["Household income group"]) == int(income_group))]
    return match.iloc[0] if len(match) else None


def _hour_values(row, cast):
    """The 168 hour-of-week values of a schedule row, in order, or ``None`` if the
    week is not complete (the survey publishes every hour, so a gap means the cell
    was not usable)."""
    if row is None:
        return None
    out = []
    for hour in range(_WEEK_HOURS):
        value = row.get(hour, row.get(str(hour)))
        try:
            value = cast(value)
        except (TypeError, ValueError):
            return None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        out.append(value)
    return out


def time_use_household_occupancy(wd, census_tract_code):
    """Weekly household occupancy agenda of a census tract, from the INE Time Use
    Survey via ``social_ES``.

    The tract is matched to the survey cell of its autonomous community and of the
    household income band its Atlas income falls into, and that cell's
    ``WeeklyScheduleHouseholdOccupancy`` row is returned: the dominant occupancy
    state at each of the 168 hours of the week, hour 0 being Monday 00:00.

    Parameters
    ----------
    wd : str
        hypercadaster working directory (``social_ES`` caches the survey profiles
        under ``{wd}/INE/TimeUseSurvey/``).
    census_tract_code : str
        10-digit CUSEC of the tract. May also arrive as a 1-element list (address
        compaction), which is unwrapped.

    Returns
    -------
    dict or None
        ``{"states": [168 state names], "shares": {state: [168 percentages]},
        "autonomous_community_code": str, "household_income_group": int,
        "household_income_group_label": str}``, or ``None`` when the survey, the
        tract or its income band is unavailable — every caller must have a
        fallback, since the Atlas does not cover every tract.
    """
    sec = _scalar(census_tract_code)
    if not sec or len(str(sec).strip()) < 10:
        return None
    sec = str(sec).strip()

    tables = _time_use_tables(wd, sec[:5])
    if tables is None:
        return None
    tracts, occupancy, shares = tables

    row = tracts[tracts["Census tract full code"].astype(str) == sec]
    if not len(row):
        return None
    row = row.iloc[0]
    community = row.get("Autonomous community code")
    group = row.get("Household income group")
    try:
        group = int(group)
    except (TypeError, ValueError):
        return None

    states = _hour_values(_cell_row(occupancy, community, group), str)
    if states is None:
        return None

    state_shares = {}
    for state in TIME_USE_OCCUPANCY_STATES:
        values = _hour_values(_cell_row(shares.get(state), community, group), float)
        if values is not None:
            state_shares[state] = values

    return {
        "states": states,
        "shares": state_shares,
        "autonomous_community_code": str(community),
        "household_income_group": group,
        "household_income_group_label": row.get("Household income group label"),
    }
