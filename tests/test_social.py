"""Offline unit tests for the social_ES enrichment helpers.

These exercise the pure transformation logic (key building, year-suffix
pivoting, share computation, graceful degradation) without hitting the network
or requiring the ``social_ES`` package to be installed.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from hypercadaster_ES import social


def test_scalar_normalises_lists_and_nan():
    assert social._scalar("0801901001") == "0801901001"
    # 1-element list, as produced by address compaction
    assert social._scalar(["0801901001"]) == "0801901001"
    assert social._scalar(np.array(["08019"])) == "08019"
    assert social._scalar([]) is None
    assert social._scalar(None) is None
    assert social._scalar(float("nan")) is None


def test_code_strips_before_padding():
    # INE pads some codes on the right — the rental index publishes districts as
    # "01 " — and zero-padding alone carried the space into a key that matched
    # nothing, silently emptying the district-level joins.
    codes = social._code(pd.Series(["01 ", " 1", "01", 1]), 2)
    assert list(codes) == ["01", "01", "01", "01"]
    assert list(social._code(pd.Series(["8019 "]), 5)) == ["08019"]


def test_census_tract_key_zero_pads():
    df = pd.DataFrame({
        "Municipality code": ["8019", "08019"],
        "District code": ["1", "01"],
        "Census tract code": ["1", "001"],
    })
    keys = social._census_tract_key(df)
    assert list(keys) == ["0801901001", "0801901001"]
    assert (keys.str.len() == 10).all()


def test_pivot_wide_year_suffix_and_last_years():
    long = pd.DataFrame({
        "_key": ["A", "A", "A", "B"],
        "Year": [2018, 2019, 2020, 2020],
        "Average household net income": [100.0, 200.0, 300.0, 400.0],
    })
    wide = social._pivot_wide(
        long, "_key", {"Average household net income": "net_income_per_household"}, n_years=2)
    # Only the two most recent years (2019, 2020) survive; suffix is the year,
    # and the ``so__`` namespace prefix is prepended.
    assert set(wide.columns) == {"_key", "so__net_income_per_household_2019",
                                 "so__net_income_per_household_2020"}
    row_a = wide[wide["_key"] == "A"].iloc[0]
    assert row_a["so__net_income_per_household_2019"] == 200.0
    assert row_a["so__net_income_per_household_2020"] == 300.0
    # B has no 2019 value -> NaN, not an error
    row_b = wide[wide["_key"] == "B"].iloc[0]
    assert np.isnan(row_b["so__net_income_per_household_2019"])


def test_pivot_wide_missing_source_columns_returns_none():
    long = pd.DataFrame({"_key": ["A"], "Year": [2020], "Other": [1.0]})
    assert social._pivot_wide(long, "_key", {"Absent": "X"}, n_years=5) is None


def test_population_shares_percentages():
    census_tracts = pd.DataFrame({
        "Municipality code": ["08019"],
        "District code": ["01"],
        "Census tract code": ["001"],
        "Year": [2021],
        "Population": [1000.0],
        "Population ~ Nationality:Foreign": [150.0],
        "Population ~ Birth origin:Extranjero": [200.0],
        "Population ~ Sex:Females": [520.0],
        "Population ~ Age:0-4": [50.0],
        "Population ~ Age:5-9": [50.0],
        "Population ~ Age:10-14": [50.0],
        "Population ~ Age:65-69": [100.0],
    })
    out = social._population_shares(census_tracts)
    row = out.iloc[0]
    assert row["_key"] == "0801901001"
    # The head count comes from here now, not from the income atlas.
    assert row["population"] == 1000.0
    assert row["pct_foreign_nationality"] == 15.0
    assert row["pct_foreign_born"] == 20.0
    # Women, not men: the two are complementary, so only one is emitted.
    assert row["pct_woman"] == 52.0
    assert "pct_male" not in row.index
    assert row["pct_under_15"] == 15.0      # (50+50+50)/1000*100
    assert row["pct_65_plus"] == 10.0       # only 65-69 present
    # Absent source column -> NaN, not a KeyError.
    assert np.isnan(row["pct_born_same_municipality"])


def test_dwelling_shares_counts_and_tenure():
    census_tracts = pd.DataFrame({
        "Municipality code": ["08019"],
        "District code": ["01"],
        "Census tract code": ["001"],
        "Year": [2021],
        "Dwellings": [1000.0],
        "Dwellings ~ Dwelling type:Main": [800.0],
        "Dwellings ~ Dwelling type:Non-main": [200.0],
        "Main dwellings ~ Tenure:Owned": [500.0],
        "Main dwellings ~ Tenure:Rented": [250.0],
        "Main dwellings ~ Tenure:Other tenure": [50.0],
        "Households": [790.0],
    })
    out = social._dwelling_shares(census_tracts)
    row = out.iloc[0]
    assert row["_key"] == "0801901001"
    assert row["dwellings"] == 1000.0
    assert row["dwellings_main"] == 800.0
    assert row["households"] == 790.0
    assert row["pct_dwellings_main"] == 80.0
    # Tenure is a share of the main dwellings it describes (500+250+50 = 800),
    # not of all dwellings, and the three classes add up to 100.
    assert row["pct_main_dwellings_owned"] == 62.5
    assert row["pct_main_dwellings_rented"] == 31.25
    assert row["pct_main_dwellings_other_tenure"] == 6.25
    # Only the curated set is emitted; the counts read to derive it are not.
    emitted = [c for c in out.columns if c not in ("Year", "_key")]
    assert set(emitted) == set(social.DWELLING_INDICATORS)


def test_dwelling_shares_zero_denominator_is_nan():
    # An empty tract must give NaN shares rather than raising or returning inf.
    census_tracts = pd.DataFrame({
        "Municipality code": ["08019"], "District code": ["01"],
        "Census tract code": ["001"], "Year": [2021],
        "Dwellings": [0.0], "Dwellings ~ Dwelling type:Main": [0.0],
        "Main dwellings ~ Tenure:Owned": [0.0],
        "Main dwellings ~ Tenure:Rented": [0.0],
        "Main dwellings ~ Tenure:Other tenure": [0.0],
    })
    row = social._dwelling_shares(census_tracts).iloc[0]
    assert np.isnan(row["pct_dwellings_main"])
    assert np.isnan(row["pct_main_dwellings_owned"])


def test_empty_dwelling_indicators_are_the_harmonised_series():
    # The 2001/2011 field classification and the 2021 electricity one are the
    # same dwellings counted twice, so only the bridge between them is emitted,
    # alongside the flag marking the modelled rows.
    assert set(social.EMPTY_DWELLING_INDICATORS.values()) == {
        "municipality_dwellings", "pct_dwellings_use_main",
        "pct_dwellings_use_secondary", "pct_dwellings_use_empty",
        "dwelling_use_predicted"}
    assert all("Comparable use" in k or k in ("Dwellings", "Predicted")
               for k in social.EMPTY_DWELLING_INDICATORS)


def test_indicator_names_are_unique_across_datasets():
    # Every indicator must come from exactly one dataset: a name emitted twice
    # would have the second join silently overwrite the first.
    names = (list(social.ATLAS_INDICATORS.values())
             + list(social.EDU_INDICATORS.values())
             + list(social.DWELLING_INDICATORS)
             + list(social.EMPTY_DWELLING_INDICATORS.values())
             + list(social.RENTAL_INDICATORS.values())
             + list(social.PRICE_INDICATORS.values())
             + list(social.ELECTRICITY_INDICATORS.values())
             # the population census names its shares in _population_shares
             + ["population", "pct_foreign_nationality", "pct_foreign_born",
                "pct_born_same_municipality", "pct_woman", "pct_under_15",
                "pct_working_age", "pct_65_plus"]
             # the education census names its attainment ladder in _education_shares
             + [f"pct_edu_lte_{level}" for _src, level in social._EDU_LEVELS])
    duplicated = {n for n in names if names.count(n) > 1}
    assert duplicated == set()


def test_join_social_ES_no_census_tract_code_is_noop():
    gdf = gpd.GeoDataFrame(
        {"building_reference": ["X"], "geometry": [Point(0, 0)]},
        crs="EPSG:25831")
    out = social.join_social_ES(gdf, wd="/nonexistent", verbose=False)
    # Unchanged frame (no census_tract_code to join on).
    assert list(out.columns) == list(gdf.columns)
    assert len(out) == 1


def test_all_datasets_constant():
    assert social.ALL_DATASETS == (
        "atlas", "population", "education", "dwellings", "empty_dwellings",
        "rental", "price", "electricity")


def test_essential_characteristics_missing_inference_cols_is_noop():
    # No br__ inference columns present -> must skip without touching the network.
    gdf = gpd.GeoDataFrame(
        {"building_reference": ["X"], "census_tract_code": ["0801901001"],
         "geometry": [Point(0, 0)]},
        crs="EPSG:25831")
    out = social.join_essential_characteristics(gdf, wd="/nonexistent", verbose=False)
    assert list(out.columns) == list(gdf.columns)
    assert len(out) == 1


def test_essential_characteristics_required_cols():
    # Guards against silently accepting a frame without the CAT-based br__ columns.
    for col in ("br__area_without_communals", "br__building_spaces",
                "ine_code", "year_of_construction", "census_tract_code",
                "building_reference"):
        assert col in social._EC_REQUIRED_COLS


def test_ec_translation_helpers_are_gone():
    # social_ES >= 1.0.3 emits English snake_case itself, so the Spanish->English
    # translation this module used to carry must not come back.
    for name in ("translate_ec_column", "rename_ec_columns", "ec_column_rename_map"):
        assert not hasattr(social, name)


def test_join_social_ES_skips_essential_characteristics_without_cat(capsys):
    # Explicitly requesting essential_characteristics without use_CAT_files must
    # skip it (with a warning) rather than attempting the downscaling. Using a
    # frame with no census_tract_code keeps the call fully offline.
    gdf = gpd.GeoDataFrame(
        {"building_reference": ["X"], "geometry": [Point(0, 0)]},
        crs="EPSG:25831")
    out = social.join_social_ES(
        gdf, wd="/nonexistent",
        datasets=[social.ESSENTIAL_CHARACTERISTICS], use_CAT_files=False,
        verbose=False)
    err = capsys.readouterr().err
    assert "use_CAT_files=False" in err
    assert list(out.columns) == list(gdf.columns)


def test_education_shares_are_cumulative():
    # INE publishes the four attainment levels as disjoint shares of the
    # population aged 16+; the emitted KPIs are their running sum, so each one
    # answers "reached at most this level".
    census_tracts = pd.DataFrame({
        "Municipality code": ["08019"],
        "District code": ["01"],
        "Census tract code": ["001"],
        "Year": [2024],
        "Population ~ Educational level:Primary education or below": [10.0],
        "Population ~ Educational level:Lower secondary education or equivalent": [25.0],
        "Population ~ Educational level:Upper secondary and post-secondary non-tertiary education": [30.0],
        "Population ~ Educational level:Tertiary education": [35.0],
        "Population ~ Labour force status:Employed": [48.0],
    })
    row = social._education_shares(census_tracts).iloc[0]
    assert row["_key"] == "0801901001"
    assert row["pct_edu_lte_primary_or_below"] == 10.0
    assert row["pct_edu_lte_lower_secondary"] == 35.0
    assert row["pct_edu_lte_upper_secondary"] == 65.0
    # The four levels partition the population, so the top of the ladder is 100 %.
    assert row["pct_edu_lte_tertiary"] == 100.0
    # The labour-force block is a straight pass-through.
    assert row["pct_employed"] == 48.0
    # The disjoint per-level shares are NOT emitted any more.
    assert "pct_edu_primary_or_below" not in row.index


def test_education_shares_missing_level_stops_the_ladder():
    # A level absent from the source must NaN every KPI from it upwards rather
    # than silently under-counting the ones above it.
    census_tracts = pd.DataFrame({
        "Municipality code": ["08019"],
        "District code": ["01"],
        "Census tract code": ["001"],
        "Year": [2024],
        "Population ~ Educational level:Primary education or below": [10.0],
        "Population ~ Educational level:Upper secondary and post-secondary non-tertiary education": [30.0],
        "Population ~ Educational level:Tertiary education": [35.0],
    })
    row = social._education_shares(census_tracts).iloc[0]
    assert row["pct_edu_lte_primary_or_below"] == 10.0
    assert np.isnan(row["pct_edu_lte_lower_secondary"])
    assert np.isnan(row["pct_edu_lte_tertiary"])


def test_atlas_carries_the_time_use_income_group():
    # The income band is the join key into the Time Use Survey schedules, so the
    # atlas map must keep publishing it.
    assert social.ATLAS_INDICATORS["Household income group"] == "household_income_group"


def test_ec_selection_keeps_social_ES_names_verbatim():
    # The names social_ES returns are kept exactly, "~" separator included: only
    # the namespace prefix is added.
    for variable in social.EC_SELECTED_VARIABLES:
        assert "~" not in variable          # a variable name, not a column name
        assert not variable.startswith(social.EC_PREFIX)
    # The selection is a subset with no duplicates.
    assert len(set(social.EC_SELECTED_VARIABLES)) == len(social.EC_SELECTED_VARIABLES)
    # The variables the energy model reads must be in it, or the envelope and the
    # system assumptions silently fall back to defaults.
    for needed in ("main_dwellings_conservation_status", "main_dwellings_heating_type",
                   "main_dwellings_heating_fuel", "main_dwellings_has_cooling"):
        assert needed in social.EC_SELECTED_VARIABLES


def test_release_empties_frames_and_dicts():
    df = pd.DataFrame({"a": [1, 2, 3]})
    tables = {"Census tracts": df, "Municipality": pd.DataFrame({"b": [1]})}
    social.release(tables, df)
    assert tables == {}
    assert df.empty
    # Freeing memory must never raise, whatever it is handed.
    social.release(None, "not a frame", 42)


def test_time_use_household_occupancy_bad_code_is_none():
    # Too short / missing tract codes must not reach social_ES at all.
    assert social.time_use_household_occupancy("/nonexistent", None) is None
    assert social.time_use_household_occupancy("/nonexistent", "0801901") is None


def test_social_es_numeric_round_shim_restores_pandas():
    # The shim works around a social_ES 1.0.3 bug (object-dtype share columns) and
    # must leave pandas exactly as it found it.
    original = pd.Series.round
    with social._social_es_numeric_round():
        # Inside: an object column of numbers rounds instead of raising.
        assert pd.Series(["1.234", "2.345"], dtype=object).round(1).tolist() == [1.2, 2.3]
        # A genuinely non-numeric column is returned untouched, not raised on.
        assert pd.Series(["a", "b"], dtype=object).round(1).tolist() == ["a", "b"]
        # Normal numeric rounding is unaffected.
        assert pd.Series([1.234]).round(1).tolist() == [1.2]
    assert pd.Series.round is original


def test_ctagg_prefix_and_coverage_threshold():
    from hypercadaster_ES import mergers
    assert mergers.CTAGG_PREFIX == "ctagg__"
    # A frame must hold most of a tract before its density is published.
    assert 0.0 < mergers.CTAGG_MIN_COVERAGE <= 1.0
