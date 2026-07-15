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


def test_section_key_zero_pads():
    df = pd.DataFrame({
        "Municipality code": ["8019", "08019"],
        "District code": ["1", "01"],
        "Section code": ["1", "001"],
    })
    keys = social._section_key(df)
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
    sections = pd.DataFrame({
        "Municipality code": ["08019"],
        "District code": ["01"],
        "Section code": ["001"],
        "Year": [2021],
        "Population": [1000.0],
        "Population ~ Nationality:Foreign": [150.0],
        "Population ~ Birth origin:Extranjero": [200.0],
        "Population ~ Sex:Males": [480.0],
        "Population ~ Age:0-4": [50.0],
        "Population ~ Age:5-9": [50.0],
        "Population ~ Age:10-14": [50.0],
        "Population ~ Age:65-69": [100.0],
    })
    out = social._population_shares(sections)
    row = out.iloc[0]
    assert row["_key"] == "0801901001"
    assert row["pct_foreign_nationality"] == 15.0
    assert row["pct_foreign_born"] == 20.0
    assert row["pct_male"] == 48.0
    assert row["pct_under_15"] == 15.0      # (50+50+50)/1000*100
    assert row["pct_65_plus"] == 10.0       # only 65-69 present


def test_join_social_ES_no_section_code_is_noop():
    gdf = gpd.GeoDataFrame(
        {"building_reference": ["X"], "geometry": [Point(0, 0)]},
        crs="EPSG:25831")
    out = social.join_social_ES(gdf, wd="/nonexistent", verbose=False)
    # Unchanged frame (no section_code to join on).
    assert list(out.columns) == list(gdf.columns)
    assert len(out) == 1


def test_all_datasets_constant():
    assert social.ALL_DATASETS == (
        "atlas", "population", "education", "rental", "price", "electricity")


def test_essential_characteristics_missing_inference_cols_is_noop():
    # No br__ inference columns present -> must skip without touching the network.
    gdf = gpd.GeoDataFrame(
        {"building_reference": ["X"], "section_code": ["0801901001"],
         "geometry": [Point(0, 0)]},
        crs="EPSG:25831")
    out = social.join_essential_characteristics(gdf, wd="/nonexistent", verbose=False)
    assert list(out.columns) == list(gdf.columns)
    assert len(out) == 1


def test_essential_characteristics_required_cols():
    # Guards against silently accepting a frame without the CAT-based br__ columns.
    for col in ("br__area_without_communals", "br__building_spaces",
                "ine_code", "year_of_construction", "section_code",
                "building_reference"):
        assert col in social._EC_REQUIRED_COLS


def test_translate_ec_column_examples():
    cases = {
        "ec__Personas de 16 años o más - Lugar de trabajo/estudio ~ En el municipio en el que resido":
            "ec__people_16plus_work_study_place_same_muni",
        "ec__Viviendas principales - Tipo de calefacción ~ Sí, individual":
            "ec__main_dwellings_heating_individual",
        "ec__Hogares/personas en viviendas propias, - Cuota mensual de la hipoteca ~ De 1.000 euros o más":
            "ec__owned_homes_monthly_mortgage_ge_1000eur",
        # trailing "(%)" is stripped
        "ec__Viviendas principales - Tipo de electrodoméstico ~ Vitocerámica/Inducción (%)":
            "ec__main_dwellings_appliance_ceramic_induction_hob",
        # digit spacing: number brackets are cleanly separated (le_45m2, not le45m2)
        "ec__Hogares - Superficie útil de la vivienda ~ Hasta 45 m2":
            "ec__households_floor_area_le_45m2",
        # non-ec columns are untouched
        "so__net_income_per_household_2020": "so__net_income_per_household_2020",
        "geometry": "geometry",
    }
    for src, expected in cases.items():
        assert social.translate_ec_column(src) == expected


def test_rename_ec_columns_frame():
    df = pd.DataFrame({
        "building_reference": ["X"],
        "ec__Hogares - Número de vehículos ~ 3 o más": [10.0],
        "ec__Hogares/personas - Número de vehículos ~ 3 o más": [20.0],
        "so__net_income_per_household_2020": [1.0],
    })
    out = social.rename_ec_columns(df)
    # Distinct groups -> distinct English names, no collision, data preserved.
    assert "ec__households_n_vehicles_3plus" in out.columns
    assert "ec__hh_people_n_vehicles_3plus" in out.columns
    assert out.columns.duplicated().sum() == 0
    assert out["so__net_income_per_household_2020"].iloc[0] == 1.0
    assert out["ec__households_n_vehicles_3plus"].iloc[0] == 10.0


def test_translate_ec_column_idempotent():
    # An already-English column (no ' - '/' ~ ' separators) must be left as-is,
    # so re-applying the rename never mangles it.
    english = "ec__MainDwellings_HeatingType_Individual"
    assert social.translate_ec_column(english) == english
    df = pd.DataFrame({english: [1.0], "geometry": [None]})
    assert list(social.rename_ec_columns(df).columns) == [english, "geometry"]


def test_translate_ec_column_unknown_fallback():
    # Unknown fragments must not raise; they get an accent-stripped CamelCase.
    got = social.translate_ec_column("ec__Grupo inventado - Variable rara ~ Opción ñandú")
    assert got.startswith("ec__")
    assert " " not in got and "~" not in got


def test_join_social_ES_skips_essential_characteristics_without_cat(capsys):
    # Explicitly requesting essential_characteristics without use_CAT_files must
    # skip it (with a warning) rather than attempting the downscaling. Using a
    # frame with no section_code keeps the call fully offline.
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
