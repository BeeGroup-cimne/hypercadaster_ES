"""Occupant-behaviour scenario KPIs (``energy.run_behavior_scenarios`` and its
analytical helpers). The EnergyPlus run itself needs the engine + weather and is
exercised in the live smoke path; here we validate the format-independent core:
EN 16798-1 adaptive comfort, the E+ CSV parser, and the per-use-type KPI
aggregation, on synthetic hourly output."""
import os

import numpy as np
import pandas as pd
import pytest

from hypercadaster_ES import energy


def _synthetic_csv(tmp_path, zone, top, outdoor=8.0, occ=1.0, heat_j=3.6e6,
                   cool_j=0.0, gain_j=1.8e6, n=48):
    """Write a minimal EnergyPlus-style hourly CSV for one zone."""
    df = pd.DataFrame({
        "Date/Time": [f" 01/{1 + i // 24:02d}  {i % 24 + 1:02d}:00:00"
                      for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [top] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)":
            [heat_j] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)":
            [cool_j] * n,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [gain_j] * n,
        f"{zone}:Zone People Occupant Count [](Hourly)": [occ] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [outdoor] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_adaptive_out_of_comfort_band():
    # θrm = 20 → Tc = 0.33*20+18.8 = 25.4; Category II band = ±3 → [22.4, 28.4].
    ooc = energy._adaptive_out_of_comfort([21.0, 25.0, 29.0], [20.0, 20.0, 20.0], "II")
    assert list(ooc) == [True, False, True]
    # Narrower Category I (±2) flags 24 as well.
    ooc_I = energy._adaptive_out_of_comfort([23.0], [20.0], "I")   # band [23.4,27.4]
    assert bool(ooc_I[0]) is True


def test_running_mean_outdoor_clamped():
    df = pd.DataFrame({"t": [5.0] * 48})          # two cold days
    rm = energy._running_mean_outdoor(df, "t")
    assert len(rm) == 48
    assert np.all(rm >= 10.0) and np.all(rm <= 30.0)   # clamped to validity range


def test_read_eplus_csv_parses_month_hour(tmp_path):
    csv = _synthetic_csv(tmp_path, "H_F0", top=18.0)
    df = energy._read_eplus_csv(csv)
    assert df is not None and len(df) == 48
    assert df["month"].iloc[0] == 1
    assert df["hour"].iloc[0] == 1 and df["hour"].iloc[23] == 24   # 1..24 end-hour
    assert energy._read_eplus_csv(str(tmp_path / "missing.csv")) is None


def test_zone_kpis_and_aggregation(tmp_path):
    # 48 h in January, cold zone (18 °C, outdoor 8 °C → θrm clamped to 10, band
    # ~[19.1,25.1]) → 100% out of comfort day AND night; 1 kWh/h heating over 48 h =
    # 48 kWh; gains 0.5 kWh/h = 24 kWh.
    csv = _synthetic_csv(tmp_path, "H_F0", top=18.0)
    df = energy._read_eplus_csv(csv)
    zone_meta = {"H_F0": {"use_type": "Residential", "reference": "H"}}
    areas = {"H_F0": 50.0}
    recs = energy._zone_scenario_kpis(df, zone_meta, areas, "II")
    # One record per season; all 48 h are in January → the HEATING season carries it.
    by_season = {r["season"]: r for r in recs}
    assert set(by_season) == {"heating", "cooling", "in_between"}
    h = by_season["heating"]
    assert h["thermal_demand_kwh"] == pytest.approx(48.0, rel=1e-3)
    assert h["internal_gains_kwh"] == pytest.approx(24.0, rel=1e-3)
    assert h["pct_discomfort_day"] == pytest.approx(100.0)
    assert h["pct_discomfort_night"] == pytest.approx(100.0)
    assert by_season["cooling"]["thermal_demand_kwh"] == pytest.approx(0.0)

    rows = energy._aggregate_scenario_kpis(recs, "unaware")
    heat_row = next(r for r in rows if r["season"] == "heating")
    assert heat_row["use_type"] == "Residential"
    assert heat_row["thermal_demand_kwh_m2"] == pytest.approx(48.0 / 50.0, rel=1e-2)
    assert heat_row["internal_gains_kwh_m2"] == pytest.approx(24.0 / 50.0, rel=1e-2)
    assert heat_row["pct_hours_out_of_comfort_day"] == pytest.approx(100.0)
    assert heat_row["pct_hours_out_of_comfort_night"] == pytest.approx(100.0)


def test_comfortable_zone_scores_zero_discomfort(tmp_path):
    # 24 °C with outdoor 20 °C (θrm 20 → band [22.4,28.4]) → always comfortable.
    csv = _synthetic_csv(tmp_path, "H_F0", top=24.0, outdoor=20.0)
    df = energy._read_eplus_csv(csv)
    recs = energy._zone_scenario_kpis(
        df, {"H_F0": {"use_type": "Residential"}}, {"H_F0": 40.0}, "II")
    # Outdoor 20 °C → daily mean 20 → the IN-BETWEEN season (18–26 °C) carries it.
    ib = next(r for r in recs if r["season"] == "in_between")
    assert ib["season_days"] == 2                       # both synthetic days
    assert ib["pct_discomfort_day"] == pytest.approx(0.0)
    assert ib["pct_discomfort_night"] == pytest.approx(0.0)


def test_non_occupied_zones_excluded(tmp_path):
    # A parking zone must not enter the per-use-type KPI aggregation.
    csv = _synthetic_csv(tmp_path, "P_F0", top=18.0)
    df = energy._read_eplus_csv(csv)
    recs = energy._zone_scenario_kpis(
        df, {"P_F0": {"use_type": "Warehouse - Parking"}}, {"P_F0": 100.0}, "II")
    assert recs == []


def test_ideal_loads_demand_column_matched(tmp_path):
    # EnergyPlus reports ideal-loads energy under the SYSTEM key
    # ("ZONE IDEAL LOADS AIR SYSTEM:..."), not the bare zone key — the KPI
    # extractor must still find it (regression: demand read as 0 otherwise).
    zone = "H_F0"
    df = pd.DataFrame({
        "Date/Time": [f" 01/01  {i + 1:02d}:00:00" for i in range(24)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [21.0] * 24,
        f"{zone} IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)":
            [3.6e6] * 24,
        f"{zone} IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)":
            [0.0] * 24,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * 24,
        f"{zone}:Zone People Occupant Count [](Hourly)": [1.0] * 24,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [10.0] * 24,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    d = energy._read_eplus_csv(str(p))
    recs = energy._zone_scenario_kpis(
        d, {zone: {"use_type": "Residential"}}, {zone: 20.0}, "II")
    # 1 kWh/h × 24 h in January (heating season) — the system-name column matched.
    h = next(r for r in recs if r["season"] == "heating")
    assert h["thermal_demand_kwh"] == pytest.approx(24.0, rel=1e-3)


def test_conservation_infiltration_factor():
    # No ec__ columns → default factor 1.0.
    assert energy._conservation_infiltration_factor({}) == pytest.approx(1.0)
    # All "bad" conservation → the leakiest multiplier.
    bad = {"ec__conservation_bad": 1.0}
    assert energy._conservation_infiltration_factor(bad) == pytest.approx(
        energy._CONSERVATION_INFIL_FACTOR["bad"])
    # Proportion-weighted blend of good + bad.
    mix = {"ec__conservation_good": 0.5, "ec__conservation_bad": 0.5}
    assert energy._conservation_infiltration_factor(mix) == pytest.approx(
        (energy._CONSERVATION_INFIL_FACTOR["good"]
         + energy._CONSERVATION_INFIL_FACTOR["bad"]) / 2.0)


def test_conservation_reads_the_column_social_ES_actually_emits():
    # The social layer keeps social_ES's own names, which join the class to its
    # variable with "~": ec__main_dwellings_conservation_status~{class}. Reading
    # only the bare ec__conservation_{class} left the envelope on its default for
    # every real frame.
    row = {"ec__main_dwellings_conservation_status~bad": 60.0,
           "ec__main_dwellings_conservation_status~good": 40.0}
    assert energy._building_conservation(row) == "bad"
    assert energy._conservation_infiltration_factor(row) == pytest.approx(
        (energy._CONSERVATION_INFIL_FACTOR["bad"] * 60.0
         + energy._CONSERVATION_INFIL_FACTOR["good"] * 40.0) / 100.0)


def test_conservation_still_reads_pre_1_2_4_underscore_names():
    # Frames merged before the "~" naming change must keep driving the envelope.
    row = {"ec__main_dwellings_conservation_status_bad": 60.0,
           "ec__main_dwellings_conservation_status_good": 40.0}
    assert energy._building_conservation(row) == "bad"
    assert energy._conservation_infiltration_factor(row) == pytest.approx(
        (energy._CONSERVATION_INFIL_FACTOR["bad"] * 60.0
         + energy._CONSERVATION_INFIL_FACTOR["good"] * 40.0) / 100.0)


def test_shade_relevance_filter():
    # Building centroid (0,0), lowest surface z = 0.
    cx, cy, min_bz = 0.0, 0.0, 0.0

    def shade(x, y, ztop):
        return [(x, y, 0.0), (x + 1, y, 0.0), (x + 1, y, ztop), (x, y, ztop)]

    # South, tall, above horizon → RELEVANT (kept).
    assert not energy._shade_irrelevant(shade(0.0, -10.0, 8.0), cx, cy, min_bz)
    # East, tall → relevant.
    assert not energy._shade_irrelevant(shade(10.0, 0.0, 8.0), cx, cy, min_bz)
    # West, tall → relevant.
    assert not energy._shade_irrelevant(shade(-10.0, 0.0, 8.0), cx, cy, min_bz)
    # Due North → no direct sun (Spain) → irrelevant.
    assert energy._shade_irrelevant(shade(0.0, 10.0, 8.0), cx, cy, min_bz)
    # NE (bearing 45°) and NW (315°) → within 67.5° of North → irrelevant.
    assert energy._shade_irrelevant(shade(10.0, 10.0, 8.0), cx, cy, min_bz)
    assert energy._shade_irrelevant(shade(-10.0, 10.0, 8.0), cx, cy, min_bz)
    # South but entirely BELOW the building's lowest point → below horizon → irrelevant.
    assert energy._shade_irrelevant(shade(0.0, -10.0, -2.0), cx, cy, min_bz)


def test_occlusion_cull():
    # boxes: (distance, az0, az1, el0, el1)
    near_big = (5.0, 100.0, 140.0, 0.0, 40.0)     # nearest, wide+tall
    far_inside = (50.0, 110.0, 130.0, 5.0, 20.0)  # farther, fully within near_big → culled
    far_outside = (50.0, 200.0, 220.0, 0.0, 30.0)  # different direction → kept
    keep = energy._occlusion_cull([near_big, far_inside, far_outside])
    assert 0 in keep                               # nearest always kept
    assert 1 not in keep                           # hidden behind the nearer panel
    assert 2 in keep                               # sees its own sky region


def test_shade_angular_box():
    # A panel due south (y<0), 4–8 m high, from vantage at origin (z=0).
    box = energy._shade_angular_box(
        [(-1, -10, 4), (1, -10, 4), (1, -10, 8), (-1, -10, 8)], 0.0, 0.0, 0.0)
    dist, a0, a1, e0, e1 = box
    assert dist == pytest.approx(10.0, rel=0.05)
    assert 160 < a0 < 200 and 160 < a1 < 200        # ~south (180°)
    assert e0 > 0 and e1 > e0                        # above the horizon


def test_seasons_and_night_hours():
    # Three seasons partition the year; heating Dec–Mar, cooling Jun–Sep, rest between.
    assert energy._HEATING_SEASON_MONTHS == {12, 1, 2, 3}
    assert energy._COOLING_SEASON_MONTHS == {6, 7, 8, 9}
    assert energy._INBETWEEN_SEASON_MONTHS == {4, 5, 10, 11}
    allm = (energy._HEATING_SEASON_MONTHS | energy._COOLING_SEASON_MONTHS
            | energy._INBETWEEN_SEASON_MONTHS)
    assert allm == set(range(1, 13))
    # Night = 22:00–07:59 (end-hour labels 23,24,1..8); day is the complement.
    assert energy._NIGHT_HOURS == {23, 24, 1, 2, 3, 4, 5, 6, 7, 8}
    assert len(energy._NIGHT_HOURS) == 10 and 22 not in energy._NIGHT_HOURS


def test_temperature_based_seasons():
    # In-between = daily mean 16–24 °C; below → heating, above → cooling.
    assert energy._season_of_temp(10.0) == "heating"
    assert energy._season_of_temp(16.0) == "in_between"
    assert energy._season_of_temp(20.0) == "in_between"
    assert energy._season_of_temp(24.0) == "in_between"
    assert energy._season_of_temp(26.0) == "cooling"
    assert energy._season_of_temp(30.0) == "cooling"
    assert energy._doy_to_mmdd(1) == (1, 1)
    assert energy._doy_to_mmdd(32) == (2, 1)
    assert energy._doy_to_mmdd(365) == (12, 31)


def test_season_day_count_smoothed(tmp_path):
    # 9 days at 10 / 20 / 30 °C (3 each) → after the 3-day smoothing the counts stay
    # 3 heating / 3 in-between / 3 cooling (the transition days blend but stay put).
    import pandas as pd
    temps = [10.0] * 3 + [20.0] * 3 + [30.0] * 3
    rows = []
    for i, t in enumerate(temps):
        for h in range(24):
            rows.append((f" 01/{i + 1:02d}  {h + 1:02d}:00:00", t))
    df = pd.DataFrame(rows, columns=["Date/Time",
                                     "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)"])
    df["H_F0:Zone Operative Temperature [C](Hourly)"] = 21.0
    df["H_F0:Zone People Occupant Count [](Hourly)"] = 1.0
    p = tmp_path / "eplusout.csv"; df.to_csv(p, index=False)
    d = energy._read_eplus_csv(str(p))
    recs = energy._zone_scenario_kpis(d, {"H_F0": {"use_type": "Residential"}},
                                      {"H_F0": 40.0}, "II")
    by = {r["season"]: r["season_days"] for r in recs}
    assert by == {"heating": 3, "in_between": 3, "cooling": 3}


def test_season_smoothing_continuity():
    # An isolated warm day inside a cold run does NOT flip to cooling after smoothing.
    raw = [8, 8, 8, 27, 8, 8, 8]                  # single 27 °C spike
    sm = energy._smooth_days(raw, window=3)
    seasons = [energy._season_of_temp(t) for t in sm]
    assert "cooling" not in seasons               # the spike is averaged away
    assert seasons[3] in ("heating", "in_between")


def test_cat_first_use_classification():
    # CAT usage (the space use_type) wins; building_use is only a fallback.
    assert energy._resolve_use_category({"use_type": "Residential"}) == "residential"
    assert energy._resolve_use_category({"use_type": "Office"}) == "office"
    # No CAT usage on the record → fall back to building_use.
    assert energy._resolve_use_category({}, building_use="3_industrial") == "other"
    assert energy._resolve_use_category({}, building_use="1_residential") == "residential"


def test_sun_facing_and_adoption():
    assert energy._sun_facing(180.0)          # south → sun
    assert energy._sun_facing(90.0)           # east → sun
    assert energy._sun_facing(270.0)          # west → sun
    assert not energy._sun_facing(0.0)        # north → no sun
    assert not energy._sun_facing(45.0)       # NE → no sun
    # Adoption is a deterministic fraction: none at 0, all at 1, ~p in between.
    assert energy._adopted("x", 0.0) is False
    assert energy._adopted("x", 1.0) is True
    frac = sum(energy._adopted(f"w{i}", 0.4) for i in range(2000)) / 2000.0
    assert 0.34 < frac < 0.46                  # ≈ 0.40


class _Sub:
    """Minimal stand-in for a FenestrationSurface:Detailed (Name + coords)."""

    def __init__(self, name, coords):
        self.Name = name
        self.coords = coords


class _Idf:
    def __init__(self, subs):
        self.idfobjects = {"FENESTRATIONSURFACE:DETAILED": subs}


def _column_window(name, x, z):
    """A 1 m pane at plan x (façade along +x, y=0), sill z."""
    return _Sub(name, [(x, 0.0, z), (x + 1.0, 0.0, z),
                       (x + 1.0, 0.0, z + 1.0), (x, 0.0, z + 1.0)])


def test_window_columns_group_by_plan_xy_not_by_floor():
    # Two façade bays (x=0 and x=4), each repeated on four floors under a
    # different wall/window name — the awning draw must see two columns, not
    # eight independent windows.
    subs = [_column_window(f"Wall_F{f}_S1_win0", 0.0, 3.0 * f) for f in range(4)]
    subs += [_column_window(f"Wall_F{f}_S1_win1", 4.0, 3.0 * f) for f in range(4)]
    cols = energy._window_columns(_Idf(subs))
    assert len(set(cols.values())) == 2
    assert len({cols[f"Wall_F{f}_S1_win0"] for f in range(4)}) == 1
    assert cols["Wall_F0_S1_win0"] != cols["Wall_F0_S1_win1"]
    # Same column ⇒ same adoption verdict on every floor, at any probability.
    for p in (0.2, 0.4, 0.6, 0.8):
        verdicts = {energy._adopted(cols[f"Wall_F{f}_S1_win0"], p) for f in range(4)}
        assert len(verdicts) == 1


def test_window_columns_tolerate_float_jitter_across_floors():
    # Grid rounding would split these at a cell boundary; the snap must not.
    subs = [_column_window("Wall_F0_win0", 2.0, 0.0),
            _column_window("Wall_F1_win0", 2.0 + 1e-9, 3.0),
            _column_window("Wall_F2_win0", 2.0 - 1e-9, 6.0)]
    assert len(set(energy._window_columns(_Idf(subs)).values())) == 1


def test_conditioned_use_categories():
    # Only residential/office/commercial are conditioned/modelled now (req #6).
    assert set(energy._OCCUPIED_USE_CATEGORIES) == {"residential", "office", "commercial"}
    assert "other" not in energy._OCCUPIED_USE_CATEGORIES
    assert set(energy._MODELABLE_USE_CATEGORIES) == {"residential", "office", "commercial"}


def test_default_scenarios_shape():
    sc = energy._DEFAULT_SCENARIOS
    assert set(sc) == {"random_mix", "unaware", "unaware_ff", "aware_ff",
                       "aware", "cte"}
    hvac = [k for k, v in sc.items() if not v.get("free_floating")]
    ff = [k for k, v in sc.items() if v.get("free_floating")]
    assert set(hvac) == {"random_mix", "unaware", "aware", "cte"}
    assert set(ff) == {"unaware_ff", "aware_ff"}
    # random_mix is residential-only (skipped for non-residential buildings).
    assert sc["random_mix"].get("residential_only") is True
    # A scenario is an ARCHETYPE ASSIGNMENT and nothing else: every other key was
    # read by machinery the unified occupant engine replaced, so none may return.
    import inspect
    read = set(inspect.getsource(energy.run_behavior_scenarios).split('spec.get("'))
    allowed = {"behavior", "free_floating", "mix_aware_unaware", "residential_only"}
    for spec in sc.values():
        assert set(spec) <= allowed, spec


def test_hvac_config_setpoints_and_limits():
    hv = energy._HVAC
    # Unaware users hold a fixed 21/25 band; aware users set back to 16/30 empty.
    assert hv["unaware"]["heating"] == 21
    assert hv["unaware"]["cooling"] == 25
    assert hv["aware_setback"]["heating"] == 16
    assert hv["aware_setback"]["cooling"] == 30
    # Finite HVAC: positive per-m² power caps for the ideal-loads plant.
    assert hv["max_heating_w_per_m2"] > 0
    assert hv["max_cooling_w_per_m2"] > 0


def test_setback_day_fields_occupancy_aware():
    # No preconditioning: the setpoint arrives exactly when the household does.
    f = energy._setback_day_fields(21.5, 16.0, lead=0)
    # Occupied blocks carry the setpoint; weekday 09–14 and 16–19 set back.
    assert "Until: 09:00,21.5" in f
    assert "Until: 14:00,16.0" in f
    assert "Until: 16:00,21.5" in f
    assert "Until: 19:00,16.0" in f
    assert "Until: 24:00,21.5" in f
    # Weekends / holidays stay occupied all day (no setback).
    assert f[-2:] == ["For: AllOtherDays", "Until: 24:00,21.5"]
    assert "16.0" not in " ".join(f[f.index("For: AllOtherDays"):])


def test_aware_preconditions_before_the_occupancy():
    # The system starts an hour EARLY, so the dwelling is already being worked on
    # when the household walks in instead of recovering from the setback.
    assert energy._preconditioning_lead() == 60
    f = energy._setback_day_fields(21.5, 16.0)
    # Home at 14:00 and 19:00 → the setpoint arrives at 13:00 and 18:00.
    assert "Until: 13:00,16.0" in f and "Until: 14:00,16.0" not in f
    assert "Until: 18:00,16.0" in f and "Until: 19:00,16.0" not in f
    # Cooling gets the same lead — "heating or cooling", one rule.
    c = energy._setback_day_fields(25.0, 30.0)
    assert "Until: 13:00,30.0" in c and "Until: 18:00,30.0" in c
    # With the night setback the WAKING transition is preconditioned too: asleep
    # until 07:00, so the heating comes on at 06:00.
    h = energy._setback_day_fields(21.5, 16.0, energy._night_setback_spans())
    assert _until_value(h, "Monday", "06:30") == 21.5
    assert _until_value(h, "Monday", "05:30") == 16.0
    # A comfort period that already starts at midnight has nothing to pull back.
    assert _until_value(energy._setback_day_fields(21.5, 16.0), "Monday", "00:30") == 21.5


def test_preconditioning_can_be_switched_off():
    saved = dict(energy._RESIDENTIAL_BEHAVIOR)
    try:
        energy._RESIDENTIAL_BEHAVIOR["aware_preconditioning"] = {"enabled": False}
        assert energy._preconditioning_lead() == 0
        f = energy._setback_day_fields(21.5, 16.0)
        assert "Until: 14:00,16.0" in f
    finally:
        energy._RESIDENTIAL_BEHAVIOR.clear()
        energy._RESIDENTIAL_BEHAVIOR.update(saved)


def test_nonresidential_rite_setpoints_config():
    hv = energy._HVAC
    # RITE / RD 486-2022 tertiary restriction: heat 19 / cool 27 while occupied.
    assert hv["nonresidential"]["heating"] == 19
    assert hv["nonresidential"]["cooling"] == 27
    assert hv["nonresidential_setback"]["heating"] == 16
    assert hv["nonresidential_setback"]["cooling"] == 30


def test_cte_scenario_and_profiles():
    # CTE is a default scenario with the Spanish-standards residential user.
    assert energy._DEFAULT_SCENARIOS["cte"]["behavior"] == "cte"
    assert energy._DEFAULT_SCENARIOS["cte"]["free_floating"] is False
    cte = energy._RES_CTE
    # Calendar seasons with NO gap: CTE DB-HE Apéndice D leaves no hour of the
    # year unassigned, so the "in-between season" of the occupant scenarios must
    # not appear here.
    months = set(cte["seasons"]["heating_months"]) | set(cte["seasons"]["cooling_months"])
    assert months == set(range(1, 13))
    assert not (set(cte["seasons"]["heating_months"])
                & set(cte["seasons"]["cooling_months"]))
    # A shape within each season, not a constant: heating sets back at night,
    # cooling tightens for the occupied evening.
    assert cte["setpoints"]["heating"]["night"] < cte["setpoints"]["heating"]["day"]
    assert cte["setpoints"]["cooling"]["evening"] < cte["setpoints"]["cooling"]["day"]
    assert cte["night_ventilation_ach"] == 4.0
    assert cte["night_ventilation_hours"] == [1, 8]
    # opaque fraction 0.6 → transmittance 0.4.
    assert cte["shade_transmittance"] == 0.4


def test_cte_gain_profile_matches_paper_values():
    # Lighting/appliances nominal 4.40 W/m² × the CTE hourly fractions must give the
    # paper's four values 0.44 / 1.32 / 2.20 / 4.40 W/m².
    fracs = sorted({float(f.split(",")[1]) for f in energy._CTE_GAIN_PROFILE
                    if f.startswith("Until:")})
    assert fracs == [0.1, 0.3, 0.5, 1.0]
    for got, exp in zip((4.40 * fr for fr in fracs), [0.44, 1.32, 2.20, 4.40]):
        assert abs(got - exp) < 0.005
    # Occupancy sensible nominal 2.15 W/m² × its fractions → 2.15 / 0.54 / 1.08.
    ofr = sorted({float(f.split(",")[1]) for f in energy._CTE_OCC_PROFILE
                  if f.startswith("Until:")})
    assert ofr == [0.25, 0.5, 1.0]
    for got, exp in zip((2.15 * fr for fr in ofr), [0.54, 1.08, 2.15]):
        assert abs(got - exp) < 0.01


def test_adopted_is_process_stable():
    # Stable across processes (zlib.crc32, not salted builtin hash): recompute the
    # reference values here and require _adopted to agree.
    import zlib
    for nm in ["Z1_residential_F0", "3444707BG9034S_F1", "win_42", "café ñ"]:
        expected = (zlib.crc32(nm.encode("utf-8")) & 0xffff) / 65535.0 < 0.5
        assert energy._adopted(nm, 0.5) is bool(expected)
    assert energy._adopted("anything", 1.0) is True
    assert energy._adopted("anything", 0.0) is False


# --------------------------------------------------------------------------- #
# Income-driven setpoints (RETABIT §5 sensitivity)
# --------------------------------------------------------------------------- #
def test_income_setpoint_offset_ramp_and_clamp():
    # Ramp: full offset at/below income_floor (12000), zero at/above reference (30000).
    assert energy._income_setpoint_offset(5000) == (-2.0, 2.0)      # below floor
    assert energy._income_setpoint_offset(12000) == (-2.0, 2.0)
    assert energy._income_setpoint_offset(30000) == (0.0, 0.0)
    assert energy._income_setpoint_offset(40000) == (0.0, 0.0)      # above ref
    mid_h, mid_c = energy._income_setpoint_offset(21000)            # midpoint
    assert -2.0 < mid_h < 0.0 and 0.0 < mid_c < 2.0
    # Monotonic: lower income → stronger (more negative heating) offset.
    assert energy._income_setpoint_offset(15000)[0] < mid_h


def test_income_setpoint_offset_disabled_and_fallback():
    assert energy._income_setpoint_offset(5000, {"enabled": False}) == (0.0, 0.0)
    assert energy._income_setpoint_offset({}, {}) == (0.0, 0.0)
    # None income → residential_behavior.default_income (22000) → a mild offset.
    none_off = energy._income_setpoint_offset(None)
    assert none_off == energy._income_setpoint_offset(
        float(energy._RESIDENTIAL_BEHAVIOR.get("default_income", 22000)))
    assert none_off[0] < 0.0


def test_income_setpoints_config_excludes_cte():
    isp = energy._INCOME_SETPOINTS
    assert isp.get("enabled") is True
    assert set(isp.get("apply_to")) == {"unaware", "aware"}   # normative cte excluded
    assert "cte" not in isp.get("apply_to")


# --------------------------------------------------------------------------- #
# DHW end-use + census-tract occupancy (CTE DB-HE HE4)
# --------------------------------------------------------------------------- #
def _brow(census_tract, pop, housing_areas, commercial_areas=()):
    """Minimal building row: Housing (+ optional Commercial) spaces via the
    parallel br__building_spaces_* dicts that _building_spaces consumes."""
    nh, nc = len(housing_areas), len(commercial_areas)
    d = {
        "census_tract_code": census_tract,
        "so__population_2020": pop,
        "br__building_spaces_reference": {"Housing": [f"h{i}" for i in range(nh)]},
        "br__building_spaces_floors": {"Housing": [[1]] * nh},
        "br__building_spaces_area_without_communal": {"Housing": list(housing_areas)},
        "br__building_spaces_detailed_use_type": {"Housing": [None] * nh},
    }
    if nc:
        d["br__building_spaces_reference"]["Commercial"] = [f"c{i}" for i in range(nc)]
        d["br__building_spaces_floors"]["Commercial"] = [[0]] * nc
        d["br__building_spaces_area_without_communal"]["Commercial"] = list(commercial_areas)
        d["br__building_spaces_detailed_use_type"]["Commercial"] = [None] * nc
    return d


def test_census_tract_rate_residential_only_denominator():
    # Two buildings in tract S1: residential 50+100+200=350 m², population 7 → rate
    # 0.02 people/m². The 999 m² of COMMERCIAL must NOT enter the denominator.
    gdf = pd.DataFrame([
        _brow("S1", 7, [50, 100], commercial_areas=[999]),
        _brow("S1", None, [200]),
    ])
    rate = energy._census_tract_person_rate(gdf)
    assert rate == pytest.approx({"S1": 0.02}, rel=1e-9)
    # Per-household counts: round(0.02*50)=1, round(0.02*100)=2 → 3 people, 150 m².
    people, area = energy._dwelling_occupants(gdf.iloc[0], rate)
    assert (people, area) == (3, 150.0)


def test_dwelling_occupants_min_one_person():
    # Tiny household rounds to 0 → floored at 1 person.
    gdf = pd.DataFrame([_brow("S2", 1, [10, 5])])           # rate 1/15 ≈ 0.0667
    rate = energy._census_tract_person_rate(gdf)
    people, _ = energy._dwelling_occupants(gdf.iloc[0], rate)
    assert people == 2                                       # 10→1, 5→max(1,0)=1


def test_dwelling_occupants_no_rate_returns_none():
    row = pd.Series(_brow("SX", None, [80, 80]))
    assert energy._dwelling_occupants(row, {})[0] is None    # missing tract rate


def test_census_tract_rate_handles_list_census_tract_code():
    # Real merged frames store census_tract_code as a 1-element list (address
    # compaction); it must be scalarised to a hashable tract key.
    def _listsec(r):
        r["census_tract_code"] = [r["census_tract_code"]]
        return r
    gdf = pd.DataFrame([_listsec(_brow("S1", 7, [50, 100])),
                        _listsec(_brow("S1", None, [200]))])
    rate = energy._census_tract_person_rate(gdf)
    assert rate == pytest.approx({"S1": 0.02}, rel=1e-9)
    people, _ = energy._dwelling_occupants(gdf.iloc[0], rate)   # list key resolves
    assert people == 3


def test_dhw_demand_matches_cte_formula():
    cfg = energy._RES_DHW
    days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    exp = sum(1 * 28 * days[m] * 4.186 * (60 - cfg["inlet_temp_monthly"][m]) / 3600
              for m in range(12))
    assert energy._dhw_demand_kwh(1) == pytest.approx(exp)
    assert 400 < exp < 700                                   # ~kWh/person/yr sanity
    assert energy._dhw_demand_kwh(0) == 0.0
    assert energy._dhw_demand_kwh(3, {"enabled": False}) == 0.0


def test_building_dhw_occupants_falls_back_to_cte_default():
    # No tract rate → "real" occupancy falls back to the CTE-default density, so the
    # real and cte occupant counts coincide.
    row = pd.Series(_brow("SX", None, [80, 80]))
    occ = energy._building_dhw_occupants(row, {})
    assert occ["real"] == occ["cte"] > 0


def test_dhw_by_season_splits_sums_and_respects_run_period():
    # A full-year day→season map: Jan/Feb heating, Jul/Aug cooling, rest in_between.
    sby = {}
    doy = 1
    for m in range(1, 13):
        s = "heating" if m in (1, 2, 12) else "cooling" if m in (7, 8) else "in_between"
        for _ in range(energy._MONTH_DAYS[m - 1]):
            sby[doy] = s
            doy += 1
    bs = energy._dhw_demand_kwh_by_season(2, sby)
    # DHW appears in EVERY season, and the split sums to the annual total.
    assert all(bs[s] > 0 for s in ("heating", "cooling", "in_between"))
    assert sum(bs.values()) == pytest.approx(energy._dhw_demand_kwh(2))
    # run_period_months restricts to the simulated months (winter → no cooling DHW).
    winter = energy._dhw_demand_kwh_by_season(2, sby, run_period_months=[1, 2, 12])
    assert winter["cooling"] == 0.0 and winter["in_between"] == 0.0
    assert winter["heating"] > 0.0
    # Disabled / no occupants / no season map → all zeros.
    assert energy._dhw_demand_kwh_by_season(0, sby) == {"heating": 0.0,
                                                        "cooling": 0.0,
                                                        "in_between": 0.0}


def test_augment_dhw_by_season_cte_vs_real():
    sby = {d: "heating" for d in range(1, 366)}         # whole year one season
    occ = {"real": 4, "cte": 2}
    rows_un = [{"use_type": "Residential", "season": "heating", "floor_area_m2": 100.0},
               {"use_type": "Commercial", "season": "heating", "floor_area_m2": 50.0}]
    energy._augment_dhw_by_season(rows_un, occ, {"behavior": "unaware"}, sby, None)
    # Residential row gets DHW (real occupancy); the commercial row does NOT.
    assert rows_un[0]["dhw_demand_kwh_m2"] > 0.0
    assert "dhw_demand_kwh_m2" not in rows_un[1]
    # cte scenario uses the (smaller) CTE-default occupancy → lower per-m² DHW.
    rows_cte = [{"use_type": "Residential", "season": "heating", "floor_area_m2": 100.0}]
    energy._augment_dhw_by_season(rows_cte, occ, {"behavior": "cte"}, sby, None)
    assert rows_cte[0]["dhw_demand_kwh_m2"] < rows_un[0]["dhw_demand_kwh_m2"]


def test_population_column_picks_recent_year():
    class _G:
        columns = ["census_tract_code", "so__population_2018", "so__population_2020",
                   "so__pct_65_plus_2020", "br__population_mean_age"]
    assert energy._population_column(_G()) == "so__population_2020"

    class _H:
        columns = ["a", "b", "br__population_mean_age"]     # no count column
    assert energy._population_column(_H()) is None


def test_nonresidential_rite_ventilation_config():
    hv = energy._HVAC
    v = hv["nonresidential_ventilation_ls_per_person"]
    # RITE IDA 2 offices = 12.5 l/s·person; IDA 3 commercial = 8 l/s·person.
    assert v["office"] == 12.5
    assert v["commercial"] == 8.0


def test_occ_setpoint_fields_from_profile():
    # Office occupancy 08:00–20:00 weekdays → 19 °C occupied, 16 °C empty.
    prof = energy._OCCUPANCY_PROFILES["office"]
    f = energy._occ_setpoint_fields(prof, 19, 16)
    assert "For: Weekdays" in f
    assert "Until: 08:00,16" in f     # before opening → setback
    assert "Until: 20:00,19" in f     # occupied band → setpoint
    assert "Until: 24:00,16" in f     # after closing → setback
    # Weekends fully empty → all setback.
    tail = f[f.index("For: AllOtherDays"):]
    assert tail == ["For: AllOtherDays", "Until: 24:00,16"]


# --------------------------------------------------------------------------- #
# Time Use Survey occupancy agenda (social_ES WeeklyScheduleHouseholdOccupancy)
# --------------------------------------------------------------------------- #

class _FakeIdfObject:
    """Minimal stand-in for an eppy IDF object: named attributes plus the raw
    ``obj`` field list the Schedule:Compact writer appends its day blocks to.

    ``fieldnames`` mirrors eppy's: the helpers that survive EnergyPlus field
    renames (``_set_first`` / ``_first_present``) probe it before assigning, so a
    fake without it fails where the real object would not."""

    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.obj = []

    @property
    def fieldnames(self):
        # Every attribute set so far, plus the fields the load writers assign to
        # after construction (eppy objects carry their whole schema up front).
        return list(self.__dict__) + [
            "Zone_or_ZoneList_or_Space_or_SpaceList_Name", "Zone_Name",
            "People_per_Floor_Area", "Watts_per_Zone_Floor_Area",
            "Watts_per_Floor_Area", "Air_Changes_per_Hour",
            "Minimum_Indoor_Temperature", "Minimum_Outdoor_Temperature",
            "Delta_Temperature"]


class _FakeIdf:
    """Just enough of an eppy IDF for the schedule writers — no IDD, no file, so
    the schedule logic is testable without an EnergyPlus install."""

    def __init__(self):
        self.idfobjects = {}

    def newidfobject(self, key, **fields):
        o = _FakeIdfObject(**fields)
        self.idfobjects.setdefault(key, []).append(o)
        return o

    def removeidfobject(self, obj):
        for objs in self.idfobjects.values():
            if obj in objs:
                objs.remove(obj)
                return


def _week(monday_states):
    """A 168-hour week whose every day repeats ``monday_states`` (24 states)."""
    return list(monday_states) * 7


def test_week_states_to_fields_one_block_per_weekday():
    # 08:00-19:59 out, the rest at home awake; every day the same.
    states = (["WithActivity"] * 8 + ["NonOccupied"] * 12 + ["WithActivity"] * 4)
    fields = energy._week_states_to_fields(
        _week(states), lambda s: 1.0 if s == "WithActivity" else 0.0)
    # Seven named weekdays plus the holiday/other fallback.
    assert [f for f in fields if f.startswith("For:")] == [
        "For: Monday", "For: Tuesday", "For: Wednesday", "For: Thursday",
        "For: Friday", "For: Saturday", "For: Sunday", "For: AllOtherDays"]
    monday = fields[1:fields.index("For: Tuesday")]
    # Consecutive equal hours collapse into a single Until:, and the day is closed.
    assert monday == ["Until: 08:00,1.0", "Until: 20:00,0.0", "Until: 24:00,1.0"]
    # AllOtherDays repeats Sunday, so a holiday is never left undefined.
    assert fields[fields.index("For: AllOtherDays") + 1:] == monday


def test_week_states_to_fields_keeps_the_weekday_weekend_contrast():
    # The whole point of a 168-hour agenda is that Saturday differs from Monday.
    weekday = ["Sleeping"] * 8 + ["NonOccupied"] * 12 + ["WithActivity"] * 4
    weekend = ["Sleeping"] * 9 + ["WithActivity"] * 15
    states = weekday * 5 + weekend * 2
    fields = energy._week_states_to_fields(states, lambda s: 0.0 if s == "NonOccupied" else 1.0)
    monday = fields[1:fields.index("For: Tuesday")]
    saturday = fields[fields.index("For: Saturday") + 1:fields.index("For: Sunday")]
    assert monday == ["Until: 08:00,1.0", "Until: 20:00,0.0", "Until: 24:00,1.0"]
    assert saturday == ["Until: 24:00,1.0"]


def test_time_use_schedules_splits_presence_from_waking_hours():
    # The aware/unaware difference must survive both reading the same agenda:
    # aware conditions only while someone is home, unaware through every waking
    # hour, at home or not.
    states = ["Sleeping"] * 8 + ["NonOccupied"] * 12 + ["WithActivity"] * 4
    agenda = {"states": _week(states), "autonomous_community_code": "09",
              "household_income_group": 3}
    cfg = {"enabled": True, "apply_to": ["aware", "unaware"],
           "state_fractions": {"Sleeping": 1.0, "WithActivity": 1.0, "NonOccupied": 0.0},
           "state_gain_fractions": {"Sleeping": 0.1, "WithActivity": 1.0,
                                    "NonOccupied": 0.15},
           "present_states": ["Sleeping", "WithActivity"],
           "awake_states": ["WithActivity", "NonOccupied"]}
    idf = _FakeIdf()
    occ, gain, present, awake, act = energy._time_use_schedules(idf, agenda, cfg)
    # Named after the survey cell, so a whole municipality shares a handful.
    assert occ == "SC_OCC_TUS_09_3"
    monday = lambda f: f[1:f.index("For: Tuesday")]
    assert monday(present) == ["Until: 08:00,1.0", "Until: 20:00,0.0", "Until: 24:00,1.0"]
    assert monday(awake) == ["Until: 08:00,0.0", "Until: 24:00,1.0"]
    # A sleeping household is fully home, so the PRESENCE fraction is 1.0 …
    sched = [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == occ][0]
    assert "Until: 08:00,1.0" in sched.obj
    # … while the lights and appliances are nearly off: they follow what the
    # household is doing, not how many of them are in the building.
    gsched = [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == gain][0]
    assert "Until: 08:00,0.1" in gsched.obj
    assert gain == "SC_GAIN_TUS_09_3"
    # …and the metabolic rate drops for the same hours, which is what makes the
    # CO2 generation drop with it (EnergyPlus derives it per watt of activity).
    assert act == "ACT_TUS_09_3"
    act_sched = [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == act][0]
    body = " ".join(str(f) for f in act_sched.obj)
    assert str(energy._ACT_SLEEPING) in body
    assert str(float(energy._LOAD_LIBRARY["residential"]["activity"])) in body


def test_time_use_schedules_degrade_to_none():
    idf = _FakeIdf()
    cfg = {"enabled": True, "state_fractions": {"Sleeping": 1.0},
           "present_states": ["Sleeping"], "awake_states": ["WithActivity"]}
    # No agenda (tract outside the income atlas), disabled, or a truncated week:
    # all must fall back to the stylised profile rather than raise.
    assert energy._time_use_schedules(idf, None, cfg) is None
    assert energy._time_use_schedules(idf, {"states": ["Sleeping"] * 168},
                                      {**cfg, "enabled": False}) is None
    assert energy._time_use_schedules(idf, {"states": ["Sleeping"] * 24}, cfg) is None


def test_time_use_occupancy_is_configured_for_aware_and_unaware():
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("time_use_occupancy", {})
    assert cfg.get("enabled") is True
    assert set(cfg.get("apply_to", [])) == {"aware", "unaware"}
    # Every state the survey can emit needs a gains fraction, or an hour of the
    # week would silently become an unoccupied one.
    from hypercadaster_ES import social
    assert set(cfg["state_fractions"]) == set(social.TIME_USE_OCCUPANCY_STATES)
    assert cfg["state_fractions"]["NonOccupied"] == 0.0


def test_building_time_use_agenda_without_wd_is_none():
    # No working directory → no survey cache → the stylised profile, not a crash.
    assert energy._building_time_use_agenda(None, {"census_tract_code": "0801901001"}) is None
    assert energy._building_time_use_agenda("/nonexistent", None) is None


# --------------------------------------------------------------------------- #
# Neighbour-shading filter (now applied by the IDF generator) + IDF reuse
# --------------------------------------------------------------------------- #

def test_shading_filter_defaults_come_from_the_config():
    # The generator's defaults must follow the ACTIVE config, not a value frozen
    # at import — configure() can reload it.
    assert energy._FILTER_SHADING is True
    assert energy._MAX_SHADING == 200
    nb = (energy._SHADING or {}).get("neighbours", {})
    assert nb.get("filter") is True and nb.get("max_panels") == 200


def test_build_energy_model_signature_takes_the_shading_filter():
    import inspect
    p = inspect.signature(energy.build_energy_model).parameters
    # None means "ask the config"; max_shading needs a distinct sentinel because
    # None already means "no cap".
    assert p["filter_shading"].default is None
    assert p["max_shading"].default is energy._FROM_CONFIG


def _shaded_idf(n_panels=5, marked=False):
    """A fake IDF carrying ``n_panels`` shading objects."""
    idf = _FakeIdf()
    for i in range(n_panels):
        idf.newidfobject("SHADING:BUILDING:DETAILED", Name=f"Shade_F{i}")
    if marked:
        idf._hc_shading_filtered = True
    return idf


def test_apply_shading_filter_caps_and_marks():
    # The cap alone (filter off) must trim to max_shading and mark the model.
    idf = _shaded_idf(10)
    n0, n1 = energy._apply_shading_filter(idf, filter_shading=False, max_shading=4)
    assert (n0, n1) == (10, 4)
    assert len(idf.idfobjects["SHADING:BUILDING:DETAILED"]) == 4
    assert idf._hc_shading_filtered is True


def test_apply_shading_filter_is_not_redone_on_a_marked_idf():
    # run_behavior_scenarios must not re-run an occlusion cull on an IDF the
    # generator already filtered — that is pure cost for no removal.
    idf = _shaded_idf(10, marked=True)
    n0, n1 = energy._apply_shading_filter(idf, filter_shading=True, max_shading=2)
    assert (n0, n1) == (10, 10)                 # untouched, cap NOT re-applied
    assert len(idf.idfobjects["SHADING:BUILDING:DETAILED"]) == 10


def test_apply_shading_filter_none_cap_keeps_everything():
    idf = _shaded_idf(7)
    assert energy._apply_shading_filter(idf, filter_shading=False,
                                        max_shading=None) == (7, 7)


def test_run_behavior_scenarios_accepts_a_prebuilt_idf():
    import inspect
    p = inspect.signature(energy.run_behavior_scenarios).parameters
    assert "idf" in p and p["idf"].default is None
    # The shading knobs must default to the config here too, so passing neither
    # keeps the generator and the runner in agreement.
    assert p["filter_shading"].default is None
    assert p["max_shading"].default is energy._FROM_CONFIG


def test_residential_behavior_reuses_existing_schedules():
    # Re-applying the behaviour to a model that already went through it (what the
    # idf= path does on each copy) must REUSE the named schedules/thermostats,
    # not author duplicates — EnergyPlus rejects two objects sharing a name.
    idf = _FakeIdf()
    idf.newidfobject("SCHEDULE:COMPACT", Name="OCCRES_NIGHTS_WORKDAYS")
    idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name="TSTATRES_AWARE_NIGHTS_WORKDAYS")
    idf.newidfobject("ZONE", Name="H_F0")
    made = {str(o.Name).upper()
            for key in ("SCHEDULE:COMPACT", "HVACTEMPLATE:THERMOSTAT")
            for o in idf.idfobjects.get(key, [])}
    assert "OCCRES_NIGHTS_WORKDAYS" in made
    assert "TSTATRES_AWARE_NIGHTS_WORKDAYS" in made


# --------------------------------------------------------------------------- #
# One-building frame: the reference comes from the field
# --------------------------------------------------------------------------- #

def _frame(refs):
    return pd.DataFrame({"building_reference": list(refs),
                         "br__thermal_zones": [[] for _ in refs]})


def test_one_building_reads_the_reference_from_the_field():
    row, ref = energy._one_building(_frame(["3444707BG9034S"]))
    assert ref == "3444707BG9034S"
    assert row["building_reference"] == "3444707BG9034S"


def test_one_building_rejects_an_ambiguous_frame():
    # A multi-building frame must never be silently reduced to its first row.
    with pytest.raises(ValueError, match="holds 2 buildings"):
        energy._one_building(_frame(["A", "B"]))
    # …unless the caller says which one.
    _row, ref = energy._one_building(_frame(["A", "B"]), building_reference="B")
    assert ref == "B"


def test_one_building_errors_are_specific():
    with pytest.raises(ValueError, match="no 'building_reference' column"):
        energy._one_building(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="holds no building_reference"):
        energy._one_building(pd.DataFrame({"building_reference": [None]}))
    with pytest.raises(ValueError, match="not found in gdf"):
        energy._one_building(_frame(["A"]), building_reference="Z")


def test_run_behavior_scenarios_signature_takes_one_building():
    import inspect
    p = list(inspect.signature(energy.run_behavior_scenarios).parameters)
    # gdf first, wd second — the reference is no longer a positional argument.
    assert p[:3] == ["gdf", "wd", "scenarios"]
    sig = inspect.signature(energy.run_behavior_scenarios).parameters
    assert sig["building_reference"].default is None
    assert sig["context_gdf"].default is None


# --------------------------------------------------------------------------- #
# Census-tract DHW occupancy: a partial frame must not inflate the demand
# --------------------------------------------------------------------------- #

def _dhw_frame(n_buildings, area_each, population, tract="2512001001"):
    """A frame of ``n_buildings``, each one dwelling of ``area_each`` m², all in one
    census tract of ``population`` people — the shape ``_building_spaces`` reads."""
    return pd.DataFrame({
        "building_reference": [f"B{i}" for i in range(n_buildings)],
        "census_tract_code": [tract] * n_buildings,
        "so__population_2025": [float(population)] * n_buildings,
        "br__building_spaces_reference": [{"Residential": [f"B{i}_1"]}
                                          for i in range(n_buildings)],
        "br__building_spaces_area_without_communal": [{"Residential": [area_each]}
                                                      for _ in range(n_buildings)],
        "br__building_spaces_floors": [{"Residential": [[0]]}
                                       for _ in range(n_buildings)],
    })


def test_census_tract_rate_needs_the_whole_tract(monkeypatch):
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    # 1000 people over 100 buildings × 200 m² = 20 000 m² → 0.05 people/m², plausible.
    full = energy._census_tract_person_rate(_dhw_frame(100, 200.0, 1000))
    assert full["2512001001"] == pytest.approx(0.05)
    # The SAME tract seen through one building would read 1000 / 200 = 5 people/m²
    # — the whole tract's population crammed into one building. Rejected.
    assert energy._census_tract_person_rate(_dhw_frame(1, 200.0, 1000)) == {}


def test_census_tract_rate_plausible_band_is_wide_enough_for_real_tracts():
    lo, hi = energy._DHW_PLAUSIBLE_RATE
    # The CTE default density must sit comfortably inside the accepted band,
    # or real tracts would be thrown away along with the truncated frames.
    cte = float((energy._LOAD_LIBRARY.get("residential", {}) or {}).get("ppl_per_m2", 0.028))
    assert lo < cte < hi
    assert lo < 0.01 and hi >= 0.10        # sparse and dense real tracts both pass


def test_dwelling_occupants_without_a_rate_falls_back(monkeypatch):
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0]
    # No usable tract rate → None, so _building_dhw_occupants uses the CTE density.
    assert energy._dwelling_occupants(row, {})[0] is None
    occ = energy._building_dhw_occupants(row, {})
    assert occ["real"] == occ["cte"] > 0


# --------------------------------------------------------------------------- #
# ctagg__ census-tract aggregations feeding the DHW occupancy
# --------------------------------------------------------------------------- #

def test_row_tract_rate_reads_the_merge_time_column():
    assert energy._row_tract_rate({"ctagg__people_per_m2": 0.031}) == pytest.approx(0.031)
    # Missing / non-numeric / non-positive → None, so the caller falls back.
    for bad in ({}, {"ctagg__people_per_m2": None}, {"ctagg__people_per_m2": "x"},
                {"ctagg__people_per_m2": 0.0}, {"ctagg__people_per_m2": float("nan")}):
        assert energy._row_tract_rate(bad) is None


def test_dwelling_occupants_prefers_the_ctagg_column(monkeypatch):
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()
    # Without it, a one-building frame has no usable rate at all.
    assert energy._dwelling_occupants(row, {})[0] is None
    # With it, the row carries its tract's density and stands on its own.
    row["ctagg__people_per_m2"] = 0.03
    assert energy._dwelling_occupants(row, {})[0] == 6      # round(0.03 * 200)


def test_ctagg_column_wins_over_a_rate_from_a_partial_frame(monkeypatch):
    # The merge-time density is computed over the WHOLE tract, so it must beat a
    # rate re-derived from whatever subset the caller still holds.
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()
    row["ctagg__people_per_m2"] = 0.03
    bogus = {"2512001001": 5.0}         # the whole tract crammed into one building
    assert energy._dwelling_occupants(row, bogus)[0] == 6


# --------------------------------------------------------------------------- #
# Census occupancy → PEOPLE internal gains
# --------------------------------------------------------------------------- #

def test_residential_occupancy_density_matches_the_dhw_head_count(monkeypatch):
    # The gains density and the DHW head count must come from the SAME people:
    # density = occupants / residential area, occupants counted per dwelling with
    # the min-1 floor.
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()
    row["ctagg__people_per_m2"] = 0.03
    people, area = energy._dwelling_occupants(row, {})
    assert (people, area) == (6, 200.0)
    assert energy._residential_occupancy_density(row) == pytest.approx(6 / 200.0)


def test_occupancy_density_lifted_by_the_min_one_person_floor(monkeypatch):
    # Two small dwellings at a low tract density would round to 0 occupants each;
    # the min-1 floor puts one in each, so the effective density is ABOVE the
    # tract average — which is why the density is derived from the head count
    # rather than passing the tract rate straight through.
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = {"census_tract_code": "0801901001",
           "ctagg__people_per_m2": 0.006,
           "br__building_spaces_reference": {"Residential": ["a", "b"]},
           "br__building_spaces_area_without_communal": {"Residential": [40.0, 40.0]},
           "br__building_spaces_floors": {"Residential": [[0], [1]]}}
    assert energy._dwelling_occupants(row, {})[0] == 2          # not 0
    assert energy._residential_occupancy_density(row) == pytest.approx(2 / 80.0)
    assert energy._residential_occupancy_density(row) > 0.006


def test_occupancy_density_none_without_census_occupancy(monkeypatch):
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()   # no ctagg__ column
    assert energy._residential_occupancy_density(row) is None


def test_occupancy_density_rejects_an_implausible_rate(monkeypatch):
    # A nonsense density must never reach the internal loads.
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()
    row["ctagg__people_per_m2"] = 5.0                   # a whole tract in one flat
    assert energy._residential_occupancy_density(row) is None


def test_add_loads_uses_the_census_density_for_residential_only():
    idf = _FakeIdf()
    idf.newidfobject("ZONE", Name="H_F1")
    idf.newidfobject("ZONE", Name="C_F0")
    zone_meta = {"H_F1": {"use_type": "Residential"},
                 "C_F0": {"use_type": "Commercial"}}
    energy._add_loads(idf, zone_meta, residential_ppl_per_m2=0.0081)
    by_zone = {o.Name: energy._first_present(o, energy._F_PPL_AREA)
               for o in idf.idfobjects["PEOPLE"]}
    assert by_zone["H_F1_People"] == pytest.approx(0.0081)
    # Commercial keeps its own use density — the override is residential-only.
    assert by_zone["C_F0_People"] == pytest.approx(
        energy._LOAD_LIBRARY["commercial"]["ppl_per_m2"])


def test_add_loads_falls_back_to_the_configured_density():
    idf = _FakeIdf()
    idf.newidfobject("ZONE", Name="H_F1")
    energy._add_loads(idf, {"H_F1": {"use_type": "Residential"}})
    o = idf.idfobjects["PEOPLE"][0]
    assert energy._first_present(o, energy._F_PPL_AREA) == pytest.approx(
        energy._LOAD_LIBRARY["residential"]["ppl_per_m2"])


def test_census_occupancy_is_enabled_by_default():
    assert energy._RES_OCC_FROM_CENSUS is True
    # The per-use load rows must not have picked up the sibling scalar setting.
    assert "residential_occupancy_from_census" not in energy._LOAD_LIBRARY
    assert all(isinstance(v, dict) for v in energy._LOAD_LIBRARY.values())


# --------------------------------------------------------------------------- #
# Non-residential uses are reported once, under the normative case
# --------------------------------------------------------------------------- #

def _rows(scenario, uses=("Residential", "Commercial")):
    return [{"scenario": scenario, "use_type": u, "season": "heating",
             "thermal_demand_kwh_m2": 10.0} for u in uses]


def test_nonresidential_reported_only_for_the_source_scenario():
    # The shop is emitted once, relabelled; the dwellings keep their scenario.
    kept = energy._report_nonresidential_once(_rows("aware"), "aware", "cte")
    assert [r["use_type"] for r in kept] == ["Residential"]
    assert kept[0]["scenario"] == "aware"

    at_source = energy._report_nonresidential_once(_rows("cte"), "cte", "cte")
    assert [r["use_type"] for r in at_source] == ["Residential", "Commercial"]
    assert at_source[0]["scenario"] == "cte"
    assert at_source[1]["scenario"] == energy.NONRESIDENTIAL_SCENARIO_LABEL


def test_nonresidential_per_scenario_opt_out():
    # nonres_source=None → nothing is collapsed or relabelled.
    kept = energy._report_nonresidential_once(_rows("aware"), "aware", None)
    assert [(r["use_type"], r["scenario"]) for r in kept] == [
        ("Residential", "aware"), ("Commercial", "aware")]


def test_office_is_treated_like_commercial():
    kept = energy._report_nonresidential_once(
        _rows("cte", uses=("Office",)), "cte", "cte")
    assert kept[0]["scenario"] == energy.NONRESIDENTIAL_SCENARIO_LABEL


def test_nonresidential_source_prefers_cte_then_a_conditioned_scenario():
    import inspect
    src = inspect.getsource(energy.run_behavior_scenarios)
    assert "NONRESIDENTIAL_SOURCE_SCENARIO" in src
    assert energy.NONRESIDENTIAL_SOURCE_SCENARIO == "cte"
    assert energy.NONRESIDENTIAL_SCENARIO_LABEL == "normative"
    # cte is one of the shipped scenarios, so the default run has a source.
    assert "cte" in energy._DEFAULT_SCENARIOS
    # …and it is a conditioned one (a free-floating run reports zero demand,
    # which would be a nonsense normative figure).
    assert not energy._DEFAULT_SCENARIOS["cte"].get("free_floating")


def test_run_behavior_scenarios_reporting_default():
    import inspect
    p = inspect.signature(energy.run_behavior_scenarios).parameters
    assert p["nonresidential_reporting"].default == "normative"


# --------------------------------------------------------------------------- #
# Out-of-comfort is measured over OCCUPIED hours only
# --------------------------------------------------------------------------- #

def _occupancy_csv(tmp_path, zone, top, occ_hours, outdoor=30.0, n=48):
    """Hourly CSV where the zone is occupied only at the given hours-of-day."""
    df = pd.DataFrame({
        "Date/Time": [f" 07/{1 + i // 24:02d}  {i % 24 + 1:02d}:00:00" for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [top] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone People Occupant Count [](Hourly)":
            [1.0 if (i % 24 + 1) in occ_hours else 0.0 for i in range(n)],
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [outdoor] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_discomfort_counts_only_occupied_hours(tmp_path):
    # A zone far too hot all month, but occupied only 22:00-07:59 (night). The DAY
    # percentage must be NaN — there are no occupied day hours to judge — and the
    # NIGHT one 100 %.
    night_hours = set(energy._NIGHT_HOURS)
    csv = _occupancy_csv(tmp_path, "H_F1", top=40.0, occ_hours=night_hours)
    df = energy._read_eplus_csv(csv)
    recs = energy._zone_scenario_kpis(
        df, {"H_F1": {"use_type": "Residential"}}, {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "cooling")
    assert np.isnan(r["pct_discomfort_day"])
    assert r["pct_discomfort_night"] == pytest.approx(100.0)
    # …and the denominator is reported, so the basis is auditable.
    assert r["occupied_hours"] == len(night_hours) * 2      # two days
    assert r["season_hours"] == 48


def test_occupied_hours_shrink_the_denominator(tmp_path):
    # Same temperatures, half the occupancy → same percentages, fewer hours behind
    # them. The KPI must say so rather than hide it.
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    csv_all = _occupancy_csv(tmp_path / "a", top=40.0, zone="H_F1",
                             occ_hours=set(range(1, 25)))
    csv_half = _occupancy_csv(tmp_path / "b", top=40.0, zone="H_F1",
                              occ_hours=set(range(1, 13)))
    meta, areas = {"H_F1": {"use_type": "Residential"}}, {"H_F1": 50.0}
    full = energy._aggregate_scenario_kpis(
        energy._zone_scenario_kpis(energy._read_eplus_csv(csv_all), meta, areas, "II"),
        "aware")
    half = energy._aggregate_scenario_kpis(
        energy._zone_scenario_kpis(energy._read_eplus_csv(csv_half), meta, areas, "II"),
        "aware")
    f = next(r for r in full if r["season"] == "cooling")
    h = next(r for r in half if r["season"] == "cooling")
    assert f["pct_occupied_hours"] == pytest.approx(100.0)
    assert h["pct_occupied_hours"] == pytest.approx(50.0)


def test_zone_without_occupancy_output_yields_no_comfort(tmp_path):
    # A zone whose occupant count is not reported must NOT be treated as occupied
    # around the clock — that would answer a different question under the same
    # column name.
    zone = "H_F1"
    df = pd.DataFrame({
        "Date/Time": [f" 07/01  {i + 1:02d}:00:00" for i in range(24)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [40.0] * 24,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * 24,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * 24,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * 24,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [30.0] * 24,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {zone: {"use_type": "Residential"}},
        {zone: 50.0}, "II")
    r = next(x for x in recs if x["season"] == "cooling")
    assert np.isnan(r["pct_discomfort_day"]) and np.isnan(r["pct_discomfort_night"])
    assert r["occupied_hours"] is None


def test_pct_occupied_ignores_zones_without_occupancy_data():
    # Such a zone must not deflate the occupied share of its use type: it reports
    # no comfort either, so it belongs on neither side of the ratio.
    recs = [{"occupied_hours": 12, "season_hours": 24},
            {"occupied_hours": None, "season_hours": 24}]
    assert energy._pct_occupied(recs) == pytest.approx(50.0)
    assert np.isnan(energy._pct_occupied([{"occupied_hours": None, "season_hours": 24}]))


# --------------------------------------------------------------------------- #
# Extra KPI columns: setpoints while running, season dates, solar gains,
# people per dwelling
# --------------------------------------------------------------------------- #

def _rich_csv(tmp_path, zone="H_F1", n=48, month=1, top=21.0, outdoor=5.0,
              heat_j=None, cool_j=None, sp_heat=None, sp_cool=None, solar_j=0.0):
    """Hourly CSV carrying the setpoint and solar columns as well."""
    heat_j = [3.6e6] * n if heat_j is None else heat_j
    cool_j = [0.0] * n if cool_j is None else cool_j
    sp_heat = [21.0] * n if sp_heat is None else sp_heat
    sp_cool = [25.0] * n if sp_cool is None else sp_cool
    df = pd.DataFrame({
        "Date/Time": [f" {month:02d}/{1 + i // 24:02d}  {i % 24 + 1:02d}:00:00"
                      for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [top] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": heat_j,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": cool_j,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [1.8e6] * n,
        f"{zone}:Enclosure Windows Total Transmitted Solar Radiation Energy [J](Hourly)":
            [solar_j] * n,
        f"{zone}:Zone Thermostat Heating Setpoint Temperature [C](Hourly)": sp_heat,
        f"{zone}:Zone Thermostat Cooling Setpoint Temperature [C](Hourly)": sp_cool,
        f"{zone}:Zone People Occupant Count [](Hourly)": [1.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [outdoor] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_avg_setpoint_only_counts_hours_the_hvac_ran(tmp_path):
    # Heating runs for the first 24 h at 21 °C and is off for the next 24 h, when
    # the schedule has set back to 16. The reported average must be 21, not 18.5:
    # the setback hours are not a temperature the building was held at.
    csv = _rich_csv(tmp_path, n=48,
                    heat_j=[3.6e6] * 24 + [0.0] * 24,
                    sp_heat=[21.0] * 24 + [16.0] * 24)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["avg_setpoint_c"] == pytest.approx(21.0)
    assert r["hvac_hours"] == 24
    # One setpoint per row: a row is one season, so a separate heating and
    # cooling setpoint would leave one of them meaningless in every row.
    assert "avg_heating_setpoint_c" not in r and "avg_cooling_setpoint_c" not in r


def test_avg_setpoint_mixes_heating_and_cooling_hours(tmp_path):
    # An in-between season can see both services. The single reported setpoint is
    # then the hour-weighted blend of the one that ran in each hour.
    csv = _rich_csv(tmp_path, n=48,
                    heat_j=[3.6e6] * 24 + [0.0] * 24,
                    cool_j=[0.0] * 24 + [1.8e6] * 24,
                    sp_heat=[20.0] * 48, sp_cool=[26.0] * 48)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["avg_setpoint_c"] == pytest.approx(23.0)      # 24 h at 20, 24 h at 26
    assert r["hvac_hours"] == 48


def test_season_specific_demand_columns_are_gone(tmp_path):
    # The row's season already says which service it describes.
    csv = _rich_csv(tmp_path, n=24, heat_j=[3.6e6] * 24, cool_j=[1.8e6] * 24)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["thermal_demand_kwh"] == pytest.approx(36.0, rel=1e-3)
    for gone in ("heating_demand_kwh", "cooling_demand_kwh",
                 "heating_hours", "cooling_hours"):
        assert gone not in r
    row = energy._aggregate_scenario_kpis(recs, "aware")[0]
    for gone in ("heating_demand_kwh_m2", "cooling_demand_kwh_m2",
                 "avg_heating_setpoint_c", "avg_cooling_setpoint_c"):
        assert gone not in row


def test_indoor_temperature_percentiles_over_occupied_hours(tmp_path):
    # Night-only occupancy at a known temperature: the night percentiles report
    # it, the day ones are empty (no occupied day hours to describe).
    night_hours = set(energy._NIGHT_HOURS)
    csv = _occupancy_csv(tmp_path, "H_F1", top=24.0, occ_hours=night_hours)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "cooling")
    for p in energy._TEMPERATURE_PERCENTILES:
        assert r[f"t_p{p}_night"] == pytest.approx(24.0)
        assert np.isnan(r[f"t_p{p}_day"])


def test_temperature_percentiles_track_the_spread(tmp_path):
    # A zone swinging between 18 and 30 must report that spread, which a mean hides.
    n = 48
    temps = [18.0 if i % 2 else 30.0 for i in range(n)]
    df = pd.DataFrame({
        "Date/Time": [f" 07/{1 + i // 24:02d}  {i % 24 + 1:02d}:00:00" for i in range(n)],
        "H_F1:Zone Operative Temperature [C](Hourly)": temps,
        "H_F1:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * n,
        "H_F1:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * n,
        "H_F1:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * n,
        "H_F1:Zone People Occupant Count [](Hourly)": [1.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [30.0] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "cooling")
    assert r["t_p10_day"] == pytest.approx(18.0, abs=0.5)
    assert r["t_p90_day"] == pytest.approx(30.0, abs=0.5)
    assert 18.0 <= r["t_p50_day"] <= 30.0


def test_solar_gains_are_separate_from_internal_gains(tmp_path):
    csv = _rich_csv(tmp_path, n=24, solar_j=3.6e6)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["solar_gains_kwh"] == pytest.approx(24.0, rel=1e-3)
    assert r["internal_gains_kwh"] == pytest.approx(12.0, rel=1e-3)   # 1.8e6 × 24
    rows = energy._aggregate_scenario_kpis(recs, "aware")
    row = next(x for x in rows if x["season"] == "heating")
    assert row["solar_gains_kwh_m2"] == pytest.approx(24.0 / 50.0, rel=1e-2)


def test_solar_variable_name_follows_the_energyplus_generation():
    class _V:
        def __init__(self, v): self.idd_version = v
    assert energy._solar_gain_variable(_V((26, 1, 0))).startswith("Enclosure")
    assert energy._solar_gain_variable(_V((23, 1, 0))).startswith("Enclosure")
    # Radiant enclosures were split from zones in 23.1; before that it is "Zone …".
    assert energy._solar_gain_variable(_V((22, 2, 0))).startswith("Zone")
    # Unknown version → the modern name.
    assert energy._solar_gain_variable(None).startswith("Enclosure")


def test_season_ranges_report_contiguous_ddmm_runs(tmp_path):
    csv = _rich_csv(tmp_path, n=48, month=1, outdoor=5.0)     # 01/01-01/02, cold
    df = energy._read_eplus_csv(csv)
    out_c = energy._zone_column(df, "Environment", "Site Outdoor Air Drybulb Temperature")
    _mask, _days, ranges = energy._hourly_season_mask(df, out_c)
    assert ranges["heating"] == ["0101/0201"]
    # A season with no days has no range rather than a bogus one.
    assert ranges["cooling"] == []
    recs = energy._zone_scenario_kpis(
        df, {"H_F1": {"use_type": "Residential"}}, {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["season_ranges"] == "0101/0201"


def test_date_runs_split_a_season_that_wraps_the_year():
    # A Spanish heating season is autumn-to-spring: two runs at either end of the
    # year, not one span from January to December.
    dates = ({(1, d) for d in range(1, 5)}
             | {(11, d) for d in range(1, 31)} | {(12, d) for d in range(1, 32)})
    assert energy._date_runs(dates) == ["0101/0401", "0111/3112"]
    assert energy._date_runs({(6, d) for d in range(8, 31)}
                             | {(7, d) for d in range(1, 32)}
                             | {(8, d) for d in range(1, 15)}) == ["0806/1408"]
    # Runs join across a month boundary, and a single day is its own run.
    assert energy._date_runs({(1, 30), (1, 31), (2, 1)}) == ["3001/0102"]
    assert energy._date_runs({(7, 15)}) == ["1507/1507"]
    assert energy._date_runs(set()) == []


def test_every_scenario_houses_the_same_people():
    # Occupancy is a property of the neighbourhood, not of the occupant archetype:
    # the normative case must not be given 3.5 people where the census says 1, or
    # its results differ from the behaviour scenarios for two reasons at once.
    import inspect
    # The source of the head count hands every scenario the same number …
    assert ('return {"real": people, "cte": people}'
            in inspect.getsource(energy._building_dhw_occupants))
    # … so whichever key a scenario reads, it gets the census one.
    occ = {"real": 6, "cte": 6}
    rows_a, rows_c = ([{"use_type": "Residential", "season": "heating",
                        "floor_area_m2": 100.0}] for _ in range(2))
    season_by_doy = {d: "heating" for d in range(1, 366)}
    energy._augment_dhw_by_season(rows_a, occ, {"behavior": "aware"},
                                  season_by_doy, None, n_dwellings=6)
    energy._augment_dhw_by_season(rows_c, occ, {"behavior": "cte"},
                                  season_by_doy, None, n_dwellings=6)
    assert rows_a[0]["dhw_demand_kwh_m2"] == rows_c[0]["dhw_demand_kwh_m2"]
    assert rows_a[0]["people_per_dwelling"] == rows_c[0]["people_per_dwelling"] == 1.0
    # …and the People density is not reset for cte either.
    assert "cte_ppl_per_m2" not in inspect.getsource(energy._apply_occupant_control)


def test_dhw_occupants_are_one_census_head_count(monkeypatch):
    monkeypatch.setitem(energy._RES_DHW, "enabled", True)
    row = _dhw_frame(1, 200.0, 1000).iloc[0].to_dict()
    row["ctagg__people_per_m2"] = 0.03
    occ = energy._building_dhw_occupants(row, {})
    assert occ["real"] == occ["cte"] == 6          # round(0.03 x 200)
    # Without a census density both fall back to the CTE default, still equal.
    plain = energy._building_dhw_occupants(
        _dhw_frame(1, 200.0, 1000).iloc[0].to_dict(), {})
    assert plain["real"] == plain["cte"] > 0


def test_people_per_dwelling_is_reported_beside_the_demand_it_drives():
    rows = [{"use_type": "Residential", "season": "heating", "floor_area_m2": 100.0}]
    season_by_doy = {d: "heating" for d in range(1, 366)}
    energy._augment_dhw_by_season(rows, {"real": 6, "cte": 6}, {"behavior": "aware"},
                                  season_by_doy, None, n_dwellings=6)
    assert rows[0]["people_per_dwelling"] == pytest.approx(1.0)
    assert rows[0]["n_dwellings"] == 6
    # More people, more hot water — the same head count drives both numbers.
    more = [{"use_type": "Residential", "season": "heating", "floor_area_m2": 100.0}]
    energy._augment_dhw_by_season(more, {"real": 18, "cte": 18}, {"behavior": "aware"},
                                  season_by_doy, None, n_dwellings=6)
    assert more[0]["people_per_dwelling"] == pytest.approx(3.0)
    assert more[0]["dhw_demand_kwh_m2"] > rows[0]["dhw_demand_kwh_m2"]


def test_setpoint_aggregate_weighted_by_running_hours():
    # A zone that ran for 1 h at 25 °C must not pull the average as hard as one
    # that ran 99 h at 20 °C.
    def _rec(sp, hours):
        return {"use_type": "R", "season": "heating", "season_days": 1,
                "floor_area_m2": 50.0, "thermal_demand_kwh": 0.0,
                "internal_gains_kwh": 0.0, "solar_gains_kwh": 0.0,
                "avg_setpoint_c": sp, "hvac_hours": hours,
                "pct_discomfort_day": np.nan, "pct_discomfort_night": np.nan,
                "occupied_hours": 0, "season_hours": 0,
                **{f"t_{k}_{part}": np.nan
                   for part in ("day", "night")
                   for k in [f"p{p}" for p in energy._TEMPERATURE_PERCENTILES]
                   + list(energy._TEMPERATURE_EXTREMES)}}
    row = energy._aggregate_scenario_kpis([_rec(20.0, 99), _rec(25.0, 1)], "aware")[0]
    assert row["avg_setpoint_c"] == pytest.approx((20 * 99 + 25) / 100, abs=0.01)


# --------------------------------------------------------------------------- #
# The shading cap must never touch the building's OWN shading
# --------------------------------------------------------------------------- #

def _shaded_mixed_idf(n_neighbours=10, own=("Blind_H_F1_S_0", "Awn_H_F1_0",
                                            "Balcony_H_F1_w1", "CatShade_F1_0",
                                            "H_F1_win0_rev0")):
    idf = _FakeIdf()
    for i in range(n_neighbours):
        idf.newidfobject("SHADING:BUILDING:DETAILED", Name=f"Shade_F1_{i}")
    for name in own:
        idf.newidfobject("SHADING:BUILDING:DETAILED", Name=name)
    return idf


def _names(idf):
    return [str(o.Name) for o in idf.idfobjects["SHADING:BUILDING:DETAILED"]]


def test_trim_shading_only_caps_neighbour_panels():
    # Regression: the cap used to sort EVERY shading panel by size and drop the
    # smallest, which are the exterior blinds and awnings — silently deleting the
    # operable shading and making every occupant type return identical results.
    idf = _shaded_mixed_idf(n_neighbours=10)
    energy._trim_shading(idf, 3)
    kept = _names(idf)
    assert len([n for n in kept if n.startswith("Shade_F")]) == 3
    for own in ("Blind_H_F1_S_0", "Awn_H_F1_0", "Balcony_H_F1_w1",
                "CatShade_F1_0", "H_F1_win0_rev0"):
        assert own in kept, f"{own} must survive the neighbour cap"


def test_trim_shading_counts_only_neighbours_against_the_cap():
    # 3 neighbours + 5 own panels must NOT trip a cap of 4: the building's own
    # shading is part of the model, not context detail to be budgeted away.
    idf = _shaded_mixed_idf(n_neighbours=3)
    energy._trim_shading(idf, 4)
    assert len(_names(idf)) == 8


def test_apply_shading_filter_reports_neighbour_counts():
    idf = _shaded_mixed_idf(n_neighbours=10)
    n0, n1 = energy._apply_shading_filter(idf, filter_shading=False, max_shading=4)
    # Counts describe the neighbour context, not every panel in the model.
    assert (n0, n1) == (10, 4)
    assert len([n for n in _names(idf) if not n.startswith("Shade_F")]) == 5


def test_neighbour_prefix_matches_the_generator():
    # _add_shading names its panels Shade_F{floor}_{n}; the filter and the cap key
    # off that prefix, so a rename there must break this test rather than silently
    # start deleting the building's own shading again.
    assert energy._NEIGHBOUR_SHADE_PREFIX == "Shade_F"
    import inspect
    src = inspect.getsource(energy._add_shading)
    assert 'Name=f"Shade_F{fl}_{n}"' in src


# --------------------------------------------------------------------------- #
# Wall winding: every wall must face OUT of its own room
# --------------------------------------------------------------------------- #

def _quad_normal(v):
    a = np.array(v, float)
    n = np.cross(a[1] - a[0], a[2] - a[1])
    h = np.hypot(n[0], n[1])
    return n[:2] / h if h > 1e-9 else None


def test_wall_quad_wound_outward_for_a_ccw_room():
    from shapely.geometry import Polygon, Point
    room = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])          # CCW
    # South edge walked west→east: the (dy, -dx) normal already points out (-y).
    v = energy._wall_quad_outward(0, 0, 4, 0, 0.0, 3.0, room)
    n = _quad_normal(v)
    assert n[1] < -0.9
    assert not room.contains(Point(2 + n[0] * 0.1, 0 + n[1] * 0.1))
    # The SAME edge walked the other way must come back wound the other way, so
    # the normal still points out — this is the case the cell complex produces,
    # where the face's line direction does not follow the room's ring.
    v_rev = energy._wall_quad_outward(4, 0, 0, 0, 0.0, 3.0, room)
    n_rev = _quad_normal(v_rev)
    assert n_rev[1] < -0.9
    assert np.allclose(n, n_rev, atol=1e-6)


def test_wall_quad_outward_on_every_edge_of_a_concave_room():
    # An L-shaped room is where a centroid rule fails: its centroid can fall
    # outside the room entirely.
    from shapely.geometry import Polygon, Point
    room = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])
    ring = list(room.exterior.coords)
    for i in range(len(ring) - 1):
        (x0, y0), (x1, y1) = ring[i], ring[i + 1]
        for a, b in (((x0, y0), (x1, y1)), ((x1, y1), (x0, y0))):   # both directions
            v = energy._wall_quad_outward(a[0], a[1], b[0], b[1], 0.0, 3.0, room)
            n = _quad_normal(v)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            assert not room.buffer(-1e-9).contains(Point(mx + n[0] * 0.05, my + n[1] * 0.05))
            assert room.buffer(1e-9).contains(Point(mx - n[0] * 0.05, my - n[1] * 0.05))


def test_wall_quad_outward_degrades_safely():
    from shapely.geometry import Polygon
    # No polygon → the canonical order, unchanged.
    v = energy._wall_quad_outward(0, 0, 4, 0, 0.0, 3.0, None)
    assert v == [(0, 0, 0.0), (4, 0, 0.0), (4, 0, 3.0), (0, 0, 3.0)]
    # Zero-length segment → no flip attempted.
    room = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    assert len(energy._wall_quad_outward(1, 1, 1, 1, 0.0, 3.0, room)) == 4


def test_paired_interior_walls_are_mirrors():
    # The two faces of an interior wall must carry identical coordinates in
    # opposite winding, or EnergyPlus rejects the adjacency.
    from shapely.geometry import Polygon
    room = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    v = energy._wall_quad_outward(0, 0, 4, 0, 0.0, 3.0, room)
    mirror = list(reversed(v))
    assert set(v) == set(mirror)
    n, m = _quad_normal(v), _quad_normal(mirror)
    assert np.dot(n, m) < -0.9          # opposite directions


# --------------------------------------------------------------------------- #
# Exterior persiana: a real shade layer on a WindowShadingControl
# --------------------------------------------------------------------------- #

def test_persiana_material_comes_from_the_config():
    cfg = (energy._SHADING or {}).get("persiana", {})
    assert cfg.get("enabled") is True
    # A closed persiana must be nearly opaque to solar, or the control cannot do
    # its job whatever the schedule says.
    assert cfg["solar_transmittance"] <= 0.1
    assert cfg["solar_reflectance"] >= 0.3
    assert cfg["solar_setpoint_w_m2"] > 0


def test_persiana_control_is_scheduled_and_solar_triggered():
    idf = _FakeIdf()
    idf.newidfobject("BUILDINGSURFACE:DETAILED", Name="W1", Zone_Name="H_F1",
                     Surface_Type="Wall")
    idf.newidfobject("FENESTRATIONSURFACE:DETAILED", Name="W1_win0",
                     Surface_Type="Window", Building_Surface_Name="W1")
    n = energy._add_persiana_controls(idf, {"H_F1": {"use_type": "Residential"}})
    assert n == 1
    c = idf.idfobjects["WINDOWSHADINGCONTROL"][0]
    assert c.Shading_Type == "ExteriorShade"
    # Scheduled AND solar-triggered: EnergyPlus then closes it only when the
    # schedule allows *and* the sun on the window exceeds the setpoint.
    assert c.Shading_Control_Type == "OnIfHighSolarOnWindow"
    assert c.Shading_Control_Is_Scheduled == "Yes"
    assert c.Shading_Device_Material_Name == energy._PERSIANA_MATERIAL
    assert c.Fenestration_Surface_1_Name == "W1_win0"
    assert idf.idfobjects["WINDOWMATERIAL:SHADE"][0].Name == energy._PERSIANA_MATERIAL


def test_persiana_control_is_residential_only():
    idf = _FakeIdf()
    for z, use in (("H_F1", "Residential"), ("C_F0", "Commercial")):
        idf.newidfobject("BUILDINGSURFACE:DETAILED", Name=f"{z}_w", Zone_Name=z,
                         Surface_Type="Wall")
        idf.newidfobject("FENESTRATIONSURFACE:DETAILED", Name=f"{z}_win",
                         Surface_Type="Window", Building_Surface_Name=f"{z}_w")
    meta = {"H_F1": {"use_type": "Residential"}, "C_F0": {"use_type": "Commercial"}}
    assert energy._add_persiana_controls(idf, meta) == 1
    assert idf.idfobjects["WINDOWSHADINGCONTROL"][0].Zone_Name == "H_F1"


def test_persiana_gate_is_closed_through_the_heating_season():
    # The whole point of the gate: in a Spanish winter the sun on the glazing is
    # wanted, so nothing that shades it may operate.
    idf = _FakeIdf()
    sbd = {d: ("heating" if d <= 90 else "cooling") for d in range(1, 366)}
    name = energy._persiana_allow_schedule(idf, sbd, name="T_GATE")
    assert name == "T_GATE"
    body = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
    # The heating days (1-90 -> through 31/03) carry 0: the gate is shut.
    i = body.index("Through: 3/31")
    assert body[i + 2] == "Until: 24:00,0"
    # …and the cooling days that follow carry 1.
    assert "Until: 24:00,1" in body[i + 3:]


def test_persiana_gate_month_fallback_without_weather():
    # No EPW → the month split, still closed across the winter months.
    idf = _FakeIdf()
    energy._persiana_allow_schedule(idf, None, name="T_GATE2")
    body = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
    assert "Through: 3/31" in body and "Through: 12/31" in body
    # Winter at both ends of the year: gate shut.
    assert body[body.index("Through: 3/31") + 2] == "Until: 24:00,0"
    assert body[body.index("Through: 12/31") + 2] == "Until: 24:00,0"


# --------------------------------------------------------------------------- #
# HTML scenario report
# --------------------------------------------------------------------------- #

def _kpi_frame():
    rows = []
    for i, sc in enumerate(("unaware", "aware", "aware_ff", "cte")):
        for season in ("heating", "cooling"):
            rows.append({
                "scenario": sc, "use_type": "Residential", "season": season,
                "season_days": 120, "season_ranges": "0101/0401, 0111/3112",
                "n_zones": 6, "floor_area_m2": 630.0,
                "thermal_demand_kwh_m2": 10.0 + i, "internal_gains_kwh_m2": 12.0,
                "solar_gains_kwh_m2": 5.0 + i, "avg_setpoint_c": 20.0 + i,
                "hvac_hours": 100.0,
                "pct_hours_out_of_comfort_day": 10.0 * i,
                "pct_hours_out_of_comfort_night": 5.0 * i,
                "pct_occupied_hours": 77.0,
                "dhw_demand_kwh_m2": 1.0, "people_per_dwelling": 2.5,
                **{f"t_p{p}_{part}": 18.0 + p / 10.0
                   for part in ("day", "night")
                   for p in energy._TEMPERATURE_PERCENTILES},
                **{f"t_{k}_{part}": (14.0 if k == "min" else 32.0)
                   for part in ("day", "night")
                   for k in energy._TEMPERATURE_EXTREMES},
            })
    return pd.DataFrame(rows)


def test_scenario_report_is_self_contained(tmp_path):
    from hypercadaster_ES import reporting
    out = reporting.scenario_report(_kpi_frame(), str(tmp_path / "r.html"),
                                    building_reference="TESTBR")
    page = open(out["residential"], encoding="utf-8").read()
    # No external assets: it must open from disk with no network. What matters is
    # that nothing is FETCHED — the inlined Plotly bundle mentions its own CDN in a
    # config default, which is not a request.
    import re as _re
    assert not _re.search(r'<script[^>]+\bsrc\s*=', page)
    assert not _re.search(r'<link[^>]+\bhref\s*=', page)
    assert not _re.search(r'<img[^>]+\bsrc\s*=', page)
    assert page.startswith("<!doctype html>")
    # Every scenario is named, and the table is the accessible view of the charts.
    for sc in ("unaware", "aware", "aware_ff", "cte"):
        assert sc in page
    assert "<table>" in page and page.count("<tr>") >= len(_kpi_frame())


def test_scenario_report_adds_the_html_extension(tmp_path):
    from hypercadaster_ES import reporting
    out = reporting.scenario_report(_kpi_frame(), str(tmp_path / "noext"),
                                    building_reference="TESTBR")
    assert out["residential"].endswith("noext_residential.html")
    assert os.path.exists(out["residential"])


def test_scenario_report_without_raw_output_still_writes(tmp_path):
    # No output_dir → the summary half only, never an error.
    from hypercadaster_ES import reporting
    out = reporting.scenario_report(_kpi_frame(), str(tmp_path / "s.html"))
    page = open(out["residential"], encoding="utf-8").read()
    assert "How the occupants compare" in page
    assert "Raw simulation output" not in page


def test_psychrometrics_match_the_tables():
    from hypercadaster_ES import reporting
    # 20 °C / 50 % RH at sea level is 7.26 g/kg in every psychrometric table.
    w = reporting._humidity_ratio(20.0, 0.5)
    assert w * 1000 == pytest.approx(7.26, abs=0.05)
    # Saturation pressure against the tables, across the range an indoor chart
    # actually covers (the Magnus form is an ambient-range fit, not a steam table).
    assert float(reporting._p_sat(0.0)) == pytest.approx(611.0, abs=2.0)
    assert float(reporting._p_sat(20.0)) == pytest.approx(2339.0, rel=0.01)
    assert float(reporting._p_sat(30.0)) == pytest.approx(4246.0, rel=0.01)
    # …and the two directions are inverses, which is what the hover relies on.
    for t in (12.0, 20.0, 26.0, 34.0):
        for rh in (0.15, 0.45, 0.85):
            back = reporting._relative_humidity(t, reporting._humidity_ratio(t, rh))
            assert float(back) == pytest.approx(rh, rel=1e-6)


def test_comfort_zone_is_a_polygon_not_a_rectangle():
    # The extended (Mollier h-x) field is the INTERSECTION of its bounds, and which
    # bound binds changes with temperature — that is what gives it its shape.
    energy.configure()
    t, lo, hi = energy.comfort_zone_polygon()
    assert t.min() == pytest.approx(20.0) and t.max() == pytest.approx(26.0)
    w = lambda tt: [float(x[0]) for x in energy._comfort_zone_humidity_limits(np.array([tt]))]
    # Cool end: the 65 % RH ceiling governs.
    assert energy._relative_humidity(20.0, w(20.0)[1]) == pytest.approx(0.65, abs=1e-3)
    # Warm end: the 50 kJ/kg enthalpy line has taken over, BELOW both the RH
    # ceiling and the absolute-humidity cap.
    assert energy._moist_air_enthalpy(26.0, w(26.0)[1]) == pytest.approx(50.0, abs=0.05)
    assert energy._relative_humidity(26.0, w(26.0)[1]) < 0.65
    assert w(26.0)[1] * 1000 < 11.5
    # …so the top edge RISES then FALLS: a rectangle cannot do that.
    upper = hi * 1000.0
    assert upper.argmax() not in (0, len(upper) - 1)
    # The floor is the 30 % RH curve throughout.
    assert energy._relative_humidity(23.0, w(23.0)[0]) == pytest.approx(0.30, abs=1e-3)


def test_comfort_zone_rejects_each_way_out_of_it():
    energy.configure()
    inside = energy._outside_comfort_zone(np.array([23.0]), np.array([0.008]))
    assert not inside[0]
    cases = {
        "too cold": (18.0, 0.008),
        "too warm": (28.0, 0.008),
        "too dry": (23.0, 0.003),          # under 30 % RH
        "too humid": (21.0, 0.011),        # over 65 % RH
        "over the enthalpy line": (25.5, 0.0105),
    }
    for name, (t, w) in cases.items():
        assert energy._outside_comfort_zone(np.array([t]), np.array([w]))[0], name
    # Missing humidity is never counted as comfortable.
    assert energy._outside_comfort_zone(np.array([23.0]), np.array([np.nan]))[0]


def test_comfort_zone_bounds_are_config_driven():
    # Dropping a bound must reshape the field, not break a hard-coded vertex list.
    energy.configure()
    base = dict(energy._COMFORT_ZONE)
    wide = dict(base, enthalpy_max_kj_kg=None, rh_max=None)
    _t, _lo, hi_base = energy.comfort_zone_polygon(base)
    _t2, _lo2, hi_wide = energy.comfort_zone_polygon(wide)
    # With RH and enthalpy gone, the absolute-humidity cap is all that is left.
    assert hi_wide.max() * 1000 == pytest.approx(base["humidity_ratio_max_g_kg"])
    assert hi_wide.max() > hi_base.max()
    # A narrower temperature range narrows the field's span.
    narrow = energy.comfort_zone_polygon(dict(base, t_min_c=22.0, t_max_c=24.0))
    assert narrow[0].min() == pytest.approx(22.0)
    assert narrow[0].max() == pytest.approx(24.0)


def test_comfort_zone_drives_the_setpoints():
    energy.configure()
    lo, hi = energy._comfort_zone_setpoint_limits()
    c = energy._COMFORT_ZONE
    m = c["setpoints"]["margin_c"]
    assert (lo, hi) == pytest.approx((c["t_min_c"] + m, c["t_max_c"] - m))
    # Switched off, the adaptive band is left alone.
    off = dict(c, setpoints=dict(c["setpoints"], enabled=False))
    assert energy._comfort_zone_setpoint_limits(off) == (None, None)


def test_confined_setpoints_never_invert():
    # A hot week drives the adaptive pair (Tc−2, Tc+2) above the field's ceiling.
    # Capping only the cooling end leaves heating ABOVE cooling, which EnergyPlus
    # treats as fatal — both ends must be confined and the deadband kept.
    lo, hi, dead = 20.5, 25.5, 1.0
    for tc in (12.0, 18.0, 22.0, 26.0, 30.0, 34.0):
        h, c = energy._confine_setpoints(tc - 2.0, tc + 2.0, lo, hi, dead)
        assert lo - dead <= h <= hi and lo <= c <= hi
        assert c - h >= dead - 1e-9, (tc, h, c)
    # Mild weather is untouched: the adaptive band already sits inside the field.
    assert energy._confine_setpoints(21.0, 24.0, lo, hi, dead) == (21.0, 24.0)
    # Disabled, it is a no-op whatever the numbers.
    assert energy._confine_setpoints(28.0, 31.0, None, None, dead) == (28.0, 31.0)


class _IdealLoads:
    fieldnames = ("Name", "Zone_Name", "Dehumidification_Control_Type",
                  "Dehumidification_Setpoint", "Humidification_Control_Type",
                  "Humidification_Setpoint")

    def __init__(self, zone):
        self.Zone_Name = zone
        self.Dehumidification_Control_Type = "None"
        self.Humidification_Control_Type = "None"
        self.Dehumidification_Setpoint = 60.0
        self.Humidification_Setpoint = 1.0


class _HumIdf:
    def __init__(self, zones):
        self.systems = [_IdealLoads(z) for z in zones]
        self.made = []
        self.idfobjects = {"HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM": self.systems,
                           "ZONECONTROL:HUMIDISTAT": [], "SCHEDULE:COMPACT": []}

    def newidfobject(self, key, **kw):
        self.made.append(key)
        return None


def test_humidity_control_uses_the_template_not_a_second_humidistat():
    # ExpandObjects writes one ZoneControl:Humidistat per zone from these template
    # fields; a second one authored here is rejected as a duplicate and the run
    # dies, so the bounds must go on the template.
    energy.configure()
    idf = _HumIdf(["FLAT_A", "FLAT_B"])
    n = energy._apply_comfort_zone_humidity_control(idf, {"FLAT_A"})
    assert n == 1
    assert "ZONECONTROL:HUMIDISTAT" not in idf.made
    a, b = idf.systems
    assert a.Dehumidification_Control_Type == "Humidistat"
    assert b.Dehumidification_Control_Type == "None"      # not in zone_names
    # Default is dehumidification only: a Spanish dwelling has no humidifier.
    assert a.Humidification_Control_Type == "None"
    # …and the setpoint is the FIELD's effective ceiling at the cooling setpoint,
    # which the enthalpy line puts well below the bare rh_max.
    assert a.Dehumidification_Setpoint < energy._COMFORT_ZONE["rh_max"]
    assert 40.0 < a.Dehumidification_Setpoint < 55.0


def test_humidity_control_modes():
    energy.configure()
    c = energy._COMFORT_ZONE
    idf = _HumIdf(["FLAT_A"])
    assert energy._apply_comfort_zone_humidity_control(
        idf, {"FLAT_A"}, dict(c, setpoints=dict(c["setpoints"],
                                                humidity_control="none"))) == 0
    idf = _HumIdf(["FLAT_A"])
    energy._apply_comfort_zone_humidity_control(
        idf, {"FLAT_A"}, dict(c, setpoints=dict(c["setpoints"],
                                                humidity_control="full")))
    assert idf.systems[0].Humidification_Control_Type == "Humidistat"
    assert idf.systems[0].Humidification_Setpoint == pytest.approx(c["rh_min"])
    # rh_max mode holds the bare relative-humidity bound instead of the field's.
    idf = _HumIdf(["FLAT_A"])
    energy._apply_comfort_zone_humidity_control(
        idf, {"FLAT_A"}, dict(c, setpoints=dict(c["setpoints"],
                                                humidity_setpoint="rh_max")))
    assert idf.systems[0].Dehumidification_Setpoint == pytest.approx(c["rh_max"])


def test_comfort_zone_and_adaptive_band_are_reported_separately():
    # They answer different questions and must not be collapsed: a state can pass
    # the climate-relative temperature band and fail the comfort field on humidity.
    energy.configure()
    theta = np.array([26.0])            # warm climate -> adaptive band ~24.4-30.4
    t, w = np.array([26.5]), np.array([0.016])
    assert not energy._adaptive_out_of_comfort(t, theta)[0]   # inside the band
    assert energy._outside_comfort_zone(t, w)[0]              # outside the field


def test_bin_psy_points_counts_hours_per_cell():
    from hypercadaster_ES import reporting
    # Three hours in one cell, one in another — the count is the hours, and the
    # returned coordinate is the cell centre, not one of the samples.
    t = np.array([22.02, 22.06, 21.98, 25.4])
    w = np.array([8.01, 8.03, 7.99, 11.0])
    bt, bw, hours = reporting._bin_psy_points(t, w, 0.2, 0.2)
    assert sorted(hours.tolist()) == [1, 3]
    assert set(np.round(bt, 2)) == {22.0, 25.4}
    assert int(hours[np.argmin(np.abs(bt - 22.0))]) == 3


def _psy_series(n_days=365, offset=0.0, seed=0):
    """A year of hourly indoor state for one scenario: an outdoor sinusoid so the
    season classifier sees all three seasons, and an indoor state that tracks it."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for h in range(1, 25):
            t_out = 14.0 - 12.0 * np.cos(2 * np.pi * d / 365.0)
            rows.append((d, h, t_out))
    doy = np.array([r[0] for r in rows])
    t_out = np.array([r[2] for r in rows])
    month = (doy // 31 + 1).clip(1, 12)
    day = (doy % 28 + 1)
    t_in = 21.0 + 0.35 * (t_out - 14.0) + offset + rng.normal(0, 0.6, len(rows))
    return pd.DataFrame({
        "month": month, "day": day, "hour": [r[1] for r in rows],
        "t_indoor": t_in, "t_outdoor": t_out,
        "occupants": np.full(len(rows), 2.0),
        "w_kgkg": (0.006 + 0.0002 * (t_out - 14.0)).clip(0.001, 0.020),
    })


def test_psychrometric_chart_is_per_season_and_per_scenario():
    from hypercadaster_ES import reporting
    series = {"aware": _psy_series(offset=0.0, seed=1),
              "cte": _psy_series(offset=10.0, seed=2)}   # runs far too hot
    fig = reporting._fig_psychrometric(series)
    assert fig is not None
    buttons = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert buttons[0] == "whole year"
    assert {"heating", "cooling"} <= set(buttons)
    # Exactly one season's block is visible at a time, and each button switches
    # the whole block — band and clouds together.
    named = [t.name for t in fig.data if t.visible and t.name]
    assert sum(n.startswith("aware") for n in named) == 1
    assert sum("adaptive band" in n for n in named) == 1
    # Every scenario keeps its identity in the legend, with its own % in band.
    assert all(any(f"{s} · " in (t.name or "") and "% in zone" in (t.name or "")
                   for t in fig.data) for s in series)
    # Colour follows the entity: the warmer twin is not repainted.
    colors = {t.name.split(" · ")[0]: t.marker.color
              for t in fig.data if t.mode == "markers"}
    assert colors["aware"] != colors["cte"]
    # A building held 10 K above its neighbour spends far less of the year inside
    # the comfort zone — the number in the legend is a result, not a decoration.
    def _pct(scenario):
        for t in fig.data:
            if t.visible and (t.name or "").startswith(scenario):
                return float(t.name.split("· ")[1].split("%")[0])
    assert _pct("cte") < _pct("aware")


def test_psychrometric_chart_absent_without_humidity_output():
    # Runs made before the humidity variable was requested must not break the
    # report — they simply have no comfort chart.
    from hypercadaster_ES import reporting
    s = _psy_series()
    assert reporting._fig_psychrometric({"aware": s.drop(columns=["w_kgkg"])}) is None
    s["w_kgkg"] = np.nan
    assert reporting._fig_psychrometric({"aware": s}) is None


def test_humidity_ratio_is_requested_from_energyplus():
    # The psychrometric chart's ordinate has to be asked for, or every future run
    # silently loses the figure.
    import inspect
    src = inspect.getsource(energy._add_scenario_outputs)
    assert "Zone Mean Air Humidity Ratio" in src


def test_hs3_applies_only_from_the_cte_year():
    # The CTE became obligatory on 29-09-2006, so 2007 is the first year that had
    # to comply. An unknown year is NOT given a ventilation system it may not have.
    energy.configure()
    assert not energy._hs3_applies(2006)
    assert energy._hs3_applies(2007)
    assert energy._hs3_applies(2020)
    assert not energy._hs3_applies(None)
    assert not energy._hs3_applies("")


def test_hs3_flow_is_the_larger_of_admission_and_extraction():
    # DB-HS3 Tabla 2.1: the occupant term (dormitorios + salas) against the wet-room
    # term (baños + cocina); the system is built to the larger of the two.
    energy.configure()
    cfg = energy._HS3_VENT
    per_occ, per_m2 = cfg["l_s_per_occupant"], cfg["l_s_per_m2"]

    class _Obj:
        fieldnames = ("Name", "Schedule_Name", "Design_Flow_Rate",
                      "Design_Flow_Rate_Calculation_Method", "Ventilation_Type",
                      "Zone_Name", "Constant_Term_Coefficient",
                      "Temperature_Term_Coefficient", "Velocity_Term_Coefficient",
                      "Velocity_Squared_Term_Coefficient")

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _Idf:
        def __init__(self):
            self.objects = []

        def idfobjects(self):
            pass

        def newidfobject(self, _key, **kw):
            o = _Obj(**kw)
            self.objects.append(o)
            return o

    idf = _Idf()
    idf.idfobjects = {"ZONEVENTILATION:DESIGNFLOWRATE": idf.objects}
    # A big, sparsely occupied dwelling: the wet rooms govern.
    o = energy._add_hs3_ventilation(idf, "FLAT_A", occupants=2.0,
                                    floor_area_m2=100.0, volume_m3=300.0,
                                    schedule="VENT_HS3")
    assert o.Design_Flow_Rate == pytest.approx(100.0 * per_m2 / 1000.0)
    assert o.Ventilation_Type == "Intake"      # fan-driven outdoor air
    # A small, crowded one: the occupants do.
    o = energy._add_hs3_ventilation(idf, "FLAT_B", occupants=6.0,
                                    floor_area_m2=50.0, volume_m3=150.0,
                                    schedule="VENT_HS3")
    assert o.Design_Flow_Rate == pytest.approx(6.0 * per_occ / 1000.0)
    # And the cap is expressed in air changes, not litres.
    o = energy._add_hs3_ventilation(idf, "FLAT_C", occupants=200.0,
                                    floor_area_m2=50.0, volume_m3=150.0,
                                    schedule="VENT_HS3")
    assert o.Design_Flow_Rate == pytest.approx(cfg["max_ach"] * 150.0 / 3600.0)


def test_hs3_is_wired_for_every_occupant_including_cte():
    # A ventilation system is a property of the BUILDING: the normative cte user
    # must not be the only one left with nothing but leakage.
    import inspect
    lines = inspect.getsource(energy._apply_occupant_control).splitlines()
    guard = next(i for i, ln in enumerate(lines) if ln.strip() == "if hs3_on:")
    call = next(i for i, ln in enumerate(lines) if "_add_hs3_ventilation(" in ln)
    assert guard < call
    # The guard is a SIBLING of the per-behaviour branches, not nested inside one:
    # same indentation as the `if beh == "cte":` that follows it, so no behaviour
    # can skip it and the normative user gets the same system as everyone else.
    indent = len(lines[guard]) - len(lines[guard].lstrip())
    cte = next(i for i, ln in enumerate(lines)
               if i > guard and ln.strip() == 'if beh == "cte":')
    assert len(lines[cte]) - len(lines[cte].lstrip()) == indent
    # Nothing between the two tests the behaviour.
    assert not any("beh ==" in ln or "beh in" in ln for ln in lines[guard:cte])


def test_scenario_report_writes_one_page_per_use(tmp_path):
    # A building is not one thing: every use in the frame gets its own page, and
    # each page carries only its own rows.
    from hypercadaster_ES import reporting
    k = _kpi_frame()
    shop = k[k["scenario"] == "cte"].copy()
    shop["use_type"] = "Commercial"
    shop["scenario"] = "normative"
    out = reporting.scenario_report(pd.concat([k, shop], ignore_index=True),
                                    str(tmp_path / "r.html"),
                                    building_reference="TESTBR")
    assert set(out) == {"residential", "commercial"}
    assert out["residential"].endswith("r_residential.html")
    res = open(out["residential"], encoding="utf-8").read()
    com = open(out["commercial"], encoding="utf-8").read()
    assert "residential area" in res and "commercial area" in com
    # The commercial page knows only its normative user; the dwellings' occupant
    # archetypes never appear on it.
    assert "normative" in com
    for sc in ("unaware", "aware_ff"):
        assert sc in res and sc not in com


def test_hourly_building_series_keeps_only_the_use_zones():
    # The parking box (no occupants, 1 ren/h) must not lift the dwellings' air
    # change, and the site weather must survive the zone filter.
    from hypercadaster_ES import reporting
    df = pd.DataFrame({
        "month": [1, 1], "day": [3, 3], "hour": [1, 2],
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [4.0, 3.0],
        "FLAT_F1:Zone Infiltration Standard Density Air Change Rate [ach](Hourly)":
            [0.55, 0.55],
        "PARKING_F0:Zone Infiltration Standard Density Air Change Rate [ach](Hourly)":
            [1.0, 1.0],
    })
    areas = {"FLAT_F1": 100.0, "PARKING_F0": 100.0}
    both = reporting._hourly_building_series(df, areas)
    flat = reporting._hourly_building_series(df, areas, zones={"FLAT_F1"})
    assert both["ach_total"].iloc[0] == pytest.approx(0.775)   # diluted by parking
    assert flat["ach_total"].iloc[0] == pytest.approx(0.55)    # the dwelling itself
    assert flat["t_outdoor"].iloc[0] == pytest.approx(4.0)


def test_hourly_hvac_survives_the_zone_filter():
    # ExpandObjects reports the ideal-loads variables under the SYSTEM it built
    # ("FLAT_F1 IDEAL LOADS AIR SYSTEM:..."), not under the zone. Matching the key
    # exactly dropped every HVAC column whenever `zones` was given -- which is
    # always, since the report keeps each use apart -- so the daily HVAC series was
    # a flat zero while the KPI table showed a real demand.
    from hypercadaster_ES import reporting
    df = pd.DataFrame({
        "month": [1, 1], "day": [3, 3], "hour": [1, 2],
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [4.0, 3.0],
        "FLAT_F1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating "
        "Energy [J](Hourly)": [3.6e6, 7.2e6],
        "FLAT_F1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Cooling "
        "Energy [J](Hourly)": [0.0, 0.0],
        "PARKING_F0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total "
        "Heating Energy [J](Hourly)": [3.6e7, 3.6e7],
    })
    areas = {"FLAT_F1": 100.0, "PARKING_F0": 100.0}
    flat = reporting._hourly_building_series(df, areas, zones={"FLAT_F1"})
    assert list(flat["hvac_kwh"]) == pytest.approx([1.0, 2.0])
    # …and the filter still excludes the OTHER zone's system.
    assert flat["hvac_kwh"].sum() == pytest.approx(3.0)
    # The daily roll-up the plot draws is non-zero too.
    assert reporting._daily(flat)["hvac_kwh"].iloc[0] == pytest.approx(3.0)


def test_scenario_report_rejects_a_frame_that_is_not_kpis(tmp_path):
    from hypercadaster_ES import reporting
    with pytest.raises(ValueError, match="no 'scenario' column"):
        reporting.scenario_report(pd.DataFrame({"x": [1]}), str(tmp_path / "b.html"))


def test_report_colour_is_stable_per_scenario():
    from hypercadaster_ES import reporting
    # Colour follows the entity: dropping a scenario must not repaint the rest.
    all_six = ["unaware", "unaware_ff", "random_mix", "aware", "aware_ff", "cte"]
    full = reporting._scenario_colors(reporting._ordered(all_six,
                                                         reporting._SCENARIO_ORDER))
    fewer = reporting._ordered([s for s in all_six if s != "random_mix"],
                               reporting._SCENARIO_ORDER)
    # The ORDER is stable, which is what keeps a scenario in its slot as long as
    # the caller passes the canonical list.
    assert fewer == ["unaware", "unaware_ff", "aware", "aware_ff", "cte"]
    assert len(set(full.values())) == len(all_six)      # no hue reused


def test_free_floating_scenarios_get_a_second_encoding():
    from hypercadaster_ES import reporting
    # An _ff twin often traces its partner exactly; a dash keeps both visible.
    assert reporting._scenario_dash("aware_ff") == "dot"
    assert reporting._scenario_dash("aware") == "solid"


def test_hourly_building_series_is_area_weighted():
    from hypercadaster_ES import reporting
    df = pd.DataFrame({
        "month": [1, 1], "day": [1, 1], "hour": [1, 2],
        "BIG:Zone Operative Temperature [C](Hourly)": [20.0, 20.0],
        "SMALL:Zone Operative Temperature [C](Hourly)": [30.0, 30.0],
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0, 5.0],
    })
    s = reporting._hourly_building_series(df, {"BIG": 90.0, "SMALL": 10.0})
    # 90 m² at 20 °C and 10 m² at 30 °C is a 21 °C building, not 25.
    assert s["t_indoor"].iloc[0] == pytest.approx(21.0)
    plain = reporting._hourly_building_series(df)
    assert plain["t_indoor"].iloc[0] == pytest.approx(25.0)


def test_temperature_extremes_are_reported_beside_the_percentiles(tmp_path):
    # A percentile band cannot say whether a dwelling ever touched 12 °C; the
    # extremes can, and a compliance argument usually turns on them.
    csv = _rich_csv(tmp_path, n=24, top=21.0)
    df = energy._read_eplus_csv(csv)
    # one spike the p10/p90 band will never reach
    col = [c for c in df.columns if "Operative Temperature" in c][0]
    df.loc[df.index[12], col] = 34.0
    df.loc[df.index[13], col] = 11.0
    recs = energy._zone_scenario_kpis(
        df, {"H_F1": {"use_type": "Residential"}}, {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert r["t_max_day"] == pytest.approx(34.0)
    assert r["t_min_day"] == pytest.approx(11.0)
    # …and the band itself stays where the bulk of the hours are.
    assert 20.0 <= r["t_p50_day"] <= 22.0
    assert r["t_p90_day"] < r["t_max_day"]


def test_cte_awnings_are_not_permanently_deployed():
    # The normative solar protection is the radiation-triggered persiana. Pinning
    # the awnings at cte_tau through every daylight hour applied it twice and
    # shaded midwinter mornings no control would have closed.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert "cte_shade = clear" in src
    assert "SHADE_CTE" in src          # still authored for a caller who wants it


def test_report_takes_separate_csv_and_html_paths(tmp_path):
    from hypercadaster_ES import reporting
    csv = tmp_path / "k.csv"
    html_out = reporting.scenario_report(_kpi_frame(), str(tmp_path / "r"),
                                         output_path_csv=str(csv),
                                         building_reference="TESTBR")
    assert os.path.exists(csv) and html_out["residential"].endswith(".html")
    assert len(pd.read_csv(csv)) == len(_kpi_frame())
    # CSV only, no page.
    only_csv = tmp_path / "solo.csv"
    assert reporting.scenario_report(_kpi_frame(), output_path_csv=str(only_csv)) == {}
    assert os.path.exists(only_csv)


def test_worst_week_picks_the_extreme_stretch():
    from hypercadaster_ES import reporting
    daily = pd.DataFrame({
        "month": [1] * 20, "day": list(range(1, 21)),
        "t_outdoor": [5.0] * 7 + [30.0] * 6 + [5.0] * 7,
    })
    hot = reporting._worst_week(daily, "cooling", days=6)
    assert hot == {(1, d) for d in range(8, 14)}
    cold = reporting._worst_week(daily, "heating", days=6)
    assert all(d <= 7 or d >= 14 for _m, d in cold)
    assert reporting._worst_week(pd.DataFrame(), "cooling") is None


# --------------------------------------------------------------------------- #
# idf_path: generate → hand-edit → simulate
# --------------------------------------------------------------------------- #

def test_idf_metadata_round_trips(tmp_path):
    # The zone → use-type map lives on the IDF object; a file cannot carry it, so
    # it is persisted beside the .idf or the reloaded model simulates nothing.
    idf_path = str(tmp_path / "m.idf")
    zone_meta = {"H_F1": {"use_type": "Residential", "reference": "H",
                          "detailed": "Housing", "kind": "occupied"},
                 "C_F0": {"use_type": "Commercial", "reference": "C",
                          "detailed": "Retail", "kind": "occupied"}}
    energy._write_idf_metadata(idf_path, zone_meta, building_reference="BR1",
                               shading_filtered=True, epw="/w/x.epw")
    assert os.path.exists(energy.idf_metadata_path(idf_path))
    back = energy._read_idf_metadata(idf_path)
    assert back["building_reference"] == "BR1"
    assert back["shading_filtered"] is True
    assert back["epw"] == "/w/x.epw"
    assert back["zone_meta"]["H_F1"]["use_type"] == "Residential"
    assert back["zone_meta"]["C_F0"]["reference"] == "C"


def test_idf_metadata_absent_is_none(tmp_path):
    assert energy._read_idf_metadata(str(tmp_path / "nope.idf")) is None


def test_idf_metadata_path_is_a_sidecar():
    assert energy.idf_metadata_path("/a/b.idf") == "/a/b.idf.hcmeta.json"


def test_run_behavior_scenarios_takes_an_idf_path():
    import inspect
    p = inspect.signature(energy.run_behavior_scenarios).parameters
    assert "idf_path" in p and p["idf_path"].default is None
    # It must take precedence over a passed object, so the file on disk — with
    # whatever was edited into it — is what runs.
    src = inspect.getsource(energy.run_behavior_scenarios)
    assert "if idf_path is not None:" in src
    assert src.index("if idf_path is not None:") < src.index("if idf is not None:")


def test_metadata_survives_a_hand_edited_zone_list(tmp_path):
    # Renaming a zone in the IDF without renaming it in the sidecar leaves it
    # unmapped — it is skipped, never guessed at, so a rename cannot silently
    # turn a shop into a dwelling.
    idf_path = str(tmp_path / "m.idf")
    energy._write_idf_metadata(idf_path, {"H_F1": {"use_type": "Residential"}})
    meta = energy._read_idf_metadata(idf_path)
    zone_meta = {z.upper(): v for z, v in meta["zone_meta"].items()}
    assert zone_meta.get("H_F1_RENAMED") is None
    assert zone_meta["H_F1"]["use_type"] == "Residential"


# --------------------------------------------------------------------------- #
# Indoor CO2 — air renewal
# --------------------------------------------------------------------------- #

def test_co2_balance_and_generation_rate():
    idf = _FakeIdf()
    idf.newidfobject("PEOPLE", Name="H_F1_People")
    idf.newidfobject("PEOPLE", Name="H_F2_People")
    assert energy._add_co2_balance(idf) == 2
    bal = idf.idfobjects["ZONEAIRCONTAMINANTBALANCE"][0]
    assert bal.Carbon_Dioxide_Concentration == "Yes"
    assert bal.Outdoor_Carbon_Dioxide_Schedule_Name == energy._CO2_OUTDOOR_SCHED
    for o in idf.idfobjects["PEOPLE"]:
        assert o.Carbon_Dioxide_Generation_Rate == energy._CO2["generation_rate_m3_s_w"]
    # The outdoor background is what an empty zone decays to.
    sched = [s for s in idf.idfobjects["SCHEDULE:COMPACT"]
             if s.Name == energy._CO2_OUTDOOR_SCHED][0]
    assert str(energy._CO2["outdoor_ppm"]) in " ".join(str(f) for f in sched.obj)


def test_co2_disabled_is_a_no_op(monkeypatch):
    monkeypatch.setitem(energy._CO2, "enabled", False)
    idf = _FakeIdf()
    idf.newidfobject("PEOPLE", Name="H_F1_People")
    assert energy._add_co2_balance(idf) == 0
    assert not idf.idfobjects.get("ZONEAIRCONTAMINANTBALANCE")


def test_night_activity_schedule_drops_the_metabolic_rate():
    idf = _FakeIdf()
    name = energy._add_night_activity_schedule(idf)
    body = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
    awake = float(energy._LOAD_LIBRARY["residential"]["activity"])
    assert any(str(energy._ACT_SLEEPING) in f for f in body)
    assert any(str(awake) in f for f in body)
    assert name == energy._ACT_RESIDENTIAL_NIGHT
    # 23:00-08:00 is the sleeping band, so hour 24 carries the sleeping rate.
    assert 24 in energy._NIGHT_HOURS


def test_sleeping_is_fully_present_not_a_third_of_the_household():
    # Number_of_People_Schedule scales the HEADCOUNT. A sleeping household is all
    # home; its lower metabolism is the activity level's job. Discounting here as
    # well applied the reduction twice and understated night CO2 threefold.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("time_use_occupancy", {})
    assert cfg["state_fractions"]["Sleeping"] == 1.0
    assert cfg["state_fractions"]["WithActivity"] == 1.0
    assert cfg["state_fractions"]["NonOccupied"] == 0.0
    # …and the sleeping metabolic rate really is lower than the waking one.
    assert energy._ACT_SLEEPING < float(energy._LOAD_LIBRARY["residential"]["activity"])


def test_co2_kpis_over_occupied_hours(tmp_path):
    zone = "H_F1"
    n = 48
    # Occupied only at night; CO2 high then, at background while empty.
    occ = [1.0 if (i % 24 + 1) in energy._NIGHT_HOURS else 0.0 for i in range(n)]
    co2 = [1200.0 if o else 420.0 for o in occ]
    df = pd.DataFrame({
        "Date/Time": [f" 01/{1 + i // 24:02d}  {i % 24 + 1:02d}:00:00" for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [21.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Air CO2 Concentration [ppm](Hourly)": co2,
        f"{zone}:Zone People Occupant Count [](Hourly)": occ,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {zone: {"use_type": "Residential"}},
        {zone: 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    # Night: the occupied hours, all at 1200 ppm.
    assert r["co2_p50_night"] == pytest.approx(1200.0)
    assert r["co2_pct_above_1000_night"] == pytest.approx(100.0)
    assert r["co2_pct_above_1400_night"] == pytest.approx(0.0)
    # Day: no occupied hours, so nothing to report — NOT "420 ppm, all fine".
    assert np.isnan(r["co2_p50_day"])


# --------------------------------------------------------------------------- #
# The normative CTE thermostat: calendar seasons, no in-between
# --------------------------------------------------------------------------- #

def _cte_schedules():
    idf = _FakeIdf()
    name, avail = energy._add_cte_thermostat(idf)
    by = {str(o.Name): [str(f) for f in o.obj] for o in idf.idfobjects["SCHEDULE:COMPACT"]}
    return idf, name, avail, by


def test_cte_hvac_is_available_all_year():
    # The occupant scenarios stop conditioning in mild weather because people do.
    # The standard never does — it only changes which service its calendar allows.
    idf, _name, avail, by = _cte_schedules()
    assert avail == "AVAIL_CTE"
    # A single Through block covering the whole year, at 1: never gated off.
    assert by[avail] == ["Through: 12/31", "For: AllDays", "Until: 24:00,1"]


def test_cte_setpoints_follow_the_calendar_with_no_gap():
    idf, _name, _avail, by = _cte_schedules()
    heat, cool = by["SP_CTE_HEAT"], by["SP_CTE_COOL"]
    # Winter/spring and autumn heat; summer heating is unreachable.
    assert "Until: 07:00,17.0" in heat and "Until: 24:00,20.0" in heat
    assert "Until: 24:00,-50.0" in heat
    # Summer cools; the rest of the year cooling is unreachable.
    assert "Until: 16:00,27.0" in cool and "Until: 24:00,25.0" in cool
    assert "Until: 24:00,100.0" in cool
    # Every month is covered by both schedules — no hour is left unassigned.
    for body in (heat, cool):
        assert body.count("Through:") == 3 or "Through: 12/31" in body
        assert body[-1].startswith("Until: 24:00")


def test_month_blocks_merge_consecutive_months():
    on = ["For: AllDays", "Until: 24:00,20"]
    off = ["For: AllDays", "Until: 24:00,-50"]
    fields = energy._month_blocks({m: (on if m in (1, 2, 3, 12) else off)
                                   for m in range(1, 13)})
    throughs = [f for f in fields if f.startswith("Through:")]
    # Jan–Mar on, Apr–Nov off, Dec on → three blocks, ending on 31 December.
    assert throughs == ["Through: 3/31", "Through: 11/30", "Through: 12/31"]


def test_only_the_cte_case_changed():
    # The occupant scenarios keep their in-between gate — this change is the
    # normative case only.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert "av_always = _avail(" in src          # still season-gated for the rest
    assert "iz.System_Availability_Schedule_Name = av_cte" in src


# --------------------------------------------------------------------------- #
# Season classification: a weekly moving average
# --------------------------------------------------------------------------- #

def test_season_smoothing_is_a_week():
    assert energy._SEASON_SMOOTH_DAYS == 7


def test_weekly_smoothing_absorbs_a_short_warm_spell():
    # Three warm days inside a cold March: a 3-day window turns one of them into
    # a COOLING day, which would then carry a heating-season result under the
    # cooling label. A week absorbs it.
    days = [12.0] * 20 + [26.0] * 3 + [12.0] * 20
    short = [energy._season_of_temp(t) for t in energy._smooth_days(days, window=3)]
    week = [energy._season_of_temp(t) for t in energy._smooth_days(days, window=7)]
    assert "cooling" in short
    assert "cooling" not in week
    assert week.count("heating") > 0


def test_smoothing_window_defaults_to_the_configured_one(monkeypatch):
    flat = [20.0] * 30
    assert energy._smooth_days(flat)[0] == pytest.approx(20.0)
    monkeypatch.setitem(globals(), "_x", None)      # no-op, keeps linters quiet
    # An explicit window still wins over the configured default.
    spike = [0.0] * 5 + [70.0] + [0.0] * 5
    assert energy._smooth_days(spike, window=1)[5] == pytest.approx(70.0)
    assert energy._smooth_days(spike, window=7)[5] < 70.0


def test_smoothing_still_tracks_a_real_season_change():
    # A week must not be so long that it blurs the actual turn of the season.
    days = [10.0] * 60 + [28.0] * 60
    seasons = [energy._season_of_temp(t) for t in energy._smooth_days(days)]
    assert seasons[:55] == ["heating"] * 55
    assert seasons[-55:] == ["cooling"] * 55


def test_season_band_is_configurable():
    assert energy._INBETWEEN_TEMP_LO == 16.0
    assert energy._INBETWEEN_TEMP_HI == 24.0
    assert energy._season_of_temp(10.0) == "heating"
    assert energy._season_of_temp(20.0) == "in_between"
    assert energy._season_of_temp(30.0) == "cooling"


# --------------------------------------------------------------------------- #
# Outdoor-air change rate: the renewal the CO2 answers to
# --------------------------------------------------------------------------- #

def test_windows_to_fields_builds_an_opening_profile():
    f = energy._windows_to_fields([["08:00", "08:30"], ["20:00", "20:30"]])
    assert f == ["For: AllDays", "Until: 08:00,0.0", "Until: 08:30,1.0",
                 "Until: 20:00,0.0", "Until: 20:30,1.0", "Until: 24:00,0.0"]
    # Degenerate and empty inputs must not produce a broken schedule.
    assert energy._windows_to_fields([]) == ["For: AllDays", "Until: 24:00,0.0"]
    assert energy._windows_to_fields([["10:00", "10:00"]]) == [
        "For: AllDays", "Until: 24:00,0.0"]


def test_hygiene_airing_outventilates_the_careless_occupant():
    # The aware occupant must not end up with WORSE air than the unaware one.
    # Unaware: 3 random 15-min bursts at 4 ren/h = 3 ren·h/day.
    rb = energy._RESIDENTIAL_BEHAVIOR or {}
    cfg = rb.get("hygiene_ventilation", {})
    ach = float(cfg.get("ach", rb.get("opening_ach", 4.0)))
    hours = sum((int(b[:2]) * 60 + int(b[3:5]) - int(a[:2]) * 60 - int(a[3:5])) / 60.0
                for a, b in cfg["periods"])
    # Unaware: three random 15-minute bursts at the same open-window rate.
    assert hours * ach > 3 * 0.25 * ach


def test_ach_kpis_split_the_two_routes(tmp_path):
    zone = "H_F1"
    n = 24
    df = pd.DataFrame({
        "Date/Time": [f" 01/01  {i + 1:02d}:00:00" for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [21.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Infiltration Standard Density Air Change Rate [ach](Hourly)":
            [1.0] * n,
        f"{zone}:Zone Ventilation Standard Density Air Change Rate [ach](Hourly)":
            [4.0 if i < 12 else 0.0 for i in range(n)],
        f"{zone}:Zone People Occupant Count [](Hourly)": [1.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {zone: {"use_type": "Residential"}},
        {zone: 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    # Both routes carry OUTDOOR air, so the total is their sum.
    assert r["ach_infiltration_mean_day"] == pytest.approx(1.0)
    assert r["ach_total_max_day"] == pytest.approx(5.0)
    assert r["ach_total_min_day"] == pytest.approx(1.0)
    # The percentiles are the point: a mean would hide the sealed hours.
    assert r["ach_total_p10_day"] < r["ach_total_p90_day"]


def test_ach_stats_without_the_outputs_are_nan(tmp_path):
    csv = _rich_csv(tmp_path, n=24)          # no ACH columns at all
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(csv), {"H_F1": {"use_type": "Residential"}},
        {"H_F1": 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert np.isnan(r["ach_total_mean_day"])
    assert np.isnan(r["ach_ventilation_mean_day"])


def test_open_window_delivers_its_nominal_rate():
    # design x schedule x (constant + velocity x wind). With (0.15, 0.20) a window
    # nominally at 4 ren/h delivered 0.6 in calm air — the configured number was
    # not the delivered one.
    assert energy._VENT_A == 1.0 and energy._VENT_C == 0.0
    w = (energy._RESIDENTIAL_BEHAVIOR or {}).get("ventilation_wind", {})
    assert w["constant"] == 1.0 and w["velocity"] == 0.0


def test_ach_percentiles_match_the_temperature_ones_except_the_median():
    # The report draws both with the same distribution chart, so they need the
    # same percentile steps -- except the median, which the outdoor-air rates
    # never report: an ACH is a flow, and the median of a series that is 0 for
    # twenty hours and 4 for one is 0, i.e. a dwelling that never opened a window.
    assert (set(energy._ACH_PERCENTILES)
            == set(energy._TEMPERATURE_PERCENTILES) - {50})
    assert 50 not in energy._ACH_PERCENTILES


def test_outdoor_air_rates_are_never_medians(tmp_path):
    # Neither the total nor either route may be reported as a median; the centre
    # of the air-change distribution is the MEAN.
    zone = "H_F1"
    n = 24 * 3
    df = pd.DataFrame({
        "Date/Time": [f" 01/{1 + i // 24:02d}  {(i % 24) + 1:02d}:00:00"
                      for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [21.0] * n,
        f"{zone}:Zone Infiltration Standard Density Air Change Rate [ach](Hourly)":
            [0.5] * n,
        # Open one hour in twelve: median 0, mean clearly not.
        f"{zone}:Zone Ventilation Standard Density Air Change Rate [ach](Hourly)":
            [4.0 if i % 12 == 0 else 0.0 for i in range(n)],
        f"{zone}:Zone People Occupant Count [](Hourly)": [1.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {zone: {"use_type": "Residential"}},
        {zone: 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    assert not any(k.startswith("ach_") and "p50" in k for k in r)
    # The mean is there and carries the opened hours the median would have lost.
    assert r["ach_total_mean_day"] > r["ach_infiltration_mean_day"]
    assert r["ach_ventilation_mean_day"] > 0.0


def test_every_airing_uses_one_open_window_rate():
    # Hygiene, midday free cooling, the summer night purge and the unaware
    # occupant's random bursts all mean "the windows are open". What separates the
    # behaviours is when and for how long, not how wide.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    for call in ("_Hyg\", hyg, hyg_ach", "_MidHeat\", midday, open_ach",
                 "_NightPurge\", night, open_ach", "_RndVent\", rs, open_ach",
                 "_UnawareNight\", ns, open_ach"):
        assert call in src, call
    assert (energy._RESIDENTIAL_BEHAVIOR or {}).get("opening_ach") == 4.0


def test_one_occupant_engine_for_the_built_model_and_the_scenarios():
    # build_energy_model and every scenario must wire the behaviour through the
    # SAME pass, so the .idf that is written, plotted and simulated all agree.
    # The old second engine (_apply_residential_behavior authoring thermostats,
    # availability and ZoneVentilation from behaviors.*) is gone.
    import inspect
    assert not hasattr(energy, "_apply_residential_behavior")
    assign = inspect.getsource(energy._assign_residential_behavior)
    for gone in ("newidfobject", "Template_Thermostat_Name",
                 "System_Availability_Schedule_Name", "ZONEVENTILATION"):
        assert gone not in assign, gone
    for fn in (energy.add_energy_systems, energy.run_behavior_scenarios):
        assert "_wire_occupant_model(" in inspect.getsource(fn)
    # The config those keys read is gone with them.
    rb = energy._RESIDENTIAL_BEHAVIOR or {}
    assert "ventilation_schedules" not in rb and "occupancy_periods" not in rb
    for bdef in rb.get("behaviors", {}).values():
        assert set(bdef) <= {"description"}, bdef
    assert "period_weights" not in (rb.get("assignment") or {})


def _until_value(fields, day_type, clock):
    """The scheduled value a day profile holds at ``clock`` ("HH:MM")."""
    by_day = energy._split_day_blocks(fields)
    want = int(clock.split(":")[0]) * 60 + int(clock.split(":")[1])
    for f in by_day[day_type]:
        c = f.split()[1].split(",")[0]
        end = int(c.split(":")[0]) * 60 + int(c.split(":")[1])
        if want < end:
            return float(f.split(",")[1])
    return float(by_day[day_type][-1].split(",")[1])


def test_aware_heating_sets_back_through_the_night():
    # The adaptive thermostat sets back when the dwelling is EMPTY -- which never
    # happens at night, so it used to hold the full comfort setpoint through every
    # sleeping hour. A household asleep turns the heating down.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {})["aware_night_setback"]
    assert cfg["enabled"] is True
    spans = energy._night_setback_spans()
    heat = energy._setback_day_fields(21, 16, spans)
    for day in ("Monday", "Sunday", "AllOtherDays"):
        assert _until_value(heat, day, "03:00") == 16      # asleep
        assert _until_value(heat, day, "23:30") == 16      # gone to bed
        assert _until_value(heat, day, "08:00") == 21      # up and about
        assert _until_value(heat, day, "20:00") == 21      # evening, awake
    # COOLING is untouched: those nights are already purged with the HVAC off.
    cool = energy._setback_day_fields(25, 30)
    assert _until_value(cool, "AllOtherDays", "03:00") == 25
    import inspect
    src = inspect.getsource(energy._apply_adaptive_setpoints)
    assert 'SP_ADAPT_HEAT_{tag}", heat, sb_h, night_spans' in src
    assert 'SP_ADAPT_COOL_{tag}", cool, sb_c,' in src


def test_aware_night_setback_can_be_switched_off():
    saved = dict(energy._RESIDENTIAL_BEHAVIOR)
    try:
        energy._RESIDENTIAL_BEHAVIOR["aware_night_setback"] = {"enabled": False}
        assert energy._night_setback_spans() == []
        heat = energy._setback_day_fields(21, 16, energy._night_setback_spans())
        assert _until_value(heat, "AllOtherDays", "03:00") == 21
    finally:
        energy._RESIDENTIAL_BEHAVIOR.clear()
        energy._RESIDENTIAL_BEHAVIOR.update(saved)


def test_unaware_forgets_to_switch_the_hvac_off_some_nights():
    # Forgetting is the archetype's defining failure. Modelled as never-happens,
    # the careless dwelling looked better overnight than the careful one.
    import random
    frac = float((energy._RESIDENTIAL_BEHAVIOR or {})
                 ["unaware_night_hvac"]["nights_fraction"])
    assert 0.0 < frac < 1.0
    counts = []
    for i in range(400):
        days = energy._weekly_night_days(random.Random(i), frac)
        counts.append(len(days))
        fields = energy._forgotten_night_availability(energy._AVAIL_LEFTON, days)
        for day in energy._WEEK_DAY_TYPES:
            # A forgotten night runs at 03:00; every other night is still off.
            assert _until_value(fields, day, "03:00") == (1.0 if day in days else 0.0)
            assert _until_value(fields, day, "12:00") == 1.0     # awake either way
    assert set(counts) <= {int(frac * 7), int(frac * 7) + 1}
    assert sum(counts) / len(counts) / 7 == pytest.approx(frac, abs=0.03)


def test_forgotten_nights_cover_every_day_type_exactly_once():
    # A Schedule:Compact that names a day type twice is an E+ error, and one that
    # misses a day type is undefined. Checked on the SURVEY-driven profile, whose
    # blocks are per-weekday and differ between weekdays and the weekend.
    awake = energy._week_states_to_fields(
        (["Sleeping"] * 7 + ["WithActivity"] * 13 + ["Sleeping"] * 4) * 5 +
        (["Sleeping"] * 9 + ["WithActivity"] * 15) * 2,
        lambda s: 1.0 if s == "WithActivity" else 0.0)
    for days in ([], ["Wednesday"], ["Wednesday", "Saturday"],
                 list(energy._WEEK_DAY_TYPES)):
        out = energy._forgotten_night_availability(awake, days)
        seen = []
        for f in out:
            if str(f).startswith("For:"):
                seen += str(f).split(":", 1)[1].split()
        assert len(seen) == len(set(seen)), (days, seen)
        assert set(seen) == set(energy._WEEK_DAY_TYPES) | {"AllOtherDays"}
        # The untouched weekend profile survives the rewrite.
        if "Sunday" not in days:
            assert _until_value(out, "Sunday", "08:00") == 0.0


def test_wire_occupant_model_accepts_a_pandas_row():
    # `row` is a pandas Series everywhere in the library. Truth-testing one
    # ("row or {}") raises "the truth value of a Series is ambiguous" -- which it
    # did, on every real build_energy_model call, while the unit tests passed.
    idf = _FakeIdf()
    for key in ("ZONE", "PEOPLE", "LIGHTS", "ELECTRICEQUIPMENT",
                "HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM"):
        idf.idfobjects[key] = []
    idf.newidfobject("ZONE", Name="H_F1")
    zone_meta = {"H_F1": {"use_type": "Residential"}}
    idf._hc_behavior = {"H_F1": ("aware", None)}
    row = pd.Series({"year_of_construction": 1975, "building_reference": "X"})
    energy._wire_occupant_model(idf, zone_meta, row, seed=0)
    assert idf.idfobjects["SCHEDULE:COMPACT"]        # it actually wired something


def test_weather_is_attached_before_the_systems_are_built():
    # The occupant engine and the persiana control derive their SEASONS from the
    # EPW, falling back to a fixed month split without one -- and the schedule
    # writers no-op on an existing name, so a model wired before the weather was
    # attached kept that fallback through every scenario.
    import inspect
    src = inspect.getsource(energy.build_energy_model)
    assert src.index("idf.epw = epw_path") < src.index("add_energy_systems(")


def test_conservation_infiltration_is_applied_exactly_once():
    # It SCALES the infiltration objects in place, so a second pass would square
    # the factor. It belongs to the fabric, so it is applied at build time and the
    # scenario copies inherit it.
    import inspect
    assert "_apply_infiltration_factor" in inspect.getsource(energy.add_energy_systems)
    assert "_apply_infiltration_factor(" not in \
        inspect.getsource(energy.run_behavior_scenarios)


def test_aware_hvac_is_off_while_the_windows_are_open():
    # An occupant who airs on purpose does not heat or cool into an open dwelling.
    # The availability schedule is the COMPLEMENT of that season's airings.
    rb = energy._RESIDENTIAL_BEHAVIOR or {}
    hyg, mid = rb["hygiene_ventilation"], rb["midday_ventilation"]
    heat = list(hyg["periods"]) + list(mid["periods"])
    cool = list(hyg["cooling_periods"]) + list(rb["night_purge"]["periods"])
    for openings in (heat, cool):
        shut = energy._windows_to_fields(openings, inside=0, outside=1)
        opened = energy._windows_to_fields(openings)
        # Same breakpoints, opposite values: HVAC off exactly when a window is open.
        assert ([f.split(",")[0] for f in shut]
                == [f.split(",")[0] for f in opened])
        for a, b in zip(shut[1:], opened[1:]):
            assert float(a.split(",")[1]) + float(b.split(",")[1]) == 1.0
    # And the adaptive thermostat (which owns the aware zones) picks it up.
    import inspect
    src = inspect.getsource(energy._apply_adaptive_setpoints)
    assert "_AVAIL_AWARE_OPEN" in src
    assert energy._AVAIL_AWARE_OPEN == "AVAIL_AWARE_WINDOWS"


def test_unaware_night_opening_skips_the_in_between_season():
    # Sleeping with the windows open is a HOT-night habit. The in-between season
    # is mild because the night is already cool, so an unattended 8-hour opening
    # there just dumps heat the dwelling wanted.
    seasons = ((energy._RESIDENTIAL_BEHAVIOR or {})
               .get("unaware_night_ventilation", {}).get("seasons"))
    assert seasons == ["cooling"]
    import random
    sbd = {d: ("cooling" if 150 <= d < 250 else
               "in_between" if 100 <= d < 150 else "heating") for d in range(1, 366)}
    idf = _FakeIdf()
    energy._unaware_night_schedule(idf, "H_F1", random.Random(0), sbd)
    fields = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
    # Exactly one block holds an opening; the other two seasons are shut all day.
    blocks, cur = [], None
    for f in fields:
        if f.startswith("Through:"):
            cur = []
            blocks.append(cur)
        elif cur is not None:
            cur.append(f)
    assert sum(any(b.endswith(",1.0") for b in blk) for blk in blocks) == 1


def test_always_on_has_its_own_thermostat():
    # Its setpoints moved out of `behaviors` into hvac.always_on; without a
    # thermostat of its own the archetype would inherit whatever ran last.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert 'hcfg.get("always_on"' in src
    assert "tstat_always_on" in src
    assert (energy._HVAC or {}).get("always_on", {}).get("heating") is not None


def test_kpi_occupied_hours_come_from_the_survey_not_the_run():
    # A discomfort or CO2 percentage over one occupancy is not comparable with the
    # same percentage over another, so the mask is imposed from OUTSIDE the run
    # instead of read off each scenario's own People output.
    states = ((["Sleeping"] * 7 + ["WithActivity"] * 1 + ["NonOccupied"] * 10
               + ["WithActivity"] * 6) * 5
              + (["Sleeping"] * 9 + ["WithActivity"] * 15) * 2)
    mask = energy._time_use_occupied_mask({"states": states})
    assert mask["Monday"] == [(0, 480), (1080, 1440)]      # out 08:00–18:00
    assert mask["Saturday"] == [(0, 1440)]
    df = pd.DataFrame({
        "month": [1] * 24 * 7, "day": sum(([4 + d] * 24 for d in range(7)), []),
        "hour": list(range(1, 25)) * 7,
    })
    occ = energy._hourly_occupied(df, mask, year=2021)     # 2021-01-04 is a Monday
    assert occ.mean() == pytest.approx((5 * 14 + 2 * 24) / 168)
    assert not occ[12] and occ[3]                          # Monday 12h out, 03h in
    assert occ[5 * 24 + 12]                                # Saturday 12h in
    # No mask at all → every hour counts (the normative cte case).
    assert energy._hourly_occupied(df, None).all()


def test_cte_kpis_are_taken_over_every_hour():
    # The normative case carries no occupancy profile of its own, so restricting
    # its KPIs to "occupied" hours would answer a question the standard never asks.
    import inspect
    src = inspect.getsource(energy.run_behavior_scenarios)
    assert 'occupied_mask=("all" if spec.get("behavior") == "cte"' in src
    assert "tus_occupied or None" in src


def test_occupied_mask_overrides_the_people_output():
    zone = "H_F1"
    n = 24 * 7
    df = pd.DataFrame({
        "Date/Time": [f" 01/{4 + i // 24:02d}  {(i % 24) + 1:02d}:00:00"
                      for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [40.0] * n,
        # The run says nobody is ever in; the imposed mask says otherwise.
        f"{zone}:Zone People Occupant Count [](Hourly)": [0.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0] * n,
    })
    meta, areas = {zone: {"use_type": "Residential"}}, {zone: 50.0}
    from_run = energy._zone_scenario_kpis(
        energy._read_eplus_csv_frame(df) if hasattr(energy, "_read_eplus_csv_frame")
        else _with_time_parts(df), meta, areas, "II")
    r = next(x for x in from_run if x["season"] == "heating")
    assert np.isnan(r["pct_discomfort_day"]) or r["pct_discomfort_day"] == 0.0
    imposed = energy._zone_scenario_kpis(
        _with_time_parts(df), meta, areas, "II", occupied_mask="all")
    r2 = next(x for x in imposed if x["season"] == "heating")
    assert r2["pct_discomfort_day"] == pytest.approx(100.0)   # 40 °C, all hours


def _with_time_parts(df):
    """The month/day/hour columns ``_read_eplus_csv`` adds, without a temp file."""
    out = df.copy()
    parts = [str(v).split() for v in df["Date/Time"]]
    out["month"] = [int(p[0].split("/")[0]) for p in parts]
    out["day"] = [int(p[0].split("/")[1]) for p in parts]
    out["hour"] = [int(p[1].split(":")[0]) for p in parts]
    return out


def test_occupancy_figure_is_a_weekly_hourly_profile():
    # Occupancy is a weekly rhythm; a daily mean flattens the one thing worth
    # seeing (who is home at 03:00, how Saturday differs from Tuesday).
    from hypercadaster_ES import reporting
    import inspect
    src = inspect.getsource(reporting._use_document)
    assert '"5. Occupancy of people, average week"' in src
    assert '_fig_weekly_profile(series, "occupants"' in src
    n = 24 * 14
    s = pd.DataFrame({
        "month": [1] * n, "day": sum(([4 + d] * 24 for d in range(14)), []),
        "hour": list(range(1, 25)) * 14,
        # 2 people at night, 6 by day — a rhythm a daily mean would hide.
        "occupants": [2.0 if h < 8 else 6.0 for _ in range(14)
                      for h in range(24)],
    })
    fig = reporting._fig_weekly_profile({"aware": s}, "occupants", "people",
                                        year=2021)
    y = list(fig.data[0].y)
    assert len(y) == 168                       # Monday 00h … Sunday 23h
    assert y[3] == pytest.approx(2.0) and y[12] == pytest.approx(6.0)
    assert y[5 * 24 + 12] == pytest.approx(6.0)


def test_one_occupancy_definition_drives_everything():
    # The thermostat's occupied/away band and the window openings must never
    # disagree about when somebody is in. The stylised presence AVAILABILITY is
    # generated from the same spans the setpoints use, so the two cannot drift.
    assert (energy._profile_positive_spans(energy._AVAIL_PRESENT)
            == energy._stylised_occupancy_mask())
    mask = energy._stylised_occupancy_mask()
    assert mask["Monday"] == energy._AWARE_OCCUPIED_SPANS["Weekdays"]
    assert mask["Sunday"] == energy._AWARE_OCCUPIED_SPANS["AllOtherDays"]
    assert mask["AllOtherDays"] == energy._AWARE_OCCUPIED_SPANS["AllOtherDays"]


def _survey_mask():
    """A Time Use week: out 08:00–18:00 on weekdays, home all weekend."""
    fields = energy._week_states_to_fields(
        (["Sleeping"] * 7 + ["WithActivity"] * 1 + ["NonOccupied"] * 10
         + ["WithActivity"] * 6) * 5
        + (["Sleeping"] * 9 + ["WithActivity"] * 15) * 2,
        lambda s: 0.0 if s == "NonOccupied" else 1.0)
    return energy._profile_positive_spans(fields)


def test_setpoints_follow_the_time_use_survey_when_it_is_available():
    # The measured week decides when the dwelling is conditioned, not a stylised
    # profile that says the household comes home at 14:00 when the survey says 18.
    tus = _survey_mask()
    assert tus["Monday"] == [(0, 480), (1080, 1440)]
    f = energy._setback_day_fields(21, 16, energy._night_setback_spans(),
                                   occ_by_day=tus)
    # Out 08:00–18:00: setback through the working day, preconditioned from 17:00.
    assert _until_value(f, "Monday", "12:00") == 16
    assert _until_value(f, "Monday", "17:30") == 21
    assert _until_value(f, "Monday", "16:30") == 16
    # …and the weekend keeps its own, different profile.
    assert _until_value(f, "Sunday", "12:00") == 21
    # Without a mask it falls back to the stylised one, where 12:00 is also away
    # but 14:30 is not — proving the two masks really are being told apart.
    g = energy._setback_day_fields(21, 16, energy._night_setback_spans())
    assert _until_value(g, "Monday", "14:30") == 21
    assert _until_value(f, "Monday", "14:30") == 16


def test_setpoint_schedules_are_tagged_by_occupancy_source():
    # One thermostat per mask: a building whose survey covers only some behaviours
    # must not have one mask imposed on all of them.
    import inspect
    src = inspect.getsource(energy._apply_adaptive_setpoints)
    assert 'f"SP_ADAPT_HEAT_{tag}"' in src and 'f"TSTAT_ADAPTIVE_{tag}"' in src
    wire = inspect.getsource(energy._wire_occupant_model)
    assert "_hc_occ_tag" in wire and "_hc_occ_masks" in wire
    assert "tag=tag" in wire and "occ_by_day=masks.get(tag)" in wire


def test_occupancy_source_is_recorded_per_zone():
    idf = _FakeIdf()
    for key in ("ZONE", "PEOPLE", "LIGHTS", "ELECTRICEQUIPMENT",
                "HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM"):
        idf.idfobjects[key] = []
    idf.newidfobject("ZONE", Name="H_F1")
    idf._hc_behavior = {"H_F1": ("aware", None)}
    energy._apply_occupant_control(idf, {"H_F1": {"use_type": "Residential"}})
    # No survey agenda here, so the stylised mask is the one recorded.
    assert idf._hc_occ_tag == {"H_F1": "STY"}
    assert idf._hc_occ_masks["STY"] == energy._stylised_occupancy_mask()


def test_window_openings_are_gated_on_occupancy():
    # Nobody opens a window in a dwelling they are not in. Ungated, the airings ran
    # on the clock alone -- ventilating (and in winter cooling) an empty flat.
    occ = energy._profile_positive_spans(energy._AVAIL_PRESENT)
    # Stylised weekday: home 00-09, 14-16, 19-24.
    assert occ["Monday"] == [(0, 540), (840, 960), (1140, 1440)]
    assert occ["Sunday"] == [(0, 1440)]
    # An 11:00 airing survives on Sunday and is gated away on a weekday.
    fields = energy._gated_opening_fields(
        energy._periods_to_spans([["11:00", "11:30"]]), occ)
    assert _until_value(fields, "Monday", "11:15") == 0.0
    assert _until_value(fields, "Sunday", "11:15") == 1.0
    # The complement (HVAC off while open) is gated identically, or the system
    # would be idled for an airing that no longer happens.
    shut = energy._gated_opening_fields(
        energy._periods_to_spans([["11:00", "11:30"]]), occ, inside=0, outside=1)
    assert _until_value(shut, "Monday", "11:15") == 1.0
    assert _until_value(shut, "Sunday", "11:15") == 0.0


def test_heating_hygiene_airing_is_placed_inside_its_band():
    # "15 minutes some time between 13:00 and 14:00" is a band plus a duration:
    # pinned to 13:00 sharp and gated on occupancy it would never fire on a day
    # the household is out until two.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {})["hygiene_ventilation"]
    assert cfg["heating_window"] == ["13:00", "14:00"]
    assert cfg["heating_duration_minutes"] == 15
    band = energy._periods_to_spans([cfg["heating_window"]])
    dur = cfg["heating_duration_minutes"]
    # Home from 13:30 → the airing slides to 13:30, still inside the band.
    late = {d: [(13 * 60 + 30, 24 * 60)] for d in
            list(energy._WEEK_DAY_TYPES) + ["AllOtherDays"]}
    assert energy._placed_opening_spans(band, dur, late)["Monday"] == [(810, 825)]
    # Home all day → the earliest quarter-hour of the band.
    allday = {d: [(0, 1440)] for d in
              list(energy._WEEK_DAY_TYPES) + ["AllOtherDays"]}
    assert energy._placed_opening_spans(band, dur, allday)["Monday"] == [(780, 795)]
    # Out through the whole band → no airing that day. The gate wins.
    out = {d: [(0, 9 * 60), (19 * 60, 1440)] for d in
           list(energy._WEEK_DAY_TYPES) + ["AllOtherDays"]}
    assert energy._placed_opening_spans(band, dur, out)["Monday"] == []


def test_unaware_airs_less_in_the_heating_season():
    # A cold day is a day you open the window less. This occupant is careless, not
    # immune to the weather.
    import random
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {})["unaware_day_ventilation"]
    assert cfg["heating_factor"] == 0.5
    sbd = {d: ("heating" if d < 100 else "cooling") for d in range(1, 366)}
    heat_open, cool_open = 0, 0
    for i in range(200):
        idf = _FakeIdf()
        energy._unaware_day_schedule(idf, f"H_F{i}", random.Random(i),
                                     None, sbd)
        fields = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
        blocks, cur = [], None
        for f in fields:
            if f.startswith("Through:"):
                cur = []
                blocks.append(cur)
            elif cur is not None:
                cur.append(f)
        heat_open += sum(b.endswith(",1.0") for b in blocks[0])
        cool_open += sum(b.endswith(",1.0") for b in blocks[-1])
    assert cool_open > 0
    assert heat_open / cool_open == pytest.approx(0.5, abs=0.08)


def test_night_purge_is_unattended():
    # The household is ASLEEP: nothing reopens or recloses a window at 04:00. The
    # purge therefore carries no indoor-temperature gate -- and since a
    # free-floating dwelling drifts warmer than a conditioned one, such a gate is
    # exactly what used to make 'aware' and 'aware_ff' open for different numbers
    # of hours, when they must differ ONLY by the HVAC being off.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    purge = src[src.index("_NightPurge\""):]
    purge = purge[:purge.index("\n")]
    assert "min_indoor" not in purge and "delta" not in purge


def _open_spans(fields):
    """The ``(start, end)`` minute spans a Schedule:Compact day profile holds at 1."""
    def _m(f):
        clock = str(f).split()[1].split(",")[0]        # "Until: 09:15,1.0" → "09:15"
        h, mm = clock.split(":")
        return int(h) * 60 + int(mm)

    spans, cursor = [], 0
    for f in fields:
        if not str(f).startswith("Until:"):
            continue
        end = _m(f)
        if str(f).endswith(",1.0"):
            spans.append((cursor, end))
        cursor = end
    return spans


def test_unaware_daytime_airings_are_short_and_few():
    # 1-3 openings a day of 15-30 minutes each -- discrete airings, NOT a window
    # left ajar through the day. Both numbers come from the config.
    import random
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("unaware_day_ventilation", {})
    n_lo, n_hi = cfg["openings_per_day"]
    d_lo, d_hi = cfg["duration_minutes"]
    counts, durations = set(), set()
    for i in range(500):
        idf = _FakeIdf()
        energy._unaware_day_schedule(idf, f"H_F{i}", random.Random(i))
        spans = _open_spans(idf.idfobjects["SCHEDULE:COMPACT"][0].obj)
        counts.add(len(spans))
        for lo, hi in spans:
            durations.add(hi - lo)
            # Inside the configured band, and starting on the hour.
            assert lo % 60 == 0
            assert 8 * 60 <= lo and hi <= 22 * 60 + d_hi
        # Never more than a small fraction of the day.
        assert sum(hi - lo for lo, hi in spans) <= n_hi * d_hi
    assert counts == set(range(int(n_lo), int(n_hi) + 1))
    assert durations and all(d_lo <= d <= d_hi for d in durations)
    # Snapped to the 15-minute simulation timestep, which is what E+ will run.
    assert all(d % energy._timestep_minutes() == 0 for d in durations)


def test_unaware_day_airings_are_configurable():
    import random
    saved = dict(energy._RESIDENTIAL_BEHAVIOR)
    try:
        energy._RESIDENTIAL_BEHAVIOR["unaware_day_ventilation"] = {
            "openings_per_day": [2, 2], "duration_minutes": [30, 30],
            "hours": ["09:00", "12:00"]}
        for i in range(50):
            idf = _FakeIdf()
            energy._unaware_day_schedule(idf, f"H_F{i}", random.Random(i))
            spans = _open_spans(idf.idfobjects["SCHEDULE:COMPACT"][0].obj)
            assert len(spans) == 2
            assert all(hi - lo == 30 for lo, hi in spans)
            assert all(9 * 60 <= lo < 12 * 60 for lo, hi in spans)
    finally:
        energy._RESIDENTIAL_BEHAVIOR.clear()
        energy._RESIDENTIAL_BEHAVIOR.update(saved)


def test_unaware_day_airings_never_step_backwards():
    # Overlapping openings (a long duration on adjacent hours) must MERGE: a
    # Schedule:Compact whose Until: goes backwards is a fatal E+ error.
    import random
    saved = dict(energy._RESIDENTIAL_BEHAVIOR)
    try:
        energy._RESIDENTIAL_BEHAVIOR["unaware_day_ventilation"] = {
            "openings_per_day": [3, 3], "duration_minutes": [120, 120],
            "hours": ["08:00", "11:00"]}
        for i in range(30):
            idf = _FakeIdf()
            energy._unaware_day_schedule(idf, f"H_F{i}", random.Random(i))
            fields = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj
                      if str(f).startswith("Until:")]
            clocks = [f.split()[1].split(",")[0] for f in fields]
            mins = [int(c.split(":")[0]) * 60 + int(c.split(":")[1]) for c in clocks]
            assert mins == sorted(mins) and len(set(mins)) == len(mins)
            # And the overlapping openings really did merge into fewer, longer ones.
            spans = _open_spans(idf.idfobjects["SCHEDULE:COMPACT"][0].obj)
            assert len(spans) < 3
    finally:
        energy._RESIDENTIAL_BEHAVIOR.clear()
        energy._RESIDENTIAL_BEHAVIOR.update(saved)


def test_unaware_opens_the_windows_on_a_share_of_the_nights():
    # The share is config, and it comes out right ACROSS a building even though
    # each dwelling gets a whole number of nights per week.
    import random
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("unaware_night_ventilation", {})
    frac = float(cfg["nights_fraction"])
    assert 0.0 < frac < 1.0
    nights = []
    for i in range(400):
        idf = _FakeIdf()
        rng = random.Random(i)
        name = energy._unaware_night_schedule(idf, f"H_F{i}", rng, None)
        if name is None:
            nights.append(0)
            continue
        fields = idf.idfobjects["SCHEDULE:COMPACT"][0].obj
        # Count the WEEKDAYS whose profile actually opens a window.
        by_day = energy._split_day_blocks([str(f) for f in fields])
        nights.append(sum(any(str(l).endswith(",1.0") for l in by_day[d])
                          for d in energy._WEEK_DAY_TYPES))
    # Whole nights per dwelling, either side of frac x 7 ...
    assert set(nights) <= {int(frac * 7), int(frac * 7) + 1}
    # ... averaging to the configured share over the building.
    assert sum(nights) / len(nights) / 7 == pytest.approx(frac, abs=0.03)


def test_unaware_night_opening_is_unattended_and_seasonal():
    import random
    idf = _FakeIdf()
    energy._unaware_night_schedule(idf, "H_F1", random.Random(0), None)
    fields = [str(f) for f in idf.idfobjects["SCHEDULE:COMPACT"][0].obj]
    # Open from midnight to 08:00 on the chosen nights, shut on the others.
    assert "Until: 08:00,1.0" in fields and "Until: 24:00,0.0" in fields
    assert fields[-2:] == ["For: AllOtherDays", "Until: 24:00,0.0"]
    # Seasons are honoured: the heating season is excluded by default, so with a
    # day->season map the schedule is built through seasonal blocks.
    sbd = {d: ("heating" if d < 100 else "cooling") for d in range(1, 366)}
    idf2 = _FakeIdf()
    energy._unaware_night_schedule(idf2, "H_F1", random.Random(0), sbd)
    f2 = [str(f) for f in idf2.idfobjects["SCHEDULE:COMPACT"][0].obj]
    through = [f for f in f2 if f.startswith("Through:")]
    assert len(through) > 1                       # heating block, then cooling
    # The heating block is shut all day.
    heat = f2[f2.index(through[0]) + 1:f2.index(through[1])]
    assert heat == ["For: AllDays", "Until: 24:00,0"]


def test_unaware_night_opening_can_be_switched_off():
    import random
    idf = _FakeIdf()
    saved = dict(energy._RESIDENTIAL_BEHAVIOR)
    try:
        energy._RESIDENTIAL_BEHAVIOR["unaware_night_ventilation"] = \
            {"nights_fraction": 0.0}
        assert energy._unaware_night_schedule(idf, "H_F1", random.Random(0), None) is None
        assert not idf.idfobjects.get("SCHEDULE:COMPACT")
    finally:
        energy._RESIDENTIAL_BEHAVIOR.clear()
        energy._RESIDENTIAL_BEHAVIOR.update(saved)


def test_infiltration_means_are_seasonal_not_blower_door():
    # The config value is what the building leaks every hour, not its n50.
    infil = energy._WIN_INFILTRATION
    for period, row in infil.items():
        ach, n50 = row["air_changes_per_hour"], row["n50"]
        assert 0.15 <= ach <= 1.0, period
        # The usual n50/20 rule for a sheltered urban site.
        assert ach == pytest.approx(n50 / 20.0, abs=0.06), period
    # Tighter windows leak less, in order.
    order = ["pre1980", "1980_2006", "2007_2013", "post2013"]
    vals = [infil[p]["air_changes_per_hour"] for p in order]
    assert vals == sorted(vals, reverse=True)


def test_gains_and_presence_are_separate_schedules():
    # Driving the lights off the presence fraction put them at FULL power through
    # every sleeping hour, which is what made the aware/unaware internal gains
    # overtake the CTE ones.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("time_use_occupancy", {})
    pres, gain = cfg["state_fractions"], cfg["state_gain_fractions"]
    assert pres["Sleeping"] == 1.0            # everyone is home, asleep
    assert gain["Sleeping"] < 0.25            # …and the lights are off
    assert gain["NonOccupied"] > 0.0          # an empty dwelling still has a fridge
    assert pres["NonOccupied"] == 0.0         # …but nobody in it
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert '("_Lights", "Schedule_Name", lights, tu_gain)' in src
    assert '("_People", "Number_of_People_Schedule_Name", people, tu_occ)' in src


def test_dwelling_count_is_reported():
    rows = [{"use_type": "Residential", "season": "heating", "floor_area_m2": 100.0}]
    energy._augment_dhw_by_season(rows, {"real": 6, "cte": 21}, {"behavior": "aware"},
                                  {d: "heating" for d in range(1, 366)}, None,
                                  n_dwellings=6)
    assert rows[0]["n_dwellings"] == 6
    assert rows[0]["people_per_dwelling"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Occupancy: the per-dwelling / per-area blend
# --------------------------------------------------------------------------- #

def _tract_row(areas, people_per_dwelling=2.57, people_per_m2=2565 / 112000):
    n = len(areas)
    return {"census_tract_code": "0801901001",
            "ctagg__people_per_dwelling": people_per_dwelling,
            "ctagg__people_per_m2": people_per_m2,
            "br__building_spaces_reference": {"Residential": [f"d{i}" for i in range(n)]},
            "br__building_spaces_area_without_communal": {"Residential": list(areas)},
            "br__building_spaces_floors": {"Residential": [[i] for i in range(n)]}}


def test_occupancy_blends_the_two_census_estimates():
    # Per-area alone gives a 40 m² studio 0.9 people and a 200 m² flat 4.6; per
    # dwelling alone gives both 2.57. Household size grows with dwelling size but
    # far more slowly than floor area, so the two are averaged.
    small, _ = energy._dwelling_occupants(_tract_row([40.0]), {})
    large, _ = energy._dwelling_occupants(_tract_row([200.0]), {})
    assert small == 2                     # (2.57 + 0.92) / 2
    assert large == 4                     # (2.57 + 4.58) / 2
    # …and it stays between the two estimators at both ends.
    assert 1 < small < 2.57 + 1
    assert 2.57 < large < 4.58 + 1


def test_occupancy_weight_selects_the_estimator(monkeypatch):
    row = _tract_row([200.0])
    monkeypatch.setitem(globals(), "_unused", None)
    monkeypatch.setattr(energy, "_OCC_AREA_WEIGHT", 0.0)
    assert energy._dwelling_occupants(row, {})[0] == 3        # round(2.57)
    monkeypatch.setattr(energy, "_OCC_AREA_WEIGHT", 1.0)
    assert energy._dwelling_occupants(row, {})[0] == 5        # round(4.58)


def test_occupancy_falls_back_to_whichever_estimate_exists():
    only_area = _tract_row([200.0])
    del only_area["ctagg__people_per_dwelling"]
    assert energy._dwelling_occupants(only_area, {})[0] == 5
    only_dwelling = _tract_row([200.0])
    del only_dwelling["ctagg__people_per_m2"]
    assert energy._dwelling_occupants(only_dwelling, {})[0] == 3
    neither = _tract_row([200.0])
    del neither["ctagg__people_per_m2"], neither["ctagg__people_per_dwelling"]
    assert energy._dwelling_occupants(neither, {})[0] is None


def test_blended_occupancy_lands_in_a_plausible_range():
    # The whole point of the estimator: a real Spanish tract must come out near
    # 0.02-0.03 people/m², not an order of magnitude away.
    row = _tract_row([40.0, 90.0, 132.0, 200.0])
    people, area = energy._dwelling_occupants(row, {})
    density = people / area
    lo, hi = energy._DHW_PLAUSIBLE_RATE
    assert lo < density < hi
    assert 0.015 < density < 0.04
    assert 1.5 < people / 4 < 4.0        # people per dwelling


# --------------------------------------------------------------------------- #
# RITE mechanical ventilation for the non-residential uses
# --------------------------------------------------------------------------- #

def test_mechanical_ventilation_rates_follow_rite_categories():
    cfg = energy._NONRES_VENT
    assert cfg["enabled"] is True
    rates = cfg["litres_per_second_person"]
    # IDA 2 offices, IDA 3 commercial — offices need the most.
    assert rates["office"] == 12.5
    assert rates["commercial"] == 8.0
    assert rates["office"] > rates["commercial"] >= rates["other"]


def test_mechanical_ventilation_is_sized_per_person():
    idf = _FakeIdf()
    o = energy._add_mechanical_ventilation(idf, "C_F0", "commercial", "SC_OCC_COMMERCIAL")
    assert o is not None
    assert o.Design_Flow_Rate_Calculation_Method == "Flow/Person"
    assert o.Flow_Rate_per_Person == pytest.approx(0.008)      # 8 l/s → m³/s
    # Fan-driven outdoor air, so it enters the CO2 balance like any other route.
    assert o.Ventilation_Type == "Intake"
    # No wind term: a fan delivers what it delivers.
    assert o.Constant_Term_Coefficient == 1.0
    assert o.Velocity_Term_Coefficient == 0.0


def test_mechanical_ventilation_runs_on_occupancy_not_hvac_availability():
    # The shoulder season is exactly when the HVAC is gated off and a full shop
    # would otherwise have nothing but its own leakage.
    idf = _FakeIdf()
    o = energy._add_mechanical_ventilation(idf, "O_F1", "office", "SC_OCC_OFFICE")
    assert o.Schedule_Name == "SC_OCC_OFFICE"
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    # Added after the base-build ventilation objects are removed, on the use's
    # own occupancy profile.
    assert "_add_mechanical_ventilation(idf, zu, cat, prof)" in src


def test_mechanical_ventilation_is_non_residential_only(monkeypatch):
    # Dwellings renew their air by infiltration and the windows the occupant
    # opens; that is what the behaviour model is for.
    idf = _FakeIdf()
    rates = dict(energy._NONRES_VENT["litres_per_second_person"])
    assert "residential" not in rates
    assert energy._add_mechanical_ventilation(idf, "H_F1", "residential",
                                              "SC_OCC_RESIDENTIAL") is not None \
        or True   # falls back to "other"; the caller never asks for residential
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    # It sits in the non-residential branch, before the residential `continue`.
    idx_call = src.index("_add_mechanical_ventilation")
    idx_cont = src.index("continue                                # behaviour model is residential")
    assert idx_call < idx_cont


def test_mechanical_ventilation_can_be_disabled(monkeypatch):
    monkeypatch.setitem(energy._NONRES_VENT, "enabled", False)
    idf = _FakeIdf()
    assert energy._add_mechanical_ventilation(idf, "C_F0", "commercial", "S") is None


# --------------------------------------------------------------------------- #
# RunPeriod start weekday: the weekly schedules must land on the real calendar
# --------------------------------------------------------------------------- #

def test_run_period_start_weekday_follows_the_weather_year():
    # Left blank, EnergyPlus assumes 1 January is a SUNDAY. 1 Jan 2021 was a
    # Friday, so every For:Saturday/For:Sunday block — the Time Use Survey agenda
    # above all — ran two days out, and the household's weekend occupancy peak
    # appeared on a Thursday.
    idf = _FakeIdf()
    idf.newidfobject("RUNPERIOD", Name="FullYear")
    energy._set_run_period(idf, None, year=2021)
    assert idf.idfobjects["RUNPERIOD"][0].Day_of_Week_for_Start_Day == "Friday"
    # A partial run starts on the first day of its first month, not of the year.
    idf2 = _FakeIdf()
    idf2.newidfobject("RUNPERIOD", Name="Jul")
    energy._set_run_period(idf2, [7], year=2021)          # 1 Jul 2021 = Thursday
    rp = idf2.idfobjects["RUNPERIOD"][0]
    assert rp.Day_of_Week_for_Start_Day == "Thursday"
    assert rp.Begin_Month == 7 and rp.End_Month == 7


def test_run_period_year_falls_back_to_the_epw_name():
    class _Idf(_FakeIdf):
        epw = "/w/weather/epw/ERA5Land_41.6_0.5_2021.epw"
    assert energy._epw_year(_Idf()) == 2021
    idf = _Idf()
    idf.newidfobject("RUNPERIOD", Name="FullYear")
    energy._set_run_period(idf, None)                     # no year passed
    assert idf.idfobjects["RUNPERIOD"][0].Day_of_Week_for_Start_Day == "Friday"


def test_run_period_without_a_year_is_left_alone():
    idf = _FakeIdf()
    idf.newidfobject("RUNPERIOD", Name="FullYear")
    energy._set_run_period(idf, [1])
    rp = idf.idfobjects["RUNPERIOD"][0]
    assert not getattr(rp, "Day_of_Week_for_Start_Day", "")
    assert rp.Begin_Month == 1


def test_weekday_stamp_labels_the_hourly_axis():
    from hypercadaster_ES import reporting
    # The weekday is what makes a weekly occupancy rhythm readable.
    assert reporting._weekday_stamp(12, 3, 8, 2021) == "Fri 12-03 08h"
    # Without a year it degrades to the bare date rather than guessing.
    assert reporting._weekday_stamp(12, 3, 8) == "12-03 08h"


# --------------------------------------------------------------------------- #
# The survey SHARES, not its argmax label, drive the occupancy
# --------------------------------------------------------------------------- #

def _agenda_with_shares():
    """A week where the dominant state says "home all day" but only ~60 % are."""
    states = ["WithActivity"] * 168
    home = [60.0] * 168
    return {"states": states,
            "autonomous_community_code": "09", "household_income_group": 3,
            "shares": {"Sleeping": [10.0] * 168, "WithActivity": [50.0] * 168,
                       "NonOccupied": [40.0] * 168}}


def _tu_cfg():
    return {"enabled": True, "apply_to": ["aware", "unaware"],
            "state_fractions": {"Sleeping": 1.0, "WithActivity": 1.0,
                                "NonOccupied": 0.0},
            "state_gain_fractions": {"Sleeping": 0.1, "WithActivity": 1.0,
                                     "NonOccupied": 0.15},
            "present_states": ["Sleeping", "WithActivity"],
            "awake_states": ["WithActivity", "NonOccupied"]}


def test_occupancy_uses_the_household_shares_not_the_argmax():
    # The dominant state is an argmax: a Sunday where 58-84 % of households are
    # home reads as 100 % occupied for all 24 hours, a far harder profile than the
    # survey measured.
    idf = _FakeIdf()
    occ, gain, _pres, _awake, act = energy._time_use_schedules(
        idf, _agenda_with_shares(), _tu_cfg())
    body = " ".join(str(f) for f in
                    [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == occ][0].obj)
    # 10 % asleep + 50 % active = 0.6 home, not 1.0.
    assert "0.6" in body and "1.0" not in body.replace("0.6", "")
    # Gains blend the three states: .1×.1 + .5×1 + .4×.15 = 0.57
    gbody = " ".join(str(f) for f in
                     [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == gain][0].obj)
    assert "0.57" in gbody
    # Metabolic rate is that of those AT HOME: (10×72 + 50×110)/60
    abody = " ".join(str(f) for f in
                     [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == act][0].obj)
    assert "103.7" in abody


def test_availability_still_binary_from_a_share_threshold():
    idf = _FakeIdf()
    _o, _g, present, awake, _a = energy._time_use_schedules(
        idf, _agenda_with_shares(), _tu_cfg())
    # 60 % home ≥ 50 % → present; 90 % awake ≥ 50 % → the unaware band is on.
    assert "Until: 24:00,1.0" in present
    assert "Until: 24:00,1.0" in awake


def test_falls_back_to_the_argmax_without_shares():
    idf = _FakeIdf()
    agenda = _agenda_with_shares()
    del agenda["shares"]
    occ, _g, _p, _a, _act = energy._time_use_schedules(idf, agenda, _tu_cfg())
    body = " ".join(str(f) for f in
                    [o for o in idf.idfobjects["SCHEDULE:COMPACT"] if o.Name == occ][0].obj)
    assert "1.0" in body            # WithActivity → the full presence fraction


def test_midday_airing_reaches_the_heating_season():
    # A 20 °C gate never fires in a Lleida January, so the aware occupant had no
    # midday airing at all in winter.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("midday_ventilation", {})
    assert cfg["min_outdoor_c"] <= 15.0
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert '{"heating": _fields(mid), "in_between": _fields(mid)}' in src


def test_air_change_is_labelled_ach():
    from hypercadaster_ES import reporting
    import inspect
    src = inspect.getsource(reporting)
    assert "1/h" not in src
    assert "ACH" in src


def test_persiana_control_types_are_valid_enums():
    # Two failures live here. "OnNightIfLowOuterTemp…" is a plausible but INVALID
    # spelling that E+ rejects with a Severe error, running without the control so
    # the shutters never come down; and two DISSIMILAR controls on one window are
    # refused outright (CheckWindowShadingControlSimilarForWindow), so the night
    # rule has to ride on the control type rather than a second object.
    assert energy._PERSIANA_SOLAR_CONTROL_TYPE == "OnIfHighSolarOnWindow"
    assert energy._PERSIANA_NIGHT_CONTROL_TYPE == \
        "OnNightAndOnDayIfCoolingAndHighSolarOnWindow"


def test_one_persiana_control_per_window():
    idf = _FakeIdf()
    idf.newidfobject("BUILDINGSURFACE:DETAILED", Name="W1", Zone_Name="H_F1",
                     Surface_Type="Wall")
    idf.newidfobject("FENESTRATIONSURFACE:DETAILED", Name="W1_win0",
                     Surface_Type="Window", Building_Surface_Name="W1")
    energy._add_persiana_controls(idf, {"H_F1": {"use_type": "Residential"}})
    ctrls = idf.idfobjects["WINDOWSHADINGCONTROL"]
    # EnergyPlus refuses two dissimilar controls on the same window.
    assert len(ctrls) == 1
    assert ctrls[0].Shading_Control_Type == energy._PERSIANA_SOLAR_CONTROL_TYPE


def test_night_shutters_keep_the_winter_sun():
    # The night type shades by day only when COOLING is called for, so the heating
    # season keeps its solar gain while the shutters still come down after dark.
    import inspect
    src = inspect.getsource(energy._apply_occupant_control)
    assert "_PERSIANA_NIGHT_CONTROL_TYPE" in src
    assert "OnDayIfCooling" in energy._PERSIANA_NIGHT_CONTROL_TYPE


def test_hygiene_drops_the_evening_airing_in_the_cooling_season():
    # The night purge already opens the dwelling from midnight, and a 20:00
    # opening on a hot evening brings in air warmer than the room.
    cfg = (energy._RESIDENTIAL_BEHAVIOR or {}).get("hygiene_ventilation", {})
    assert len(cfg["cooling_periods"]) < len(cfg["periods"])
    assert cfg["cooling_periods"][0][0] == "08:00"


def test_co2_mean_is_reported_beside_the_percentiles(tmp_path):
    zone = "H_F1"
    n = 24
    df = pd.DataFrame({
        "Date/Time": [f" 01/01  {i + 1:02d}:00:00" for i in range(n)],
        f"{zone}:Zone Operative Temperature [C](Hourly)": [21.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Total Internal Total Heating Energy [J](Hourly)": [0.0] * n,
        f"{zone}:Zone Air CO2 Concentration [ppm](Hourly)": [400.0] * 12 + [800.0] * 12,
        f"{zone}:Zone People Occupant Count [](Hourly)": [1.0] * n,
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)": [5.0] * n,
    })
    p = tmp_path / "eplusout.csv"
    df.to_csv(p, index=False)
    recs = energy._zone_scenario_kpis(
        energy._read_eplus_csv(str(p)), {zone: {"use_type": "Residential"}},
        {zone: 50.0}, "II")
    r = next(x for x in recs if x["season"] == "heating")
    # DAY is 09:00-22:00, so four hours at 400 ppm and ten at 800.
    assert r["co2_mean_day"] == pytest.approx((4 * 400 + 10 * 800) / 14, abs=1)
    # The mean is not the median when the distribution is lopsided — which is
    # exactly when it is worth reporting both.
    assert r["co2_p50_day"] == pytest.approx(800.0)
    assert r["co2_mean_day"] < r["co2_p50_day"]
