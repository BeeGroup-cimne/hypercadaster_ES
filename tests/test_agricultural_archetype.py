"""Agricultural buildings model as uninsulated, unconditioned warehouses.

A rural shed takes the same assumptions as a warehouse (no people, no
equipment, no HVAC, vehicle door) but is uninsulated whatever its year and very
leaky. See ``loads.agricultural`` / ``archetype_envelope.uninsulated`` in
energy_config.yaml.
"""
import pytest

from hypercadaster_ES import energy


def test_agricultural_is_its_own_unconditioned_use_category():
    assert energy._use_category("Agricultural") == "agricultural"
    p = energy._LOAD_LIBRARY["agricultural"]
    assert p["conditioned"] is False
    assert p["ppl_per_m2"] == 0.0
    assert p["equip"] == 0.0


def test_agricultural_assumptions_match_warehouse_except_infiltration():
    """'All the assumptions should be equal to warehouse' — warehouse maps to
    the `parking` category — 'very high infiltration' is the exception."""
    agri = energy._LOAD_LIBRARY["agricultural"]
    warehouse = energy._LOAD_LIBRARY[energy._use_category("Warehouse - Parking")]

    shared = {k: v for k, v in warehouse.items() if k != "infil_ach"}
    assert {k: agri[k] for k in shared} == shared
    assert agri["infil_ach"] > warehouse["infil_ach"]


def test_agricultural_gets_no_insulation_whatever_the_year():
    """No wall/roof/floor U target -> _add_constructions drops the INS layer."""
    for year in (1950, 1996, 2010, 2020):
        env = energy.building_archetype(year, "Agricultural")["envelope"]
        assert "wall_U" not in env
        assert "roof_U" not in env
        assert "floor_U" not in env


def test_other_uses_keep_their_period_envelope():
    arch = energy.building_archetype(1996, "Residential")
    assert arch["use_category"] == "residential"
    assert arch["envelope"]["wall_U"] == pytest.approx(0.95)
    # the period is still reported for an agricultural building
    assert energy.building_archetype(1996, "Agricultural")["period"] == "1980_2006"


def test_agricultural_zone_gets_a_vehicle_door():
    assert energy._zone_is_parking({"Z": {"use_type": "Agricultural"}}, "z") is True
    assert energy._zone_is_parking({"Z": {"use_type": "Warehouse - Parking"}}, "z") is True
    assert energy._zone_is_parking({"Z": {"use_type": "Residential"}}, "z") is False


def test_dominant_use_falls_back_when_no_zone_is_occupied():
    """A barn's only zone is non_occupied; without the fallback the archetype
    would resolve to the generic 'other' instead of agricultural."""
    barn = [{"kind": "non_occupied", "use_type": "Agricultural", "floor_polys": {}}]
    assert energy._dominant_use_type(barn) == "Agricultural"

    # occupied zones still win when present
    mixed = [{"kind": "non_occupied", "use_type": "Agricultural", "floor_polys": {}},
             {"kind": "occupied", "use_type": "Residential", "floor_polys": {}}]
    assert energy._dominant_use_type(mixed) == "Residential"

    # a communal-only building has no characteristic use
    assert energy._dominant_use_type(
        [{"kind": "communal", "use_type": "Communal", "floor_polys": {}}]) is None
