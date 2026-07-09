"""Offline INE <-> cadaster code conversions (backed by a bundled xlsx)."""
import pytest

import hypercadaster_ES as hc
from hypercadaster_ES import utils


def test_ine_to_cadaster_known_pair():
    # ABENGIBRE (Albacete): INE and cadaster codes coincide.
    assert utils.ine_to_cadaster_codes(["02001"]) == ["02001"]


def test_cadaster_to_ine_known_pair():
    assert utils.cadaster_to_ine_codes(["44003"]) == ["44003"]


def test_ine_cadaster_round_trip():
    ine = ["02001", "44003", "08019"]
    cadaster = utils.ine_to_cadaster_codes(ine)
    assert utils.cadaster_to_ine_codes(cadaster) == ine


def test_none_passthrough():
    assert utils.ine_to_cadaster_codes(None) is None
    assert utils.cadaster_to_ine_codes(None) is None


def test_unknown_code_raises():
    with pytest.raises(KeyError):
        utils.ine_to_cadaster_codes(["99999"])
