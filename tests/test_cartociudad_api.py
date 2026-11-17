"""CartoCiudad API-Features fallback — thoroughfare split and INE name matching."""
import pytest

from hypercadaster_ES.downloaders import (
    _normalize_province_name,
    _uninvert_ine_name,
    split_cartociudad_thoroughfare,
)


@pytest.mark.parametrize("thoroughfare, expected", [
    ("CALLE MAJOR", ("CALLE", "MAJOR")),
    ("CARRETERA C-1313", ("CARRETERA", "C-1313")),
    ("PLAÇA DE L'ESGLÉSIA", ("PLAÇA", "DE L'ESGLÉSIA")),
    # multi-word raw types win over their one-word prefix
    ("PASAJE, PASADIZO SANT JOAN", ("PASAJE, PASADIZO", "SANT JOAN")),
    ("GRAN VIA DE LES CORTS CATALANES", ("GRAN VIA", "DE LES CORTS CATALANES")),
    ("CALLE / KALEA NAGUSIA", ("CALLE / KALEA", "NAGUSIA")),
    # whole-word prefixes only: CALLEJA is not CALLE + "JA ..."
    ("CALLEJA DEL SOL", ("CALLEJA", "DEL SOL")),
    # a type with no name
    ("DISEMINADOS", ("DISEMINADOS", None)),
    # unknown leading word -> no type, whole string is the name
    ("MASIA PERDUTO", ("MASIA", "PERDUTO")),
    ("EL PLA", (None, "EL PLA")),
    # whitespace is collapsed
    ("  CALLE   SANTA   MARIA ", ("CALLE", "SANTA MARIA")),
    (None, (None, None)),
])
def test_split_cartociudad_thoroughfare(thoroughfare, expected):
    assert split_cartociudad_thoroughfare(thoroughfare) == expected


def test_split_falls_back_to_resolvable_first_word():
    # not in the GeoPackage vocabulary, but a known regional street type
    assert split_cartociudad_thoroughfare("ETORBIDEA GASTEIZ") == ("ETORBIDEA", "GASTEIZ")


@pytest.mark.parametrize("ine_name, cartociudad_name", [
    ("Roda, La", "La Roda"),
    ("Atzúbia, l'", "l'Atzúbia"),
    ("Camp de Mirra, el/Campo de Mirra", "el Camp de Mirra/Campo de Mirra"),
    ("Cubells", "Cubells"),
])
def test_uninvert_ine_name_matches_cartociudad(ine_name, cartociudad_name):
    assert (_normalize_province_name(_uninvert_ine_name(ine_name))
            == _normalize_province_name(cartociudad_name))
