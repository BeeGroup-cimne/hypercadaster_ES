"""Selector normalisation for download()/merge() — accept ints, fail loudly."""
import pytest

import hypercadaster_ES as hc
from hypercadaster_ES.functions import _as_code_list


def test_int_province_is_zero_padded():
    assert _as_code_list([25], 2) == ["25"]
    assert _as_code_list([8], 2) == ["08"]
    assert _as_code_list(8, 2) == ["08"]


def test_int_ine_is_zero_padded_to_five():
    assert _as_code_list([8019], 5) == ["08019"]


def test_digit_strings_are_also_zero_padded():
    # strings missing their leading zero must be normalised too
    assert _as_code_list(["8"], 2) == ["08"]
    assert _as_code_list(["8019"], 5) == ["08019"]


def test_already_padded_and_nonnumeric_kept_verbatim():
    assert _as_code_list(["08900"], 5) == ["08900"]   # already width -> unchanged
    assert _as_code_list("25", 2) == ["25"]
    assert _as_code_list(["088900"], 5) == ["088900"]  # longer -> not truncated
    assert _as_code_list(None, 2) is None


def test_merge_requires_a_selector():
    with pytest.raises(ValueError):
        hc.merge("/tmp/does-not-matter")


def test_merge_unresolved_selection_raises_clear_error(tmp_path, monkeypatch):
    # Simulate a province code that matches no municipality (offline): the guard
    # must raise a clear ValueError instead of the cryptic downstream KeyError.
    monkeypatch.setattr(hc.utils, "list_municipalities", lambda **kw: [])
    with pytest.raises(ValueError, match="No municipalities resolved"):
        hc.merge(str(tmp_path), province_codes=["99"])
