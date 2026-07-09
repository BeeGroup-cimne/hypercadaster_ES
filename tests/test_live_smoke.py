"""End-to-end smoke test on a tiny municipality.

Downloads real cadastral data for AGUATON (Teruel, cadaster ``44003`` — ~65
urban buildings), so it is gated behind the ``live`` marker and skipped by
default. Run with::

    HC_RUN_LIVE=1 pytest tests/test_live_smoke.py
    # or
    pytest -m live

The download is cached under a temp working directory for the session.
"""
import pytest

import hypercadaster_ES as hc

CADASTER = "44003"  # AGUATON, Teruel — small municipality


@pytest.fixture(scope="module")
def aguaton_wd(tmp_path_factory):
    wd = str(tmp_path_factory.mktemp("hc_live"))
    hc.download(
        wd,
        cadaster_codes=[CADASTER],
        neighborhood_layer=False,
        open_data_layers=False,
    )
    return wd


@pytest.mark.live
def test_download_and_merge_small_municipality(aguaton_wd):
    gdf = hc.merge(
        aguaton_wd,
        cadaster_codes=[CADASTER],
        building_parts_inference=False,
        # neighborhood + open-data layers only exist for Barcelona
        neighborhood_layer=False,
        open_data_layers=False,
    )
    assert len(gdf) > 0
    assert "building_reference" in gdf.columns
    assert gdf["building_reference"].notna().any()
    assert gdf.geometry.notna().any()
