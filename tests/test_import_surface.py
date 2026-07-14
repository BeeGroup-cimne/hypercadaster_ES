"""The public API surface must import cleanly and stay stable."""
import hypercadaster_ES as hc


# Minimal top-level API — everything else lives on the submodules.
PUBLIC = ["download", "merge", "ResultsHandle"]

# Functions that were intentionally moved off the top level; they must be
# reachable through their submodule instead of `hc.<name>`.
SUBMODULE_API = {
    "utils": ["get_ine_codes_from_bounding_box", "ine_to_cadaster_codes",
              "municipality_name"],
    "mergers": ["aggregate_to_grid", "aggregate_to_h3"],
    "energy": ["build_energy_model", "building_archetype", "building_wwr",
               "plot_idf_geometry", "configure", "get_config"],
    "weather": ["get_municipality_weather", "municipality_epw",
                "weather_statistics", "join_weather"],
}


def test_version_is_a_string():
    assert isinstance(hc.__version__, str)
    assert hc.__version__.count(".") >= 1


def test_public_api_present():
    missing = [name for name in PUBLIC if not hasattr(hc, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_demoted_functions_not_top_level():
    demoted = [n for names in SUBMODULE_API.values() for n in names
               if n not in ("configure", "get_config")]
    leaked = [n for n in demoted if hasattr(hc, n)]
    assert not leaked, f"these should no longer be top-level: {leaked}"


def test_submodule_api_reachable():
    for sub, names in SUBMODULE_API.items():
        module = getattr(hc, sub)
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, f"hc.{sub} missing: {missing}"


def test_all_matches_module_attrs():
    for name in hc.__all__:
        assert hasattr(hc, name), f"__all__ advertises missing symbol {name}"


def test_submodules_importable():
    for sub in ("mergers", "utils", "building_inference", "downloaders",
                "interoperability", "energy", "weather", "social"):
        assert hasattr(hc, sub)
