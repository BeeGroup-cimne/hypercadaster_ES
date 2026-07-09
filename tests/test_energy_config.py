"""Energy-model configuration loading and overriding (offline)."""
import textwrap

from hypercadaster_ES import energy


def test_get_energy_config_is_dict():
    cfg = energy.get_config()
    assert isinstance(cfg, dict)
    assert cfg  # non-empty defaults loaded from energy_config.yaml


def test_configure_energy_merges_user_yaml(tmp_path):
    baseline = energy.get_config()
    try:
        override = tmp_path / "override.yaml"
        override.write_text(textwrap.dedent(
            """
            geometry:
              cat_area_shrink: false
            """
        ))
        cfg = energy.configure(str(override))
        assert cfg["geometry"]["cat_area_shrink"] is False
    finally:
        # restore package defaults so we don't leak state into other tests
        energy.configure(None)
        assert energy.get_config()["geometry"] == baseline["geometry"]
