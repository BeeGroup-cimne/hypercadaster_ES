"""Corrupt-tile detection used by the DEM downloader's self-healing cache."""
import numpy as np
import pytest

from hypercadaster_ES.downloaders import _is_readable_raster


@pytest.fixture
def valid_tif(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    path = tmp_path / "tile.tif"
    data = (np.random.rand(1, 64, 64) * 100).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(0, 0, 1, 1, 64, 64)) as ds:
        ds.write(data)
    return path


def test_valid_tif_is_readable(valid_tif):
    assert _is_readable_raster(str(valid_tif)) is True


def test_truncated_tif_is_not_readable(valid_tif):
    size = valid_tif.stat().st_size
    with open(valid_tif, "r+b") as f:      # chop the file in half -> corrupt strips
        f.truncate(size // 2)
    assert _is_readable_raster(str(valid_tif)) is False


def test_missing_file_is_not_readable(tmp_path):
    assert _is_readable_raster(str(tmp_path / "nope.tif")) is False


def test_non_raster_file_is_not_readable(tmp_path):
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff at all")
    assert _is_readable_raster(str(junk)) is False
