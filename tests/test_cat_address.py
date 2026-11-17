"""CAT parcel-record (type 11) address fallback for buildings CartoCiudad misses."""
import gzip
import os

import pandas as pd

from hypercadaster_ES import downloaders, mergers


def _type11_line(values):
    """Fixed-width CAT type-11 record with the given catstruct[11] fields."""
    line = [" "] * 1000
    line[0:2] = "11"
    spec = {c[3]: (c[0], c[1]) for c in downloaders.catstruct[11]}
    for name, value in values.items():
        start, length = spec[name]
        line[start - 1:start - 1 + length] = str(value).ljust(length)[:length]
    return "".join(line).rstrip() + "\n"


def _write_cat(cat_dir, kind, code, lines):
    folder = os.path.join(cat_dir, f"{code[:2]}_{kind}_01012026_CAT")
    os.makedirs(folder, exist_ok=True)
    with gzip.open(os.path.join(folder, f"{code}{kind}_01012026.CAT.gz"), "wt", encoding="latin-1") as f:
        f.writelines(lines)


RURAL = {
    "parcela_catastral": "25999A00100086", "codigo_postal": "25737",
    "codigo_poligono": "001", "codigo_parcela": "00086", "nombre_paraje": "TRILLA",
    "coordenada_x_por_cien": "032984930", "coordenada_y_por_cien": "0463624029",
    "codigo_epsg": "EPSG:25831",
}
URBAN = {
    "parcela_catastral": "0354404CG3305N", "tipo_via": "CL", "nombre_via": "CALVARI",
    "primer_numero_policia": "0000", "codigo_postal": "00000",
    "coordenada_x_por_cien": "033023833", "coordenada_y_por_cien": "0463525259",
    "codigo_epsg": "EPSG:25831",
}


def test_rural_and_urban_parcel_addresses(tmp_path):
    cat_dir = str(tmp_path)
    _write_cat(cat_dir, "U", "25999", [_type11_line(URBAN)])
    _write_cat(cat_dir, "R", "25999", [_type11_line(RURAL)])

    out = mergers.get_CAT_address(cat_dir, ["25999"]).set_index("building_reference")

    rural = out.loc["25999A00100086"]
    assert (rural.street_type, rural.street_name) == ("PARAJE", "TRILLA")
    assert rural.street_number_label == "POLIGONO 1 PARCELA 86"
    assert rural.street_number_clean is None
    assert rural.postal_code == "25737"
    assert rural.address_source == mergers.CAT_ADDRESS_SOURCE
    # Coordinates keep their centimetres (Float32 would round y by ~0.3 m).
    assert (rural.location.x, rural.location.y) == (329849.30, 4636240.29)

    urban = out.loc["0354404CG3305N"]
    assert (urban.street_type, urban.street_name, urban.street_number_label) == ("CALLE", "CALVARI", "S-N")
    assert pd.isna(urban.postal_code)  # urban records leave it as 00000
    assert str(out.crs) == "EPSG:25831"


def test_building_reference_filter_and_missing_files(tmp_path):
    cat_dir = str(tmp_path)
    _write_cat(cat_dir, "R", "25999", [_type11_line(RURAL)])
    # Only the rural file exists: the urban lookup must neither wait nor pick it up.
    assert downloaders.parse_CAT_file("25999", cat_dir, [11], cat_kind="U", wait_timeout=0) is None
    assert mergers.get_CAT_address(cat_dir, ["25999"], building_references=["OTHER"]).empty
    assert len(mergers.get_CAT_address(cat_dir, ["25999"], building_references=["25999A00100086"])) == 1
    # No CAT file at all for the municipality.
    assert mergers.get_CAT_address(cat_dir, ["25998"]).empty
