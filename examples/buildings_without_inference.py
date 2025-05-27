import hypercadaster_ES as hc

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]

hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

province_codes=None
ine_codes=None
neighborhood_layer=True
postal_code_layer=True
census_layer=True
elevations_layer=True
open_data_layers=True
building_parts_inference=False
building_parts_plots=False
use_CAT_files=True
CAT_files_rel_dir="CAT_files"

gdf = hc.merge(
    wd=wd, cadaster_codes=cadaster_codes, province_codes=province_codes, ine_codes=ine_codes,
    neighborhood_layer=neighborhood_layer, postal_code_layer=postal_code_layer, census_layer=census_layer,
    elevations_layer=elevations_layer, open_data_layers=open_data_layers,
    building_parts_inference=building_parts_inference, building_parts_plots=building_parts_plots,
    use_CAT_files=use_CAT_files, CAT_files_rel_dir=CAT_files_rel_dir
)
gdf.to_pickle(f"{wd}/{'~'.join(cadaster_codes)}_only_addresses.pkl", compression="gzip")
