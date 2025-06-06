import pandas as pd
import geopandas as gpd
import hypercadaster_ES as hc

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"] #Barcelona

## Generate the elements geojson for the UI
gdf = pd.read_pickle(f"{wd}/{"~".join(cadaster_codes)}_only_addresses.pkl", compression="gzip")
## Get only those buildings with residential area > 0
gdf = gdf.drop_duplicates("building_reference")
gdf = gdf[(gdf.building_area_residential>0) & ~gdf.building_reference.isin(["6873901DF2767D"])]
gdf = gdf[["building_reference", "building_geometry"]]
gdf = gdf.rename(columns={"building_reference": "reference", "building_geometry": "geometry"})
gdf = gdf.set_geometry("geometry")
gdf.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/bcn_buildings_v2.geojson")

gdf_ct = gpd.read_file("/home/gmor/Downloads/seccionado_2025/España_Seccionado2025_ETRS89H30/SECC_CE_20250101.shp")
gdf_ct = gdf_ct[gdf_ct.CLAU2.isin(hc.functions.utils.cadaster_to_ine_codes(
    cadaster_dir=hc.functions.utils.cadaster_dir_(wd),
    cadaster_codes=cadaster_codes))]
gdf_ct = gdf_ct[["CUSEC","geometry"]]
gdf_ct.rename(columns={"CUSEC": "reference"}, inplace=True)
gdf_ct.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/census_tracts.geojson")