import pandas as pd
import geopandas as gpd
import hypercadaster_ES as hc
from shapely import wkt

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
gdf = gdf.to_crs(epsg=25831)
gdf.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/bcn_buildings_v2.geojson")

gdf_ct = gpd.read_file("/home/gmor/Downloads/seccionado_2025/España_Seccionado2025_ETRS89H30/SECC_CE_20250101.shp")
gdf_ct = gdf_ct[gdf_ct.CLAU2.isin(hc.functions.utils.cadaster_to_ine_codes(
    cadaster_dir=hc.functions.utils.cadaster_dir_(wd),
    cadaster_codes=cadaster_codes))]
gdf_ct = gdf_ct[["CUSEC","geometry"]]
gdf_ct.rename(columns={"CUSEC": "reference"}, inplace=True)
gdf_ct = gdf_ct.to_crs(epsg=25831)
gdf_ct.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/census_tracts.geojson")

gdf_d = gpd.read_file("/home/gmor/Downloads/seccionado_2025/España_Seccionado2025_ETRS89H30/SECC_CE_20250101.shp")
gdf_d = gdf_d[gdf_d.CLAU2.isin(hc.functions.utils.cadaster_to_ine_codes(
    cadaster_dir=hc.functions.utils.cadaster_dir_(wd),
    cadaster_codes=cadaster_codes))]
gdf_d = gdf_d[["CUDIS","geometry"]]
gdf_d.rename(columns={"CUDIS": "reference"}, inplace=True)
gdf_d = gdf_d.dissolve(by="reference", as_index=False)
gdf_d = gdf_d.to_crs(epsg=25831)
gdf_d.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/districts.geojson")

if any([c=="08900" for c in cadaster_codes]):
    gdf_n = gpd.read_file("/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES/neighbourhoods/neighbourhoods.csv")
    columns = \
        {
            "codi_barri": "neighborhood_code",
            "nom_barri": "neighborhood_name",
            "nom_districte": "district_name",
            "geometria_etrs89": "geometry"
        }
    gdf_n.rename(columns = columns, inplace = True)
    gdf_n = gdf_n[columns.values()]
    gdf_n["geometry"] = gdf_n["geometry"].apply(wkt.loads)
    gdf_n = gpd.GeoDataFrame(gdf_n, geometry="geometry", crs='EPSG:25831')
    gdf_n.rename(columns={"neighborhood_code": "reference"}, inplace=True)
    gdf_n["reference_name"] = gdf_n["district_name"] + ', ' + gdf_n["neighborhood_name"]
    gdf_n = gdf_n.drop(columns=["district_name", "neighborhood_name"])
    gdf_n = gdf_n.dissolve(by="reference", as_index=False)
    gdf_n = gdf_n.to_crs(epsg=25831)
    gdf_n.to_file("/home/gmor/Nextcloud2/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Data/NAZKA/neighborhoods.geojson")