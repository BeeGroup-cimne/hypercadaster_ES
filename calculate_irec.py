import hypercadaster_ES as hc
from hypercadaster_ES import interoperability
import social_ES.utils_INE as sc
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]

# hc.download(
#     wd=wd,
#     cadaster_codes=cadaster_codes
# )

province_codes=None
ine_codes=None
neighborhood_layer=True
postal_code_layer=True
census_layer=True
elevations_layer=True
open_data_layers=True
building_parts_inference=True
building_parts_plots=False
use_CAT_files=True
CAT_files_rel_dir="CAT_files"

# gdf = hc.merge(
#     wd=wd, cadaster_codes=cadaster_codes, province_codes=province_codes, ine_codes=ine_codes,
#     neighborhood_layer=neighborhood_layer, postal_code_layer=postal_code_layer, census_layer=census_layer,
#     elevations_layer=elevations_layer, open_data_layers=open_data_layers,
#     building_parts_inference=building_parts_inference, building_parts_plots=building_parts_plots,
#     use_CAT_files=use_CAT_files, CAT_files_rel_dir=CAT_files_rel_dir
# )
# gdf.to_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")

# Read the pickle file
gdf = pd.read_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")
gdf = gdf[gdf["location"].notnull()]

# Transform our general GDF to the IREC's format
gdf_irec = interoperability.input_files_for_IREC_simulations(gdf)

# Load weather data
weather = pd.read_parquet(f"{wd}/results/third_party_datasets/weatherStation_by_cluster.parquet")

# Create geometry column correctly (lon, lat order)
weather["geometry"] = weather.apply(lambda x: Point(float(x["longitude"]), float(x["latitude"])), axis=1)
weather_gdf = gpd.GeoDataFrame(weather, geometry="geometry", crs="EPSG:4326")

# Ensure gdf_irec has a proper geometry column
if "geometry" not in gdf_irec.columns:
    gdf_irec = gpd.GeoDataFrame(gdf_irec, geometry="Location", crs="EPSG:25831")
else:
    gdf_irec = gdf_irec.set_geometry("geometry")

# Project both datasets to projected CRS (e.g., EPSG:25831) for accurate distance calculation
weather_gdf = weather_gdf.to_crs("EPSG:25831")

# Spatial join (nearest) to assign nearest weather station's Cluster to each building
gdf_irec = gpd.sjoin_nearest(
    gdf_irec,
    weather_gdf[["Cluster", "geometry"]],
    how="left",
    distance_col="distance"
)

# Rename assigned cluster column
gdf_irec.rename(columns={"Cluster": "WeatherCluster"}, inplace=True)

# Drop unnecessary columns if desired
gdf_irec = gdf_irec.drop(columns=["index_right"])

# Social
# essential_characteristics = sc.INEEssentialCharacteristicsOfPopulationAndHouseholds(wd)
# population = sc.INEPopulationAnualCensus(wd)
# population = population["Sections"]
# population["census_tract"] = population["Municipality code"] + population["District code"] + population["Section code"]
atlas = sc.INERentalDistributionAtlas(wd)
atlas = atlas["Sections"]
atlas["census_tract"] = atlas["Municipality code"] + atlas["District code"] + atlas["Section code"]
atlas = atlas[atlas["Year"]==2022]
gdf_irec = gdf_irec.merge(atlas[["census_tract", "Tamaño medio del hogar"]],
                          left_on="CensusTract", right_on="census_tract", how="left")
gdf_irec.drop(columns=["census_tract"], inplace=True)
gdf_irec.rename(columns={"Tamaño medio del hogar": "NumberOfPeoplePerHousehold"}, inplace=True)

# EPCs
epc = pd.read_parquet(f"{wd}/results/third_party_datasets/epc_predictor_results.parquet")
gdf_irec = gdf_irec.merge(epc[["building_reference", "WindowToWallRatio", "EPCs_ratio"]],
                          left_on="BuildingReference", right_on="building_reference", how="left")
gdf_irec.drop(columns=["building_reference"], inplace=True)

# Last changes
gdf_irec["AverageDwellingArea"] = gdf_irec["UsefulResidentialArea"] / gdf_irec["NumberOfDwelling"]
gdf_irec = gdf_irec.to_crs("EPSG:4326")
gdf_irec["Latitude"] = gdf_irec["Location"].y
gdf_irec["Longitude"] = gdf_irec["Location"].x
gdf_irec["Projection"] = "EPSG:4326"

# Plot Weather Stations
interoperability.plot_weather_stations(gdf_irec, "WeatherCluster",
                                       f"{wd}/results/building_weather_clusters.png")
interoperability.plot_weather_stations(weather_gdf, "Cluster",
                                       f"{wd}/results/weather_stations_clusters.png")

# Export
gdf_irec.drop(columns=["Location"], inplace=True)
gdf_irec.to_pickle(f"{wd}/results/IREC_bcn_input.pkl")

# failing_zones = [
#     '03735DF3807C', '05192DF3801H', '08169DF3801F', '11859DF3718E', '12078DF3910E', '13182DF3811G', '14063DF3810E',
#     '15923DF3819C', '18654DF3816F', '22043DF3820C', '22157DF3821E', '22343DF3823C', '23157DF3821E', '23166DF3821E',
#     '25281DF3822H', '25423DF3824D', '26474DF3824H', '27737DF2837A', '28394DF3823H', '28833DF3828D', '29646DF3826D',
#     '31326DF3833C', '31458DF3834E', '44546DF2845C', '47518DF2845B', '49494DF3844H', '54525DF2855A', '59243DF2852D',
#     '65192DF2861H', '75688DF2876H', '81041DF2880C', '83526DF2885A', '84654DF2786E', '86789DF2887H', '88235DF2882D',
#     '92827DF2898C', '93655DF2796E', 'unknown'
# ]