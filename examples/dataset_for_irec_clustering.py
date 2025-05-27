import pandas as pd
from hypercadaster_ES import interoperability
import social_ES.utils_INE as sc
import geopandas as gpd
from shapely.geometry import Point

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]

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