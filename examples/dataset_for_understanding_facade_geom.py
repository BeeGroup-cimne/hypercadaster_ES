import pandas as pd
# import pprint

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]

# Read the pickle file
gdf = pd.read_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")
gdf = gdf[gdf["location"].notnull()]

building_reference = "9325519DF2892E"
gdf_ = gdf[gdf.building_reference == building_reference]
# pprint.pp(gdf_.iloc[0].to_dict())

air_contact = pd.DataFrame(
    [(int(k), float(v)) for k, v in gdf_.iloc[0]["br__air_contact_wall"].items()],
    columns=["angle", "value"]
)
avg_height_of_walls = 3
adiabatic = gdf_.iloc[0]["br__adiabatic_wall"]
patios = gdf_.iloc[0]['br__patios_wall_total']
walls_abs = [
    (air_contact[(air_contact["angle"]<=22.5) | (air_contact["angle"]>337.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=67.5) & (air_contact["angle"]>22.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=112.5) & (air_contact["angle"]>67.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=157.5) & (air_contact["angle"]>112.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=202.5) & (air_contact["angle"]>157.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=247.5) & (air_contact["angle"]>202.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=292.5) & (air_contact["angle"]>247.5)].value.sum() * avg_height_of_walls),
    (air_contact[(air_contact["angle"]<=337.5) & (air_contact["angle"]>292.5)].value.sum() * avg_height_of_walls),
    air_contact.value.sum() * avg_height_of_walls,
    adiabatic * avg_height_of_walls,
    patios * avg_height_of_walls
]
walls_ratio = [round(wall_abs * 100 / (adiabatic * avg_height_of_walls +
                                       air_contact.value.sum() * avg_height_of_walls +
                                       patios * avg_height_of_walls ), 2) for wall_abs in walls_abs]
walls_abs = [round(i, 2) for i in walls_abs]
orient_ind = int(((int(gdf_.iloc[0]['br__parcel_main_orientation']) + 22.5) % 360) // 45)
air_contact_labels = ["N","NE","E", "SE", "S", "SW", "W", "NW", "Total air contact", "Adiabatic", "Patio"]
main_orientation = air_contact_labels[orient_ind]
pd.DataFrame({"orientation": air_contact_labels, "facade_ratio": walls_ratio, "facade_area": walls_abs})
main_orientation