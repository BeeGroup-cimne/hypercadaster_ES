"""Interoperability functions for external systems integration.

This module provides functions for converting hypercadaster_ES data 
into formats compatible with external simulation and analysis tools.

Main functions:
    - input_files_for_IREC_simulations(): Convert data for IREC building energy simulations
    - plot_weather_stations(): Visualize weather station clustering
    - converter_(): Provide data structure mappings for external tools

TODO:
- Add real street width calculations using parcel geometries
- Add comprehensive rehabilitation analysis
- Investigate empty orientation cases: 9505427DF2890F
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def input_files_for_IREC_simulations(gdf):

    gdf_filt = gdf.drop_duplicates(subset="building_reference")
    gdf_filt = gdf_filt[gdf_filt["br__building_spaces"].apply(lambda d: isinstance(d, dict) and "Residential" in d)]

    def classify_building_type(spaces, detached):
        residential_units = spaces.get("Residential", 0)
        non_residential_units = sum(v for k, v in spaces.items() if k != "Residential")
        if residential_units == 1:
            return "SF"
        elif residential_units > 1:
            return "MFI" if detached else "MFNI"
        elif residential_units == 0:
            return "NR"
        return "Unknown"


    def calculate_typology_percentages(areas):
        if not isinstance(areas, dict):
            areas = areas.values[0]
        total = sum(areas.values())
        return {
            "BuildingResidentialArea": areas.get("Residential", 0) / total * 100 if total else 0,
            "BuildingCommercialArea": areas.get("Commercial", 0) / total * 100 if total else 0,
            "BuildingOfficesArea": areas.get("Offices", 0) / total * 100 if total else 0,
            "BuildingParkingArea": areas.get("Warehouse - Parking", 0) / total * 100 if total else 0,
            "BuildingOtherUsesArea": 100 - (
                    areas.get("Residential", 0) +
                    areas.get("Commercial", 0) +
                    areas.get("Offices", 0) +
                    areas.get("Warehouse - Parking", 0)
            ) / total * 100 if total else 0
        }

    def extract_floors_by_use(floor_data, use_type):
        return sorted([
            int(f) for building_use, floors in floor_data.items()
            if building_use == use_type
            for f, area in floors.items()
            if area > 0
        ])

    def process_row(row):
        try:
            building_spaces = row["br__building_spaces"]
            area_wo_communal = row["br__area_without_communals"]
            area_w_communal = row["br__area_with_communals"]
            floor_data = row["br__area_with_communals_by_floor"]
            eff_years = row["br__mean_building_space_effective_year"]
            year_of_construction = row["year_of_construction"] if pd.notna(row["year_of_construction"]) else eff_years["Residential"]
            if year_of_construction<1850 and eff_years["Residential"]>1850:
                year_of_construction = eff_years["Residential"]
            elif year_of_construction<1850:
                year_of_construction = 1850
            avg_eff_year = np.mean(list(eff_years.values())) if eff_years else row["year_of_construction"]
            use_percentages = calculate_typology_percentages(area_w_communal)
            return pd.Series({
                "BuildingReference": row["building_reference"],
                "BuildingType": classify_building_type(building_spaces, row["br__detached"]),
                "Location": row["location"],
                "CensusTract": row["section_code"],
                "PostalCode": row["postal_code"],
                "AllParcelOrientations": row['br__parcel_orientations'],
                "MainParcelOrientation": row['br__parcel_main_orientation'],
                "AllParcelOrientationsStreetWidths": row['br__street_width_by_orientation'],
                "MainParcelOrientationStreetWidth": row['br__street_width_main_orientation'],
                "NumberOfDwelling": building_spaces.get("Residential", 0),
                "UsefulResidentialArea": area_wo_communal.get("Residential", 0),
                "YearOfConstruction": year_of_construction,
                "BuildingWasRetroffited": year_of_construction < avg_eff_year,
                "YearOfRetroffiting": avg_eff_year if year_of_construction < avg_eff_year else year_of_construction,
                **use_percentages,
                "BuildingResidentialFloors": extract_floors_by_use(floor_data, "Residential"),
                "BuildingCommercialFloors": extract_floors_by_use(floor_data, "Commercial"),
                "BuildingOfficesFloors": extract_floors_by_use(floor_data, "Offices"),
                "BuildingParkingFloors": extract_floors_by_use(floor_data, "Warehouse - Parking"),
                "NumberOfFloorsAboveGround": 1 + max([max(floors.keys()) for floors in list(floor_data.values())]),
                "NumberOfFloorsBelowGround": min([min(floors.keys()) for floors in list(floor_data.values())])
            })
        except Exception as e:
            print((row["building_reference"],e))


    # Assuming `df` is your original DataFrame
    new_df = gdf_filt.apply(process_row, axis=1)
    return new_df

def converter_():
    return {
        'Edad': {
            'Menos de 30 años': {},
            'De 30 a 39 años': {},
            'De 40 a 49 años': {},
            'De 50 a 59 años': {},
            'De 60 a 69 años': {},
            'De 70 y más años': {}
        },
        'Sexo': {
            'Hombre': {},
            'Mujer': {}
        },
        'Tipo de núcleo familiar': None,
        'Tipo de unión': None,
        'Nivel educativo alcanzado de la pareja': None,
        'Situación laboral de la pareja': None,
        'Nivel de ingresos mensuales netos del hogar': {
            'Menos de 1.000 euros': {},
            'De 1.000 euros a menos de 1.500 euros': {},
            'De 1.500 euros a menos de 2.000 euros': {},
            'De 2.000 euros a menos de 3.000 euros': {},
            '3.000 euros o más': {}
        },
        'Sexo del progenitor': None,
        'Estado civil del progenitor': None,
        'Nivel educativo del progenitor': None,
        'Situación laboral del progenitor': None,
        'Tipo de hogar': None,
        'Número de miembros del hogar': {
            '1 persona': {},
            '2 personas': {},
            '3 personas': {},
            '4 personas o más': {}
        },
        'Nivel de estudios alcanzado por los miembros del hogar': None,
        'Situación laboral de los miembros del hogar': None,
        'Tipo de edificio': {
            'Total': {},
            'Vivienda unifamiliar (chalet, adosado, pareado...)': {},
            'Edificio de 2 o más viviendas': {}
        },
        'Año de construcción del edificio': {
            'Total': {},
            '2000 y anterior': {},
            'Posterior a 2000': {}
        },
        'Nacionalidad de los miembros del hogar': {
            'Total': {},
            'Hogar exclusivamente español': {},
            'Hogar mixto (con españoles y extranjeros)': {},
            'Hogar exclusivamente extranjero': {}
        },
        'Superficie útil de la vivienda': {
            'Total': {},
            'Hasta 75 m2': {},
            'Entre 76 y 90 m2': {},
            'Entre 91 y 120 m2': {},
            'Más de 120 m2': {}
        }
    }

def plot_weather_stations(gdf, weather_clusters_column, filename):
    fig, ax = plt.subplots(figsize=(15, 12))

    # Correct GeoPandas plotting syntax
    gdf.plot(
        ax=ax,
        column=weather_clusters_column,
        categorical=True,
        legend=True,
        cmap='tab10',
        markersize=20,
        alpha=0.7,
        edgecolor='black'
    )

    ax.set_title('Buildings Colored by WeatherCluster', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)