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
import folium
import geopandas as gpd
import json


def create_parcel_address_map(result_gdf, target_parcel_ref='6161378DF2766A'):
    """
    Create an interactive Folium map showing the target parcel geometry and its assigned addresses

    Parameters:
    result_gdf: GeoDataFrame - Your final merged result with addresses and building references
    target_parcel_ref: str - The parcel reference to visualize
    """

    print(f"\n=== CREATING INTERACTIVE MAP FOR PARCEL {target_parcel_ref} ===")

    # Filter addresses assigned to target parcel
    target_addresses = result_gdf[result_gdf['building_reference'] == target_parcel_ref].copy()

    if target_addresses.empty:
        print(f"❌ No addresses found assigned to parcel {target_parcel_ref}")
        return None

    print(f"📍 Found {len(target_addresses)} addresses assigned to parcel {target_parcel_ref}")

    # Debug: Check input data type and structure
    print(f"🔍 Debug info:")
    print(f"   - Input type: {type(result_gdf)}")
    print(f"   - Target addresses type: {type(target_addresses)}")
    print(f"   - Available columns: {list(target_addresses.columns)}")

    # Check if it's a proper GeoDataFrame and ensure geometry column exists
    if not isinstance(target_addresses, gpd.GeoDataFrame):
        print(f"❌ Input data is not a GeoDataFrame: {type(target_addresses)}")

        # Try to find geometry-like columns
        geom_cols = [col for col in target_addresses.columns if
                     'geom' in col.lower() or 'point' in col.lower() or col in ['lat', 'lon', 'x', 'y']]
        if geom_cols:
            print(f"💡 Found potential geometry columns: {geom_cols}")
            print("Consider converting to GeoDataFrame first")
        return None

    # Check geometry column
    if not hasattr(target_addresses, 'geometry') or target_addresses.geometry is None:
        print(f"❌ No geometry column found")
        print(f"Available columns: {list(target_addresses.columns)}")
        return None

    if target_addresses.geometry.empty:
        print(f"❌ Geometry column is empty")
        return None

    print(f"   - Geometry column type: {type(target_addresses.geometry)}")
    print(f"   - CRS: {target_addresses.crs}")

    # Remove any rows without valid geometry
    valid_geom_before = len(target_addresses)
    target_addresses = target_addresses[target_addresses.geometry.notna()]
    valid_geom_after = len(target_addresses)

    if target_addresses.empty:
        print(f"❌ No valid geometries found for addresses")
        return None

    print(f"📍 Valid geometries: {valid_geom_after}/{valid_geom_before}")

    # Check first few geometry types
    if len(target_addresses) > 0:
        sample_geoms = target_addresses.geometry.head(3)
        print(f"   - Sample geometries: {[type(g).__name__ for g in sample_geoms]}")

    # Extract parcel geometry directly from the data
    try:
        # Get the parcel geometry from the first row (they should all be the same for the target parcel)
        parcel_geom = target_addresses["parcel_geometry"].iloc[0]

        # Create a proper GeoDataFrame for the parcel
        target_parcel_geom = gpd.GeoDataFrame(
            {"parcel_ref": [target_parcel_ref]},
            geometry=[parcel_geom],
            crs=target_addresses.crs
        )

        print(f"🗺️  Extracted parcel geometry (area: {parcel_geom.area:.2f} m²)")
        print(f"    Parcel type: {type(parcel_geom).__name__}")

    except Exception as e:
        print(f"❌ Error extracting parcel geometry: {e}")
        print(f"    Available parcel_geometry type: {type(target_addresses['parcel_geometry'].iloc[0])}")
        return None

    # Set the correct geometry column for addresses (use address_location instead of geometry)
    if 'address_location' in target_addresses.columns:
        # Create a proper GeoDataFrame with address_location as geometry
        target_addresses = target_addresses.set_geometry('address_location')
        print(f"✅ Using 'address_location' column for address points")

    # Convert to WGS84 for Folium
    web_crs = 'EPSG:4326'
    target_addresses_web = target_addresses.to_crs(web_crs)
    target_parcel_web = target_parcel_geom.to_crs(web_crs)

    # Get center point for map
    parcel_bounds = target_parcel_web.total_bounds
    center_lat = (parcel_bounds[1] + parcel_bounds[3]) / 2
    center_lon = (parcel_bounds[0] + parcel_bounds[2]) / 2

    print(f"🎯 Map center: {center_lat:.6f}, {center_lon:.6f}")

    # Create Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        tiles='OpenStreetMap'
    )

    # Add alternative tile layers
    folium.TileLayer('CartoDB positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)

    # Add parcel geometry with proper error handling
    try:
        # Convert parcel to GeoJSON safely
        parcel_geojson = json.loads(target_parcel_web.to_json())

        # Create parcel popup content
        parcel_area = target_parcel_geom.geometry.iloc[0].area
        popup_content = f"""
        <div style="font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: darkred;">Parcel {target_parcel_ref}</h4>
            <p style="margin: 5px 0;"><strong>Area:</strong> {parcel_area:.2f} m²</p>
            <p style="margin: 5px 0;"><strong>Area (hectares):</strong> {parcel_area / 10000:.2f} ha</p>
            <p style="margin: 5px 0;"><strong>Assigned addresses:</strong> {len(target_addresses)}</p>
        </div>
        """

        folium.GeoJson(
            parcel_geojson,
            style_function=lambda feature: {
                'fillColor': 'red',
                'color': 'darkred',
                'weight': 3,
                'fillOpacity': 0.3,
                'opacity': 1.0
            },
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"Click for parcel {target_parcel_ref} details"
        ).add_to(m)

        print("✅ Parcel geometry added to map")

    except Exception as e:
        print(f"⚠️ Error adding parcel geometry to map: {e}")
        # Add a simple marker at parcel centroid as fallback
        try:
            centroid = target_parcel_geom.geometry.iloc[0].centroid
            centroid_web = gpd.GeoSeries([centroid], crs=target_parcel_geom.crs).to_crs(web_crs)
            folium.Marker(
                [centroid_web.iloc[0].y, centroid_web.iloc[0].x],
                popup=f"Parcel {target_parcel_ref} (centroid)",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
            print("✅ Added parcel centroid marker as fallback")
        except Exception as e2:
            print(f"⚠️ Could not add parcel marker: {e2}")

    # Add address points with detailed popups
    address_count = 0
    for idx, address in target_addresses_web.iterrows():
        try:
            # Check if geometry exists and is valid
            if pd.isna(address.address_location) or address.address_location is None:
                print(f"⚠️ Skipping address {idx} - no valid geometry")
                continue

            address_count += 1

            # Get coordinates safely
            try:
                lat = address.address_location.y
                lon = address.address_location.x
            except AttributeError:
                print(f"⚠️ Skipping address {address_count} - invalid geometry type: {type(address.address_location)}")
                continue

            # Build detailed popup information with better formatting
            street_name = address.get('street_name', 'Unknown')
            street_number = address.get('street_number', '')
            street_type = address.get('street_type', '')

            # Create nicely formatted address
            full_address = f"{street_type} {street_name} {street_number}".strip()

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <h4 style="margin: 0 0 10px 0; color: #2E86AB;">Address #{address_count}</h4>
                <p style="margin: 5px 0;"><strong>📍 Address:</strong> {full_address}</p>
                <p style="margin: 5px 0;"><strong>🏢 Parcel:</strong> {target_parcel_ref}</p>
                <p style="margin: 5px 0;"><strong>📮 Postal Code:</strong> {address.get('postal_code', 'N/A')}</p>
                <p style="margin: 5px 0;"><strong>🏘️ District:</strong> {address.get('district_name', 'N/A')}</p>
                <p style="margin: 5px 0;"><strong>🗺️ Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
            </div>
            """

            # Create marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Address {address_count}",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)

        except Exception as e:
            print(f"⚠️ Error processing address {idx}: {e}")
            continue

    # Add marker cluster for better visualization if many addresses
    if len(target_addresses_web) > 20:
        print("📍 Adding marker cluster for better visualization...")
        try:
            from folium.plugins import MarkerCluster
            marker_cluster = MarkerCluster().add_to(m)

            cluster_count = 0
            for idx, address in target_addresses_web.iterrows():
                try:
                    if pd.notna(address.address_location) and hasattr(address.address_location, 'y'):
                        lat, lon = address.address_location.y, address.address_location.x

                        # Build the same detailed address info as regular markers
                        street_name = address.get('street_name', 'Unknown')
                        street_number = address.get('street_number', '')
                        street_type = address.get('street_type', '')
                        full_address = f"{street_type} {street_name} {street_number}".strip()

                        # Create detailed popup for clustered marker
                        cluster_popup_html = f"""
                        <div style="font-family: Arial, sans-serif; min-width: 200px;">
                            <h4 style="margin: 0 0 10px 0; color: #17a2b8;">📍 Clustered Address</h4>
                            <p style="margin: 5px 0;"><strong>🏠 Address:</strong> {full_address}</p>
                            <p style="margin: 5px 0;"><strong>🏢 Parcel:</strong> {target_parcel_ref}</p>
                            <p style="margin: 5px 0;"><strong>📮 Postal Code:</strong> {address.get('postal_code', 'N/A')}</p>
                            <p style="margin: 5px 0;"><strong>🏘️ District:</strong> {address.get('district_name', 'N/A')}</p>
                            <p style="margin: 5px 0;"><strong>🗺️ Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                            <p style="margin: 5px 0; font-size: 12px; color: #6c757d;"><em>Part of marker cluster</em></p>
                        </div>
                        """

                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(cluster_popup_html, max_width=350),
                            tooltip=f"📍 {full_address}",
                            icon=folium.Icon(color='lightblue', icon='info-sign')
                        ).add_to(marker_cluster)
                        cluster_count += 1
                except:
                    continue
            print(f"📍 Added {cluster_count} markers to cluster")
        except ImportError:
            print("⚠️ MarkerCluster not available, skipping cluster visualization")

    # Add controls and plugins
    try:
        # Add layer control
        folium.LayerControl().add_to(m)
        print("✅ Added layer control")

        # Add fullscreen button
        try:
            from folium.plugins import Fullscreen
            Fullscreen().add_to(m)
            print("✅ Added fullscreen control")
        except ImportError:
            print("⚠️ Fullscreen plugin not available")

        # Add measure tool
        try:
            from folium.plugins import MeasureControl
            MeasureControl().add_to(m)
            print("✅ Added measure control")
        except ImportError:
            print("⚠️ MeasureControl plugin not available")

        # Add minimap
        try:
            from folium.plugins import MiniMap
            minimap = MiniMap(toggle_display=True)
            m.add_child(minimap)
            print("✅ Added minimap")
        except ImportError:
            print("⚠️ MiniMap plugin not available")

    except Exception as e:
        print(f"⚠️ Error adding map controls: {e}")

    # Save the map with error handling
    output_file = f'/tmp/parcel_{target_parcel_ref}_addresses_interactive.html'
    try:
        m.save(output_file)
        print(f"✅ Interactive map saved to: {output_file}")

        # Verify the file was created and has content
        import os
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ Map file size: {file_size:,} bytes")
            if file_size < 1000:  # Very small file might indicate an error
                print("⚠️ Warning: Map file seems very small, might be corrupted")
        else:
            print("❌ Error: Map file was not created")

    except Exception as e:
        print(f"❌ Error saving map: {e}")
        return None

    # Show statistics
    print(f"\n📊 STATISTICS:")
    print(f"   • Parcel area: {target_parcel_geom.geometry.area.iloc[0]:.2f} m²")
    print(
        f"   • Address density: {len(target_addresses) / (target_parcel_geom.geometry.area.iloc[0] / 10000):.1f} addresses/ha")
    print(f"   • Total assigned addresses: {len(target_addresses)}")
    print(f"   • Map bounds: {parcel_bounds}")

    # Check for address types if available
    if 'address_type' in target_addresses.columns:
        addr_types = target_addresses['address_type'].value_counts()
        print(f"   • Address types: {dict(addr_types)}")

    # Print available columns for debugging
    print(f"   • Available address columns: {list(target_addresses.columns)}")

    return m

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