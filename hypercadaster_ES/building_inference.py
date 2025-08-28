from hypercadaster_ES import utils
from hypercadaster_ES import downloaders
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import multiprocessing
from datetime import date
from itertools import chain
from itertools import zip_longest
import tqdm
import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from shapely.ops import linemerge
from shapely.geometry.polygon import orient
from shapely.geometry import (Polygon, LineString, LinearRing, MultiPolygon, MultiLineString)
import math
from tqdm import tqdm
import polars as pl
import warnings
import re
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(
    "ignore",
    message=re.escape("FigureCanvasAgg is non-interactive, and thus cannot be shown")
)
# from joblib import Parallel, delayed
# from joblib.externals.loky import get_reusable_executor

def process_building_parts(code, building_part_gdf_, buildings_CAT, parcels_gdf, results_dir=None, cadaster_dir=None,
                           open_street_dir=None, plots=False, orientation_discrete_interval_in_degrees=5,
                           num_workers=max(1, math.ceil(multiprocessing.cpu_count()/3))):

    if (not os.path.exists(f"{results_dir}/{code}_sbr_results.pkl") and
        not os.path.exists(f"{results_dir}/{code}_br_results.pkl")):

        os.makedirs(f"{results_dir}/cache", exist_ok=True)
        if plots:
            if results_dir is None:
                results_dir = "results"
            os.makedirs(f"{results_dir}/plots", exist_ok=True)

        if (not os.path.exists(f"{results_dir}/cache/{code}_aux1.pkl")):
            gdf_global = utils.detect_close_buildings_parallel(gdf_building_parts=building_part_gdf_,
                                                         buffer_neighbours=80,
                                                         neighbours_column_name="nearby_buildings",
                                                         neighbours_id_column_name="building_reference",
                                                         num_workers=-1,
                                                         column_name_to_split="zone_reference")
            parcels_gdf["zone_reference"] = (
                parcels_gdf["building_reference"].str[0:5] + parcels_gdf["building_reference"].str[7:14])
            gdf_parcels_global = utils.detect_close_parcels_parallel(gdf_parcels=parcels_gdf,
                                                             buffer_neighbours=150,
                                                             neighbours_column_name="nearby_parcels",
                                                             neighbours_id_column_name="building_reference",
                                                             num_workers=-1,
                                                             column_name_to_split="zone_reference")
            gdf_global = gdf_global.merge(
                gdf_parcels_global[["building_reference","nearby_parcels"]],
                on="building_reference", how="left")
            gdf_global = gdf_global.set_geometry("building_part_geometry")
            gdf_global.to_pickle(f"{results_dir}/cache/{code}_aux1.pkl")
        else:
            gdf_global = pd.read_pickle(f"{results_dir}/cache/{code}_aux1.pkl")

        if (not os.path.exists(f"{results_dir}/cache/{code}_aux2.pkl")):
            gdf_footprints_global = utils.calculate_floor_footprints(
                gdf=gdf_global,
                group_by="building_reference",
                geometry_name="building_part_geometry",
                only_exterior_geometry=True,
                min_hole_area=0.5,
                gap_tolerance=0.05,
                chunk_size=500,
                num_workers=-1)
            gdf_footprints_global.to_pickle(f"{results_dir}/cache/{code}_aux2.pkl")
        else:
            gdf_footprints_global = pd.read_pickle(f"{results_dir}/cache/{code}_aux2.pkl")


        grouped = gdf_global.groupby("zone_reference")
        zone_references = list(grouped.groups.keys())

        ine_code = utils.cadaster_to_ine_codes(cadaster_dir, [code])[0]
        streets_gdf = utils.get_municipality_open_street_maps(
            open_street_dir=open_street_dir,
            query_location=f"{utils.municipality_name(ine_code)}, Spain",
            crs=gdf_global.crs)
        streets_gdf = streets_gdf[streets_gdf.geometry.apply(lambda geom: isinstance(geom, LineString))]

        # Define a wrapper for joblib's delayed processing
        def process_zone_delayed(zone_reference):
            #zone_reference='00112DF3801A'#"00329DF3803A"#"00054DF3800E"
            gdf_zone = grouped.get_group(zone_reference).reset_index(drop=True)
            return process_zone(
                gdf_zone,
                zone_reference,
                gdf_footprints_global,
                parcels_gdf,
                streets_gdf,
                buildings_CAT,
                results_dir,
                plots,
                orientation_discrete_interval_in_degrees
            )

        # Parallel processing for each zone
        results = []
        for zone_reference in tqdm(zone_references, desc="Inferring geometrical KPIs for each building..."):
            results.append(process_zone_delayed(zone_reference))
            # with utils.tqdm_joblib(tqdm(total=len(zone_references), desc="Inferring geometrical KPIs for each building...")):
        #     results = Parallel(n_jobs=num_workers)(
        #         delayed(process_zone_delayed)(zone_reference) for zone_reference in zone_references
        #     )
        # get_reusable_executor().shutdown(wait=True)

        # del streets_gdf
        # del gdf_global
        # del gdf_footprints_global
        # del grouped

        # Concatenate all results into a single DataFrame
        sbr_results = pd.concat([i[0] for i in results if i is not None])[results[0][0].columns]
        br_results = pd.concat([i[1] for i in results if i is not None])[results[0][1].columns]
        del results

        sbr_results.to_pickle(f"{results_dir}/{code}_sbr_results.pkl",
                              compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        br_results.to_pickle(f"{results_dir}/{code}_br_results.pkl",
                             compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})

    else:
        sbr_results = pd.read_pickle(f"{results_dir}/{code}_sbr_results.pkl", compression="gzip")
        br_results = pd.read_pickle(f"{results_dir}/{code}_br_results.pkl", compression="gzip")

    return sbr_results, br_results


def process_zone(gdf_zone, zone_reference, gdf_footprints_global, parcels_gdf, streets_gdf, buildings_CAT,
                 results_dir, plots, orientation_discrete_interval_in_degrees):

    try:
        if (not os.path.exists(f"{results_dir}/cache/{zone_reference}_sbr_results.pkl") and
            not os.path.exists(f"{results_dir}/cache/{zone_reference}_br_results.pkl")):

            # PDF setup for the zone if pdf_plots is True
            if plots:
                pdf = PdfPages(f"{results_dir}/plots/zone_{zone_reference}.pdf")

            # Calculation of the single building references and adjacent buildings
            gdf_zone = utils.detect_close_buildings(gdf_building_parts = gdf_zone,
                                   buffer_neighbours = 0.5,
                                   neighbours_column_name = "adjacent_buildings",
                                   neighbours_id_column_name = "single_building_reference",
                                   geometry_variable="building_part_geometry")

            # Calculation of the single building references and adjacent buildings
            gdf_zone = utils.detect_close_buildings(gdf_building_parts=gdf_zone,
                                              buffer_neighbours=1,
                                              neighbours_column_name="adjacent_parcels",
                                              neighbours_id_column_name="single_building_reference",
                                              geometry_variable="parcel_geometry")

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries(gdf_zone.building_part_geometry,
                                        labels=gdf_zone.n_floors_above_ground,
                                        clusters=gdf_zone.single_building_reference,
                                        ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            gdf_aux_footprints = utils.calculate_floor_footprints(
                gdf=gdf_zone,
                group_by="single_building_reference",
                geometry_name="building_part_geometry",
                min_hole_area=0.5,
                gap_tolerance=0.05,
                num_workers=1)

            gdf_aux_footprints_global = utils.calculate_floor_footprints(
                gdf=gdf_zone,
                geometry_name="building_part_geometry",
                min_hole_area=0.5,
                gap_tolerance=0.05,
                num_workers=1)

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries(gdf_aux_footprints.geometry, clusters=gdf_aux_footprints.group, ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            # Detect all the patios in a zone
            patios_detected = {}
            for floor in range(gdf_aux_footprints.floor.max() + 1):
                patios_detected[floor] = (
                    utils.get_all_patios(gdf_aux_footprints_global[gdf_aux_footprints_global['floor']==floor].geometry))

            unique_patios_detected = utils.unique_polygons(list(itertools.chain(*patios_detected.values())),tolerance=0.1)

            if plots:
                fig, ax = plt.subplots()
                utils.plot_shapely_geometries([i.exterior for i in gdf_zone.building_part_geometry] + unique_patios_detected, ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            # Close the PDF file for the current zone
            if plots:
                pdf.close()

            results_ = []

            for single_building_reference, building_gdf_item in gdf_zone.groupby('single_building_reference'):
                # grouped2 = gdf_zone.groupby("single_building_reference")
                # single_building_reference = list(grouped2.groups.keys())[0]
                # building_gdf_item = grouped2.get_group(single_building_reference).reset_index().drop(['index'], axis=1).copy()

                # PDF setup for the building if plots is True
                if plots:
                    pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}.pdf")

                # Obtain a general geometry of the building, considering all floors.
                building_geom = utils.union_geoseries_with_tolerance(building_gdf_item['building_part_geometry'],
                                                               gap_tolerance=0.05, resolution=16)
                parcel_geom = utils.union_geoseries_with_tolerance(building_gdf_item["parcel_geometry"],
                                                             gap_tolerance=0.5, resolution=0.5)

                # Extract and clean the set of neighbouring buildings
                adjacent_buildings_set = {id for ids in building_gdf_item['adjacent_buildings'] for id in ids.split(",")}
                adjacent_buildings_set.discard(
                    single_building_reference)  # Safely remove the single_building_reference itself if present
                adjacent_buildings = sorted(adjacent_buildings_set)

                # Extract and clean the set of nearby buildings
                nearby_buildings_set = {id for ids in building_gdf_item['nearby_buildings'] for id in ids.split(",")}
                nearby_buildings_set.discard(
                    single_building_reference.split("_")[0])  # Safely remove the single_building_reference itself
                nearby_buildings = sorted(nearby_buildings_set)

                # Extract and clean the set of nearby parcels
                nearby_parcels_set = {id for ids in building_gdf_item['nearby_parcels'] for id in ids.split(",")}
                nearby_parcels_set.discard(
                    single_building_reference.split("_")[0]) # Safely remove the single_building_reference itself
                nearby_parcels = sorted(nearby_parcels_set)

                # Extract and clean the set of adjacent parcels
                adjacent_parcels_set = {id for ids in building_gdf_item['adjacent_parcels'] for id in ids.split(",")}
                adjacent_parcels_set.discard(
                    single_building_reference)  # Safely remove the single_building_reference itself if present
                adjacent_parcels = sorted(adjacent_parcels_set)

                # Is there any premises in ground floor?
                premises = False
                premises_activity_typologies = []
                premises_names = []
                premises_last_revision = []
                if "number_of_ground_premises" in building_gdf_item.columns:
                    if (~np.isfinite(building_gdf_item.iloc[0]["number_of_ground_premises"]) and
                            building_gdf_item.iloc[0]["number_of_ground_premises"] > 0):
                        premises = True
                        premises_activity_typologies = building_gdf_item.iloc[0]["ground_premises_activities"]
                        premises_names = building_gdf_item.iloc[0]["ground_premises_names"]
                        premises_last_revision = building_gdf_item.iloc[0]["ground_premises_last_revision"]

                # Filter building in study
                gdf_aux_footprint_building = gdf_aux_footprints[gdf_aux_footprints['group'] == single_building_reference]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(list(gdf_aux_footprint_building.geometry), ax=ax,
                                            title = "Building in analysis")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related buildings footprint
                gdf_aux_footprints_ = gdf_aux_footprints[gdf_aux_footprints['group'].isin(adjacent_buildings)]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(gdf_aux_footprints_.geometry, clusters=gdf_aux_footprints_.group, ax=ax,
                                            contextual_geometry = building_geom,
                                            title = "Adjacent buildings")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related buildings footprint
                gdf_aux_footprints_nearby = gdf_footprints_global[gdf_footprints_global['group'].isin(nearby_buildings)]
                if plots:
                    fig, ax = plt.subplots()
                    utils.plot_shapely_geometries(gdf_aux_footprints_nearby.geometry, clusters=gdf_aux_footprints_nearby.group,
                                            ax=ax, contextual_geometry=building_geom,
                                            title="Nearby buildings")
                    pdf.savefig(fig)
                    plt.close(fig)

                # Filter related parcels
                related_parcels_geom = []
                nearby_related_parcels_geom = []
                related_union = None
                if len(adjacent_parcels)>0:
                    related_parcels_geom = gdf_zone.loc[gdf_zone["single_building_reference"].isin(adjacent_parcels), "parcel_geometry"]
                    related_parcels_geom = [utils.union_geoseries_with_tolerance(item,gap_tolerance=0.05, resolution=2) for item
                                            in list(related_parcels_geom.drop_duplicates())]
                if len(nearby_parcels)>0:
                    nearby_related_parcels_geom = parcels_gdf.loc[parcels_gdf['building_reference'].isin(nearby_parcels), "parcel_geometry"]
                    nearby_related_parcels_geom = [utils.union_geoseries_with_tolerance(item, gap_tolerance=0.05, resolution=2) for item
                       in list(nearby_related_parcels_geom.drop_duplicates())]

                # Detect which is the part of the parcel in contact with the street
                filtered_streets = streets_gdf[streets_gdf.geometry.distance(parcel_geom.centroid) <= 200]
                filtered_streets.reset_index(inplace=True)
                filtered_streets = filtered_streets[filtered_streets.element == "way"]

                if isinstance(parcel_geom, MultiPolygon):
                    parcel_limits = [walls_i for walls_i in list(parcel_geom)[0].geoms]
                else:
                    parcel_limits = [parcel_geom]

                all_parcel_segments = []
                touching_street_and_non_adjacent_parcel_segments = []
                non_adjacent_parcel_segments = []

                for geom in parcel_limits:
                    filtered_streets = filtered_streets[~filtered_streets.geometry.intersects(geom)]
                    exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
                    #utils.plot_points_with_indices(exterior_coords)
                    for i in range(len(exterior_coords) - 1):
                        segment = LineString([exterior_coords[i], exterior_coords[i + 1]])
                        # utils.plot_polygons_group(related_parcels_geom,
                        #                     utils.shorten_linestring(segment, 0.1).buffer(2, cap_style="flat"),
                        #                     pdf_file="test2.pdf")
                        all_parcel_segments.append(segment)
                        if (len(related_parcels_geom)==0 or
                            not np.any([related_parcel_geom.intersects(
                                utils.shorten_linestring(segment, 0.1).buffer(2,cap_style="flat")) for
                            related_parcel_geom in related_parcels_geom])):
                            if utils.is_segment_in_contact_with_street(
                                    segment, streets = filtered_streets, max_distance=8, orientation_tolerance=15,
                                    orientation_discrete_interval_in_degrees=orientation_discrete_interval_in_degrees):
                                touching_street_and_non_adjacent_parcel_segments.append(segment)
                            else:
                                non_adjacent_parcel_segments.append(segment)
                street_parcel_limits = {"by_orientation": {}}

                if len(touching_street_and_non_adjacent_parcel_segments) > 0:
                    street_parcel_limits_segments = touching_street_and_non_adjacent_parcel_segments
                elif len(non_adjacent_parcel_segments) > 0:
                    street_parcel_limits_segments = non_adjacent_parcel_segments
                else:
                    street_parcel_limits_segments = all_parcel_segments

                for segment in street_parcel_limits_segments:
                    segment_orientation = str(utils.calculate_wall_outdoor_normal_orientation(
                            segment,
                            orientation_interval=orientation_discrete_interval_in_degrees))
                    if segment_orientation in street_parcel_limits["by_orientation"]:
                        street_parcel_limits["by_orientation"][segment_orientation] += segment.length
                    else:
                        street_parcel_limits["by_orientation"][segment_orientation] = segment.length
                if utils.detect_number_of_orientations(street_parcel_limits["by_orientation"])==1:
                    street_parcel_limits["main_orientation"] = utils.weighted_circular_mean(
                        degrees=[float(i) for i in street_parcel_limits["by_orientation"].keys()],
                        weights=[float(i) for i in street_parcel_limits["by_orientation"].values()])
                else:
                    main_orientation = max(street_parcel_limits["by_orientation"].items(), key=lambda item: item[1])[0]
                    street_parcel_limits["main_orientation"] = float(main_orientation)

                if plots:
                    pdf2 = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_streets.pdf")
                    fig, ax = plt.subplots(figsize=(15,15))
                    utils.plot_shapely_geometries(geometries=list(filtered_streets.geometry), ax=ax,
                                            contextual_geometry=[parcel_geom],
                                            contextual_geometry_options={
                                                "linestyle": "solid",
                                                "linewidth": 1.0,
                                                "color": "red",
                                                "alpha": 1,
                                            })
                    pdf2.savefig(fig)
                    fig, ax = plt.subplots(figsize=(15,15))
                    utils.plot_shapely_geometries(geometries=list(filtered_streets.geometry), ax=ax,
                                            contextual_geometry=street_parcel_limits_segments,
                                            contextual_geometry_options={
                                                "linestyle": "solid",
                                                "linewidth": 1.0,
                                                "color": "red",
                                                "alpha": 1,
                                            })
                    pdf2.savefig(fig)
                    plt.close(fig)
                    pdf2.close()

                floor_area = {}
                underground_floor_area = {}
                roof_area = {}
                patios_area = {}
                patios_n = {}
                perimeter = {}
                air_contact_wall = {}
                shadows_at_distance = {}
                parcel_at_distance = {}
                adiabatic_wall = {}
                patios_wall = {}
                significant_orientations = []
                significant_orientations_by_floor = {}
                n_floors = gdf_aux_footprint_building.floor.max() if gdf_aux_footprint_building.floor.max() >= 0 else np.nan
                n_underground_floors = -gdf_aux_footprint_building.floor.min() if gdf_aux_footprint_building.floor.min() < 0 else 0
                n_buildings = len(
                    gdf_zone[
                            gdf_zone["building_reference"] == building_gdf_item.iloc[0]["building_reference"]
                        ]["single_building_reference"].unique()
                )

                if n_underground_floors > 0:

                    for underground_floor in range(1,(n_underground_floors+1)):
                        underground_floor_area[(underground_floor)] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == -(underground_floor)]['geometry'].area.sum(),
                            2
                        )

                if n_floors is not np.nan:

                    for floor in range(n_floors + 1):
                        floor_area[floor] = round(
                            gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                            2
                        )
                        if floor == n_floors:
                            roof_area[floor] = round(
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry'].area.sum(),
                                2
                            )
                        else:
                            roof_area[floor] = round(
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor][
                                    'geometry'].area.sum() -
                                gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == (floor + 1)][
                                    'geometry'].area.sum(),
                                2
                            )
                        patios = utils.patios_in_the_building(
                            patios_geoms=patios_detected[floor],
                            building_geom=building_geom,
                            tolerance=0.8
                        )
                        patios_n[floor] = len(patios)
                        patios_area[floor] = round(sum([patio.area for patio in patios]), 2)
                        walls = gdf_aux_footprint_building[gdf_aux_footprint_building["floor"] == floor]['geometry']
                        if isinstance(list(walls)[0], MultiPolygon):
                            walls = [walls_i for walls_i in list(walls)[0].geoms]
                        elif isinstance(list(walls)[0], Polygon):
                            walls = list(walls)

                        # Initialize the lengths for each type of wall contact and for each orientation
                        air_contact_wall[floor] = {direction: 0.0 for direction in [str(i) for i in
                                                                                    list(range(0, 360, orientation_discrete_interval_in_degrees))]}
                        adiabatic_wall[floor] = 0.0
                        patios_wall[floor] = 0.0
                        perimeter[floor] = round(sum([peri.exterior.length for peri in walls]), 2)

                        for geom in walls:

                            # Indoor patios
                            patios_wall[floor] += sum([item.length for item in list(geom.interiors)])

                            # Break down the exterior into segments
                            exterior_coords = list(orient(geom, sign=-1.0).exterior.coords)
                            for i in range(len(exterior_coords) - 1):
                                segment_assigned = False
                                segment = LineString([exterior_coords[i], exterior_coords[i + 1]])
                                if segment.length < 0.001:
                                    continue

                                # Determine the orientation of this segment
                                segment_orientation = str(utils.calculate_wall_outdoor_normal_orientation(
                                    segment,
                                    orientation_interval=orientation_discrete_interval_in_degrees))

                                # Check if the segment is in contact with patios
                                for patio in patios:
                                    if utils.segment_intersects_with_tolerance(
                                            segment, patio,
                                            buffer_distance=0.1,
                                            area_percentage_threshold=15
                                    ):
                                        patios_wall[floor] += round(segment.length, 2)
                                        if plots:
                                            fig, ax = plt.subplots()
                                            utils.plot_shapely_geometries(
                                                geometries = [geom] + [segment],
                                                title = f"Wall ID:{i}, floor: {floor},\n"
                                                        f"orientation: {segment_orientation},"
                                                        f"type: patio",
                                                ax = ax,
                                                contextual_geometry = building_geom)
                                            pdf.savefig(fig)
                                            plt.close(fig)
                                        segment_assigned = True
                                        break

                                # Check if the segment is in contact with nearby buildings
                                if not segment_assigned:
                                    for aux_geom in gdf_aux_footprints_[gdf_aux_footprints_.floor == floor].geometry:
                                        if utils.segment_intersects_with_tolerance(
                                                segment, aux_geom,
                                                buffer_distance=0.1,
                                                area_percentage_threshold=15
                                        ):
                                            adiabatic_wall[floor] += round(segment.length, 2)
                                            if plots:
                                                fig, ax = plt.subplots()
                                                utils.plot_shapely_geometries(
                                                    geometries = [geom] + [segment],
                                                    title = f"Wall ID: {i}, floor: {floor},\n"
                                                            f"orientation: {segment_orientation}, "
                                                            f"type: adiabatic",
                                                    ax = ax,
                                                    contextual_geometry = building_geom)
                                                pdf.savefig(fig)
                                                plt.close(fig)
                                            segment_assigned = True
                                            break

                                # Check if the segment is in contact with outdoor (air contact)
                                if not segment_assigned:
                                    air_contact_wall[floor][segment_orientation] += round(segment.length, 2)
                                    if plots:
                                        fig, ax = plt.subplots()
                                        utils.plot_shapely_geometries(
                                            geometries = [geom] + [segment],
                                            title = f"Wall ID: {i}, floor: {floor},\n"
                                                    f"orientation: {segment_orientation}, "
                                                    f"type: air contact",
                                            ax = ax,
                                            contextual_geometry = building_geom)
                                        pdf.savefig(fig)
                                        plt.close(fig)
                    for floor, air_contact_walls in air_contact_wall.items():
                        significant_orientations_ = []
                        significant_threshold = 0.1 * perimeter[floor]

                        for orientation, wall_length in air_contact_walls.items():
                            if wall_length > significant_threshold:
                                significant_orientations_.append(int(orientation))

                        # Sort the significant orientations in ascending order for better readability
                        significant_orientations_ = sorted(list(set(significant_orientations_)))
                        significant_orientations.extend(significant_orientations_)
                        significant_orientations_by_floor[floor] = significant_orientations_
                    significant_orientations = list(set(significant_orientations))

                    # Close the PDF file for the current building
                    if plots:
                        pdf.close()

                    # if building_type == "Non-residential":
                    #     starting_residential_floor = np.nan
                    # elif premises:
                    #     starting_residential_floor = 1
                    #     if building_type == "Multi-family" and n_floors >= n_dwellings:
                    #         n_dwellings = n_dwellings - 1
                    # else:
                    #     if n_floors > 6 and building_type == "Multi-family":
                    #         starting_residential_floor = 1
                    #     else:
                    #         starting_residential_floor = 0
                    # floor_area_with_possible_residential_use = [floor_area[floor] if floor >= starting_residential_floor or starting_residential_floor is np.nan else 0.0 for floor in range(n_floors + 1)]

                if max(gdf_aux_footprint_building.floor.max(), gdf_aux_footprints_nearby.floor.max()) > 0:

                    # Shadows depending orientation
                    if plots:
                        pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_shadows.pdf")
                    else:
                        pdf = None
                    for floor in range(max(gdf_aux_footprint_building.floor.max(),
                                           gdf_aux_footprints_nearby.floor.max()) + 1):
                        shadows_at_distance[floor] = utils.distance_from_centroid_to_polygons_by_orientation(
                            geom=building_geom.exterior if isinstance(building_geom, Polygon) else (
                                ([LinearRing(polygon.exterior.coords) for polygon in building_geom.geoms])),
                            other_polygons=gdf_aux_footprints_nearby[gdf_aux_footprints_nearby.floor == floor].geometry,
                            centroid=building_geom.centroid,
                            orientation_interval=orientation_discrete_interval_in_degrees,
                            plots=plots,
                            pdf=pdf,
                            floor=floor,
                            max_distance=150
                        )
                    if plots:
                        pdf.close()

                if len(nearby_related_parcels_geom)>0:
                    # Street widths
                    if plots:
                        pdf = PdfPages(f"{results_dir}/plots/building_{single_building_reference}_street_widths.pdf")
                    else:
                        pdf = None
                    parcel_at_distance = utils.distance_from_points_to_polygons_by_orientation(
                        geom=street_parcel_limits_segments,
                        other_polygons=nearby_related_parcels_geom,
                        num_points_per_geom_element=10,
                        orientations_to_analyse=sorted(street_parcel_limits["by_orientation"].keys(), key=int),
                        force_one_orientation_per_geom_element=True,
                        plots=plots,
                        pdf=pdf,
                        max_distance=150
                    )
                    if plots:
                        pdf.close()

                street_parcel_limits["next_building_by_orientation"] = {}
                if len(street_parcel_limits["by_orientation"]) > 0 and len(shadows_at_distance)>0:
                    for i in street_parcel_limits["by_orientation"].keys():
                        street_parcel_limits["next_building_by_orientation"][i] = shadows_at_distance[0]["shadows"][i]

                street_parcel_limits["next_building_main_orientation"] = np.inf
                if street_parcel_limits["main_orientation"] is not np.nan and len(shadows_at_distance)>0:
                    street_parcel_limits["next_building_main_orientation"] = shadows_at_distance[0]["shadows"][
                        str(utils.discrete_orientation(
                            street_parcel_limits["main_orientation"],
                            orientation_discrete_interval_in_degrees))
                    ]

                street_parcel_limits["street_width_by_orientation"] = parcel_at_distance[0]
                filtered = {k: v for k, v in street_parcel_limits["street_width_by_orientation"].items() if np.isfinite(v)}
                if not filtered:
                    street_parcel_limits["street_width_main_orientation"] = np.inf
                else:
                    street_parcel_limits["main_orientation"] = max(filtered, key=filtered.get)
                    street_parcel_limits["street_width_main_orientation"] = parcel_at_distance[0][
                            max(filtered, key=filtered.get)]

                # Store results
                results_.append({
                    'building_reference': single_building_reference.split("_")[0],
                    'single_building_reference': single_building_reference,
                    'zone_reference': zone_reference,
                    'sbr__n_buildings': n_buildings,
                    'sbr__detached': (np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0) < 1.0,
                    'br__parcel_orientations': street_parcel_limits["by_orientation"],
                    'br__parcel_main_orientation': street_parcel_limits["main_orientation"],
                    'br__next_building_by_orientation': street_parcel_limits["next_building_by_orientation"],
                    'br__next_building_main_orientation': street_parcel_limits["next_building_main_orientation"],
                    'br__street_width_by_orientation': street_parcel_limits["street_width_by_orientation"],
                    'br__street_width_main_orientation': street_parcel_limits["street_width_main_orientation"],
                    'br__exists_ground_commercial_premises': premises,
                    'br__ground_commercial_premises_names': premises_names,
                    'br__ground_commercial_premises_typology': premises_activity_typologies,
                    'br__ground_commercial_premises_last_revision': premises_last_revision,
                    'sbr__floors_above_ground': n_floors+1,
                    'sbr__floors_below_ground': n_underground_floors,
                    'sbr__below_ground_built_area_by_floor': [underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)] if n_underground_floors > 0 else [],
                    'sbr__below_ground_built_area': np.sum([underground_floor_area[floor] for floor in range(1,n_underground_floors + 1)]) if n_underground_floors > 0 else 0.0,
                    'sbr__above_ground_built_area_by_floor': [floor_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                    'sbr__above_ground_built_area': np.sum([floor_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                    'sbr__above_ground_roof_area_by_floor': [roof_area[floor] for floor in range(n_floors + 1)] if n_floors is not np.nan else [],
                    'sbr__above_ground_roof_area': np.sum([roof_area[floor] for floor in range(n_floors + 1)]) if n_floors is not np.nan else 0.0,
                    'sbr__building_footprint_area': round(building_geom.area if isinstance(building_geom, Polygon)
                                                     else sum(polygon.area for polygon in building_geom.geoms) if isinstance(building_geom, MultiPolygon)
                                                     else 0.0, 2),
                    'sbr__building_footprint_geometry': building_geom.exterior if isinstance(building_geom, Polygon) else (
                                MultiLineString([LineString(polygon.exterior.coords) for polygon in building_geom.geoms])),
                    'sbr__building_footprint_by_floor': gdf_aux_footprint_building[["group","floor","geometry"]].to_dict(index=False, orient='split')['data'],
                    'sbr__patios_area_by_floor': [patios_area[floor] for floor in range(n_floors + 1)] if patios_area != {} else [],
                    'sbr__patios_number_by_floor': [patios_n[floor] for floor in range(n_floors + 1)] if patios_n != {} else [],
                    'sbr__walls_between_slabs': np.sum([perimeter[floor] for floor in
                                                         range(n_floors + 1)] if perimeter != {} else []),
                    'sbr__building_perimeter_by_floor': [perimeter[floor] for floor in range(n_floors + 1)] if perimeter != {} else [],
                    'sbr__air_contact_wall_by_floor': {key: [air_contact_wall[d][key] for d in air_contact_wall] for key in
                                         air_contact_wall[0]} if air_contact_wall != {} else {},
                    'sbr__air_contact_wall':  {key: np.sum([air_contact_wall[d][key] for d in air_contact_wall]) for key in
                                         air_contact_wall[0]} if air_contact_wall != {} else {},
                    'sbr__air_contact_wall_significant_orientations': significant_orientations,
                    'sbr__air_contact_wall_significant_orientations_by_floor': significant_orientations_by_floor,
                    'sbr__adiabatic_wall_by_floor': [adiabatic_wall[floor] for floor in range(n_floors + 1)] if adiabatic_wall != {} else [],
                    'sbr__adiabatic_wall': np.sum([adiabatic_wall[floor] for floor in range(n_floors + 1)]) if adiabatic_wall != {} else 0.0,
                    'sbr__patios_wall_by_floor': [patios_wall[floor] for floor in range(n_floors + 1)] if patios_wall != {} else [],
                    'sbr__patios_wall_total': np.sum([patios_wall[floor] for floor in range(n_floors + 1)]) if patios_wall != {} else 0.0,
                    'sbr__shadows_at_distance': {key: [shadows_at_distance[d]['shadows'][key] for d in shadows_at_distance] for key in
                                            shadows_at_distance[0]['shadows']} if shadows_at_distance != {} else {},
                    'sbr__building_contour_at_distance': {key: np.mean([shadows_at_distance[d]['contour'][key] for d in shadows_at_distance]) for key in
                                            shadows_at_distance[0]['contour']} if shadows_at_distance != {} else {}
                })

            if len(results_) > 0:
                results_ = pd.DataFrame(results_)
                sbr_results_concat = pd.DataFrame()
                br_results_concat = pd.DataFrame()
                for building_reference in results_.building_reference.unique():
                    # building_reference = "0004701DF3800C"#"9505401DF2890F"

                    sbr_results = results_[results_.building_reference == building_reference]
                    br_agg_dict = {
                        'building_reference': [building_reference],
                        'br__detached': [np.any(sbr_results['sbr__detached'])],
                        'br__parcel_orientations': [sbr_results.iloc[0]['br__parcel_orientations']],
                        'br__parcel_main_orientation': [sbr_results.iloc[0]['br__parcel_main_orientation']],
                        'br__street_width_by_orientation': [sbr_results.iloc[0]['br__street_width_by_orientation']],
                        'br__street_width_main_orientation': [sbr_results.iloc[0]['br__street_width_main_orientation']],
                        'br__next_building_by_orientation': [sbr_results.iloc[0]['br__next_building_by_orientation']],
                        'br__next_building_main_orientation': [sbr_results.iloc[0]['br__next_building_main_orientation']],
                        'br__exists_ground_commercial_premises': [sbr_results.iloc[0]['br__exists_ground_commercial_premises']],
                        'br__ground_commercial_premises_names': [sbr_results.iloc[0]['br__ground_commercial_premises_names']],
                        'br__ground_commercial_premises_typology': [sbr_results.iloc[0]['br__ground_commercial_premises_typology']],
                        'br__ground_commercial_premises_last_revision': [sbr_results.iloc[0]['br__ground_commercial_premises_last_revision']],
                        'br__floors_above_ground': [max(sbr_results['sbr__floors_above_ground'])],
                        'br__floors_below_ground': [max(sbr_results['sbr__floors_below_ground'])],
                        'br__below_ground_built_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__below_ground_built_area_by_floor'], fillvalue=0)]],
                        'br__below_ground_built_area': [sum(sbr_results['sbr__below_ground_built_area'])],
                        'br__above_ground_built_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__above_ground_built_area_by_floor'], fillvalue=0)]],
                        'br__above_ground_built_area': [sum(sbr_results['sbr__above_ground_built_area'])],
                        'br__above_ground_roof_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__above_ground_roof_area_by_floor'], fillvalue=0)]],
                        'br__above_ground_roof_area': [sum(sbr_results['sbr__above_ground_roof_area'])],
                        'br__building_footprint_area': [sum(sbr_results['sbr__building_footprint_area'])],
                        'br__building_footprint_geometry': [linemerge(
                            [line for geom in sbr_results['sbr__building_footprint_geometry'].tolist() for line in
                             (geom.geoms if geom.geom_type == "MultiLineString" else [geom])])],
                        'br__building_footprint_by_floor':
                            [[item for sublist in sbr_results['sbr__building_footprint_by_floor'] for item in sublist]],
                        'br__patios_area_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_area_by_floor'], fillvalue=0)]],
                        'br__patios_number_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_number_by_floor'], fillvalue=0)]],
                        'br__walls_between_slabs': [sum(sbr_results['sbr__walls_between_slabs'])],
                        'br__building_perimeter_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__building_perimeter_by_floor'], fillvalue=0)]],
                        'br__air_contact_wall_by_floor': [dict(sorted({
                            k: [
                                sum(x)
                                for x in zip_longest(
                                    *(
                                        d.get(
                                            k,
                                            [0] * (
                                                max(
                                                    (len(v) for v in d.values() if isinstance(v, list)),
                                                    default=1,  # Default to 1 if no valid lists are found
                                                )
                                            )
                                        )
                                        for d in sbr_results["sbr__air_contact_wall_by_floor"]
                                    ),
                                    fillvalue=0,
                                )
                            ]
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__air_contact_wall_by_floor"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__air_contact_wall': [dict(sorted({
                            k: sum(
                                d.get(k, 0)  # Get scalar value or default to 0
                                for d in sbr_results["sbr__air_contact_wall"]
                            )
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__air_contact_wall"]
                                )
                            )
                        }.items(),key=lambda item: int(item[0])))],
                        'br__air_contact_wall_significant_orientations': [
                            dict(sorted({
                                k: sum(
                                    d.get(k, 0)
                                    for d in sbr_results["sbr__air_contact_wall_significant_orientations"]
                                    if isinstance(d, dict)
                                )
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__air_contact_wall_significant_orientations"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__air_contact_wall_significant_orientations_by_floor': [
                            dict(sorted({
                                k: [
                                    sum(floor_vals)
                                    for floor_vals in zip_longest(
                                        *[
                                            d.get(k, []) for d in sbr_results["sbr__air_contact_wall_significant_orientations_by_floor"]
                                            if isinstance(d, dict)
                                        ],
                                        fillvalue=0
                                    )
                                ]
                                for k in set(
                                    chain.from_iterable(
                                        d.keys()
                                        for d in sbr_results["sbr__air_contact_wall_significant_orientations_by_floor"]
                                        if isinstance(d, dict)
                                    )
                                )
                            }.items(), key=lambda item: int(item[0])))
                        ],
                        'br__adiabatic_wall_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__adiabatic_wall_by_floor'], fillvalue=0)]],
                        'br__adiabatic_wall': [sum(sbr_results['sbr__adiabatic_wall'])],
                        'br__patios_wall_by_floor': [[sum(values) for values in zip_longest(
                            *sbr_results['sbr__patios_wall_by_floor'], fillvalue=0)]],
                        'br__patios_wall_total': [sum(sbr_results['sbr__patios_wall_total'])],
                        'br__shadows_at_distance': [dict(sorted({
                            k: [
                                np.mean(x)
                                for x in zip_longest(
                                    *(
                                        d.get(
                                            k,
                                            [0] * (
                                                max(
                                                    (len(v) for v in d.values() if isinstance(v, list)),
                                                    default=1,  # Default to 1 if no valid lists are found
                                                )
                                            )
                                        )
                                        for d in sbr_results["sbr__shadows_at_distance"]
                                    ),
                                    fillvalue=0,
                                )
                            ]
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__shadows_at_distance"]
                                )
                            )
                        }.items(), key=lambda item: int(item[0])))],
                        'br__building_contour_at_distance': [dict(sorted({
                            k: np.mean(
                                list(
                                    d.get(k, 0)  # Get scalar value or default to 0
                                    for d in sbr_results["sbr__building_contour_at_distance"]
                                )
                            )
                            for k in set(
                                chain.from_iterable(
                                    d.keys() for d in sbr_results["sbr__building_contour_at_distance"]
                                )
                            )
                        }.items(),
                        key=lambda item: int(item[0])))]
                    }
                    br_results = pd.DataFrame(br_agg_dict)
                    geoms_by_floor = list(sbr_results['sbr__building_footprint_by_floor'])
                    building_gdf_by_floor = pd.concat([gpd.GeoDataFrame(geom_by_floor) for geom_by_floor in geoms_by_floor])

                    if len(building_gdf_by_floor)>0:

                        building_gdf_by_floor.columns = ["single_building_reference", "floor_number", "geometry"]
                        building_gdf_by_floor = building_gdf_by_floor.set_geometry("geometry")
                        building_gdf_by_floor["area_single_building_reference"] = building_gdf_by_floor.area
                        building_aggregates = (
                            building_gdf_by_floor.groupby("floor_number").agg(
                                n_single_building_references=("single_building_reference", "nunique"),
                                area_building_reference=("area_single_building_reference", "sum")
                            ))
                        building_gdf_by_floor = building_gdf_by_floor.merge(building_aggregates, on="floor_number", how="left")
                        building_gdf_by_floor["building_reference"] = building_reference

                        # Consider the information from CAT files
                        if buildings_CAT is not None:

                            # Integrate all CAT file information
                            agg_CAT = aggregate_CAT_file_building_spaces(
                                building_CAT_df=buildings_CAT.filter(pl.col("building_reference") == building_reference))

                            building_gdf_by_floor = building_gdf_by_floor.drop(columns="geometry")

                            # Reduce agg_CAT["by_floors"] to each single building reference
                            sbr_with_CAT = building_gdf_by_floor.merge(agg_CAT["by_floors"].to_pandas(),
                                                                            on=["floor_number", "building_reference"])
                            def process_variable(df, variable, ops, multiplier=1, divider=1):
                                return {
                                    f"sbr__{variable}_by_floor": utils.agg_op(
                                        df,
                                        funcs=ops,
                                        grouping=["use_type","floor_number"],
                                        variable=f"br__{variable}_by_floor",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }
                            agg_list = [("building_spaces",[lambda x: round(x, 0)],"area_single_building_reference","area_building_reference"),
                                        ("area_with_communals",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("area_without_communals",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("communal_area",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("economic_value",[lambda x: round(x, 2)],"area_single_building_reference","area_building_reference"),
                                        ("mean_building_space_area_with_communals",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_area_without_communals",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_communal_area",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_effective_year",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_category",[lambda x: round(x, 0)],1,1),
                                        ("mean_building_space_economic_value",[lambda x: round(x, 0)],1,1),
                                        ]
                            sbr_with_CAT_by_floors = (
                                sbr_with_CAT.groupby(["single_building_reference","building_reference"])
                                .apply(lambda df: pd.Series({
                                        key: value
                                        for agg_elem in agg_list
                                        for key, value in
                                        process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3]).items()
                                    }),
                                    include_groups=False)
                                .reset_index()
                            )

                            # Reduce agg_CAT["building"] to each single building reference
                            sbr_with_CAT = building_gdf_by_floor.merge(agg_CAT["building"].to_pandas(),
                                                                            on=["building_reference"])
                            def process_variable(df, variable, summarise_op=None, digits=0, multiplier=1, divider=1):
                                return {
                                    f"sbr__{variable}": utils.agg_op(
                                        df,
                                        funcs=[summarise_op, lambda x: round(x, digits) if summarise_op is not None else
                                            [lambda x: round(x, digits)]],
                                        grouping="use_type",
                                        variable=f"br__{variable}",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }
                            agg_list = [("building_spaces",sum,0,"area_single_building_reference","area_building_reference"),
                                        ("area_with_communals",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("area_without_communals",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("communal_area",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("economic_value",sum,2,"area_single_building_reference","area_building_reference"),
                                        ("mean_building_space_area_with_communals",np.mean,0,1,1),
                                        ("mean_building_space_area_without_communals",np.mean,0,1,1),
                                        ("mean_building_space_communal_area",np.mean,0,1,1),
                                        ("mean_building_space_effective_year",np.mean,0,1,1),
                                        ("mean_building_space_category",np.mean,0,1,1),
                                        ("mean_building_space_economic_value",np.mean,0,1,1)
                                        ]
                            sbr_with_CAT_summarised = (
                                sbr_with_CAT.groupby(["single_building_reference","building_reference"])
                                .apply(lambda df: pd.Series({
                                    key: value
                                    for agg_elem in agg_list
                                    for key, value in process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3],
                                                                       agg_elem[4]).items()
                                }),
                                include_groups=False)
                                .reset_index()
                            )

                            # Reduce agg_CAT["by_floors"] to each building reference
                            def process_variable(df, variable, summarise_ops=None, multiplier=1, divider=1):
                                return {
                                    f"br__{variable}_by_floor": utils.agg_op(
                                        df,
                                        funcs=summarise_ops,
                                        grouping=["use_type","floor_number"],
                                        variable=f"br__{variable}_by_floor",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }

                            agg_list = [("building_spaces", [lambda x: round(x, 2)], 1, 1),
                                        ("area_with_communals", [lambda x: round(x, 2)], 1, 1),
                                        ("area_without_communals", [lambda x: round(x, 2)], 1, 1),
                                        ("communal_area", [lambda x: round(x, 2)], 1, 1),
                                        ("economic_value", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_area_with_communals", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_area_without_communals", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_communal_area", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_effective_year", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_category", [lambda x: round(x, 2)], 1, 1),
                                        ("mean_building_space_economic_value", [lambda x: round(x, 2)], 1, 1)
                                        ]
                            br_with_CAT_by_floors = (
                                agg_CAT["by_floors"].to_pandas().groupby(["building_reference"])
                                .apply(lambda df: pd.Series({
                                        key: value
                                        for agg_elem in agg_list
                                        for key, value in
                                        process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3]).items()
                                    }),
                                    include_groups=False)
                                .reset_index()
                            )

                            # Reduce agg_CAT["building"] to each building reference
                            def process_variable(df, variable, ops, multiplier=1, divider=1):
                                return {
                                    f"br__{variable}": utils.agg_op(
                                        df,
                                        funcs=ops,
                                        grouping="use_type",
                                        variable=f"br__{variable}",
                                        multiplier=multiplier,
                                        divider=divider
                                    )
                                }
                            agg_list = [("building_spaces",[sum],1,1),
                                        ("area_with_communals",[sum],1,1),
                                        ("area_without_communals",[sum],1,1),
                                        ("communal_area",[sum],1,1),
                                        ("economic_value",[sum],1,1),
                                        ("mean_building_space_area_with_communals",[sum],1,1),
                                        ("mean_building_space_area_without_communals",[sum],1,1),
                                        ("mean_building_space_communal_area",[sum],1,1),
                                        ("mean_building_space_effective_year",[sum],1,1),
                                        ("mean_building_space_category",[sum],1,1),
                                        ("mean_building_space_economic_value",[sum],1,1),
                                        ('building_spaces_reference',[chain.from_iterable, list],1,1),
                                        ('building_spaces_postal_address',[chain.from_iterable, list],1,1),
                                        ('building_spaces_inner_address',[chain.from_iterable, list],1,1),
                                        ('building_spaces_floor_number',[chain.from_iterable, list],1,1),
                                        ('building_spaces_category',[chain.from_iterable, list],1,1),
                                        ('building_spaces_effective_year',[chain.from_iterable, list],1,1),
                                        ('building_spaces_area_without_communal',[chain.from_iterable, list],1,1),
                                        ('building_spaces_area_with_communal',[chain.from_iterable, list],1,1),
                                        ('building_spaces_economic_value',[chain.from_iterable, list],1,1),
                                        ('building_spaces_detailed_use_type',[chain.from_iterable, list],1,1)
                                        ]
                            br_with_CAT_summarised = (
                                agg_CAT["building"].to_pandas().groupby(["building_reference"])
                                .apply(lambda df: pd.Series({
                                    key: value
                                    for agg_elem in agg_list
                                    for key, value in process_variable(df, agg_elem[0], agg_elem[1], agg_elem[2], agg_elem[3]).items()
                                }),
                                       include_groups=False)
                                .reset_index()
                            )

                            sbr_results = (sbr_results.
                                merge(sbr_with_CAT_by_floors, on=["building_reference","single_building_reference"], how="left").
                                merge(sbr_with_CAT_summarised, on=["building_reference","single_building_reference"], how="left"))

                            br_results = (br_results.
                                merge(br_with_CAT_by_floors, on="building_reference", how="left").
                                merge(br_with_CAT_summarised, on="building_reference", how="left"))

                    sbr_results_concat = pd.concat([sbr_results_concat, sbr_results], ignore_index=True)
                    br_results_concat = pd.concat([br_results_concat, br_results], ignore_index=True)

                sbr_results_concat.to_pickle(f"{results_dir}/cache/{zone_reference}_sbr_results.pkl")
                br_results_concat.to_pickle(f"{results_dir}/cache/{zone_reference}_br_results.pkl")

                return sbr_results_concat, br_results_concat

            else:
                return None, None

        else:
            return (pd.read_pickle(f"{results_dir}/cache/{zone_reference}_sbr_results.pkl"),
                    pd.read_pickle(f"{results_dir}/cache/{zone_reference}_br_results.pkl"))

    except Exception as e:
        print(f" This cadastral zone failed: '{zone_reference}'. Error: {e}", file=sys.stderr)

def parse_horizontal_division_buildings_CAT_files(cadaster_code, CAT_files_dir):

    combined_dfs = downloaders.parse_CAT_file(cadaster_code, CAT_files_dir, allowed_dataset_types=[14, 15])

    # Join the building space and building spaces detailed datasets. Order the items by floor.
    building_spaces_detailed = combined_dfs[14]
    building_spaces = combined_dfs[15]
    building_spaces = building_spaces.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference"),
        pl.concat_str(['space2_reference', 'space3_reference'], separator=""
                      ).alias("building_space_reference_last_digits")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.concat_str(['building_reference', 'space1_reference'], separator=""
                      ).alias("building_space_reference")
    )
    building_spaces_detailed = building_spaces_detailed.filter(
        pl.col("distribution_method_for_communal_areas") == "")
    floor_names = sorted(building_spaces_detailed["building_space_floor_name"].unique().to_list())
    df = pd.DataFrame({'floor_name': floor_names})
    df['order'] = df['floor_name'].apply(utils.classify_cadaster_floor_names)
    floor_names_sorted = list(df.sort_values(by='order').reset_index(drop=True).floor_name)

    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces.select(
            'building_space_reference', 'building_space_reference_last_digits', 'building_space_year',
            'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
            'building_space_total_area', 'building_space_participation_rate', 'building_space_use_type'),
        on = "building_space_reference").select(
        'building_reference', 'building_space_reference', 'building_space_reference_last_digits',
        'street_type', 'street_name', 'street_number1', 'street_letter1', 'street_number2', 'street_letter2', 'km',
        'building_space_block_name', 'building_space_stair_name', 'building_space_floor_name', 'building_space_door_name',
        'building_space_year', 'retrofitted', 'building_space_retroffiting_year', 'building_space_effective_year',
        'building_space_total_area', 'building_space_area_without_communal', 'building_space_area_balconies_terraces',
        'building_space_area_imputable_to_other_floors', 'building_space_participation_rate',
        'building_space_use_type', 'building_space_detailed_use_type', 'building_space_typology'
    )
    order_mapping = {value: index for index, value in enumerate(floor_names_sorted)}
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_floor_name").map_elements(lambda x: order_mapping.get(x, -1),
                                                         return_dtype=pl.Int32).alias("custom_sort")
    )
    building_spaces_detailed = building_spaces_detailed.sort("custom_sort").drop("custom_sort").sort("building_space_reference")

    # Calculate areas and affected communal areas per building space
    building_spaces_detailed = building_spaces_detailed.join(
        building_spaces_detailed.group_by("building_space_reference").agg(
            pl.col("building_space_area_without_communal").sum().alias("building_space_total_area_without_communal"),
            pl.len().alias("building_spaces_considered"),
        ),
        on = "building_space_reference"
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_participation_rate") / 1000000).alias("building_space_participation_rate"),
        pl.concat_str(['building_space_reference', 'building_space_reference_last_digits'], separator=""
                      ).alias("building_space_reference"),
        ((pl.col("building_space_total_area") - pl.col("building_space_total_area_without_communal")) *
         pl.col("building_space_area_without_communal") / pl.col("building_space_total_area_without_communal")).alias(
            "building_space_communal_area"))
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_communal_area") + pl.col("building_space_area_without_communal")).alias(
            "building_space_area_with_communal")
    )

    # Rename items of categorical columns (Typology, category, use...)
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_typology").str.tail(1).alias("building_space_typology_category"),
        pl.col("building_space_typology").str.head(4).alias("building_space_typology_id"),
        (pl.lit(date.today().year) - pl.col("building_space_effective_year")).alias("building_space_age")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.col("building_space_typology_id").replace(
            {k: v.get("Use") for k, v in downloaders.building_space_typologies.items() if "Use" in v}).
            alias("building_space_typology_use"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClass") for k, v in downloaders.building_space_typologies.items() if "UseClass" in v}).
            alias("building_space_typology_use_class"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseClassModality") for k, v in downloaders.building_space_typologies.items() if "UseClassModality" in v}).
            alias("building_space_typology_use_class_modality"),
        pl.col("building_space_typology_id").replace(
            {k: v.get("UseLevel") for k, v in downloaders.building_space_typologies.items() if "UseLevel" in v}).
            alias("building_space_typology_use_level"),
        pl.when(pl.col("building_space_age") >= 90).
            then(pl.lit(90).cast(pl.Int16).cast(pl.Utf8)).
            otherwise(((pl.col("building_space_age") / 5).floor() * 5).cast(pl.Int16).cast(pl.Utf8)).
            alias("building_space_age_key"),
        pl.col("building_space_use_type").replace(downloaders.building_space_use_types).
            alias("building_space_use_type"),
        pl.col("building_space_detailed_use_type").replace(downloaders.building_space_detailed_use_types).
            alias("building_space_detailed_use_type")
    )
    building_space_age_value_dict = \
        {str(entry["Age"][0]): {k: v for k, v in entry.items() if k != "Age"} for entry in downloaders.building_space_age_value}

    # Economical value coefficients of the constructions
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.struct(["building_space_typology_id", "building_space_typology_category"]).map_elements(
            lambda row:
                downloaders.building_space_typologies[
                    row["building_space_typology_id"]]["ConstructionValue"][
                    int(row["building_space_typology_category"]) - 1], # Adjust index to match zero-based indexing
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("construction_relative_economic_value"),
        pl.struct(["building_space_age_key", "building_space_typology_use_level",
                   "building_space_typology_category"]).map_elements(
            lambda row:
            building_space_age_value_dict[
                row["building_space_age_key"]][
                row["building_space_typology_use_level"]][
                int(row["building_space_typology_category"]) - 1],  # Adjust index to match zero-based indexing
            return_dtype=pl.Float64  # Specify the return dtype explicitly
        ).alias("age_correction_relative_economic_value")
    )

    # Calculate relative economic value
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("construction_relative_economic_value") * pl.col("age_correction_relative_economic_value")).alias("building_space_relative_economic_value")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.col("building_space_relative_economic_value") * pl.col("building_space_area_with_communal")).alias("building_space_economic_value")
    )

    # Calculate the complete address
    building_spaces_detailed = building_spaces_detailed.with_columns(
        (pl.when(pl.col("street_number1") == "0000").then(pl.lit("")).otherwise(
            pl.col("street_number1").cast(pl.Int16))).alias("street_number1"),
        (pl.when(pl.col("street_number2") == "0000").then(pl.lit("")).otherwise(
            pl.col("street_number2").cast(pl.Int16))).alias("street_number2"),
        (pl.when(pl.col("km") == "00000").then(pl.lit("")).otherwise(pl.col("km").cast(pl.Int16))).alias("km")
    )
    building_spaces_detailed = building_spaces_detailed.with_columns(
        pl.concat_str(["street_type", "street_name", "street_number1", "street_letter1", "street_number2",
                       "street_letter2", "km", "building_space_block_name", "building_space_stair_name"],
                      separator=" ").
        str.strip_chars().str.replace_all(r"\s+", " ").alias("building_space_address"),
        pl.concat_str(['building_space_block_name', 'building_space_stair_name', 'building_space_floor_name',
                       'building_space_door_name'],
                      separator=" ").
        str.strip_chars().str.replace_all(r"\s+", " ").alias("building_space_inner_address")
    )

    use_type_mapping = {
        "Restaurant 4 Stars": "Leisure and Hospitality",
        "Professional education": "Cultural",
        "Water treatment plants": "Singular building",  # or whichever fallback you prefer
        "Other office activities": "Offices",
        "Weaver's office": "Offices",
        "Sports facilities": "Sports facilities",
        "Bookstore commerce": "Commercial",
        "Food industry": "Industrial",
        "Parking in a household": "Warehouse - Parking",
        "Commerce": "Commercial",
        "Religious": "Religious",
        "Open terrace (100%)": "Urbanization and landscaping works, undeveloped land",
        "Public town hall (<20,000)": "Offices",
        "Hotel Cafe 3 Stars": "Leisure and Hospitality",
        "Hostel Standard 1": "Leisure and Hospitality",
        "Hotel 5 Stars": "Leisure and Hospitality",
        "Station": "Singular building",
        "Financial commerce": "Commercial",
        "Club": "Entertainment venues",
        "Printing commerce": "Commercial",
        "Restaurant 5 Stars": "Leisure and Hospitality",
        "Musician's office": "Offices",
        "Hotel Cafe 5 Stars": "Leisure and Hospitality",
        "Luxury apartments 3 Stars": "Residential",
        "Clinic": "Healthcare and Charity",
        "Swimming pool": "Sports facilities",
        "University education": "Cultural",
        "Other provincial union": "Singular building",
        "Sports complex": "Sports facilities",
        "Stadium": "Sports facilities",
        "Education": "Cultural",
        "Luxury apartments 1 Star": "Residential",
        "Personal/Home commerce": "Commercial",
        "Other outpatient clinic": "Healthcare and Charity",
        "Nursing office": "Healthcare and Charity",
        "Hostel Standard 3": "Leisure and Hospitality",
        "Wholesale commerce": "Commercial",
        "Jewelry commerce": "Commercial",
        "Hotel 4 Stars": "Leisure and Hospitality",
        "Public courthouse": "Offices",
        "Education (Library)": "Cultural",
        "Silos, solid storage": "Warehouse - Parking",
        "Museum education": "Cultural",
        "Liquid storage tanks": "Warehouse - Parking",
        "Porch (100%)": "Urbanization and landscaping works, undeveloped land",
        "Galleries commerce": "Commercial",
        "Casino (>20,000)": "Entertainment venues",
        "Shoe commerce": "Commercial",
        "Machinery industry": "Industrial",
        "Wood industry": "Industrial",
        "Petroleum industry": "Industrial",
        "Parking": "Warehouse - Parking",
        "Transformer hut": "Singular building",
        "Warehouse": "Warehouse - Parking",
        "Other uses": "Singular building",
        "Bazaar commerce": "Commercial",
        "Provincial court": "Offices",
        "Industry": "Industrial",
        "Other dispensary": "Healthcare and Charity",
        "Textile industry": "Industrial",
        "Metal industry": "Industrial",
        "Other rescue facilities": "Singular building",
        "Plumbing commerce": "Commercial",
        "Public town hall (>20,000)": "Offices",
        "Entertainment": "Entertainment venues",
        "Chemical industry": "Industrial",
        "Casino (<20,000)": "Entertainment venues",
        "Drugstore commerce": "Commercial",
        "Colonnade (50%)": "Urbanization and landscaping works, undeveloped land",
        "Agent office": "Offices",
        "Construction industry": "Industrial",
        "Pharmacy commerce": "Commercial",
        "Government building": "Offices",
        "Undeveloped land": "Urbanization and landscaping works, undeveloped land",
        "Gas storage": "Warehouse - Parking",
        "Butcher commerce": "Commercial",
        "Daycare": "Healthcare and Charity",
        "Residence": "Residential",
        "Institute education": "Cultural",
        "Medical/Law office": "Offices",
        "Other sanatorium": "Healthcare and Charity",
        "Furniture commerce": "Commercial",
        "Public": "Singular building",
        "Hotel 1 Star": "Leisure and Hospitality",
        "Hotel": "Leisure and Hospitality",
        "Religious chapel": "Religious",
        "Hostel Standard 2": "Leisure and Hospitality",
        "Housing": "Residential",
        "Restaurant 2 Stars": "Leisure and Hospitality",
        "Cinema": "Entertainment venues",
        "Hotel Cafe 2 Stars": "Leisure and Hospitality",
        "Restaurant 1 Star": "Leisure and Hospitality",
        "Retail commerce": "Commercial",
        "Hotel 3 Stars": "Leisure and Hospitality",
        "Entertainment hall": "Entertainment venues",
        "Auditorium": "Entertainment venues",
        "Salesperson office": "Offices",
        "Sports": "Sports facilities",
        "Hotel 2 Stars": "Leisure and Hospitality",
        "Electric industry": "Industrial",
        "Office": "Offices",
        "Watchmaking commerce": "Commercial",
        "Hotel Cafe 4 Stars": "Leisure and Hospitality",
        "Theater": "Entertainment venues",
        "Private garden (100%)": "Urbanization and landscaping works, undeveloped land",
        "Hotel Cafe 1 Star": "Leisure and Hospitality",
        "Restaurant 3 Stars": "Leisure and Hospitality",
        "Religious hermitage": "Religious",
        "Basic education": "Cultural",
        "Urbanization works": "Urbanization and landscaping works, undeveloped land",
        "Perfumery commerce": "Commercial",
        "Automobile commerce": "Commercial",
        "Hospital": "Healthcare and Charity",
        "Religious parish": "Religious",
        "Glass industry": "Industrial",
        "Public delegation": "Offices",
        "Fabric commerce": "Commercial",
        "Storage": "Warehouse - Parking",
        "Supermarket commerce": "Commercial",
        "Manufacturing industry": "Industrial",
        "Covered terrace (100%)": "Urbanization and landscaping works, undeveloped land",
        "Cultural house education": "Cultural",
    }
    mapper = pl.DataFrame([{'keys': x, 'values': y} for x, y in use_type_mapping.items()])

    building_spaces_detailed = building_spaces_detailed.with_columns(
        building_spaces_detailed["building_space_detailed_use_type"].
        to_frame("keys").join(mapper, on="keys", how="left").
        to_series(1).alias("building_space_inferred_use_type"))

    return building_spaces_detailed

    # rc = building_spaces_detailed.filter((building_spaces_detailed["building_floor_name"] == 'CUB'))["building_reference"][0]
    # result = (building_spaces.filter(building_spaces["building_reference"] == rc)[
    #     ["building_reference","building_stair_name","building_space_reference","building_floor_name","building_door_name","building_space_area"]].join(
    #     building_spaces_detailed.filter(building_spaces_detailed["building_reference"] == rc)[
    #         ["building_space_reference","building_stair_name","building_floor_name", "building_door_name", "building_space_area_without_communal","building_space_detailed_use_type"]],
    #     on="building_space_reference").sort("building_space_reference").join(
    #         pl.DataFrame({
    #             "building_reference": rc,
    #             "area": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["building_part_geometry"].area,
    #             "n_floors_above_ground": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["n_floors_above_ground"],
    #             "n_floors_below_ground": building_part_gdf_[
    #                 building_part_gdf_["building_reference"] == rc
    #                 ]["n_floors_below_ground"],
    #         }).select(
    #             pl.col("building_reference").first(),
    #             pl.col("n_floors_above_ground").max().alias("max_floors_above"),
    #             pl.col("n_floors_below_ground").max().alias("max_floors_below"),
    #         ),
    #         on="building_reference"))
    # result["building_floor_name_right"].to_list()
    # result.write_csv("joined_spaces.csv",quote_style='always')


def aggregate_CAT_file_building_spaces(building_CAT_df):
    building_CAT_df = building_CAT_df.with_columns(
        pl.col("building_space_floor_name").map_elements(
            lambda x: utils.classify_above_ground_floor_names(x) if x is not None else None,
            return_dtype=pl.Int32
        ).alias("building_space_floor_order_above_ground"),
        pl.col("building_space_floor_name").map_elements(
            lambda x: utils.classify_below_ground_floor_names(x) if x is not None else None,
            return_dtype=pl.Int32
        ).alias("building_space_floor_order_below_ground")
    )

    # Map the original Series to ordered values
    unique_values = building_CAT_df[
        "building_space_floor_order_above_ground"].unique().drop_nulls().sort()
    value_to_order = {val: i for i, val in enumerate(unique_values)}
    building_CAT_df = building_CAT_df.with_columns(
        building_CAT_df["building_space_floor_order_above_ground"]
        .map_elements(lambda x: value_to_order.get(x, None), return_dtype=pl.Int32)
        .alias("building_space_floor_order_above_ground")
    )
    unique_values = building_CAT_df[
        "building_space_floor_order_below_ground"].unique().drop_nulls().sort()
    value_to_order = {val: -(i + 1) for i, val in enumerate(unique_values)}
    building_CAT_df = building_CAT_df.with_columns(
        building_CAT_df["building_space_floor_order_below_ground"]
        .map_elements(lambda x: value_to_order.get(x, None), return_dtype=pl.Int32)
        .alias("building_space_floor_order_below_ground")
    )

    # Join floor orders below and above ground
    building_CAT_df = building_CAT_df.with_columns(
        pl.when(pl.col("building_space_floor_order_above_ground").is_null()).
        then(pl.col("building_space_floor_order_below_ground")).
        otherwise(pl.col("building_space_floor_order_above_ground")).alias(
            "building_space_floor_order"
        )
    )

    def adjust_to_consecutive_floor(series):
        unique_sorted = series.unique().sort()
        unique_sorted = unique_sorted.drop_nulls()
        value_to_consecutive = {val: idx + unique_sorted[0] for idx, val in enumerate(unique_sorted)}
        return series.map_elements(lambda x: value_to_consecutive[x], return_dtype=pl.Int32)

    adjusted_floors = building_CAT_df.group_by("building_space_address").agg(
        [
            pl.col("building_space_floor_name"),
            pl.col("building_space_floor_order").map_elements(adjust_to_consecutive_floor,
                                                              return_dtype=pl.List(pl.Int32)).alias(
                "building_space_floor_number")
        ]
    )
    adjusted_floors = (
        adjusted_floors.explode(["building_space_floor_name", "building_space_floor_number"]).
        unique().sort(["building_space_address", "building_space_floor_number"]))
    building_CAT_df = building_CAT_df.join(adjusted_floors,
                                           on=["building_space_address", "building_space_floor_name"])

    # Compute total areas of the buildings
    building_CAT_df = building_CAT_df.join(
        building_CAT_df.group_by("building_space_floor_number").agg(
            pl.col("building_space_area_with_communal").sum().alias("building_floor_total_area")
        ), on=["building_space_floor_number"])
    building_CAT_df = building_CAT_df.join(
        building_CAT_df.group_by("building_reference").agg(
            pl.col("building_space_area_with_communal").sum().alias("building_total_area")
        ), on=["building_reference"])
    building_CAT_df = building_CAT_df.with_columns(
        pl.col("building_space_area_without_communal").round(2).alias(
            "building_space_area_without_communal"),
        pl.col("building_space_area_with_communal").round(2).alias("building_space_area_with_communal"),
        pl.col("building_space_economic_value").round(2).alias("building_space_economic_value"),
    )

    # Sort the dataset
    building_CAT_df = building_CAT_df.sort(
        ["building_space_address", "building_space_inner_address"]
    )

    agg_CAT_by_floors = (
        building_CAT_df.group_by("building_space_floor_number", "building_space_inferred_use_type").agg(
            pl.col("building_reference").first().alias("building_reference"),
            pl.col("building_space_reference").count().alias("br__building_spaces_by_floor"),
            # Average values per building space
            pl.col("building_space_area_with_communal").mean().round(2).alias(
                "br__mean_building_space_area_with_communals_by_floor"),
            pl.col("building_space_area_without_communal").mean().round(2).alias(
                "br__mean_building_space_area_without_communals_by_floor"),
            pl.col("building_space_communal_area").mean().round(2).alias(
                "br__mean_building_space_communal_area_by_floor"),
            pl.col("building_space_effective_year").mean().round(1).alias(
                "br__mean_building_space_effective_year_by_floor"),
            pl.col("building_space_typology_category").
                replace({"A":"1","B":"2","C":"3","0":"1","O":"1"}).
                cast(pl.Int8).mean().round(1).alias(
                    "br__mean_building_space_category_by_floor"),
            pl.col("building_space_economic_value").mean().round(2).alias(
                "br__mean_building_space_economic_value_by_floor"),
            # Total value for the whole building
            pl.col("building_space_area_with_communal").sum().alias("br__area_with_communals_by_floor"),
            pl.col("building_space_area_without_communal").sum().alias(
                "br__area_without_communals_by_floor"),
            pl.col("building_space_communal_area").sum().alias("br__communal_area_by_floor"),
            pl.col("building_space_economic_value").sum().alias("br__economic_value_by_floor"),
            ((pl.col("building_space_area_with_communal").sum() / pl.col(
                "building_floor_total_area").mean()).round(2) * 100).alias(
                "br__relative_area_with_communals_by_floor"),
            ((pl.col("building_space_area_without_communal").sum() / pl.col(
                "building_floor_total_area").mean()).round(2) * 100).alias(
                "br__relative_area_without_communals_by_floor"),
            ((pl.col("building_space_communal_area").sum() / pl.col(
                "building_floor_total_area").mean()).round(2) * 100).alias(
                "br__relative_communal_area_by_floor")
        ))
    agg_CAT_by_floors = agg_CAT_by_floors.rename(
        {"building_space_floor_number": "floor_number", "building_space_inferred_use_type": "use_type"})
    agg_CAT_by_floors = agg_CAT_by_floors.sort("floor_number")

    agg_CAT_building = building_CAT_df.group_by("building_space_inferred_use_type").agg(
        pl.col("building_reference").first().alias("building_reference"),
        pl.col("building_space_reference").count().alias("br__building_spaces"),
        pl.col("building_space_floor_number").unique().sort().alias("br__floors"),
        # Average values per building space
        pl.col("building_space_area_with_communal").mean().round(2).alias(
            "br__mean_building_space_area_with_communals"),
        pl.col("building_space_area_without_communal").mean().round(2).alias(
            "br__mean_building_space_area_without_communals"),
        pl.col("building_space_communal_area").mean().round(2).alias(
            "br__mean_building_space_communal_area"),
        pl.col("building_space_effective_year").mean().round(1).alias(
            "br__mean_building_space_effective_year"),
        pl.col("building_space_typology_category").
            replace({"A":"1","B":"2","C":"3","0":"1","O":"1"}).
            cast(pl.Int8).mean().round(1).alias(
            "br__mean_building_space_category"),
        pl.col("building_space_economic_value").mean().round(2).alias(
            "br__mean_building_space_economic_value"),
        # Total value for the whole building
        pl.col("building_space_area_with_communal").sum().alias("br__area_with_communals"),
        pl.col("building_space_area_without_communal").sum().alias(
            "br__area_without_communals"),
        pl.col("building_space_communal_area").sum().alias("br__communal_area"),
        pl.col("building_space_economic_value").sum().alias("br__economic_value"),
        ((pl.col("building_space_area_with_communal").sum() / pl.col(
            "building_total_area").mean()).round(2) * 100).alias(
            "br__relative_area_with_communals"),
        ((pl.col("building_space_area_without_communal").sum() / pl.col(
            "building_total_area").mean()).round(2) * 100).alias(
            "br__relative_area_without_communals"),
        ((pl.col("building_space_communal_area").sum() / pl.col("building_total_area").mean()).round(
            2) * 100).alias("br__relative_communal_area"),
        # Arrays of values per building
        pl.col("building_space_reference").alias("br__building_spaces_reference"),
        pl.col("building_space_address").alias("br__building_spaces_postal_address"),
        pl.col("building_space_inner_address").alias("br__building_spaces_inner_address"),
        pl.col("building_space_floor_number").alias("br__building_spaces_floor_number"),
        (pl.col("building_space_typology_category").
            replace({"A":"1","B":"2","C":"3","0":"1","O":"1"}).
            cast(pl.Int8).alias("br__building_spaces_category")),
        pl.col("building_space_effective_year").alias("br__building_spaces_effective_year"),
        pl.col("building_space_area_without_communal").alias(
            "br__building_spaces_area_without_communal"),
        pl.col("building_space_area_with_communal").alias("br__building_spaces_area_with_communal"),
        pl.col("building_space_economic_value").alias("br__building_spaces_economic_value"),
        pl.col("building_space_detailed_use_type").alias("br__building_spaces_detailed_use_type")
    )
    agg_CAT_building = agg_CAT_building.rename(
        {"building_space_inferred_use_type": "use_type"})

    return {
        "by_floors": agg_CAT_by_floors,
        "building": agg_CAT_building
    }
