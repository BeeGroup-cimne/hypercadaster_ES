import hypercadaster_ES as hc
if __name__ == '__main__':
    building_parts_inference = True
    building_parts_plots = False
    building_parts_inference_using_CAT_files = True
    CAT_files_rel_dir = "CAT_files"
    open_data_layers = True
    # wd = "/Users/gmor-air/Nextcloud/Beegroup/data/hypercadaster_ES"  # laptop wd
    wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"  # desktop wd
    cadaster_codes = ['08900'] # Cadaster code for Barcelona municipality
    hc.download(wd=wd, cadaster_codes=cadaster_codes)  # download method
    gdf = hc.merge(wd=wd, cadaster_codes=cadaster_codes, building_parts_inference=True, building_parts_plots=False,
                   building_parts_inference_using_CAT_files = True)  # merge method
    gdf.to_parquet(f"{wd}/results/{'~'.join(cadaster_codes)}.parquet", index=False)  # save to parquet