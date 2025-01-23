import hypercadaster_ES as hc
if __name__ == '__main__':
    # wd = "/Users/gmor-air/Nextcloud/Beegroup/data/hypercadaster_ES"  # laptop wd
    wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"  # desktop wd
    cadaster_codes = ['08900'] # Cadaster code for Barcelona municipality
    hc.download(wd=wd, cadaster_codes=cadaster_codes)  # download method
    gdf = hc.merge(wd=wd, cadaster_codes=cadaster_codes, building_parts_inference=False, building_parts_plots=False,
                   use_CAT_files = True)  # merge method
    gdf.to_pickle(f"{wd}/results/{'~'.join(cadaster_codes)}_no_inference.pkl", compression="gzip")  # save to pickle
