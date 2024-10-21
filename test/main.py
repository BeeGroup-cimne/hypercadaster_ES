import hypercadaster_ES as hc

if __name__ == '__main__':
    wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"  # working directory
    cadaster_codes = ['08900'] # Cadaster code for Barcelona municipality
    hc.download(wd=wd, cadaster_codes=cadaster_codes)  # download method
    gdf = hc.merge(wd=wd, cadaster_codes=cadaster_codes, building_parts_inference=True, building_parts_plots=False)  # merge method
    gdf.to_parquet(f"{wd}/results/{"~".join(cadaster_codes)}.parquet", index=False)  # save to parquet

