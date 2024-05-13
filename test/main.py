from streetmaps import download, merge

if __name__ == '__main__':
    wd = "../dataa"  # working directory
    ine_codes = ["08019"]  # ine Code, only one in the first library version
    download(wd=wd, ine_codes=ine_codes)  # download method
    streetmap_gdf = merge(wd=wd, ine_codes=ine_codes)  # merge method
    streetmap_gdf.to_csv(f"{wd}/inspire/streetmap_{ine_codes[0]}.csv", index=False)  # save to csv
