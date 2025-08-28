import hypercadaster_ES as hc

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["25023","25285"]

hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

gdf = hc.merge(
    wd=wd,
    cadaster_codes=cadaster_codes,
    province_codes=None,
    ine_codes=None,
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    building_parts_inference=True,
    building_parts_plots=False,
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"
)
gdf.to_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")

# failing_zones_bcn = [
#     '03735DF3807C', '05192DF3801H', '08169DF3801F', '11859DF3718E', '12078DF3910E', '13182DF3811G', '14063DF3810E',
#     '15923DF3819C', '18654DF3816F', '22043DF3820C', '22157DF3821E', '22343DF3823C', '23157DF3821E', '23166DF3821E',
#     '25281DF3822H', '25423DF3824D', '26474DF3824H', '27737DF2837A', '28394DF3823H', '28833DF3828D', '29646DF3826D',
#     '31326DF3833C', '31458DF3834E', '44546DF2845C', '47518DF2845B', '49494DF3844H', '54525DF2855A', '59243DF2852D',
#     '65192DF2861H', '75688DF2876H', '81041DF2880C', '83526DF2885A', '84654DF2786E', '86789DF2887H', '88235DF2882D',
#     '92827DF2898C', '93655DF2796E', 'unknown'
# ]