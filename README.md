# hypercadaster_ES

This is a Python library to ingest the most updated Spanish cadaster data regarding addresses, parcels and buildings,
Digital Elevation Model datasets, and multiple administrative divisions layers. In addition, we can obtain a single
GeoPandas dataframe with all the cadaster data joined by buildings with the attributes related to administrative layers
and height above sea level.

## How to install the library?
1. Download the repository in your own system: git clone
2. Use the virtualenv where you want to install the library
3. Install setuptools: pip install setuptools
4. Create the library: python setup.py sdist
5. Install library: pip install dist/hypercadaster_es-1.0.0.tar.gz

## How to use it?
```
import hypercadaster_ES as hc
wd = "<introduce the directory where the data will be downloaded>"

hc.download(
    wd=wd, 
    cadaster_codes=cadaster_codes
)
gdf = hc.merge(
    wd=wd, 
    cadaster_codes=cadaster_codes, 
    building_parts_inference=False, 
    building_parts_plots=False,
    use_CAT_files=True
)
gdf.to_pickle(f"{wd}/{'~'.join(cadaster_codes)}_no_inference.pkl", compression="gzip")
```

## Authors
- Jose Manuel Broto - jmbroto@cimne.upc.edu
- Gerard Mor - gmor@cimne.upc.edu
  
Copyright (c) 2025 Jose Manuel Broto, Gerard Mor
