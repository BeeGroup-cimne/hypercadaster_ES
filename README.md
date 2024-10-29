# hypercadaster_ES

This is a Python library to ingest the most updated Spanish cadaster data regarding addresses, parcels and buildings,
Digital Elevation Model datasets, and multiple administrative divisions layers. In addition, we can obtain a single
GeoPandas dataframe with all the cadaster data joined by buildings with the attributes related to administrative layers
and height above sea level.

## How to install the library?
1- Download the repository in your own system: git clone
2- Create library: python setup.py sdist
2- Install library: pip install dist/hypercadaster_ES-0.0.1.tar.gz

## How to use it?
```
import hypercadaster_ES as hc

hc.download()
gdf = hc.merge()
```

## Authors
- Jose Manuel Broto - jmbroto@cimne.upc.edu
- Gerard Mor - gmor@cimne.upc.edu
  
Copyright (c) 2024 Jose Manuel Broto, Gerard Mor
