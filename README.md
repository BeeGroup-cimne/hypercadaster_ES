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

cadaster_codes = ["08900"]
province_codes=None
ine_codes=None

hc.download(
    wd=wd,
    cadaster_codes=cadaster_codes
)

neighborhood_layer=True
postal_code_layer=True
census_layer=True
elevations_layer=True
open_data_layers=True
building_parts_inference=True
building_parts_plots=False
# If you use CAT files, you need to download these datasets by province from this website:
# https://www.sedecatastro.gob.es/Accesos/SECAccDescargaDatos.aspx
# By clicking to: Descarga de información alfanumérica por provincia (formato CAT) 
use_CAT_files=True
CAT_files_rel_dir="CAT_files"

gdf = hc.merge(
    wd=wd, cadaster_codes=cadaster_codes, province_codes=province_codes, ine_codes=ine_codes,
    neighborhood_layer=neighborhood_layer, postal_code_layer=postal_code_layer, census_layer=census_layer,
    elevations_layer=elevations_layer, open_data_layers=open_data_layers,
    building_parts_inference=building_parts_inference, building_parts_plots=building_parts_plots,
    use_CAT_files=use_CAT_files, CAT_files_rel_dir=CAT_files_rel_dir
)
gdf.to_pickle(f"{wd}/{'~'.join(cadaster_codes)}.pkl", compression="gzip")
```

# Install PDAL (optional) and GDAL on your system

Make sure the unstable PPA of Ubuntu GIS is present
```
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt update
# in your venv: hypercadaster_ES
python3 -m pip install setuptools wheel
```

## GDAL installation in Ubuntu 24.04
```
sudo apt install gdal-bin libgdal-dev

# make sure headers are visible (usually not strictly needed, but safe)
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# match the system GDAL version to avoid ABI issues
GDAL_VERSION=$(gdal-config --version)
python3 -m pip install --no-binary=GDAL "GDAL==${GDAL_VERSION}.*"
```

## (optional) PDAL installation in Ubuntu 24.04 without CONDA

Install PDAL (CLI + dev headers)
```

sudo apt install -y pdal libpdal-dev
```
Python bindings
```
# in your venv: hypercadaster_ES
python3 -m pip install pdal
```
Sanity checks
```
pdal --version
pdal info --summary
# in your venv: hypercadaster_ES
python3 -c "import pdal; print('python pdal OK')"
```

## Authors
- Jose Manuel Broto - jmbroto@cimne.upc.edu
- Gerard Mor - gmor@cimne.upc.edu
  
Copyright (c) 2025 Jose Manuel Broto, Gerard Mor
