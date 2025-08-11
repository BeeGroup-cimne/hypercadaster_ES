import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.0'
PACKAGE_NAME = 'hypercadaster_ES'
AUTHOR = 'Jose Manuel Broto Vispe'
AUTHOR_EMAIL = 'jmbrotovispe@gmail.com'
URL = 'https://github.com/BeeGroup-cimne'

LICENSE = 'MIT'
DESCRIPTION = 'Python library to obtain the Spanish cadaster data joined with external attributes.'
LONG_DESCRIPTION = (HERE / "README.md").read_text(
    encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    "affine==2.4.0",
    "attrs==23.2.0",
    "beautifulsoup4==4.12.3",
    "certifi==2024.2.2",
    "charset-normalizer==3.3.2",
    "click==8.1.7",
    "click-plugins==1.1.1",
    "cligj==0.7.2",
    "cloudpickle==3.1.0",
    "contourpy==1.2.1",
    "cycler==0.12.1",
    "dask==2024.10.0",
    "et-xmlfile==1.1.0",
    "fastkml==0.12",
    "fiona==1.9.6",
    "fonttools==4.53.1",
    "fsspec==2024.10.0",
    "geopandas==1.0.1",
    "idna==3.7",
    "joblib==1.4.2",
    "kiwisolver==1.4.5",
    "locket==1.0.0",
    "lxml==5.2.2",
    "matplotlib==3.9.1",
    "networkx==3.3",
    "numpy==2.2.6",
    "openpyxl==3.1.2",
    "osmnx==2.0.1",
    "packaging==24.0",
    "pandas==2.2.2",
    "partd==1.4.2",
    "pillow==10.4.0",
    "polars==1.10.0",
    "pyarrow==16.1.0",
    "pycatastro==0.1.3",
    "pygeoif==0.7",
    "pyogrio==0.10.0",
    "pyparsing==3.1.2",
    "pyproj==3.6.1",
    "python-dateutil==2.9.0.post0",
    "pytz==2024.1",
    "PyYAML==6.0.2",
    "rasterio==1.3.10",
    "regex==2024.11.6",
    "requests==2.31.0",
    "scikit-learn==1.6.0",
    "scipy==1.14.1",
    "setuptools==70.3.0",
    "shapely==2.1.1",
    "six==1.16.0",
    "snuggs==1.4.7",
    "soupsieve==2.5",
    "threadpoolctl==3.5.0",
    "toolz==1.0.0",
    "tqdm==4.66.5",
    "tzdata==2024.1",
    "urllib3==2.2.1",
    "xmltodict==0.14.2",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
