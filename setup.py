import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = "1.0.0"
PACKAGE_NAME = "hypercadaster_ES"
AUTHOR = "Jose Manuel Broto Vispe"
AUTHOR_EMAIL = "jmbrotovispe@gmail.com"
URL = "https://github.com/BeeGroup-cimne"

LICENSE = "MIT"
DESCRIPTION = "Python library to obtain the Spanish cadaster data joined with external attributes."
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    # --- Core numerics & ML ---
    "numpy>=1.24,<3.0",
    "scipy>=1.10,<2.0",
    "pandas>=2.1,<3.0",
    "polars>=1.0,<2.0",
    "scikit-learn>=1.3,<2.0",
    "joblib>=1.3",
    "threadpoolctl>=3.3",

    # --- Parallel & task orchestration ---
    "dask>=2024.4",
    "cloudpickle>=3.0",
    "toolz>=0.12",
    "partd>=1.4",

    # --- Geospatial core (ABI-sensitive caps) ---
    "shapely>=2.0,<3.0",
    "pyproj>=3.5,<4.0",
    "rasterio>=1.3,<2.0",
    "fiona>=1.9,<2.0",
    "geopandas>=0.13,<1.1",
    "pyogrio>=0.8,<1.0",
    "affine>=2.4",
    "snuggs>=1.4",
    "pygeoif>=0.7",

    # --- OSM / Networks ---
    "osmnx>=1.8,<2.1",
    "networkx>=3.2",

    # --- File formats & tabular IO ---
    "pyarrow>=14.0",
    "openpyxl>=3.1",
    "et-xmlfile>=1.1",

    # --- Web / HTTP ---
    "requests>=2.31,<3.0",
    "urllib3>=2.1,<3",
    "charset-normalizer>=3.3",
    "idna>=3.6",
    "certifi>=2024.2",
    "beautifulsoup4>=4.12",
    "soupsieve>=2.5",
    "xmltodict>=0.13",

    # --- Plotting & visuals ---
    "matplotlib>=3.8,<4",
    "contourpy>=1.2",
    "cycler>=0.12",
    "kiwisolver>=1.4",
    "fonttools>=4.53",
    "pillow>=10.2",

    # --- CLI & packaging helpers ---
    "click>=8.1",
    "click-plugins>=1.1",
    "cligj>=0.7",
    "packaging>=23.2",

    # --- Dates, timezones & text utils ---
    "python-dateutil>=2.8.2",
    "pytz>=2024.1",
    "tzdata>=2024.1",
    "regex>=2023.12,<2027.0",
    "tqdm>=4.66",
    "six>=1.16",

    # --- YAML / config ---
    "PyYAML>=6.0",
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
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
    extras_require={
        "dev": ["black>=24.0", "ruff>=0.4", "pytest>=7.0", "pytest-cov>=4.0"],
        # "pdal": ["pdal>=3.0"],  # if/when you wire PDAL features
    },
)
