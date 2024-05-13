import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'streetmaps'
AUTHOR = 'Jose Manuel Broto Vispe'
AUTHOR_EMAIL = 'jmbrotovispe@gmail.com'
URL = 'https://github.com/josemanuel97'

LICENSE = 'MIT'
DESCRIPTION = 'Librería para obtener callejero geolocalizado en España.'
LONG_DESCRIPTION = (HERE / "README.md").read_text(
    encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'attrs==23.2.0',
    'beautifulsoup4==4.12.3',
    'certifi==2024.2.2',
    'charset-normalizer==3.3.2',
    'click==8.1.7',
    'click-plugins==1.1.1',
    'cligj==0.7.2',
    'fastkml==0.12',
    'fiona==1.9.6',
    'geopandas==0.14.4',
    'idna==3.7',
    'numpy==1.26.4',
    'packaging==24.0',
    'pandas==2.2.2',
    'pyproj==3.6.1',
    'python-dateutil==2.9.0.post0',
    'pytz==2024.1',
    'rasterio==1.3.10',
    'requests==2.31.0',
    'shapely==2.0.4',
    'six==1.16.0',
    'tzdata==2024.1',
    'urllib3==2.2.1',
    'openpyxl==3.1.2',
    'et-xmlfile==1.1.0'
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
