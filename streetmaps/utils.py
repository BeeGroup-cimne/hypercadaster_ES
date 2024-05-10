import copy
import sys
import os
from zipfile import ZipFile
import requests
from bs4 import BeautifulSoup


def create_dir(data_dir):
    os.makedirs(f"{data_dir}/census_tracts", exist_ok=True)
    os.makedirs(f"{data_dir}/districts", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/buildings", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/buildings/zip", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/buildings/unzip", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/address", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/address/zip", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/address/unzip", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/parcels", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/parcels/zip", exist_ok=True)
    os.makedirs(f"{data_dir}/inspire/parcels/unzip", exist_ok=True)
    os.makedirs(f"{data_dir}/municipalities", exist_ok=True)
    os.makedirs(f"{data_dir}/neighbourhoods", exist_ok=True)
    os.makedirs(f"{data_dir}/postal_codes", exist_ok=True)
    os.makedirs(f"{data_dir}/postal_codes/raw", exist_ok=True)


def list_municipalities(province_codes=None,
                        inspire_url="https://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml"):
    response = requests.get(inspire_url)
    soup = BeautifulSoup(response.content, "html.parser")
    municipalities = soup.find_all("div", id='scrolltable')

    urls = [x.get('href') for x in soup.find_all("link", rel="enclosure")]
    list_municipalities = []
    for j in range(len(municipalities)):
        x = municipalities[j]
        url = urls[j]

        if province_codes is None or url[48:50] in province_codes or url[55:57] in province_codes:
            sys.stdout.write('Downloading province: %s\n' % (url[48:50]))
            # Obtain the municipality name
            x = copy.deepcopy(x)
            x = [line.strip() for line in
                 x.get_text(separator='\n').strip().replace("\t", "").replace("\r", "").replace(' ', '').replace('\n\n',
                                                                                                                 '\n').split(
                     '\n') if line.strip()]
            x = copy.deepcopy(x)
            z = []
            for y in x:
                if y:
                    z.append(y)
            z.sort()
            # Obtain the URL's
            url_soup = BeautifulSoup(requests.get(url).content, "html.parser")
            municip_urls = [x.get('href') for x in url_soup.find_all("link", rel="enclosure")]
            municip_urls = [x for _, x in sorted(zip([y[50:56] for y in municip_urls], municip_urls))]
            # Extend the list of municipalities
            for i in range(len(z)):
                list_municipalities.append(
                    {
                        "name": z[i],
                        "url": municip_urls[i]
                    })
    return list_municipalities


def unzip_directory(zip_directory, unzip_directory):
    for file in os.listdir(zip_directory):
        if file.endswith(".zip"):
            with ZipFile(f"{zip_directory}{file}", 'r') as zip:
                zip.extractall(unzip_directory)
