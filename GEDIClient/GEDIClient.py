import glob
import os.path
import warnings

import geopandas as gpd
import requests
import yaml
from matplotlib import pyplot as plt
from shapely.ops import orient
import datetime as dt
from pydap.cas.urs import setup_session
import netCDF4 as nc
import pandas as pd
from requests.adapters import HTTPAdapter, Retry

with open("config/credential.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
warnings.filterwarnings('ignore')


class GEDIClient:
    def __init__(self):
        return

    def visualize_geojson(self, json_path):
        aca = gpd.read_file(json_path)

        aca.crs = "EPSG:4326"
        aca.geometry = aca.geometry.apply(orient, args=(1,))

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        base = world.plot(color='white', edgecolor='black', figsize=(7, 7))
        ax = aca.plot(ax=base, color='red')
        plt.show()

    def query_with_json(self, json_path, id):
        aca = gpd.read_file(json_path)
        aca = aca.iloc[[id]]
        aca.crs = "EPSG:4326"
        aca.geometry = aca.geometry.apply(orient, args=(1,))
        # GEDI L4A DOI
        doi = '10.3334/ORNLDAAC/2056'

        # CMR API base url
        cmrurl = 'https://cmr.earthdata.nasa.gov/search/'
        doisearch = f"{cmrurl}collections.json?doi={doi}"
        concept_id = requests.get(doisearch).json()['feed']['entry'][0]['id']

        # defining geojson
        geojson = {"shapefile": ("aca.geojson", aca.geometry.to_json(), "application/geo+json")}
        page_num = 1
        page_size = 2000  # CMR page size limit

        # time bound for 2020
        start_date = dt.datetime(2020, 1, 1)  # specify your own start date
        end_date = dt.datetime(2020, 12, 31)  # specify your end start date

        # CMR formatted start and end times
        dt_format = '%Y-%m-%dT%H:%M:%SZ'
        temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)

        opendap_arr = []

        while True:

            # defining parameters
            cmr_param = {
                "collection_concept_id": concept_id,
                "temporal": temporal_str,
                "page_size": page_size,
                "page_num": page_num,
                "simplify-shapefile": 'true'  # this is needed to bypass 5000 coordinates limit of CMR
            }

            granulesearch = f"{cmrurl}granules.json"
            response = requests.post(granulesearch, data=cmr_param, files=geojson)
            granules = response.json()['feed']['entry']
            if granules:
                for g in granules:
                    # Get OPeNDAP URLs
                    for links in g['links']:
                        if 'title' in links and links['title'].startswith('OPeNDAP'):
                            opendap_url = links['href']
                            opendap_arr.append(opendap_url)
                page_num += 1
            else:
                break

        total_granules = len(opendap_arr)
        print("Total granules found: ", total_granules)
        print(opendap_arr[:3])
        return opendap_arr, total_granules

    def download(self, json_path, file_url_list, region, id, total_granules):
        aca = gpd.read_file(json_path)
        aca = aca.iloc[[id]]
        aca.crs = "EPSG:4326"
        aca.geometry = aca.geometry.apply(orient, args=(1,))
        username = config.get('earthdata_username')
        password = config.get('earthdata_password')
        session = setup_session(username, password, check_url="https://opendap.earthdata.nasa.gov/")
        variables = ['agbd', 'l4_quality_flag', 'land_cover_data/pft_class']
        beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
        out_csv = 'subsets/'+region+'/'+region+'_gedi_l4a' + str(id) + '.csv'
        if not os.path.exists('subsets/'+region+'/'):
            os.mkdir('subsets/'+region+'/')
        headers = ['lat_lowestmode', 'lon_lowestmode', 'elev_lowestmode', 'shot_number']
        headers.extend(variables)
        with open(out_csv, "w") as f:
            f.write(','.join(headers) + '\n')

        # setting up maximum retries to get around Hyrax 500 error
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        c = 0
        for i, g_name in enumerate(file_url_list):
            print(g_name, i, total_granules)
            c += 1
            # loop over all beams
            for beam in beams:
                # print(beam)
                # 1. Retrieving lat, lon coordinates for the file
                hyrax_url = f"{g_name}.dap.nc4?dap4.ce=/{beam}/lon_lowestmode;/{beam}/lat_lowestmode"
                r = session.get(hyrax_url)
                if (r.status_code == 200) and r.content != b'':

                    try:
                        ds = nc.Dataset('hyrax', memory=r.content)
                        lat = ds[beam]['lat_lowestmode'][:]
                        lon = ds[beam]['lon_lowestmode'][:]
                        ds.close()
                        df = pd.DataFrame({'lat_lowestmode': lat, 'lon_lowestmode': lon})  # creating pandas dataframe

                        # 2. Subsetting by bounds of the area of interest
                        # converting to geopandas dataframe
                        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_lowestmode, df.lat_lowestmode))
                        gdf_aca = gdf[gdf['geometry'].within(aca.geometry[id])]
                        if not gdf_aca.empty:
                            # creating empty columns for variables
                            for v in headers[2:]:
                                gdf_aca[v] = None
                            # 3. retrieving variables of interest, agbd, agbd_t in this case.
                            # We are only retriving the shots within subset area.
                            for _, df_gr in gdf_aca.groupby((gdf_aca.index.to_series().diff() > 1).cumsum()):
                                i = df_gr.index.min()
                                j = df_gr.index.max()
                                for v in headers[2:]:
                                    var_s = f"/{beam}/{v}%5B{i}:{j}%5D"
                                    hyrax_url = f"{g_name}.dap.nc4?dap4.ce={var_s}"
                                    r = session.get(hyrax_url)
                                    if (r.status_code == 200) and r.content != b'':
                                        try:
                                            ds = nc.Dataset('hyrax.nc', memory=r.content)
                                            gdf_aca.loc[i:j, (v)] = ds[beam][v][:]
                                            ds.close()
                                        except:
                                            print('unable to parse')
                                            print(r.content)
                                            continue

                            # saving the output file
                            gdf_aca.to_csv(out_csv, mode='a', index=False, header=False, columns=headers)
                    except:
                        print('unable to parse')
                        print(r.content)
                        continue
    def concatcsv(self, path, region):
        file_list = glob.glob(path)
        li = []
        headers = ['lat', 'lon', 'elev_lowestmode', 'shot_number', 'agbd', 'l4_quality_flag', 'land_cover_data/pft_class']
        for file in file_list:
            df = pd.read_csv(file, index_col=None, header=0)
            df.rename(columns={'lat_lowestmode': 'lat', 'lon_lowestmode': 'lon'}, inplace=True)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=True)
        frame.to_csv('csv/conbine_csv_'+region+'.csv', mode='a', index=False, header=headers)
