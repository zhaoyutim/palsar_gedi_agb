import argparse

from GEDIClient.GEDIClient import GEDIClient

def gedi_download(region, id):
    print(id)
    # time.sleep(3)
    json_path = 'gedi_train_sample_json/'+region+'_json.geojson'
    gedi_client=GEDIClient()
    # gedi_client.visualize_geojson(json_path)
    file_url_list = gedi_client.query_with_json(json_path, id)
    gedi_client.download(json_path, file_url_list, region, id)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-region', type=str, help='region to be executed')
    args = parser.parse_args()
    region = args.region
    json_path = 'gedi_train_sample_json/'+region+'_json.geojson'
    gedi_client = GEDIClient()
    file_url_list, total_granules = gedi_client.query_with_json(json_path, 0)
    gedi_client.download(json_path, file_url_list, region, 0, total_granules)
    # n_geometries = len(gpd.read_file(json_path))
    # pool = Pool(processes=n_geometries)
    # for i in range(n_geometries):
    #     pool.apply_async(gedi_download, args=(region, i))
    # print('test')
    # pool.close()
    # pool.join()
    # gedi_client.concatcsv('subsets/'+region+'/*.csv', region)

