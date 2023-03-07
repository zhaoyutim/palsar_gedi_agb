import glob
import os
import subprocess
import ee
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from google.cloud import storage
from scipy.stats import stats
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

def get_dataset(channel_first=False):
    trainset = h5py.File("africa-biomass-challenge/09072022_1154_train.h5", "r")
    validateset = h5py.File("africa-biomass-challenge/09072022_1154_val.h5", "r")
    testset = h5py.File("africa-biomass-challenge/09072022_1154_test.h5", "r")
    def feature_engineering(array, scl, cloud, lat, lon, channel_first):
        dvi = array[:, :, :, [7]] / (array[:, :, :, [3]] + 1e6)
        ndvi = (array[:, :, :, [7]] - array[:, :, :, [3]]) / (array[:, :, :, [7]] + array[:, :, :, [3]] + 1e6)
        ndvi2 = (array[:, :, :, [8]] - array[:, :, :, [3]]) / (array[:, :, :, [8]] + array[:, :, :, [3]] + 1e6)
        gndvi = (array[:, :, :, [7]] - array[:, :, :, [2]]) / (array[:, :, :, [7]] + array[:, :, :, [2]] + 1e6)
        ndi45 = (array[:, :, :, [4]] - array[:, :, :, [3]]) / (array[:, :, :, [4]] + array[:, :, :, [3]] + 1e6)
        ndre = (array[:, :, :, [7]] - array[:, :, :, [4]]) / (array[:, :, :, [7]] + array[:, :, :, [4]] + 1e6)
        MEAN = array.mean((0, 1, 2))
        STD = array.std((0, 1, 2))
        array = (array[:, :, :, :12] - MEAN[None, None, None, :12]) / STD[None, None, None, :12]
        array = np.concatenate([array, cloud, dvi, gndvi, ndi45, ndre, ndvi, ndvi2, scl, lat, lon], axis=3)
        array = np.where(scl!=4, 0, array)
        array = array[:,3:13, 3:13, :]
        array = np.concatenate(array[:, :, :, [1]], array[:, :, :, [4]],array[:, :, :, [5]],array[:, :, :, [10]],
                               array[:, :, :, [11]], array[:, :, :, [15]],array[:, :, :, [16]],array[:, :, :, [21]])
        if channel_first:
            array = array.transpose(0, 3, 1, 2)
        return array

    # train
    train_images = np.array(trainset['images'], dtype=np.float64)
    train_scl = np.array(trainset['scl'], dtype=np.float64)
    train_cloud = np.array(trainset['cloud'], dtype=np.float64)
    train_lat = np.array(trainset['lat'], dtype=np.float64)
    train_lon = np.array(trainset['lon'], dtype=np.float64)
    train_biomasses = np.array(trainset['agbd'], dtype=np.float64)
    train_images_norm = feature_engineering(train_images, train_scl, train_cloud, train_lat, train_lon, channel_first)
    corr_max = []
    corr_min = []
    for i in range(train_images_norm.shape[-1]):
        corr_position = np.zeros((train_images_norm.shape[1],train_images_norm.shape[2]))
        for j in range(train_images_norm.shape[1]):
            for k in range(train_images_norm.shape[2]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    train_images_norm[:, j, k, i], train_biomasses)
                corr_position[j,k] = r_value
        plt.title('Correlation between Band {} with Biomass with max {:.3f} and min r value {:.3f}'.format(i+1, corr_position.max(), corr_position.min()))
        plt.imshow(corr_position)
        plt.colorbar()
        plt.savefig('plt/corr_band'+str(i+1)+'.png')
        plt.show()
        corr_max.append(corr_position.max())
        corr_min.append(corr_position.min())

    df = pd.DataFrame({'corr_max':corr_max, 'corr_min':corr_min})
    df.head(5)
    df.to_csv('all_bands_10x10.csv')
def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))

def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile

def upload_to_gcloud(file):
    print('Upload to gcloud')
    file_name = file.split('/')[-1]
    storage_client = storage.Client()
    bucket = storage_client.bucket('ai4wildfire')
    upload_cmd = 'gsutil cp ' + file + ' gs://ai4wildfire/abc_chan/' + file_name
    print(upload_cmd)
    os.system(upload_cmd)
    print('finish uploading' + file)

def upload_to_gee(file):
    print('start uploading to gee')
    file_name = file.split('/')[-1]
    cmd = 'earthengine upload image --asset_id=users/zhaoyutim/abc_challenge_label/' + \
          file.split('/')[-1].split('.')[0] + ' --pyramiding_policy=sample gs://ai4wildfire/abc_chan/' + file_name
    subprocess.call(cmd.split())
def get_infer_tif():
    infer_images = h5py.File("africa-biomass-challenge/images_test.h5", "r")
    infer_images = np.array(infer_images["images"])
    infer_scl = h5py.File("africa-biomass-challenge/scl_test.h5", "r")
    infer_scl = np.array(infer_scl["scl"])
    infer_cloud = h5py.File("africa-biomass-challenge/cloud_test.h5", "r")
    infer_cloud = np.array(infer_cloud["cloud"])
    infer_lat = h5py.File("africa-biomass-challenge/lat_test.h5", "r")
    infer_lat = np.array(infer_lat["lat"])
    infer_lon = h5py.File("africa-biomass-challenge/lon_test.h5", "r")
    infer_lon = np.array(infer_lon["lon"])

    for i in range(infer_images.shape[0]):
        output_array = infer_scl[i,:,:,0]
        lat = infer_lat[i,:,:,:]
        lon = infer_lon[i,:,:,:]
        nx = output_array.shape[0]
        ny = output_array.shape[0]
        xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(float(xmin), float(ymin))
        ring.AddPoint(float(xmin), float(ymax))
        ring.AddPoint(float(xmax), float(ymax))
        ring.AddPoint(float(xmax), float(ymin))

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.CreateDataSource('polygon'+str(i)+'.shp')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer("polygon", srs, ogr.wkbPolygon)
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        layer.CreateField(idField)
        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", i)
        layer.CreateFeature(feature)
        feature = None
        ds = None
def export_label():
    ee.Initialize()
    table = ee.FeatureCollection("projects/grand-drive-285514/assets/abc_polygon")
    gedi4 = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY").select('agbd').mosaic()
    biomasscci = ee.ImageCollection("projects/ee-zhaoyutim/assets/biomasscci2018").mosaic()
    polygons = table.toList(90)
    for i in range(90):
        polygon = ee.Feature(polygons.get(i))
        output = ee.Image([biomasscci, gedi4])
        dir = 'abc_label' + '/' + str(i)
        image_task = ee.batch.Export.image.toCloudStorage(
            image=output.toFloat(),
            description='Image Export:' + 'id_' + str(i),
            fileNamePrefix=dir,
            bucket='ai4wildfire',
            scale=20,
            maxPixels=1e11,
            region=polygon.geometry(),
        )
        image_task.start()
        print('Start with image task (id: {}).'.format(i))
def get_fake_label():
    fake_label = np.zeros(90)
    file_list = glob.glob('/Users/zhaoyu/PycharmProjects/palsar_gedi_agb/fake_labels/*.tif')
    for i, file in enumerate(file_list):
        array, _ = read_tiff(file)
        fake_label[i] = np.nanmean(array[1, :, :])
    ID_S2_pair = pd.read_csv('africa-biomass-challenge/UniqueID-SentinelPair.csv')
    fake_label = pd.DataFrame({'Target': fake_label}).rename_axis('S2_idx').reset_index()
    fake_label = ID_S2_pair.merge(fake_label, on='S2_idx').drop(columns=['S2_idx'])

    pred = pd.read_csv('africa-biomass-challenge/predictions/biomass_predictionsvit_tiny_customnum_heads_3num_layers_4mlp_dim_512hidden_size_192batchsize_128.csv')
    import tensorflow as tf
    x = pred.merge(fake_label, on='ID').rename_axis('asd')
    x.Target_y.fillna(x.Target_x, inplace=True)
    x = x.drop(['Target_x'], axis=1).rename(columns={'Target_y':'Target'}).drop(columns=['asd'])
    x.to_csv('fake_label.csv')


if __name__=='__main__':
    get_fake_label()
    # file_list = glob.glob('/Users/zhaoyu/PycharmProjects/palsar_gedi_agb/label/*.tif')
    # for file in file_list:
    #     upload_to_gcloud(file)
        # upload_to_gee(file)