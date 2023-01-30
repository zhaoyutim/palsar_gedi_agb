import datetime
import os
from glob import glob

import ee
import numpy as np
import plotly.express as px
import rasterio
import yaml
from google.cloud import storage
from matplotlib import pyplot as plt

from ParamsFetching import ParamsFetching
from run_cnn_model_gedi import create_model_cpu

class Gedi:

    def qualityMask(self, img):
        return img.updateMask(img.select('quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))

    def qualityMask_l4a(self, img):
        return img.updateMask(img.select('l4_quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read().astype(np.float32).transpose((1,2,0))
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))

    def random_blind(self, array, percentage):
        '''
        This function is used in test section, to randomly mask out some percentage of GEDI L2A RH data in the input.
        :param array: Input Array.
        :param percentage: Percentage of footprints to be masked.
        :return:
        '''
        sample_filter = np.random.binomial(1, percentage, array.shape)
        filter = np.logical_and(sample_filter == 1, np.logical_not(np.isnan(array)))
        return np.where(filter, array, np.nan)

    def download_to_gcloud(self, region_ids=['na'], mode='test', year = 2019, custom_region=None):
        '''
        This function is to process GEDI and PALSAR-2 data in Google Earth Engine。
        :param region_ids: Regions to be processed ['na', 'sa', 'af', 'eu', 'au', 'sas', 'nas']
        :param mode: Download test or train samples if custom_region is set to None
        :param year: The year of PALSAR-2 and GEDI data
        :param custom_region: When custom_region is assigned, only images of given region will be downloaded.
        :return:
        '''
        dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH') \
            .filter(ee.Filter.date(str(year)+'-01-01', str(year + 1)+'-01-01')) # PALSAR-2 data import
        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83) # SAR Digital Numebr to Backscatter
        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
        sarhvhh = sarHv_log.subtract(sarHh_log).rename('HV-HH') # HV-HH is considered as the third channel
        composite = ee.Image([sarHh_log, sarHv_log, sarhvhh])
        lc = ee.ImageCollection("ESA/WorldCover/v100").first() # landcover data
        l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select(['PS', 'MU']) # GEDI L4B data (Not in use but included as reference)
        if custom_region == None:
            for region_id in region_ids:
                # Select test region or train region
                if mode=='test':
                    roi_col = ee.FeatureCollection('users/zhaoyutim/GEDI_Test_' + region_id.upper())
                else:
                    roi_col = ee.FeatureCollection('users/zhaoyutim/GEDI_SAMPLE_'+region_id.upper())
                size = roi_col.size().getInfo()
                roi_col = roi_col.toList(size)
                for i in range(size):
                    # For each polygon getch data from GEE to Google cloud storage
                    roi = ee.Feature(roi_col.get(i).getInfo())
                    class_id = roi.args['metadata'].get('class')
                    date_pre = dataset.select('epoch').median().clip(roi).reduceRegion(
                        reducer=ee.Reducer.max(),
                        geometry=roi.geometry(),
                        scale=1000,
                        crs='EPSG:4326').getNumber('epoch')
                    date_pre = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(date_pre).divide(1000), 'second')) # calculate capture date of PALSAR-2 image
                    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                        .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                        .map(self.qualityMask)\
                        .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
                    # get GEDI images within 15 days before and after date of PALSAR-2 image
                    agbd_l4a = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY').filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day")).map(self.qualityMask_l4a).select(['agbd']).mosaic()
                    output = ee.Image([composite, lc, l4b, gedi, agbd_l4a])
                    dir = 'gedi_palsar_data' + '/' + region_id.upper() + str(year) + '/' + 'year'+ str(year)+ 'class_' + class_id + '_' + str(i)
                    if mode=='test':
                        mode_str=mode
                    else:
                        mode_str=''
                    # Export to cloud storage
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=output.toFloat(),
                        description='Image Export:' + 'GEDI_PALSAR_' + region_id.upper() + mode_str + str(year)+'_CLASS_'+class_id,
                        fileNamePrefix=dir,
                        bucket='ai4wildfire',
                        scale=25,
                        maxPixels=1e11,
                        region=roi.geometry(),
                        fileDimensions=256*5
                    )
                    image_task.start()
                    if mode=='test':
                        print('Start with image task (id: {}).'.format(
                            'GEDI-PALSAR Image Export:' + 'GEDI_Test_'+region_id.upper() + str(year)+'_INDEX_'+str(i)+'_CLASS_'+class_id))

                    else:
                        print('Start with image task (id: {}).'.format(
                            'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_'+region_id.upper() + str(year)+'_INDEX_'+str(i)+'_CLASS_'+class_id))
        else:
            # initiate an polygon with given coordinates
            region = ee.Geometry.Rectangle(custom_region)
            roi = ee.Feature(region)
            date_pre = dataset.select('epoch').median().clip(roi).reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=roi.geometry(),
                scale=1000,
                crs='EPSG:4326').getNumber('epoch')
            date_pre = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(date_pre).divide(1000), 'second'))
            gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                .map(self.qualityMask)\
                .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
            agbd_l4a = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY').filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day")).map(self.qualityMask_l4a).select(
                ['agbd']).mosaic()
            output = ee.Image([composite, lc, l4b, gedi, agbd_l4a])
            dir = 'gedi_palsar_data' + '/' + 'custom_region' + str(year) + '/' + 'year'+ str(year)
            image_task = ee.batch.Export.image.toCloudStorage(
                image=output.toFloat(),
                description='Image Export:' + 'GEDI_PALSAR_' + 'custom_region' + str(year),
                fileNamePrefix=dir,
                bucket='ai4wildfire',
                scale=25,
                maxPixels=1e11,
                region=roi.geometry(),
                fileDimensions=256*5
            )
            image_task.start()
            print('Start with image task (id: {}).'.format(
                'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_' + 'custom_region' + str(year)))

    def download_to_local_proj4(self, create_time='2022-06-18'):
        '''
        Download images from cloud storage to local with images created later than given date
        :param create_time: Only files created after this time will be downloaded.
        :return:
        '''
        storage_client = storage.Client()
        bucket = storage_client.bucket('ai4wildfire')
        blobs = bucket.list_blobs(prefix='gedi_palsar_data')
        for blob in blobs:
            if blob.time_created.date() < datetime.datetime.strptime(create_time, '%Y-%m-%d').date():
                continue
            filename = blob.name
            path = os.path.dirname(filename)
            if not os.path.exists(path):
                os.makedirs(path)
            blob.download_to_filename(filename)
            print(
                "Blob {} downloaded to {}.".format(
                    filename, filename
                )
            )

    def slice_into_small_tiles(self, array, new_shape, concat=False):
        '''
        Clip GeoTIFF into the wanted shape.
        :param array: Input array
        :param new_shape: Cliped shape
        :param concat: Concatenation or stack operation
        :return:
        '''
        shape_x = array.shape[0]
        shape_y = array.shape[1]
        loop_x = shape_x // new_shape
        loop_y = shape_y // new_shape
        new_array = []
        for i in range(loop_x):
            for j in range(loop_y):
                piece = array[new_shape*i:new_shape*(i+1), new_shape*j:new_shape*(j+1), :]
                new_array.append(piece)
        if concat==False:
            array = np.stack(new_array, axis=0)
        else:
            array = np.concatenate(new_array, axis=0)
        return array

    def slice_into_small_tiles_inference(self, array, new_shape=64, overlap=32):
        '''
        Clip and stack inference images into smaller shape with overlap between each other. During the inference time, to
        have better qualitative results, only the center part of the image is used.
        :param array: Input array
        :param new_shape: Cliped shape
        :param overlap: Overlap between two samples
        :return:
        '''
        shape = array.shape[0]
        center_size = new_shape-overlap
        loop_size = (shape-overlap)//center_size
        new_array = []
        for i in range(loop_size):
            for j in range(loop_size):
                piece = array[center_size*i:center_size*i+new_shape, center_size*j:center_size*j+new_shape, :]
                if np.nanmean(piece[:,:,8])==-1.0:
                    continue
                new_array.append(piece)
        array = np.stack(new_array, axis=0)
        return array, loop_size

    def combine_images_inference(self, array, loop_size, overlap=32):
        '''
        Combine images after the inference into the orginal shape
        :param array: input array
        :param loop_size: Number of horizontal loops and vertical loops
        :param overlap: overlap between patches
        :return:
        '''
        current_shape = array.shape[1]
        desired_shape = (current_shape-overlap)*loop_size
        center_shape = current_shape-overlap
        edge_shape=overlap//2
        new_array = np.zeros((desired_shape, desired_shape, array.shape[3]))
        for i in range(loop_size):
            for j in range(loop_size):
                new_array[center_shape*i:center_shape*(i+1), center_shape*j:center_shape*(j+1), :] = array[i*loop_size+j, edge_shape:current_shape-edge_shape, edge_shape:current_shape-edge_shape, :]
        return new_array

    def remove_outliers(self, x, outlierConstant):
        upper_quartile = np.nanpercentile(x, 75)
        lower_quartile = np.nanpercentile(x, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = x * (x >= quartileSet[0]) * (x <= quartileSet[1])
        return result

    def remove_outliers_nan(self, x, outlierConstant):
        upper_quartile = np.nanpercentile(x, 75)
        lower_quartile = np.nanpercentile(x, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        x = np.nan_to_num(x)
        result = x * (x >= quartileSet[0]) * (x <= quartileSet[1])
        result = np.where(result == 0, np.nan, result)
        return result

    def standardization(self, x):
        x = (x - x.mean()) / x.std()
        return x

    def normalization(self, x):
        return 255 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

    def generate_dataset_proj4(self, region_ids = ['na', 'sa', 'af', 'eu', 'au', 'sas', 'nas'], random_blind=False, year=2020, custom_region=None, mode='test', rb=0.5):
        #params_fetching = ParamsFetching()
        if custom_region!=None:
            region_ids=['custom_region']
        for region_id in region_ids:
            if year == 2019:
                path = os.path.join('gedi_palsar_data', region_id.upper(), '*.tif')
            else:
                path = os.path.join('gedi_palsar_data', region_id.upper()+str(year), '*.tif')
            file_list = glob(path)
            dataset_list = []
            print('region_id:', region_id)
            index=0
            # Read file
            for file in file_list:
                if mode=='test' and 'Test' not in file and region_id!='custom_region':
                    continue
                elif mode!='test' and 'Test' in file and region_id!='custom_region':
                    continue
                array, _ = self.read_tiff(file)
                index += 1
                if array.shape[0]<64 or array.shape[1]<64 or array.shape[2]!=12:
                    continue
                for i in range(5):
                    rh = array[:, :, 6+i]
                    array[:, :, 6+i] = self.remove_outliers_nan(rh, 1)
                output_array = np.zeros((array.shape[0], array.shape[1], 10)).astype(np.float32)
                # Previous solution we neet to transfer L2A to L4A since L4A is not availble in GEE
                # agbd_l2a = params_fetching.get_agbd(array[:, :, 4:])
                # agbd_l2a = np.where(agbd_l2a==-1, np.nan, agbd_l2a)

                # Remove outlier and nan for PALSAR-2 images
                for i in range(3):
                    output_array[:, :, i] = self.remove_outliers(array[:, :, i], 1)
                    output_array[:, :, i] = np.nan_to_num(output_array[:, :, i])

                # Randomly drop GEDI L2A RH at input
                if random_blind:
                    output_array[:, :, 4:9] = np.nan_to_num(self.random_blind(array[:, :, 6:11], rb), nan=-1)
                else:
                    output_array[:, :, 4:9] = np.nan_to_num(array[:, :, 6:11], nan=-1)

                # Remove nan for GEDI L4A for masked MSE
                agbd_l4a = np.nan_to_num(array[:, :, 11], nan=-1)
                output_array[:, :, 9] = agbd_l4a
                print(index)
                output_array[:, :, 3] = array[:, :, 3]
                output_array = self.slice_into_small_tiles(output_array, 64)
                dataset_list.append(output_array)
                del output_array
                del array
                del rh

                if index % 10==0:
                    print('{:.2f}% completed'.format(index*100/len(file_list)))

            if len(dataset_list)==1:
                dataset = dataset_list[0]
                del dataset_list
            else:
                dataset = np.concatenate(dataset_list, axis=0)
                del dataset_list


            np.save('dataset/proj4_train_'+region_id+str(year)+mode+str(rb)+'.npy', dataset)
            del dataset
            print('finish')

    def evaluate_and_plot(self, test_array_path='dataset/proj4_train_na2020.npy', model_path='model/proj4_unet_pretrained_resnet18_nchannels_', nchannels=4):
        import segmentation_models as sm
        region_id='custom_region'
        sm.set_framework('tf.keras')
        test_array= np.load(test_array_path)
        agbd = test_array[:,:,:,[9]]
        agbd = np.where(np.isnan(agbd), -1, agbd)

        # Inference
        if not os.path.exists('dataset_pred/'+region_id+'agbd_resnet18_unet_nchannels_'+str(nchannels)+'.npy'):
            model = create_model_cpu('unet', 'resnet18', 0.0003, nchannels=nchannels, nclass=1)
            model.load_weights(model_path+str(nchannels))
            agbd_pred = model.predict(test_array[:, :, :, :nchannels])
        else:
            agbd_pred = np.load('dataset_pred/'+region_id+'agbd_resnet18_unet_nchannels_'+str(nchannels)+'.npy')
        x_scatter = agbd[np.squeeze(agbd) != -1]
        y_scatter = agbd_pred[np.squeeze(agbd) != -1]
        from scipy import stats
        # Calculate R squared
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_scatter.flatten(), y_scatter.flatten())
        res = stats.linregress(x_scatter.flatten(), y_scatter.flatten())
        random_id = np.random.choice(x_scatter.shape[0], 10000)
        fig = px.scatter(x=x_scatter[random_id, 0], y=y_scatter[random_id, 0] * 100, range_x=[0, 1000], range_y=[0, 1000], trendline='ols')
        fig.update_layout(title='Correlation with ' + str(nchannels) + ' channels. R-squared: {0:.2f}'.format(r_value ** 2), xaxis_title="AGB Groundtruth (GEDI L4A AGB)", yaxis_title="AGB Predicted")
        fig.show()

    def inference(self, rb, path='gedi_palsar_data/CUSTOM_REGION2020/*.tif', random_blind=False, model_path='model/proj4_unet_pretrained_resnet18_nchannels_', nchannels=9, overlap=32):
        # Inference and generate image with original scale
        file_list=glob(path)
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        model = create_model_cpu('unet', 'resnet18', 0.0003, nchannels, 1)
        model.load_weights(model_path+str(nchannels))
        for file_dir in file_list:
            array, pf = self.read_tiff(file_dir)
            if array.shape[0] != 1280 or array.shape[1] != 1280 or array.shape[2] != 11:
                print('invalid shape')
                continue
            output_array = np.zeros((array.shape[0], array.shape[1], 9)).astype(np.float32)
            for i in range(3):
                output_array[:, :, i] = self.remove_outliers(array[:, :, i], 1)
                output_array[:, :, i] = np.nan_to_num(output_array[:, :, i])
            output_array[:, :, 3] = array[:, :, 3]
            if not random_blind:
                output_array[:, :, 4:] = np.nan_to_num(array[:, :, 6:])
            else:
                output_array[:, :, 4:] = np.nan_to_num(self.random_blind(array[:, :, 6:], rb))
            input, loop_size = self.slice_into_small_tiles_inference(output_array, new_shape=64, overlap=overlap)
            agbd_pred = model.predict(input[:,:,:,:nchannels])
            agbd_pred = self.combine_images_inference(array=agbd_pred, loop_size=loop_size, overlap=overlap)
            pf.data['count'] = 1
            self.write_tiff(file_dir.replace('gedi_palsar_data', 'recon'), agbd_pred.transpose((2,0,1)), pf)
            print('successfully reconstruct agbd predicted')



if __name__=='__main__':

    array=np.array([[np.nan,np.nan,np.nan,np.nan,4],
                   [np.nan,2,np.nan,np.nan,np.nan],
                   [np.nan,np.nan,1,np.nan,np.nan],
                   [np.nan,3,np.nan,np.nan,np.nan],
                   [np.nan,np.nan,np.nan,5,np.nan]])
    gedi=Gedi()
    print(array)
    print(gedi.random_blind(array, 1))
