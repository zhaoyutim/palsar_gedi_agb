import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import stats


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

    # train_biomasses_norm = (train_biomasses-train_biomasses.mean())/(train_biomasses.std())

    #
    # # validate
    # validate_images = np.array(validateset['images'], dtype=np.float64)
    # validate_scl = np.array(validateset['scl'], dtype=np.float64)
    # validate_cloud = np.array(validateset['cloud'], dtype=np.float64)
    # validate_lat = np.array(validateset['lat'], dtype=np.float64)
    # validate_lon = np.array(validateset['lon'], dtype=np.float64)
    # validate_biomasses = np.array(validateset['agbd'], dtype=np.float64)
    # validate_biomasses_norm = (validate_biomasses - train_biomasses.mean()) / (train_biomasses.std())
    # validate_images_norm = feature_engineering(validate_images, validate_scl, validate_cloud, validate_lat, validate_lon, channel_first)
    #
    #
    # # test
    # test_images = np.array(testset['images'], dtype=np.float32)
    # test_scl = np.array(testset['scl'], dtype=np.float64)
    # test_cloud = np.array(testset['cloud'], dtype=np.float64)
    # test_lat = np.array(testset['lat'], dtype=np.float64)
    # test_lon = np.array(testset['lon'], dtype=np.float64)
    # test_biomasses = np.array(testset['agbd'], dtype=np.float32)
    # test_biomasses_norm = (test_biomasses - train_biomasses.mean()) / (train_biomasses.std())
    # test_images_norm = feature_engineering(test_images, test_scl, test_cloud, test_lat, test_lon, channel_first)

if __name__=='__main__':
    get_dataset()