import argparse
import os
import platform

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import KFold

import wandb
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from wandb.integration.keras import WandbCallback

from model.gru.gru_model import GRUModel
from model.lstm.lstm_model import LSTMModel
from model.vit_keras import vit

warnings.filterwarnings('ignore')
if platform.system() == 'Darwin':
    root_path = '/Users/zhaoyu/PycharmProjects/PALSAR-GEDI-AGB/africa-biomass-challenge'
else:
    root_path = '/geoinfo_vol1/zhao2/abc_challenge_models'

def wandb_config(model_name, num_layers, hidden_size, validation_fold):
    wandb.login()
    wandb.init(project="abc_challenge_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)+str(validation_fold)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "embed_dim": hidden_size
    }

def get_dataset(channel_first=True):
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
        # array = np.where(scl!=4, 0, array)
        # array = array[:,4:13, 4:13, :]
        # array = np.concatenate([array[:, :, :, [10]],
        #                        array[:, :, :, [11]]], axis=3)
        if channel_first:
            array = array.transpose(0, 3, 1, 2)
        return array
    # train
    train_images = np.concatenate([np.array(trainset['images'], dtype=np.float64), np.array(validateset['images'], dtype=np.float64), np.array(testset['images'], dtype=np.float64)], axis=0)
    train_scl = np.concatenate([np.array(trainset['scl'], dtype=np.float64), np.array(validateset['scl'], dtype=np.float64), np.array(testset['scl'], dtype=np.float64)], axis=0)
    train_cloud = np.concatenate([np.array(trainset['cloud'], dtype=np.float64), np.array(validateset['cloud'], dtype=np.float64), np.array(testset['cloud'], dtype=np.float64)], axis=0)
    train_lat = np.concatenate([np.array(trainset['lat'], dtype=np.float64), np.array(validateset['lat'], dtype=np.float64), np.array(testset['lat'], dtype=np.float64)], axis=0)
    train_lon = np.concatenate([np.array(trainset['lon'], dtype=np.float64), np.array(validateset['lon'], dtype=np.float64), np.array(testset['lon'], dtype=np.float64)], axis=0)
    train_biomasses_norm = np.concatenate([np.array(trainset['agbd'], dtype=np.float64), np.array(validateset['agbd'], dtype=np.float64), np.array(testset['agbd'], dtype=np.float64)], axis=0)
    train_images_norm = feature_engineering(train_images, train_scl, train_cloud, train_lat, train_lon, channel_first)
    # Filter
    train_images_norm = train_images_norm[np.logical_and(train_biomasses_norm <= 150, train_biomasses_norm > 50)]
    train_biomasses_norm = train_biomasses_norm[np.logical_and(train_biomasses_norm <= 150, train_biomasses_norm > 50)]
    if channel_first:
        train_images_norm = train_images_norm.reshape((train_images_norm.shape[0], train_images_norm.shape[1], -1))

    # infer
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
    infer_images_norm = feature_engineering(infer_images, infer_scl, infer_cloud, infer_lat, infer_lon, channel_first)
    if not channel_first:
        infer_images_norm = infer_images_norm
    else:
        infer_images_norm = infer_images_norm.reshape((infer_images_norm.shape[0], infer_images_norm.shape[1], -1))

    def make_dataset(X_data, y_data, n_splits):

        def gen():
            for train_index, test_index in KFold(n_splits).split(X_data):
                X_train, X_val = X_data[train_index], X_data[test_index]
                y_train, y_val = y_data[train_index], y_data[test_index]
                yield X_train, y_train, X_val, y_val

        return tf.data.Dataset.from_generator(gen, (tf.float64, tf.float64, tf.float64, tf.float64))
    dataset = make_dataset(train_images_norm, train_biomasses_norm, 5)
    return dataset, infer_images_norm

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')
    args = parser.parse_args()
    model_name = args.m
    batch_size = args.b
    num_heads=args.nh
    mlp_dim=args.md
    num_layers=args.nl
    hidden_size=args.ed

    num_classes = 1

    lr = args.lr
    learning_rate = lr
    weight_decay = lr/10
    MAX_EPOCHS = 1

    if model_name == 'vit_tiny_custom':
        channel_first = False
    else:
        channel_first = True
    dataset, infer_images_norm = get_dataset(channel_first=channel_first)

    if not channel_first:
        input_shape = (infer_images_norm.shape[1], infer_images_norm.shape[2], infer_images_norm.shape[3])
    else:
        input_shape = (infer_images_norm.shape[1], infer_images_norm.shape[2])

    i=0
    for x_train, y_train, x_val, y_val in iter(dataset):
        wandb_config(model_name, num_layers=num_layers, hidden_size=hidden_size, validation_fold=i)
        if model_name == 'gru_custom':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
        elif model_name == 'lstm_custom':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
        elif model_name=='vit_tiny_custom':
            model = vit.vit_tiny_custom(
                input_shape=input_shape,
                patch_size=3,
                num_patches=infer_images_norm.shape[2]//3**2,
                classes=num_classes,
                activation='linear',
                pretrained=True,
                include_top=True,
                pretrained_top=True,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                hidden_size=hidden_size
            )
        model.summary()
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        def root_mean_squared_log_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name="rmse")
            ],
        )
        checkpoint = ModelCheckpoint(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size), 'fold_'+str(i)),
                                     monitor="val_loss", mode="min", save_best_only=True, verbose=1)

        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=MAX_EPOCHS,
            callbacks=[checkpoint, WandbCallback()],
        )
        # model.save(os.path.join(root_path, 'proj3_' + model_name + str(hidden_size) + '_' + str(num_layers)))

        model.load_weights(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size), 'fold_'+str(i)))

        pred_giz = model.predict(infer_images_norm)
        ID_S2_pair = pd.read_csv('africa-biomass-challenge/UniqueID-SentinelPair.csv')
        preds = pd.DataFrame({'Target': pred_giz[:,0]}).rename_axis('S2_idx').reset_index()
        preds = ID_S2_pair.merge(preds, on='S2_idx').drop(columns=['S2_idx'])
        if not os.path.exists('africa-biomass-challenge/predictions'):
            os.mkdir('africa-biomass-challenge/predictions')
        preds.to_csv('africa-biomass-challenge/predictions/biomass_predictions'+ model_name + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)+'_fold_'+str(i)+'.csv', index=False)
        i+=1