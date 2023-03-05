import argparse
import os
import platform

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

import wandb
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_addons as tfa
from wandb.integration.keras import WandbCallback

from model.gru.gru_model import GRUModel
from model.lstm.lstm_model import LSTMModel
from model.vit_keras import vit

warnings.filterwarnings('ignore')
if platform.system() == 'Darwin':
    root_path = '/Users/zhaoyu/PycharmProjects/palsar_gedi_agb/africa-biomass-challenge'
else:
    root_path = '/geoinfo_vol1/zhao2/abc_challenge_models'

def wandb_config(model_name, num_layers, hidden_size):
    wandb.login()
    wandb.init(project="abc_challenge_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)
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
    def feature_engineering(array, scl, cloud, lat, lon):
        dvi = array[:, :, :, [7]] / (array[:, :, :, [3]] + 1e6)
        ndvi = (array[:, :, :, [7]] - array[:, :, :, [3]]) / (array[:, :, :, [7]] + array[:, :, :, [3]] + 1e6)
        ndvi2 = (array[:, :, :, [8]] - array[:, :, :, [3]]) / (array[:, :, :, [8]] + array[:, :, :, [3]] + 1e6)
        gndvi = (array[:, :, :, [7]] - array[:, :, :, [2]]) / (array[:, :, :, [7]] + array[:, :, :, [2]] + 1e6)
        ndi45 = (array[:, :, :, [4]] - array[:, :, :, [3]]) / (array[:, :, :, [4]] + array[:, :, :, [3]] + 1e6)
        ndre = (array[:, :, :, [7]] - array[:, :, :, [4]]) / (array[:, :, :, [7]] + array[:, :, :, [4]] + 1e6)
        array = np.concatenate([array, scl, cloud, lat, lon, dvi, ndvi, ndvi2, gndvi, ndi45, ndre], axis=3)
        if channel_first:
            array = array.transpose(0, 3, 1, 2)
        return array
    def patch_dividing(array):
        patches = tf.image.extract_patches(
            images=array,
            sizes=[1, 5, 5, 1],
            strides=[1, 5, 5, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return patches.numpy().reshape(patches.shape[0], -1, patches.shape[3])
    # train
    train_images = np.array(trainset['images'], dtype=np.float64)
    train_scl = np.array(trainset['scl'], dtype=np.float64)
    train_cloud = np.array(trainset['cloud'], dtype=np.float64)
    train_lat = np.array(trainset['lat'], dtype=np.float64)
    train_lon = np.array(trainset['lon'], dtype=np.float64)
    train_biomasses = np.array(trainset['agbd'], dtype=np.float64)
    train_images = feature_engineering(train_images, train_scl, train_cloud, train_lat, train_lon)
    if not channel_first:
        MEAN = train_images.mean((0, 1, 2))
        STD = train_images.std((0, 1, 2))
        train_images_norm = (train_images - MEAN[None, None, None, :]) / STD[None, None, None, :]
        train_images_norm = patch_dividing(train_images_norm)
    else:
        MEAN = train_images.mean((0, 2, 3))
        STD = train_images.std((0, 2, 3))
        train_images_norm = (train_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
        train_images_norm = train_images_norm.reshape((train_images_norm.shape[0], train_images_norm.shape[1], -1))

    # validate
    validate_images = np.array(validateset['images'], dtype=np.float64)
    validate_scl = np.array(validateset['scl'], dtype=np.float64)
    validate_cloud = np.array(validateset['cloud'], dtype=np.float64)
    validate_lat = np.array(validateset['lat'], dtype=np.float64)
    validate_lon = np.array(validateset['lon'], dtype=np.float64)
    validate_biomasses = np.array(validateset['agbd'], dtype=np.float64)
    validate_images = feature_engineering(validate_images, validate_scl, validate_cloud, validate_lat, validate_lon)
    if not channel_first:
        validate_images_norm = (validate_images - MEAN[None, None, None, :]) / STD[None, None, None, :]
        validate_images_norm = patch_dividing(validate_images_norm)
    else:
        validate_images_norm = (validate_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
        validate_images_norm = validate_images_norm.reshape(
            (validate_images_norm.shape[0], validate_images_norm.shape[1], -1))


    # test
    test_images = np.array(testset['images'], dtype=np.float32)
    test_scl = np.array(testset['scl'], dtype=np.float64)
    test_cloud = np.array(testset['cloud'], dtype=np.float64)
    test_lat = np.array(testset['lat'], dtype=np.float64)
    test_lon = np.array(testset['lon'], dtype=np.float64)
    test_biomasses = np.array(testset['agbd'], dtype=np.float32)
    test_images = feature_engineering(test_images, test_scl, test_cloud, test_lat, test_lon)
    if not channel_first:
        test_images_norm = (test_images - MEAN[None, None, None, :]) / STD[None, None, None, :]
        test_images_norm = patch_dividing(test_images_norm)
    else:
        test_images_norm = (test_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
        test_images_norm = test_images_norm.reshape((test_images_norm.shape[0], test_images_norm.shape[1], -1))

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
    infer_images = feature_engineering(infer_images, infer_scl, infer_cloud, infer_lat, infer_lon)
    if not channel_first:
        infer_images_norm = (infer_images - MEAN[None, None, None, :]) / STD[None, None, None, :]
        infer_images_norm = patch_dividing(infer_images_norm)
    else:
        infer_images_norm = (infer_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
        infer_images_norm = infer_images_norm.reshape((infer_images_norm.shape[0], infer_images_norm.shape[1], -1))

    return train_images_norm, train_biomasses, validate_images_norm, validate_biomasses, test_images_norm, test_biomasses, infer_images_norm

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
    MAX_EPOCHS = 50

    train_images_norm, train_biomasses, validate_images_norm, validate_biomasses, test_images_norm, test_biomasses, infer_images_norm = get_dataset(channel_first=False)
    input_shape = (train_images_norm.shape[1], train_images_norm.shape[2])
    wandb_config(model_name, num_layers=num_layers, hidden_size=hidden_size)


    if model_name == 'gru_custom':
        gru = GRUModel(input_shape, num_classes)
        model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
    elif model_name == 'lstm_custom':
        lstm = LSTMModel(input_shape, num_classes)
        model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
    elif model_name=='vit_tiny_custom':
        model = vit.vit_tiny_custom(
            input_shape=input_shape,
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
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ],
    )
    checkpoint = ModelCheckpoint(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)),
                                 monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit(
        x=train_images_norm,
        y=train_biomasses,
        validation_data=(validate_images_norm, validate_biomasses),
        epochs=MAX_EPOCHS,
        callbacks=[checkpoint, WandbCallback()],
    )
    # model.save(os.path.join(root_path, 'proj3_' + model_name + str(hidden_size) + '_' + str(num_layers)))

    model.load_weights(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)))

    score = model.evaluate(test_images_norm, test_biomasses, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    wandb.run.log({'Test loss': score[0], 'Test accuracy': score[1]})

    pred_giz = model.predict(infer_images_norm)
    ID_S2_pair = pd.read_csv('africa-biomass-challenge/UniqueID-SentinelPair.csv')
    preds = pd.DataFrame({'Target': pred_giz[:,0]}).rename_axis('S2_idx').reset_index()
    preds = ID_S2_pair.merge(preds, on='S2_idx').drop(columns=['S2_idx'])
    if not os.path.exists('africa-biomass-challenge/predictions'):
        os.mkdir('africa-biomass-challenge/predictions')
    preds.to_csv('africa-biomass-challenge/predictions/biomass_predictions'+ model_name + 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)+'.csv', index=False)
