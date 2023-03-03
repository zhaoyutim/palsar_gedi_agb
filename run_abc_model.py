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

warnings.filterwarnings('ignore')
if platform.system() == 'Darwin':
    root_path = '/Users/zhaoyu/PycharmProjects/palsar_gedi_agb/africa-biomass-challenge'
else:
    root_path = '/geoinfo_vol1/zhao2/abc_challenge_models'

def wandb_config(model_name, num_layers, hidden_size):
    wandb.login()
    wandb.init(project="abc_challenge_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_layers_'+ str(num_layers)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "embed_dim": hidden_size
    }

def get_dataset():
    trainset = h5py.File("africa-biomass-challenge/09072022_1154_train.h5", "r")
    validateset = h5py.File("africa-biomass-challenge/09072022_1154_val.h5", "r")
    testset = h5py.File("africa-biomass-challenge/09072022_1154_test.h5", "r")
    def feature_engineering(array):
        dvi = array[:, [7], :, :] / array[:, [3], :, :]
        ndvi = (array[:, [7], :, :] - array[:, [3], :, :]) / (array[:, [7], :, :] + array[:, [3], :, :] + 1e6)
        ndvi2 = (array[:, [8], :, :] - array[:, [3], :, :]) / (array[:, [8], :, :] + array[:, [3], :, :] + 1e6)
        gndvi = (array[:, [7], :, :] - array[:, [2], :, :]) / (array[:, [7], :, :] + array[:, [2], :, :] + 1e6)
        ndi45 = (array[:, [4], :, :] - array[:, [3], :, :]) / (array[:, [4], :, :] + array[:, [3], :, :] + 1e6)
        ndre = (array[:, [7], :, :] - array[:, [4], :, :]) / (array[:, [7], :, :] + array[:, [4], :, :] + 1e6)
        array = np.concatenate([array, dvi, ndvi, ndvi2, gndvi, ndi45, ndre], axis=1)
        return array

    # train
    train_images = np.array(trainset['images'], dtype=np.float64)
    train_images = train_images.transpose(0, 3, 1, 2)
    train_biomasses = np.array(trainset['agbd'], dtype=np.float64)
    train_images = feature_engineering(train_images)

    # validate
    validate_images = np.array(validateset['images'], dtype=np.float64)
    validate_images = validate_images.transpose(0, 3, 1, 2)
    validate_biomasses = np.array(validateset['agbd'], dtype=np.float64)
    validate_images = feature_engineering(validate_images)

    # test
    test_images = np.array(testset['images'], dtype=np.float32)
    test_images = test_images.transpose(0, 3, 1, 2)
    test_biomasses = np.array(testset['agbd'], dtype=np.float32)
    test_images = feature_engineering(test_images)

    # infer
    infer_images = h5py.File("africa-biomass-challenge/images_test.h5", "r")
    infer_images = np.array(infer_images["images"])
    infer_images = infer_images.transpose(0, 3, 1, 2)
    infer_images = feature_engineering(infer_images)

    MEAN = train_images.mean((0, 2, 3))
    STD = train_images.std((0, 2, 3))
    train_images_norm = (train_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
    validate_images_norm = (validate_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
    test_images_norm = (test_images - MEAN[None, :, None, None]) / STD[None, :, None, None]
    infer_images_norm = (infer_images - MEAN[None, :, None, None]) / STD[None, :, None, None]

    train_images_norm = train_images_norm.reshape((train_images_norm.shape[0], train_images_norm.shape[1], -1))
    validate_images_norm = validate_images_norm.reshape((validate_images_norm.shape[0], validate_images_norm.shape[1], -1))
    test_images_norm = test_images_norm.reshape((test_images_norm.shape[0], test_images_norm.shape[1], -1))
    infer_images_norm = infer_images_norm.reshape((infer_images_norm.shape[0], infer_images_norm.shape[1], -1))

    return train_images_norm, train_biomasses, validate_images_norm, validate_biomasses, test_images_norm, test_biomasses, infer_images_norm

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')
    args = parser.parse_args()
    model_name = args.m
    batch_size = args.b
    num_layers=args.nl
    hidden_size=args.ed

    num_classes = 1

    lr = args.lr
    learning_rate = lr
    weight_decay = lr/10
    MAX_EPOCHS = 50

    train_images_norm, train_biomasses, validate_images_norm, validate_biomasses, test_images_norm, test_biomasses, infer_images_norm = get_dataset()
    input_shape = (train_images_norm.shape[1], train_images_norm.shape[2])
    wandb_config(model_name, num_layers=num_layers, hidden_size=hidden_size)


    if model_name == 'gru_custom':
        gru = GRUModel(input_shape, num_classes)
        model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
    elif model_name == 'lstm_custom':
        lstm = LSTMModel(input_shape, num_classes)
        model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequences=False)
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
    checkpoint = ModelCheckpoint(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + model_name + str(hidden_size) + '_' + str(num_layers)),
                                 monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit(
        x=train_images_norm,
        y=train_biomasses,
        validation_data=(validate_images_norm, validate_biomasses),
        epochs=MAX_EPOCHS,
        callbacks=[checkpoint, WandbCallback()],
    )
    # model.save(os.path.join(root_path, 'proj3_' + model_name + str(hidden_size) + '_' + str(num_layers)))

    model.load_weights(os.path.join(root_path.replace('africa-biomass-challenge', 'abc_challenge_models'), 'abc_' + model_name + str(hidden_size) + '_' + str(num_layers)))

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
    preds.to_csv('africa-biomass-challenge/predictions/biomass_predictions'+ model_name + str(hidden_size) + '_' + str(num_layers)+'.csv', index=False)
