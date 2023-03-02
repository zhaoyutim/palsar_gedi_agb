import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TokenizeLayer(layers.Layer):
    def __init__(self, window_size, batch_size):
        super(TokenizeLayer, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(45, window_size, padding='same')
        self.batch_size = batch_size

    def call(self, batch):
        tokenized = self.conv2d(batch)
        tokenized = tf.reshape(tokenized, (-1, 10, 45))
        return tokenized


if __name__=='__main__':
    tokenize = TokenizeLayer(3, 1)
    data = np.load('/Users/zhaoyu/PycharmProjects/ViirsTimeSeriesModel/data/proj3_test_img.npy')
    print(data.shape)
    data = data.transpose((2, 3, 0, 1))
    print(data.shape)
    print(tokenize(data[np.newaxis,np.newaxis, :, :, :, :5]))