from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed, MaxPooling2D, ConvLSTM2D, UpSampling2D, \
    Convolution2D, Concatenate, Conv2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf


def get_convlstm_unet1(input_shape):
    inputs = Input(input_shape)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    # conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    convlstm1 = ConvLSTM2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    # conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    convlstm2 = ConvLSTM2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    # conv3 =TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
    convlstm3 = ConvLSTM2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    # conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)
    # convlstm4 = ConvLSTM2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv4)
    # drop4 = TimeDistributed(Dropout(0.5))(conv4)
    # pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop4)

    # conv5 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool4)
    # conv5 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
    # drop5 = TimeDistributed(Dropout(0.5))(conv5)

    # up6 = TimeDistributed(Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(drop4)
    # merge6 = tf.concat([conv4,up6], axis = 4)
    conv6 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)
    # conv6 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv6)

    up7 = TimeDistributed(Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv6))
    merge7 = tf.concat([convlstm3,up7], axis = 4)
    conv7 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge7)
    conv7 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv7)

    up8 = TimeDistributed(Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv7))
    merge8 = tf.concat([convlstm2,up8], axis = 4)
    conv8 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge8)
    conv8 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv8)

    up9 = TimeDistributed(Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv8))
    merge9 = tf.concat([convlstm1,up9], axis = 4)
    conv9 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge9)
    conv9 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv9 = TimeDistributed(Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv10 = TimeDistributed(Conv2D(1, 1, activation = 'sigmoid'))(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model

def get_convlstm_unet2(input_shape):
    inputs = Input(input_shape)
    conv1 = ConvLSTM2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(inputs)
    # conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = ConvLSTM2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(pool1)
    # conv2 = ConvLSTM2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = ConvLSTM2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(pool2)
    # conv3 = ConvLSTM2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv6 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)

    up7 = TimeDistributed(Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv6))
    merge7 = tf.concat([conv3,up7], axis = 4)
    conv7 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge7)
    conv7 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv7)

    up8 = TimeDistributed(Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv7))
    merge8 = tf.concat([conv2,up8], axis = 4)
    conv8 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge8)
    conv8 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv8)

    up9 = TimeDistributed(Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv8))
    merge9 = tf.concat([conv1,up9], axis = 4)
    conv9 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge9)
    conv9 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv9 = TimeDistributed(Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv10 = TimeDistributed(Conv2D(1, 1, activation = 'sigmoid'))(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model

def unet(input_size = (256,256,5)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.concat([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.concat([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.concat([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = tf.concat([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model