import keras.backend as K
import tensorflow as tf


class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def get_model_custom(self, input_shape, num_class, num_layers, hidden_size, return_sequences=False):
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(hidden_size, input_shape=input_shape, return_sequences=return_sequences),
        ])
        for i in range(num_layers-1):
            lstm_model.add(tf.keras.layers.LSTM(hidden_size, dropout=0.2,return_sequences=return_sequences))
        lstm_model.add(tf.keras.layers.Dense(num_class,activation='linear'))
        return lstm_model