import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow.python.keras.utils.version_utils import callbacks


class ValidationAccuracy(callbacks.Callback):
    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, batch, logs={}):

        self.scores = {
            'recall_score': [],
            'precision_score': [],
            'f1_score': []
        }

        for batch_index in range(self.validation_steps):
            features, y_true = next(self.validation_generator)
            y_pred = np.asarray(self.model.predict(features))
            y_pred = y_pred.round().astype(int)
            self.scores['recall_score'].append(recall_score(y_true[:,0], y_pred[:,0]))
            self.scores['precision_score'].append(precision_score(y_true[:,0], y_pred[:,0]))
            self.scores['f1_score'].append(f1_score(y_true[:,0], y_pred[:,0]))
        return