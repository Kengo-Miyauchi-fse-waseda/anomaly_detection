import sys
sys.path.append('.')
import matplotlib as mpl
mpl.use('Agg')
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras import backend
from keras.models import load_model
import numpy as np
from logging import getLogger, StreamHandler, DEBUG
import os
from util.preparing_dir import preparing_dir

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class DeepAutoencoderFive(object):
    def __init__(self):
        pass

    def structure(self, bottleneck, input_dim=75, mid=128, optimizer='SGD', loss='mse'):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(mid, activation='relu')(input_layer)
        encoded = Dense(bottleneck, activation='relu')(encoded)
        decoded = Dense(mid, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='relu')(decoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder


class DeepAutoencoderFiveBottleneck(object):
    def __init__(self, bottleneck=8, input_dim=15, epochs=10, batch_size=256, shuffle=True, mid1=128,
                 model_save_dir='', optimizer='adam', loss='mse'):
        self._input_extractor = None
        self._output_extractor = None
        self._bottleneck = bottleneck
        self._mid1 = mid1
        self._epochs = epochs
        self._autoencoder = None
        self._model_save_dir = model_save_dir
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(self._bottleneck, self._mid1, self._epochs)
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._input_dim = input_dim
        self._optimizer = optimizer
        self._loss = loss

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, input_dim=75, mid1=128):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(mid1, activation='relu', kernel_initializer='he_normal')(input_layer)
        encoded = Dropout(0.25, seed=1234)(encoded)
        encoded = Dense(bottleneck, activation='relu', kernel_initializer='he_normal', name='bottleneck')(encoded)
        decoded = Dropout(0.25, seed=1234)(encoded)
        decoded = Dense(mid1, activation='relu', kernel_initializer='he_normal')(decoded)
        decoded = Dense(input_dim, activation='sigmoid', kernel_initializer='he_normal')(decoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=self._optimizer, loss=self._loss)
        self._autoencoder = autoencoder
        print(autoencoder.summary())
        return None

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(self._bottleneck, input_dim=self._input_dim, mid1=self._mid1)
        if not os.path.isfile(self._save) or force_training:
            self._autoencoder.fit(x_train, y_train,
                                  epochs=self._epochs,
                                  batch_size=self._batch_size,
                                  shuffle=self._shuffle,
                                  validation_data=(x_test, y_test))
            self._autoencoder.save(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottleneck').output)
            return self._autoencoder, self._output_extractor
        else:
            self._autoencoder = load_model(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottleneck').output)
            return self._autoencoder, self._output_extractor

    def model_save(self, save_dir, roc, f_value):
        preparing_dir(save_dir)
        save_file_name = 'bn{}_{}_{}epochs_AUC_{:.4f}_F_{:.4f}.h5'.format(self._bottleneck, self._mid1,
                                                                          self._epochs, roc, f_value)
        save_path = os.path.join(save_dir, save_file_name)
        self._autoencoder.save(save_path)


class DeepAutoencoderSevenBottleneck(object):
    def __init__(self, bottleneck=8, input_dim=15, epochs=10, batch_size=256, shuffle=True, mid1=128, mid2=256,
                 model_save_dir='', optimizer='adam', loss='mse'):
        self._input_extractor = None
        self._output_extractor = None
        self._bottleneck = bottleneck
        self._mid1 = mid1
        self._mid2 = mid2
        self._epochs = epochs
        self._autoencoder = None
        self._model_save_dir = model_save_dir
        self._save_file_name = 'bn{}_{}_{}_{}epochs.h5'.format(self._bottleneck, self._mid1, self._mid2, self._epochs)
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._input_dim = input_dim
        self._optimizer = optimizer
        self._loss = loss

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, input_dim=75, mid1=128, mid2=256):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(mid1, activation='relu', kernel_initializer='he_normal')(input_layer)
        encoded = Dense(mid2, activation='relu', kernel_initializer='he_normal')(encoded)
        encoded = Dense(bottleneck, activation='relu', kernel_initializer='he_normal', name='bottleneck')(encoded)
        decoded = Dense(mid2, activation='relu', kernel_initializer='he_normal')(encoded)
        decoded = Dense(mid1, activation='relu', kernel_initializer='he_normal')(decoded)
        decoded = Dense(input_dim, activation='sigmoid', kernel_initializer='he_normal')(decoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=self._optimizer, loss=self._loss)
        self._autoencoder = autoencoder
        print(autoencoder.summary())
        return None

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(self._bottleneck, input_dim=self._input_dim, mid1=self._mid1, mid2=self._mid2)
        if not os.path.isfile(self._save) or force_training:
            self._autoencoder.fit(x_train, y_train,
                                  epochs=self._epochs,
                                  batch_size=self._batch_size,
                                  shuffle=self._shuffle,
                                  validation_data=(x_test, y_test))
            self._autoencoder.save(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottleneck').output)
            return self._autoencoder, self._output_extractor
        else:
            print('Loading autoencoder')
            self._autoencoder = load_model(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottleneck').output)
            return self._autoencoder, self._output_extractor

    def model_save(self, save_dir, roc, f_value):
        preparing_dir(save_dir)
        save_file_name = 'bn{}_{}_{}_{}epochs_AUC_{:.4f}_F_{:.4f}.h5'.format(self._bottleneck, self._mid1,
                                                                             self._mid2, self._epochs, roc, f_value)
        save_path = os.path.join(save_dir, save_file_name)
        self._autoencoder.save(save_path)


class DeepAutoencoderSevenBottleneckBn(object):
    def __init__(self, bottleneck=8, input_dim=15, epochs=10, batch_size=512, shuffle=True, mid1=128, mid2=256,
                 model_save_dir='', save_file_name=''):
        self._input_extractor = None
        self._output_extractor = None
        self._bottleneck = bottleneck
        self._mid1 = mid1
        self._mid2 = mid2
        self._epochs = epochs
        self._autoencoder = None
        self._model_save_dir = model_save_dir
        self._save_file_name = 'batch_bn{}_{}_{}_{}epochs.h5'.format(self._bottleneck, self._mid1, self._mid2, self._epochs)
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._input_dim = input_dim

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, input_dim=75, mid1=128, mid2=256, optimizer='adam', loss='mse'):
        input_layer = Input(shape=(input_dim,))
        encoded1 = Dense(mid1, activation='relu')(input_layer)
        encoded1_bn = Dropout(.3)(encoded1)
        encoded2 = Dense(mid2, activation='relu')(encoded1_bn)
        encoded2_bn = Dropout(.3)(encoded2)
        bottle_neck = Dense(bottleneck, activation='relu', name='bottle_neck')(encoded2_bn)
        decoded1 = Dense(mid2, activation='relu')(bottle_neck)
        decoded1_dp = Dropout(.3)(decoded1)
        decoded2 = Dense(mid1, activation='relu')(decoded1_dp)
        decoded2_dp = Dropout(.3)(decoded2)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2_dp)
        autoencoder = Model(input_layer, decoded3)
        autoencoder.compile(optimizer=optimizer, loss=loss)
        self._autoencoder = autoencoder
        print(self._autoencoder.summary())
        return autoencoder

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(self._bottleneck, input_dim=self._input_dim, mid1=self._mid1, mid2=self._mid2)
        if not os.path.isfile(self._save) or force_training:
            self._autoencoder.fit(x_train, y_train,
                                  epochs=self._epochs,
                                  batch_size=self._batch_size,
                                  shuffle=self._shuffle,
                                  validation_data=(x_test, y_test))
            self._autoencoder.save(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottle_neck').output)
            self._input_extractor = Model(inputs=self._autoencoder.input,
                                          outputs=self._autoencoder.get_layer('bottle_neck').input)
            return self._input_extractor, self._output_extractor
        else:
            self._autoencoder = load_model(self._save)
            self._output_extractor = Model(inputs=self._autoencoder.input,
                                           outputs=self._autoencoder.get_layer('bottle_neck').output)
            self._input_extractor = Model(inputs=self._autoencoder.input,
                                          outputs=self._autoencoder.get_layer('bottle_neck').input)
            return self._input_extractor, self._output_extractor
