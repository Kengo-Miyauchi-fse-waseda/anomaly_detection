import sys
sys.path.append('.')
import matplotlib as mpl
mpl.use('Agg')
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Reshape
from keras.models import Model
from keras import backend
from keras.models import load_model
import numpy as np
from logging import getLogger, StreamHandler, DEBUG
from multiprocessing import Pool
from util.manage_list import unroll
import gc
from loading.feature import Ishibashi
from loading.feature import GetFeature
from scoring.score import Score
from visualization.plot_eer import PlottingAutoencoder
import argparse
import os
from util.preparing_dir import preparing_dir

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class DeepConvolutionalAutoencoder(object):
    def __init__(self, bottleneck=8, input_dim=15, epochs=30, batch_size=256, shuffle=True, mid1=128, mid2=256,
                 model_save_dir='', save_file_name=''):
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

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, input_dim=75, mid1=50, mid2=25, optimizer='adam', loss='mse'):
        pool_size = 4
        input_layer = Input(batch_shape=(None, input_dim, 1))
        encoded = Conv1D(filters=mid1, kernel_size=8, strides=pool_size, padding='same', activation='relu')(input_layer)
        #encoded = MaxPooling1D(pool_size=pool_size, strides=None, padding='valid')(encoded)
        #encoded = Conv1D(filters=mid2, kernel_size=5, strides=2, activation='relu')(encoded)
        #encoded = MaxPooling1D(pool_size=pool_size, strides=None, padding='valid')(encoded)
        bottle_neck = Conv1D(filters=bottleneck, kernel_size=8, strides=pool_size, 
                             padding='same', activation='relu', name='bottleneck')(encoded)
        #bottle_neck = MaxPooling1D(pool_size=pool_size, strides=None, padding='valid', name='bottleneck')(encoded)

        decoded = Conv1D(filters=bottleneck, kernel_size=8, padding='same', activation='relu')(bottle_neck)
        decoded = UpSampling1D(size=pool_size)(decoded)
        #decoded = Conv1D(filters=mid2, kernel_size=5, activation='relu')(decoded)
        #decoded = UpSampling1D(size=pool_size)(decoded)
        decoded = Conv1D(filters=mid1, kernel_size=8, padding='same', activation='relu')(decoded)
        decoded = UpSampling1D(size=pool_size)(decoded)

        decoded = Conv1D(filters=1, kernel_size=1, padding='same',  activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=optimizer, loss=loss)
        print(autoencoder.summary())
        self._autoencoder = autoencoder
        return autoencoder

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        if not os.path.isfile(self._save) or force_training:
            self._structure(self._bottleneck, input_dim=self._input_dim, mid1=self._mid1, mid2=self._mid2)
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
