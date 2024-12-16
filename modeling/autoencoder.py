import sys
sys.path.append('.')
import matplotlib as mpl
mpl.use('Agg')
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
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

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class ConventionalAutoencoder(object):
    def __init__(self, epochs):
        self._epochs = epochs

    def structure(self, bottleneck):
        input_layer = Input(shape=(75,))
        encoded = Dense(bottleneck, activation='relu')(input_layer)
        decoded = Dense(75, activation='relu')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adadelta', loss='mse')
        return autoencoder
