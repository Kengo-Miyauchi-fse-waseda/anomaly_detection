import os
from keras import backend
from keras import Sequential
from keras import callbacks
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from util.preparing_dir import preparing_dir


class ClassificationNetwork(object):
    def __init__(self, model_save_dir='', save_file_name='', bottleneck=16, input_dim=75,
                 epochs=10, batch_size=256, shuffle=True, sparse_units=1024):
        self._model = None
        self._extractor = None
        self._model_save_dir = model_save_dir
        self._sparse_units = sparse_units
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(bottleneck, sparse_units, epochs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._bottleneck = bottleneck
        self._input_dim = input_dim

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, sparse=1024):
        num_bottleneck = int(bottleneck)
        inputs = Input(shape=(self._input_dim, ))
        x = Dense(sparse, activation='relu')(inputs)
        x = Dense(sparse, activation='relu')(x)
        x = Dense(sparse, activation='relu')(x)
        bottleneck = Dense(num_bottleneck, activation='relu')(x)
        x = Dense(sparse, activation='relu')(bottleneck)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=prediction)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        extractor = Model(inputs=inputs, outputs=bottleneck)
        self._model = model
        self._extractor = extractor

    def training(self, x_train, y_train, x_test, y_test):
        backend.clear_session()
        self._structure(bottleneck=self._bottleneck, sparse=self._sparse_units)
        if not os.path.isfile(self._save):
            self._model.fit(x_train, y_train,
                            epochs=self._epochs,
                            batch_size=self._batch_size,
                            shuffle=self._shuffle,
                            validation_data=(x_test, y_test))
            self._model.save(self._save)
            return self._model, self._extractor
        else:
            self._model = load_model(self._save)
            return self._model, self._extractor


class TempClassificationNetwork(object):
    def __init__(self, model_save_dir='', save_file_name='', bottleneck=16, input_dim=75,
                 epochs=10, batch_size=256, shuffle=True, sparse_units=1024):
        self._model = None
        self._extractor = None
        self._model_save_dir = model_save_dir
        self._sparse_units = sparse_units
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(bottleneck, sparse_units, epochs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._bottleneck = bottleneck
        self._input_dim = input_dim

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, sparse=1024):
        self._bottleneck = int(bottleneck)
        self._sparse_units = sparse
        inputs = Input(shape=(self._input_dim, ))
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(inputs)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(x)
        bottleneck = Dense(self._bottleneck, activation='relu', kernel_initializer='he_normal', name='bottleneck')(x)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(bottleneck)
        prediction = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
        model = Model(inputs=inputs, outputs=prediction)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        extractor = Model(inputs=inputs, outputs=bottleneck)
        self._model = model
        self._extractor = extractor

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(bottleneck=self._bottleneck, sparse=self._sparse_units)
        if (not os.path.isfile(self._save)) or force_training:
            self._model.fit(x_train, y_train,
                            epochs=self._epochs,
                            batch_size=self._batch_size,
                            shuffle=self._shuffle,
                            validation_data=(x_test, y_test))
            self._model.save(self._save)
            return self._model, self._extractor
        else:
            self._model = load_model(self._save)
            self._extractor = Model(inputs=self._model.input,
                                    outputs=self._model.get_layer('bottleneck').output)
            return self._model, self._extractor

    def model_save(self, save_dir, roc, f_value):
        preparing_dir(save_dir)
        save_file_name = 'bn{}_{}_{}epochs_AUC_{:.4f}_F_{:.4f}.h5'.format(self._bottleneck, self._sparse_units,
                                                                          self._epochs, roc, f_value)
        save_path = os.path.join(save_dir, save_file_name)
        self._model.save(save_path)


class ClassificationNetworkVer2(object):
    def __init__(self, model_save_dir='', save_file_name='', bottleneck=16, input_dim=75,
                 epochs=10, batch_size=256, shuffle=True, sparse_units=1024):
        self._model = None
        self._extractor = None
        self._model_save_dir = model_save_dir
        self._sparse_units = sparse_units
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(bottleneck, sparse_units, epochs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._bottleneck = bottleneck
        self._input_dim = input_dim

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, sparse=1024):
        self._bottleneck = int(bottleneck)
        self._sparse_units = sparse
        inputs = Input(shape=(self._input_dim, ))
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(inputs)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(x)
        bottleneck = Dense(self._bottleneck, activation='relu', kernel_initializer='he_normal', name='bottleneck')(x)
        x = Dense(self._sparse_units, activation='relu', kernel_initializer='he_normal')(bottleneck)
        prediction = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
        model = Model(inputs=inputs, outputs=prediction)
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])
        extractor = Model(inputs=inputs, outputs=bottleneck)
        self._model = model
        self._extractor = extractor

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(bottleneck=self._bottleneck, sparse=self._sparse_units)
        if (not os.path.isfile(self._save)) or force_training:
            self._model.fit(x_train, y_train,
                            epochs=self._epochs,
                            batch_size=self._batch_size,
                            shuffle=self._shuffle,
                            validation_data=(x_test, y_test))
            self._model.save(self._save)
            return self._model, self._extractor
        else:
            self._model = load_model(self._save)
            self._extractor = Model(inputs=self._model.input,
                                    outputs=self._model.get_layer('bottleneck').output)
            return self._model, self._extractor

    def model_save(self, save_dir, roc, f_value):
        preparing_dir(save_dir)
        save_file_name = 'bn{}_{}_{}epochs_AUC_{:.4f}_F_{:.4f}.h5'.format(self._bottleneck, self._sparse_units,
                                                                          self._epochs, roc, f_value)
        save_path = os.path.join(save_dir, save_file_name)
        self._model.save(save_path)


class KerasClassificationNetwork19(object):
    def __init__(self, model_save_dir='', save_file_name='', bottleneck=16, input_dim=75,
                 epochs=10, batch_size=256, shuffle=True, sparse_units=1024):
        self._model = None
        self._extractor = None
        self._model_save_dir = model_save_dir
        self._sparse_units = sparse_units
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(bottleneck, sparse_units, epochs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._bottleneck = bottleneck
        self._input_dim = input_dim

        preparing_dir(self._model_save_dir)

    def _structure(self, bottleneck, sparse=1024):
        num_bottleneck = int(bottleneck)
        inputs = Input(shape=(self._input_dim,))
        x = Dense(sparse, activation='relu', kernel_initializer='he_normal')(inputs)
        x = Dense(sparse, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(sparse, activation='relu', kernel_initializer='he_normal')(x)
        bottleneck = Dense(num_bottleneck, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(sparse, activation='relu', kernel_initializer='he_normal')(bottleneck)
        prediction = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        model = Model(inputs=inputs, outputs=prediction)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        extractor = Model(inputs=inputs, outputs=bottleneck)
        self._model = model
        self._extractor = extractor

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        # backend.clear_session()
        self._structure(bottleneck=self._bottleneck, sparse=self._sparse_units)
        if (not os.path.isfile(self._save)) or force_training:
            self._model.fit(x_train, y_train,
                            epochs=self._epochs,
                            batch_size=self._batch_size,
                            shuffle=self._shuffle,
                            validation_data=(x_test, y_test))
            self._model.save(self._save)
            return self._model, self._extractor
        else:
            self._model = load_model(self._save)
            self._extractor = Model(inputs=self._model.input,
                                    outputs=self._model.get_layer('dense_4').output)
            return self._model, self._extractor


class ConvolutionalNeuralNework(object):
    def __init__(self, input_dim: int, model_save_dir='', bottleneck=16,
                 epochs=30, batch_size=512, shuffle=True,):
        self._model_save_dir = model_save_dir
        preparing_dir(self._model_save_dir)
        self._bottleneck = bottleneck
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._save_file_name = 'bn{}_{}epochs.h5'.format(bottleneck, epochs)
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._model = self._build_model(input_dim, bottleneck)
        self._extractor = None

    def _build_model(self, input_dim: int, bottleneck: int):
        input_layer = Input(shape=(input_dim, 1))
        x = Conv1D(15, 5, strides=3, activation='relu')(input_layer)
        x = Conv1D(10, 5, strides=3, activation='relu')(x)
        x = Conv1D(10, 5, strides=3, activation='relu')(x)
        x = Conv1D(5, 5, strides=2, activation='relu', name='last_conv_layer')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(bottleneck, activation='relu', name='bnf')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(input_layer, x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        if (not os.path.isfile(self._save)) or force_training:
            self._model.fit(x_train, y_train,
                            epochs=self._epochs,
                            batch_size=self._batch_size,
                            shuffle=self._shuffle,
                            validation_data=(x_test, y_test))
            self._model.save(self._save)
        else:
            self._model = load_model(self._save)
        self._extractor = Model(inputs=self._model.input,
                                outputs=self._model.get_layer('bnf').output)
        return self._model, self._extractor
