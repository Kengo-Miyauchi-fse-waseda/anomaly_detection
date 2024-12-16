from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from util.preparing_dir import preparing_dir
import os


class VariationalAutoencoder(object):
    def __init__(self, bottleneck=8, input_dim=75, epochs=10, batch_size=256, shuffle=True, mid1=64, mid2=32,
                 model_save_dir='', save_file_name=''):
        self._bottleneck = bottleneck
        self._mid1 = mid1
        self._mid2 = mid2
        self._epochs = epochs
        self._autoencoder = None
        self._model_save_dir = model_save_dir
        self._save_file_name = 'bn{}_{}_{}epochs.h5'.format(self._bottleneck, self._mid1, self._epochs)
        self._save = os.path.join(self._model_save_dir, self._save_file_name)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._input_dim = input_dim
        self._epsilon_std = 1.0

        preparing_dir(self._model_save_dir)

    def build_encoder(self):
        x = Input(shape=(self._input_dim,))
        hidden = Dense(self._mid1, activation='relu')(x)
        hidden = Dense(self._mid2, activation='relu')(hidden)
        z_mean = Dense(self._bottleneck, activation='linear', name='z_m')(hidden)
        z_sigma = Dense(self._bottleneck, activation='linear', name='z_sigma')(hidden)
        return Model(x, [z_mean, z_sigma])

    def build_decoder(self):
        z_mean = Input(shape=(self._bottleneck, ))
        z_sigma = Input(shape=(self._bottleneck, ))
        z = Lambda(self.sampling, output_shape=(self._bottleneck, ))([z_mean, z_sigma])
        h_decoded = Dense(self._mid2, activation='relu')(z)
        h_decoded = Dense(self._mid1, activation='relu')(h_decoded)
        x_decoded_mean = Dense(self._input_dim, activation='sigmoid')(h_decoded)
        return Model([z_mean, z_sigma], x_decoded_mean)

    def sampling(self, args):
        z_mean, z_sigma = args
        epsilpn = K.random_normal(shape=(self._bottleneck, ), mean=0, stddev=self._epsilon_std)
        return z_mean + z_sigma * epsilpn

    def build_vae(self, encoder, decoder):
        _, encoder_dense, encoder_dense2, encoder_mean, encoder_sigma = encoder.layers

        x = Input(shape=(self._input_dim, ))
        hidden = encoder_dense(x)
        hidden2 = encoder_dense2(hidden)
        z_mean = encoder_mean(hidden2)
        z_sigma = encoder_sigma(hidden2)

        self.z_m = z_mean
        self.z_s = z_sigma

        _, _, decoder_lambda, decoder_dense1, decoder_dense2, decoder_dense3 = decoder.layers
        z = decoder_lambda([z_mean, z_sigma])
        h_decoded = decoder_dense1(z)
        x_decoded = decoder_dense2(h_decoded)
        x_decoded_mean = decoder_dense3(x_decoded)
        return Model(x, x_decoded_mean)

    def binary_crossentropy(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)

    def vae_loss(self, x, x_decoded_mean):
        z_mean = self.z_m
        z_sigma = self.z_s

        latent_loss = - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma),
                                           axis=-1))
        reconst_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean), axis=-1)
        return latent_loss + reconst_loss

    def model_compile(self, model):
        return model.compile(optimizer='rmsprop',
                             loss=self.vae_loss)

    def training(self, x_train, y_train, x_test, y_test, force_training=False):
        self.vae_model = self.build_vae(self.build_encoder(), self.build_decoder())
        self.model_compile(self.vae_model)
        self.vae_model.fit(x_train, x_train,
                           epochs=self._epochs,
                           batch_size=self._batch_size,
                           shuffle=self._shuffle)
        intermediate_model = self.build_encoder()

        return self.vae_model, intermediate_model

    def model_save(self, save_dir, roc, f_value):
        preparing_dir(save_dir)
        save_file_name = 'ROC_{}_F_{}.h5'.format(roc, f_value)
        save_path = os.path.join(save_dir, save_file_name)
        self.vae_model.save(save_path)
