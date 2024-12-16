import os
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture
from util.preparing_dir import preparing_dir
import joblib


class Gmm(object):
    def __init__(self,
                 model_save_dir='./data/Ishibashi/model/gmm'):
        self._model_save_dir = model_save_dir

    def modeling(self, data_set, components=2,
                 force_training=False, covar='full'):
        """training GMM
        Parameters
            dirname: str, feature_name dimenstion etc...
            data_set: array_like, shape(n_sample, n_features)
            mix: int, GMM mixture
            preproc: str, preprocessing flag

        Returns
        """
        target_dir = self._model_save_dir
        save_filename = "{}/model_mix_{}_cov_{}.pkl".format(target_dir, components, covar)
        if os.path.isfile(save_filename) and not force_training:
            print("already trained")
        else:
            print("training components {}".format(components))
            gmm = GMM(components, covariance_type=covar)
            model = gmm.fit(data_set)

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            joblib.dump(model, save_filename, compress=True)
        return save_filename


class DpGmm(object):
    def __init__(self,
                 model_save_dir=''):
        self._model_save_dir = model_save_dir

    def modeling(self, data_set, components=512,
                 force_training=False, covar='full'):
        """training GMM
        Parameters
            data_set: array_like, shape(n_sample, n_features)
            mix: int, GMM mixture
            preproc: str, preprocessing flag

        Returns
        """
        target_dir = self._model_save_dir
        save_filename = "{}/DPGMM_model_cov_{}.pkl".format(target_dir, covar)
        if os.path.isfile(save_filename) and not force_training:
            print("already trained")
        else:
            print("training dpgmm components max {}".format(components))
            gmm = BayesianGaussianMixture(n_components=components, covariance_type=covar,
                                          weight_concentration_prior_type='dirichlet_process')
            model = gmm.fit(data_set)

            preparing_dir(target_dir)
            joblib.dump(model, save_filename)
        return save_filename
