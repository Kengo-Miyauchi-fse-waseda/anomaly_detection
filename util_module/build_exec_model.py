from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import yaml
from util_module.callback import LogCallback
import logging
import os

class ExecModel:
    def __init__(self, X_train, X_valid, device, config, dataset_name,
                 out_dir=None,feature_dim=None, n_steps=None, optimizer_params=None, batch_size=None, 
                 gmm_multi=None, log_plot=None, covariance_type=None, pretraining_ratio=None, max_epochs=None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.device = device
        self.config = config
        self.dataset_name = dataset_name
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.gmm_multi = gmm_multi
        self.log_plot = log_plot
        self.covariance_type = covariance_type
        self.pretraining_ratio = pretraining_ratio
        self.max_epochs = max_epochs
        
        self.set_params_from_file(config)
        self.conditions=self.gather_conditions()
        self.out_dir = self.set_out_dir()
        self.set_log()
        self.unsupervised_model=self.set_unsupervised_model()
        
        logging.info("Starting TabNet pretraining...")
        self.fit_model()
        logging.info("TabNet pretraining finished.")
    
    def set_unsupervised_model(self):
        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            mask_type='entmax', # "sparsemax",
            n_d = self.feature_dim,
            n_a = self.feature_dim,
            verbose=100,
        )
        return unsupervised_model
    
    def fit_model(self):
        self.unsupervised_model.fit(
            X_train=self.X_train,
            eval_set=[self.X_valid],
            max_epochs=self.max_epochs , patience=1000000,
            batch_size=self.batch_size, virtual_batch_size=self.batch_size,
            pretraining_ratio=self.pretraining_ratio,
            callbacks=[LogCallback()]
        )
    
    def set_log(self):
        filepath=self.out_dir + '/pretraining.log'
        logging.basicConfig(
            filename=filepath,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        print(f"log file : {filepath}")
    
    def set_out_dir(self):
        out_dir = "result/" + self.dataset_name + "_tabnet/" + str(self.feature_dim) + "_dim"
        if(self.covariance_type!='full'):
            out_dir += ("_" + self.covariance_type)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
        
    def set_params_from_dict(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self 

    def set_params_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)
                self.set_params_from_dict(params)
        except FileNotFoundError:
            print(f"Error: Config file not found at {file_path}")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML file {file_path}: {e}")
        return self

    def gather_conditions(self):
        conditions = [("feature_dim",self.feature_dim),("optimizer_params",self.optimizer_params),("pretraining_ratio",self.pretraining_ratio),("max_epochs",self.max_epochs)]
        return conditions