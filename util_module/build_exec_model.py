from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import yaml
from util_module.callback import LogCallback
import logging
import os
import json

class ExecModel:
    def __init__(self, X_train, X_valid, device, config, dataset_name, model_name,
                 feature_dim=None, n_steps=None, optimizer_params=None, batch_size_pre=None, batch_size_tr=None,
                 gmm_multi=None, log_plot=None, covariance_type=None, pretraining_ratio=None, max_epochs=None,
                 path_to_pretrained=None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.device = device
        self.config = config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.optimizer_params = optimizer_params
        self.batch_size_pre = batch_size_pre
        self.batch_size_tr = batch_size_tr
        self.gmm_multi = gmm_multi
        self.log_plot = log_plot
        self.covariance_type = covariance_type
        self.pretraining_ratio = pretraining_ratio
        self.max_epochs = max_epochs
        self.path_to_pretrained=path_to_pretrained
        
        # configとlogの設定
        self.set_params_from_file(config)
        self.conditions=self.gather_conditions()
        self.out_dir = self.set_out_dir()
        os.makedirs(self.out_dir, exist_ok=True)
        self.path_to_pretrained = './model/' + dataset_name + "/tabnet-pretrain-" + str(self.feature_dim) + "dim"
        os.makedirs(self.path_to_pretrained, exist_ok=True)
        self.set_log()
        
        self.path_to_pretrained +="/pretrained.pth"
        if(os.path.exists(self.path_to_pretrained)):
            self.unsupervised_model=torch.load(self.path_to_pretrained)
        else:
            self.unsupervised_model=self.set_unsupervised_model()
            logging.info("Starting TabNet pretraining...")
            self.fit_model()
            logging.info("TabNet pretraining finished.")
            torch.save(self.unsupervised_model, self.path_to_pretrained)
            # self.unsupervised_model.save_model(path_to_pretrained)
    
    # 事前学習モデルの設定
    def set_unsupervised_model(self):
        unsupervised_model = TabNetPretrainer(
            device_name=self.device,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            mask_type='entmax', # "sparsemax",
            n_d = self.feature_dim,
            n_a = self.feature_dim,
            verbose=100,
        )
        return unsupervised_model
    
    # 事前学習
    def fit_model(self):
        self.unsupervised_model.fit(
            X_train=self.X_train,
            eval_set=[self.X_valid],
            max_epochs=self.max_epochs , patience=1000000,
            batch_size=self.batch_size_pre, virtual_batch_size=256,
            pretraining_ratio=self.pretraining_ratio,
            callbacks=[LogCallback()]
        )
    
    # ログの設定
    def set_log(self):
        filepath=self.path_to_pretrained + '/pretraining.log'
        logging.basicConfig(
            filename=filepath,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        print(f"log file : {filepath}")
    
    def set_out_dir(self):
        out_dir = "result/" + self.dataset_name + "/" + self.model_name + "/" + str(self.feature_dim) + "dim"
        if(self.covariance_type!='full'):
            out_dir += ("-" + self.covariance_type)
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
    

    def custom_save_model(model, filepath):
        model_dict = model.__dict__
        for key, value in model_dict.items():
            if isinstance(value, torch.device):
                model_dict[key] = str(value)
        with open(filepath, 'w') as f:
            json.dump(model_dict, f)