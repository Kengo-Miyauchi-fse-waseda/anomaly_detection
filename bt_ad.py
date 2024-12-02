import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download dataset
dataset_name = "BATADAL"
train = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_train.csv')
test = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_test.csv')
train = train.fillna(method="ffill")
test = test.fillna(method="ffill")

# データとtimestampを分離
X_train_base = train.drop('DATETIME',axis='columns').values
X_test = test.drop('DATETIME',axis='columns').values
timestamp = test['DATETIME']
X_train, X_valid = train_test_split(X_train_base, test_size=0.2, random_state=42)

config = set_config_file()
exec_model = ExecModel(X_train,X_valid,device,config)

out_dir = "result/" + dataset_name + "_tabnet/" + str(exec_model.feature_dim) + "_dim"
if(exec_model.covariance_type!='full'):
    out_dir += ("_" + exec_model.covariance_type)
os.makedirs(out_dir, exist_ok=True)

# data to features as encoder output data
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)

if exec_model.gmm_multi:
    for n in range(10):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,exec_model.covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(exec_model.log_plot,anomaly_score,timestamp,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,exec_model.covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(exec_model.log_plot,anomaly_score,timestamp,threshold,img_path)

show_info(out_dir,exec_model)