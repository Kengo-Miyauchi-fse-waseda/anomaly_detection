import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
dataset_name = "haenkaze"
parquet_file = f'data/{dataset_name}/2023_fixed.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
print("Finished loading data.")

# preprocessing
stampcol = "DateTime"
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
valid_range = ('2023-05-01','2023-05-04')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]
del data

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train

X_valid = valid.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del valid

#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test

# memory management
gc.collect()
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)
X_valid = X_valid.astype(np.float16)

# scaling data
""" from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid) """

# set execute model
model_name = "tabnet-gmm"
config = set_config_file()
exec_model = ExecModel(X_train,X_valid,device,config,dataset_name,model_name)
out_dir = exec_model.out_dir

# convert data to tabnet encoder features
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
print("Finish feature extraction")

show_info(out_dir,exec_model)
event_file = "data/haenkaze/events.csv"
if exec_model.gmm_multi:
    for n in range(20):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,exec_model.covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_file)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,exec_model.covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)