import torch
import pandas as pd
import os
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download dataset
dataset_name = "wind_edp"
stampcol = "Timestamp"
full_data = pd.read_csv('data/'+dataset_name+'/SCADA.csv')
full_data = utils.arrange_data(full_data,stampcol)
data = utils.extract_specific_id(full_data,'T07')
train_range = ('2017-02-01T00:00:00+00:00','2017-04-30T23:50:00+00:00')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = utils.extract_specific_terms(data,'2017-05-01T00:00:00+00:00','2017-05-31T23:50:00+00:00',stampcol)

# データとtimestampを分離
X_train = train.drop(columns=['Timestamp','Turbine_ID']).values
X_test = data.drop(columns=['Timestamp','Turbine_ID']).values
X_valid = valid.drop(columns=['Timestamp','Turbine_ID']).values
timestamp = data['Timestamp']

# scale data
""" from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid) """

config = set_config_file()
exec_model = ExecModel(X_train,X_valid,device,config,dataset_name)

out_dir = "result/" + dataset_name + "_tabnet/" + str(exec_model.feature_dim) + "_dim"
if(exec_model.covariance_type!='full'):
    out_dir += ("_" + exec_model.covariance_type)
os.makedirs(out_dir, exist_ok=True)

# data to features as encoder outputs
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)

if exec_model.gmm_multi:
    for n in range(40):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,exec_model.covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,exec_model.covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)

show_info(out_dir,exec_model)