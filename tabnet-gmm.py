import torch
import pandas as pd
import numpy as np
import os
import gc
import pickle
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"b
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
stampcol = "DateTime"
frequency = '1S'
dataset_name = "haenkaze"
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
#data = utils.fix_data(data)
data = utils.arrange_data(data,stampcol)
print("Finished loading data.")

# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]
del data

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test

# memory management
gc.collect()
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)


# set execute model
model_name = "tabnet-gmm"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train,path_to_pretrained="model/haenkaze/tabnet-pretrain-out2023-40dim")
out_dir = exec_model.out_dir

# convert data to tabnet encoder features
print("Start feature extraction")
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
del X_train
del X_test
print("Finish feature extraction")

from sklearn.mixture import GaussianMixture
import numpy as np

event_files = ["data/haenkaze/events.csv"]
event_files.append("data/haenkaze/event_range.csv")
score_file = "result/haenkaze/tabnet-gmm/"+frequency+"scores.csv"
#os.makedirs(score_file, exist_ok=True)
n_components = 10


gmm_model = f"model/haenkaze/gmm_{frequency}.pkl"
if os.path.exists(gmm_model):
    print(f"Load from {gmm_model}")
    with open(gmm_model, "rb") as file:
        gmm = pickle.load(file)
else:
    print("GMM Training")
    gmm = GaussianMixture(n_components=n_components, covariance_type=exec_model.covariance_type, random_state=42, n_init=10, max_iter=25)
    gmm.fit(feature_train)
    print("Finish GMM Training")
    with open(gmm_model, "wb") as file:
        pickle.dump(gmm, file)
    print("Model saved as gmm_model.pkl")  

log_likelihood = -gmm.score_samples(feature_train)
threshold_line = 100 - 0.001
threshold = np.percentile(log_likelihood,threshold_line)
#threshold = max(log_likelihood)
del feature_train
anomaly_score = -gmm.score_samples(feature_test)

#import pdb; pdb.set_trace()

#img_path = out_dir + "/"+frequency+"_gmm_" + str(n_components) + "components.png"
img_path = f"{out_dir}/{frequency}.png"
plot_by_date(False,anomaly_score,timestamp,train_range,threshold,img_path,frequency,event_files,score_file=score_file)
print(f"result: {img_path}")
show_info(out_dir,exec_model)
print(f"Threshold line: {threshold_line}%")
print(f"Score result: {score_file}")

""" event_file = "data/haenkaze/events.csv"
if exec_model.gmm_multi:
    for n in range(20):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,exec_model.covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_file)
        print(f"result: {img_path}")
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,exec_model.covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)
    print(f"result: {img_path}")
show_info(out_dir,exec_model) """