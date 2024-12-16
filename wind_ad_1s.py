import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
dataset_name = "haenkaze"
data_dir1 = 'data/' + dataset_name + '/2023/1s_data'
data_dir2 = 'data/' + dataset_name + '/2024/1s_data'
parquet_file = f'data/{dataset_name}/merged_data.parquet'

if os.path.exists(parquet_file):
    print("Loading data from parquet_file...")
    data = pd.read_parquet(parquet_file)
else:
    print("Loading data from CSV files...")
    dfs = []
    file_list1 = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
    file_list2 = [os.path.join(data_dir2, file) for file in os.listdir(data_dir2)]
    for file_path in file_list1:
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        except UnicodeDecodeError as e:
            print(f"Error in file: {file_path}")
            print(e)
        dfs.append(df)
    print("dir1 downloaded")
    for file_path in file_list2:
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        except UnicodeDecodeError as e:
            print(f"Error in file: {file_path}")
            print(e)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data.to_parquet(parquet_file, engine='pyarrow')
    print("Data saved in Parquet format.")
print("Finished loading data.")

# preprocessing
stampcol = "DateTime"
data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
valid_range = ('2023-05-01','2023-05-04')
test_range = ('2023-04-01','2024-03-02')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
#data = utils.arrange_data(utils.fix_data(data),stampcol)
train = utils.arrange_data(utils.fix_data(train),stampcol)
valid = utils.arrange_data(utils.fix_data(valid),stampcol)
test = utils.arrange_data(utils.fix_data(test),stampcol)

# split timestamp from data
X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
X_valid = valid.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
timestamp = test['DateTime']

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

if exec_model.gmm_multi:
    for n in range(40):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,exec_model.covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,exec_model.covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path)

show_info(out_dir,exec_model)