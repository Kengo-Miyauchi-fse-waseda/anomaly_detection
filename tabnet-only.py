import torch
import pandas as pd
import os
import numpy as np
import gc
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from util_module.end_info import show_info
from util_module.data_to_plot import plot_by_date
from util_module.TabNet_reconstruntion_errors import calc_reconstruction_errors
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
# download data
dataset_name = "haenkaze"
data_dir1 = 'data/' + dataset_name + '/2023/1s_data'
data_dir2 = 'data/' + dataset_name + '/2024/1s_data'
parquet_file = f'data/{dataset_name}/2023_fixed.parquet'

if os.path.exists(parquet_file):
    print("Loading data from parquet_file...")
    data = pd.read_parquet(parquet_file)
else:
    print("Loading data from CSV files...")
    dfs = []
    file_list1 = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
    for file_path in file_list1:
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        except UnicodeDecodeError as e:
            print(f"Error in file: {file_path}")
            print(e)
        dfs.append(df)
        del df
    data = pd.concat(dfs, ignore_index=True)
    del dfs
print("Finished loading data.")

# preprocessing
stampcol = "DateTime"
data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
valid_range = ('2023-05-01','2023-05-04')
test_range = ('2023-04-01','2023-06-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

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
model_name = "tabnet-only"
config = set_config_file()
exec_model = ExecModel(X_train,X_valid,device,config,dataset_name,model_name)
out_dir = exec_model.out_dir

# 学習済みモデルを使用して再構成誤差を計算
reconstructed_X_test = exec_model.unsupervised_model.predict(X_test)
reconstruction_errors = np.mean((X_test - reconstructed_X_test) ** 2, axis=1)

# 再構成誤差の閾値を設定（正常データの上位5%の誤差を閾値に）
threshold = np.percentile(reconstruction_errors[:len(X_test)], 95)
import pdb; pdb.set_trace()

plot_by_date(exec_model.log_plot,reconstruction_errors,timestamp,train_range,threshold,out_dir)

""" # 再構成誤差を計算
reconstruction_errors = calc_reconstruction_errors(exec_model, X_test)

# 正常データの再構成誤差から閾値を設定（95パーセンタイルを閾値とする例）
threshold = torch.quantile(reconstruction_errors, 0.95) """

show_info(out_dir,exec_model)