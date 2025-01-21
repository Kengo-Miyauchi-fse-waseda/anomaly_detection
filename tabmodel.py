import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
dataset_name = "haenkaze"

# preprocessing
parquet_file = f'data/{dataset_name}/2024_fixed.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
stampcol = "DateTime"
train_range = ('2024-04-01','2024-08-31')
valid_range = ('2024-09-01','2024-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train,data
X_valid = valid.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del valid
gc.collect()
print("Finished loading data.")

data_dir ='data/haenkaze/fixed_data'
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
for file_path in file_list:
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    df.fillna(method="ffill",inplace=True)
    data = df.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
    if(np.isnan(data).any() or np.isinf(data).any()):
        print(f"isNAN: {np.isnan(data).any()}")
        print(f"isINF: {np.isinf(data).any()}")
    X_train = np.concatenate([X_train,data])
    del df,data
    print("Finished loading data.")
    gc.collect()

# set execute model
model_name = "tabnet-pretrain-out2023"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train,X_valid)
out_dir = exec_model.out_dir

print(f"finish model build: {exec_model.path_to_pretrained}")