import torch
import pandas as pd
import os
import gc
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
dataset_name = "haenkaze"
parquet_file = f'data/{dataset_name}/2023_fixed.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
print("Finished loading data.")

import pdb; pdb.set_trace()
# preprocessing
stampcol = "DateTime"
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
valid_range = ('2023-05-01','2023-05-04')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
train.to_parquet(train_range[0]+"~"+train_range[1]+"data.parquet", engine='pyarrow')
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)