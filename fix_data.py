import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from wind_module import utils

# download dataset
dataset_name = "haenkaze"
data_dir1 = 'data/' + dataset_name + '/2023/1s_data'
data_dir2 = 'data/' + dataset_name + '/2024/1s_data'
parquet_file = f'data/{dataset_name}/2024_fixed.parquet'

stampcol = "DateTime"
dfs = []
file_list1 = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
file_list2 = [os.path.join(data_dir2, file) for file in os.listdir(data_dir2)]

print(f"file path:{parquet_file}")
for file_idx, file_path in enumerate(file_list2):
    try:
        df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
    except UnicodeDecodeError as e:
        print(f"Error in file: {file_path}")
        print(e)
    df[stampcol] = pd.to_datetime(df[stampcol])
    utils.fix_data(df)
    df.fillna(method="ffill")
    dfs.append(df)
    print(f"Download {100*(file_idx+1)/len(file_list2):.1f}% completed")
data = pd.concat(dfs, ignore_index=True)
data.to_parquet(parquet_file, engine='pyarrow')