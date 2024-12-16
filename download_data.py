import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
# download dataset
dataset_name = "haenkaze"
data_dir1 = 'data/' + dataset_name + '/2023/1s_data'
data_dir2 = 'data/' + dataset_name + '/2024/1s_data'
parquet_file = f'data/{dataset_name}/2024_data.parquet'

# check cache file
if os.path.exists(parquet_file):
    print("Loading data from parquet_file...")
    data = pd.read_parquet(parquet_file)
else:
    print("Loading data from CSV files...")
    dfs = []
    file_list1 = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
    file_list2 = [os.path.join(data_dir2, file) for file in os.listdir(data_dir2)]
    for file_idx, file_path in enumerate(file_list2):
        print(f"Download {file_idx/len(file_list2):.2f}% completed")
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