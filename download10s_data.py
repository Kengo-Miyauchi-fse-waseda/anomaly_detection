import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# データセット情報
dataset_name = "haenkaze"
data_dir1 = f'data/{dataset_name}/2023/1s_data'
data_dir2 = f'data/{dataset_name}/2024/1s_data'
output_dir = f'data/{dataset_name}/2024/10s_avg_data'
os.makedirs(output_dir, exist_ok=True)  # 出力フォルダを作成
final_output_file = f'{output_dir}/2024_10s_avg_data.parquet'

# CSVファイルリストの取得
file_list1 = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
file_list2 = [os.path.join(data_dir2, file) for file in os.listdir(data_dir2)]
all_files = file_list1 + file_list2

# 各ファイルでの平均化処理関数
def process_file(file_path):
    try:
        # ファイル読み込み
        df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        
        # タイムスタンプ列を処理
        timestamp_col = "DateTime"  # 必要に応じて列名を変更
        if timestamp_col not in df.columns:
            print(f"Warning: '{timestamp_col}' column not found in {file_path}. Skipping...")
            return None
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])  # タイムスタンプをdatetime型に変換
        df = df.set_index(timestamp_col)  # タイムスタンプをインデックスに設定

        # 10秒ごとの平均化
        df_10s_avg = df.resample('10S').mean()

        # 平均化結果を保存
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_10s_avg.parquet'))
        df_10s_avg.reset_index(inplace=True)  # インデックスをリセット
        df_10s_avg.to_parquet(output_file, engine='pyarrow')
        print(f"Processed and saved: {output_file}")
        return df_10s_avg
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# 全ファイルを処理
all_avg_dfs = []
for file_idx, file_path in enumerate(file_list2):
    print(f"Processing file {file_idx + 1}/{len(file_list2)}: {file_path}")
    avg_df = process_file(file_path)
    if avg_df is not None:
        all_avg_dfs.append(avg_df)

# 全ファイルの結果を結合して保存
if all_avg_dfs:
    final_data = pd.concat(all_avg_dfs, ignore_index=True)
    final_data.to_parquet(final_output_file, engine='pyarrow')
    print(f"All files processed and saved to {final_output_file}.")
else:
    print("No valid data to save.")
