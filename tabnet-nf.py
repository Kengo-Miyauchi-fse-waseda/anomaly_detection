import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.create_dataloader import create_dataloader
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
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
valid = utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train

X_valid = valid.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del valid
gc.collect()
#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test

# memory management
gc.collect()
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)
X_valid = X_valid.astype(np.float16)

# data scaling
""" from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid) """

# set execute model
model_name = "tabnet-NF"
config = set_config_file()
exec_model = ExecModel(X_train,X_valid,device,config,dataset_name,model_name)
out_dir = exec_model.out_dir

# convert data to tabnet encoder features
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
print("Finish feature extraction")


# Normalizing Flow
from modeling.Flow_based.RealNVP import RealNVP
from torch.distributions import MultivariateNormal
nf_layers = 32
# RealNVPの基底分布を追加
class RealNVPWithBaseDistribution(RealNVP):
    def __init__(self, dims, cfg=None):
        #import pdb; pdb.set_trace()
        super().__init__(dims=[dims], cfg=cfg)
        self.base_distribution = MultivariateNormal(
            loc=torch.zeros(dims).to(device),
            covariance_matrix=torch.eye(dims).to(device)
        )

# RealNVPモデルのインスタンス化
#import pdb; pdb.set_trace()
nf_model = RealNVPWithBaseDistribution(
    dims=feature_train.shape[1], 
    cfg={"layers": nf_layers}
)
nf_model.to(device)

# DataLoaderの作成
train_dataloader = create_dataloader(feature_train.numpy(), batch_size=256, need_shuffle=True)
test_dataloader = create_dataloader(feature_test.numpy(), batch_size=256, need_shuffle=False)

# Optimizerの設定
optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-3)

# 学習ループ
print("NF Training")
for epoch in range(50):
    nf_model.train()
    epoch_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        loss = torch.mean(-nf_model.base_distribution.log_prob(z) - log_df_dz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")

# 正常データの異常スコアを計算
normal_scores = []
nf_model.eval()
with torch.no_grad():
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_jacobian = nf_model.forward(batch)
        log_likelihood = nf_model.base_distribution.log_prob(z) + log_jacobian
        normal_scores.extend(-log_likelihood.cpu().numpy())

threshold = np.percentile(normal_scores, 99)

# テストデータの異常スコアを計算
test_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        z, log_jacobian = nf_model.forward(batch)
        log_likelihood = nf_model.base_distribution.log_prob(z) + log_jacobian
        test_scores.extend(-log_likelihood.cpu().numpy())


plot_by_date(exec_model.log_plot,test_scores,timestamp,train_range,threshold,out_dir)
show_info(out_dir,exec_model)