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
stampcol = "DateTime"
frequency = '1S'
dataset_name = "haenkaze"
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
print("Finished loading data.")

#import pdb; pdb.set_trace()
# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
gc.collect()
#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test
gc.collect()



# set execute model
model_name = "tabnet-NF"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = exec_model.out_dir

# convert data to tabnet encoder features
print("Start feature extraction")
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
print("Finish feature extraction")


# Normalizing Flow
from modeling.Flow_based.RealNVP import RealNVP
nf_layers = 32
dim = feature_train.shape[1]
normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.zeros(dim).to(device),
    covariance_matrix=torch.eye(dim).to(device)
)

# RealNVPモデルのインスタンス化
#import pdb; pdb.set_trace()
nf_model = RealNVP(dims=[dim],cfg = {"layers": nf_layers})
nf_model.to(device)

# DataLoaderの作成
train_dataloader = create_dataloader(feature_train.numpy(), batch_size=256, need_shuffle=True)
test_dataloader = create_dataloader(feature_test.numpy(), batch_size=256, need_shuffle=False)
del feature_train,feature_test

# Optimizerの設定
optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-3)


# 学習ループ
print("NF Training...")
epoch_losses = []
for epoch in range(20):
    nf_model.train()
    epoch_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        loss = torch.mean(-normal_dist.log_prob(z) - log_df_dz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        del batch, z, log_df_dz
    epoch_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")
    epoch_losses.append(epoch_loss)
gc.collect()
print("Finish NF Training")

print("Calculate Train Score")
# 正常データの異常スコアを計算
normal_scores = []
nf_model.eval()
with torch.no_grad():
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        log_likelihood = normal_dist.log_prob(z) + log_df_dz
        normal_scores.extend(-log_likelihood.cpu().numpy())
        del batch, z, log_df_dz
threshold = np.percentile(normal_scores, 99)
del train_dataloader
gc.collect()

print("Calculate Test Score")
# テストデータの異常スコアを計算
test_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        log_likelihood = normal_dist.log_prob(z) + log_df_dz
        test_scores.extend(-log_likelihood.cpu().numpy())
        del batch, z, log_df_dz
del test_dataloader
gc.collect()
event_files = ["data/haenkaze/events.csv","data/haenkaze/event_range.csv"]
score_file = "result/haenkaze/tabnet-NF/scores.csv"
plot_by_date(exec_model.log_plot,test_scores,timestamp,train_range,threshold,(out_dir+"/tabnet-nf.png"),event_files,score_file=score_file)
show_info(out_dir,exec_model)

from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(len(epoch_losses)), epoch_losses, label="Loss")
plt.title("Reconstruction Loss during Pretraining", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
img_path = "result/haenkaze/tabnet-NF/train_curve.png"
plt.savefig(img_path)
print(img_path)