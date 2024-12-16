import torch
import pandas as pd
import numpy as np
np.random.seed(0)
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer
from util_module.create_dataloader import create_dataloader
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from used.record_info import recordInfo
from util_module.tabnet_feature import data_to_TabNetFeatures

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# set parameter
feature_dim = 16
optimizer_params=dict(lr=1e-3)
gmm_multi = True
log_plot = True
covariance_type = 'diag'
pretraining_ratio = 0.5
conditions = [("feature_dim",feature_dim),("optimizer_params",optimizer_params),("pretraining_ratio",pretraining_ratio)]

dataset_name = "BATADAL"
out_dir = "result/" + dataset_name + "_tabnet/" + str(feature_dim) + "_dim"
if(covariance_type!='full'):
    out_dir += ("_" + covariance_type)
os.makedirs(out_dir, exist_ok=True)

# Download dataset
train = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_train.csv')
test = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_test.csv')

# データとtimestampを分離
X_train_base = train.drop('DATETIME',axis='columns').values
X_test = test.drop('DATETIME',axis='columns').values
timestamp = test['DATETIME']
X_train, X_valid = train_test_split(X_train_base, test_size=0.2, random_state=42)

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=optimizer_params,
    mask_type='entmax', # "sparsemax",
    n_d = feature_dim,
    n_a = feature_dim,
    verbose=25,
)

# Self Supervised Training
max_epochs = 250 if not os.getenv("CI", False) else 2 # 1000

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs , patience=10000,
    batch_size=128, virtual_batch_size=128,
    pretraining_ratio=pretraining_ratio,
)

# data to features as encoder output data
train_dataloader = create_dataloader(X_train,batch_size=128,need_shuffle=True)
test_dataloader = create_dataloader(X_test,batch_size=128,need_shuffle=False)
feature_train, feature_test = data_to_TabNetFeatures(unsupervised_model,X_train,X_test,device)

if gmm_multi:
    for n in range(10):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)

print("result: "+img_path)
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print((f"End Time : {formatted_time}\n"))
recordInfo(out_dir,conditions)