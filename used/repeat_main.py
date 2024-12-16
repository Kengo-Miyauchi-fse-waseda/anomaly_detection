import torch
import pandas as pd
import numpy as np
np.random.seed(0)
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer
from custom_module.create_dataloader import create_dataloader
from custom_module.gmm_module import calc_AnomalyScore
from custom_module.data_to_plot import plot_by_date
from custom_module.record_info import recordInfo

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def repeat(feature_dim, covariance_type):
    # set parameter
    feature_dim = feature_dim
    optimizer_params=dict(lr=5e-3)
    gmm_multi = True
    covariance_type = covariance_type
    conditions = [("feature_dim",feature_dim),("optimizer_params",optimizer_params), ("gmm_multi",gmm_multi), ("covariance_type",covariance_type)]

    dataset_name = "BATADAL"
    out_dir = "result/"+dataset_name + "_tabnet/" + str(feature_dim) + "_dim2"
    if(covariance_type!='full'):
        out_dir += ("_" + covariance_type)
    os.makedirs(out_dir, exist_ok=True)
    recordInfo(out_dir,conditions)

    # Download dataset
    train = pd.read_csv('data/'+dataset_name+'_train.csv')
    test = pd.read_csv('data/'+dataset_name+'_test.csv')

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
        pretraining_ratio=1.0,
    )

    # data to features as encoder output data
    train_dataloader = create_dataloader(X_train,batch_size=128,need_shuffle=True)
    test_dataloader = create_dataloader(X_test,batch_size=128,need_shuffle=False)
    feature_train = []
    for batch in train_dataloader:
        encoder_out = unsupervised_model.network.encoder(batch.to(device))[0][-1]
        feature_train.append(encoder_out)
    feature_train = torch.cat(feature_train).detach().cpu()
    feature_test = []
    for batch in test_dataloader:
        encoder_out = unsupervised_model.network.encoder(batch.to(device))[0][-1]
        feature_test.append(encoder_out)
    feature_test = torch.cat(feature_test).detach().cpu()


    if gmm_multi:
        for n in range(10):
            anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,covariance_type,feature_train,feature_test,out_dir)
            plot_by_date(anomaly_score,timestamp,img_path)
    else:
        n_components=3
        anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(anomaly_score,timestamp,img_path)
    print("result folder: "+out_dir)

covariance_types = ['full','tied','diag']
for i in range(8,23):
    feture_dim = i
    for j in range(len(covariance_types)):
        repeat(feture_dim,covariance_types[j])