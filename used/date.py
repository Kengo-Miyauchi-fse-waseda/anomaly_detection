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


dataset_name = "BATADAL"


# Download dataset
train = pd.read_csv('data/'+dataset_name+'_train.csv')
test = pd.read_csv('data/'+dataset_name+'_test.csv')

# データとtimestampを分離
X_train_base = train.drop('DATETIME',axis='columns').values
X_test = test.drop('DATETIME',axis='columns').values
timestamp = test['DATETIME']
values = np.random.rand(len(timestamp))

plot_by_date(values,timestamp,"test.png")
