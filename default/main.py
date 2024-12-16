import torch
import pandas as pd
import numpy as np
np.random.seed(0)
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer

# end_info.py
def show_info(out_dir,conditions):
    print("result: "+out_dir)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print((f"End Time : {formatted_time}\n"))
    recordInfo(out_dir,conditions)
def recordInfo(out_dir, info):
    file_path = out_dir + "/info.txt"
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "w") as file:
        file.write(f"End Time : {formatted_time}\n")
        for x in info:
            file.write(x[0]+" : "+str(x[1])+"\n")

# data_to_plot.py
from matplotlib import pyplot as plt
def visualize_events(type):
    event_dates=pd.read_csv("data/BATADAL/BATADAL_events.csv")
    if(type=="range"):
        start_date=event_dates['start_date']
        end_date=event_dates['end_date']
        # イベント区間の可視化
        for i in range(len(start_date)):
            start = datetime.strptime(start_date[i], "%Y-%m-%d")
            end = datetime.strptime(end_date[i], "%Y-%m-%d")
            plt.axvspan(start, end, color='orange', alpha=0.2)
    if(type=="line"):
        for i in range(len(start_date)):
            event = pd.Timestamp(datetime.strptime(event_dates[i], "%Y-%m-%d"))
            plt.axvline(event, color='orange', alpha=0.5)
def plot_by_date(log_plot, anomaly_score, timestamp, threshold, img_path):
    df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
    # DATATIMEを日付部分だけに変換（Y軸を同じにするため）
    if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
        df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
    df['Date'] = df['DATETIME'].dt.date

    # 散布図をプロット
    plt.figure(figsize=(12, 6))
    if log_plot:
        plt.yscale('log')
    for date, group in df.groupby('Date'):
        plt.scatter([date]*len(group), group['AnomalyScore'], alpha=0.8, marker='o', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores by Date")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
    # イベントの可視化
    visualize_events(type="range")
    # 凡例の表示を防止するための設定（重複する日付を削除）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(img_path)

# gmm_module.py
from sklearn.mixture import GaussianMixture
def calc_AnomalyScore(n_components, covariance_type, feature_train, feature_test, out_dir):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=1)
    gmm.fit(feature_train)
    log_likelihood = -gmm.score_samples(feature_train)
    threshold = np.percentile(log_likelihood,99)
    anomaly_score = -gmm.score_samples(feature_test)
    img_path = out_dir + "/gmm_" + str(n_components) + "components"
    return anomaly_score, threshold, img_path

# tabnet_feature.py
from torch.utils.data import Dataset, DataLoader
class TorchDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x
def create_dataloader(X,batch_size,need_shuffle):
    dataloader = DataLoader(
        TorchDataset(X.astype(np.float32)),
        batch_size=batch_size,
        shuffle=need_shuffle,
    )
    return dataloader
def data_to_TabNetFeatures(unsupervised_model,X_train,X_test):
    train_dataloader = create_dataloader(X_train,batch_size=256,need_shuffle=True)
    test_dataloader = create_dataloader(X_test,batch_size=256,need_shuffle=False)
    # data to features as encoder output data
    feature_train = []
    for batch in train_dataloader:
        #import pdb; pdb.set_trace()
        batch = batch.to(device)
        step_outputs = unsupervised_model.network.encoder(batch)[0]
        # 最終的にstep_outputの総和を取る必要があるかどうか
        encoder_out = sum(step for step in step_outputs)
        feature_train.append(encoder_out)
    feature_train = torch.cat(feature_train).detach().cpu()
    feature_test = []
    for batch in test_dataloader:
        step_outputs = unsupervised_model.network.encoder(batch.to(device))[0]
        encoder_out = sum(step for step in step_outputs)
        feature_test.append(encoder_out)
    feature_test = torch.cat(feature_test).detach().cpu()
    return feature_train, feature_test


# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set parameter
feature_dim = 32
optimizer_params=dict(lr=1e-3)
gmm_multi = True
log_plot = False
covariance_type = 'full'
pretraining_ratio = 0.5
conditions = [("feature_dim",feature_dim),("optimizer_params",optimizer_params),("pretraining_ratio",pretraining_ratio)]

dataset_name = "BATADAL"
out_dir = "result/" + dataset_name + "_tabnet/" + str(feature_dim) + "_dim_test"
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
max_epochs = 50 if not os.getenv("CI", False) else 2 # 1000

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs , patience=10000,
    batch_size=256, virtual_batch_size=256,
    pretraining_ratio=pretraining_ratio,
)

# data to features as encoder output data
feature_train, feature_test = data_to_TabNetFeatures(unsupervised_model,X_train,X_test)

if gmm_multi:
    for n in range(10):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,covariance_type,feature_train,feature_test,out_dir)
        plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)

show_info(out_dir,conditions)