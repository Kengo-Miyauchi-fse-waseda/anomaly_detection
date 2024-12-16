import torch
import pandas as pd
import numpy as np
np.random.seed(0)
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer
import utils

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
    # visualize_events(type="range")
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
def data_to_TabNetFeatures(unsupervised_model,data):
    dataloader = create_dataloader(data,batch_size=128,need_shuffle=True)
    # data to features as encoder output data
    features = []
    #import pdb;pdb.set_trace()
    for batch_index, batch in enumerate(dataloader):
        batch = batch.to(device)
        #print(f"Batch index: {batch_index}")
        try:
            step_outputs = unsupervised_model.network.encoder(batch)[0]
        except Exception as e:
            print("\n"+f"Error occurred in batch {batch_index}: {e}")
            print(f"Batch shape: {batch.shape}")
            raise e
        encoder_out = sum(step for step in step_outputs)
        features.append(encoder_out)
    #import pdb; pdb.set_trace()
    features = torch.cat(features)
    features = features.detach()
    features = features.cpu()
    return features

def extract_feature_by_month(unsupervised_model,df):
    features = []
    # 月ごとにデータを分割
    month_data = utils.get_months(df)
    base_time = df["Timestamp"].min()
    for i in range(len(month_data)):
        #import pdb; pdb.set_trace()
        start = utils.new_time(base_time,i,"months")
        end = utils.new_time(start,int(month_data[i]),"minutes")
        df_i = utils.extract_specific_terms(df,start,end)
        X = df_i.drop(columns=['Timestamp','Turbine_ID']).values
        features.append(data_to_TabNetFeatures(unsupervised_model,X))
        print(f"--- {start} ---")
    features = torch.cat(features)
    features = features.detach()
    features = features.cpu()
    return features

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set parameter
feature_dim = 32
optimizer_params=dict(lr=3e-3)
gmm_multi = True
log_plot = False
covariance_type = 'full'
pretraining_ratio = 0.5
conditions = [("feature_dim",feature_dim),("optimizer_params",optimizer_params),("pretraining_ratio",pretraining_ratio)]


# Download dataset
dataset_name = "wind_edp"
full_data = pd.read_csv('data/'+dataset_name+'/SCADA.csv')
full_data = utils.arrange_data(full_data)
data = utils.extract_specific_id(full_data,'T07')
train = utils.extract_specific_terms(data,'2017-04-01T00:00:00+00:00','2017-05-17T23:50:00+00:00')
valid = utils.extract_specific_terms(data,'2017-05-18T00:00:00+00:00','2017-05-31T23:50:00+00:00')
test = utils.extract_specific_terms(data,'2017-04-01T00:00:00+00:00','2017-08-18T23:50:00+00:00')
#test2 = utils.extract_specific_terms(data,'2017-08-20T00:00:00+00:00','2017-12-31T23:50:00+00:00')
#test = pd.concat([test1, test2], axis=0)


out_dir = "result/" + dataset_name + "_tabnet/" + str(feature_dim) + "_dim"
if(covariance_type!='full'):
    out_dir += ("_" + covariance_type)
os.makedirs(out_dir, exist_ok=True)

# データとtimestampを分離
X_train = train.drop(columns=['Timestamp','Turbine_ID']).values
#X_test = test.drop(columns=['Timestamp','Turbine_ID']).values
X_test = test.drop(columns=['Timestamp','Turbine_ID']).values
X_valid = valid.drop(columns=['Timestamp','Turbine_ID']).values
timestamp = test['Timestamp']
#import pdb; pdb.set_trace()

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
max_epochs = 200 if not os.getenv("CI", False) else 2 # 1000

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs , patience=10000,
    batch_size=128, virtual_batch_size=128,
    pretraining_ratio=pretraining_ratio,
)

#import pdb; pdb.set_trace()
# data to features as encoder output data
feature_train = data_to_TabNetFeatures(unsupervised_model,X_train)
feature_test = data_to_TabNetFeatures(unsupervised_model,X_test)

""" feature_train = extract_feature_by_month(unsupervised_model,train)
feature_test = extract_feature_by_month(unsupervised_model,test) """
# import pdb; pdb.set_trace()

if gmm_multi:
    for n in range(10):
        anomaly_score, threshold, img_path = calc_AnomalyScore(n+1,covariance_type,feature_train,feature_test,out_dir)
        #import pdb;pdb.set_trace()
        plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)
else:
    n_components=3
    anomaly_score, threshold, img_path = calc_AnomalyScore(n_components,covariance_type,feature_train,feature_test,out_dir)
    plot_by_date(log_plot,anomaly_score,timestamp,threshold,img_path)

show_info(out_dir,conditions)