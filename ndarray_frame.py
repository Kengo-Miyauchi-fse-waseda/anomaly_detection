import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.gmm_module import calc_AnomalyScore
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet_feature import create_dataloader, data_to_framedFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
dataset_name = "haenkaze"
stampcol = "DateTime"
frequency = '1S'
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = utils.arrange_data(data,stampcol)
print("Finish loading data.")

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
model_name = "tabnet-gmm-3frame"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = exec_model.out_dir


# convert data to tabnet encoder features
num_frame = 3
print("Start Feature Extraction")
feature_train = data_to_framedFeatures(exec_model,X_train,num_frame,True)
del X_train
print("Finish Feature Extraction")

from sklearn.mixture import GaussianMixture
import numpy as np
n_components = 10
print("GMM Training")
gmm = GaussianMixture(n_components=n_components, covariance_type=exec_model.covariance_type, random_state=42, n_init=10, max_iter=500)
gmm.fit(feature_train)
log_likelihood = -gmm.score_samples(feature_train)
threshold = np.percentile(log_likelihood,99.9)
del feature_train, log_likelihood
print("Finish GMM Training")

print("Start Test Calculation")
#feature_test = data_to_framedFeatures(exec_model,X_test,num_frame,False)
dataloader = create_dataloader(X_test,exec_model.batch_size_tr,False)
# data to features as encoder output data
anomaly_score = []
for batch_index, batch in enumerate(dataloader):
    batch_output = []
    batch = batch.to(exec_model.device)
    try:
        step_outputs = exec_model.unsupervised_model.network.encoder(batch)[0]
    except Exception as e:
        print("\n"+f"Error occurred in batch {batch_index}: {e}")
        print(f"Batch shape: {batch.shape}")
        raise e
    encoder_out = sum(step for step in step_outputs)
    encoder_out = encoder_out.detach().cpu()
    del step_outputs, batch
    torch.cuda.empty_cache()
    i = 0
    #import pdb; pdb.set_trace()
    while((len(encoder_out)-i)>=num_frame):
        framed = []
        for j in range(num_frame):
            framed.extend(encoder_out[j])
        i+=1
        batch_output.append(framed)
    for j in range(num_frame-1):
        batch_output.append(framed)
    del framed
        #import pdb; pdb.set_trace()
    anomaly_score.extend(-gmm.score_samples(batch_output))
    del encoder_out, batch_output
    if ((batch_index+1)%100==0):print(f"{batch_index+1}/{len(dataloader)} batch:")
del X_test
print("Finish Test Calculation")

event_files = ["data/haenkaze/events.csv","data/haenkaze/event_range.csv"]
score_file = "result/haenkaze/tabnet-gmm/"+frequency+"scores.csv"
img_path = out_dir + "/"+frequency+"_3frame_gmm_" + str(n_components) + "components.png"
plot_by_date(exec_model.log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_files,score_file)
print(f"result: {img_path}")
show_info(out_dir,exec_model)