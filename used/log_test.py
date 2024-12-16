import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.callbacks import Callback
import torch

class LogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """エポック終了時に呼び出されるコールバック"""
        loss = logs.get("loss")  # トレーニング損失
        val_loss = logs.get("val_0_unsup_loss_numpy")  # 検証損失
        logging.info(f"epoch {epoch+1 :>3} | loss: {loss:.4f}, val_0_unsup_loss_numpy: {val_loss:.4f}")

# ログの設定
logging.basicConfig(
    filename='training.log',  # ログファイルのパス
    filemode='w',             # 書き込みモード ('w'は上書き、'a'は追記)
    level=logging.INFO,       # ログレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download dataset
dataset_name = "BATADAL"
train = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_train.csv')
test = pd.read_csv('data/'+dataset_name+'/'+dataset_name+'_test.csv')
train = train.fillna(method="ffill")
test = test.fillna(method="ffill")

# データとtimestampを分離
X_train_base = train.drop('DATETIME',axis='columns').values
X_test = test.drop('DATETIME',axis='columns').values
timestamp = test['DATETIME']
X_train, X_valid = train_test_split(X_train_base, test_size=0.2, random_state=42)

# TabNetPretrainerのインスタンスを作成
unsupervised_model = TabNetPretrainer(
    verbose=0  # ログ出力を有効化
)

# 学習を実行
logging.info("Starting TabNet pretraining...")
unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=100,
    batch_size=256,
    pretraining_ratio=0.5,
    num_workers=0,
    callbacks=[LogCallback()]
)
logging.info("TabNet pretraining finished.")
