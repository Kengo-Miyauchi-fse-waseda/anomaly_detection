import pandas as pd
from sklearn.model_selection import train_test_split
from util_module.build_exec_model import ExecModel

dataset_name = "BATADAL"

# Download dataset
train = pd.read_csv('data/'+dataset_name+'_train.csv')
test = pd.read_csv('data/'+dataset_name+'_test.csv')

# データとtimestampを分離
X_train_base = train.drop('DATETIME',axis='columns').values
X_test = test.drop('DATETIME',axis='columns').values
timestamp = test['DATETIME']
X_train, X_valid = train_test_split(X_train_base, test_size=0.2, random_state=42)

exec_model = ExecModel(X_train,X_valid)
import pdb; pdb.set_trace()