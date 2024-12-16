import pandas as pd
import numpy as np
np.random.seed(0)
from wind_module import utils

# Download dataset
dataset_name = "wind_edp"
full_data = pd.read_csv('data/'+dataset_name+'/SCADA.csv')
full_data = utils.arrange_data(full_data)
data = utils.extract_specific_id(full_data,'T07')
test = utils.extract_specific_terms(data,'2017-01-19T14:20:00+00:00','2017-08-19T14:20:00+00:00')
arr = test.drop(columns=['Timestamp','Turbine_ID']).values
""" for i in range(len(arr[0])):
    print(f"{i}th data: {arr[0][i]}  type:{type(arr[0][i])}")
#print(X_test)
#print(f"Data type: {arr.dtype}")
if np.isnan(arr).any():
    print("NaN values found!")
if np.isinf(arr).any():
    print("Inf values found!") """
print(utils.get_months(test))
