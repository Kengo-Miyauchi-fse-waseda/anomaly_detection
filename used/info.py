from custom_module.record_info import recordInfo
feature_dim = 16
optimizer_params=dict(lr=3e-3)
gmm_multi = True
covariance_type = 'full'
conditions = [("feature_dim",feature_dim), ("optimizer_params",optimizer_params), ("gmm_multi",gmm_multi), ("covariance_type",covariance_type)]
dataset_name = "BATADAL"
out_dir = "result/"+dataset_name + "_tabnet/" + str(feature_dim) + "_dim"
if(covariance_type!='full'):
    out_dir += covariance_type
recordInfo(out_dir,conditions)