import torch
from util_module.create_dataloader import create_dataloader

# 再構成誤差の計算
def calc_reconstruction_errors(exec_model, data):
    dataloader = create_dataloader(data, exec_model.batch_size_tr, need_shuffle=False)
    reconstruction_errors = []

    for batch in dataloader:
        batch = batch.to(exec_model.device)
        try:
            # 入力データを再構成
            reconstructed = exec_model.unsupervised_model.network.decoder(
                exec_model.unsupervised_model.network.encoder(batch)[0]
            )
            
            # 再構成誤差を計算（MSE：平均二乗誤差）
            error = ((batch - reconstructed) ** 2).mean(dim=1)
            reconstruction_errors.append(error.cpu().detach())
        except Exception as e:
            print(f"Error occurred during reconstruction: {e}")
            raise e

    reconstruction_errors = torch.cat(reconstruction_errors)
    return reconstruction_errors
