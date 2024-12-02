import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

def data_to_TabNetFeatures(exec_model,data):
    dataloader = create_dataloader(data,batch_size=exec_model.batch_size,need_shuffle=True)
    # data to features as encoder output data
    features = []
    for batch_index, batch in enumerate(dataloader):
        batch = batch.to(exec_model.device)
        try:
            step_outputs = exec_model.unsupervised_model.network.encoder(batch)[0]
        except Exception as e:
            print("\n"+f"Error occurred in batch {batch_index}: {e}")
            print(f"Batch shape: {batch.shape}")
            raise e
        encoder_out = sum(step for step in step_outputs)
        features.append(encoder_out)
    features = torch.cat(features)
    features = features.detach()
    features = features.cpu()
    return features