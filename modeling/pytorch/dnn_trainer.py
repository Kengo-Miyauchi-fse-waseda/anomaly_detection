"""
DNN trainer
"""
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DnnTrainer(object):
    def __init__(self, n_epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                 device: str = 'cuda', verbose: bool = False):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.verbose = verbose

    def train(self, dataset: Dataset, model: nn.Module):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print('DNN training...')
        model.train()
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss().to(self.device)

        for i_epoch in range(1, self.n_epochs + 1):
            sum_loss = 0.
            _s = time.time()

            for batch_idx, (x_input, y_label) in enumerate(dataloader):
                x_input, y_label = x_input.to(self.device), y_label.to(self.device)
                # ===============forward=================
                output, _ = model(x_input)
                loss = criterion(output, y_label)
                # ===============backward================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =================log===================
                sum_loss += loss
            _e = time.time()
            # if self.verbose:
            print(f"Train Epoch: {i_epoch}\tloss: {sum_loss / len(dataloader.dataset):.4f}\tElapsed time: {_e - _s:.3f}")
        return model
