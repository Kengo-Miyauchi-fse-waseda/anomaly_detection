"""
Deep Autoencoder trainer
"""
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DaeTrainer(object):
    def __init__(self, n_epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                 device: str = 'cuda', verbose: bool = False):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.verbose = verbose

    def train(self, dataset: Dataset, model: nn.Module):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print('DAE training...')
        model.train()
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss().to(self.device)

        sum_loss = 0.
        for i_epoch in range(1, self.n_epochs + 1):
            for batch_idx, (x_input, x_output) in enumerate(dataloader):
                x_input, x_output = x_input.to(self.device), x_output.to(self.device)
                # ===============forward=================
                x_bar, _ = model(x_input)
                loss = criterion(x_output, x_bar)
                # ===============backward================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =================log===================
                sum_loss += loss

            if self.verbose:
                print("{} epoch\tDecoder loss: {:.4f}".format(epoch, sum_loss / len(dataloader)))

        return model
