import torch.nn as nn


class SingleAE(nn.Module):
    def __init__(self, input_size, bottleneck, mid1=32):
        super(SingleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, mid1),
            nn.ReLU(),
            nn.Linear(mid1, bottleneck),
        )

        self.decoder = nn.Sequential(nn.Linear(bottleneck, mid1), nn.ReLU(), nn.Linear(mid1, input_size), nn.Sigmoid())

    def forward(self, x):
        bn = self.encoder(x)
        x_bar = self.decoder(bn)
        return x_bar, bn


# Denoising AutoEncoder
class DAE(nn.Module):
    def __init__(self, input_dim, bottleneck, mid1=64, mid2=64):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid1),
            nn.BatchNorm1d(mid1),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid1, mid2),
            nn.BatchNorm1d(mid2),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid2, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(bottleneck, mid2),
            nn.BatchNorm1d(mid2),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid2, mid1),
            nn.BatchNorm1d(mid1),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bn = self.encoder(x)
        x_bar = self.decoder(bn)
        return x_bar, bn


# Denoising AutoEncoder without BatchNorm
class DAEwoBatchNorm(nn.Module):
    def __init__(self, input_dim, bottleneck, mid1=64, mid2=64):
        super(DAEwoBatchNorm, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid1),
            # nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid1, mid2),
            # nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid2, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(bottleneck, mid2),
            # nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid2, mid1),
            # nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(mid1, input_dim),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        bn = self.encoder(x)
        x_bar = self.decoder(bn)
        return x_bar, bn
