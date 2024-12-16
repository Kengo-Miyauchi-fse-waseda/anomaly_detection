import torch.nn as nn
import torch

# Convolutional AutoEncoder
class CAE(nn.Module):
    def __init__(self, bottleneck, mid1=8, mid2=16, mid3=32):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mid1, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid2, mid3, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.Flatten(),
            nn.Linear(8 * 8 * mid3, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8 * 8 * mid3),
            nn.ReLU(True),
            nn.Unflatten(1, (mid3, 8, 8)),
            nn.ConvTranspose2d(mid3, mid2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid2, mid1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        bn = self.encoder(x)
        x_bar = self.decoder(bn)
        return x_bar, bn


# Convolutional Variational AutoEncoder
class CVAE(nn.Module):
    def __init__(self, bottleneck, mid1=8, mid2=16, mid3=32):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mid1, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid2, mid3, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.Flatten(),
        )

        self.lr_ave = nn.Linear(32 * 1 * mid3, bottleneck)  # Linear layer for average
        self.lr_dev = nn.Linear(32 * 1 * mid3, bottleneck)  # Linear layer for log of standard deviation

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32 * 1 * mid3),
            nn.ReLU(True),
            nn.Unflatten(1, (mid3, 32, 1)),
            nn.ConvTranspose2d(mid3, mid2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid2, mid1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        ave = self.lr_ave(x)
        log_dev = self.lr_dev(x)
        bn = ave + torch.exp(log_dev / 2) * torch.randn_like(ave)
        x_bar = self.decoder(bn)
        return x_bar, bn, ave, log_dev


# Convolutional Variational AutoEncoder
class CVAE2(nn.Module):
    def __init__(self, bottleneck, mid1=8, mid2=16, mid3=32):
        super(CVAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mid1, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.Flatten(),
        )

        self.lr_ave = nn.Linear(64 * 2 * mid2, bottleneck)  # Linear layer for average
        self.lr_dev = nn.Linear(64 * 2 * mid2, bottleneck)  # Linear layer for log of standard deviation

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64 * 2 * mid2),
            nn.ReLU(True),
            nn.Unflatten(1, (mid2, 64, 2)),
            nn.ConvTranspose2d(mid2, mid1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        ave = self.lr_ave(x)
        log_dev = self.lr_dev(x)
        bn = ave + torch.exp(log_dev / 2) * torch.randn_like(ave)
        x_bar = self.decoder(bn)
        return x_bar, bn, ave, log_dev


# Convolutional Variational AutoEncoder
class CVAE3(nn.Module):
    def __init__(self, bottleneck, mid1=8, mid2=16, mid3=32, mid4=64):
        super(CVAE3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mid1, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid2, mid3, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Conv2d(mid3, mid4, kernel_size=1, stride=1),
            nn.Flatten(),
        )

        self.lr_ave = nn.Linear(32 * 1 * mid4, bottleneck)  # Linear layer for average
        self.lr_dev = nn.Linear(32 * 1 * mid4, bottleneck)  # Linear layer for log of standard deviation

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32 * 1 * mid4),
            nn.ReLU(True),
            nn.Unflatten(1, (mid4, 32, 1)),
            nn.ConvTranspose2d(mid4, mid3, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid3, mid2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid2, mid1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        ave = self.lr_ave(x)
        log_dev = self.lr_dev(x)
        bn = ave + torch.exp(log_dev / 2) * torch.randn_like(ave)
        x_bar = self.decoder(bn)
        return x_bar, bn, ave, log_dev
