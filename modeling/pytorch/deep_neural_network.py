"""
Deep Neural Network with bottleneck
"""

import torch.nn as nn


class Dnn(nn.Module):
    """
    Deep Neural Network with bottleneck

    output: probability, bottlenck-feature
    """

    def __init__(self, input_size, bottleneck, mid1=64, mid2=64, mid3=64):
        super(Dnn, self).__init__()
        self.mid1 = mid1
        self.mid2 = mid2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.mid1),
            nn.ReLU(True),
            nn.Linear(self.mid1, self.mid2),
            # nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(self.mid2, bottleneck),
        )

        self.decoder = nn.Sequential(nn.ReLU(True), nn.Linear(bottleneck, mid3), nn.ReLU(True), nn.Linear(mid3, 1), nn.Sigmoid())

    def forward(self, x):
        bn = self.encoder(x)
        output = self.decoder(bn)
        return output, bn


class DnnForClass(nn.Module):
    """
    Deep Neural Network for classification

    output: probability for each class, bottlenck-feature
    """

    def __init__(self, input_size, bottleneck, output, mid1=64, mid2=64, mid3=64):
        super(DnnForClass, self).__init__()
        self.mid1 = mid1
        self.mid2 = mid2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.mid1),
            nn.ReLU(True),
            nn.Linear(self.mid1, self.mid2),
            # nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(self.mid2, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(bottleneck, mid3),
            nn.ReLU(True),
            nn.Linear(mid3, output),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        bn = self.encoder(x)
        output = self.decoder(bn)
        return output, bn


class DnnConvForClass(nn.Module):
    """
    Deep Neural Network for classification

    output: probability for each class, bottlenck-feature
    """

    def __init__(self, bottleneck, output, filter1=64, filter2=64):
        super(DnnConvForClass, self).__init__()

        self.encoder = nn.Sequential(
            # (N, 1, 75)
            nn.Conv1d(1, filter1, kernel_size=7, stride=2, padding=40, padding_mode="replicate"),
            # (N, filter1, 75)
            # nn.BatchNorm1d(filter1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=5),
            # (N, filter1, 15)
            nn.Conv1d(filter1, filter2, kernel_size=5, stride=4, padding=23, padding_mode="replicate"),
            # (N, filter2, 15)
            # nn.BatchNorm1d(filter2),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3),
            # (N, filter2, 5)
            nn.Conv1d(filter2, bottleneck, kernel_size=3),
            # (N, bottleneck, 3)
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3),
            # (N, bottleneck, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, output),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        N = x.size()[0]
        x = x.view(N, 1, -1)
        bn = self.encoder(x)
        bn = bn.view(N, -1)
        output = self.decoder(bn)
        return output, bn
