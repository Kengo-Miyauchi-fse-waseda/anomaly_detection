"""
Deep Neural Network with bottleneck
"""
import torch.nn as nn


class Dnn(nn.Module):
    """
    Deep Neural Network with bottleneck

    output: probability, bottlenck-feature
    """

    def __init__(self, input, bottleneck, output, mid1=64, mid2=32):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input, mid1)
        self.fc2 = nn.Linear(mid1, mid2)
        self.fc3 = nn.Linear(mid2, bottleneck)
        self.fc4 = nn.Linear(bottleneck, mid2)
        self.fc5 = nn.Linear(mid2, mid1)
        self.fc6 = nn.Linear(mid1, input)
        self.fc7 = nn.Linear(input, output)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        bn = self.fc3(x)
        output = self.relu(bn)
        output = self.fc4(output)
        output = self.relu(output)
        output = self.fc5(output)
        output = self.relu(output)
        output = self.fc6(output)
        output = self.relu(output)
        output = self.fc7(output)
        output = self.log_softmax(output)
        return output, bn


class Dnn2(nn.Module):
    """
    Deep Neural Network with bottleneck

    output: probability, bottlenck-feature
    """

    def __init__(self, input, bottleneck, output, mid1, mid2, mid3):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input, mid1)
        self.fc2 = nn.Linear(mid1, mid2)
        self.fc3 = nn.Linear(mid2, bottleneck)
        self.fc4 = nn.Linear(bottleneck, mid3)
        self.fc5 = nn.Linear(mid3, output)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        bn = self.fc3(x)
        output = self.relu(bn)
        output = self.fc4(output)
        output = self.relu(output)
        output = self.fc5(output)
        output = self.log_softmax(output)
        return output, bn


class Dnn3(nn.Module):
    """
    Deep Neural Network with bottleneck

    output: probability, bottlenck-feature
    """

    def __init__(self, input, bottleneck, output, mid1, mid2):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input, mid1)
        self.fc2 = nn.Linear(mid1, mid2)
        self.fc3 = nn.Linear(mid2, bottleneck)
        self.fc4 = nn.Linear(bottleneck, output)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        bn = self.fc3(x)
        output = self.relu(bn)
        output = self.fc4(output)
        output = self.log_softmax(output)
        return output, bn


class Dnn4(nn.Module):
    """
    Deep Neural Network with bottleneck

    output: probability, bottlenck-feature
    """

    def __init__(self, input, bottleneck, output, mid1, mid2, mid3):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input, mid1)
        self.fc2 = nn.Linear(mid1, mid2)
        self.fc3 = nn.Linear(mid2, mid3)
        self.fc4 = nn.Linear(mid3, bottleneck)
        self.fc5 = nn.Linear(bottleneck, output)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        bn = self.fc4(x)
        output = self.relu(bn)
        output = self.fc5(output)
        output = self.log_softmax(output)
        return output, bn
