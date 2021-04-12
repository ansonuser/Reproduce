import torch.nn as nn


class BNConvRe(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, size, stride)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self, x)
        return self.bn1(self.relu1(self.conv1(x)))


class DenseBlock(nn.Module):
    def __init__(self, init_k, n=6):
        super().__init__()
        self.n = n
        self.layers = []
        cum_k = 1
        for i in range(n):
            cum_k += i
            self.layers.append([BNConvRe(cum_k*init_k, init_k, 1), BNConvRe(init_k, init_k, 3)])
    def forward(self, x):
        cum_list = []
        for l in self.layers:
            cum_list.append(x)
            x = torch.cat(*cum_list, 1)
            x = l(x)
        return x        

class Densenet(nn.Module):
    def __init__(self):
        super().__init__()
        block_list = [6, 12, 24, 16]
        n = len(block_list)
        self.layers = []
        self.layers.append(BNConv(3, 32, 7, 2))
        self.layers.append(nn.MaxPool2d(3, 2))
        for idx, b in enumerate(block_list):
            self.layers.append(DenseBlock(32, b))
            if idx != n:
                self.layers.append(BNConvRe(32, 32, 1))
                self.layers.append(nn.AvgPool2d(2, 2))
            else:
                self.layers.append(nn.AvgPool2d(7, 1))
        self.liner = nn.Linear(32, 1000)
        self.layers.append(self.liner)
        self.seq = nn.Sequential(*self.liner)
    def forward(self, x):
        return nn.Softmax(self.seq(x))