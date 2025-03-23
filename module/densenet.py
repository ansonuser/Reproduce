import torch.nn as nn
import torch

class BNConvRe(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride=1):
        super().__init__()
        self.block = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, size, stride, padding=size//2),
             nn.ReLU(),
             nn.BatchNorm2d(out_channels)
        )

    def __str__(self):
        return "BNC"
    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):
    def __init__(self, init_k=64, growth_rate=32, n=6):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(n):
            self.layers.append( nn.Sequential(
                BNConvRe(init_k + i*growth_rate, growth_rate, size=1), 
                BNConvRe(growth_rate, growth_rate, size=3)
            ))
        
    def __str__(self):
        return f"DenseBlock(n={self.n})"
    
    def forward(self, x):
        cat_x = [x]
        for layer in self.layers:
            cat_x.append(layer(x))
            x = torch.cat(cat_x, dim=1)
        return x



class Densenet(nn.Module):
    def __init__(self, growth_rate=32, init_k=64):
        super().__init__()
        block_list = [6, 12, 24, 16]
        n = len(block_list)
        
        self.layers = nn.ModuleList()
        self.layers.append(BNConvRe(3, init_k, 7, 2))
        self.layers.append(nn.MaxPool2d(3, 2, padding=1))
        for idx, b in enumerate(block_list):
            self.layers.append(DenseBlock(init_k, growth_rate, n=b))
            init_k += growth_rate*b
            if idx != len(block_list)-1:
                self.layers.append(BNConvRe(init_k , init_k//2, 1))
                self.layers.append(nn.AvgPool2d(2, 2))
                init_k = init_k//2
            else:
                self.layers.append(nn.AdaptiveAvgPool2d((1,1)))
                self.layers.append(nn.Flatten())
        self.linear = nn.Linear(init_k, 1000)
        self.layers.append(self.linear)
     
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        for layer in self.layers:   
            x = layer(x)
        return self.softmax(x)