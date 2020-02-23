import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), 
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X
def DenseBlockShape():
    blk = DenseBlock(2, 3, 10)
    X = torch.rand(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

def TransitionBlock(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

def DenseNet():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlosk_%d" % i, DB)
        # 上一个稠密块的输出通道数
        num_channels = DB.out_channels
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, TransitionBlock(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, 10))) 

    return net

def DenseNetShape():
    net = DenseNet()
    X = torch.rand((1, 1, 96, 96))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, 'DenseNet output shape:\t', X.shape)

if __name__ == '__main__':
    #DenseBlockShape()
    #TransitionBlockShape()
    #DenseNetShape()
    net = DenseNet()
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
