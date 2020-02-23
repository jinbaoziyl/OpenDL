# 神经网络加深网络深度的两个思路:
# 1. 批量归一化层（batch normalization）, 通过数值稳定性，但是问题依然存在
# 2. ResNet
import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def InceptionShape():
    blk = d2l.Inception(3, 3)
    X = torch.rand((4, 3, 6, 6))
    print(blk(X).shape)

def ResnetBlock(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(d2l.Inception(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(d2l.Inception(out_channels, out_channels))
    return nn.Sequential(*blk)


def ResNet():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", ResnetBlock(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", ResnetBlock(64, 128, 2))
    net.add_module("resnet_block3", ResnetBlock(128, 256, 2))
    net.add_module("resnet_block4", ResnetBlock(256, 512, 2))

    # 和GooLeNet一样先接平均池，然后全连接层输出
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10))) 
    return net

def ResNetShape():
    net = ResNet()
    X = torch.rand((1, 1, 224, 224))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, 'ResNet output shape:\t', X.shape)


if __name__ == '__main__':
    #InceptionShape()
    #ResNetShape()
    net = ResNet()

    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


