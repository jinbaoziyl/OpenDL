# GoogLeNet主要模块是inception
# inception的设计: 
#   通过不同窗口形状的卷积层和最大池化层来并行抽取信息,并使用1×11×1卷积层减少通道数从而降低模型复杂度
# inception的思想: 
#   并行网络块，保存同样的宽高, 最后通过通道连接起来, 通过深度网络自动学习到使用卷积核的尺寸
#  

import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class Inception(nn.Module):
    # 4条并行网络
    def __init__(self, in_channel, L1_channel, L2_channel, L3_channel, L4_channel):
        super(Inception, self).__init__()
        # 线路1, 1x1卷积层
        self.p1_1 = nn.Conv2d(in_channel, L1_channel, kernel_size=1)
        # 线路2, 1x1卷积层+3x3卷积层
        self.p2_1 = nn.Conv2d(in_channel, L2_channel[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(L2_channel[0], L2_channel[1], kernel_size=3, padding=1)
        # 线路3, 1x1卷积层+5x5卷积层
        self.p3_1 = nn.Conv2d(in_channel, L3_channel[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(L3_channel[0], L3_channel[1], kernel_size=5, padding=2)        
        # 线路4, 3x3最大池化+1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channel, L4_channel, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        
        return torch.cat((p1, p2, p3, p4), dim=1)

def GoogLeNet():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        d2l.GlobalAvgPool2d()
        )
    net = nn.Sequential(b1, b2, b3, b4, b5, 
        d2l.FlattenLayer(), nn.Linear(1024, 10))
    
    return net

def google_nets_shape():
    net = GoogLeNet()
    X = torch.rand(1, 1, 96, 96)
    for name, blk in net.named_children(): 
        X = blk(X)
        print(name, 'GooleLeNet output shape: ', X.shape)

if __name__ == '__main__':
    #google_nets_shape()
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    net = GoogLeNet()

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

