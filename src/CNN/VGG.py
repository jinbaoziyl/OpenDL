# vvg网络意义: 对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核优于采用大的卷积核，
# 因为可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）
import time
import torch
from torch import nn, optim
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# vgg块的参数， 卷积层的个数+输入通道数+输出通道数
convs_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 5个vgg块，宽高会减半5次， --> 221/2^5 = 7
fc_features = 512*7*7
fc_hidden_units = 4096

# vgg网络都是由基本网络块组成，最后输入全连接层
# vgg块组成: 连续使用数个相同的填充为1、窗口形状为3×3的卷积层后接上一个步幅为2、窗口形状为2×2的最大池化层
# 目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果
def vgg_block(num_convs, in_channels, out_channels):
    blks = []
    for i in range(num_convs):
        if i == 0:
            blks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  #第一个卷积层完成通道转换
        else:
            blks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blks.append(nn.ReLU())
    blks.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blks)

def vgg11(convs_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层-连续的vgg块
    for i, (num_convs, in_channels, out_channels) in enumerate(convs_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    
    # fully connect layers
    net.add_module("fc", nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10),
    ))
    return net

def vgg_nets_shape():
    net = vgg11(convs_arch, fc_features, fc_hidden_units)
    X = torch.rand(1, 1, 224, 224)

    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)

if __name__ == "__main__": 
    #vgg_nets_shape()
    # 构造网络
    ratio = 8  #减少通道数
    small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                    (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
    net = vgg11(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)

    # 读取数据集
    batch_size = 64
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # 训练模型  
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

