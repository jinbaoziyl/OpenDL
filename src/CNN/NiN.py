# LeNet、AlexNet和VGG在设计上的共同之处: 都是先卷积层抽取空间特征，然后由全连接层构成的模块来输出分类结果
# NiN: 串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络
# 1x1卷积层可替代全连接层
import time
import torch
from torch import nn, optim
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# nin块: 一个卷积层加两个充当全连接层的1×11×1卷积层串联而成
# 其中第一个卷积层进行通道转换，而第二和第三个卷积层的输入、输出由第一个卷积层的输出通道决定
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blks = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
    return blks


def nin_net():
    net = nn.Sequential(
        nin_block(1, 96,kernel_size=11, stride=4, padding=0 ),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),

        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        d2l.GlobalAvgPool2d(), #全局池化层,这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合
        d2l.FlattenLayer(),
    )    

    return net

def vgg_nets_shape():
    net = nin_net()
    X = torch.rand(1, 1, 224, 224)
    for name, blk in net.named_children(): 
        X = blk(X)
        print(name, 'NiN CNN output shape: ', X.shape)

if __name__ == '__main__':
    #vgg_nets_shape()
    net = nin_net()

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
   
