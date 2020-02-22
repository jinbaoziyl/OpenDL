# 卷积神经网络- 含卷积层的网络
# 模型设计: LeNet交替使用卷积层和最大池化层(2个)后接3个全连接层处理图像分类
# 意义: 一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别
#       另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大
import torch
import time
from torch import nn, optim
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),   # p=0 s=1 -> 28 - 5 + 1 = 24
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),  # p=0 s=2 -> 24/2 = 12
            nn.Conv2d(6,16,5),  # p=0 s=1 -> 12 - 5 + 1 = 8  尺寸减少，增加输出通道使两个卷积层的参数尺寸类似
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),  # p=0 s=2 -> 8/2 = 4  ==> 16channes  h,w = 4,4
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            nn.Sigmoid(),            
        )

    def forward(self, img):
        feature = self.conv(img)
        out = self.fc(feature.view(img.shape[0], -1))  # batch size图像: [batch_size, flatten的图像数据]
        return out 

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.001, 5, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    net = LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

