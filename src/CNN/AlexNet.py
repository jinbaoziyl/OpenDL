import time
import torch
from torch import nn, optim
import torchvision
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # 参数列表in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, 96, 11, 4),  # out_size = (in_size + 2*padding - kernel_size / stride) + 1, 224*2*0-11/4 + 1 = 59
            nn.ReLU(),   # 1. ReLU激活函数的计算更简单 2.另一方面，ReLU激活函数在不同的参数初始化方法下使模型更容易训练,应对梯度弥散、爆炸
            nn.MaxPool2d(3, 2),       # (59 + 2*0 - 3) / 2 + 1 = 24
            nn.Conv2d(96, 256, 5, 1, 2),    # (24 + 2*2 - 5 / 2) + 1 = 12
            nn.ReLU(),
            nn.MaxPool2d(3, 2),             # (12 + 2*0 - 3 / 2) + 1 = 5
            
            # 连续三个卷积层后接池化层
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5 ),   # 两个输出个数为4096的全连接层，带来将近1 GB的模型参数，使用dropout层应对过拟合

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output



if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.001, 5, 128
    net = AlexNet()

    # 图像高和宽扩大到AlexNet使用的图像高和宽224
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


