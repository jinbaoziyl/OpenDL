# V2的改进: Inverted Residuals and Linear Bottlenecks
# 1.Inverted Residuals: “升维-卷积-降维”   Residuals: “降维-卷积-升维”
#                 原因: 因为depthwise本身没有改变通道数的能力，而其在低维度上的表现又比较糟糕，因此，需要升维，从而提高准确率
# 2.Linear Bottlenecks: 把最后那个ReLU6给替换成了线性的激活函数
#                 原因: 对低维度做ReLU运算，信息丢失的很多，但是如果是对高维度做ReLU运算，信息的丢失就会相对减少了

import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
from torch.autograd import Variable

import math
import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Standard convolutional layer with batchnorm and ReLU
def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)       
    )

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Pointwise layer, when expand_ratio=1, 1x1conv and oup=input ----> the same
                # Depthwise layer
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-Linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise layer
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), # Dw层前加一层Pw:因为Dw层没有改变通道数量能力，可能导致Dw只能在低维提取特征
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise layer
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-Linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # Linear Bottlenecks: 去掉激活函数，因为在高维空间增加非线性，而低维破坏非特性，线性激活更好
                nn.BatchNorm2d(oup),                             
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # building first layer
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))  # conv2d 1x1
        self.features.append(nn.Sequential(nn.AvgPool2d(7)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(self.last_channel, 1000)
        ) 
        # init paramaters
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        output = self.fc(features.view(-1, self.last_channel))  #将最后一层卷积层 打平成全连接层输入
        return output


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def MobileNetV2Shape():
    net = MobileNetV2()
    print(net)
    X = torch.rand(1,3,224,224)
    for name, layer in net.named_children():
        X = layer(X)
        print('MobileNet', name, 'output shape:\t', X.shape)     

if __name__ == '__main__':
    MobileNetV2Shape()
    