import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
from torch.autograd import Variable

import sys
sys.path.append("D:/NN/OpenDL/src/")
import open_dl_utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # Standard convolutional layer with batchnorm and ReLU
        def conv_normal(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Depthwise Separable convolutions with Depthwise and Pointwise layers followed by batchnorm and ReLU
        def conv_depthwise(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_normal(     3,  32, 2), 
            conv_depthwise( 32,  64, 1),
            conv_depthwise( 64, 128, 2),
            conv_depthwise(128, 128, 1),
            conv_depthwise(128, 256, 2),
            conv_depthwise(256, 256, 1),
            conv_depthwise(256, 512, 2),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 1024, 2),
            conv_depthwise(1024, 1024, 1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(1024, 1000)
        )
    
    def forward(self, x):
        features = self.model(x)
        output = self.fc(features.view(-1, 1024))  #将最后一层卷积层 打平成全连接层输入
        return output

def Speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cuda()
    input = Variable(input)  #Varibale 默认时不要求梯度的，如果要求梯度，需要说明

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))
    #print(model)

def MobileNetShape():
    net = MobileNet()
    print(net)
    X = torch.rand(1,3,224,224)
    for name, layer in net.named_children():
        X = layer(X)
        print('MobileNet', name, 'output shape:\t', X.shape)


if __name__ == '__main__':
    resnet18 = models.resnet18().cuda()
    alexnet = models.alexnet().cuda()
    vgg16 = models.vgg16().cuda()
    squeezenet = models.squeezenet1_0().cuda()
    mobilenet = MobileNet().cuda()

    Speed(resnet18, 'resnet18')
    Speed(alexnet, 'alexnet')
    Speed(vgg16, 'vgg16')
    Speed(squeezenet, 'squeezenet')
    Speed(mobilenet, 'mobilenet')

    # mobileNet Shape
    MobileNetShape()