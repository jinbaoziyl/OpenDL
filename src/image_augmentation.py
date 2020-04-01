import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import pylab
import sys
sys.path.append(".")
import open_dl_utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('/home/linyang/workspace/projects/OpenDL/src/person.jpg')
# d2l.plt.imshow(img)
# pylab.show()

# 一半概率的图像水平（左右）翻转
d2l.apply(img, torchvision.transforms.RandomHorizontalFlip())

# 一半概率的图像垂直（上下）翻转
d2l.apply(img, torchvision.transforms.RandomVerticalFlip())

shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
d2l.apply(img, shape_aug)

# 图像的亮度随机变化为原图亮度的0.5
d2l.apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 随机变化图像的色调
d2l.apply(img, torchvision.transforms.ColorJitter(hue=0.5))

# 随机变化图像的对比度
d2l.apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
d2l.apply(img, color_aug)

# 叠加多个图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
d2l.apply(img, augs)

all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)

flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root='~/Datasets/CIFAR'):
    dataset = torch.datasets.CIFAR10(root=root,train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_worker)



def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size ,net = 256, resnet18(10)
    optimizer = torch.optim.Adam(net.params(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(True, test_augs, batch_size)

    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)