import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import sys
import open_dl_utils as d2l

batch_size = 256
num_inputs = 784
num_outputs = 10
num_epochs, lr = 5, 0.1 # 训练超参数

W = []
b = []

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

if __name__ == '__main__':
    # 初始化参数
    print("set parameters init!")
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True) 

    # 加载数据集
    print("load data sets!")
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 训练模型
    print("softmax regression model training begin!")
    d2l.train_softmax_regression(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    # 预测
    print("predict classfication!")
    X, y = iter(test_iter).next()

    plt.plot([5,4,3,2,1])   
    plt.show()

    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])






 