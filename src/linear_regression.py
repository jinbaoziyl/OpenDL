import torch
from IPython import display
from matplotlib import pyplot as plt 
import numpy as np
import random
import open_dl_utils as d2l

# 声明变量
num_inputs = 2
num_example = 1000
true_w = [2,3.4]
true_b = 4.2

def linear_model_train():
    print("linear model set parameters init!")
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True) 

    # 人工构造数据集
    print("linear data set generate begin!")
    features = torch.randn(num_example, num_inputs, dtype=torch.float32)
    labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                        dtype=torch.float32)
    
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = d2l.linreg
    loss = d2l.squared_loss

    print("linear model training begin!")
    for epoch in range(num_epochs):  
        for X, y in d2l.data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            d2l.sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
            
            # 开始下次迭代前，需要梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

    print("linear model training ouput show!")
    print(true_w, '\n', w)
    print(true_b, '\n', b)


if __name__ == '__main__':
    linear_model_train()