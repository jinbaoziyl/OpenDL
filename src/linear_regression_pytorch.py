import torch
from IPython import display
from matplotlib import pyplot as plt 
import numpy as np
import random
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        y = self.linear(x)
        return y

batch_size = 10
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

def linear_model_train():
    # 网络模型，损失函数和梯度下降函数
    net = LinearNet(num_inputs)
    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    # 初始化参数
    print("linear model set parameters init!")
    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)

    # 构造人工数据集
    print("linear data set generate begin!")
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    # 读取数据集函数
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    print("linear model training begin!")
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad() # 更新参数之前，梯度清零
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))

    print("linear model training ouput show!")
    dense = net.linear
    print(true_w, dense.weight)
    print(true_b, dense.bias)

if __name__ == '__main__':
    linear_model_train()