import torch
import torch.nn as nn
import numpy as np 
import open_dl_utils as d2l

# 模型参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256 #两个隐藏层的MLP
drop_prob1, drop_prob2 = 0.2, 0.5  #两个隐藏层丢弃概率

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

# 以drop_prob的概率丢弃X中的元素
def dropout(X, drop_pro):
    X = X.float()
    assert 0 <= drop_pro <= 1

    keep_prob = 1 - drop_pro
    if(keep_prob == 0):
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob

# 定义模型
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2, W3) + b3


if __name__ == '__main__':
    num_epochs, lr, batch_size = 5, 100.0, 256
    loss = torch.nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_softmax_regression(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
