import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from torch.nn import init
import open_dl_utils as d2l
from collections import OrderedDict

batch_size = 256
num_inputs = 784
num_outputs = 10
num_epochs = 5

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x):
        y = nn.linear(x.view(x.shape[0], -1))
        return y

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
            ('flatten', d2l.FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
        ])
    )

    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0) 

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    d2l.train_softmax_regression(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


    
