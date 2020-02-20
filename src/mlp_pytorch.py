import torch
from torch import nn
from torch.nn import init
import numpy as np 
import sys
import open_dl_utils as d2l

batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    num_epochs = 5
    d2l.train_softmax_regression(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
