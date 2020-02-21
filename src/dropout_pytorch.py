import torch
import torch.nn as nn
import numpy as np 
import open_dl_utils as d2l

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256 #两个隐藏层的MLP
drop_prob1, drop_prob2 = 0.2, 0.5  #两个隐藏层丢弃概率
num_epochs, lr, batch_size = 5, 100.0, 256

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = torch.nn.CrossEntropyLoss()
    net = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Dropout(drop_prob1),
            nn.Linear(num_hiddens1, num_hiddens2), 
            nn.ReLU(),
            nn.Dropout(drop_prob2),
            nn.Linear(num_hiddens2, 10)
            )

    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    d2l.train_softmax_regression(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

