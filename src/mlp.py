# multilayer perceptron，MLP 多层感知机
import torch
import numpy as np 
import sys
import open_dl_utils as d2l

batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256

W1 = []
b1 = []
W2 = []
b2 = []

def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view((-1,num_inputs)) 
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)

    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)   

    loss = torch.nn.CrossEntropyLoss()

    # 训练模型
    num_epochs, lr = 5, 100.0
    d2l.train_softmax_regression(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)



 
