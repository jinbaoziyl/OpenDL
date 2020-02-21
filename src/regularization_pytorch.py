# L2范数正则化，又称权重衰减
import torch
import torch.nn as nn
import numpy as np 
import open_dl_utils as d2l
from matplotlib import pyplot as plt


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1)*0.01, 0.05

batch_size, num_epochs, lr = 1, 100, 0.003

def data_set_generator():
    features = torch.randn((n_train + n_test, num_inputs))
    labels = torch.matmul(features, true_w) + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    return [train_features, test_features, train_labels, test_labels]

def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def train_pytorch_model_and_plot(wd):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)

    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    loss = d2l.squared_loss
    
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())           

if __name__ == '__main__':
    # 构造人工数据集，y=0.05+∑0.01x + ϵ
    train_features, test_features, train_labels, test_labels = data_set_generator()

    # 读取数据集
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    plt.plot([5,4,3,2,1])   
    plt.show()

    train_pytorch_model_and_plot(0)
    train_pytorch_model_and_plot(3)




