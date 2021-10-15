import torch
from torch.nn import *

import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision


class GroupNN(Module):
    def __init__(self, base_num, base_dim):
        super(GroupNN, self).__init__()
        self.mlp = [
            Linear(base_dim, base_dim),
            ReLU(),
            Linear(base_dim, base_dim),
            ReLU(),
        ]
        self.mlp = Sequential(*self.mlp)

    def forward(self, x1):
        res = x1
        res = self.mlp(res)
        return res


def base_test():
    base_num = 10
    base_dim = 2

    data1 = torch.randn(size=(32, base_num, base_dim), device=dev, requires_grad=False)
    group = GroupNN(base_num, base_dim).to(dev)

    # plt.quiver(torch.zeros_like(data1)[:, :, 0].cpu(), data1[:, :, 1].cpu())
    data2 = data1.cpu()
    plt.figure(1)
    plt.quiver(np.zeros_like(data2)[:, :, 0], np.zeros_like(data2)[:, :, 1], data2[:, :, 0], data2[:, :, 1])

    optimizer = torch.optim.Adam(group.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss().to(dev)
    EPOCH = 500
    for epoch in range(EPOCH):
        loss = criterion(group(data1) * group(data1), group(data1 * data1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss = {loss}')

    res = group(data1).cpu().detach().numpy()
    plt.figure(2)
    plt.quiver(np.zeros_like(res)[:, :, 0], np.zeros_like(res)[:, :, 1], res[:, :, 0], res[:, :, 1])
    # print(res)

    plt.figure(3)
    data2 = torch.randn(size=(1, base_num, base_dim), device=dev, requires_grad=False)
    res = group(data2).cpu().detach().numpy()
    plt.quiver(np.zeros_like(res)[:, :, 0], np.zeros_like(res)[:, :, 1], res[:, :, 0], res[:, :, 1])

    plt.show()


def train(EPOCH, train_loader, model):
    criterion = MSELoss().to(dev)
    optimizer = torch.optim.Adam(group.parameters(), lr=0.0005)

    for epoch in range(EPOCH):
        for sub, (x, _) in enumerate(train_loader):
            x = x.to(dev)
            r = torch.rand_like(x, requires_grad=False)
            loss = criterion(model(x @ r).detach(), (model(x) @ model(r)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (sub + 1) % 100 == 0:
                print(f'epoch = {epoch}, loss = {loss}')

    torch.save(model.state_dict(), '../../test/group.pkl')


def test(model, test_loader):
    model.eval()

    for sub, (x, _) in enumerate(test_loader):
        x = x.to(dev)
        r = torch.rand_like(x, requires_grad=False)
        x_cpu = x.cpu()
        plt.figure(1)
        plt.imshow(x_cpu[0, 0, :, :], cmap='gray', interpolation='none')

        plt.figure(2)
        r_res = model(r).cpu().detach().numpy()
        plt.imshow(r_res[0, 0, :, :], cmap='gray', interpolation='none')

        plt.figure(4)
        r_res = (r @ x).cpu().detach().numpy()
        plt.imshow(r_res[0, 0, :, :], cmap='gray', interpolation='none')

        plt.show()
        input()


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    train_loader = DataLoader(
        MNIST(
            '../../test/mnist/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        ),
        batch_size=32,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../test/mnist/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=32, shuffle=True)

    group = GroupNN(28, 28).to(dev)

    EPOCH = 5

    # train(EPOCH, train_loader, group)

    group.load_state_dict(torch.load('../../test/group.pkl'))

    test(group, test_loader)
