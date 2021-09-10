import torch

from torch.nn.modules import Module
from torch.nn import *
import numpy as np


class AttentionMap(Module):
    def __init__(self, channels, in_dim):
        super().__init__()
        adj = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        ]
        self.A = Parameter(torch.Tensor(adj), requires_grad=False)
        d = np.sum(adj, axis=1)
        d = np.diag(d)
        d = np.linalg.cholesky(d)
        d = np.linalg.inv(d)
        self.D = Parameter(torch.Tensor(d), requires_grad=False)
        self.W = Parameter(torch.ones(size=(in_dim, in_dim)), requires_grad=True)

    def forward(self, x):
        # print(x.size(), self.attention.size())
        res = torch.mm(self.D, self.A)
        res = torch.mm(res, self.D)
        res = torch.matmul(res, x)
        res = torch.matmul(res, self.W)
        return res


class KeyPointLearner(Module):
    """
    input 1 shape = (batch, keypoints_num, 3)
    input 1 shape = (batch, keypoints_num, keypoints_num)
    """

    def __init__(self, keypoints_num=26):
        super().__init__()
        self.attention = AttentionMap(1, 11)
        self.kpm_model = [
            # Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            AttentionMap(1, 11),
            LeakyReLU(inplace=True),
            # AttentionMap(8, keypoints_num),
            AttentionMap(1, 11),
            LeakyReLU(inplace=True),
            # AttentionMap(8, keypoints_num),
            AttentionMap(1, 11),
            LeakyReLU(inplace=True),
            Flatten(),
            Linear(11 * 11, 64, bias=True),
            LeakyReLU(inplace=True),
            Linear(64, 3, bias=True),
            Softmax(dim=1),
        ]
        self.kpm_model = Sequential(*self.kpm_model)

    def forward(self, kp, kpm):
        res = self.kpm_model(kpm)
        return res


if __name__ == '__main__':
    kp = torch.randn(size=(100, 1, 26, 3))
    kpm = torch.randn(size=(100, 1, 26, 26))
    kpl = KeyPointLearner(26)
    result = kpl(kp, kpm)
    print(result.size())
