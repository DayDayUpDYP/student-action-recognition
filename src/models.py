import torch

from torch.nn.modules import Module
from torch.nn import *
import numpy as np


class AttentionLayer(Module):
    def __init__(self, channels, in_dim):
        super().__init__()
        self.atten = Parameter(torch.Tensor(np.diag([1] * in_dim)), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.atten)


class GCNLayer(Module):
    def __init__(self, channels, in_dim):
        super().__init__()
        adj = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
        ]
        self.A = Parameter(torch.Tensor(adj), requires_grad=False)
        self.W = Parameter(torch.randn(size=(in_dim, in_dim)), requires_grad=True)

    def forward(self, x):
        # print(x.size(), self.attention.size())
        # res = torch.mm(self.D, self.A)
        # res = torch.mm(res, self.D)
        # res = self.A * x
        res = torch.matmul(x, self.W)
        sf = Softmax(dim=3)
        res = sf(res)

        res = torch.matmul(res, x)
        # res = ax * x
        # res = torch.matmul(x, self.W)
        return res


class KeyPointLearner(Module):
    """
    input 1 shape = (batch, keypoints_num, 3)
    input 1 shape = (batch, keypoints_num, keypoints_num)
    """

    def __init__(self, keypoints_num=26):
        super().__init__()
        self.attention = AttentionLayer(1, keypoints_num)

        self.kpm_model = [
            # Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            # AttentionLayer(1, keypoints_num),
            LayerNorm(26),
            GCNLayer(1, keypoints_num),
            LeakyReLU(),
            # AttentionMap(8, keypoints_num),
            # AttentionLayer(1, keypoints_num),
            GCNLayer(1, keypoints_num),
            LeakyReLU(),
            # AttentionMap(8, keypoints_num),
            # AttentionLayer(1, keypoints_num),
            GCNLayer(1, keypoints_num),
            LeakyReLU(),
        ]

        self.end_model = [
            # BatchNorm2d(1),
            Flatten(),
            Linear(keypoints_num, 13),
            LeakyReLU(),
            Dropout(0.3),
            Linear(13, 3, bias=True),
            Softmax(dim=1),
        ]

        self.prob_atten = Parameter(torch.randn(size=(keypoints_num, keypoints_num)), requires_grad=True)

        self.end_model = Sequential(*self.end_model)

        self.kpm_model = Sequential(*self.kpm_model)

    def forward(self, kp, kpm):
        # kpm = self.mlp(kpm)
        # shape = kpm.shape
        # kpm = torch.reshape(kpm, (shape[0], 1, 26, 26))

        # res = self.kpm_model(kpm)
        res = self.kpm_model(kpm)

        kp_permute = torch.matmul(kp, kp.permute(0, 1, 3, 2))
        kp_atten = self.prob_atten * kp_permute
        sf = Softmax(2)
        kp_atten = sf(kp_atten)

        kp = torch.matmul(kp_atten, kp)

        res = torch.matmul(res, kp)

        # res = torch.sum(res, dim=1)
        res = self.end_model(res)
        return res


if __name__ == '__main__':
    kp = torch.randn(size=(100, 1, 26, 3))
    kpm = torch.randn(size=(100, 1, 26, 26))
    kpl = KeyPointLearner(26)
    result = kpl(kp, kpm)
    print(result.size())
