import torch

from torch.nn.modules import Module
from torch.nn import *
import numpy as np
from gat.GAT import GAT


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
        # torch.nn.init.normal_(self.W)

    def forward(self, x):
        # print(x.size(), self.attention.size())
        # res = torch.mm(self.D, self.A)
        # res = torch.mm(res, self.D)
        # res = self.A * x
        res = torch.matmul(x, self.W)
        # sf = Softmax(dim=3)
        # res = sf(res)
        # res = res.permute(0, 1, 3, 2)
        # res = torch.matmul(x, res)
        # res = ax * x
        # res = torch.matmul(x, self.W)
        return res


class IntensifyLayer(Module):
    def __init__(self, decay_rate, N):
        super(IntensifyLayer, self).__init__()
        self.decay_rate = decay_rate
        self.N = N

    def forward(self, x):
        return x + self.decay_rate * (x - 1. / self.N)


class KeyPointLearner(Module):
    """
    input 1 shape = (batch, keypoints_num, 3)
    input 1 shape = (batch, keypoints_num, keypoints_num)
    """

    def __init__(self, keypoints_num=26, intensify_num=1.):
        super().__init__()
        self.intensify_num = intensify_num

        self.attention = AttentionLayer(1, keypoints_num)
        self.softmax = Softmax(2)
        self.linear_intensify = [
            IntensifyLayer(decay_rate=intensify_num, N=26),
        ]

        self.linear_intensify = Sequential(*self.linear_intensify)

        self.nonlinear_intensify = [
            Flatten(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 26),
            LeakyReLU(),
            Softmax(1),
        ]

        self.nonlinear_intensify_W = Parameter(torch.randn(size=(64, 26)), requires_grad=True)

        self.nonlinear_intensify = Sequential(*self.nonlinear_intensify)

        self.gcn = GCNLayer(1, keypoints_num)

        self.kpm_model = [

            self.gcn,
            BatchNorm2d(1),
            # GCNLayer(1, keypoints_num),
            ReLU(),

            self.gcn,
            BatchNorm2d(1),
            # GCNLayer(1, keypoints_num),
            ReLU(),

            self.gcn,
            BatchNorm2d(1),
            # GCNLayer(1, keypoints_num),
            ReLU(),
        ]

        self.end_model = [
            # BatchNorm2d(1),
            Flatten(),
            Linear(keypoints_num, 13),
            ReLU(),
            Linear(13, 3, bias=True),
            Softmax(dim=1),
        ]

        self.end_model = Sequential(*self.end_model)

        self.kpm_model = Sequential(*self.kpm_model)

    def forward(self, kp, kpm):
        # linear intensify function
        # kp = self.linear_intensify(kp)
        # kp = self.softmax(kp)

        # nonlinear intensify function
        # for i in range(3):
        kpp = torch.matmul(self.nonlinear_intensify_W, kp)
        gama = self.nonlinear_intensify(kpp)
        gama = torch.reshape(gama, (gama.shape[0], 1, gama.shape[1], 1))
        kp = kp + self.intensify_num * (kp - gama)

        res = self.kpm_model(kpm)
        res = torch.matmul(res.transpose(-2, -1), kp)

        res = self.end_model(res)
        return res


class KeyPointLearnerGAT(Module):
    def __init__(self, gat_layer_num, multi_num):
        super(KeyPointLearnerGAT, self).__init__()
        self.gats = ModuleList()
        for i in range(gat_layer_num):
            self.gats.append(GAT(26, 26, multi_num))

        self.mlp = [
            Flatten(),
            Linear(26 * 26, 256),
            LeakyReLU(negative_slope=0.2),
            Linear(256, 3),
        ]
        self.mlp = Sequential(*self.mlp)

    def forward(self, _, x):
        for layer in self.gats:
            x = layer(x)
        return self.mlp(x)


if __name__ == '__main__':
    # kp = torch.randn(size=(100, 1, 26, 3))
    kpm = torch.randn(size=(100, 1, 26, 26))
    kpl = KeyPointLearnerGAT(1, 3)
    result = kpl(kpm)
    print(result.size())
