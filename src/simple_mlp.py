import torch

from torch.nn import *


class SimpleMLP(Module):
    def __init__(self, keypoints_num, dis_num, angle_num):
        super(SimpleMLP, self).__init__()
        self.mlp = [
            Linear(keypoints_num * 2 + dis_num + angle_num, 128),
            Sigmoid(),
            Linear(128, 64),
            Sigmoid(),
            Linear(64, 16),
            Sigmoid(),
            Linear(16, 3),
            Softmax(dim=1)
        ]
        self.mlp = Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)
