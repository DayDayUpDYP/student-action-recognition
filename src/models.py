import torch

from torch.nn.modules import Module
from torch.nn import *


class KeyPointLearner(Module):
    """
    input 1 shape = (batch, keypoints_num, 3)
    input 1 shape = (batch, keypoints_num, keypoints_num)
    """

    def __init__(self, keypoints_num=26):
        super().__init__()
        self.kp_num = keypoints_num
        self.kp_model = [
            # 1x26x3 -> 8x26x1
            Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            # 8x26x1 -> 4x7x1
            Conv2d(8, 4, kernel_size=(2, 1), stride=(4, 1), padding=0),
            Flatten()
        ]
        self.kpm_model = [
            # 26 / 2 -> 13
            Conv2d(1, 8, kernel_size=2, stride=2, padding=0),
            # 13 / 2 -> 6
            Conv2d(8, 1, kernel_size=2, stride=2, padding=0),
            Flatten(),
            Linear(6 * 6, 7),
            ReLU(inplace=True)
        ]
        self.kp_model = Sequential(*self.kp_model)
        self.kpm_model = Sequential(*self.kpm_model)

        self.end_model = [
            Linear(31, 16, bias=True),
            ReLU(),
            # 3 class
            Linear(16, 3, bias=False),
            Softmax(dim=1)
        ]
        self.end_model = Sequential(*self.end_model)

    def forward(self, kp, kpm):
        kpv = self.kp_model(kp)
        kpmv = self.kpm_model(kpm)
        end = torch.cat([kpv, kpmv], dim=1)
        res = self.end_model(end)
        return res


if __name__ == '__main__':
    kp = torch.randn(size=(100, 1, 26, 3))
    kpm = torch.randn(size=(100, 1, 26, 26))
    kpl = KeyPointLearner(26)
    result = kpl(kp, kpm)
    print(result.size())
