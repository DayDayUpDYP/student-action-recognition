import torch
import numpy as np

from torch.utils.data import Dataset
from utils.ImageProcess import ImageProcess


class KeyPointDataset(Dataset):
    def __init__(self, base_path, keypoint_num, std_h, std_w):
        self.base_path = base_path
        self.ip = ImageProcess(base_path)
        self.img_data = list(self.ip.get_data(keypoint_num, std_h, std_w))
        self.data_len = len(self.img_data)

    def __getitem__(self, index):
        return self.img_data[index]

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    kpd = KeyPointDataset('../test/resource/res', 26, 5, 5)
    print(kpd.__getitem__(5))
