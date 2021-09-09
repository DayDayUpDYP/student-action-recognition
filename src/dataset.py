import torch
import numpy as np

from conf import NAME_MAP
from torch.utils.data import Dataset
from utils.ImageProcess import ImageProcess
from torchvision.transforms import transforms


class KeyPointDataset(Dataset):
    def __init__(self, base_path, keypoint_num=26, std_h=5, std_w=5):
        self.transform_1 = transforms.Compose([
            transforms.ToTensor()
        ])

        self.transform_2 = transforms.Compose([
            transforms.ToTensor()
        ])

        self.base_path = base_path
        self.ip = ImageProcess(base_path)
        self.img_data = list(self.ip.get_data(keypoint_num, std_h, std_w))
        self.data_len = len(self.img_data)

    def __getitem__(self, index):
        img_name, keypoints, dism = self.img_data[index]
        _index = img_name.find('_')
        img_name = img_name[:_index]
        keypoints = self.transform_1(keypoints)
        dism = self.transform_2(dism)
        return NAME_MAP[img_name], keypoints, dism

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    kpd = KeyPointDataset('../test/resource/res', 26, 5, 5)
    print(kpd.__getitem__(5))
