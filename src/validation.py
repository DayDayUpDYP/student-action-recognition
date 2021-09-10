import json
from pathlib import Path

import torch
from torchvision.transforms import transforms
import numpy as np

from conf import *
from models import KeyPointLearner
from main import load_model
from utils.ImageProcess import *
from utils.Visualizer import *


class Validation:
    def __init__(self, learner, path):
        self.path = Path(path)
        with open(self.path / 'alphapose-results.json', 'r') as fp:
            self.json_file = json.load(fp)
        self.learner = learner

    def inference(self, cnt):
        scan_cnt = 0
        for img_file in self.path.rglob('*jpg'):
            origin = cv2.imread(str(img_file))
            for i, element in enumerate(self.json_file):
                if element['image_id'] == img_file.name and element['score'] > 1.:
                    np_keypoints = np.array(element['keypoints'])
                    np_keypoints = std_coordinate(1, 1, element['box'], np_keypoints)[:11, :]
                    keypoints_m = ImageProcess.__get_matrix__(np_keypoints, 11)
                    keypoints = transforms.ToTensor()(np_keypoints)
                    keypoints_m = transforms.ToTensor()(keypoints_m)

                    keypoints = torch.unsqueeze(keypoints, dim=0).to(torch.float)
                    keypoints_m = torch.unsqueeze(keypoints_m, dim=0).to(torch.float)

                    pred = learner(keypoints, keypoints_m).argmax(dim=1)
                    for k, v in NAME_MAP.items():
                        if v == pred.item():
                            img = Visualizer.show_anchor(origin.copy(), element)
                            img = Visualizer.show_keypoint(img, element)
                            img = Visualizer.show_line(img, element)
                            clr = (0, 0, 255)
                            if k == 'stand':
                                clr = (255, 0, 0)
                            elif k == 'handsup':
                                clr = (0, 255, 0)
                            img = Visualizer.show_label(img, element, k, clr)
                            # print(keypoints_m)
                            cv2.imshow(img_file.name, img)
                            cv2.waitKey(0)
                    scan_cnt += 1
                    if scan_cnt == cnt:
                        break


    def inference_video(self, cnt):
        pass


if __name__ == '__main__':
    learner = KeyPointLearner()
    load_model('../test/resource/model.pkl', learner)
    val = Validation(learner, path='../test/resource/scene')
    val.inference(-1)
