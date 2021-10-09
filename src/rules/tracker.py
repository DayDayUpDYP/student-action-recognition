import json

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from conf import NAME_MAP, AT_LAYER, AT_MULTI
from models import KeyPointLearner, KeyPointLearnerGAT
from utils.ImageProcess import std_coordinate, ImageProcess
from utils.Visualizer import Visualizer
from vis_video import split_frame_json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def load_model(path, model: torch.nn.Module):
    model.load_state_dict(torch.load(path))


class Tracker:
    def __init__(self, video_path, json_file, learner):
        super(Tracker, self).__init__()
        with open(json_file, 'r') as fp:
            json_file = json.load(fp)
            self.frame_data = split_frame_json(json_file)

        self.learner = learner
        self.video_path = video_path

        self.y_data = []

    @staticmethod
    def predict(learner, element, keypoints_num):
        np_keypoints = np.array(element['keypoints'])
        np_keypoints = std_coordinate(1, 1, element['box'], np_keypoints, 26)[:keypoints_num, :]
        keypoints_m = ImageProcess.__get_matrix__(np_keypoints, keypoints_num)
        keypoints_m = transforms.ToTensor()(keypoints_m)
        keypoints_m = torch.unsqueeze(keypoints_m, dim=0).to(torch.float)

        keypoints_pm = np.array([np_keypoints[:, 2]])
        keypoints_pm = np.matmul(np.transpose(keypoints_pm), keypoints_pm)
        keypoints_pm = keypoints_pm / np.sum(keypoints_pm, axis=1)
        keypoints_pm = transforms.ToTensor()(keypoints_pm)
        keypoints_pm = torch.unsqueeze(keypoints_pm, dim=0).to(torch.float)

        keypoints_pm = keypoints_pm.to(device)
        keypoints_m = keypoints_m.to(device)
        return learner(keypoints_pm, keypoints_m).argmax(dim=1)

    def track(self):
        cap = cv2.VideoCapture(self.video_path)

        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        id_total = 0
        for person_list in self.frame_data:
            for person in person_list:
                id_total = max(id_total, person['idx'])

        self.y_data = [[0. for j in range(frame_total + 1)] for i in range(id_total + 1)]

        for frame_sub in range(frame_total):
            person_list = self.frame_data[frame_sub]
            for element in person_list:
                if element['score'] > 1.:
                    idx = int(element['idx'])
                    keypoints = element['keypoints']
                    np_points = np.array(keypoints).reshape((len(keypoints) // 3, 3))
                    self.y_data[idx][frame_sub] = np_points[17][1]


    def plot_track(self, idx):
        cap = cv2.VideoCapture(self.video_path)

        frame_cnt = 0

        x = [i for i in range(len(self.y_data[idx]))]

        while True:
            ret, frame = cap.read()
            if ret:
                for element in self.frame_data[frame_cnt]:
                    if element['score'] > 1. and element['idx'] == idx:
                        keypoints = element['keypoints']
                        np_points = np.array(keypoints).reshape((len(keypoints) // 3, 3))
                        frame = Visualizer.show_anchor(frame, element)
                        frame = cv2.circle(frame, (int(np_points[17][0]), int(np_points[17][1])), radius=2, thickness=6,
                                           color=(0, 0, 255))
                    # for i in range(len(self.y_data[idx])):
                plt.ion()
                plt.plot(x[:frame_cnt], self.y_data[idx][:frame_cnt])
                plt.show()
                plt.pause(0.005)
                plt.clf()
                cv2.imshow('tracker', frame)

                frame_cnt += 1
            else:
                break
        # plt.plot(x, self.y_data[idx])
        # plt.show()


if __name__ == '__main__':
    learner = KeyPointLearnerGAT(AT_LAYER, AT_MULTI).to(device)
    load_model('../../test/resource/model.pkl', learner)

    learner.eval()

    tracker = Tracker(video_path='../../test/resource/video/src_videos/2/demo_all_Trim_2.avi',
                      json_file='../../test/resource/video/src_videos/2/alphapose-results-track-2.json',
                      learner=learner)
    tracker.track()

    tracker.plot_track(36)
