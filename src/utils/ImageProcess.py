import json
from pathlib import Path

import cv2
import numpy as np


def find_person(img_data):
    index = 0
    max_area = 0
    for i, person in enumerate(img_data):
        area = person['box'][2] * person['box'][3]
        if area > max_area and person['score'] > 1.:
            max_area = area
            index = i
    # print(index, img_data)
    if len(img_data) == 0:
        return None
    return img_data[index]


def split_person_dict(img_name, dict_all):
    # print(img_name)
    return find_person([meta for meta in dict_all if meta['image_id'] == img_name])


def std_coordinate(std_h, std_w, box, keypoints):
    x, y, h, w = box
    np_points = np.array(keypoints).reshape((26, 3))
    # np_points = np.array(keypoints).reshape((136, 3))
    np_points[:, 0] = (np_points[:, 0] - x) / h * std_h
    np_points[:, 1] = (np_points[:, 1] - y) / w * std_w
    np_points[np_points < 0] = 0
    return np_points


class ImageProcess:
    def __init__(self, in_path):
        self.in_path = Path(in_path)
        # self.out_path = Path(out_path)
        with open(self.in_path / 'alphapose-results.json', 'r') as fp:
            self.json_file = json.load(fp)

    def __get_keypoints__(self, img_path, keypoints_num, std_h, std_w):
        person_dict = split_person_dict(img_path.name, self.json_file)
        if person_dict is None:
            return None
        keypoints_xyp = std_coordinate(std_h, std_w, person_dict['box'], person_dict['keypoints'])
        return keypoints_xyp[:keypoints_num]

    @staticmethod
    def __get_matrix__(keypoints, num):
        result = np.zeros(shape=(num, num))
        for i, line in enumerate(keypoints):
            l = np.array([line[:2]])
            temp = (np.repeat(l, num, -2) - keypoints[:, :2]) ** 2
            temp = np.sqrt(np.sum(temp, axis=1))
            # temp = temp / np.sum(temp / (1 + np.exp(keypoints[:, 2])))
            # temp = temp / keypoints[:, 2]
            # print(temp.size())
            # temp = (temp - np.mean(temp)) / np.std(temp)
            result[i] = temp
        return result

    def get_keypoints(self, keypoints_num, std_h, std_w):
        for img_path in self.in_path.rglob('*.jpg'):
            keypoints = self.__get_keypoints__(img_path, keypoints_num, std_h, std_w)
            if keypoints is None:
                continue
            yield img_path.stem, keypoints

    def get_data(self, keypoints_num, std_h, std_w):
        for img_name, keypoints in self.get_keypoints(keypoints_num, std_h, std_w):
            yield img_name, keypoints, ImageProcess.__get_matrix__(keypoints, keypoints_num)


if __name__ == '__main__':
    ip = ImageProcess(in_path='../../test/resource/res')
    for name, m in ip.get_data(26, 5, 5):
        print(name, m)
        input()
