import json
from pathlib import Path

import cv2
import numpy as np


def find_person(img_data):
    index = 0
    max_area = 0
    for i, person in enumerate(img_data):
        area = person['box'][2] * person['box'][3]
        if area > max_area:
            max_area = area
            index = i
    return img_data[index]


def split_person_dict(img_name, dict_all):
    return find_person([meta for meta in dict_all if meta['image_id'] == img_name])


def std_coordinate(std_h, std_w, box, keypoints):
    x, y, h, w = box
    np_points = np.array(keypoints).reshape((136, 3))
    np_points[:, 0] = (np_points[:, 0] - x) / h * std_h
    np_points[:, 1] = (np_points[:, 1] - y) / w * std_w
    return np_points


class ImageProcess:
    def __init__(self, in_path, out_path):
        self.in_path = Path(in_path)
        self.out_path = Path(out_path)
        with open(self.in_path / 'alphapose-results.json', 'r') as fp:
            self.json_file = json.load(fp)

    def __get_keypoints__(self, img_path, keypoints_num, std_h, std_w):
        person_dict = split_person_dict(img_path.name, self.json_file)
        keypoints_xyp = std_coordinate(std_h, std_w, person_dict['box'], person_dict['keypoints'])
        return keypoints_xyp[:keypoints_num]

    @staticmethod
    def __get_matrix__(keypoints, num):
        result = np.zeros(shape=(num, num))
        for i, line in enumerate(keypoints):
            l = np.array([line[:2]])
            temp = (np.repeat(l, num, -2) - keypoints[:, :2]) ** 2
            temp = np.sqrt(np.sum(temp, axis=1))
            result[i] = temp
        return result

    def get_keypoints(self, keypoints_num, std_h, std_w):
        for img_path in self.in_path.rglob('*.jpg'):
            keypoints = self.__get_keypoints__(img_path, keypoints_num, std_h, std_w)
            yield keypoints

    def get_dism(self, keypoints_num, std_h, std_w):
        for keypoints in self.get_keypoints(keypoints_num, std_h, std_w):
            yield ImageProcess.__get_matrix__(keypoints, keypoints_num)


if __name__ == '__main__':
    ip = ImageProcess(in_path='../../test/resource/res', out_path='')
    for m in ip.get_dism(26, 5, 5):
        print(m)
