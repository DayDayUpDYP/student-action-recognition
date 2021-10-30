import json
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np

from utils.ImageProcess import ImageProcess
from utils.Visualizer import Visualizer


def dis_coord(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dis_keypoints(sub1, sub2, keypoints):
    sub1_x = keypoints[sub1 * 3]
    sub1_y = keypoints[sub1 * 3 + 1]
    sub2_x = keypoints[sub2 * 3]
    sub2_y = keypoints[sub2 * 3 + 1]
    return dis_coord(sub1_x, sub1_y, sub2_x, sub2_y)


def unit_vector(sub1, sub2, keypoints):
    (x1, y1) = keypoints[sub1 * 3] - keypoints[sub2 * 3], keypoints[sub1 * 3 + 1] - keypoints[sub2 * 3 + 1]
    x1, y1 = x1 / (np.sqrt(x1 ** 2 + y1 ** 2) + 0.000003), y1 / (np.sqrt(x1 ** 2 + y1 ** 2) + 0.000003)
    return (x1, y1)


def inner_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def is_sit(keypoints):
    cos1 = \
        (dis_keypoints(10, 8, keypoints) ** 2 +
         dis_keypoints(8, 6, keypoints) ** 2 -
         dis_keypoints(10, 6, keypoints) ** 2) / (
                2 * dis_keypoints(8, 10, keypoints) * dis_keypoints(8, 6, keypoints) + 0.0003)
    cos2 = \
        (dis_keypoints(6, 17, keypoints) ** 2 +
         dis_keypoints(6, 8, keypoints) ** 2 -
         dis_keypoints(8, 17, keypoints) ** 2) / (
                2 * dis_keypoints(6, 8, keypoints) * dis_keypoints(6, 17, keypoints) + 0.0003)

    return keypoints[8 * 3 + 1] >= keypoints[6 * 3 + 1] and keypoints[7 * 3 + 1] >= keypoints[5 * 3 + 1]


def is_handsup(keypoints):
    v = (
        keypoints[10 * 3] - keypoints[8 * 3],
        keypoints[10 * 3 + 1] - keypoints[8 * 3 + 1]
    )

    val1 = -v[1] / (dis_keypoints(10, 8, keypoints) + 0.0003)

    v = (
        keypoints[9 * 3] - keypoints[7 * 3],
        keypoints[9 * 3 + 1] - keypoints[7 * 3 + 1]
    )

    val2 = -v[1] / (dis_keypoints(9, 7, keypoints) + 0.0003)

    dis_68 = dis_keypoints(6, 8, keypoints)
    dis_810 = dis_keypoints(8, 10, keypoints)

    dis_57 = dis_keypoints(5, 7, keypoints)
    dis_79 = dis_keypoints(7, 9, keypoints)

    return (0.9 < val2 <= 1 and dis_79 >= 2 * dis_57 / 3) or (0.9 < val1 <= 1 and dis_810 >= 2 * dis_68 / 3)


def is_stand(keypoints):
    v0 = unit_vector(18, 19, keypoints)
    v1 = unit_vector(12, 14, keypoints)

    v2 = unit_vector(11, 13, keypoints)
    inp_1 = inner_product(v0, v1)
    inp_2 = inner_product(v0, v2)

    ul1 = unit_vector(12, 14, keypoints)
    ul2 = unit_vector(14, 16, keypoints)

    ur1 = unit_vector(11, 13, keypoints)
    ur2 = unit_vector(13, 15, keypoints)

    flag = True
    if keypoints[16 * 3 + 2] >= 0.8 and keypoints[15 * 3 + 2] >= 0.8:
        # if abs(ul1[0] * ul2[0] + ul1[1] * ul2[1] - 1) <= 0.2 and abs(ur1[0] * ur2[0] + ur1[1] * ur2[1] - 1) <= 0.2:
        #     flag = True
        # else:
        #     flag = False
        dis_12_16 = dis_keypoints(12, 25, keypoints)
        dis_19_18 = dis_keypoints(19, 24, keypoints)
        if abs(dis_12_16 / (dis_12_16 + dis_19_18) - 0.5) <= 0.2:
            flag = True
        else:
            flag = False

    # and abs(dis_12_14 * dis_13_15 - dis_14_16 * dis_11_13) <= 0.5
    return inp_1 > 0.9 and inp_2 > 0.9 and flag


def is_look_aside(keypoints):
    x1, y1 = keypoints[17 * 3], keypoints[17 * 3 + 1]
    x2, y2 = keypoints[18 * 3], keypoints[18 * 3 + 1]
    A = y2 - y1
    B = -(x2 - x1)
    C = -x1 * (y2 - y1) + y2 * (x2 - x1)
    nose_kp = int(keypoints[0]), int(keypoints[1])

    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    v = nose_kp[0] - x, y - nose_kp[1]
    inp = v[0] / np.sqrt(v[0] ** 2 + v[1] ** 2 + 0.00003)

    return abs(inp) >= 0.8


def validation(dir_path):
    with open(dir_path / 'alphapose-results.json', 'r') as fp:
        json_file = json.load(fp)
    total = 0.
    cor_total = 0

    stand_total = 0
    cor_stand = 0

    sit_total = 0
    cor_sit = 0

    handsup_total = 0
    cor_handsup = 0

    for file in dir_path.rglob('*.jpg'):
        file_name = file.name
        label_name = file.name[:file.name.find('_')]
        for item in json_file:
            if item['image_id'] == file_name and item['box'][2] * item['box'][3] >= 200000:
                keypoints = item['keypoints']

                k = 'unknown'
                if is_handsup(keypoints):
                    k = 'handsup'
                elif is_stand(keypoints):
                    k = 'stand'
                elif is_sit(keypoints):
                    k = 'sit'
                total += 1

                if label_name == 'stand':
                    stand_total += 1
                    if k == 'stand':
                        cor_stand += 1

                if label_name == 'sit':
                    sit_total += 1
                    if k == 'sit':
                        cor_sit += 1

                if label_name == 'handsup':
                    handsup_total += 1
                    if k == 'handsup':
                        cor_handsup += 1

                if k == label_name:
                    cor_total += 1

                if k != label_name and label_name == 'sit':
                    v0 = unit_vector(18, 19, keypoints)
                    v1 = unit_vector(12, 14, keypoints)

                    v2 = unit_vector(11, 13, keypoints)
                    inp_1 = inner_product(v0, v1)
                    inp_2 = inner_product(v0, v2)

                    # print(file_name, k, label_name)
                    # print(inp_1, inp_2)
                    # print(item['score'], item['box'][2] * item['box'][3])
                    # I = cv2.imread(str(file))
                    # I = Visualizer.show_line(I, item)
                    # I = Visualizer.show_anchor(I, item)
                    # cv2.namedWindow('1', 0)
                    # cv2.imshow('1', I)
                    # cv2.waitKey(0)

    print(f'total correct rate = {cor_total * 1. / total * 100}%')
    print(f'stand correct rate = {cor_stand * 1. / stand_total * 100}%')
    print(f'sit correct rate = {cor_sit * 1. / sit_total * 100}%')
    print(f'handsup correct rate = {cor_handsup * 1. / handsup_total * 100}%')


if __name__ == '__main__':
    validation(Path('../../test/resource/output'))
