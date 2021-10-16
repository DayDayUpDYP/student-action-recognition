import json
from pathlib import Path

import numpy as np


def dis_coord(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dis_keypoints(sub1, sub2, keypoints):
    sub1_x = keypoints[sub1 * 3]
    sub1_y = keypoints[sub1 * 3 + 1]
    sub2_x = keypoints[sub2 * 3]
    sub2_y = keypoints[sub2 * 3 + 1]
    return dis_coord(sub1_x, sub1_y, sub2_x, sub2_y)


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

    return (0 <= cos1 <= 1) and (-1 <= cos2 <= 0)


def unit_vector(sub1, sub2, keypoints):
    (x1, y1) = keypoints[sub1 * 3] - keypoints[sub2 * 3], keypoints[sub1 * 3 + 1] - keypoints[sub2 * 3 + 1]
    x1, y1 = x1 / (np.sqrt(x1 ** 2 + y1 ** 2) + 0.000003), y1 / (np.sqrt(x1 ** 2 + y1 ** 2) + 0.000003)
    return (x1, y1)


def inner_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def is_handsup(keypoints):
    v0 = unit_vector(6, 5, keypoints)
    v1 = unit_vector(10, 8, keypoints)
    v2 = unit_vector(9, 7, keypoints)
    inp_1 = inner_product(v0, v1)
    inp_2 = inner_product(v0, v2)
    return is_sit(keypoints) and (abs(inp_1) < 0.4 or abs(inp_2) < 0.4)


def is_stand(keypoints):
    v0 = unit_vector(18, 19, keypoints)
    v1 = unit_vector(12, 14, keypoints)

    v2 = unit_vector(11, 13, keypoints)
    inp_1 = inner_product(v0, v1)
    inp_2 = inner_product(v0, v2)

    return inp_1 > 0.99 and inp_2 > 0.99


def validation(dir_path):
    with open(dir_path / 'alphapose-results.json', 'r') as fp:
        json_file = json.load(fp)
    total = 0.
    correction = 0

    for file in dir_path.rglob('*.jpg'):
        file_name = file.name
        label_name = file.name[:file.name.find('_')]
        for item in json_file:
            if item['image_id'] == file_name:
                np_keypoints = item['keypoints']
                k = 'unknown'
                if is_stand(np_keypoints):
                    k = 'stand'
                elif is_handsup(np_keypoints):
                    k = 'handsup'
                elif is_sit(np_keypoints):
                    k = 'sit'
                total += 1
                if k == label_name:
                    correction += 1

    print(f'correct rate = {correction * 1. / total * 100}%')


if __name__ == '__main__':
    validation(Path('../../test/resource/val'))
