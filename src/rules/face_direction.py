import json
from pathlib import Path

import cv2

import numpy as np


class FaceDirection:
    def __init__(self, json_path):
        with open(json_path, 'r') as fp:
            self.json_file = json.load(fp)

    def get_node(self, x1, y1, x2, y2, x3, y3, x4, y4):
        det = (y2 - y1) * (x3 - x4) - (y4 - y3) * (x1 - x2)
        c1 = x1 * y2 - x2 * y1
        c2 = x3 * y4 - x4 * y3
        if det == 0: return None
        x = (c1 * (x3 - x4) - c2 * (x1 - x2)) / det
        y = ((y2 - y1) * c2 - (y4 - y3) * c1) / det
        return x, y

    def run(self, img, img_name):
        for element in self.json_file:
            if element['image_id'] == img_name:
                return self.draw_direction(img, element['keypoints'])

    @staticmethod
    def draw_direction(img, keypoints):
        # mid_12 = int((keypoints[1 * 3] + keypoints[2 * 3]) / 2), int((keypoints[1 * 3 + 1] + keypoints[2 * 3 + 1]) / 2)
        # mid_12 = int(keypoints[0]), int(keypoints[1])
        # mid_34 = int((keypoints[3 * 3] + keypoints[4 * 3]) / 2), int((keypoints[4 * 3 + 1] + keypoints[4 * 3 + 1]) / 2)
        # mid_34 = int(mid_34[0]), int((mid_34[1] + keypoints[17]) / 2)

        x1, y1 = keypoints[17 * 3], keypoints[17 * 3 + 1]
        x2, y2 = keypoints[18 * 3], keypoints[18 * 3 + 1]
        A = y2 - y1
        B = -(x2 - x1)
        C = -x1 * (y2 - y1) + y2 * (x2 - x1)
        nose_kp = int(keypoints[0]), int(keypoints[1])

        # if A * A + B * B == 0:
        #     return img
        # x = int((B * B * nose_kp[0] - A * B * nose_kp[1] - A * C) / (A * A + B * B))
        # y = int((-A * B * nose_kp[0] + A * A * nose_kp[1] - B * C) / (A * A + B * B))

        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        img = cv2.arrowedLine(img, (x, y), nose_kp, color=(255, 0, 0), thickness=3, tipLength=0.3)
        return img


if __name__ == '__main__':
    fd = FaceDirection('../../test/resource/res2/alphapose-results.json')
    img_name = 'handsup_1_410.jpg'

    img_path = Path('../../test/resource/res2/vis')
    for file in img_path.rglob('*.jpg'):
        img = cv2.imread(str(file))
        img = fd.run(img, str(file.name))

        cv2.imshow('1', img)
        cv2.waitKey(0)
