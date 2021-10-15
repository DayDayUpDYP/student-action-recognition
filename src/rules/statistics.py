import cv2
import numpy as np
from matplotlib import pyplot as plt

import json

from utils.Visualizer import Visualizer


class Statistic:
    def __init__(self):
        self.all_active_data = []

    def find_person_index(self, box_x, box_y, box_h, box_w):
        _min = 700
        _min_sub = -1

        for sub, ((x, y, h, w), data) in enumerate(self.all_active_data):
            if (abs((box_x - x) / box_h)) < 0.3 and (abs((box_y - y) / box_w)) < 0.3:
                temp = abs(box_x - x) + abs(box_y - y)
                if temp < _min:
                    _min = temp
                    _min_sub = sub
        return _min_sub

    def update_person(self, box_x, box_y, box_h, box_w, behave, frame):
        index = self.find_person_index(box_x, box_y, box_h, box_w)
        if index == -1:
            self.all_active_data.append(([box_x, box_y, box_h, box_w], []))
            for j in range(frame + 1):
                self.all_active_data[-1][1].append(-1)
        else:
            det = frame - len(self.all_active_data[index][1])
            if det > 0:
                for j in range(det):
                    self.all_active_data[index][1].append(-1)
                self.all_active_data[index][1].append(behave)
            elif det == 0:
                self.all_active_data[index][1].append(behave)

    def show_data(self, index):
        print(f'total person number = {len(self.all_active_data)}')

        plt.scatter(range(len(self.all_active_data[index][1])), self.all_active_data[index][1])
        plt.show()

    def load_data(self, path):
        with open(path, 'r') as fp:
            self.all_active_data = json.load(fp)
            print('json load finished.')

    def save_data(self, path):
        with open(path, 'w') as fp:
            s = json.dumps(self.all_active_data)
            fp.write(s)
            print('write to json finished.')

    def paint_frame(self, frame, frame_sub, index):
        person_data = self.all_active_data[index]

        if frame_sub < len(person_data[1]):
            bev = person_data[1][frame_sub]
            if bev != -1:
                (x, y, h, w) = person_data[0]
                color = (0, 128, 128)
                k = 'unknown'
                if bev == 1:
                    color = (255, 0, 0)
                    k = 'stand'
                elif bev == 2:
                    color = (0, 255, 0)
                    k = 'handsup'
                elif bev == 3:
                    color = (0, 0, 255)
                    k = 'sit'

                frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + h), int(y + w)), (0, 255, 0), 2)
                # print(k, color)
                frame = Visualizer.show_label(frame, int(x), int(y), k, color)

        return frame

    def show_video_by_index(self, in_path, out_path, index):
        cap = cv2.VideoCapture(in_path)  # 读取视频文件
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, 30, (frame_w, frame_h))
        frame_cnt = 0

        while True:
            ret, frame = cap.read()
            if ret:
                frame = self.paint_frame(frame, frame_cnt, index)

                out.write(frame)
                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_cnt += 1
            else:
                break
        cap.release()


if __name__ == '__main__':
    log = Statistic()
    log.load_data('../../test/resource/logging.json')

    index = 2

    log.show_data(index)
    log.show_video_by_index(
        in_path='../../test/resource/video/src_videos/classroom/classroom_Trim.avi',
        out_path='../../test/resource/video/tracker.avi',
        index=index
    )
