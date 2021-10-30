import json
from queue import Queue

import numpy as np
from matplotlib import pyplot as plt
import cv2

from rules.face_direction import FaceDirection
from utils.Visualizer import Visualizer


def split_frame_json(json_file):
    res = []
    for elem in json_file:
        image_id = elem['image_id'][:elem['image_id'].find('.')]
        frame_sub = eval(image_id)
        if frame_sub >= len(res):
            res.append([elem])
        else:
            res[frame_sub].append(elem)
    return res


def check_frame(sub, data):
    for elem in data:
        image_id = elem[0]['image_id'][:elem[0]['image_id'].find('.')]
        frame_sub = eval(image_id)
        if sub == frame_sub:
            return True
    return False


def get_bpf(frame, element):
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    x, y, w, h = element['box']
    x, y, w, h = int(x), int(y), int(w), int(h)
    H = h
    R = h / w
    M = cv2.moments(temp_frame[y:y + h, x:x + w])
    xc, yc = M['m10'] * 1. / M['m00'], M['m01'] * 1. / M['m00']
    theta = np.arctan(
        2 * (M['m11'] / M['m00'] - xc * yc) /
        (M['m20'] / M['m00'] - xc * xc - (M['m02'] / M['m00'] - yc * yc))) * 0.5 * 180 / np.pi
    return x, y, x + w, y + h, H, R, xc, yc, theta


buffer = []


def paint_rule(frame, frame_sub, frame_data, learner, scan_cnt, keypoints_num):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]

    global buffer

    for element in person_list:
        if element['score'] > 1.:
            np_keypoints = np.array(element['keypoints'])

            cur_data = get_bpf(frame, element)

            (box_x, box_y, box_h, box_w) = element['box']
            frame = Visualizer.show_line(frame, element)
            frame = Visualizer.show_anchor(frame, element)
            frame = FaceDirection.draw_direction(frame, element['keypoints'])

            frame = Visualizer.show_label(frame, int(box_x), int(box_y),
                                          f'{cur_data[4]:.0f}, '
                                          f'{cur_data[5]:.0f}, '
                                          f'({cur_data[6]:.0f}, {cur_data[7]:.0f}), '
                                          f'{cur_data[8]:.0f}',
                                          (0, 0, 255))

            # GAP = 1
            # if len(buffer) == GAP:
            #     data = np.zeros(9)
            #
            #     for d in buffer:
            #         data = data + np.array(list(d))
            #
            #     data /= GAP
            #
            #     frame = Visualizer.show_label(frame, int(box_x), int(box_y),
            #                                   f'{data[4].item():.0f}, '
            #                                   f'{data[5].item():.0f}, '
            #                                   f'({data[6].item():.0f}, {data[7].item():.0f}), '
            #                                   f'{data[8].item():.0f}',
            #                                   (0, 0, 255))
            #     buffer.pop()
            #     buffer.append(cur_data)
            # else:
            #     buffer.append(np.array(list(cur_data)))

    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture("../../test/resource/mv/4.mp4")

    with open("../../test/resource/mv/alphapose-4.json", 'r') as fp:
        json_file = json.load(fp)
    frame_data = split_frame_json(json_file)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_cnt = 0
    print('start...')

    while True:
        ret, frame = cap.read()
        if ret:
            frame = paint_rule(frame, frame_cnt, frame_data, None, -1, keypoints_num=26)

            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_cnt += 1
        else:
            break
    cap.release()

    print('finished...')
