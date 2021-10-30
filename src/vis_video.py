import torch
import json
import cv2
import numpy as np
from torchvision.transforms import transforms

from rules.face_direction import FaceDirection
from rules.statistics import Statistic
from utils.ImageProcess import std_coordinate
from utils.ImageProcess import ImageProcess
from models import KeyPointLearner, KeyPointLearnerGAT
from conf import *
from utils.Visualizer import Visualizer

from rules.simple_rules import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

logger = Statistic()


def load_model(path, model: torch.nn.Module):
    model.load_state_dict(torch.load(path))


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


def keypoints_moment(keypoints, p, q):
    res = 0.
    for i in range(0, len(keypoints), 3):
        res += np.power(keypoints[i], p) * np.power(keypoints[i + 1], q) * keypoints[i + 2]
    return res


def get_part_moment(s, e, element):
    keypoints = element['keypoints'][s * 3:e * 3]
    xc, yc = int(keypoints_moment(keypoints, 1, 0) / keypoints_moment(keypoints, 0, 0)) \
        , int(keypoints_moment(keypoints, 0, 1) / keypoints_moment(keypoints, 0, 0))
    return xc, yc


def get_bpf(frame, element):
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    x, y, w, h = element['box']
    x, y, w, h = int(x), int(y), int(w), int(h)
    H = h
    R = h / w

    # M = cv2.moments(temp_frame[y:y + h, x:x + w])
    # xc, yc = M['m10'] * 1. / M['m00'], M['m01'] * 1. / M['m00']
    # theta = np.arctan(
    #     2 * (M['m11'] / M['m00'] - xc * yc) /
    #     (M['m20'] / M['m00'] - xc * xc - (M['m02'] / M['m00'] - yc * yc))) * 0.5 * 180 / np.pi
    xc, yc = get_part_moment(0, 19, element)

    return x, y, x + w, y + h, H, R, xc, yc


def fix_keypoints(person_list):
    select_keypoints = list(range(11)) + [17, 18]

    for person in person_list:
        x_min, y_min = person['keypoints'][0], person['keypoints'][1]
        x_max, y_max = person['keypoints'][0], person['keypoints'][1]

        for i in range(0, 26 * 3, 3):
            if i // 3 in select_keypoints:
                x_min = min(x_min, person['keypoints'][i])
                x_max = max(x_max, person['keypoints'][i])
                y_min = min(y_min, person['keypoints'][i + 1])
                y_max = max(y_max, person['keypoints'][i + 1])

        person['box'] = [x_min, y_min, x_max - x_min, y_max - y_min]
    return person_list


def fix_symmetric(person_list):
    def fix_points(a, b, keypoints):
        dis1 = abs(keypoints[a * 3] - keypoints[17 * 3]) + \
               abs(keypoints[a * 3 + 1] - keypoints[17 * 3 + 1])
        dis2 = abs(keypoints[b * 3] - keypoints[17 * 3]) + \
               abs(keypoints[b * 3 + 1] - keypoints[17 * 3 + 1])

        if dis1 / (dis2 + 0.0001) > 2:
            keypoints[a * 3] = 2 * keypoints[17 * 3] - keypoints[b * 3]
            keypoints[a * 3 + 1] = keypoints[b * 3 + 1]

        if dis2 / (dis1 + 0.0001) > 2:
            keypoints[b * 3] = 2 * keypoints[17 * 3] - keypoints[a * 3]
            keypoints[b * 3 + 1] = keypoints[a * 3 + 1]

    def fix_single_points(a, b, c, keypoints):
        dis1 = abs(keypoints[a * 3] - keypoints[b * 3]) + \
               abs(keypoints[a * 3 + 1] - keypoints[b * 3 + 1])
        dis2 = abs(keypoints[a * 3] - keypoints[c * 3]) + \
               abs(keypoints[a * 3 + 1] - keypoints[c * 3 + 1])

        if dis1 / (dis2 + 0.0001) > 2 or dis2 / (dis1 + 0.0001) > 2:
            keypoints[a * 3] = (keypoints[b * 3] + keypoints[c * 3]) / 2.

    for person in person_list:
        keypoints = person['keypoints']
        fix_points(2, 1, keypoints)
        fix_points(4, 3, keypoints)
        fix_points(6, 5, keypoints)
        fix_points(8, 7, keypoints)
        fix_points(10, 9, keypoints)

        fix_single_points(0, 4, 3, keypoints)
        fix_single_points(18, 6, 5, keypoints)

    return person_list


def defect_anchor_finding(frame, frame_sub, frame_data):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]
    person_list = fix_symmetric(person_list)
    person_list = fix_keypoints(person_list)
    ret = False
    for element in person_list:
        x, y, _, _, H, R, xc, yc = get_bpf(frame, element)

        if H / R >= 150 and element['score'] > 1.5:
            frame = Visualizer.show_anchor(frame, element)
            frame = Visualizer.show_line(frame, element, sub_index=12)
            # frame = Visualizer.show_label(frame, int(x), int(y), f'{theta :.1f}', (0, 0, 255))
            frame = cv2.circle(frame, (xc, yc), 1, (255, 255, 255), 4)

            keypoints = element['keypoints']

            # print(dis1, dis2)

            # frame = cv2.circle(frame, theta, 1, (255, 255, 255), 3)
            # frame = cv2.line(frame, theta0, (xc, yc), color=(0, 0, 0), thickness=2)
            # frame = cv2.line(frame, (xc, yc), theta, color=(0, 0, 0), thickness=2)
            # print(np.array(element['keypoints']).reshape(26, 3))
            # print('*' * 10)
            ret = True
    return frame, ret


def paint_rule(frame, frame_sub, frame_data, learner, scan_cnt, keypoints_num):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]
    cnt = 0
    for element in person_list:
        if element['score'] > 1.7:
            np_keypoints = np.array(element['keypoints'])

            k = 0
            if is_stand(np_keypoints):
                # stand
                k = 1
            elif is_handsup(np_keypoints):
                # hansup
                k = 2
            elif is_look_aside(np_keypoints):
                k = 4
            elif is_sit(np_keypoints):
                # sit
                k = 3

            clr = (0, 128, 128)
            if k == 1:
                clr = (255, 0, 0)
            elif k == 2:
                clr = (0, 255, 0)
            elif k == 3:
                clr = (0, 0, 255)
            elif k == 4:
                clr = (128, 128, 0)

            # if k == 4:
            # frame = Visualizer.show_line(frame, element)
            (box_x, box_y, box_h, box_w) = element['box']

            frame = Visualizer.show_anchor(frame, element)

            sub = logger.find_person_index(box_x, box_y, box_h, box_w)

            frame = Visualizer.show_label(frame, int(box_x), int(box_y), f'{k}__{sub}', clr)

            # frame = FaceDirection.draw_direction(frame, element['keypoints'])

            logger.update_person(box_x, box_y, box_h, box_w, k, frame_sub)

            cnt += 1
            if cnt == scan_cnt:
                return frame
    return frame


def paint_model(frame, frame_sub, frame_data, learner, scan_cnt, keypoints_num):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]
    cnt = 0
    for element in person_list:
        if element['score'] > 1.:
            np_keypoints = np.array(element['keypoints'])
            np_keypoints = std_coordinate(1, 1, element['box'], np_keypoints, 26)[:keypoints_num, :]
            keypoints_m = ImageProcess.__get_matrix__(np_keypoints, keypoints_num)
            keypoints_m = transforms.ToTensor()(keypoints_m)
            keypoints_m = torch.unsqueeze(keypoints_m, dim=0).to(torch.float)

            keypoints_pm = np.array([np_keypoints[:, 2]]).T
            # keypoints_pm = np.matmul(np.transpose(keypoints_pm), keypoints_pm)
            # keypoints_pm = keypoints_pm / np.sum(keypoints_pm, axis=1)
            keypoints_pm = transforms.ToTensor()(keypoints_pm)
            keypoints_pm = torch.unsqueeze(keypoints_pm, dim=0).to(torch.float)

            keypoints_pm = keypoints_pm.to(device)
            keypoints_m = keypoints_m.to(device)
            pred = learner(keypoints_pm, keypoints_m).argmax(dim=1)
            for k, v in NAME_MAP.items():
                if v == pred.item():
                    clr = (0, 0, 255)
                    if k == 'stand':
                        clr = (255, 0, 0)
                    elif k == 'handsup':
                        clr = (0, 255, 0)
                    if k == 'handsup':
                        frame = Visualizer.show_line(frame, element)
                        # frame = Visualizer.show_anchor(frame, element)
                        # frame = Visualizer.show_label(frame, int(element['box'][0]), int(element['box'][1]), k, clr)
            cnt += 1
            if cnt == scan_cnt:
                return frame
    return frame


if __name__ == '__main__':

    with open(f'../test/resource/video/src_videos/only_class/alphapose-results-1.json', 'r') as fp:
        json_file = json.load(fp)

    frame_data = split_frame_json(json_file)
    learner = KeyPointLearnerGAT(AT_LAYER, AT_MULTI).to(device)
    # load_model('../test/resource/model.pkl', learner)

    learner.eval()

    cap = cv2.VideoCapture(f"../test/resource/video/src_videos/only_class/only_class_Trim_1.mp4")  # 读取视频文件
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../test/resource/video/output001.avi', fourcc, 30, (frame_w, frame_h))
    frame_cnt = 0

    while True:
        ret, frame = cap.read()
        if ret:
            # frame = paint_rule(frame, frame_cnt, frame_data, learner, -1, keypoints_num=26)
            # frame = paint_model(frame, frame_cnt, frame_data, learner, -1, keypoints_num=26)
            frame, ret = defect_anchor_finding(frame, frame_cnt, frame_data)

            out.write(frame)
            cv2.imshow("frame", frame)

            if ret:
                cv2.waitKey(0)
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_cnt += 1
        else:
            break
    cap.release()

    print(f'total frame = {frame_cnt}')

    # logger.save_data('../test/resource/logging.json')
    # logger.show_data(10)
