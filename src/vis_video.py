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

from rules.simple_rules import is_sit, is_handsup, is_stand

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


def paint_rule(frame, frame_sub, frame_data, learner, scan_cnt, keypoints_num):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]
    cnt = 0
    for element in person_list:
        if element['score'] > 1.:
            np_keypoints = np.array(element['keypoints'])

            k = 0
            if is_stand(np_keypoints):
                # stand
                k = 1
            elif is_handsup(np_keypoints):
                # hansup
                k = 2
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

            # if k == 'stand':
            #     frame = Visualizer.show_line(frame, element)
            (box_x, box_y, box_h, box_w) = element['box']

            frame = Visualizer.show_anchor(frame, element)

            sub = logger.find_person_index(box_x, box_y, box_h, box_w)

            frame = Visualizer.show_label(frame, int(box_x), int(box_y), f'{k}__{sub}', clr)

            frame = FaceDirection.draw_direction(frame, element['keypoints'])

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
                    if k == 'sit' and element['score'] >= 1.6:
                        frame = Visualizer.show_line(frame, element)
                        frame = Visualizer.show_anchor(frame, element)
                        frame = Visualizer.show_label(frame, int(element['box'][0]), int(element['box'][1]), k, clr)
            cnt += 1
            if cnt == scan_cnt:
                return frame
    return frame


if __name__ == '__main__':

    video_name = 'classroom'

    with open(f'../test/resource/video/src_videos/{video_name}/alphapose-results.json', 'r') as fp:
        json_file = json.load(fp)

    frame_data = split_frame_json(json_file)
    learner = KeyPointLearnerGAT(AT_LAYER, AT_MULTI).to(device)
    load_model('../test/resource/model.pkl', learner)

    learner.eval()

    cap = cv2.VideoCapture(f"../test/resource/video/src_videos/{video_name}/{video_name}_Trim.avi")  # 读取视频文件
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../test/resource/video/output001.avi', fourcc, 30, (frame_w, frame_h))
    frame_cnt = 0

    while True:
        ret, frame = cap.read()
        if ret:
            frame = paint_rule(frame, frame_cnt, frame_data, learner, -1, keypoints_num=26)

            out.write(frame)
            cv2.imshow("frame", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_cnt += 1
        else:
            break
    cap.release()

    print(f'total frame = {frame_cnt}')

    logger.save_data('../test/resource/logging.json')
    logger.show_data(10)
