import torch
import json
import cv2
import numpy as np
from torchvision.transforms import transforms

from utils.ImageProcess import std_coordinate
from utils.ImageProcess import ImageProcess
from models import KeyPointLearner
from conf import *
from utils.Visualizer import Visualizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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


def paint(frame, frame_sub, frame_data, learner, scan_cnt, keypoints_num):
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
                    if k == 'stand':
                        frame = Visualizer.show_anchor(frame, element)
                        frame = Visualizer.show_line(frame, element)
                        frame = Visualizer.show_label(frame, int(element['box'][0]), int(element['box'][1]), k, clr)
            cnt += 1
            if cnt == scan_cnt:
                return frame
    return frame


if __name__ == '__main__':
    with open('../test/resource/video/src_videos/2/alphapose-results.json', 'r') as fp:
        json_file = json.load(fp)

    frame_data = split_frame_json(json_file)
    learner = KeyPointLearner().to(device)
    load_model('../test/resource/model.pkl', learner)

    learner.eval()

    cap = cv2.VideoCapture("../test/resource/video/src_videos/2/demo_all_Trim_2.avi")  # 读取视频文件
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../test/resource/video/output001.avi', fourcc, 30, (frame_w, frame_h))
    frame_cnt = 0

    cap_cnt = 0
    gap_cnt = 0
    is_painting = False

    while True:
        ret, frame = cap.read()
        if ret:
            frame_cnt += 1
            frame = paint(frame, frame_cnt, frame_data, learner, -1, keypoints_num=26)

            # if is_painting:
            #     cap_cnt += 1
            #     if cap_cnt == 120:
            #         is_painting = False
            #         cap_cnt = 0
            #     frame = paint(frame, frame_cnt, frame_data, learner, 5, keypoints_num=26)
            # else:
            #     gap_cnt += 1
            #     if gap_cnt == 90:
            #         is_painting = True
            #         gap_cnt = 0

            out.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
