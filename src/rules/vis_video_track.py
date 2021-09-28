import torch
import json
import cv2

from utils.Visualizer import Visualizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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


def paint(frame, frame_sub, frame_data, scan_cnt):
    if not check_frame(frame_sub, frame_data):
        return frame
    person_list = frame_data[frame_sub]
    for element in person_list:
        if element['score'] > 1.:
            # if element['idx'] == 8:
            frame = Visualizer.show_anchor(frame, element)
            frame = Visualizer.show_line(frame, element)
            frame = Visualizer.show_label(frame, int(element['box'][0]), int(element['box'][1]), str(element['idx']),
                                          (255, 0, 0))
            frame = Visualizer.show_label(frame, 5, 18, 'frame:' + str(frame_sub), (255, 0, 0))
            if element['idx'] is not None:
                frame = Visualizer.show_label(frame, int(element['box'][0]), int(element['box'][1]),
                                              str(element['idx']), (255, 0, 0))
            # return frame
    return frame


if __name__ == '__main__':
    with open('../../test/resource/video/src_videos/2/alphapose-results-track-2.json', 'r') as fp:
        json_file = json.load(fp)

    frame_data = split_frame_json(json_file)

    cap = cv2.VideoCapture("../../test/resource/video/src_videos/2/demo_all_Trim_2.avi")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../../test/resource/video/output001.avi', fourcc, 30, (frame_w, frame_h))
    frame_cnt = 0

    cap_cnt = 0
    gap_cnt = 0
    is_painting = False

    while True:
        ret, frame = cap.read()
        if ret:
            frame_cnt += 1
            frame = paint(frame, frame_cnt, frame_data, 3)

            out.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
