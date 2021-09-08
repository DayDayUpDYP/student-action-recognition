import json
from pathlib import Path
import cv2
from utils.ImageProcess import split_person_dict


class Visualizer:
    @staticmethod
    def show_anchor(img, img_data):
        x = int(img_data['box'][0])
        y = int(img_data['box'][1])
        h = int(img_data['box'][2])
        w = int(img_data['box'][3])
        return cv2.rectangle(img, (x, y), (x + h, y + w), (0, 0, 255), 2)

    @staticmethod
    def show_keypoint(img, img_data):
        for i in range(0, 26, 3):
            x = img_data['keypoints'][i]
            y = img_data['keypoints'][i + 1]
            img = cv2.rectangle(
                img, (int(x), int(y)), (int(x), int(y)),
                (0, 0, 255), 4)
        return img

    @staticmethod
    def show_line(img, img_data):
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
            (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
            (36, 37),
            (37, 38),  # Face
            (38, 39), (39, 40), (40, 41), (41, 42), (43, 44), (44, 45), (45, 46), (46, 47), (48, 49), (49, 50),
            (50, 51),
            (51, 52),  # Face
            (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61), (62, 63), (63, 64), (64, 65),
            (65, 66),
            (66, 67),  # Face
            (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79),
            (79, 80),
            (80, 81),  # Face
            (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91),
            (91, 92),
            (92, 93),  # Face
            (94, 95), (95, 96), (96, 97), (97, 98), (94, 99), (99, 100), (100, 101), (101, 102), (94, 103), (103, 104),
            (104, 105),  # LeftHand
            (105, 106), (94, 107), (107, 108), (108, 109), (109, 110), (94, 111), (111, 112), (112, 113), (113, 114),
            # LeftHand
            (115, 116), (116, 117), (117, 118), (118, 119), (115, 120), (120, 121), (121, 122), (122, 123), (115, 124),
            (124, 125),  # RightHand
            (125, 126), (126, 127), (115, 128), (128, 129), (129, 130), (130, 131), (115, 132), (132, 133), (133, 134),
            (134, 135)  # RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

        nodes = []

        for i in range(0, 408, 3):
            x = int(img_data['keypoints'][i])
            y = int(img_data['keypoints'][i + 1])
            p = img_data['keypoints'][i + 2]
            nodes.append((x, y, p))
        for index in range(26):
            node1_index = l_pair[index][0]
            node2_index = l_pair[index][1]
            if node1_index >= 26 or node2_index >= 26:
                continue
            x1, y1, p1 = nodes[node1_index]
            x2, y2, p2 = nodes[node2_index]
            if p1 > 0.02 or p2 > 0.02:
                img = cv2.circle(img, (x1, y1), 1, p_color[node1_index], 2)
                img = cv2.circle(img, (x2, y2), 1, p_color[node2_index], 2)
                img = cv2.line(img, (x1, y1), (x2, y2), line_color[index], 2)
        return img

    def __init__(self, img_path, json_path):
        self.img_path = Path(img_path)
        with open(json_path, 'r') as fp:
            self.json_file = json.load(fp)
        self.img_data = split_person_dict(self.img_path.name, self.json_file)
        # print(self.img_data)

    def show_img(self, img):
        shape = img.shape[:2]
        img = cv2.resize(img, (int(shape[1] // 1.5), int(shape[0] // 1.5)))
        cv2.imshow(str(self.img_path.name), img)
        cv2.waitKey(0)

    def visualize(self, display):
        img = cv2.imread(str(self.img_path))
        img = Visualizer.show_anchor(img, self.img_data)
        img = Visualizer.show_line(img, self.img_data)
        img = Visualizer.show_keypoint(img, self.img_data)
        if display:
            self.show_img(img)
        return img


if __name__ == '__main__':

    path = Path('../../test/resource/output/sit')
    for img_path in path.rglob("*.jpg"):
        vis = Visualizer(img_path=str(img_path),
                         json_path='../../test/resource/res/alphapose-results.json')
        vis.visualize(1)
