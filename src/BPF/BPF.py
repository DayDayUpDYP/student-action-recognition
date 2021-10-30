import numpy as np


class BPF:
    @staticmethod
    def dis_coord(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def dis_keypoints(sub1, sub2, keypoints):
        sub1_x = keypoints[sub1 * 3]
        sub1_y = keypoints[sub1 * 3 + 1]
        sub2_x = keypoints[sub2 * 3]
        sub2_y = keypoints[sub2 * 3 + 1]
        return BPF.dis_coord(sub1_x, sub1_y, sub2_x, sub2_y)

    @staticmethod
    def pack_feature(keypoints):
        v1 = (
            keypoints[10 * 3] - keypoints[8 * 3],
            keypoints[10 * 3 + 1] - keypoints[8 * 3 + 1]
        )
        val1 = -v1[1] / (BPF.dis_keypoints(10, 8, keypoints) + 0.0003)
        v1 = (
            keypoints[9 * 3] - keypoints[7 * 3],
            keypoints[9 * 3 + 1] - keypoints[7 * 3 + 1]
        )
        val2 = -v1[1] / (BPF.dis_keypoints(9, 7, keypoints) + 0.0003)

        features = {
            '17_y': keypoints[17 * 3 + 1],
            '18_y': keypoints[18 * 3 + 1],
            '8_y': keypoints[8 * 3 + 1],
            '6_y': keypoints[6 * 3 + 1],
            '5_y': keypoints[5 * 3 + 1],
            '7_y': keypoints[7 * 3 + 1],

            'theta_0': val1,  # left elbow
            'theta_1': val2,  # right elbow

            'dis_0': BPF.dis_keypoints(8, 10, keypoints) / (BPF.dis_keypoints(6, 8, keypoints) + 0.0003),
            # dis_8_10 / dis_8_6
            'dis_1': BPF.dis_keypoints(7, 9, keypoints) / (BPF.dis_keypoints(5, 7, keypoints) + 0.0003),
            # dis_7_9 / dis_5_7
        }

        return features

    def __init__(self, x, y):
        self.location = (x, y)

        self.features = []

        self.behavior = 'sit'

    def update_features(self, feature):
        if len(self.features) == 10:
            self.features.pop(0)
        self.features.append(feature)

        if self.is_stand() is True:
            self.behavior = 'standing'
        elif self.is_stand() is False:
            self.behavior = 'sitting'
        elif self.behavior == 'standing':
            self.behavior = 'stand'
        elif self.behavior != 'stand':
            if self.is_hands_up():
                self.behavior = 'hands_up'
            elif self.is_sit():
                self.behavior = 'sit'

    def is_stand(self):
        sum = 0
        for i in range(1, len(self.features)):
            sum += -(self.features[i]['17_y'] - self.features[i - 1]['17_y'])
        length = (self.features[-1]['18_y'] - self.features[-1]['17_y']) // 2
        if sum >= length:
            return True
        elif sum <= -length:
            return False
        else:
            return None

    def is_hands_up(self):
        return \
            (0.9 < self.features[-1]['theta_1'] <= 1 and self.features[-1]['dis_1'] >= 2 / 3.) \
            or \
            (0.9 < self.features[-1]['theta_0'] <= 1 and self.features[-1]['dis_0'] >= 2 / 3.)

    def is_look_aside(self):
        pass

    def is_sit(self):
        return \
            self.features[-1]['8_y'] >= self.features[-1]['6_y'] \
            and \
            self.features[-1]['7_y'] >= self.features[-1]['5_y']

    def equal(self, other_location):
        det = abs(self.location[0] - other_location[0]) + abs(self.location[1] - other_location[1])
        return det <= 30


if __name__ == '__main__':
    pass
