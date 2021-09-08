import cv2
from pathlib import Path


class VideoToImage:
    def __init__(self, fps, in_path, out_path):
        self.in_path = Path(in_path)
        self.out_path = Path(out_path)
        self.fps = fps
        self.context = {}
        self.total_img_cnt = 0

    def convert_all(self):
        for mp4_path in self.in_path.rglob('*.mp4'):
            x = self.context.setdefault(f'{mp4_path.parent.name}', 0)
            self.context[f'{mp4_path.parent.name}'] += 1
            self.convert(mp4_path.parent.name + '_' + str(self.context[f'{mp4_path.parent.name}']), self.fps,
                         mp4_path,
                         self.out_path)

    def statistics(self):
        print(f'total img count is {self.total_img_cnt}')
        print(self.context)

    def convert(self, sub, fps, in_path, out_path):
        capture = cv2.VideoCapture(str(in_path))
        total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frame):
            ret = capture.grab()
            if not ret:
                print("[ERROR] Can't get frame from error videos.")
                break
            if i % fps == 0:
                print(f'[Processing] {sub}_{i} jpg...')
                ret, frame = capture.retrieve()
                if ret:
                    self.total_img_cnt += 1
                    (out_path / f'{in_path.parent.name}').mkdir(exist_ok=True)
                    frame = cv2.flip(cv2.transpose(frame), 1)
                    cv2.imwrite(str(out_path / f'{in_path.parent.name}/{sub}_{i}.jpg'), frame)


if __name__ == '__main__':
    vtm = VideoToImage(10, '../../test/resource/input', '../../test/resource/output')
    vtm.convert_all()
    vtm.statistics()
