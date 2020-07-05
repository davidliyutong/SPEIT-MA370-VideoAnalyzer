import cv2
from Videoloader import VideoLoader
import datetime
import time
import numpy as np
import os


class VideoCorrector:
    def __init__(self, video_path=None, save_dir='./output'):
        self.video_path = video_path
        self.VL = VideoLoader(self.video_path)
        self.save_dir = save_dir
        self.VW = None

    def _preprocess(self):
        fps = self.VL.get_fps()
        size = self.VL.get_size()
        codec = cv2.VideoWriter_fourcc(*'MP4V')
        ext = os.path.split(self.video_path)[-1].split('.')[1]
        filename = os.path.split(self.video_path)[-1].split('.')[0] + '_filted' + '.mp4'
        self.save_path = os.path.join(self.save_dir, filename)
        self.VW = cv2.VideoWriter(self.save_path, codec, fps, size)

    def correct(self, delta, log=True):
        start_t = time.time()
        cnt = 0
        self._preprocess()
        for delta_x, delta_y in delta:
            curr_frame = self.VL.read()
            if curr_frame is None:
                break
            A = np.array([[1, 0, delta_y], [0, 1, delta_x]])
            H, W = curr_frame.shape[0], curr_frame.shape[1]
            dest_frame = cv2.warpAffine(curr_frame, A, (W, H))
            self.VW.write(dest_frame)
            cnt += 1
            if time.time() - start_t > 5 and log:
                print(cnt / 5)
                cnt = 0
                start_t = time.time()
        self.VW.release()
        print('saved filted video to: ' + self.save_path)