import numpy as np
import cupy as cp
import time
from utils import gen_track_P, decompose, get_2d_sample, BGR_2_gray
from NCC import ncc, ncc_neighbor
from Videoloader import VideoLoader
import datetime
import os


class MotionAnalyzer:
    def __init__(self, video_path, n=32, M=64):
        self.VL = VideoLoader(video_path)
        self.n = n
        self.M = M
        self.last_frame = None

    def _preprocess(self):
        self.last_frame = self.VL.read_gray()
        self.P = gen_track_P(self.last_frame, self.M)
        self.Tx = [0]
        self.Ty = [0]
        self.Rot = [0]

    def process(self, log=True):
        start_t = time.time()
        cnt = 0
        self._preprocess()
        while True:
            curr_frame = self.VL.read_gray()
            if curr_frame is None:
                break
            P_1 = ncc(self.last_frame, curr_frame, self.P, self.n, self.M)
            tx, ty, rot = decompose(self.P, P_1)
            self.Tx.append(tx + self.Tx[-1])
            self.Ty.append(ty + self.Ty[-1])
            self.Rot.append(rot + self.Rot[-1])
            self.last_frame = curr_frame
            cnt += 1
            if time.time() - start_t > 5 and log:
                print(cnt / 5)
                cnt = 0
                start_t = time.time()
        return self.Tx, self.Ty, self.Rot

    def get_motion(self):
        return self.Tx, self.Ty, self.Rot

    def save(self, save_dir='./data'):
        np.save(os.path.join(save_dir, 'Tx{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), np.array(self.Tx))
        np.save(os.path.join(save_dir, 'Ty{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), np.array(self.Ty))
        np.save(os.path.join(save_dir, 'Rot{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), np.array(self.Rot))

