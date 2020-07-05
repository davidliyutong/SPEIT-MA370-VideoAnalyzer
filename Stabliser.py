from MotionAnalyser import MotionAnalyzer
import numpy as np
from KalmanFilter import KalmanFilter
from VideoProcessor import VideoCorrector
import argparse


def main(args):
    assert args.path_to_video is not None
    KFilter = KalmanFilter(args.Q, args.R)
    MAnly = MotionAnalyzer(args.path_to_video)
    Vcorr = VideoCorrector(args.path_to_video)

    Tx, Ty, _ = MAnly.process()
    res_x, res_y = KFilter.filter(Tx), KFilter.filter(Ty)
    delta_x = np.array(Tx) - np.array(res_x)
    delta_y = np.array(Ty) - np.array(res_x)
    Vcorr.correct(zip(delta_x, delta_y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stablise a video")
    parser.add_argument("--path_to_video", default=None, type=str, help="select a video")
    parser.add_argument("--Q", default=0.001, type=float, help="Q")
    parser.add_argument("--R", default=0.01, type=float, help="R")

    args = parser.parse_args()
    main(args)
    print('finish')