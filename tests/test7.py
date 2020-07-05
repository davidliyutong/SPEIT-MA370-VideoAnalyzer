from MotionAnalyser import MotionAnalyzer
from utils import plot_Tx_Ty_Rot, plot_Tx_Ty
import numpy as np
from KalmanFilter import KalmanFilter
from VideoProcessor import VideoCorrector

KFilter = KalmanFilter()
MAnly = MotionAnalyzer('./output/IMG_1432_filted.mp4')
Vcorr = VideoCorrector('./output/IMG_1432_filted.mp4')

Tx, Ty, _ = MAnly.process()
# MAnly.save()
plot_Tx_Ty(Tx, Ty)
res_x, res_y = KFilter.filter(Tx), KFilter.filter(Ty)
plot_Tx_Ty(res_x, res_y)
delta_x = np.array(Tx) - np.array(res_x)
delta_y = np.array(Ty) - np.array(res_y)
Vcorr.correct(zip(delta_x, delta_y))

print('finish')