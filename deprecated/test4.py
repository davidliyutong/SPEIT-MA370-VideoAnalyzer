import cv2
from utils import gen_track_P, decompose,get_2d_sample,BGR_2_gray
from NCC import ncc, ncc_neighbor

import matplotlib.pyplot as plt
import time

coord_X, coord_Y = 1000, 1200
window_sz = 32
neighbor_sz = 64
input_img1 = cv2.imread('../images/4.png')
input_img2 = cv2.imread('../images/3.png')

P = gen_track_P(input_img1, neighbor_sz)
P_1, A = ncc(input_img1, input_img2, P, window_sz, neighbor_sz)
tx, ty, rot = decompose(A)

print(P)
print(P_1)
print(A)
print('finish')
