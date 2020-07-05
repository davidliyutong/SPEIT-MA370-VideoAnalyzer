import cv2
from utils import get_3d_sample, get_2d_sample, BGR_2_gray, get_2d_neighbor
import numpy as np
import cupy as cp
from NCC import ncc_neighbor
import matplotlib.pyplot as plt
import time

coord_X, coord_Y = 1000, 1200
window_sz = 32
neighbor_sz = 64
input_img1 = cv2.imread('../images/1.png')
input_img2 = cv2.imread('../images/1.png')

sample_neigber = get_2d_neighbor(BGR_2_gray(input_img2), coord_X, coord_X, window_sz, neighbor_sz, gpu=True)
sample_label = get_2d_sample(BGR_2_gray(input_img1), coord_X, coord_X, window_sz, gpu=True)


_, ans = ncc_neighbor(sample_neigber, sample_label)
plt.figure(figsize=(8,8))
plt.imshow(cp.asnumpy(sample_neigber), vmin=40, vmax=100, cmap='gray')
plt.savefig('neighbor.svg')
plt.show()
plt.imshow(cp.asnumpy(sample_label), vmin=40, vmax=100, cmap='gray')
plt.savefig('window.svg')
plt.show()
plt.imshow(cp.asnumpy(ans))
plt.savefig('ncc.svg')
plt.show()
print('finish')