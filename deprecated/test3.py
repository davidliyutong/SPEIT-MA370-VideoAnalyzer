import cv2
from utils import get_3d_sample, get_2d_sample, BGR_2_gray, get_2d_neighbor
import numpy as np
import cupy as cp
from NCC import ncc
import matplotlib.pyplot as plt
import time

def ncc_single(X, Y):
    if isinstance(X, np.ndarray):
        X = cp.array(X)
    if isinstance(Y, np.ndarray):
        Y = cp.array(Y)
    n = int(np.prod(X.shape))
    NCC = (cp.sum(X * Y) - (cp.sum(X)*cp.sum(Y) / n)) / cp.power((cp.sum(X * X) - cp.sum(X)**2 / n) * (cp.sum(Y * Y) - cp.sum(Y)**2 / n), 0.5)
    return NCC


def ncc_field(neigbor, label):
    assert neigbor.shape[0] == neigbor.shape[1] and label.shape[0] == label.shape[1]
    n = label.shape[0]
    M = neigbor.shape[0]
    L = M - n + 1
    res = np.zeros(shape=(L, L))
    for i in range(L):
        for j in range(L):
            res[i, j] = ncc_single(neigbor[i:i + n, j:j + n], label)
    return res


coord_X, coord_Y = 1000, 1000
window_sz = 32
neighbor_sz = 64
input_img1 = cv2.imread('../images/1.png')
input_img2 = cv2.imread('../images/2.png')

sample_neigber = get_2d_neighbor(BGR_2_gray(input_img1), coord_X, coord_X, window_sz, neighbor_sz )
sample_label = get_2d_sample(BGR_2_gray(input_img2), coord_X, coord_X, window_sz)

print(sample_label)
print(sample_neigber)
start_t = time.time()
for i in range(100):
    ans = ncc(sample_neigber, sample_label)
print(time.time() - start_t)
start_t = time.time()
for i in range(100):
    ans = ncc_field(sample_neigber, sample_label)
print(time.time() - start_t)

