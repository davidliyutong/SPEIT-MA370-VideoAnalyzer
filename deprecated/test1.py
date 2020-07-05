import cv2
from utils import get_3d_sample, get_2d_sample, BGR_2_gray, get_2d_neighbor
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


def ncc(X, Y):
    if isinstance(X, np.ndarray):
        X = cp.array(X)
    if isinstance(Y, np.ndarray):
        Y = cp.array(Y)
    n = int(np.prod(X.shape))
    # mu_X, mu_Y = cp.average(X), cp.average(Y)
    # sigma_X, sigma_Y = cp.power(cp.var(X), 0.5), cp.power(cp.var(Y), 0.5)
    # X = (X - mu_X) / sigma_X
    # Y = (Y - mu_Y) / sigma_Y
    # NCC = cp.sum(cp.multiply((X - mu_X), (Y - mu_Y)) / (sigma_X * sigma_Y)) / (n - 1)
    NCC = (cp.sum(X * Y) - (cp.sum(X)*cp.sum(Y) / n)) / cp.power((cp.sum(X * X) - cp.sum(X)**2 / n) * (cp.sum(Y * Y) - cp.sum(Y)**2 / n), 0.5)
    return NCC

def ncc_field(neigbor, label, n, M):
    L = M - n + 1
    ans = np.zeros(shape=(L, L))
    for i in range(L):
        for j in range(L):
            ans[i, j] = ncc(sample_neigbor[i:i + n, j:j+n], sample_label)
    return ans

coord_X, coord_Y = 1000, 1000
window_sz = 32
neighbor_sz = 64
input_img1 = cv2.imread('../images/1.png')
input_img2 = cv2.imread('../images/2.png')

sample_neigbor = get_2d_neighbor(BGR_2_gray(input_img1), coord_X, coord_X, window_sz, neighbor_sz )
sample_label = get_2d_sample(BGR_2_gray(input_img2), coord_X, coord_X, window_sz)




ans = ncc_field(sample_neigbor, sample_label, window_sz, neighbor_sz)
print(ans)
plt.imshow(sample_label)
plt.show()
plt.imshow(sample_neigbor)
plt.show()
plt.imshow(ans)
plt.show()
print('finish')