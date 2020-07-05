import cupy as cp
import numpy as np
import math
from utils import get_3d_sample, get_2d_sample, BGR_2_gray, get_2d_neighbor
import matplotlib.pyplot as plt

def auto_detect(shape: tuple, matmul=False):
    assert isinstance(shape, tuple)
    if len(shape) == 4:
        raise NotImplementedError
    elif len(shape) == 3:
        block_dim = (1, 16, 16)
        grid_dim = (
        math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1]), math.ceil(shape[2] / block_dim[2]))
    elif len(shape) == 2:
        if matmul:
            block_dim = (32, 32)
            grid_edge = max(math.ceil(shape[0] / block_dim[0]), math.ceil(shape[0] / block_dim[1]))
            grid_dim = (grid_edge, grid_edge)
            return grid_dim, block_dim
        if shape[0] < 32 and shape[1] < 32:
            block_dim = (shape[0], shape[1])
            grid_dim = (1, 1)
        elif shape[1] < 32:
            block_dim = (math.floor(1024 / shape[1]), shape[1])
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
        elif shape[0] < 32:
            block_dim = (shape[0], math.floor(1024 / shape[0]))
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
        else:
            block_dim = (32, 32)
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
    elif len(shape) == 1:
        if matmul:
            block_dim = (32, 32)
            grid_dim = (math.ceil(shape[0] / 32), math.ceil(shape[0] / 32))
            return grid_dim, block_dim
        if shape[0] < 1024:
            block_dim = (shape[0], 1)
            grid_dim = (1, 1)
        else:
            block_dim = (1024, 1)
            grid_dim = (1, 1)
    else:
        raise ValueError

    return grid_dim, block_dim


ncc_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cal_nccf(float * X, float * Y, int n, int L, int M, float * ans, float * _S_Y, float * _S_YY) {
        float S_X = 0;
        float S_XX = 0;
        float S_XY = 0;
        float S_Y = _S_Y[0];
        float S_YY = _S_YY[0];
        float res = 0;
        float tmp = 0;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= L || y >= L) return;
        int idx_X = 0;
        int idx_Y = 0;
        int m = L * L;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                idx_X = (x + i) * M + (y + j);
                idx_Y = i * n + j;
                S_X += X[idx_X];
                S_XX += X[idx_X] * X[idx_X];
                tmp = max(Y[idx_Y], tmp);
                S_XY += X[idx_X] * Y[idx_Y];
            }
        }
        res = (S_XY - S_X * S_Y / m) / sqrtf((S_XX - S_X * S_X / m) * (S_YY - S_Y * S_Y / m));
        if (isnan(res)) {
            ans[x * L + y] = res;
        } else {
            ans[x * L + y] = res;
        }
        
        // ans[x * L + y] = S_XX;
    }
    ''', 'cal_nccf')


def ncc_neighbor(X, Y, threshhold=0.05):
    """
    :param X: The neighbor
    :param Y: The label
    search the position of label in the neighbor
    :return:
    """
    assert X.shape[0] == X.shape[1] and Y.shape[0] == Y.shape[1]

    if not isinstance(X, cp.ndarray):
        X = cp.array(X)
    if not isinstance(Y, cp.ndarray):
        Y = cp.array(Y)

    X, Y = X.astype(cp.float32), Y.astype(cp.float32)

    M, n = X.shape[0], Y.shape[0]
    L = M - n + 1
    S_Y, S_YY = cp.sum(Y), cp.sum(Y * Y)

    grid_dim, block_dim  = (16, 16), (math.ceil(L / 16), math.ceil(L / 16))
    ans = cp.zeros(shape=(L, L), dtype=cp.float32)
    ncc_kernel(grid_dim, block_dim, (X, Y, n, L, M, ans, S_Y, S_YY))

    max_idx = cp.asnumpy(cp.argmax(ans))
    max_coord = (math.ceil(max_idx / L), max_idx % L)
    delta = math.floor((M - n + 1) / 2)
    dpoint = np.array([max_coord[0] - delta - 1, max_coord[1] - delta])

    return dpoint, ans


def ncc(X, Y, P, n, M):
    """
    X: The i-1 frame
    Y: The i frame
    P: matrix/list of points
    n: size of window
    M: size of neighbor
    :param X:
    :param Y:
    :param P:
    :param n:
    :param M:
    :return: a matrix of tracked
    """
    P_1 = P.copy()

    assert len(X.shape) == len(Y.shape) == 2

    for i in range(len(P)):
        point = P[0:2, i]
        neigbor = get_2d_neighbor(X, point[0], point[1], n, M, gpu=True)
        label = get_2d_sample(Y, point[0], point[1], n, gpu=True)
        dpoint, _ = ncc_neighbor(neigbor, label)
        P_1[0:2, i] = (point + dpoint).T
        # if i == 1:
        #     plt.imshow(cp.asnumpy(neigbor))
        #     plt.show()
        #     plt.imshow(cp.asnumpy(label))
        #     plt.show()
        #     plt.imshow(cp.asnumpy(_))
        #     plt.show()

    # plt.scatter(P[0, :], P[1, :], c='b')
    # plt.scatter(P_1[0, :], P_1[1, :], c='r')
    # plt.show()
    return P_1



