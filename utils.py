import math
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def get_3d_sample(X, x, y, n):
    if x + n >= X.shape[0] or y + n >= X.shape[1]:
        return None
    else:
        return X[x:x + n, y:y + n, :]


def get_2d_sample(X, x, y, n, gpu=False):
    if x + n >= X.shape[0] or y + n >= X.shape[1]:
        return None
    else:
        if gpu and (not isinstance(X, cp.ndarray)):
            return cp.array(X[x:x + n, y:y + n], dtype=cp.float32)
        else:
            return X[x:x + n, y:y + n]


def get_2d_neighbor(X, x, y, n, M, gpu=False):
    delta = math.floor((M - n + 1) / 2)
    if x - delta < 0 or x + delta >= X.shape[0] or y - delta < 0 or y + delta >= X.shape[1]:
        return None
    else:
        if gpu and (not isinstance(X, cp.ndarray)):
            return cp.array(X[x - delta:x + delta + n, y - delta:y + delta + n], dtype=cp.float32)
        else:
            return X[x - delta:x + delta + n, y - delta:y + delta + n]


def RGB_2_gray(X):
    return X[:, :, 0] * 0.299 + X[:, :, 1] * 0.587 + X[:, :, 2] * 0.114


def BGR_2_gray(X):
    return X[:, :, 2] * 0.299 + X[:, :, 1] * 0.587 + X[:, :, 0] * 0.114


def gen_track_P(X, M):
    """
    :param X: A frame
    :param n: size of winodw
    :param M: size of neighbor
    :return: a list of points
    """
    H, W = X.shape[0], X.shape[1]
    offset = 10 * M
    P = np.zeros(shape=(3, 3), dtype=np.int)
    P[2,:] = np.array([0, 0, 1])
    P[0:2, 0] = np.array([np.clip(H / 2 + offset, M, H - M), np.clip(W / 2 - offset, M, W - M)]).T
    P[0:2, 1] = np.array([np.clip(H / 2 - offset, M, H - M), np.clip(W / 2 - offset, M, W - M)]).T
    P[0:2, 2] = np.array([np.clip(H / 2 - offset, M, H - M), np.clip(W / 2 + offset, M, W - M)]).T
    P[2,:] = 1
    return P


def decompose(P, P_1):
    P_0 = P - np.array([P[0, 1], P[1, 1], 0]).T
    P_1 = P_1 - np.array([P[0, 1], P[1, 1], 0]).T
    A = np.linalg.solve(P_0, P_1)
    # Q, R = np.linalg.qr(A)
    # tx = R[0, 2] * (P[0, 0] - P[1, 0])
    # ty = R[1, 2] * (P[0, 0] - P[1, 0])
    # tx = A[0, 2] * (P[0, 0] - P[1, 0])
    # ty = A[1, 2] * (P[0, 0] - P[1, 0])
    tx = np.average(P_1[0, :] - P_0[0, :])
    ty = np.average(P_1[1, :] - P_0[1, :])
    rotation = 0
    # rotation = math.atan2(-A[1, 0], A[1, 1])
    return tx, ty, rotation


def plot_Tx_Ty_Rot(Tx, Ty, Rot):
    plt.figure(figsize=(15, 15))
    frame_index = np.linspace(1, len(Tx), len(Tx))
    plt.axis('on')
    plt.subplot(311)
    plt.title('Tx')
    plt.ylabel('Tx')
    plt.xlabel('Index')
    plt.plot(frame_index, Tx)
    plt.grid()

    plt.subplot(312)
    plt.title('Ty')
    plt.ylabel('Ty')
    plt.xlabel('Index')
    plt.plot(frame_index, Ty)
    plt.grid()

    plt.subplot(313)
    plt.title('Rot')
    plt.ylabel('Rot')
    plt.xlabel('Index')
    plt.plot(frame_index, Rot)
    plt.grid()

    plt.savefig('analyse.svg')
    plt.show()


def plot_Tx_Ty(Resx, Resy):
    plt.figure(figsize=(15, 15))
    frame_index = np.linspace(1, len(Resx), len(Resx))
    plt.axis('on')
    plt.subplot(211)
    plt.title('Tx')
    plt.ylabel('Tx')
    plt.xlabel('Index')
    plt.plot(frame_index, Resx)
    plt.grid()

    plt.subplot(212)
    plt.title('Ty')
    plt.ylabel('Ty')
    plt.xlabel('Index')
    plt.plot(frame_index, Resy)
    plt.grid()

    plt.savefig('analyse.svg')
    plt.show()