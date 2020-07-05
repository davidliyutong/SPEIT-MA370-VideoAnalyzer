import numpy as np
import cupy as cp
import time
from utils import gen_track_P, decompose, get_2d_sample, BGR_2_gray
from NCC import ncc, ncc_neighbor
from Videoloader import VideoLoader
VL = VideoLoader('../videos/IMG_1441.MOV')
import matplotlib.pyplot as plt
n = 32
M = 64

last_frame = VL.read_gray()
P = gen_track_P(last_frame, M)
Tx = [0]
Ty = [0]
Rot = [0]

start_t = time.time()
cnt = 0

while True:
    curr_frame = VL.read_gray()
    if curr_frame is None:
        break

    P_1 = ncc(last_frame, curr_frame, P, n, M)
    tx, ty, rot = decompose(P, P_1)
    # tx, ty, rot = P_1[0, 1] - P[0, 1], P_1[1, 1] - P[1, 1], 0
    Tx.append(tx + Tx[-1])
    Ty.append(ty + Ty[-1])
    Rot.append(rot + Rot[-1])
    last_frame = curr_frame
    cnt += 1
    if time.time() - start_t > 5:
        print(cnt / 5)
        cnt = 0
        start_t = time.time()


plt.figure(figsize=(15,15))
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
print('finish')