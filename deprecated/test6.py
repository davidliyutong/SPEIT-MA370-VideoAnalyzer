import numpy as np
P = np.array([[1,0,0],
              [0,0,1],
              [1,1,1]])
P_1 = np.array([[2,1,1],
               [1,1,2],
               [1,1,1]])
A = np.array([[1,0,1],
              [0,1,1],
              [0,0,1]])
PP_1 = np.dot(A, P)
A_1 = np.linalg.solve(P, P_1)
print(A)