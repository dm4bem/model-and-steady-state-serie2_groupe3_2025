import numpy as np
import pandas as pd

def Matrice():
  A = np.zeros([26, 18])       # n° of branches X n° of nodes
  A[0, 0] = 1                 # branch 0: -> node 0
  A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
  A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
  A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
  A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
  A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5
  A[6, 4], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
  A[7, 5], A[7, 6] = -1, 1    # branch 7: node 5 -> node 6
  A[8, 7] = 1                 # branch 8: -> node 7
  A[9, 5], A[9, 7] = 1, -1    # branch 9: node 5 -> node 7
  A[10, 6] = 1                # branch 10: -> node 6
  A[11, 6] = 1                # branch 11: -> node 6
  A[12, 8] = 1               # branch 0': -> node 12
  A[13, 9], A[13, 8] = 1, -1         # branch 1': node 12 -> node 13
  A[14, 10], A[14, 9] = 1, -1         # branch 2': node 13 -> node 14
  A[15, 11], A[15, 10] = 1, -1         # branch 3': node 14 -> node 15
  A[16, 12], A[16, 11] = 1, -1         # branch 4': node 13 -> node 14
  A[17, 13] = 1               # branch 0": -> node 17
  A[18, 14], A[18, 13] = 1, -1         # branch 1": node 17 -> node 18
  A[19, 15], A[19, 14] = 1, -1         # branch 2": node 18 -> node 19
  A[20, 16], A[20, 15] = 1, -1         # branch 3": node 19 -> node 20
  A[21, 15], A[21, 16] = 1, -1         # branch 4": node 18 -> node 19
  A[22, 5], A[22, 12] = 1, -1         # branch 5': node 16 -> node 5
  A[23, 5], A[23, 17] = 1, -1         # branch 5": node 21 -> node 5
  A[24, 6], A[24, 12] = 1, -1         # branch 6': node 16 -> node 6
  A[25, 6], A[25, 17] = 1, -1         # branch 6": node 21 -> node 6
  return A

A=Matrice()
