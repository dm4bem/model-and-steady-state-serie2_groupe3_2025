import numpy as np

def Matrice():
  A = np.zeros([29, 22])       # n° of branches X n° of nodes
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
  A[12, 12] = 1               # branch 0': -> node 12
  A[13, 13], A[13, 12] = 1, -1         # branch 1': node 12 -> node 13
  A[14, 14], A[14, 13] = 1, -1         # branch 2': node 13 -> node 14
  A[15, 15], A[15, 14] = 1, -1         # branch 3': node 14 -> node 15
  A[16, 16], A[16, 15] = 1, -1         # branch 4': node 13 -> node 14
  A[17, 17] = 1               # branch 0": -> node 17
  A[18, 18], A[18, 17] = 1, -1         # branch 1": node 17 -> node 18
  A[19, 19], A[19, 18] = 1, -1         # branch 2": node 18 -> node 19
  A[20, 20], A[20, 19] = 1, -1         # branch 3": node 19 -> node 20
  A[21, 21], A[21, 20] = 1, -1         # branch 4": node 18 -> node 19
  A[22, 5], A[22, 16] = 1, -1         # branch 5': node 16 -> node 5
  A[23, 5], A[23, 21] = 1, -1         # branch 5": node 21 -> node 5
  A[24, 6], A[24, 16] = 1, -1         # branch 6': node 16 -> node 6
  A[25, 6], A[25, 21] = 1, -1         # branch 6": node 21 -> node 6
  A[26, 16], A[26, 4] = 1, -1         # branch 12: node 4 -> node 16
  A[27, 21], A[27, 16] = 1, -1         # branch 12': node 16 -> node 21
  A[28, 4], A[28, 21] = 1, -1         # branch 12": node 21 -> node 4
  return A

A=Matrice()
print (A)
