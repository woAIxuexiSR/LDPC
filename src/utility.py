import scipy.io as sio
import numpy as np

NZ = 56
N_NZ = 36
K_NZ = 18
N = 2016
K = 1008

def create_block(n):
    ans = np.zeros((NZ, NZ), dtype=np.int)

    if n == 0:
        return ans

    n = n - 1
    for i in range(NZ):
        ans[i, n] = 1
        n = (n + 1) % NZ

    return ans

def create_H(M):
    h, w = M.shape
    ans = np.zeros((h * NZ, w * NZ), dtype=np.int)

    for i in range(h):
        for j in range(w):
            ans[i * NZ : i * NZ + NZ, j * NZ : j * NZ + NZ] = create_block(M[i, j])

    return ans

def load_H(path):
    data = sio.loadmat(path)

    H_block = data["H_block"]

    Hp = create_H(H_block[:, 0 : N_NZ - K_NZ])
    Hp[0, -1] = 0

    Hs = create_H(H_block[:, N_NZ - K_NZ : N_NZ])

    H = np.concatenate((Hp, Hs), axis=1)

    # print(Hp.shape, Hs.shape, H.shape)

    return Hp, Hs, H