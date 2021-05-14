import scipy.io as sio
import numpy as np
from utility import *

def method_1(s, Hp, Hs):
    Hs_t = np.transpose(Hs)
    Hp_i_t = np.abs(np.linalg.inv(np.transpose(Hp)).astype(np.int))

    p = np.matmul(np.matmul(s, Hs_t), Hp_i_t) % 2

    return np.concatenate((p, s), axis=1)

def method_2(s, Hp, Hs):
    p = np.zeros((1, N - K), dtype=np.int)
    w = np.matmul(s, np.transpose(Hs)) % 2

    for i in range(NZ):
        for j in range(N_NZ - K_NZ):
            if j == 0 and i == 0:
                p[0, i] = w[0, i]
            elif j == 0:
                p[0, i] = (w[0, i] + p[0, (N_NZ - K_NZ - 1) * NZ + i - 1]) % 2
            else:
                p[0, j * NZ + i] = (w[0, j * NZ + i] + p[0, (j - 1) * NZ + i]) % 2

    return np.concatenate((p, s), axis=1)

if __name__ == "__main__":
    MATRIX_PATH = "Matrix(2016,1008)Block56.mat"
    
    Hp, Hs, H = load_H(MATRIX_PATH)

    for i in range(1000):
        s = np.random.randint(0, 2, (1, K), dtype=np.int)
        out = method_2(s, Hp, Hs)

        check = np.matmul(H, np.transpose(out)) % 2
        if check.max() != 0:
            print("Error occured!")
    
    print("Success!")