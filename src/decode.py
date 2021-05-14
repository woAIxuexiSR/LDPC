import numpy as np
import math
from utility import *

def LLR(y, sigma2):
    return 2 / sigma2 * y

def SP(y_llr, H_block, max_it):
    u = np.zeros((N - K, N))
    v = np.zeros((N, N - K))
    x = np.zeros((1, N), dtype=np.int)

    l_idx = [np.where(H_block[i, :] == 1)[0] for i in range(N - K)]
    r_idx = [np.where(H_block[:, i] == 1)[0] for i in range(N)]

    for i in range(N):
        u[r_idx[i], i] = y_llr[i]

    for _ in range(max_it):
        
        for i in range(N):
            tmp_idx = r_idx[i].copy()
            for idx in r_idx[i]:
                v[i, idx] = y_llr[i] + np.sum(u[tmp_idx[tmp_idx != idx], i])

        for i in range(N - K):
            tmp_idx = l_idx[i].copy()
            for idx in l_idx[i]:
                product = np.prod(np.tanh(v[tmp_idx[tmp_idx != idx], i] / 2))
                if product == 1 or product == -1 :
                    u[i, idx] = product * 1e10
                else :
                    u[i, idx] = 2 * np.arctanh(product)
        
        for i in range(N):
            if np.sum(u[r_idx[i], i]) + y_llr[i] < 0:
                x[0, i] = 1
            else:
                x[0, i] = 0
        
        check = np.matmul(H_block, np.transpose(x)) % 2
        if check.max() == 0:
            # print("check success at the ", _, " th iteration.")
            break

    return x[:, N - K : N]

def MS(y_llr, H_block, max_it):
    u = np.zeros((N - K, N))
    v = np.zeros((N, N - K))
    x = np.zeros((1, N), dtype=np.int)

    l_idx = [np.where(H_block[i, :] == 1)[0] for i in range(N - K)]
    r_idx = [np.where(H_block[:, i] == 1)[0] for i in range(N)]

    for i in range(N):
        u[r_idx[i], i] = y_llr[i]

    for _ in range(max_it):
        
        for i in range(N):
            tmp_idx = r_idx[i].copy()
            for idx in r_idx[i]:
                v[i, idx] = y_llr[i] + np.sum(u[tmp_idx[tmp_idx != idx], i])

        for i in range(N - K):
            tmp_idx = l_idx[i].copy()
            for idx in l_idx[i]:
                arr = v[tmp_idx[tmp_idx != idx], i]
                u[i, idx] = np.prod(np.sign(arr)) * np.abs(arr).min()
        
        for i in range(N):
            if np.sum(u[r_idx[i], i]) + y_llr[i] < 0:
                x[0, i] = 1
            else:
                x[0, i] = 0
        
        check = np.matmul(H_block, np.transpose(x)) % 2
        if check.max() == 0:
            # print("check success at the ", _, " th iteration.")
            break

    return x[:, N - K : N]

def NMS(y_llr, H_block, max_it, alpha):
    u = np.zeros((N - K, N))
    v = np.zeros((N, N - K))
    x = np.zeros((1, N), dtype=np.int)

    l_idx = [np.where(H_block[i, :] == 1)[0] for i in range(N - K)]
    r_idx = [np.where(H_block[:, i] == 1)[0] for i in range(N)]

    for i in range(N):
        u[r_idx[i], i] = y_llr[i]

    for _ in range(max_it):
        
        for i in range(N):
            tmp_idx = r_idx[i].copy()
            for idx in r_idx[i]:
                v[i, idx] = y_llr[i] + np.sum(u[tmp_idx[tmp_idx != idx], i])

        for i in range(N - K):
            tmp_idx = l_idx[i].copy()
            for idx in l_idx[i]:
                arr = v[tmp_idx[tmp_idx != idx], i]
                u[i, idx] = np.prod(np.sign(arr)) * np.abs(arr).min() * alpha
        
        for i in range(N):
            if np.sum(u[r_idx[i], i]) + y_llr[i] < 0:
                x[0, i] = 1
            else:
                x[0, i] = 0
        
        check = np.matmul(H_block, np.transpose(x)) % 2
        if check.max() == 0:
            # print("check success at the ", _, " th iteration.")
            break

    return x[:, N - K : N]

def OMS(y_llr, H_block, max_it, beta):
    u = np.zeros((N - K, N))
    v = np.zeros((N, N - K))
    x = np.zeros((1, N), dtype=np.int)

    l_idx = [np.where(H_block[i, :] == 1)[0] for i in range(N - K)]
    r_idx = [np.where(H_block[:, i] == 1)[0] for i in range(N)]

    for i in range(N):
        u[r_idx[i], i] = y_llr[i]

    for _ in range(max_it):
        
        for i in range(N):
            tmp_idx = r_idx[i].copy()
            for idx in r_idx[i]:
                v[i, idx] = y_llr[i] + np.sum(u[tmp_idx[tmp_idx != idx], i])

        for i in range(N - K):
            tmp_idx = l_idx[i].copy()
            for idx in l_idx[i]:
                arr = v[tmp_idx[tmp_idx != idx], i]
                u[i, idx] = np.prod(np.sign(arr)) * max(np.abs(arr).min() - beta, 0)
        
        for i in range(N):
            if np.sum(u[r_idx[i], i]) + y_llr[i] < 0:
                x[0, i] = 1
            else:
                x[0, i] = 0
        
        check = np.matmul(H_block, np.transpose(x)) % 2
        if check.max() == 0:
            # print("check success at the ", _, " th iteration.")
            break

    return x[:, N - K : N]