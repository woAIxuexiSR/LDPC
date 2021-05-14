import encode
import modulate
import decode
from utility import *

import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle
import sys

def search_alpha():
    MATRIX_PATH = "Matrix(2016,1008)Block56.mat"
    Hp, Hs, H = load_H(MATRIX_PATH)

    SNR = 1.0
    sigma2 = 1.0 / pow(10.0, SNR / 10.0)

    MAX_ERROR_FRAME = 50 if SNR <= 1.0 else 3 

    alpha = np.arange(0.0, 1.01, 0.1)
    ber = np.zeros_like(alpha)
    fer = np.zeros_like(alpha)

    for i in range(len(alpha)):

        MAX_IT = 30
        ROUND = 0
        BER = 0
        FER = 0

        while True:
            print("alpha ", alpha[i], ", round ", ROUND)

            s = np.random.randint(0, 2, (1, K), dtype=np.int)
            x = encode.method_2(s, Hp, Hs)

            y = modulate.awgn(modulate.BPSK(x), SNR)

            y_llr = decode.LLR(y, sigma2).reshape(-1)
            x_hat = decode.NMS(y_llr, H, MAX_IT, alpha[i])

            cnt = np.sum(np.abs(s - x_hat))
            BER += cnt
            if cnt != 0:
                FER += 1

            ROUND += 1
            if FER > MAX_ERROR_FRAME:
                break
        
        ber[i] = BER / ROUND
        fer[i] = FER / ROUND
        
        # print("BER : ", BER / ROUND,  "  FER : ", FER / ROUND)
    
    f = open("./data/alpha_ber.pkl", "wb")
    pickle.dump(ber, f)
    f.close()
    f = open("./data/alpha_fer.pkl", "wb")
    pickle.dump(fer, f)
    f.close()

    # plt.plot(alpha, ber)
    # plt.show()

def search_beta():
    MATRIX_PATH = "Matrix(2016,1008)Block56.mat"
    Hp, Hs, H = load_H(MATRIX_PATH)

    SNR = 1.0
    sigma2 = 1.0 / pow(10.0, SNR / 10.0)

    MAX_ERROR_FRAME = 50 if SNR <= 1.0 else 3 

    beta = np.arange(0.0, 1.01, 0.1)
    ber = np.zeros_like(beta)
    fer = np.zeros_like(beta)

    for i in range(len(beta)):

        MAX_IT = 30
        ROUND = 0
        BER = 0
        FER = 0

        while True:
            print("beta ", beta[i], ", round ", ROUND)

            s = np.random.randint(0, 2, (1, K), dtype=np.int)
            x = encode.method_2(s, Hp, Hs)

            y = modulate.awgn(modulate.BPSK(x), SNR)

            y_llr = decode.LLR(y, sigma2).reshape(-1)
            x_hat = decode.OMS(y_llr, H, MAX_IT, beta[i])

            cnt = np.sum(np.abs(s - x_hat))
            BER += cnt
            if cnt != 0:
                FER += 1

            ROUND += 1
            if FER > MAX_ERROR_FRAME:
                break
        
        ber[i] = BER / ROUND
        fer[i] = FER / ROUND
        
        # print("BER : ", BER / ROUND,  "  FER : ", FER / ROUND)
    
    f = open("./data/beta_ber.pkl", "wb")
    pickle.dump(ber, f)
    f.close()
    f = open("./data/beta_fer.pkl", "wb")
    pickle.dump(fer, f)
    f.close()

    # plt.plot(beta, ber)
    # plt.show()

def simulate(method):
    MATRIX_PATH = "Matrix(2016,1008)Block56.mat"
    Hp, Hs, H = load_H(MATRIX_PATH)

    SNR = np.arange(-1.0, 2.1, 0.5)
    ber = np.zeros_like(SNR)
    fer = np.zeros_like(SNR)

    for i in range(len(SNR)):

        sigma2 = 1.0 / pow(10.0, SNR[i] / 10.0)

        MAX_ERROR_FRAME = 50 if SNR[i] <= 1.0 else 3 

        MAX_IT = 30
        ROUND = 0
        BER = 0
        FER = 0

        while True:
            print("SNR ", SNR[i], ", round ", ROUND)

            s = np.random.randint(0, 2, (1, K), dtype=np.int)
            x = encode.method_2(s, Hp, Hs)

            y = modulate.awgn(modulate.BPSK(x), SNR[i])

            y_llr = decode.LLR(y, sigma2).reshape(-1)

            if method == "SP":
                x_hat = decode.SP(y_llr, H, MAX_IT)
            elif method == "MS":
                x_hat = decode.MS(y_llr, H, MAX_IT)
            elif method == "NMS":
                x_hat = decode.NMS(y_llr, H, MAX_IT, 0.7)
            else:
                x_hat = decode.OMS(y_llr, H, MAX_IT, 0.5)

            cnt = np.sum(np.abs(s - x_hat))
            BER += cnt
            if cnt == 0:
                FER += 1

            ROUND += 1
            if FER > MAX_ERROR_FRAME:
                break

        ber[i] = BER / ROUND
        fer[i] = FER / ROUND

        # print("BER : ", BER / ROUND,  "  FER : ", FER / ROUND)

    f = open("./data/" + method + "_ber.pkl", "wb")
    pickle.dump(ber, f)
    f.close()
    f = open("./data/" + method + "_fer.pkl", "wb")
    pickle.dump(fer, f)
    f.close()

    # plt.plot(SNR, ber)
    # plt.show()

if __name__ == "__main__":
    start = time()

    if len(sys.argv) < 2:
        simulate("SP")
    elif sys.argv[1] == "alpha":
        search_alpha()
    elif sys.argv[1] == "beta":
        search_beta()
    else:
        simulate(sys.argv[1])

    end = time()
    print(end - start, " Seconds")