import numpy as np

def BPSK(x):
    return 1 - x * 2

def awgn(x, SNR):
    xpower = np.mean(x ** 2)
    sigma2 = xpower / pow(10.0, SNR / 10.0)
    np.random.randn()
    noise = np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(sigma2)
    return x + noise