import pickle
import numpy as np
import matplotlib.pyplot as plt
from utility import *

def load_data(path):
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

alpha = np.arange(0.0, 1.01, 0.1)
alpha_ber = load_data("./data/alpha_ber.pkl")
# fer = load_data("./data/alpha_fer.pkl")

plt.xlabel("alpha")
plt.ylabel("BER")

# print(alpha_ber / K)
plt.semilogy(alpha, alpha_ber / K)
# plt.savefig("./data/alpha_ber.jpg")
plt.show()

beta = np.arange(0.0, 1.01, 0.1)
beta_ber = load_data("./data/beta_ber.pkl")

plt.xlabel("beta")
plt.ylabel("BER")

# print(beta_ber / K)
plt.semilogy(beta, beta_ber / K)
# plt.savefig("./data/beta_ber.jpg")
plt.show()

SNR = np.arange(-1.0, 2.1, 0.5)
SP_ber = load_data("./data/SP_ber.pkl")
MS_ber = load_data("./data/MS_ber.pkl")
NMS_ber = load_data("./data/NMS_ber.pkl")
OMS_ber = load_data("./data/OMS_ber.pkl")

plt.xlabel("Eb/N0")
plt.ylabel("BER")

# print(SP_ber / K)
# print(MS_ber / K)
# print(NMS_ber / K)
# print(OMS_ber / K)

plt.semilogy(SNR, SP_ber / K, c="r", label="SP")
plt.semilogy(SNR, MS_ber / K, c="b", label="MS")
plt.semilogy(SNR, NMS_ber / K, c="g", label="NMS")
plt.semilogy(SNR, OMS_ber / K, c="y", label="OMS")

plt.legend()
# plt.savefig("./data/Eb_N0_ber.jpg")
plt.show()


SP_fer = load_data("./data/SP_fer.pkl")
MS_fer = load_data("./data/MS_fer.pkl")
NMS_fer = load_data("./data/NMS_fer.pkl")
OMS_fer = load_data("./data/OMS_fer.pkl")

plt.xlabel("Eb/N0")
plt.ylabel("FER")

# print(SP_fer)
# print(MS_fer)
# print(NMS_fer)
# print(OMS_fer)

plt.semilogy(SNR, SP_fer, c="r", label="SP")
plt.semilogy(SNR, MS_fer, c="b", label="MS")
plt.semilogy(SNR, NMS_fer, c="g", label="NMS")
plt.semilogy(SNR, OMS_fer, c="y", label="OMS")

plt.legend()
# plt.savefig("./data/Eb_N0_fer.jpg")
plt.show()