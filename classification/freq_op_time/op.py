import numpy as np
from numpy import log10
import matplotlib.pyplot as plt

BASE_TIME = 0.5
BASE_FREQ = 10e3
T = 1/BASE_FREQ
NBR_ECHAN = BASE_TIME/T
plt.figure()

def ope(n):
    return (n*log10(n))

def plot_op_time():
    time = np.linspace(0.02,0.1,1000)
    echan = time*BASE_FREQ
    print(time)
    print(ope(echan)/time)
    plt.xlabel("time slice [s]")
    plt.ylabel("operation/sec")
    plt.plot(time, ope(echan)/time)
    plt.show()

def plot_op_freq():
    freq = np.linspace(5000,20000,1000)
    echan = BASE_TIME*freq
    plt.xlabel("fréquence d'échantillonage [Hz]")
    plt.ylabel("operation/sec")
    plt.plot(freq, ope(echan)/BASE_TIME)
    plt.show()

def plot_op(freq, time):
    echan = time*freq
    print(f"base : \t {ope(NBR_ECHAN)/BASE_TIME} perso : \t {ope(echan)/time}")


plot_op_time()
plot_op_freq()

