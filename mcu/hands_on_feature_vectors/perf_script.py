import numpy as np
import matplotlib.pyplot as plt

file = open("perf.txt", "r")

data = file.readlines()
labels = ["0.1: Increase fixed-point scale","0.2: Remove DC Component",\
          "1: Windowing of input samples","2: Discrete Fourier Transform",\
            "3.1: Compute the complex magnitude of the FFT",\
                "3.2: Find the extremum value (maximum of absolute values",\
                    "3.3: Normalize the vector","3.4: Compute the complex magnitude",\
                        "4:Denormalize the vector","Apply MEL transform"]
dico = {}

for i in data:
    text = i.split(" ")
    if text[0] == "[PERF]":
        if text[1] in dico.keys():
            dico[text[1]].append(text[2])
        else:
            dico[text[1]] = [text[2]]


for i in dico.keys():
    dico[i] = np.mean([int(a) for a in dico[i]])

print(dico.values())

plt.figure(figsize=(16,9))
plt.title("Nombre de cycle par sequence de code")
plt.bar(labels,list(dico.values()))
plt.savefig("nbr_cycle.pdf")
plt.show()