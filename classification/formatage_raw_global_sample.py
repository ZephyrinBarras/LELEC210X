import pickle

import numpy as np

a = pickle.load(open("data/raw_global_samples/birds_globalsample.pickle", "rb"))
b = pickle.load(open("data/raw_global_samples/fire_globalsample.pickle", "rb"))
c = pickle.load(open("data/raw_global_samples/handsaw_globalsample.pickle", "rb"))
d = pickle.load(open("data/raw_global_samples/chainsaw_globalsample.pickle", "rb"))
e = pickle.load(open("data/raw_global_samples/helicopter_globalsample.pickle", "rb"))

# Nombre d'échantillons (512 * 70 * 260 -> voir uart-reader.py)
nbr_samples = len(a[1][0]) * len(a[1])

# np.ravel : Return a contiguous flattened array.
new_a = np.ravel(a[1])
new_b = np.ravel(b[1])
new_c = np.ravel(c[1])
new_d = np.ravel(d[1])
new_e = np.ravel(e[1])

print(len(new_a))
"""
pickle.dump(["birds", new_a], open("./birds_reformated_globalsample.pickle", "wb"))
pickle.dump(["fire", new_b], open("./fire_reformated_globalsample.pickle", "wb"))
pickle.dump(["handsaw", new_c], open("./handsaw_reformated_globalsample.pickle", "wb"))
pickle.dump(["chainsaw", new_d], open("./chainsaw_reformated_globalsample.pickle", "wb"))
pickle.dump(["helicopter", new_e], open("./helicopter_reformated_globalsample.pickle", "wb"))
"""

test = pickle.load(open("data/raw_global_samples/helicopter_reformated_globalsample.pickle", "rb"))
print(test)
print(len(test[1]))

# C'est carré : on a bien l'étiquette suivie de tout l'échantillon qui tient en une seule liste

a = np.arange(10)
b = np.arange(11, 20)

print(a)
print(b)

c = np.concatenate((a, b), axis=0)
print(c)