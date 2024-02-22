### TO RUN
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
import pickle
import numpy as np

from classification.utils.plots import plot_specgram, show_confusion_matrix, plot_decision_boundaries
from classification.utils.utils import accuracy
from classification.datasets import Dataset

# from classification.src.classification.utils.audio_student import AudioUtil, Feature_vector_DS

# import torch
# from DL_model import *

# -----------------------------------------------------------------------------
"""
Synthesis of the functions in :

- accuracy : Compute the accuracy between prediction and ground truth.
- fixed_to_float(x,e) : Convert signal from integer fixed format to floating point. 
- float2fixed : Convert signal from float format to integer fixed point, here q15_t.
- fixed2binary : convert int in q15_t format into a 16-bit binary string.
- resize_and_fix_origin : Pads sig to reach length `L`, and shift it in order to cancel phase.
- convole : compute the convolution through Fourier transform
- correlate : Compute the correlation through Fourier transform
- threshold : Threshold an audio signal based on its energy per packet of Nft samples
- quantize  : quantize a signal on n-bits
- flatten : Flattens a multidimensional array into a 1D array 
- STFT_subsamp : compute an undersampled STFT
- STFT : compute an averaged compressed STFT
- load_model : Load pretrained model 
- eval_model : Run inference on trained model with the validation set 
"""

a = pickle.load(open("./" + "birds" + ".pickle", 'rb'))
b = pickle.load(open("./" + "birds2.pickle", 'rb'))
c = pickle.load(open("./" + "chainsaw1.pickle", 'rb'))
d = pickle.load(open("./" + "chainsaw2.pickle", 'rb'))
e = pickle.load(open("./" + "fire1.pickle", 'rb'))
f = pickle.load(open("./" + "fire2" + ".pickle", 'rb'))
g = pickle.load(open("./" + "handsaw1" + ".pickle", 'rb'))
h = pickle.load(open("./" + 'helicopter' + ".pickle", 'rb'))

data1_list = []  # Toutes les données côté à côte
data2_list = []  # Tous les tag de chaque fichier côte à côte

# Variables contenant les données
data_variables = [a, b, c, d, e, f, g, h]

# Itération sur chaque variable pour extraire les données
for data_variable in data_variables:
    # Vérification de la forme [[data1],[data2]]
    # Extraction et ajout des données à data1_list et data2_list
    # print(data_variable[0])
    for i in range(len(data_variable[0])):
        data1_list.append(data_variable[0][i])
        data2_list.append(data_variable[1][i])

print(data2_list)

pickle.dump([data1_list, data2_list], open("./all.pickle", 'wb'))

model = RandomForestClassifier(n_estimators=100, min_samples_split=2)

pca = PCA(n_components=35, whiten=True)
X_learn_reduced = pca.fit_transform(np.array(data1_list))

model.fit(X_learn_reduced, np.array(data2_list))

print(model.predict(pca.transform([data1_list[0]])))

filename = 'random_forest_Q1_parameters.pickle'
model_dir = "data/models/"
pickle.dump(model, open(model_dir + filename, 'wb'))
pickle.dump(pca, open(model_dir + "pca_Q1_parameters", 'wb'))
