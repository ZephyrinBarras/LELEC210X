### TO RUN
import os
import numpy as np
import matplotlib.pyplot as plt

"Machine learning tools"
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
#print(a[1])
data1_list = []
data2_list = []

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

#print(len(data1_list))

begin_n_trees = 75
end_n_trees = 150
step_n_trees = 5

range_n_trees = range(begin_n_trees, end_n_trees, step_n_trees)

begin_pca = 5
end_pca = 200
step_pca = 5

range_pca = range(begin_pca, end_pca, step_pca)

# 10 pour que ça prenne moins de temps
k_splits_cross_validation = 5

precis = np.zeros((len(range_n_trees), len(range_pca)))
ecart_type = np.zeros((len(range_n_trees), len(range_pca)))

pickle.dump([data1_list, data2_list], open("./all.pickle", 'wb'))

accuracy_knn = np.zeros(k_splits_cross_validation)

kf = StratifiedKFold(n_splits=k_splits_cross_validation, shuffle=True)

np.set_printoptions(threshold=9)
print(data1_list[0])
print(len(data1_list[0]))

for i in range(begin_n_trees, end_n_trees, step_n_trees):
    model = RandomForestClassifier(n_estimators=i, min_samples_split=2)
    for j in range(begin_pca, end_pca, step_pca):
        for k, idx in enumerate(kf.split(data1_list, data2_list)):
            (idx_learn, idx_val) = idx

            pca = PCA(n_components=j, whiten=True)
            X_learn_reduced = pca.fit_transform(np.array(data1_list)[idx_learn])
            X_val_reduced = pca.transform(np.array(data1_list)[idx_val])

            model.fit(X_learn_reduced, np.array(data2_list)[idx_learn])

            prediction_knn = model.predict(X_val_reduced)
            accuracy_knn[k] = accuracy(prediction_knn, np.array(data2_list)[idx_val])

        temp_mean = np.mean(accuracy_knn)
        temp_ecart_type = np.std(accuracy_knn)

        precis[(i - begin_n_trees) // step_n_trees][(j - begin_pca) // step_pca] = temp_mean
        ecart_type[(i - begin_n_trees) // step_n_trees][(j - begin_pca) // step_pca] = temp_ecart_type

        print("i : {}, j : {}, accuracy : {}, écart-type = {}".format(i, j, temp_mean, temp_ecart_type))

print("Accuracy:")
print(precis)

print("Écart-type:")
print(ecart_type)

# plt.figure()
# plt.imshow(100 * precis, cmap='jet', origin='lower')
# plt.show()
