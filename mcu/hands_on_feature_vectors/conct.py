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

from classification.datasets import Dataset
from classification.utils.plots import plot_specgram, show_confusion_matrix, plot_decision_boundaries
from classification.utils.utils import accuracy
from classification.utils.audio_student import AudioUtil, Feature_vector_DS


a = pickle.load(open("./"+"birds"+".pickle", 'rb'))
b = pickle.load(open("./"+"birds2.pickle", 'rb'))
c = pickle.load(open("./"+"chainsaw1.pickle", 'rb'))
d = pickle.load(open("./"+"chainsaw2.pickle", 'rb'))
e = pickle.load(open("./"+"fire1.pickle", 'rb'))
f = pickle.load(open("./"+"fire2"+".pickle", 'rb'))
g = pickle.load(open("./"+"handsaw1"+".pickle", 'rb'))
h = pickle.load(open("./"+'helicopter'+".pickle", 'rb'))
print(a[1])
data1_list = []
data2_list = []

# Variables contenant les données
data_variables = [a, b, c, d, e, f, g, h]

# Itération sur chaque variable pour extraire les données
for data_variable in data_variables:
    # Vérification de la forme [[data1],[data2]]
    # Extraction et ajout des données à data1_list et data2_list
    #print(data_variable[0])
    for i in range(len(data_variable[0])):
        data1_list.append(data_variable[0][i])
        data2_list.append(data_variable[1][i])

print(len(data1_list))
precis = np.zeros((15,16))
pickle.dump([data1_list, data2_list], open("./all.pickle", 'wb'))
accuracy_knn = np.zeros(5)
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits,shuffle=True)

for i in range(75,150,5):
    model = RandomForestClassifier(i)
    for j in range(4,5):
        model_rf = RandomForestClassifier(i)

        for k, idx in enumerate(kf.split(data1_list,data2_list)):
            
            (idx_learn, idx_val) = idx
            tot = len(idx_learn)+len(idx_val)
            print(len(idx_learn)/tot, len(idx_val)/tot)
            
            pca = PCA(n_components=j,whiten=True)
            X_learn_reduced = pca.fit_transform(np.array(data1_list)[idx_learn])
            X_val_reduced = pca.transform(np.array(data1_list)[idx_val])

            model.fit(X_learn_reduced,np.array(data2_list)[idx_learn])

            prediction_knn = model.predict(X_val_reduced)
            accuracy_knn[k] = accuracy(prediction_knn, np.array(data2_list)[idx_val])
        precis[(i-75)//5][j-4]=np.mean(accuracy_knn)
        print(np.mean(accuracy_knn))
print(precis)
            
    

plt.figure()
plt.imshow(100*precis, cmap='jet', origin='lower')
plt.show()