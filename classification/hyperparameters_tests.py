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
import librosa
from sklearn.impute import SimpleImputer
import seaborn as sns

import classification.utils.audio_student as audio
from classification.utils.plots import plot_specgram, show_confusion_matrix, show_confusion_matrix_with_save, \
    plot_decision_boundaries
from classification.utils.utils import accuracy
from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS

from classification.utils.plots import plot_specgram, show_confusion_matrix, plot_decision_boundaries
from classification.datasets import Dataset

MELVEC_LENGTH_DEFAULT = 20  # hauteur (longueur de chaque vecteur)
N_MELVECS_DEFAULT = 20  # Nombre de vecteurs mel
fs_down = 11111  # Target sampling frequency


def compute_accuracy(prediction, target):
    """
    Compute the accuracy between prediction and ground truth.
    """
    return np.sum(prediction == target) / len(prediction)


def remove_dc_component(signal):
    mean_value = np.mean(signal)  # Calculer la moyenne du signal
    return signal - mean_value


def specgram(y, Nft=512, mellength=MELVEC_LENGTH_DEFAULT):
    """Build the spectrogram of a downsample and filtered (low-pass to avoid aliasing) sound record.

    Args:
      y (array of float): sound record after filtering (low pass filter to avoid aliasing) and downsampling.
      Nft (int): number of sample by fft

    Returns:
      stft (array of float): short time fourier transform
    """

    # Homemade computation of stft
    "Crop the signal such that its length is a multiple of Nft"
    "Pas besoin de crop car je le fais au moment de l'appel de la fonction tout en bas dans la boucle"
    #y = y[: mellength * Nft]

    L = len(y)
    #print("Taille de y :", L)

    "Reshape the signal with a piece for each row"
    audiomat = np.reshape(y, (L // Nft, Nft))
    audioham = audiomat * np.hamming(Nft)  # Windowing. Hamming, Hanning, Blackman,..
    z = np.reshape(audioham, -1)  # y windowed by pieces
    "FFT row by row"
    stft = np.fft.fft(audioham, axis=1)
    stft = np.abs(
        stft[:, : Nft // 2].T
    )  # Taking only positive frequencies and computing the magnitude

    # Enlever la dernière ligne pour que le produit matriciel fonctionne
    return stft[:-1, :]


def melspecgram(x, Nmel=N_MELVECS_DEFAULT, mellength=MELVEC_LENGTH_DEFAULT, Nft=512, fs=44100, fs_down=11025):
    """Get a audio record as input. Apply a low-pass filter then downsample before transforming the signal in melspectogram.

    Args:
      x (array de float): sound record
      Nmel (int): number of mels class --> TODO : je mettrais plutôt 20 = N_MELVECS
      Nft (int): number of sample by fft
      fs (int): frequency of the sampling
      fs_down (int): frequecy after the downsampling

    Returns:
      melspec (array of int): Melspectogram of the STFT
    """

    "Obtain the Hz2Mel transformation matrix"
    # y = resample(x)

    # A décommenter si on veut la calculer ici (la fonction buggait donc j'utilise une matrice de transformation calculée précédemment)
    # mels = librosa.filters.mel(sr=fs_down, n_fft=Nft, n_mels=Nmel)

    mels = pickle.load(open("data/mel_matrix/{}_N_MELVECS_mel_matrix.pickle".format(Nmel), "rb"))
    mels = mels[:, :-1]
    mels = mels / np.max(mels)
    """    
    "Plot de la matrice de transformation"
    plt.figure(figsize=(5, 4))
    plt.imshow(mels, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Hz2Mel transformation matrix")
    plt.xlabel("$N_{FT}$")
    plt.ylabel("$N_{Mel}$")
    plt.show()
    """

    stft = specgram(x, mellength=mellength, Nft=Nft)

    """print("=====================================")
    print("Fonction de calcul du mel spectrogamme :")
    print("Dimensions de la matrice de transfo (la matrice est correcte) :", "({},{})".format(len(mels), len(mels[0])),
          "Dimensions de la matrice de stft :", "({},{})".format(len(stft), len(stft[0])))
    """
    melspec = np.dot(mels, stft)
    return melspec


a = pickle.load(open("data/raw_global_samples/birds_reformated_globalsample.pickle", "rb"))
b = pickle.load(open("data/raw_global_samples/fire_reformated_globalsample.pickle", "rb"))
c = pickle.load(open("data/raw_global_samples/handsaw_reformated_globalsample.pickle", "rb"))
d = pickle.load(open("data/raw_global_samples/chainsaw_reformated_globalsample.pickle", "rb"))
e = pickle.load(open("data/raw_global_samples/helicopter_reformated_globalsample.pickle", "rb"))

print("Classes in the dataset:", ["birds", "fire", "handsaw", "chainsaw", "helicopter"])

N_MELVECS_begin = 10
N_MELVECS_end = 20
N_MELVECS_step = 2

MELVEC_LENGTH_begin = 20
MELVEC_LENGTH_end = 24
MELVEC_LENGTH_step = 2

N_MELVECS_arange = np.arange(N_MELVECS_begin, N_MELVECS_end, N_MELVECS_step)  # ICI LA RANGE DU PARAM NMEL
MELVEC_LENGTH_arange = np.arange(MELVEC_LENGTH_begin, MELVEC_LENGTH_end,
                                 MELVEC_LENGTH_step)  # ICI LA RANGE DU PARAM MELVEC_LENGTH

# La création des matrices a été adaptée à la modification des boucles i et j
accuracy_matrix = np.zeros((len(MELVEC_LENGTH_arange), len(N_MELVECS_arange)))
std_matrix = np.zeros((len(MELVEC_LENGTH_arange), len(N_MELVECS_arange)))

"Les deux boucles étaient incorrectes, il fallait les inverser"
"""for i in N_MELVECS_arange:
    for j in MELVEC_LENGTH_arange:"""

for i in MELVEC_LENGTH_arange:
    for j in N_MELVECS_arange:
        print("Nouvelle boucle pour : N_MELVECS = {}, MELVEC_LENGTH = {}".format(i, j))

        # Retirer la composante continue de chaque signal pour chaque classe
        a_signal_without_DC_component = remove_dc_component(a[1])
        b_signal_without_DC_component = remove_dc_component(b[1])
        c_signal_without_DC_component = remove_dc_component(c[1])
        d_signal_without_DC_component = remove_dc_component(d[1])
        e_signal_without_DC_component = remove_dc_component(e[1])

        data1_list = [] # Liste contenant les spectrogrammes de toutes les classes

        # Calculer les spectrogrammes
        a_spec = []
        size_a = len(a_signal_without_DC_component)
        cropped_a_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
        for m in range(0, len(cropped_a_spec), 512):
            temp_spec = np.log(np.abs(melspecgram(cropped_a_spec[m:m + 512], Nmel=i, mellength=j, fs_down=fs_down)))
            a_spec.append(temp_spec)

        # print("Taille de a_spec :", len(a_spec))
        # print("Taille d'un melvec :", len(a_spec[0]))

        a_spec_reshaped = []
        for k in range(0, len(a_spec) - (len(a_spec) % j), j): # Avancer par pas de j et tronquer le nombre de melvecs pour avoir un multiple de j
            a_spec_reshaped.append(np.ravel(a_spec[k:k+j])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

        data1_list = a_spec_reshaped # Pour la suite, il faudra concaténer les autres classes à la suite de cette liste

        """
        "A décommenter pour vérifier le formatage des spectrogrammes"    
        print("Taille de data1_list", len(data1_list))

        print("Taille de a_spec_reshaped :", len(a_spec_reshaped))
        #print(a_spec_reshaped[0])
        print(len(a_spec_reshaped[0]))
        print("Taille de cropped_a_spec :", len(cropped_a_spec))

        print("Spectrogrammes oiseau calculés !")
        """

        b_spec = []
        size_b = len(b_signal_without_DC_component)
        cropped_b_spec = b_signal_without_DC_component[:size_b - size_b % 512]
        for m in range(0, len(cropped_b_spec), 512):
            temp_spec = np.log(np.abs(melspecgram(cropped_b_spec[m:m + 512], Nmel=i, mellength=j, fs_down=fs_down)))
            b_spec.append(temp_spec)

        b_spec_reshaped = []
        for k in range(0, len(b_spec) - (len(b_spec) % j), j): # Avancer par pas de j
            b_spec_reshaped.append(np.ravel(b_spec[k:k+j])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

        data1_list = np.concatenate((data1_list, b_spec_reshaped))

        print("Spectrogrammes feu calculés !")


        c_spec = []
        size_c = len(c_signal_without_DC_component)
        cropped_c_spec = c_signal_without_DC_component[:size_c - size_c % 512]
        for m in range(0, len(cropped_c_spec), 512):
            temp_spec = np.log(np.abs(melspecgram(cropped_c_spec[m:m + 512], Nmel=i, mellength=j, fs_down=fs_down)))
            c_spec.append(temp_spec)

        c_spec_reshaped = []
        for k in range(0, len(c_spec) - (len(c_spec) % j), j): # Avancer par pas de j
            c_spec_reshaped.append(np.ravel(c_spec[k:k+j])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

        data1_list = np.concatenate((data1_list, c_spec_reshaped), axis=0)

        print("Spectrogrammes scie calculés !")


        d_spec = []
        size_d = len(d_signal_without_DC_component)
        cropped_d_spec = d_signal_without_DC_component[:size_d - size_d % 512]
        for m in range(0, len(cropped_d_spec), 512):
            temp_spec = np.log(np.abs(melspecgram(cropped_d_spec[m:m + 512], Nmel=i, mellength=j, fs_down=fs_down)))
            d_spec.append(temp_spec)

        d_spec_reshaped = []
        for k in range(0, len(d_spec) - (len(d_spec) % j), j): # Avancer par pas de j
            d_spec_reshaped.append(np.ravel(d_spec[k:k+j])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

        data1_list = np.concatenate((data1_list, d_spec_reshaped), axis=0)

        print("Spectrogrammes tronçonneuse calculés !")


        e_spec = []
        size_e = len(e_signal_without_DC_component)
        cropped_e_spec = e_signal_without_DC_component[:size_e - size_e % 512]
        for m in range(0, len(cropped_e_spec) - (len(e_spec) % j), 512):
            temp_spec = np.log(np.abs(melspecgram(cropped_e_spec[m:m + 512], Nmel=i, mellength=j, fs_down=fs_down)))
            e_spec.append(temp_spec)

        e_spec_reshaped = []
        for k in range(0, len(e_spec) - (len(e_spec) % j), j): # Avancer par pas de j
            e_spec_reshaped.append(np.ravel(e_spec[k:k+j])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

        data1_list = np.concatenate((data1_list, e_spec_reshaped), axis=0)

        print("Spectrogrammes hélicoptère calculés !")

        "Création de data2_list avec tous les labels"
        data2_list = []
        labels = ["birds", "fire", "handsaw", "chainsaw", "helicopter"]
        for label in range(0, 5):
            data2_list = np.concatenate((data2_list, [labels[label] for _ in range(len(a_spec_reshaped))]), axis=0)
            # en partant du principe que len(a_spec_reshaped) = len(b_spec_reshaped) = len(c_spec_reshaped) = len(d_spec_reshaped) = len(e_spec_reshaped)


        print("Data2_list longeur", len(data2_list))
        #print(data2_list)
        print("Longueur de tous les spectrogrammes : ", len(data1_list))
        print("Nombre d'éléments dans le premier spectrogramme", len(data1_list[0]))

        """
        myds = Feature_vector_DS(dataset, Nft=512, nmel=10, duration=950, shift_pct=0.0,
                                 fs=10980)  # METTRE i Á FS POUR FREQ OU NFT
        train_pct = 0.70

        featveclen = len(myds["fire", 0])  # number of items in a feature vector
        nitems = len(myds)  # number of sounds in the dataset
        naudio = dataset.naudio  # number of audio files in each class
        nclass = dataset.nclass  # number of classes
        nlearn = round(naudio * train_pct)  # number of sounds among naudio for training

        data_aug_factor = 1
        class_ids_aug = np.repeat(classnames, naudio * data_aug_factor)

        "Compute the matrixed dataset, this takes some seconds, but you can then reload it by commenting this loop and decommenting the np.load below"
        X = np.zeros((data_aug_factor * nclass * naudio, featveclen))
        for s in range(data_aug_factor):
            for class_idx, classname in enumerate(classnames):
                for idx in range(naudio):
                    featvec = myds[classname, idx]
                    X[s * nclass * naudio + class_idx * naudio + idx, :] = featvec
        np.save("data/feature_matrices/" + "feature_matrix_2D.npy", X)
        # X = np.load(fm_dir+"feature_matrix_2D.npy")
        """
        "Labels"
        # y = class_ids_aug.copy()
        # X_train, X_test, y_train, y_test = ...
        X_train, X_test, y_train, y_test = train_test_split(data1_list, data2_list, test_size=0.3, stratify=data2_list)  # random_state=1
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        n_trees = 100
        model = RandomForestClassifier(n_trees)
        accuracy = np.zeros((n_splits,))

        # best pca = PCA(n_components=5,whiten=True)with n=7+1=8
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        pca = PCA(n_components=35, whiten=True)
        for k, idx in enumerate(kf.split(X_train, y_train)):
            (idx_learn, idx_val) = idx

            # [2] (optional) Data normalization
            X_learn_normalised = X_train[idx_learn] / np.linalg.norm(X_train[idx_learn], axis=1, keepdims=True)
            # print(len(X_learn_normalised))
            X_val_normalised = X_train[idx_val] / np.linalg.norm(X_train[idx_val], axis=1, keepdims=True)
            # print(len(X_val_normalised))

            # [3] (optional) dimensionality reduction.
            imp.fit(X_learn_normalised)
            X_learn_normalised = imp.transform(X_learn_normalised) # Remplacer tous les NaN par la moyenne
            X_learn_reduced = pca.fit_transform(X_learn_normalised)

            imp.fit(X_val_normalised)
            X_val_normalised = imp.transform(X_val_normalised)
            X_val_reduced = pca.transform(X_val_normalised)
            pca.transform([X_val_normalised[0]])
            model.fit(X_learn_reduced, y_train[idx_learn])
            prediction = model.predict(X_val_reduced)
            # print(len(prediction_knn))
            accuracy[k] = compute_accuracy(prediction, y_train[idx_val])

        # accuracy est un tableau avec 5 élements car il y a 5 plis pour la validation croisée
        # accuracy.mean() est la moyenne des 5 élements
        # accuracy.std() est l'écart-type des 5 élements
        temp_mean = np.mean(accuracy)
        temp_std = np.std(accuracy)

        # Remplir les matrices globales (ont été modifiées en fonction des nouvelles boucles i et j)
        accuracy_matrix[(i - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step][(j - N_MELVECS_begin) // N_MELVECS_step] = temp_mean
        std_matrix[(i - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step][(j - N_MELVECS_begin) // N_MELVECS_step] = temp_std

        print("N_MELVECS : {}, MELVEC_LENGTH : {}, accuracy : {}, std : {}".format(i, j, 100 * temp_mean, 100 * temp_std))
        print("=====================================")


"""ax = sns.heatmap(accuracy_matrix, linewidth=0.5)
ax2 = sns.heatmap(std_matrix, linewidth=0.5)
plt.show()"""

plt.imshow(accuracy_matrix, cmap='hot', extent=np.concatenate((N_MELVECS_arange, MELVEC_LENGTH_arange)))
plt.colorbar()
plt.xlabel("N_MELVECS")
plt.ylabel("MELVEC_LENGTH")
plt.savefig("accuracy_matrix.png")
plt.show()

plt.imshow(std_matrix, cmap='hot', extent=np.concatenate((N_MELVECS_arange, MELVEC_LENGTH_arange)))
plt.colorbar()
plt.xlabel("N_MELVECS")
plt.ylabel("MELVEC_LENGTH")
plt.savefig("std_matrix.png")
plt.show()
