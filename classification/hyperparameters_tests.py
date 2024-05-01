import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config

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
from scipy.signal import fftconvolve, resample

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
Nft = 512  # Number of samples by fft


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
    # y = y[: mellength * Nft]

    L = len(y)
    # print("Taille de y :", L)

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


def melspecgram(x, Nmel=N_MELVECS_DEFAULT, mellength=MELVEC_LENGTH_DEFAULT, Nft=512, fs=44100, fs_down=11111):
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


def turbo_lol_spectrograms(signal, i_melvec_length, j_n_melvecs, Nft=512, fs_down=11111) -> list:
    """
    Fonction qui calcule les vecteurs mel et les agrège en spectrogrammes
    Args:
        signal: signal audio d'entrée
        i_melvec_length: longueur des vecteurs mel
        j_n_melvecs: nombre de vecteurs mel à agréger pour former un spectrgrogramme
        Nft: nombre d'échantillons audio par fft
        fs_down: fréquence d'échantillonnage du signal

    Returns: liste de spectrogrammes, chacun de taille j_n_melvecs * i_melvec_length

    """
    # Calculer les spectrogrammes
    a_spec = []
    size_a = len(signal)
    cropped_a_spec = signal[:size_a - size_a % Nft]  # Tronquer le signal pour qu'il soit un multiple de 512
    for m in range(0, len(cropped_a_spec), Nft):
        temp_spec = np.log(np.abs(
            melspecgram(cropped_a_spec[m:m + Nft], Nmel=j_n_melvecs, mellength=i_melvec_length, fs_down=fs_down)))
        a_spec.append(temp_spec)

    a_spec_reshaped = []
    for k in range(0, len(a_spec) - (len(a_spec) % j_n_melvecs),
                   j_n_melvecs):  # Avancer par pas de j et tronquer le nombre de melvecs pour avoir un multiple de j
        a_spec_reshaped.append(np.ravel(a_spec[
                                        k:k + j_n_melvecs]))  # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

    return a_spec_reshaped


def audio_signal_troncated(signal, i_melvec_length, j_n_melvecs, Nft=512, fs_down=11111):
    a_spec = []
    size_a = len(signal)
    cropped_a_spec = signal[
                     :size_a - size_a % Nft * i_melvec_length]  # Tronquer le signal pour qu'il soit un multiple de 512*20
    for m in range(0, len(cropped_a_spec), Nft * i_melvec_length):
        a_spec.append(cropped_a_spec[m:m + Nft * i_melvec_length])

    return a_spec

def add_lol_effects_and_specgram(signal, i_melvec_length, j_n_melvecs, Nft=512, fs_down=11111):
    array = signal
    array = remove_dc_component(array)
    array = array * np.random.uniform(0.1, 3)  # amplitude
    echo = np.zeros(len(array))
    # TODO : vérifier que c'est bien j_n_melvecs et pas i_melvec_length
    echo_sig = np.zeros(512 * j_n_melvecs) # Même taille que les fichiers audio de la base de données
    echo_sig[0] = 1
    n_echo = np.random.randint(1, 3)
    echo_sig[(np.arange(1) / 1 * 512 * j_n_melvecs).astype(int)] = (1 / 2) ** np.arange(1) #TODO : vérifier que c'est bien j_n_melvecs et pas i_melvec_length

    # print("len array avant convolve", len(array))
    array = fftconvolve(array, echo_sig, mode="full")
    "Tronquer le signal car il est plus long après la convolution"
    # print("len array après convolve", len(array))
    array = array[:512 * i_melvec_length] #TODO : vérifier que c'est bien j_n_melvecs et pas i_melvec_length
    # print("len array après troncage", len(array))
    array = array + np.random.normal(0, np.random.uniform(0.05, 0.7), len(array))  # noise
    sound_to_add = data1_list[np.random.randint(0, len(data1_list))].astype(np.float32)
    sound_to_add = sound_to_add - np.mean(sound_to_add)
    # print("len sound_to_add", len(sound_to_add))
    array = array + sound_to_add * np.random.uniform(0, 0.7) / np.max(sound_to_add) * np.max(array)

    x = array - np.mean(array)  # Moyenne quasi nulle donc ligne superflue
    x = x / np.linalg.norm(x)

    #print("len x", len(x))  # Taille de chaque petit signal sonore

    spec = np.ravel(np.log(np.abs(melspecgram(x, Nmel=j_n_melvecs, mellength=i_melvec_length, fs_down=fs_down))))
    #print("len spec", len(spec))  # Taille du spectrogramme
    return (spec - np.mean(spec))


def convert_x_spec_to_regular_array(x_spec):
    temp = []
    # for i in range(len(x_spec)):
    #     for j in range(len(x_spec[i])):
    #         temp[i][j] = x_spec[i][j]

    for i in range(len(x_spec)):
        temp.append(np.ravel(x_spec[i]))
        print(temp[i])

    return temp

a = pickle.load(open("data/raw_global_samples/birds_reformated_globalsample.pickle", "rb"))
b = pickle.load(open("data/raw_global_samples/fire_reformated_globalsample.pickle", "rb"))
c = pickle.load(open("data/raw_global_samples/handsaw_reformated_globalsample.pickle", "rb"))
d = pickle.load(open("data/raw_global_samples/chainsaw_reformated_globalsample.pickle", "rb"))
e = pickle.load(open("data/raw_global_samples/helicopter_reformated_globalsample.pickle", "rb"))

# Retirer la composante continue de chaque signal pour chaque classe
a_signal_without_DC_component = remove_dc_component(a[1])
b_signal_without_DC_component = remove_dc_component(b[1])
c_signal_without_DC_component = remove_dc_component(c[1])
d_signal_without_DC_component = remove_dc_component(d[1])
e_signal_without_DC_component = remove_dc_component(e[1])

print("Classes in the dataset:", ["birds", "fire", "handsaw", "chainsaw", "helicopter"])

N_MELVECS_begin = 16
N_MELVECS_end = 24
N_MELVECS_step = 2

MELVEC_LENGTH_begin = 16 # au départ on avait mis 20
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
        print("Nouvelle boucle pour : MELVEC_LENGTH = {}, N_MELVECS = {}".format(i, j))

        """
        "Ancienne méthode sans fonction (conservée ici au cas où)"
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
        """

        """
        "Utiliser ça si on ne rajoute aucun effet aux signaux sonores"
        a_spec_reshaped = turbo_lol_spectrograms(a_signal_without_DC_component, i, j, Nft=512, fs_down=fs_down)
        b_spec_reshaped = turbo_lol_spectrograms(b_signal_without_DC_component, i, j, Nft=512, fs_down=fs_down)
        c_spec_reshaped = turbo_lol_spectrograms(c_signal_without_DC_component, i, j, Nft=512, fs_down=fs_down)
        d_spec_reshaped = turbo_lol_spectrograms(d_signal_without_DC_component, i, j, Nft=512, fs_down=fs_down)
        e_spec_reshaped = turbo_lol_spectrograms(e_signal_without_DC_component, i, j, Nft=512, fs_down=fs_down)

        data1_list = a_spec_reshaped
        data1_list = np.concatenate((data1_list, b_spec_reshaped))
        data1_list = np.concatenate((data1_list, c_spec_reshaped))
        data1_list = np.concatenate((data1_list, d_spec_reshaped))
        data1_list = np.concatenate((data1_list, e_spec_reshaped))
        """


        # Zéphyrin ne prenait pas les signaux avec la composante continue enlevée
        # a_spec = convert_x_spec_to_regular_array(audio_signal_troncated(a[1], i, j, Nft=512, fs_down=fs_down))
        # b_spec = convert_x_spec_to_regular_array(audio_signal_troncated(b[1], i, j, Nft=512, fs_down=fs_down))
        # c_spec = convert_x_spec_to_regular_array(audio_signal_troncated(c[1], i, j, Nft=512, fs_down=fs_down))
        # d_spec = convert_x_spec_to_regular_array(audio_signal_troncated(d[1], i, j, Nft=512, fs_down=fs_down))
        # e_spec = convert_x_spec_to_regular_array(audio_signal_troncated(e[1], i, j, Nft=512, fs_down=fs_down))


        a_spec = audio_signal_troncated(a[1], i, j, Nft=512, fs_down=fs_down)
        b_spec = audio_signal_troncated(b[1], i, j, Nft=512, fs_down=fs_down)
        c_spec = audio_signal_troncated(c[1], i, j, Nft=512, fs_down=fs_down)
        d_spec = audio_signal_troncated(d[1], i, j, Nft=512, fs_down=fs_down)
        e_spec = audio_signal_troncated(e[1], i, j, Nft=512, fs_down=fs_down)

        if len(a_spec) != len(b_spec) != len(c_spec) != len(d_spec) != len(e_spec):
            raise ValueError("Dimensions des tableaux différentes")

        sub_size = len(a_spec[0])
        if len(a_spec[-1]) != sub_size:
            a_spec.pop(-1)

        sub_size = len(b_spec[0])
        if len(b_spec[-1]) != sub_size:
            b_spec.pop(-1)

        sub_size = len(c_spec[0])
        if len(c_spec[-1]) != sub_size:
            c_spec.pop(-1)

        sub_size = len(d_spec[0])
        if len(d_spec[-1]) != sub_size:
            d_spec.pop(-1)

        sub_size = len(e_spec[0])
        if len(e_spec[-1]) != sub_size:
            e_spec.pop(-1)

        data1_list = np.concatenate((a_spec, b_spec, c_spec, d_spec, e_spec))

        # print("len data1_list", len(data1_list))

        "Création de data2_list avec tous les labels"
        data2_list = []
        labels = ["birds", "fire", "handsaw", "chainsaw", "helicopter"]
        for label in range(0, 5):
            data2_list = np.concatenate((data2_list, [labels[label] for _ in range(len(a_spec))]), axis=0)
            # en partant du principe que len(a_spec_reshaped) = len(b_spec_reshaped) = len(c_spec_reshaped) = len(d_spec_reshaped) = len(e_spec_reshaped)

        # print("Data2_list longeur", len(data2_list))
        # print("Longueur de tous les spectrogrammes : ", len(data1_list))
        # print("Nombre d'éléments dans le premier spectrogramme", len(data1_list[0]))

        data1_list = np.array(data1_list)
        data2_list = np.array(data2_list)

        # print("len data1_list", len(data1_list))
        # print("len data2_list", len(data2_list))

        temp_acc_split = [] # une accuracy par split
        n_splits = 1
        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(data1_list, data2_list, test_size=0.3)
            # print("Taille de X_test et de y_test : ", len(X_test), len(y_test))
            # print("Taille de X_train et de y_train : ", len(X_train), len(y_train))
            X_val_spec = []
            y_val = y_test
            X_train_spec = []

            "Ajouter les effets LOL aux signaux sonores de test puis en calculer les spectrogrammes"
            for test_elem in range(len(X_test)):
                X_val_spec.append(add_lol_effects_and_specgram(X_test[test_elem].astype(np.float32), i_melvec_length=i, j_n_melvecs=j, Nft=512, fs_down=fs_down))

            "Spectrogrammes des signaux sonores d'entraînement"
            for train_elem in range(len(X_train)):
                x = X_train[train_elem] - np.mean(X_train[train_elem])

                # Normalisation (Zéphyrin ne l'a pas fait mais c'est fait pour le dataset de test)
                x = x / np.linalg.norm(x)
                spec = np.ravel(np.log(np.abs(melspecgram(x, Nmel=j, mellength=i, fs_down=fs_down))))
                X_train_spec.append(spec - np.mean(spec))

            "Modèle et PCA"
            model = RandomForestClassifier(n_estimators=100)
            pca = PCA(n_components=29, whiten=True)

            "Entraîner le modèle et la PCA"
            X_learn_reduced = pca.fit_transform(np.array(X_train_spec))
            X_val_reduced = pca.transform(np.array(X_val_spec))

            for reduced_elem in range(len(X_val_reduced)):
                X_val_reduced[reduced_elem] = X_val_reduced[reduced_elem] / np.linalg.norm(X_val_reduced[reduced_elem])

            model.fit(X_learn_reduced, y_train)
            prediction = model.predict(X_val_reduced)

            temp_acc_split.append(accuracy(prediction, y_val))

            """
            print(prediction)
            print("len prediction", len(prediction))
            print("y_val", y_val)
            print("computed accuracy", accuracy(prediction, y_val)) # rappel : y_val = y_test
            """

        mean_acc = np.mean(temp_acc_split)
        accuracy_matrix[(i - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step][(j - N_MELVECS_begin) // N_MELVECS_step] = mean_acc

        print("MELVEC_LENGTH : {}, N_MELVECS : {}, mean accuracy of the {}-splits : {}".format(i, j, n_splits,100 * mean_acc))
        print("=====================================")

plt.imshow(accuracy_matrix, cmap='hot', extent=np.concatenate((N_MELVECS_arange, MELVEC_LENGTH_arange)))
plt.colorbar()
plt.xlabel("N_MELVECS")
plt.ylabel("MELVEC_LENGTH")
plt.savefig("accuracy_matrix.png")
plt.show()

"""
        "========================================="
        "Ancien code"
        "Labels"
        X_train, X_test, y_train, y_test = train_test_split(data1_list, data2_list, test_size=0.3, stratify=data2_list)
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
            X_learn_normalised = imp.transform(X_learn_normalised)  # Remplacer tous les NaN par la moyenne
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
        accuracy_matrix[(i - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step][
            (j - N_MELVECS_begin) // N_MELVECS_step] = temp_mean
        std_matrix[(i - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step][(j - N_MELVECS_begin) // N_MELVECS_step] = temp_std

        print(
            "N_MELVECS : {}, MELVEC_LENGTH : {}, accuracy : {}, std : {}".format(i, j, 100 * temp_mean, 100 * temp_std))
        print("=====================================")

# ax = sns.heatmap(accuracy_matrix, linewidth=0.5)
# ax2 = sns.heatmap(std_matrix, linewidth=0.5)
# plt.show()

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
"""