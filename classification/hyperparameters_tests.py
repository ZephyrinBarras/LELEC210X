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

from classification.utils.plots import plot_specgram, show_confusion_matrix, show_confusion_matrix_with_save, \
    plot_decision_boundaries
from classification.utils.utils import accuracy
from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS

from classification.utils.plots import plot_specgram, show_confusion_matrix, plot_decision_boundaries
from classification.datasets import Dataset

MELVEC_LENGTH_DEFAULT = 20  # hauteur (longueur de chaque vecteur)
N_MELVECS_DEFAULT = 20  # Nombre de vecteurs mel
fs_down = 11025  # Target sampling frequency

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
    y = y[: mellength * Nft]
    # y = y[: L - L % Nft]
    L = len(y)
    print("Taille de y :", L)

    "Reshape the signal with a piece for each row"
    audiomat = np.reshape(y, (L // Nft, Nft))
    audioham = audiomat * np.hamming(Nft)  # Windowing. Hamming, Hanning, Blackman,..
    z = np.reshape(audioham, -1)  # y windowed by pieces
    "FFT row by row"
    stft = np.fft.fft(audioham, axis=1)
    stft = np.abs(
        stft[:, : Nft // 2].T
    )  # Taking only positive frequencies and computing the magnitude

    return stft


def melspecgram(x, Nmel=N_MELVECS_DEFAULT, Nft=512, fs=44100, fs_down=11025):
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

    mels = librosa.filters.mel(sr=fs_down, n_fft=Nft, n_mels=Nmel)
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

    stft = specgram(x, Nft=Nft)
    print("=====================================")
    print("Fonction de calcul du mel spectrogamme :")
    print("Dimensions de la matrice de transfo (la matrice est correcte) :", "({},{})".format(len(mels), len(mels[0])),
          "Dimensions de la matrice de stft :", "({},{})".format(len(stft), len(stft[0])))
    melspec = np.dot(mels, stft)
    return melspec


dataset = Dataset("birds_globalsample.pickle")
classnames = dataset.list_classes()

print("Classes in the dataset:", classnames)

# abss = np.arange(256, 2560, 256)  # ICI LA RANGE DU PARAM NFT OU FREQ

N_MELVECS_begin = 10
N_MELVECS_end = 100
N_MELVECS_step = 2

MELVEC_LENGTH_begin = 10
MELVEC_LENGTH_end = 100
MELVEC_LENGTH_step = 2


N_MELVECS_arange = np.arange(N_MELVECS_begin, N_MELVECS_end, N_MELVECS_step)  # ICI LA RANGE DU PARAM NMEL
MELVEC_LENGTH_arange = np.arange(MELVEC_LENGTH_begin, MELVEC_LENGTH_end, MELVEC_LENGTH_step)  # ICI LA RANGE DU PARAM MELVEC_LENGTH

accuracy_matrix = np.zeros((len(N_MELVECS_arange), len(MELVEC_LENGTH_arange)))
std_matrix = np.zeros((len(N_MELVECS_arange), len(MELVEC_LENGTH_arange)))

for i in N_MELVECS_arange:
    for j in MELVEC_LENGTH_arange:
        #TODO calculer les spectrogrammes à partir d'ici

        signal_without_DC_component = remove_dc_component(sample)
        spectrogram = np.log(np.abs(melspecgram(signal_without_DC_component, fs_down=10980)))

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

        "Labels"
        y = class_ids_aug.copy()
        # X_train, X_test, y_train, y_test = ...
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)  # random_state=1
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

        # Remplir les matrices globales
        accuracy_matrix[(i - N_MELVECS_begin) // N_MELVECS_step][(j - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step] = temp_mean
        std_matrix[(i - N_MELVECS_begin) // N_MELVECS_step][(j - MELVEC_LENGTH_begin) // MELVEC_LENGTH_step] = temp_std

        print("N_MELVECS : {}, MELVEC_LENGTH : {}, accuracy : {}, std : {}".format(i, j, 100 * temp_mean, 100 * temp_std))


ax = sns.heatmap(accuracy_matrix, linewidth=0.5)
ax2 = sns.heatmap(std_matrix, linewidth=0.5)
plt.show()


