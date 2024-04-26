# -*- coding: utf-8 -*-
"""
uart-reader.py
ELEC PROJECT - 210x
"""

import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from classification.utils.utils import accuracy

import librosa
import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from classification.utils.plots import plot_specgram

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20  # hauteur (longueur de chaque vecteur)
N_MELVECS = 20 # Nombre de vecteurs mel

fs_down_old = 11025  # Target sampling frequency
fs_down = 11111.11

dt = np.dtype(np.uint16).newbyteorder("<")

"""

INIT CLASSIFICATION

"""

"""
model_knn = pickle.load(
    open("C:/Users/valer/Documents/Master1/Projetmaster/classification/data/models/random_forest_Q1_parameters.pickle",
         'rb'))  # Write your path to the model here!

normalize = True
pca = pickle.load(
    open("C:/Users/valer/Documents/Master1/Projetmaster/classification/data/models/pca_Q1_parameters", 'rb'))
"""

def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    else:
        print(line)
        return None


def remove_dc_component(signal):
    mean_value = np.mean(signal)  # Calculer la moyenne du signal
    return signal - mean_value


def specgram(y, Nft=512):
    """Build the spectrogram of a downsample and filtered (low-pass to avoid aliasing) sound record.

    Args:
      y (array of float): sound record after filtering (low pass filter to avoid aliasing) and downsampling.
      Nft (int): number of sample by fft

    Returns:
      stft (array of float): short time fourier transform
    """

    # Homemade computation of stft
    "Crop the signal such that its length is a multiple of Nft"
    L = len(y)
    y = y[: MELVEC_LENGTH * Nft]
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


def melspecgram(x, Nmel=N_MELVECS, Nft=512, fs=44100, fs_down=11025):
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


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=int(2e6))
    while True:
        line = ""
        while not line.endswith("\n"):
            # SAMPLE_PER_MELVEC = 512 (voir config.c)
            line += ser.read_until(b"\n", size=512).decode(
                "ascii"
            )
            # melcalcul(line)
            # print(line) # print the line in hex
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


sound_loaded = []
classe_loaded = []
sound_record = []
classe_record = []


def add_to_save_data(data, classe):
    sound_record.append(data)
    classe_record.append(classe)


def save():
    x = input("select the file name >>")
    filename = 'x'
    pickle.dump((sound_record, classe_record), open("./" + filename, 'wb'))


def load():
    global sound_loaded, classe_loaded
    b = input("name the file>>")
    a = pickle.load(open("./" + b + ".pickle", 'rb'))
    print(a)
    return a


def train():
    """X , y = load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y) # random_state=1
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits,shuffle=True)

    model_knn = RandomForestClassifier()
    #model_knn = KNeighborsClassifier(n_neighbors=9)
    accuracy_knn = np.zeros((n_splits,))
    for k, idx in enumerate(kf.split(X_train,y_train)):
        (idx_learn, idx_val) = idx

        # [2] (optional) Data normalization
        X_learn_normalised = X_train[idx_learn]/ np.linalg.norm(X_train[idx_learn], axis=1, keepdims=True)
        #print(len(X_learn_normalised))
        X_val_normalised = X_train[idx_val]/ np.linalg.norm(X_train[idx_val], axis=1, keepdims=True)
        #print(len(X_val_normalised))
        # [3] (optional) dimensionality reduction.
        n=8 # Number of principal components kept
        pca = PCA(n_components=n,whiten=True)
        X_learn_reduced = pca.fit_transform(X_learn_normalised)
        X_val_reduced = pca.transform(X_val_normalised)
        pca.transform([X_val_normalised[0]])
        model_knn.fit(X_learn_reduced, y_train[idx_learn])
        prediction_knn = model_knn.predict(X_val_reduced)
        #print(len(prediction_knn))
        accuracy_knn[k] = accuracy(prediction_knn, y_train[idx_val])
        #show_confusion_matrix(prediction_knn, y_train[idx_val], classnames) #add package
    print('Mean accuracy of KNN with 5-Fold CV: {:.1f}%'.format(100*accuracy_knn.mean()))
    print('Std deviation in accuracy of KNN with 5-Fold CV: {:.1f}%'.format(100*accuracy_knn.std()))"""

    # [4] Model training and selection.

    # [5] Save the trained model, eventually the pca.
    """filename = 'model.pickle'
    pickle.dump(model_knn, open("./"+filename, 'wb'))
    pickle.dump(pca, open("./"+"pca", 'wb'))"""
    print("a")


def micro_model():
    global model_knn, normalize, pca
    model_knn = pickle.load(open("./model.pickle", 'rb'))  # Write your path to the model here!

    normalize = True
    pca = pickle.load(open("./pca", 'rb'))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        input_stream = reader(port=args.port)
        msg_counter = 0
        # plt.figure()
        i = 0

        global_sample = []

        for sample in input_stream:
            # 260 paquets envoyés par le MCU en taille 512 * 40
            # Jouer avec les paramètres Nft et N_MELVECS tant que leur produit est inférieur à 512 * 40

            number_of_packets = 260  # A changer en fonction du code du MCU (config.h)
            # Dans les faits, il y a peut-être 259 paquets envoyés

            msg_counter += 1

            print("--------------------")
            print("Échantillon #{}".format(msg_counter))
            print("Taille de l'échantillon {} reçu : {}".format(msg_counter, len(sample)))


            global_sample.append(sample)

            taille = 0
            for i in range(len(global_sample)):
                taille += len(global_sample[i])

            print("Taille de global_sample", taille)
            print("--------------------")

            if msg_counter == number_of_packets:
                pickle.dump(["helicopter", global_sample], open("./helicopter_globalsample.pickle", 'wb'))
                print("Fichier enregistré")

                break

        print("Fin de l'enregistrement")

        print("Fin du programme")
        exit()

        for sample in input_stream:
            break  # Ne pas oublier d'enlever
            # 129 paquets envoyés par le MCU en taille 512 * 40
            # Jouer avec les paramètres Nft et N_MELVECS tant que leur produit est inférieur à 512 * 40
            msg_counter += 1

            print(sample)
            print("Taille du sample reçu", len(sample))

            # Calcul du spectrogramme mel avec la fonction melcalcul / melspecgram
            signal_without_DC_component = remove_dc_component(sample)
            spectrogram = np.log(np.abs(melspecgram(signal_without_DC_component, fs_down=10980)))
            #print(spectrogram)
            print("Dimensions du spectrogramme mel :", "({},{})".format(len(spectrogram), len(spectrogram[0])))

            """ A décommenter si le MCU envoie des spectrogrammes (et pas les données brutes)        
            # print("MEL Spectrogram #{}".format(msg_counter))
            sgram = sample.reshape((N_MELVECS, MELVEC_LENGTH)).T
            ncol = int(1000 * 10200 / (1e3 * 512))
            sgram = sgram[:, :ncol]
            sgram = sgram / np.linalg.norm(sgram, keepdims=True)  # normalisation
            sgram = np.nan_to_num(sgram, nan=1e-16)
            fv = sgram.reshape(-1)  # spectrogramme mis en une seule ligne
            # print("must be 400 or change code")
            # print(len(fv))"""

            ### TO COMPLETE - Eventually normalize and reduce feature vector dimensionality

            ''' Sert un peu rien j'ai l'impression... Ou alors c'est caduc
            try:
                fv = pca.transform([fv[:400]])
                probs = model_knn.predict(fv)[0]
                # print(probs)
            except:
                print("hum")
            '''

            # Enregistre chaque spectrogramme dans deux tableaux (un tableau pour les données et un tableau pour les classes)
            # add_to_save_data(fv, "helicopter")
            print("added")

            """ A decommenter pour enregistrer les données dans un fichier pickle au bout de 120 spectrogrammes reçus
            # Au bout de 120 spectrogrammes, enregistre les données dans un fichier pickle.
            if i == 120:
                pickle.dump([sound_record, classe_record], open("./helicopter.pickle", 'wb'))
                print("Fichier enregistré")
            i += 1"""

            # print(spectrogram.reshape((N_MELVECS, MELVEC_LENGTH)).T)

            print("Temps de l'enregistrement :", len(sample) / fs_down, "s")

            # J'ai enlevé la transposée pour que le spectrogramme soit dans le bon sens : sans doute alors qu'on a
            # échangé NMEL et MELVEC car en augmentant NMEL, on augmente la hauteur du spectrogramme alors que ça devrait être l'abscisse
            plot_specgram(spectrogram.reshape((N_MELVECS, MELVEC_LENGTH)), ax=plt.gca(), is_mel=True,
                          title="MEL Spectrogram #{}".format(msg_counter), xlabel="Temps (s)",
                          tf=len(sample) / fs_down)
            plt.draw()
            plt.pause(0.05)
            plt.clf()
