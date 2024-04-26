import numpy as np
import matplotlib.pyplot as plt

"Machine learning tools"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
import pickle
import numpy as np
import librosa
from sklearn.impute import SimpleImputer
import copy

from classification.utils.utils import accuracy

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

    mels = librosa.filters.mel(sr=11111, n_fft=512, n_mels=20)
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

    stft = specgram(x, mellength=20, Nft=512)

    """print("=====================================")
    print("Fonction de calcul du mel spectrogamme :")
    print("Dimensions de la matrice de transfo (la matrice est correcte) :", "({},{})".format(len(mels), len(mels[0])),
          "Dimensions de la matrice de stft :", "({},{})".format(len(stft), len(stft[0])))
    """
    melspec = np.dot(mels[:,:-1], stft)
    return melspec


a = pickle.load(open("data/raw_global_samples/birds_reformated_globalsample.pickle", "rb"))
b = pickle.load(open("data/raw_global_samples/fire_reformated_globalsample.pickle", "rb"))
c = pickle.load(open("data/raw_global_samples/handsaw_reformated_globalsample.pickle", "rb"))
d = pickle.load(open("data/raw_global_samples/chainsaw_reformated_globalsample.pickle", "rb"))
e = pickle.load(open("data/raw_global_samples/helicopter_reformated_globalsample.pickle", "rb"))

print("Classes in the dataset:", ["birds", "fire", "handsaw", "chainsaw", "helicopter"])

pca_begin = 1
pca_end = 50
pca_step = 1

pca_arange = np.arange(pca_begin,pca_end,pca_step)
accuracy_matrix = np.zeros(len(pca_arange))
std_matrix = np.zeros(len(pca_arange))

a_signal_without_DC_component = remove_dc_component(a[1])
b_signal_without_DC_component = remove_dc_component(b[1])
c_signal_without_DC_component = remove_dc_component(c[1])
d_signal_without_DC_component = remove_dc_component(d[1])
e_signal_without_DC_component = remove_dc_component(e[1])

data1_list = [] # Liste contenant les spectrogrammes de toutes les classes
a_spec = []
size_a = len(a_signal_without_DC_component)
cropped_a_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_a_spec), 512):
    a_spec.append(cropped_a_spec[m:m + 512])

data1_list = (a_spec)
b_spec = []
size_a = len(a_signal_without_DC_component)
cropped_b_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_b_spec), 512):
    b_spec.append(cropped_b_spec[m:m + 512])

data1_list = np.concatenate((data1_list,b_spec), axis=0)

c_spec = []
size_a = len(a_signal_without_DC_component)
cropped_c_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_c_spec), 512):
    c_spec.append(cropped_c_spec[m:m + 512])

data1_list = np.concatenate((data1_list,c_spec), axis=0)

d_spec = []
size_a = len(a_signal_without_DC_component)
cropped_d_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_d_spec), 512):
    d_spec.append(cropped_d_spec[m:m + 512])

data1_list = np.concatenate((data1_list,d_spec), axis=0)

e_spec = []
size_a = len(a_signal_without_DC_component)
cropped_e_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_e_spec), 512):
    e_spec.append(cropped_e_spec[m:m + 512])

data1_list = np.concatenate((data1_list,e_spec), axis=0)

"Création de data2_list avec tous les labels"
data2_list = []
labels = ["birds", "fire", "handsaw", "chainsaw", "helicopter"]
for label in range(0, 5):
    data2_list = np.concatenate((data2_list, [labels[label] for _ in range(len(a_spec))]), axis=0)
# en partant du principe que len(a_spec_reshaped) = len(b_spec_reshaped) = len(c_spec_reshaped) = len(d_spec_reshaped) = len(e_spec_reshaped)
data1_list = np.array(data1_list)
data2_list = np.array(data2_list)

print(data1_list.shape, data2_list.shape)

X_train, X_test, y_train, y_test = train_test_split(data1_list, data2_list, test_size=0.3, stratify=data2_list)  # random_state=1
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
for k, idx in enumerate(kf.split(X_train, y_train)):
    (idx_learn, idx_val) = idx
    X_val = copy.copy(data1_list[idx_val])
    y_val = data2_list[idx_val]
    X_train = data1_list[idx_learn]
    X_val = data2_list[idx_learn]
    #ADD DEFECT
    for i in range(len(X_val)):
        X_val[i] = X_val[i]*np.random.uniform(0.6,3)   #amplitude
        X_val[i] =X_val[i] + np.random.normal(0, np.random.random(), len(X_val[i]))   #noise
        X_val[i] =data1_list[np.random.randint(0,len(data1_list))]*np.random.random()   #background
        echo = np.zeros(len(X_val[i]))
        debut = np.random.randint(10,500)
        factor = np.random.uniform(0.5,1.0)
        for j in range(debut, len(X_val[i])):
            X_val[i][j] =X_val[i][j] + X_val[i][j-debut]*factor #echo
    





    """for i in range(pca_begin,pca_end, pca_step):
        

        print("Nouvelle boucle pour : N_Componenet = {}".format(i))

        n_trees = 100
        model = RandomForestClassifier(n_trees)
        accuracy = np.zeros((n_splits,))

        # best pca = PCA(n_components=5,whiten=True)with n=7+1=8
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        pca = PCA(n_components=i, whiten=True)

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
        model.fit(X_learn_reduced, y_train_u[idx_learn])
        prediction = model.predict(X_val_reduced)
        # print(len(prediction_knn))
        accuracy[k] = compute_accuracy(prediction, y_train_u)

        # accuracy est un tableau avec 5 élements car il y a 5 plis pour la validation croisée
        # accuracy.mean() est la moyenne des 5 élements
        # accuracy.std() est l'écart-type des 5 élements
    temp_mean = np.mean(accuracy)
    temp_std = np.std(accuracy)

    # Remplir les matrices globales
    accuracy_matrix[i-pca_begin] = temp_mean
    std_matrix[i-pca_begin] = temp_std

    print("N_Componenet : {}, accuracy : {}, std : {}".format(i, 100 * temp_mean, 100 * temp_std))
    print("=====================================")

plt.plot(pca_arange, accuracy_matrix, label="accuracy")
plt.show()
plt.plot(pca_arange, std_matrix, label="std")
plt.show()














# Calculer les spectrogrammes
a_spec = []
size_a = len(a_signal_without_DC_component)
cropped_a_spec = a_signal_without_DC_component[:size_a - size_a % 512] # Tronquer le signal pour qu'il soit un multiple de 512
for m in range(0, len(cropped_a_spec), 512):
    temp_spec = np.log(np.abs(melspecgram(cropped_a_spec[m:m + 512], Nmel=20, mellength=20, fs_down=fs_down)))
    a_spec.append(temp_spec)

a_spec_reshaped = []
for k in range(0, len(a_spec) - (len(a_spec) % 20), 20): # Avancer par pas de j et tronquer le nombre de melvecs pour avoir un multiple de j
    a_spec_reshaped.append(np.ravel(a_spec[k:k+20])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

data1_list = a_spec_reshaped # Pour la suite, il faudra concaténer les autres classes à la suite de cette liste


print("Spectrogrammes oiseau calculés !")


b_spec = []
size_b = len(b_signal_without_DC_component)
cropped_b_spec = b_signal_without_DC_component[:size_b - size_b % 512]
for m in range(0, len(cropped_b_spec), 512):
    temp_spec = np.log(np.abs(melspecgram(cropped_b_spec[m:m + 512], Nmel=20, mellength=20, fs_down=fs_down)))
    b_spec.append(temp_spec)

b_spec_reshaped = []
for k in range(0, len(b_spec) - (len(b_spec) % 20), 20): # Avancer par pas de j
    b_spec_reshaped.append(np.ravel(b_spec[k:k+20])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

data1_list = np.concatenate((data1_list, b_spec_reshaped))

print("Spectrogrammes feu calculés !")


c_spec = []
size_c = len(c_signal_without_DC_component)
cropped_c_spec = c_signal_without_DC_component[:size_c - size_c % 512]
for m in range(0, len(cropped_c_spec), 512):
    temp_spec = np.log(np.abs(melspecgram(cropped_c_spec[m:m + 512], Nmel=20, mellength=20, fs_down=fs_down)))
    c_spec.append(temp_spec)

c_spec_reshaped = []
for k in range(0, len(c_spec) - (len(c_spec) % 20), 20): # Avancer par pas de j
    c_spec_reshaped.append(np.ravel(c_spec[k:k+20])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

data1_list = np.concatenate((data1_list, c_spec_reshaped), axis=0)

print("Spectrogrammes scie calculés !")


d_spec = []
size_d = len(d_signal_without_DC_component)
cropped_d_spec = d_signal_without_DC_component[:size_d - size_d % 512]
for m in range(0, len(cropped_d_spec), 512):
    temp_spec = np.log(np.abs(melspecgram(cropped_d_spec[m:m + 512], Nmel=20, mellength=20, fs_down=fs_down)))
    d_spec.append(temp_spec)

d_spec_reshaped = []
for k in range(0, len(d_spec) - (len(d_spec) % 20), 20): # Avancer par pas de j
    d_spec_reshaped.append(np.ravel(d_spec[k:k+20])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

data1_list = np.concatenate((data1_list, d_spec_reshaped), axis=0)

print("Spectrogrammes tronçonneuse calculés !")


e_spec = []
size_e = len(e_signal_without_DC_component)
cropped_e_spec = e_signal_without_DC_component[:size_e - size_e % 512]
for m in range(0, len(cropped_e_spec), 512):
    temp_spec = np.log(np.abs(melspecgram(cropped_e_spec[m:m + 512], Nmel=20, mellength=20, fs_down=fs_down)))
    e_spec.append(temp_spec)

e_spec_reshaped = []
for k in range(0, len(e_spec) - (len(e_spec) % 20), 20): # Avancer par pas de j
    e_spec_reshaped.append(np.ravel(e_spec[k:k+20])) # Prendre des paquets de j spectrogrammes et les applatir en un seul vecteur de dimension 1

data1_list = np.concatenate((data1_list, e_spec_reshaped), axis=0)

print("Spectrogrammes hélicoptère calculés !")"""