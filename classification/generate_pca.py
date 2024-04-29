import numpy as np
import matplotlib.pyplot as plt
import time

"Machine learning tools"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
import pickle
import numpy as np
import librosa
import sounddevice as sd
from sklearn.impute import SimpleImputer
import copy
from scipy.signal import fftconvolve, resample

from classification.utils.utils import accuracy

MELVEC_LENGTH_DEFAULT = 20  # hauteur (longueur de chaque vecteur)
N_MELVECS_DEFAULT = 20  # Nombre de vecteurs mel
fs_down = 11111  # Target sampling frequency
import numpy as np
import matplotlib.pyplot as plt

# Fréquence de la sinusoïde en Hz
frequency = 1000

# Fréquence d'échantillonnage en Hz
sampling_rate = 11111

# Nombre d'échantillons
num_samples = 512 * 20 *50

# Temps d'échantillonnage
t = np.arange(num_samples) / sampling_rate

# Générer le signal sinusoïdal
signal = 32767*np.sin(2 * np.pi * frequency * t)
signal = np.reshape(signal,(50,512*20))
signal = signal.astype(np.int16)
print(signal)




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



a_signal_without_DC_component = a[1]
b_signal_without_DC_component = b[1]
c_signal_without_DC_component = c[1]
d_signal_without_DC_component = d[1]
e_signal_without_DC_component = e[1]

data1_list = [] # Liste contenant les spectrogrammes de toutes les classes
a_spec = []
size_a = len(a_signal_without_DC_component)
cropped_a_spec = a_signal_without_DC_component[:size_a - size_a % 512*20] # Tronquer le signal pour qu'il soit un multiple de 512*20
for m in range(0, len(cropped_a_spec), 512*20):
    a_spec.append(cropped_a_spec[m:m + 512*20])
data1_list = (a_spec)
b_spec = []
size_b = len(b_signal_without_DC_component)
cropped_b_spec = b_signal_without_DC_component[:size_b - size_b % 512*20] # Tronquer le signal pour qu'il soit un multiple de 512*20
for m in range(0, len(cropped_b_spec), 512*20):
    b_spec.append(cropped_b_spec[m:m + 512*20])

data1_list = np.concatenate((data1_list,b_spec), axis=0)

c_spec = []
size_c = len(c_signal_without_DC_component)
cropped_c_spec = c_signal_without_DC_component[:size_c - size_c % 512*20] # Tronquer le signal pour qu'il soit un multiple de 512*20
for m in range(0, len(cropped_c_spec), 512*20):
    c_spec.append(cropped_c_spec[m:m + 512*20])

data1_list = np.concatenate((data1_list,c_spec), axis=0)

d_spec = []
size_d = len(d_signal_without_DC_component)
cropped_d_spec = d_signal_without_DC_component[:size_d - size_d % 512*20] # Tronquer le signal pour qu'il soit un multiple de 512*20
for m in range(0, len(cropped_d_spec), 512*20):
    d_spec.append(cropped_d_spec[m:m + 512*20])

data1_list = np.concatenate((data1_list,d_spec), axis=0)

e_spec = []
size_e = len(e_signal_without_DC_component)
cropped_e_spec = e_signal_without_DC_component[:size_e - size_e % 512*20] # Tronquer le signal pour qu'il soit un multiple de 512*20
for m in range(0, len(cropped_e_spec), 512*20):
    e_spec.append(cropped_e_spec[m:m + 512*20])

data1_list = np.concatenate((data1_list,e_spec), axis=0)

"Création de data2_list avec tous les labels"
data2_list = []
labels = ["birds", "fire", "handsaw", "chainsaw", "helicopter"]
for label in range(0, 5):
    data2_list = np.concatenate((data2_list, [labels[label] for _ in range(len(e_spec))]), axis=0)
# en partant du principe que len(a_spec_reshaped) = len(b_spec_reshaped) = len(c_spec_reshaped) = len(d_spec_reshaped) = len(e_spec_reshaped)
data2_list = list(data2_list)
data1_list = np.array(data1_list)
data2_list = np.array(data2_list)

X_train_spec = []
for i in range(len(data1_list)):
    x = data1_list[i]-np.mean(data1_list[i])  
    x=x/np.linalg.norm(x)      
    spec = np.ravel(melspecgram(x, 20, 20,512, fs_down, fs_down=fs_down))
    
    X_train_spec.append(spec-np.mean(spec))
np.set_printoptions(threshold=np.inf)

pca = PCA(n_components=29, whiten=True)
pca.fit(np.array(X_train_spec))
components = pca.components_
new_compo = np.zeros((29,400))
for i in range(len(components)):
    for j in range(20):
        mean = np.mean(components[i][j*20:j*20+20])
        for k in range(20):
            new_compo[i][j*20+k] = mean
pca.components_ = new_compo



scaled_components = pca.components_ * 32767
scaled_components = (scaled_components).astype(np.int16)
#pca.components_ = scaled_components
new = np.zeros((29,20),np.int16)
for i in range(0,len(scaled_components)):
    for j in range(0,20,1):
        new[i][j] = np.mean(scaled_components[i][j*20:j*20+20])
        new[i][j] = new[i][j]
model = RandomForestClassifier(100)
X_learn_reduced = pca.transform(np.array(X_train_spec))
for i in range(len(X_learn_reduced)):
    X_learn_reduced[i] = X_learn_reduced[i]/np.linalg.norm(X_learn_reduced[i])
model.fit(X_learn_reduced,data2_list)

pickle.dump(model, open("./model_pca_29.pickle", "wb"))
first=1
pp = np.zeros((50,29))
mel = np.zeros((50,20,20))
pca_test = np.zeros((50,29), dtype=np.int16)
pca_test2 = np.zeros((50,29), dtype=np.int32)
for i in range(len(signal)):
    temp = (melspecgram(signal[i], 20,20,512,11111,11111).T)
    mel[i] = ((temp-np.mean(temp))/200).astype(np.int16)
    for j in range(20):
        pca_test2[i]+= (new @ np.abs(mel[i][j])).astype(np.int32)

pca_test = pca_test2//np.max(pca_test2)
print(np.max(np.abs(mel))/32767)
print(np.max(mel[0]))
print(type(new[0][0]),type(mel[0][0][0]))
plt.imshow(pca_test)
plt.show()
plt.imshow(pca_test2)
plt.show()


text = str(new)
text= text.replace("]","")
text= text.replace("[","")
text= text.replace(" ", ",")
text= text.replace(",,",",")
text= text.replace(",,",",")
text= text.replace(",,",",")
text= text.replace(",,",",")
text= text.replace(",,",",")
text= text.replace(",,",",")
#print(text)