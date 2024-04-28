import numpy as np
import matplotlib.pyplot as plt

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
"""s1 = data1_list[20].astype(np.float32)
s1 = s1-np.mean(s1)
s1=s1/np.linalg.norm(s1)
s2 = data1_list[720].astype(np.float32)
s2 = s2-np.mean(s2)
s2=s2/np.linalg.norm(s2)
s3 = data1_list[1440].astype(np.float32)
s3 = s3-np.mean(s3)
s3=s3/np.linalg.norm(s3)
s4 = data1_list[2160].astype(np.float32)
s4 = s4-np.mean(s4)
s4=s4/np.linalg.norm(s4)
print(len(data1_list))
sd.play(s1, 11111)
sd.wait()
sd.play(s2, 11111)
sd.wait()
sd.play(s3, 11111)
sd.wait()
sd.play(s4, 11111)
sd.wait()"""

pca_begin = 29
pca_end = 30
pca_step = 1
n_splits=1
pca_arange = np.arange(pca_begin,pca_end,pca_step)
accuracy_matrix = np.zeros((len(pca_arange),n_splits))
std_matrix = np.zeros((n_splits,len(pca_arange)))
count = 0
for k in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(data1_list, data2_list, test_size=0.3)  # random_state=1
    print(len(X_test), len(y_test))
    X_val_spec = []
    y_val = y_test
    X_train_spec = []
    #ADD DEFECT +conversion en mel  test

    print(f"loop {k}")
    for i in range(len(X_test)):
        
        array = X_test[i].astype(np.float32)
        array = array -np.mean(array)
        array = array*np.random.uniform(0.1,3)   #amplitude
        echo = np.zeros(len(array))
        echo_sig = np.zeros(512*20)
        echo_sig[0] = 1
        n_echo = np.random.randint(1,3)
        echo_sig[(np.arange(1) / 1 * 512*20).astype(int)] = (
            1 / 2
        ) ** np.arange(1)

        array = fftconvolve(array, echo_sig, mode="full")[:512*20]
        array =array + np.random.normal(0, np.random.uniform(0.05,0.7), len(array))   #noise
        sound_to_add = data1_list[np.random.randint(0,len(data1_list))].astype(np.float32)
        sound_to_add = sound_to_add-np.mean(sound_to_add)
        array =   array+sound_to_add*np.random.uniform(0,0.7)/np.max(sound_to_add)*np.max(array)

        x = array-np.mean(array)
        x=x/np.linalg.norm(x)
        """imp = np.log(np.abs(melspecgram(x, Nmel=20, mellength=20, fs_down=fs_down)))
        sd.play(x, 11111)
        sd.wait()
        print(y_test[i])
        plt.imshow(imp)
        plt.show()"""
        spec = np.ravel(np.log(np.abs(melspecgram(x, Nmel=20, mellength=20, fs_down=fs_down))))
        X_val_spec.append(spec-np.mean(spec))
    
    #conversion en mel train
    for i in range(len(X_train)):
        x = X_train[i]-np.mean(X_train[i])  
        #x=x/np.linalg.norm(x)      
        spec = np.ravel(np.log(np.abs(melspecgram(x, Nmel=20, mellength=20, fs_down=fs_down))))
        
        X_train_spec.append(spec-np.mean(spec))

    for b in range(len(pca_arange)):
        n_compo = pca_arange[b]
        n_trees = 100
        model = RandomForestClassifier(n_trees)
        pca = PCA(n_components=n_compo, whiten=True)
        pca.fit(np.array(X_train_spec))
        components = pca.components_
        new_compo = np.zeros((n_compo,400))
        for d in range(len(components)):
            for e in range(20):
                mean = np.mean(components[d][e*20:e*20+20])
                for f in range(20):
                    new_compo[d][e*20+f] = mean
        pca.components_ = new_compo
        X_learn_reduced = pca.transform(np.array(X_train_spec))
        array_fire = []
        array_birds = []
        array_hel = []
        array_hand = []
        array_chain = []
        for i in range(len(X_learn_reduced)):
            X_learn_reduced[i]=X_learn_reduced[i]/np.linalg.norm(X_learn_reduced[i])
            if y_train[i]=="birds":
                array_birds.append(X_learn_reduced[i])
            elif y_train[i]=="fire":
                array_fire.append(X_learn_reduced[i])
            elif y_train[i]=="helicopter":
                array_hel.append(X_learn_reduced[i])
            elif y_train[i]=="chainsaw":
                array_chain.append(X_learn_reduced[i])
            elif y_train[i]=="handsaw":
                array_hand.append(X_learn_reduced[i])
        print("birds")
        plt.imshow(array_birds)
        plt.savefig("imbirds.svg")
        print("helicopter")
        plt.imshow(array_hel)
        plt.savefig("imhel.svg")
        print("handsaw")
        plt.imshow(array_hand)
        plt.savefig("imhand.svg")
        print("chainsaw")
        plt.imshow(array_chain)
        plt.savefig("imchain.svg")
        plt.imshow(array_fire)
        plt.savefig("imfire.svg")
        X_val_reduced = pca.transform(X_val_spec)
        for i in range(len(X_val_reduced)):
            X_val_reduced[i]=X_val_reduced[i]/np.linalg.norm(X_val_reduced[i])
        model.fit(X_learn_reduced, y_train)
        prediction = model.predict(X_val_reduced)
        a = compute_accuracy(prediction, y_val)
        accuracy_matrix[b][k] = a
        print(f"accuracy {n_compo}, {a}")
        
acc = np.zeros(len(pca_arange))
for i in range(len(accuracy_matrix)):
    acc[i] = np.mean(accuracy_matrix[i])
print(acc)

plt.plot(pca_arange, acc)
plt.show()
