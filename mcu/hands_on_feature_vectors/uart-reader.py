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

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from classification.utils.plots import plot_specgram

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

dt = np.dtype(np.uint16).newbyteorder("<")


"""

INIT CLASSIFICATION

"""
"""model_knn = pickle.load(open("/home/zephyrin/Desktop/project-2103-2102-a/classification/data/models/model.pickle", 'rb')) # Write your path to the model here!

normalize = True
pca = pickle.load(open("/home/zephyrin/Desktop/project-2103-2102-a/classification/data/models/pca", 'rb'))"""





def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        print("line\t"+line[len(PRINT_PREFIX) :])
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            print(ser.read_until(b"\n", size=2  * MELVEC_LENGTH).decode(
                "ascii"
            ))
            line += ser.read_until(b"\n", size=2  * MELVEC_LENGTH).decode(
                "ascii"
            )
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
    pickle.dump((sound_record, classe_record), open("./"+filename, 'wb'))

def load():
    global sound_loaded, classe_loaded
    b = input("name the file>>")
    a = pickle.load(open("./"+b+".pickle", 'rb'))
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
    model_knn = pickle.load(open("./model.pickle", 'rb')) # Write your path to the model here!

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
        plt.figure()
        i=0

        for melvec in input_stream:
            print("here")
            msg_counter += 1

            #print("MEL Spectrogram #{}".format(msg_counter))
            sgram = melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T
            ncol = int(1000*10200 /(1e3*512)) 
            sgram = sgram[:, :ncol]
            sgram = sgram/ np.linalg.norm(sgram, keepdims=True)
            sgram = np.nan_to_num(sgram,nan=1e-16)
            fv = sgram.reshape(-1)
            #print("must be 400 or change code")
            #print(len(fv))

            ### TO COMPLETE - Eventually normalize and reduce feature vector dimensionality
            try:
                fv = pca.transform([fv[:400]])
                probs = model_knn.predict(fv)[0]
                #print(probs)
            except:
                print("hum")
            """classe = ["fire", "birds", "handsaw", "chainsaw", "helicopter"]
            choice = input("1 to add at the file tosave otherwise skip\n\
                         2 to save the file\n\
                         3 to train the model on the file\n\
                         4 load the model trained\n\
                          Please be sure of the format before saving the len must be in accord with the parameter\n\t>>> ")
                          
            print("choice")
            if choice =="1":
                print("here")
                for i in range(len(classe)):
                    print(i,"\t",classe[i])
                classe_ind = input("enter the class index\n\t >>> ")
                classe_ind = int(classe_ind)
                add_to_save_data(fv, classe[classe_ind])
            elif choice =="2":
                save()
            elif choice =="3":
                train()
            elif choice =="4":
                micro_model()"""
            add_to_save_data(fv, "helicopter")
            print("added")
            if i==120:
                pickle.dump([sound_record, classe_record], open("./helicopter.pickle", 'wb'))
                print("salut")
            i+=1

            #print(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T)

            
            plot_specgram(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, ax=plt.gca(), is_mel=True, title="MEL Spectrogram #{}".format(msg_counter), xlabel="Mel vector")
            plt.draw()
            plt.pause(0.05)
            plt.clf()
            
