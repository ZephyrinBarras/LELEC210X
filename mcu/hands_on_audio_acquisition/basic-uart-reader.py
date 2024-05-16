# -*- coding: utf-8 -*-
"""
uart-reader.py 
ELEC PROJECT - 210x
"""
import argparse
import serial
from serial.tools import list_ports
import time
import pickle
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

listeclass = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]
name = sorted(listeclass)
max_size = 4  # à changer
memory = np.zeros((max_size, 5))
pos = 0
commencement = 0


def ajouter(elem):
    global pos
    memory[pos] = elem
    pos = (pos + 1) % max_size


def get_prob(prob):
    result = np.mean(memory)
    print(result)
    return np.mean(memory)


def result():
    return np.argmax(get_prob())


def send(classe):
    hostname = "http://lelec210x.sipr.ucl.ac.be/lelec210x/"
    key = "QMJf02nSY3SvesXt1I8Vi9bKZT520NmM4LpIhHNN"
    response = requests.post(f"{hostname}/leaderboard/submit/{key}/{classe}", timeout=1)


argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--port", help="Port for serial communication")
args = argParser.parse_args()
print('basic-uart-reader launched...\n')

if args.port is None:
    print("No port specified, here is a list of serial communication port available")
    print("================")
    port = list(list_ports.comports())
    for p in port:
        print(p.device)
    print("================")
    print("Launch this script with [-p PORT_REF] to access the communication port")
else:
    serialPort = serial.Serial(
        port=args.port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    serialString = ""  # Used to hold data coming over UART
    count = 0
    max_count = 300
    data = []
    model = pickle.load(open("./model_def.pickle", "rb"))
    parity = 0
    memory_size = 5
    memory_value = np.zeros((memory_size, len(name)))
    position = 0
    vote = np.zeros(5)
    classe_vote = []


    def voteur(array):
        global vote, name
        for k in range(len(vote)):
            count = 0
            for i in array:
                if i == name[k]:
                    count += 1
            vote[k] = count


    while 1:
        # Wait until there is data waiting in the serial buffer
        if serialPort.in_waiting > 0:

            # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline()

            # Print the contents of the serial data
            text = serialString.decode("Ascii")
            feature = np.zeros((2, 24))  # 24=MEL_LENGTH

            if "[MEAN]" in text:

                count += 1
                data.append(text)

                temp = text[len("[MEAN]"):].strip("\n").strip(",")
                temp = temp.split(",")
                for j in range(len(temp)):
                    temp[j] = np.int16(temp[j])
                temp = np.array(temp)
                temp = temp / np.linalg.norm(temp)
                feature[parity] = temp
                parity = (parity + 1) % 2
                if parity:
                    elem = model.predict_proba([np.ravel(feature)])
                    predict = model.predict([np.ravel(feature)])
                    print(elem, model.predict([np.ravel(feature)]))
                    classe_vote.append(predict[0])
                    if len(classe_vote) > 10:
                        del (classe_vote[0])
                    voteur(classe_vote)

                    if np.max(vote) >= 5:
                        print("critere vote")
                        # print("Envoie de ", name[np.argmax(vote)])


                    if np.max(elem) > 0.7:
                        print("je déconne à fond")
                    loga = np.log(elem)
                    memory_value[position] = loga

                    position = (position + 1) % 5
                    proba_sum = np.sum(memory_value, axis=0)
                    retour = np.exp(proba_sum / 5)
                    var = np.var(retour)

                    if np.max(np.exp(proba_sum / 5)) > 0.4:
                        print("nouveau threshold")
                        print("Envoie de ", name[np.argmax(vote)])
                        send(name[np.argmax(vote)])

                    trie = np.flip(np.sort(retour))
                    if np.var(var > 0.2 and trie[0] - trie[1] > 0.2):
                        print("treash variance")
                    if np.var(trie[0] - trie[1] > 0.3):
                        print("treash diff max")

    for i in range(len(data)):
        temp = data[i]
        temp = temp[len("[MEAN]"):].strip("\n").strip(",")
        temp = temp.split(",")
        for j in range(len(temp)):
            temp[j] = np.int16(temp[j])
        temp = np.array(temp)
        temp = temp / np.linalg.norm(temp)
        data[i] = temp
    data = np.array(data)
    classe = input("enter the class recorded>>")
    name = input("enter the file name>>")
    pickle.dump((data, classe), open(name + ".pickle", "wb"))
