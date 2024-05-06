# -*- coding: utf-8 -*-
"""
uart-reader.py 
ELEC PROJECT - 210x
"""
import argparse
import serial
from   serial.tools import list_ports
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
else :
    serialPort = serial.Serial(
        port=args.port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    serialString = ""  # Used to hold data coming over UART
    count = 0
    max_count = 100
    data = []
    rep = []
    model = pickle.load(open("./classification/model_quick.pickle","rb"))
    parity= 0
    while 1:
        # Wait until there is data waiting in the serial buffer
        if serialPort.in_waiting > 0:

            # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline()

            # Print the contents of the serial data
            text =serialString.decode("Ascii")
            feature=np.zeros((2,24))    #24=MEL_LENGTH
            
            if "[MEAN]" in text:
                
                count +=1
                print(str(count/max_count*100)+"%")
                data.append(text)
                
                temp = text[len("[MEAN]"):].strip("\n").strip(",")
                temp = temp.split(",")
                for j in range(len(temp)):
                    temp[j] = np.int16(temp[j])
                temp = np.array(temp)
                temp =temp/np.linalg.norm(temp)
                feature[parity] = temp
                parity = (parity+1)%2
                if parity==0:
                    print(model.predict([np.ravel(feature)]))
            if count==max_count:
                break

    print(rep)
    for i in range(len(data)):
        temp = data[i]
        temp = temp[len("[MEAN]"):].strip("\n").strip(",")
        temp = temp.split(",")
        for j in range(len(temp)):
            temp[j] = np.int16(temp[j])
        temp = np.array(temp)
        print(temp)
        print(temp.astype(np.float64))
        temp =temp/np.linalg.norm(temp)
        data[i] = temp
    data = np.array(data)
    classe = input("enter the class recorded>>")
    name = input("enter the file name>>")
    pickle.dump((data,classe), open(name+".pickle", "wb"))