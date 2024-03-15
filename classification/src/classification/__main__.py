import pickle
import time
from pathlib import Path
from typing import Optional

import requests

import click
import matplotlib.pyplot as plt
import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from numpy import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from classification.utils.plots import plot_specgram
from sklearn.impute import SimpleImputer

from .utils import payload_to_melvecs

load_dotenv()
pca = pickle.load(open("classification/data/models/pca_Q1_parameters","rb"))
model = pickle.load(open("classification/data/models/random_forest_Q1_parameters.pickle","rb"))

plt.figure()
#plt.show()

def myfct(feature_vector,n_melvecs, melvec_length, counter):
    #plt.figure(clear=True)
    feature_vector = feature_vector.reshape(-1)
    #logger.info(f"Parsed payload into Mel vectors: {melvecs}")
    mp = pca.transform([feature_vector[:-20]])
    #logger.info(f"Parsed payload into Mel vectors: {mp}")
    classe = model.predict(mp)
    #print(f"classe:\t{classe}")
    #print(feature_vector.reshape((n_melvecs, melvec_length)).T)
    
    #logger.info(f"Parsed payload into Mel vectors: {classe}")
    """plot_specgram(feature_vector.reshape((n_melvecs, melvec_length)).T, ax=plt.gca(), is_mel=True, title="MEL Spectrogram #{}".format(counter), xlabel="Mel vector")
    plt.draw()
    plt.pause(0.01)
    plt.clf()"""

def send(classe):
    hostname = "http://lelec210x.sipr.ucl.ac.be/lelec210x/"
    key = "QMJf02nSY3SvesXt1I8Vi9bKZT520NmM4LpIhHNN"
    response = requests.post(f"{hostname}/leaderboard/submit/{key}/{classe}", timeout=1)
    



@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        poetry run auth | poetry run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """
    imp = SimpleImputer(missing_values=nan, strategy="mean")
    model = pickle.load(open("classification/data/models/random_forest_Q1_parameters.pickle","rb"))
    data = zeros((20,20))
    data2 = None
    first=0
    done = 0                    #nombre de mel récupéré pour un feature vector
    t0  = 0                     #temps de la première acuisition d'un feature vector
    t1 = 0                      #permet de comparé à t0 pour savoir si on a perdu un mel
    lastmissFlag = False
    missing = 0
    nbr_nmel =0
    nbr_packet = 0
    error = []
    are_error = 0
    data = full((20, 20), nan)
    #print(f"nmel : {n_melvecs}, length {melvec_length}")

    for payload in _input:
        nbr_packet+=1
        
        if PRINT_PREFIX in payload:
            logger.info(f"detect")
            previous = 0

            #RECUPERE ET TRAITE LE MEL
            #serial_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+5],16)
            
            packet_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+7],16)%20
            payload = payload[len(PRINT_PREFIX)+7:]
            melvecs = payload_to_melvecs(payload, melvec_length, 1)
            print(packet_number)

            if are_error:
                print("error")
                are_error=0
                data[previous] = error

            if previous>packet_number:
                print("detect error")
                are_error=1
                error = melvecs
                previous = packet_number
                continue
            else:
                print("ici", packet_number) 
                previous = packet_number
                data[packet_number] = melvecs.reshape(-1)
                #print(melvecs.reshape(-1))

            if (packet_number==19 ):
                #melvecs1 = data.reshape(-1)
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                #print(data)
                data = imp.fit_transform(data)
                #print(data)
                mp = pca.transform([data.reshape(-1)[:-20]])
                #plt.imshow(data)
                classe = model.predict(mp)
                print(classe)
                


                """plot_specgram(data.reshape((20, 20)).T, ax=plt.gca(), is_mel=True,
                          title="MEL Spectrogram", xlabel="Mel vector", tf=512 / 11000)
                plt.draw()
                plt.pause(0.05)
                plt.clf()"""
                send(classe)
                data = full((20, 20), nan)
                main()
            

                

                

                


            first=1
            #logger.info(f"Parsed payload into Mel vectors: {melvecs}")
            #logger.info(str(len(melvecs))+str( len(melvecs[0])))
            #logger.info("ananas")
            #melvecs1 = melvecs.reshape(-1)
            #logger.info(f"Parsed payload into Mel vectors: {melvecs}")
            #mp = pca.transform([melvecs1[:-20]])
            #logger.info(f"Parsed payload into Mel vectors: {mp}")
            #classe = model.predict(mp)
            #logger.info(f"Parsed payload into Mel vectors: {classe}")
            
            #plt.show()

            

            """hostname = "http://130.104.96.147:5000"
            key = "QMJf02nSY3SvesXt1I8Vi9bKZT520NmM4LpIhHNN"
            guess = classe

            response = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}")"""

            

            """if done==0:
                
                t0 = time.time()
                if lastmissFlag:
                    lastmissFlag=False
                    data[0] = data2
                    done+=1


            #ASSIGNE LE MEL DANS LE FEATURE VECTOR
            #!!!!!!!!! 2 erreurs de réception d'affilé non géré !!!!!!!!!!
            
            t1 = time.time()
            if done<n_melvecs-1:
                if t1-t0>0.1:
                    nbr = (t1-t0-0.05)//0.05
                    error.append(nbr)
                    #print(f"delta t:\t{t1-t0:.4f}\tt0:\t{t0:.4f}")
                    #print("packet missing (replace)")
                    missing +=1
                    
                    data[done] = mean(data[0:done])#remplacer par le moyenne
                    done +=1
                    if done ==n_melvecs-1:
                        lastmissFlag=True
                        data2 = melvecs.T[0]
                        done = 0
                    else:
                        data[done] = melvecs.T[0]
                done +=1
            else:
                data[done] = melvecs.T[0]
                done =0
                nbr_nmel+=1 
                #myfct(data,n_melvecs,melvec_length, nbr_nmel)
                if nbr_nmel==100:
                    pickle.dump(error, open("./error_distrib", 'wb'))
                    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            t0=t1"""
