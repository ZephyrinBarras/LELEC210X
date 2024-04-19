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

listeclass=["birds","chainsaw","fire","handsaw","helicopter"]
name = sorted(listeclass)
max_size = 4 #à changer
memory = zeros((max_size,5))
pos = 0
commencement=0
def ajouter(elem):
    global pos
    memory[pos] = elem
    pos = (pos+1)%max_size
    
def get_prob(prob):
    result =mean(memory)
    print(result)
    return mean(memory)

def result():
    return argmax(get_prob())



load_dotenv()
pca = pickle.load(open("classification/data/models/pca_vite","rb"))
model = pickle.load(open("classification/data/models/model_vite","rb"))



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
            previous = 0

            #RECUPERE ET TRAITE LE MEL
            #serial_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+5],16)
            
            packet_number = int(payload[len(PRINT_PREFIX):len(PRINT_PREFIX)+7],16)%20
            payload = payload[len(PRINT_PREFIX)+7:]
            melvecs = payload_to_melvecs(payload, melvec_length, 1)

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
                previous = packet_number
                data[packet_number] = melvecs.reshape(-1)
                #print(melvecs.reshape(-1))

            if (packet_number==19 ):
                pca = pickle.load(open("classification/data/models/pca_vite","rb"))
                model = pickle.load(open("classification/data/models/model_vite","rb"))
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                data = imp.fit_transform(data)
                data = data-mean(data)
                mp = data/linalg.norm(data, axis=1,keepdims=True)
                mp = pca.transform([data.reshape(-1)])
                
                #plt.imshow(data)
                elem = model.predict_log_proba(mp)[0]
                ajouter(elem)
                global commencement
                commencement+=1
                probs=memory
                maxvrai= zeros(5) 
                for m in range(5) :
                    for n in range(len(probs)):
                        maxvrai[m]=+probs[n,m]
                #pour le seuille
                """seuille=0.6
                if max(10**(maxvrai/5))<seuille:
                    classe=" "
                else:
                    classe=name[argmax(maxvrai)]"""
                print(max(10**(maxvrai/5)))
                classe=name[argmax(maxvrai)]
        
            
            
                logger.info(f"Parsed payload into Mel vectors: {classe}")
                


                plot_specgram(data.reshape((20, 20)).T, ax=plt.gca(), is_mel=True,
                          title="MEL Spectrogram", xlabel="Mel vector", tf=512 / 11000)
                plt.draw()
                plt.pause(0.05)
                plt.clf()
                #send(classe)
                data = full((20, 20), nan)
                main()